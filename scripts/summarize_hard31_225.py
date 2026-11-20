"""Audit the stopped 31-task extension and the fixed prior ten, without reruns."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_reddit_hard10_225 as run
from scripts.summarize_hard10_60 import LOGS
from web2kg.evaluation.actor_output import validate_single_action
from web2kg.common.responses import ResponsesError

BATCH = ROOT / 'runs/reddit-hard31-225-flash-20261106-v1'


def usage(events):
    starts = [e for e in events if e['event'] == 'responses_started']
    received = {e['request_id']: e for e in events if e['event'] == 'responses_received'}
    retries = [e for e in events if e['event'] == 'responses_retry']
    return dict(physical_requests=len(starts),
        logical_calls=sum(not e.get('output_retry_number') for e in starts)
                      - sum(e.get('persistent', False) for e in retries),
        input_tokens=sum(e.get('usage', {}).get('input_tokens', 0) for e in received.values()),
        output_tokens=sum(e.get('usage', {}).get('output_tokens', 0) for e in received.values()),
        unknown_usage=sum(not {'input_tokens','output_tokens'} <= set(received.get(e['request_id'],{}).get('usage') or {}) for e in starts),
        format_retries=sum(e['event'] == 'responses_output_retry' for e in events),
        transport_retries=len(retries),
        request_seconds=sum(e.get('seconds',0) for e in events if e['event']=='responses_received' or
                            (e['event']=='responses_failed' and e.get('request_id') not in received)),
        actual_models=sorted({e.get('actual_model','unknown') for e in received.values()}))


def href(path):
    return Path(os.path.relpath(path, BATCH)).as_posix()


def main():
    config, state = run.read(BATCH/'protocol.json'), run.read(BATCH/'state.json')
    assert state['status']=='needs_attention' and state['tasks']['66']['status']=='needs_attention'
    assert len(state['tasks'])==10 and Path(state['current_graph'])==BATCH/'407/execution-graph'
    grouped, roles, rows = defaultdict(list), defaultdict(list), []
    selected = [(Path(x['directory']),True,x['state']) for x in config['inherited_results']]
    selected += [(BATCH/str(t),False,state['tasks'][str(t)]) for t in config['task_ids'] if str(t) in state['tasks']]
    for folder, inherited, item in selected:
        result, official = run.read(folder/'result.json'), run.read(folder/'official.json')
        assert official.get('score') in (0,1)
        source=run.read(folder.parent/'protocol.json'); update=result.get('execution_graph') or {}
        group='prior_ten' if inherited else 'new_attempts'
        events=[]
        for role, name in LOGS.items():
            part=run.read_events(folder/name); events.extend(part); roles[(group,role)].extend(part)
        grouped[group].extend(events)
        trace=folder.parent/'trajectories'/f'task-{folder.name}.html';assert trace.is_file()
        recovery=run.read(folder/'recovery/recovery.json',{})
        rows.append(dict(task_id=int(folder.name),inherited=inherited,state=item['status'],
            official_score=official['score'],source_run=folder.parent.name,source_version=source.get('version'),
            frozen_manifest_sha256=run.file_hash(folder.parent/'frozen-runtime/manifest.json'),
            graph_capacity=source['max_graph_nodes'],model=source['model'],
            result_status=result['status'],error=result.get('error'),steps=result['steps'],
            task_seconds=result.get('seconds',0),total_seconds=result.get('total_seconds',0),
            maintenance_seconds=update.get('finalization_seconds',0),
            graph_after=(update.get('retention') or {}).get('after'),
            new_nodes=len(update.get('new_states',[])),removed_nodes=len((update.get('retention') or {}).get('removed',[])),
            index_status=update.get('index_status'),usage=usage(events),
            diagnosis_decisions=len(recovery.get('decisions',[])),return_attempted=recovery.get('attempted',False),
            return_result=recovery.get('return'),step_refund=recovery.get('step_refund'),
            learning=run.read(folder/'recovery/learning.json'),trace=href(trace),
            result_sha256=run.file_hash(folder/'result.json'),official_sha256=run.file_hash(folder/'official.json')))
    assert len({r['task_id'] for r in rows})==20 and all(r['graph_capacity']==225 for r in rows)
    ids=[e['request_id'] for es in grouped.values() for e in es if e['event']=='responses_started']
    assert len(ids)==len(set(ids)), 'Never count inherited requests twice'
    failed=BATCH/'66';actor=run.read_events(failed/'events.jsonl')
    rejected=[e for e in actor if e['event']=='responses_failed' and e.get('failure_kind')=='output:actor_action_invalid']
    if not rejected:
        rejected=[e for e in actor if e['event']=='responses_failed' and 'ACTION section' in e.get('error','')]
    evidence=[]
    for event in rejected:
        p=failed/'llm'/event['request_id'];request=run.read(p/'chat-request.json');response=run.read(p/'chat-response.json')
        text=response['choices'][0]['message']['content']
        actions=re.findall(r'^ACTION:\s*([^\n]+)',text,re.M)
        assert actions==['go_back','go_back']
        try:validate_single_action(text)
        except ResponsesError as error:assert 'multiple sections' in str(error)
        else:raise AssertionError('Saved failure must reproduce offline')
        inputs=[m for m in request['messages'] if m['role']!='system']
        evidence.append(dict(request_id=event['request_id'],actions=actions,
            same_observation_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
            repeated_answer_marker='</think>' in text,
            correction_in_system='Your previous response was rejected:' in request['messages'][0]['content'],
            response_sha256=run.file_hash(p/'chat-response.json')))
    assert len(evidence)==4 and len({e['same_observation_sha256'] for e in evidence})==1
    first_failed=next(i for i,e in enumerate(actor) if e.get('request_id')==evidence[0]['request_id'] and e['event']=='responses_started')
    assert not any(e['event']=='agent_action' for e in actor[first_failed:])
    diagnosis=dict(status='confirmed_output_format_incompatibility',task_id=66,
        cause='All four successful model responses contain two identical ACTION: go_back sections. The registered single-action-v2 validator rejects duplicate headings even when commands match. Three responses also contain a repeated answer separated by a stray closing think tag.',
        retry_count=3,same_observation=True,browser_actions_from_rejected_responses=0,
        model_budget_exhausted=False,network_error=False,evidence=evidence,
        retained_graph=str(BATCH/'407/execution-graph'),failed_graph_inherited=False,
        possible_repair='Consider lossless normalization of exactly identical complete action sections; still reject different actions and trailing commands. Not implemented or rerun in this frozen batch.',
        recovery=run.read(failed/'recovery/recovery.json',{}).get('return'))
    run.write_json(BATCH/'failure-diagnosis.json',diagnosis)
    excluded=[]
    for item in config['excluded_attempts']:
        folder=Path(item['directory']);events=[e for name in LOGS.values() for e in run.read_events(folder/name)]
        excluded.append(dict(task_id=item['task_id'],reason=item['reason'],official_score=run.read(folder/'official.json')['score'],
            usage=usage(events),trace=href(folder.parent/'trajectories'/f'task-{folder.name}.html')))
    graph=run.Graph.load(Path(state['current_graph']));index=run.read(graph.directory/'index.json')
    assert len(graph.nodes)==225 and set(index['ids'])==set(graph.nodes)
    full_checks=[e for e in run.read_events(BATCH/'timings/events.jsonl') if e['event']=='integrity_verified' and e.get('scope')=='full']
    assert full_checks[-1]['time'] > actor[-1]['time'], 'Require post-stop full verification'
    remaining=[t for t in config['task_ids'] if str(t) not in state['tasks']]
    audit=dict(created_at=datetime.now(timezone.utc).isoformat(),status='stopped_incomplete',planned_selected=41,
        inherited_scored=10,new_planned=31,new_scored=10,unstarted=remaining,rows=rows,excluded=excluded,
        prior_successes=sum(r['official_score']==1 for r in rows if r['inherited']),
        new_successes=sum(r['official_score']==1 for r in rows if not r['inherited']),
        scored_successes=sum(r['official_score']==1 for r in rows),scored_count=len(rows),
        overall41_success_rate=None,usage={g:usage(es) for g,es in grouped.items()},
        roles={g:{role:usage(es) for (group,role),es in roles.items() if group==g} for g in grouped},
        new_queue_seconds=state['total_seconds'],
        new_task_seconds=sum(r['task_seconds'] for r in rows if not r['inherited']),
        new_maintenance_seconds=sum(r['maintenance_seconds'] for r in rows if not r['inherited']),
        guard_receipt=full_checks[-1],frozen_files=len(run.read(BATCH/'frozen-runtime/manifest.json')),
        original_graph_and_published_pointer_preserved=True,final_graph_nodes=len(graph.nodes),
        final_graph_edges=len(graph.data['edges']),final_index_pages=len(index['ids']),
        source_code_unchanged_during_run=True,paid_preflight=0,prices_estimated=False,
        limits=config['limits']+' Incomplete:21 unstarted tasks have no scores;18/20 is only the scored subset, never the final41-task success rate.')
    run.write_json(BATCH/'combined-audit.json',audit)
    esc=html.escape
    table=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in [f'<a href="{esc(r["trace"],quote=True)}">{r["task_id"]}</a>',
        '历史' if r['inherited'] else '新增',int(r['official_score']),r['steps'],r['state'],esc(r['source_run'])])+'</tr>' for r in rows)
    new=audit['usage']['new_attempts'];old=audit['usage']['prior_ten']
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>225节点 Reddit Hard · 阶段汇总</title><style>body{{font:17px/1.65 system-ui;max-width:1180px;margin:36px auto;padding:0 20px;color:#17202a}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow-x:auto}}.notice{{padding:18px;background:#fff3dd;border-radius:8px}}a{{color:#155f9d}}</style>
<h1>225节点 Reddit Hard · 阶段汇总</h1><div class="notice"><b>本批因66格式错误停止，尚未完成31题。</b>旧十题9/10；新增已评分10题中9题成功。当前合计18/20，剩余21题未开始，不能把它写成41题的最终成功率。724按此前授权排除，原失败及成本另外保留。</div>
<h2>为什么停止</h2><p>66在9次实际行动之后，模型连续四次把同一个 <code>ACTION: go_back</code> 输出两遍。原响应加三次格式重生成都被当前严格单段校验拒绝，其中三次还重复了整段回答。纠错提示已发送，四次观察输入哈希一致，没有执行这些失败响应中的动作。这不是网络失败，也没有耗尽31次调用额度。</p>
<p>保留了66的官方0分、原始响应、全部请求成本和不完整图。模型错误使收尾制卡停止，该图没有交接；最近完整图仍是407后的225节点、{audit['final_graph_edges']}边和225页索引。队列、worker和自有Jina已退出。没有修改冻结规则或自动重跑。</p>
<p>可以考虑仅对完全相同的完整动作段做无损去重，仍拒绝不同动作及一个字段中的多个命令。这是后续修复建议，本批没有实施。66此前触发诊断并核验返回成功，记录路径返回动作数为0、返还6行动步；随后任务仍失败，未学习成功经验，不能据此宣称诊断提高成功率。</p>
<h2>逐题来源与评分</h2><div class="scroll"><table><tr><th>任务/轨迹</th><th>来源</th><th>官方分</th><th>实际步</th><th>状态</th><th>独立版本目录</th></tr>{table}</table></div>
<p>未开始：{', '.join(map(str,remaining))}。</p>
<h2>用量与时间</h2><p>新增十次尝试共{new['physical_requests']}次物理请求，输入{new['input_tokens']:,}、输出{new['output_tokens']:,} token，格式重生成{new['format_retries']}次，传输重试{new['transport_retries']}次，未知用量{new['unknown_usage']}。已包括失败66。新队列约{state['total_seconds']/60:.2f}分钟，其中各题执行合计{audit['new_task_seconds']/60:.2f}分钟、维护合计{audit['new_maintenance_seconds']/60:.2f}分钟；登记与停止后的人工分析/核验另计。</p>
<p>旧十题引用一次：{old['physical_requests']}次物理请求，输入{old['input_tokens']:,}、输出{old['output_tokens']:,} token。724单列在审计中；建图、旧准备调用和未选中的早期失败版本仍留在原记录，不并入此处选定任务用量。没有新增付费预检或估算价格。</p>
<h2>汇总边界</h2><p>所有选定任务均为Qwen3.7-flash、225节点容量。旧十题来自多个授权修复版本；新题采用当前单动作提示词及格式重生成，并从旧十题最后完整644图副本开始，不继承735单题测试或60节点实验结果。每行来源及源码清单哈希都保留。这是开发记录汇总，不是单一冻结版本的盲测、图大小对照或组件因果实验。</p>
<p>停止后完整核验{audit['guard_receipt']['protected_files']}项保护文件及冻结源通过；原始225图、发布指针和用户修改的提示词未改变。</p>
<p><a href="combined-audit.json">分角色用量与审计</a> · <a href="failure-diagnosis.json">66原因与四次请求证据</a> · <a href="report.html">全部41行状态</a> · <a href="result-lineage.json">固定来源清单</a> · <a href="initial-graph/viewer.html">保留原始图</a> · <a href="407/execution-graph/viewer.html">最后完整图</a></p></html>'''
    (BATCH/'analysis.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['new_successes','scored_successes','scored_count','unstarted','usage','new_queue_seconds','final_graph_edges']},ensure_ascii=False))


if __name__=='__main__':main()
