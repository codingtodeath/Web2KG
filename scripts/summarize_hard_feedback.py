"""Audit the authorized format-feedback retry and its stopped continuation."""
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_reddit_hard10_225 as run
from scripts.summarize_hard31_225 import usage
from scripts.summarize_hard10_60 import LOGS
from web2kg.stage1.function_cards import validate_card
from jsonschema import Draft202012Validator

BATCH=ROOT/'runs/reddit-hard22-feedback-flash-20261106-v1'


def href(p):return Path(os.path.relpath(p,BATCH)).as_posix()


def main():
    c,s=run.read(BATCH/'protocol.json'),run.read(BATCH/'state.json')
    assert s['status']=='needs_attention' and s['tasks']['624']['status']=='needs_attention'
    assert len(s['tasks'])==15 and Path(s['current_graph'])==BATCH/'28/execution-graph'
    rows,feedback,groups,roles=[],[],defaultdict(list),defaultdict(list)
    selected=[(Path(x['directory']),True,x['state']) for x in c['inherited_results']]
    selected += [(BATCH/str(t),False,s['tasks'][str(t)]) for t in c['task_ids'] if str(t) in s['tasks']]
    for f,inherited,state in selected:
        r,o,p=run.read(f/'result.json'),run.read(f/'official.json'),run.read(f.parent/'protocol.json')
        assert o.get('score') in (0,1)
        group='inherited19' if inherited else 'new15';events=[]
        for role,log in LOGS.items():
            part=run.read_events(f/log);events.extend(part);roles[(group,role)].extend(part)
        groups[group].extend(events)
        trace=f.parent/'trajectories'/f'task-{f.name}.html';assert trace.exists()
        g=r.get('execution_graph') or {};rec=run.read(f/'recovery/recovery.json',{})
        rows.append(dict(task_id=int(f.name),inherited=inherited,source_run=f.parent.name,source_version=p['version'],
            source_manifest_sha256=run.file_hash(f.parent/'frozen-runtime/manifest.json'),
            state=state['status'],result_status=r['status'],error=r.get('error'),official_score=o['score'],
            graph_capacity=p['max_graph_nodes'],actor_model=p['model'],steps=r['steps'],task_seconds=r.get('seconds',0),
            total_seconds=r.get('total_seconds',0),maintenance_seconds=g.get('finalization_seconds',0),
            graph_after=(g.get('retention') or {}).get('after'),new_nodes=len(g.get('new_states',[])),
            removed_nodes=len((g.get('retention') or {}).get('removed',[])),index_status=g.get('index_status'),
            global_update_status=(g.get('template_global') or {}).get('status'),
            diagnoses=len(rec.get('decisions',[])),return_attempted=rec.get('attempted',False),
            return_result=rec.get('return'),step_refund=rec.get('step_refund'),learning=run.read(f/'recovery/learning.json'),
            usage=usage(events),trace=href(trace),result_sha256=run.file_hash(f/'result.json'),official_sha256=run.file_hash(f/'official.json')))
        if inherited:continue
        actor=run.read_events(f/'events.jsonl')
        for i,e in enumerate(actor):
            if e['event']!='responses_output_feedback':continue
            rejected=f/'llm'/e['request_id'];a=run.read(rejected/'request.json')
            j=next(j for j in range(i+1,len(actor)) if actor[j]['event']=='responses_started')
            corrected=f/'llm'/actor[j]['request_id'];z=run.read(corrected/'request.json')
            original=a['input'] if e['retry_number']==1 else a['input'][:-2]
            assert z['input'][:-2]==original and z['input'][-2]['role']=='assistant' and z['input'][-1]['role']=='user'
            text=z['input'][-2]['content'][0]['text']
            assert hashlib.sha256(text.encode()).hexdigest()==e['rejected_sha256']
            assert not any(x['event']=='agent_action' for x in actor[i+1:j])
            feedback.append(dict(task_id=int(f.name),rejected_request=e['request_id'],corrected_request=actor[j]['request_id'],
                retry_number=e['retry_number'],original_observation_unchanged=True,previous_visible_output_included=True,
                no_browser_action_between=True,corrected_parsed=(corrected/'parsed.txt').exists(),
                rejected_action_lines=re.findall(r'^ACTION:\s*([^\n]+)',text,re.M),
                request_sha256=run.file_hash(corrected/'request.json')))
    assert len(rows)==34 and len({r['task_id'] for r in rows})==34
    starts=[e['request_id'] for es in groups.values() for e in es if e['event']=='responses_started']
    assert len(starts)==len(set(starts))
    old=[]
    for item in c['excluded_attempts']:
        f=Path(item['directory']);es=[e for log in LOGS.values() for e in run.read_events(f/log)]
        old.append(dict(task_id=item['task_id'],reason=item['reason'],official_score=run.read(f/'official.json')['score'],
            usage=usage(es),trace=href(f.parent/'trajectories'/f'task-{f.name}.html')))
    request_id='a69cbcfa7d6d40a8ab9e1632fd74ad49';folder=BATCH/'624/graph-cards/llm'/request_id
    request=run.read(folder/'request.json');response=run.read(folder/'response.json')
    text='\n'.join(p['text'] for m in response['messages'] for p in m['content'] if p['type']=='output_text')
    value=json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
    # Reproduce the acceptance/rejection boundary on the saved output, no model call.
    Draft202012Validator(request['text']['format']['schema']).validate(value)
    empty=[i for i,v in enumerate(value['variants']) if not v['evidence_nodes']]
    assert empty==[1,2]
    try:validate_card(value,{'representatives':[{'node_id':'P1'}]})
    except ValueError as e:assert str(e)=='Invalid variant or invented evidence node'
    else:raise AssertionError('Expected the saved card failure')
    r624=run.read(BATCH/'624/result.json');failed_graph=r624['execution_graph']
    global_cards=run.read(BATCH/'624/execution-graph/function-cards/global/cards.json')['cards']
    diagnosis=dict(status='confirmed_schema_and_business_validation_mismatch',task_id=624,official_score=1,
        failed_role='template_card',failed_template='T124',request_id=request_id,
        cause='Model generated two hypothetical variants without evidence_nodes. JSON Schema accepts empty arrays; post-response validate_card requires a nonempty evidence list. Therefore the failure happens after structured-output retries, not inside them.',
        empty_evidence_variant_indices=empty,schema_passed_offline=True,post_validator_failed_offline=True,
        failed_response_sha256=run.file_hash(folder/'response.json'),global_cards=dict(Counter(v['status'] for v in global_cards.values())),
        selected_clusters=failed_graph['template_global']['selection']['clusters'],
        global_assets_published=False,retained_complete_graph=str(BATCH/'28/execution-graph'),
        failed_task_graph_inherited=False,
        next_repair='Align card request schema with semantic constraints, especially nonempty evidence references, and give validation feedback within bounded regeneration. Do not invent references or silently drop unsupported variants.',
        scope='Different issue from Actor ACTION feedback; retained without another paid restart.')
    run.write_json(BATCH/'failure-diagnosis.json',diagnosis)
    graph=run.Graph.load(Path(s['current_graph']));index=run.read(graph.directory/'index.json')
    assert len(graph.nodes)==225 and set(index['ids'])==set(graph.nodes)
    checks=[e for e in run.read_events(BATCH/'timings/events.jsonl') if e['event']=='integrity_verified' and e.get('scope')=='full']
    assert checks[-1]['time'] > run.read_events(BATCH/'624/events.jsonl')[-1]['time']
    unstarted=[t for t in c['task_ids'] if str(t) not in s['tasks']]
    audit=dict(created_at=datetime.now(timezone.utc).isoformat(),status='stopped_incomplete',planned=41,scored=34,
        successes=sum(r['official_score']==1 for r in rows),unstarted=unstarted,rows=rows,excluded_attempts=old,
        new_starts=15,new_successes=sum(r['official_score']==1 for r in rows if not r['inherited']),
        overall41_success_rate=None,usage={g:usage(es) for g,es in groups.items()},
        roles={g:{role:usage(es) for (group,role),es in roles.items() if group==g} for g in groups},
        actor_feedback=feedback,new_queue_seconds=s['total_seconds'],
        new_task_seconds=sum(r['task_seconds'] for r in rows if not r['inherited']),
        new_maintenance_seconds=sum(r['maintenance_seconds'] for r in rows if not r['inherited']),
        final_graph_nodes=len(graph.nodes),final_graph_edges=len(graph.data['edges']),final_index_pages=len(index['ids']),
        integrity=checks[-1],frozen_files=len(run.read(BATCH/'frozen-runtime/manifest.json')),
        original_graph_and_published_pointer_unchanged=True,old66_preserved=True,paid_preflight=0,prices_estimated=False,
        scope='Fixed multiversion developmental selection; old66 is superseded by the authorized new66 regardless of score and costed separately. Seven unstarted tasks have no scores.')
    run.write_json(BATCH/'combined-audit.json',audit)
    esc=html.escape
    table=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in [f'<a href="{esc(r["trace"],quote=True)}">{r["task_id"]}</a>',
        '保留' if r['inherited'] else '本批',int(r['official_score']),r['steps'],r['state'],esc(r['source_run'])])+'</tr>' for r in rows)
    u=audit['usage']['new15'];f66=[x for x in feedback if x['task_id']==66]
    assert len(f66)==1 and f66[0]['corrected_parsed']
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>纠错反馈修复 · Reddit Hard 阶段结果</title>
<style>body{{font:17px/1.65 system-ui;max-width:1180px;margin:36px auto;padding:0 20px;color:#17202a}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow-x:auto}}.note{{background:#fff3dd;padding:18px;border-radius:8px}}a{{color:#155f9d}}</style>
<h1>纠错反馈修复 · Reddit Hard 阶段结果</h1><div class="note"><b>Actor纠错已验证有效，本批因另一处制卡校验问题停止。</b>已执行15/22题，8题通过；固定合并此前19题后为26/34，仍有7题未开始，不能写成41题的最终成功率。</div>
<h2>这次改了什么、验证到了什么</h2><p>格式不合格时，在原观察之后附上最近一次失败的可见响应和具体错误，明确动作尚未执行；再次失败只替换临时反馈，不堆积错误回答。严格单动作规则、观察、额度和最多三次格式重生成保持不变。111项相关离线检查通过。</p>
<p>重跑66再次遇到重复ACTION，真实请求附带错误回答后，一次纠正就生成合法单动作，并执行到结束。11步，官方0分：格式纠错恢复成功，不等于任务答案正确。本批共有{len(feedback)}次Actor反馈，均核验了原观察未变、附带最新失败响应、期间未操作浏览器。旧66失败和成本单列，未选择最高分。</p>
<h2>为何在624停止</h2><p>624七步完成，官方1分。收尾全局聚类选出20个模板，制卡产出17张有效卡、1张失败卡，尚有2张未完成。T124响应包含两个没有证据页的假设变体，evidence_nodes为空。请求Schema允许空数组，后置校验却要求非空，导致模型响应已被记录为解析成功后才失败，没有进入格式重生成。</p>
<p>该新问题未在本批修改，也未自动重跑。未完成的全局分区和卡片没有发布，624任务图没有交接，最近完整图仍为28之后的225节点、{audit['final_graph_edges']}边、225页索引。一次制卡连接错误曾按登记策略重试，最终停止原因是上述证据校验失败。</p>
<h2>全部已评分来源</h2><div class="scroll"><table><tr><th>任务/轨迹</th><th>来源</th><th>官方分</th><th>实际步</th><th>状态</th><th>版本目录</th></tr>{table}</table></div>
<p>未开始：{', '.join(map(str,unstarted))}。724继续按用户授权排除。617因导航站点参数错误结束，其余未通过题的官方分也原样保留。</p>
<h2>本批成本和边界</h2><p>新增{u['physical_requests']}次物理请求，输入{u['input_tokens']:,}、输出{u['output_tokens']:,} token；未知用量{u['unknown_usage']}，传输重试{u['transport_retries']}，格式重生成{u['format_retries']}。包括66重跑和624失败制卡。队列约{s['total_seconds']/60:.2f}分钟，各题执行合计{audit['new_task_seconds']/60:.2f}分钟、维护{audit['new_maintenance_seconds']/60:.2f}分钟；登记和停止后核验另计。旧19题、旧66失败和724分别统计，不重复计费，不估算价格。</p>
<p>全角色Qwen3.7-flash、容量225。任务和知识已用于开发，结果跨多个冻结版本；本次证明了具体纠错链路能工作，不证明成功率提升或泛化。停止后的完整核验通过，原始图、发布指针、用户提示词和旧失败保留，自有队列/worker/Jina已退出。</p>
<p><a href="combined-audit.json">逐题、角色成本和真实纠错审计</a> · <a href="failure-diagnosis.json">624制卡原因</a> · <a href="report.html">全部41行状态</a> · <a href="repair-lineage.json">修复和来源</a> · <a href="initial-graph/viewer.html">保留原始图</a> · <a href="28/execution-graph/viewer.html">最后完整图</a></p></html>'''
    (BATCH/'analysis.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['new_successes','successes','scored','unstarted','usage','new_queue_seconds','final_graph_edges']},ensure_ascii=False))


if __name__=='__main__':main()
