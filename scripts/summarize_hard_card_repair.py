"""Audit the authorized624 repair and seven-task remainder, including fixed prior rows."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_reddit_hard10_225 as run
from scripts.summarize_hard31_225 import usage
from scripts.summarize_hard10_60 import LOGS


def summarize(batch):
    c,s=run.read(batch/'protocol.json'),run.read(batch/'state.json')
    if s['status'] not in ('completed','needs_attention','user_stopped'):
        raise ValueError('Only summarize a stopped queue')
    def link(p):return Path(os.path.relpath(p,batch)).as_posix()
    groups,roles,rows,feedback=defaultdict(list),defaultdict(list),[],[]
    selected=[(Path(x['directory']),True,x['state']) for x in c['inherited_results']]
    selected += [(batch/str(t),False,s['tasks'][str(t)]) for t in c['task_ids'] if str(t) in s['tasks']]
    for f,inherited,state in selected:
        r,o,p=run.read(f/'result.json',{}),run.read(f/'official.json',{}),run.read(f.parent/'protocol.json')
        group='inherited33' if inherited else 'new8'; events=[]
        for role,log in LOGS.items():
            es=run.read_events(f/log);events.extend(es);roles[(group,role)].extend(es)
        groups[group].extend(events)
        trace=f.parent/'trajectories'/f'task-{f.name}.html'
        if o.get('score') in (0,1):assert trace.exists()
        g=r.get('execution_graph') or {}; rec=r.get('recovery') or {}
        rows.append(dict(task_id=int(f.name),inherited=inherited,source_run=f.parent.name,source_version=p['version'],
            state=state['status'],official_score=o.get('score'),result_status=r.get('status'),error=r.get('error'),
            graph_capacity=p['max_graph_nodes'],actor_model=p['model'],steps=r.get('steps'),
            task_seconds=r.get('seconds',0),total_seconds=r.get('total_seconds',0),
            maintenance_seconds=g.get('finalization_seconds',0),queue_task_seconds=state.get('seconds'),
            graph_after=(g.get('retention') or {}).get('after'),new_nodes=len(g.get('new_states',[])),
            removed_nodes=len((g.get('retention') or {}).get('removed',[])),index_status=g.get('index_status'),
            global_update=g.get('template_global'),card_refresh=g.get('card_refresh'),recovery=rec,
            learning=run.read(f/'recovery/learning.json'),usage=usage(events),trace=link(trace) if trace.exists() else None))
        if inherited:continue
        # Cards run concurrently: associate feedback by the actual rejected text,
        # never by whichever unrelated request happens to be next in the log.
        card_events=run.read_events(f/LOGS['cards_framework'])
        requests={q.parent.name:run.read(q) for q in (f/'graph-cards/llm').glob('*/request.json')}
        for e in card_events:
            if e['event']!='responses_output_feedback' or e.get('feedback_version')!='validated-card-v1':continue
            a=requests[e['request_id']];matches=[]
            for rid,z in requests.items():
                inp=z.get('input',[])
                if len(inp)<3 or inp[-2]['role']!='assistant':continue
                raw=inp[-2]['content'][0]['text']
                if hashlib.sha256(raw.encode()).hexdigest()!=e['rejected_sha256']:continue
                original=a['input'][:-2] if len(a['input'])>=3 and a['input'][-2]['role']=='assistant' else a['input']
                assert inp[:-2]==original and z.get('local_images')==a.get('local_images')
                matches.append(dict(request_id=rid,validated=any(x['event']=='responses_semantic_validated'
                    and x['request_id']==rid for x in card_events),request_sha256=run.file_hash(f/'graph-cards/llm'/rid/'request.json')))
            feedback.append(dict(task_id=int(f.name),rejected_request=e['request_id'],retry_number=e['retry_number'],
                original_evidence_unchanged=bool(matches),corrected_requests=matches))
    excluded=[]
    for item in c.get('excluded_attempts',[]):
        f=Path(item['directory']);es=[e for log in LOGS.values() for e in run.read_events(f/log)]
        groups['excluded_attempts'].extend(es)
        excluded.append(dict(task_id=item['task_id'],directory=str(f),reason=item['reason'],
            official_score=run.read(f/'official.json',{}).get('score'),usage=usage(es),
            trace=link(f.parent/'trajectories'/f'task-{f.name}.html')))
    starts=[e['request_id'] for es in groups.values() for e in es if e['event']=='responses_started']
    assert len(starts)==len(set(starts)), 'Do not double count any physical request'
    assert len({r['task_id'] for r in rows})==len(rows)
    unstarted=[t for t in c['task_ids'] if str(t) not in s['tasks']]
    graph=run.Graph.load(Path(s['current_graph']));index=run.read(graph.directory/'index.json')
    assert len(graph.nodes)==225 and set(index['ids'])==set(graph.nodes)
    receipts=[e for e in run.read_events(batch/'timings/events.jsonl') if e['event']=='integrity_verified' and e.get('scope')=='full']
    if s['tasks']:assert receipts and datetime.fromisoformat(receipts[-1]['time']).timestamp()>max(v['start'] for v in s['tasks'].values())
    scored=[r for r in rows if r['official_score'] in (0,1)];new=[r for r in scored if not r['inherited']]
    audit=dict(created_at=datetime.now(timezone.utc).isoformat(),status=s['status'],error=s.get('error'),
        selected_count=len(rows),scored_count=len(scored),successes=sum(r['official_score'] for r in scored),
        new_scored=len(new),new_successes=sum(r['official_score'] for r in new),unstarted=unstarted,rows=rows,
        excluded_attempts=excluded,usage={k:usage(v) for k,v in groups.items()},
        usage_by_role={f'{g}/{role}':usage(es) for (g,role),es in roles.items()},card_feedback=feedback,
        queue_seconds=s.get('total_seconds'),last_complete_graph=str(graph.directory),
        graph=dict(nodes=len(graph.nodes),edges=len(graph.data['edges']),indexed=len(index['ids'])),
        guard=receipts[-1],frozen_files=len(run.read(batch/'frozen-runtime/manifest.json')),
        preflight_calls=0,preparation_calls=0,limits=c['limits'])
    run.write_json(batch/'combined-audit.json',audit)
    esc=html.escape
    def table(records,keys):
        return '<table><thead><tr>'+''.join(f'<th>{esc(k)}</th>' for k in keys)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+esc(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in records)+'</tbody></table>'
    links=''.join(f'<li><a href="{esc(r["trace"],quote=True)}">任务 {r["task_id"]} · {r["source_run"]} · 官方 {r["official_score"]}</a></li>'
                  for r in rows if r['trace'])
    costs=[dict(group=k,**v) for k,v in audit['usage'].items()]
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Reddit Hard41 · 制卡校验修复</title>
<style>body{{max-width:1160px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#183039;background:#f5f7f7}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}td,th{{border:1px solid #d3dfe2;padding:8px;text-align:left;overflow-wrap:anywhere}}a{{color:#086a7c}}.box{{padding:16px;background:white;border-left:4px solid #17777b}}code{{overflow-wrap:anywhere}}</style>
<h1>Reddit Hard41 · 制卡校验修复</h1><p>状态：{esc(s['status'])}。全部模型 Qwen3.7-flash，225 节点容量。</p>
<div class="box">本批 {len(new)}/8 题有官方评分，成功 {audit['new_successes']} 题。与此前固定 33 题合并：{audit['successes']}/{len(scored)}。
未开始：{esc(str(unstarted))}。队列错误：{esc(str(s.get('error')))}。</div>
<h2>本次改变</h2><p>制卡 Schema 明确每个变体需 1–5 个实际证据编号；没有支持的变体可以不写。业务校验进入同一个最多三次重生成循环，反馈最新失败卡片与具体错误。原证据、截图、行动预算不变；不猜测证据或自动删除输出。</p>
<p>624 从原始 28 题后的完整图重跑。旧 624 的官方 1 分、收尾失败与成本单独保留，当前结果无论高低均按授权替换。新实验没有额外付费预检或制卡准备。</p>
<h2>新任务</h2>{table([r for r in rows if not r['inherited']],['task_id','official_score','state','steps','task_seconds','maintenance_seconds'])}
<h2>请求成本</h2>{table(costs,['group','physical_requests','logical_calls','input_tokens','output_tokens','unknown_usage','format_retries','transport_retries'])}
<p>队列墙钟 {float(s.get('total_seconds',0))/60:.2f} 分钟；登记与完整性核验单列。并行请求时长不能当作墙钟。此表覆盖所选任务及登记的排除尝试；早期建图、准备和其他故障成本仍见历史报告，没有计入本批，也不估算价格。</p>
<h2>卡片纠错与完整性</h2><p>本批记录 {len(feedback)} 次卡片反馈，详见审计文件。最后完整图 {len(graph.nodes)} 节点、{len(graph.data['edges'])} 边、{len(index['ids'])} 页索引。停止后完整核验 {receipts[-1].get('files',receipts[-1].get('protected_files','见审计'))} 项，冻结源码 {audit['frozen_files']} 个文件。</p>
<p>这些是已接触任务上的跨代码版本累计开发结果，不是单一冻结版本的盲测，也不能据此断言泛化或组件因果收益。旧 60 节点批次未混入；724 按原授权排除。</p>
<p><a href="combined-audit.json">逐题、分角色和反馈审计</a> · <a href="repair-lineage.json">修复来源</a> · <a href="report.html">队列报告</a></p>
<h2>实际轨迹</h2><ul>{links}</ul></html>'''
    (batch/'analysis.html').write_text(content,encoding='utf-8')
    print({k:audit[k] for k in ('status','scored_count','successes','new_scored','new_successes','unstarted','graph')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    summarize(p.parse_args().directory.resolve())
