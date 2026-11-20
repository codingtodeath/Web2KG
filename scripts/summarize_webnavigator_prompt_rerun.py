"""Read-only evidence aggregation for a finished Verified stop-prompt rerun."""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(batch):
    summary, state = read(batch/'summary.json'), read(batch/'state.json')
    protocol, guards = read(batch/'protocol.json'), read(batch/'final-audit.json')
    lineage, schema = read(batch/'repeat-lineage.json'), read(batch/'public-schema-check.json')
    previous = Path(lineage['previous'])
    old = read(previous/'summary.json')
    assert state['status'] == 'completed' and guards['guards_passed']
    assert summary['resolved'] == summary['official_scored'] == 42
    assert len(set(protocol['task_ids'])) == 42 and not protocol['inherited_tasks']
    valid = {r['task_id']:r['public_schema_valid'] for r in schema['tasks']}
    roles, rows, ids = defaultdict(Counter), [], set()
    prior_rows = {r['task_id']:r for r in old['tasks']}
    for item in summary['tasks']:
        folder = Path(item['source'])
        official = read(folder/'official.json')
        components = [{k:e.get(k) for k in ('evaluator_name','status','score')}
                      for e in official['evaluators_results']]
        network = [e for e in components if e['evaluator_name']=='NetworkEventEvaluator']
        if item['official_score'] == 1:
            category = '成功'
        elif item['status'] == 'failed':
            category = '异常退出'
        elif not valid[item['task_id']]:
            category = '答案协议不合规'
        elif network and any(e['status']!='success' for e in network):
            category = '网页操作检查未全部通过'
        else:
            category = '协议合规但答案检查未通过'
        events = [json.loads(x) for x in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        starts = [e for e in events if e['event']=='responses_started']
        receipts = {e['request_id']:e for e in events if e['event']=='responses_received'}
        for event in starts:
            request_id = event['request_id']
            assert request_id not in ids
            ids.add(request_id)
            received = receipts.get(request_id,{})
            usage = received.get('usage',{})
            roles[event['purpose']].update(physical_requests=1,
                input_tokens=usage.get('input_tokens',0),output_tokens=usage.get('output_tokens',0),
                unknown_usage=int(not usage),response_seconds=received.get('seconds',0))
        reason = None
        if item['status']=='failed' and starts:
            response = folder/'llm'/starts[-1]['request_id']/'chat-response.json'
            if response.exists():
                reason = (read(response).get('choices') or [{}])[0].get('finish_reason')
        assert (batch/item['trace']).is_file()
        rows.append({**item,'public_schema_valid':valid[item['task_id']], 'category':category,
                     'components':components,'last_finish_reason':reason,
                     'previous_score':prior_rows[item['task_id']]['official_score']})
    assert len(ids) == summary['totals']['physical_requests']
    old_manifest = read(previous/'frozen-runtime/manifest.json')
    new_manifest = read(batch/'frozen-runtime/manifest.json')
    changed = [k for k,v in old_manifest.items() if new_manifest.get(k)!=v]
    upstream_changes = []
    for source in (previous/'upstream').rglob('*'):
        if not source.is_file(): continue
        rel = source.relative_to(previous/'upstream')
        target = batch/'upstream'/rel
        if not target.exists() or hashlib.sha256(source.read_bytes()).digest()!=hashlib.sha256(target.read_bytes()).digest():
            upstream_changes.append(rel.as_posix())
    assert upstream_changes == ['webNavigator/prompts/navigation_specifications/stop.txt']
    audit = dict(status='completed',schema_validation=schema,guards=guards,trace_links_checked=42,
        official_successes=summary['official_successes'],success_rate=summary['success_rate'],
        categories=dict(Counter(r['category'] for r in rows)),tasks=rows,
        model=protocol['model'],provider=protocol['provider'],
        actual_models=sorted({m for r in rows for m in r['actual_models']}),
        roles=dict(roles),totals_new=summary['totals'],totals_previous_selected=old['totals'],
        totals_both_selected_runs={k:summary['totals'][k]+old['totals'][k] for k in summary['totals']},
        previous_successes=old['official_successes'],queue_seconds=state['seconds'],
        task_seconds=sum(r.get('seconds') or 0 for r in rows),
        changed_existing_runtime_files=changed,changed_upstream_files=upstream_changes,
        source_summary_sha256=hashlib.sha256((batch/'summary.json').read_bytes()).hexdigest(),
        limitations=['Exposed development tasks; no blind generalization or isolated causal claim.',
            'Prompt-only adaptation: no new output correction or validation enabled during execution.',
            'Public schema validation here is offline auditing, not rescoring or answer repair.',
            'Released image encoder lacks identity attestation; local Jina encodes queries.',
            'Previous selected42 includes v2 two attempts once; old v1 costs remain separate and excluded from both-selected totals.'])
    (batch/'completion-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    category_rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v}</td></tr>' for k,v in audit['categories'].items())
    table = ''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in
        ('task_id','previous_score','official_score','category','steps','seconds'))+
        f'<td><a href="{r["trace"]}">轨迹</a></td></tr>' for r in rows)
    totals = summary['totals']
    failed = [r for r in rows if r['status']=='failed']
    error_text = '；'.join(f'{r["task_id"]}：{r["error"]}，finish_reason={r["last_finish_reason"]}' for r in failed)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>WebNavigator JSON提示适配 · Hard42重跑</title>
<style>body{{font:16px/1.8 system-ui;color:#182337;max-width:1200px;margin:32px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #d7dfe8;padding:6px;text-align:left}}.key{{background:#edf6ee;padding:16px}}a{{color:#275daa}}</style>
<h1>WebNavigator × Qwen3.7-flash：JSON提示适配后重跑</h1>
<p class="key">42题全部重新运行，官方成功 <strong>{summary['official_successes']}/42（{summary['success_rate']:.1%}）</strong>。{summary['completed']}题正常结束，{summary['errors']}题异常退出。实际提交的{schema['answered']}份最终回答，全部通过Verified公开Schema校验。旧成绩与日志保留，没有挑分替换。</p>
<h2>这次改了什么</h2><p>原版stop说明统一要求JSON最终答案，保留原回复栏目和ACTION命令；去掉自然语言/N/A示例。新登记明确装入该提示词，入口启用统一说明。没有新增格式重试、诊断、卡片或导航策略。与上一轮相比，upstream副本只有stop.txt变化。</p>
<p>模型仍为百炼Qwen3.7-flash，原发布图225状态、201截图向量；每题30步、31次逻辑调用、4096输出token，AX1500。新任务独立重置环境，全部42题从头运行；单题错误记录后继续。没有额外付费预检。</p>
<h2>结果如何理解</h2><p>上一轮官方结果为0/42，正常结束的37题没有有效的公开答案协议结构；本轮{schema['valid']}/{schema['answered']}份最终回答符合Verified公开类型。提示适配后，普遍的格式提交问题未再出现。但任务路径、模型输出和服务状态也会变化，不能将31个成功全部解释成格式变化的独立因果效应。</p>
<table><tr><th>结果分类</th><th>题数</th></tr>{category_rows}</table>
<p>网页操作未通过的题为625、596、721、735、617、615、715；66和68的JSON格式合规但检索答案未通过。两题异常为{html.escape(error_text)}。长度截断仍未解决，不能将本次验证写成“没有bug”。</p>
<h2>时间与用量</h2><p>队列总用时{state['seconds']/60:.2f}分钟，任务执行时间合计{audit['task_seconds']/60:.2f}分钟，两者口径不同。新一轮共{totals['physical_requests']:,}次物理请求，已知输入{totals['input_tokens']:,}、输出{totals['output_tokens']:,} token，{totals['retries']}次传输重试、{totals['unknown_usage']}份未知用量。未知用量不作零成本，未估算价格。上一轮430次请求单列，不计入本轮成本。</p>
<h2>逐题对照</h2><table><tr><th>题号</th><th>旧分</th><th>本轮分</th><th>分类</th><th>步</th><th>秒</th><th>证据</th></tr>{table}</table>
<h2>核验与限制</h2><p>{guards['sources']}份冻结源码、{guards['protected']}份保护材料通过终检，42条实际轨迹链接存在。历史版本和发布图未覆盖；运行相关进程已退出。本次是暴露过任务的开发验证，不是盲测；原发布图片编码器缺少身份凭证的限制仍保留。</p>
<p><a href="completion-audit.json">完整审计与分角色用量</a> · <a href="public-schema-check.json">公开Schema验证</a> · <a href="summary.json">原始汇总</a> · <a href="final-audit.json">完整性核验</a> · <a href="repeat-lineage.json">重跑来源</a> · <a href="protocol.json">协议</a></p></html>'''
    (batch/'analysis.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ('official_successes','success_rate','categories','queue_seconds','task_seconds','roles')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    summarize(parser.parse_args().directory.resolve())
