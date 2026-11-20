"""Summarize completed baseline attempts without changing scores or raw evidence."""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(batch):
    summary = read(batch / 'summary.json')
    protocol = read(batch / 'protocol.json')
    state = read(batch / 'state.json')
    guards = read(batch / 'final-audit.json')
    assert summary['resolved'] == summary['official_scored'] == 42
    assert guards['guards_passed'] and state['status'] == 'completed'
    rows, roles, requests = [], defaultdict(Counter), set()
    for item in summary['tasks']:
        folder = Path(item['source'])
        result, official = read(folder / 'result.json'), read(folder / 'official.json')
        answer = result.get('answer')
        try:
            parsed = json.loads(answer) if isinstance(answer, str) else answer
            answer_type = 'json_object' if isinstance(parsed, dict) else 'json_other'
        except (ValueError, TypeError):
            parsed, answer_type = None, 'non_json' if answer else 'empty'
        # Only public schema presence, never the answer contents or reference answers.
        fields_present = isinstance(parsed, dict) and all(k in parsed for k in
            ('task_type', 'status', 'retrieved_data', 'error_details'))
        components = [{k: e.get(k) for k in ('evaluator_name', 'status', 'score')}
                      for e in official.get('evaluators_results', [])]
        network = [e for e in components if e['evaluator_name'] == 'NetworkEventEvaluator']
        events = [json.loads(line) for line in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        starts = [e for e in events if e['event'] == 'responses_started']
        receipts = {e['request_id']: e for e in events if e['event'] == 'responses_received'}
        for event in starts:
            request_id = event['request_id']
            assert request_id not in requests
            requests.add(request_id)
            receipt = receipts.get(request_id, {})
            usage = receipt.get('usage', {})
            roles[event['purpose']].update(physical_requests=1,
                input_tokens=usage.get('input_tokens', 0), output_tokens=usage.get('output_tokens', 0),
                unknown_usage=int(not usage), response_seconds=receipt.get('seconds', 0))
        trace = batch / item['trace']
        assert trace.is_file(), trace
        finish_reason = None
        if starts:
            response = folder / 'llm' / starts[-1]['request_id'] / 'chat-response.json'
            if response.is_file():
                data = read(response)
                finish_reason = (data.get('choices') or [{}])[0].get('finish_reason')
        rows.append({**item, 'answer_type': answer_type, 'public_answer_fields_present': fields_present,
            'components': components, 'all_network_checks_passed': bool(network) and all(e['status'] == 'success' for e in network),
            'last_provider_finish_reason': finish_reason})
    assert len({r['task_id'] for r in rows}) == 42
    assert len(requests) == summary['totals']['physical_requests']
    previous = read(batch.parent / 'webnavigator-flash-hard42-20261108-v1' / 'summary.json')['totals']
    v2_state = read(batch.parent / 'webnavigator-flash-hard42-20261108-v2' / 'state.json')
    audit = dict(status='completed', guards=guards, tasks=rows,
        official_successes=summary['official_successes'], official_scored=42,
        official_success_rate=summary['success_rate'], errors=summary['errors'],
        answer_types=dict(Counter(r['answer_type'] for r in rows)),
        public_answer_fields_present=sum(r['public_answer_fields_present'] for r in rows),
        all_network_checks_passed=[r['task_id'] for r in rows if r['all_network_checks_passed']],
        totals_selected_42=summary['totals'], roles=dict(roles),
        previous_v1_excluded=previous,
        totals_all_attempts={k: summary['totals'][k] + previous[k] for k in summary['totals']},
        v3_queue_seconds=state['seconds'], v2_queue_seconds=v2_state['seconds'],
        selected_task_seconds=sum(r['seconds'] or 0 for r in rows),
        inherited_tasks=[r['task_id'] for r in rows if r['inherited']],
        provider=protocol['provider'], actual_models=sorted({m for r in rows for m in r['actual_models']}),
        source_summary_sha256=hashlib.sha256((batch/'summary.json').read_bytes()).hexdigest(),
        trace_links_checked=42,
        limitations=['Network checks are component results, not alternative task success scores.',
            'Public answer schema presence is only a format check, not semantic correctness.',
            'No reference answers or raw HAR contents were exported.',
            'Original released image encoder identity is not attested; local Jina encodes queries.',
            'Exposed development tasks; no blind generalization or component causal claims.',
            'Selected 42 include the two v2 attempts once; v1 is excluded and separately costed.'])
    (batch/'completion-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    table = ''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in
        ('task_id','status','official_score','steps','seconds','answer_type','all_network_checks_passed'))+
        '<td>'+html.escape(r['error'] or '—')+'</td><td><a href="'+r['trace']+'">轨迹</a></td></tr>' for r in rows)
    errors = ''.join('<li>'+str(r['task_id'])+'：'+html.escape(r['error'] or '')+
        ('；服务端 finish_reason='+html.escape(str(r['last_provider_finish_reason'])) if r['last_provider_finish_reason'] else '')+'</li>'
        for r in rows if r['status']=='failed')
    t = summary['totals']
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>WebNavigator Flash · Hard42 完整结果</title>
<style>body{{font:16px/1.8 system-ui;max-width:1350px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #d9e0e9;padding:6px;text-align:left;overflow-wrap:anywhere}}.notice{{padding:16px;background:#fff4d9}}a{{color:#2459b5}}</style>
<h1>WebNavigator × Qwen3.7-flash：Reddit Hard42</h1>
<p>42题均已尝试并获得官方评分。正常结束37题，异常退出5题；官方成功 <strong>{summary['official_successes']}/42（{summary['success_rate']:.1%}）</strong>。错误保留后继续下一题，没有挑分或重跑。</p>
<p class="notice">不能把0/42直接解释成42题都没有完成网页操作：28题的全部网络操作检查通过，但42题的答案检查全部失败。32份最终回答不是JSON，5题没有最终回答，3份只有数据对象，2份是其他JSON值；没有一份含齐公开回答协议的四个字段。格式是明确障碍，但未经修复重跑，不能据此推算成功率，也不能认定其余失败全由格式造成。</p>
<h2>运行条件与来源</h2><p>原发布225状态、201截图向量；原版Agent、提示词和导航。Actor及页面选择均使用百炼Qwen3.7-flash，每题30步、31逻辑调用、4096输出token、AX1500。保留已登记的传输、动作语法及Verified接口兼容，以及最小栏目解析修复；没有本方法的卡片、诊断、学习或增图。</p>
<p>本次v3继承v2的646与598原始结果及成本，只新增剩余40题。与v2相比只修改队列和报告逻辑，逐题错误不再结束整个队列。此前v1解析异常的两次尝试单独保留，不进入这42题成绩。公开答案协议已经提供，但输出没有按协议形成有效最终提交。以上是当前适配配置的结果，不是原项目论文成绩，也不是对模型能力的单独测量。</p>
<h2>异常条目</h2><ul>{errors}</ul><p>410的HTTP400为服务端内容审核拒绝；730为逻辑调用额度耗尽。其他三项为未完成响应。没有把这些错误当作浏览器动作重试，也没有补写答案。</p>
<h2>时间与用量</h2><p>选定42题共{t['physical_requests']:,}次物理请求，已知输入{t['input_tokens']:,}、输出{t['output_tokens']:,} token，{t['retries']}次传输重试、{t['unknown_usage']}份未知用量。未知部分不能当作零成本，未估算价格。</p>
<p>v3队列用时{state['seconds']/60:.2f}分钟（40个新任务及准备、重置、核验）；前两题所在v2队列另用{v2_state['seconds']/60:.2f}分钟。42题记录的执行时间合计{audit['selected_task_seconds']/60:.2f}分钟，口径不含全部队列开销。旧v1另有27次请求、343,411输入和28,663输出token，不与v2重复计数。</p>
<h2>逐题结果</h2><p>“网络全通过”仅表示该题的全部NetworkEventEvaluator检查通过，并非另一套任务成功分。完整原始评分保持不变。</p>
<table><tr><th>题号</th><th>状态</th><th>官方分</th><th>步</th><th>秒</th><th>回答类型</th><th>网络全通过</th><th>异常</th><th>证据</th></tr>{table}</table>
<h2>核验与限制</h2><p>246份冻结源码、2136份保护材料通过终检；42份实际轨迹链接存在。原始图片编码器缺少身份凭证，本地Jina用于查询编码，此限制仍存在。任务已用于开发，不能作盲测泛化或组件因果结论。</p>
<p><a href="completion-audit.json">完整汇总审计</a> · <a href="summary.json">原队列汇总</a> · <a href="final-audit.json">守卫核验</a> · <a href="continuation.json">继续运行登记</a> · <a href="protocol.json">协议</a></p></html>'''
    (batch/'analysis.html').write_text(page, encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ('official_successes','official_scored','answer_types','public_answer_fields_present','trace_links_checked')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    summarize(parser.parse_args().directory.resolve())
