"""Audit completed AgentOccam evidence without model calls or rescoring."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        self.urls.extend(v for k, v in attrs if v and k in ('href', 'src'))


def check_links(page):
    parser = Links()
    parser.feed(page.read_text(encoding='utf-8'))
    checked = 0
    for value in parser.urls:
        link = urlsplit(value)
        if link.scheme or not link.path:
            continue
        target = (page.parent / unquote(link.path)).resolve()
        if target.suffix.lower() == '.har' or 'auth' in target.parts:
            raise ValueError(f'Private evidence linked from {page.name}')
        if not target.is_file():
            raise FileNotFoundError(target)
        checked += 1
    return checked


def summarize(batch):
    s, p = read(batch/'summary.json'), read(batch/'protocol.json')
    state, guards = read(batch/'state.json'), read(batch/'final-audit.json')
    lineage = read(batch/'continuation.json')
    schema = read(batch/'public-schema-check.json')
    old = Path(lineage['previous'])
    failed = Path(lineage['prior_repair_lineage']['previous'])
    reference = Path(p['reference'])
    ws = read(reference/'summary.json')
    assert state['status'] == s['status'] == 'completed' and guards['guards_passed']
    assert s['resolved'] == s['official_scored'] == len(s['tasks']) == 42
    assert [r['task_id'] for r in s['tasks']] == p['task_ids'] == [r['task_id'] for r in ws['tasks']]
    assert len(set(p['task_ids'])) == 42
    assert len(p['inherited_tasks']) == 26 and len(p['run_task_ids']) == 16
    assert set(map(int, state['tasks'])) == set(p['run_task_ids'])
    for path, expected in read(batch/'registration-integrity.json').items():
        assert digest(Path(path)) == expected, path
    old_manifest, manifest = read(old/'frozen-runtime/manifest.json'), read(batch/'frozen-runtime/manifest.json')
    changed = [k for k, v in old_manifest.items() if manifest.get(k) != v]
    added = [k for k in manifest if k not in old_manifest]
    assert changed == ['scripts/run_agentoccam_flash_hard.py']
    assert added == ['scripts/continue_agentoccam_flash_hard.py']
    assert sum(manifest.get(k) == v for k, v in old_manifest.items()) == 334
    valid = {r['task_id']: r for r in schema['tasks']}
    wr = {r['task_id']: r for r in ws['tasks']}
    roles, pairs, rows, requests = defaultdict(Counter), defaultdict(list), [], set()
    local_links = 0
    for item in s['tasks']:
        folder = Path(item['result_folder'])
        result, official = read(folder/'result.json'), read(folder/'official.json')
        assert official['score'] == item['official_score'] in (0, 1)
        assert read(folder/'eval_result.json') == official
        assert result['steps'] == item['steps'] and result['status'] == item['agent_status']
        local_links += check_links((batch/item['trace']).resolve())
        components = [{k: e.get(k) for k in ('evaluator_name', 'status', 'score')}
                      for e in official['evaluators_results']]
        answer_fields = []
        for e in official['evaluators_results']:
            a, expected = e.get('actual_normalized'), e.get('expected')
            if isinstance(a, dict) and isinstance(expected, dict) and 'task_type' in expected:
                # Export field names only, never evaluator reference answers.
                answer_fields = [k for k in expected if a.get(k) != expected[k]]
        network_failed = any(e['evaluator_name'] == 'NetworkEventEvaluator' and e['status'] != 'success'
                             for e in components)
        answer_failed = any(e['evaluator_name'] == 'AgentResponseEvaluator' and e['status'] != 'success'
                            for e in components)
        if item['official_score'] == 1:
            category = '成功'
        elif item['agent_status'] == 'step_limit' or result.get('error') == 'ResponsesError: Responses call budget exhausted':
            category = '步数或调用额度耗尽'
        elif not valid[item['task_id']]['public_schema_valid']:
            category = '最终答案格式不合规'
        elif network_failed and answer_failed:
            category = '网页操作与答案均未通过'
        elif network_failed:
            category = '网页操作未全部通过'
        else:
            category = '网页操作已通过或无此项，答案未通过'
        es = [json.loads(line) for line in (folder/'events.jsonl').read_text('utf-8').splitlines() if line.strip()]
        starts = [e for e in es if e['event'] == 'responses_started']
        received = {e['request_id']: e for e in es if e['event'] == 'responses_received'}
        assert len(starts) == item['physical_requests']
        assert len(received) == len(starts)  # This completed selection has no unknown usage.
        tokens = Counter()
        for event in starts:
            key = event['request_id']
            assert key not in requests
            requests.add(key)
            receipt = received[key]
            usage = receipt.get('usage') or {}
            assert usage and receipt['actual_model'] == p['model']
            values = dict(physical_requests=1, input_tokens=usage.get('input_tokens', 0),
                          output_tokens=usage.get('output_tokens', 0), unknown_usage=int(not usage))
            tokens.update(values)
            roles[event['purpose']].update(values)
            roles[event['purpose']]['response_seconds'] += receipt.get('seconds', 0)
        assert all(tokens[k] == item[k] for k in ('physical_requests', 'input_tokens', 'output_tokens', 'unknown_usage'))
        a, w = int(item['official_score']), int(wr[item['task_id']]['official_score'])
        pairs[f'{a}/{w}'].append(item['task_id'])
        rows.append({**item, 'category': category, 'components': components,
                     'answer_field_differences': answer_fields,
                     'public_schema_valid': valid[item['task_id']]['public_schema_valid'],
                     'webnavigator_score': w,
                     'webnavigator_trace': Path(os.path.relpath(reference/wr[item['task_id']]['trace'], batch)).as_posix()})
    assert len(requests) == s['totals']['physical_requests']
    assert sum(r['official_score'] for r in rows) == s['official_successes']
    for field in s['totals']:
        assert sum(r[field] for r in rows) == s['totals'][field]
    interrupted = s['interrupted_attempts']
    assert len(interrupted) == 1 and interrupted[0]['task_id'] == 615
    interrupted_usage = {k: sum(r[k] for r in interrupted) for k in s['totals']}
    failed_summary = read(failed/'summary.json')
    costs = {'selected_42': s['totals'], 'user_interrupted_615': interrupted_usage,
             'engineering_failed_v1': failed_summary['totals']}
    costs['all_agentoccam_attempts'] = {k: sum(v[k] for v in costs.values()) for k in s['totals']}
    task_seconds = sum(r['seconds'] for r in rows)
    queue_seconds = lineage['previous_queue_seconds'] + s['seconds']
    audit = dict(status='completed', created_utc=datetime.now(timezone.utc).isoformat(),
                 official_successes=s['official_successes'], success_rate=s['success_rate'],
                 selected_tasks=42, inherited_tasks=26, new_tasks=16,
                 new_successes=sum(r['official_score'] for r in rows if not r['inherited']),
                 missing_tasks=[], unrun_tasks=[], official_scores_checked=42,
                 schema_validation=schema, guards=guards,
                 registration_hashes_checked=len(read(batch/'registration-integrity.json')),
                 inherited_method_files_unchanged=334, changed_runtime_files=changed,
                 added_runtime_files=added, trace_pages_checked=42, trace_local_links_checked=local_links,
                 categories=dict(Counter(r['category'] for r in rows)),
                 agent_statuses=dict(Counter(r['agent_status'] for r in rows)),
                 paired_ids=dict(pairs), model=p['model'], provider=p['provider'], roles=dict(roles),
                 cost_groups=costs, tasks=rows, selected_task_seconds=task_seconds,
                 active_queue_seconds_including_interruption=queue_seconds,
                 new_queue_seconds=s['seconds'], failed_v1_queue_seconds=failed_summary['seconds'],
                 reference=dict(directory=str(reference), successes=ws['official_successes'],
                                totals=ws['totals'], selected_task_seconds=sum(r['seconds'] for r in ws['tasks'])),
                 limits=[
                     'Development tasks exposed in prior runs, one selected attempt per task; no generalization or causal claim.',
                     '26 completed results retained by reference; only the user-cancelled attempt is excluded and separately costed.',
                     'AgentOccam native text observation/history/branch/prune; 1000-row serializer, no graph or selector.',
                     'WebNavigator uses AX1500, released 225-state graph/201 vectors and a selector sharing its call budget.',
                     'Both use Qwen3.7-flash and Verified JSON stop contract, but action syntax compatibility differs.',
                     'AgentOccam retains native type parsing; WebNavigator normalizes terminal press_enter_after=N. Task625 illustrates the gap.',
                     'No graph/cards/construction/preflight costs for this AgentOccam experiment. No price estimate.',
                     '39 valid JSON answers do not imply semantic correctness. Three budget endings have no answer.',
                 ],
                 evidence_hashes={str(q): digest(q) for q in [batch/'summary.json', batch/'state.json',
                     batch/'final-audit.json', batch/'public-schema-check.json', reference/'summary.json']})
    (batch/'completion-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    esc = html.escape
    cat_table = ''.join(f'<tr><td>{esc(k)}</td><td>{v}</td></tr>' for k, v in audit['categories'].items())
    cost_labels = {'selected_42': '选定42题（继承26＋新跑16）', 'user_interrupted_615': '用户暂停的旧615（不计成功率）',
                   'engineering_failed_v1': '早期工程失败v1（2次尝试）', 'all_agentoccam_attempts': 'AgentOccam全部45次尝试合计'}
    cost_table = ''.join('<tr><td>'+cost_labels[k]+'</td>'+''.join(f'<td>{v[f]:,}</td>' for f in
                          ('physical_requests','input_tokens','output_tokens','unknown_usage','retries'))+'</tr>' for k, v in costs.items())
    task_table = ''.join(f'<tr><td>{r["task_id"]}</td><td>{int(r["official_score"])}</td><td>{r["webnavigator_score"]}</td>'
                        f'<td>{esc(r["category"])}</td><td>{"继承" if r["inherited"] else "新跑"}</td><td>{r["steps"]}</td>'
                        f'<td>{r["seconds"]:.1f}</td><td><a href="{esc(r["trace"])}">Occam</a> · '
                        f'<a href="{esc(r["webnavigator_trace"])}">WN</a></td></tr>' for r in rows)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgentOccam × Qwen3.7-flash · Reddit Hard42</title>
<style>body{{font:16px/1.8 system-ui,sans-serif;color:#192b3f;max-width:1160px;margin:30px auto;padding:0 22px}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d6dfe7;padding:7px;text-align:left}}th{{background:#f0f4f8}}a{{color:#2259a7}}.key{{background:#edf5ef;border-left:4px solid #34765a;padding:16px}}.scroll{{overflow-x:auto}}.note{{color:#485e73}}</style>
<h1>AgentOccam × Qwen3.7-flash：Reddit Hard42</h1>
<p class="key">42题全部结束，官方成功 <strong>27/42（64.3%）</strong>。本次续跑16题成功12题；暂停前的26题成功15题，全部按原结果引用，没有挑分。队列与worker均已退出，最终保护核验通过。</p>
<h2>与WebNavigator的对照</h2><table><tr><th>方法</th><th>模型</th><th>成功</th><th>成功率</th><th>任务执行时间合计</th></tr>
<tr><td>AgentOccam</td><td>Qwen3.7-flash</td><td>27/42</td><td>64.3%</td><td>{task_seconds/60:.2f}分钟</td></tr>
<tr><td>WebNavigator</td><td>Qwen3.7-flash</td><td>31/42</td><td>73.8%</td><td>{audit['reference']['selected_task_seconds']/60:.2f}分钟</td></tr></table>
<p>两者共同成功23题；AgentOccam独自成功4题（721、66、715、67），WebNavigator独自成功8题（724、409、408、407、597、638、581、618），共同失败7题。AgentOccam本轮少成功4题，差9.5个百分点。这是一次开发样本上的整套系统比较，不能推出模型或某个组件本身的因果结论。</p>
<h2>保留了什么，适配了什么</h2>
<p>AgentOccam基于官方仓库提交 <code>c078ba629212ea7cee35b2718dc8df72c05e57c7</code>，采用基础Actor配置，保留原生页面裁剪、历史、highlight、branch/prune和动作处理。没有图谱、Jina、selector、额外卡片或根因诊断。两组均使用百炼直连、Qwen3.7-flash、每题30轮决策／31次逻辑调用／4096输出token、相同任务顺序与站点重置、相同Verified评分。</p>
<p>适配包含供应商传输、独立worker导入、整行栏目识别、Verified JSON stop说明、本地登录与评分采集。早期工程修复避免复制随后就丢弃的整棵父子树，并跳过全文AX模式下未使用的逐节点几何预查询；真实动作定位仍查当前DOM。修复后从头登记v2，旧失败版本单独保留。</p>
<p class="note">两组输入与接口仍不完全相同：AgentOccam是文本观察，原生序列化上限1000行；WebNavigator使用AX1500、225状态发布图、201截图向量和selector。AgentOccam的note/规划也计决策轮次，原生格式重生成计调用额度。WebNavigator另有动作语法兼容；AgentOccam保留原生type解析。不能把这张表当作等输入、等计算或论文成绩复现。</p>
<h2>失败原因与格式检查</h2><table><tr><th>分类</th><th>题数</th></tr>{cat_table}</table>
<p>39份真实最终答案全部通过Verified公开Schema，另外617、615、618没有最终答案：617和618达到30轮，615耗尽31次调用。这三次保留官方零分，没有补写回答。其余失败中，409、408的网页操作检查通过，但JSON里的任务类型填错；68的检索结果未通过。JSON合法不代表答案内容正确。</p>
<p>625还有明确的接口兼容限制：原生提示同时写了命名参数和位置参数示例。模型输出末尾的 <code>[press_enter_after=0]</code> 被原生解析器当成正文的一部分，页面实际出现了这段文字。WebNavigator已有将该末尾参数转为 <code>[0]</code> 的兼容处理，而本次AgentOccam未启用。未在运行中修补或重跑；这一点会影响比较口径，也不能据此断言修复后该题一定成功。</p>
<h2>暂停与续跑的来源</h2><p>用户暂停时，v2完成26题，615在14步后中断。新目录引用这26个完成结果，从615重新开始并完成后续15题；旧615仍保留原始评分、日志和STOP，用量单列。334份原方法文件与v2完全一致，变化只在续跑调度与报告。没有把工程失败当作用户暂停续跑，也没有继承早期v1成绩。</p>
<h2>时间与用量</h2><p>选定42题的任务执行时间合计{task_seconds/60:.2f}分钟。v2与本次续跑的队列实际运行时间合计{queue_seconds/60:.2f}分钟，包含旧615中断尝试和环境重置等开销，不包含用户暂停期间的隔夜间隔；早期失败v1另耗时{failed_summary['seconds']/60:.2f}分钟。本次新16题队列用时{s['seconds']/60:.2f}分钟。</p>
<div class="scroll"><table><tr><th>口径</th><th>物理请求</th><th>已知输入token</th><th>已知输出token</th><th>未知用量</th><th>传输重试</th></tr>{cost_table}</table></div>
<p>选定42题的449次请求全部为Actor，实际返回模型均为Qwen3.7-flash。WebNavigator对照为433次请求（Actor395、selector38），已知输入6,341,539、输出550,437 token，另有3份未知用量和3次重试。输出token含供应商计入的推理用量，不另加一遍。此次AgentOccam没有建图、制卡、编码或付费预检；未估算货币价格。旧v1的未知用量不能当作零成本。</p>
<h2>逐题评分与实际轨迹</h2><div class="scroll"><table><tr><th>题号</th><th>Occam</th><th>WN</th><th>Occam结果</th><th>来源</th><th>决策轮</th><th>执行秒</th><th>轨迹</th></tr>{task_table}</table></div>
<h2>核验记录</h2><p>{guards['sources']}份冻结源码、{guards['protected']}份保护材料通过结束核验；42份官方评分与结果一致，42条Occam轨迹及其本地资源存在。公开Schema检查只是离线审计，没有重评分、改答案或额外模型调用。旧版本证据、用户修改的提示词和发布图保持原样。</p>
<p><a href="completion-audit.json">汇总审计、配对结果与用量</a> · <a href="public-schema-check.json">公开Schema核验</a> · <a href="final-audit.json">守卫终检</a> · <a href="continuation.json">续跑来源</a> · <a href="summary.json">原始汇总</a> · <a href="protocol.json">协议</a> · <a href="../webnavigator-flash-hard42-20261110-v4/analysis.html">WebNavigator完整报告</a></p></html>'''
    (batch/'analysis.html').write_text(page, encoding='utf-8')
    check_links(batch/'analysis.html')
    print(json.dumps({k: audit[k] for k in ('official_successes','success_rate','categories','cost_groups',
                                          'selected_task_seconds','active_queue_seconds_including_interruption')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    summarize(parser.parse_args().directory.resolve())
