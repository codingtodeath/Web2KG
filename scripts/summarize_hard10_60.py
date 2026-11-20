"""Combine the ten unique scored attempts, retaining the budget stop and queue boundary."""
from collections import defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_reddit_hard10_225 as run
from web2kg.evaluation.actor_output import single_action

FIRST = ROOT / 'runs/reddit-hard10-60-flash-20261106-v1'
REST = ROOT / 'runs/reddit-hard10-60-flash-20261106-remainder-v1'
ORDER = [646, 598, 625, 596, 721, 31, 730, 409, 651, 644]
LOGS = {'actor_selector': 'events.jsonl', 'description': 'graph-annotations/events.jsonl',
        'cards_framework': 'graph-cards/events.jsonl', 'diagnosis_learning': 'diagnosis-calls/events.jsonl'}


def aggregate(events):
    starts = [e for e in events if e['event'] == 'responses_started']
    received = {e['request_id']: e for e in events if e['event'] == 'responses_received'}
    return dict(physical_requests=len(starts), logical_calls=sum(not e.get('output_retry_number') for e in starts),
        input_tokens=sum(e.get('usage', {}).get('input_tokens', 0) for e in received.values()),
        output_tokens=sum(e.get('usage', {}).get('output_tokens', 0) for e in received.values()),
        unknown_usage=sum(e['request_id'] not in received or not received[e['request_id']].get('usage') for e in starts),
        format_retries=sum(e['event'] == 'responses_output_retry' for e in events),
        transport_retries=sum(e['event'] == 'responses_retry' for e in events),
        actual_models=sorted({e.get('actual_model', 'unknown') for e in received.values()}))


def main():
    states = {b: run.read(b / 'state.json') for b in (FIRST, REST)}
    assert states[FIRST]['status'] == 'needs_attention' and states[REST]['status'] == 'completed'
    rows, retry_rows, all_events, by_role = [], [], [], defaultdict(list)
    guards = []
    for b in (FIRST, REST):
        checks = [e for e in run.read_events(b / 'timings/events.jsonl')
                  if e['event'] == 'integrity_verified' and e.get('scope') == 'full']
        guards.append(dict(directory=str(b), receipt=checks[-1],
            frozen_files=len(run.read(b / 'frozen-runtime/manifest.json'))))
    for task in ORDER:
        b = FIRST if task in ORDER[:7] else REST
        folder = b / str(task); result = run.read(folder / 'result.json')
        official = run.read(folder / 'official.json')
        assert official.get('score') in (0, 1)
        update = result.get('execution_graph') or {}; retention = update.get('retention') or {}
        events = []
        for role, sub in LOGS.items():
            part = run.read_events(folder / sub); events.extend(part); by_role[role].extend(part)
        all_events.extend(events)
        trace = b / 'trajectories' / f'task-{task}.html'; assert trace.is_file()
        recovery = result.get('recovery') or {}
        row = dict(task_id=task, run=b.name, state=states[b]['tasks'][str(task)]['status'],
            official_score=official['score'], result_status=result['status'], error=result.get('error'),
            steps=result['steps'], task_seconds=result['seconds'], total_seconds=result['total_seconds'],
            maintenance_seconds=update.get('finalization_seconds'),
            queue_task_seconds=states[b]['tasks'][str(task)]['seconds'],
            graph_before_pruning=retention.get('before'), graph_after=retention.get('after'),
            removed_nodes=len(retention.get('removed', [])), index_status=update.get('index_status'),
            pending_cards=(update.get('card_refresh') or {}).get('pending', []),
            unavailable_cards=(update.get('card_refresh') or {}).get('unavailable', []),
            diagnoses=recovery.get('diagnoses', 0), return_attempted=recovery.get('attempted', False),
            return_result=recovery.get('return'), refunded_steps=recovery.get('step_refund', 0),
            usage=aggregate(events), trace=trace.relative_to(FIRST).as_posix() if b == FIRST
              else '../' + REST.name + '/' + str(trace.relative_to(REST)).replace('\\', '/'),
            result_sha256=run.file_hash(folder / 'result.json'), official_sha256=run.file_hash(folder / 'official.json'))
        rows.append(row)
        actor_events = run.read_events(folder / 'events.jsonl')
        for i, event in enumerate(actor_events):
            if event['event'] != 'responses_output_retry':
                continue
            failed = event['request_id']
            following_index = next(j for j in range(i+1, len(actor_events)) if actor_events[j]['event'] == 'responses_started')
            following = actor_events[following_index]['request_id']
            a = run.read(folder / 'llm' / failed / 'request.json')
            z = run.read(folder / 'llm' / following / 'request.json')
            response = run.read(folder / 'llm' / failed / 'response.json')
            text = '\n'.join(c.get('text', '') for m in response.get('messages', []) for c in m.get('content', []))
            same_input = a['input'] == z['input']
            no_action = not any(e['event'] == 'agent_action' for e in actor_events[i+1:following_index])
            assert same_input and no_action
            retry_rows.append(dict(task_id=task, failed_request=failed, corrected_request=following,
                retry_number=event['retry_number'], same_input=same_input, browser_action_between=False,
                failure_kind=event.get('failure_kind'),
                shape='bare_single_action_without_ACTION_heading' if single_action(text.strip()) else 'other'))
    received = [e['request_id'] for e in all_events if e['event'] == 'responses_received']
    assert all(row['graph_after'] == 60 for row in rows)
    assert len(received) == len(set(received)), 'No inherited request may be counted twice'
    usage = aggregate(all_events); assert usage['transport_retries'] == 0  # logical count formula above
    source_receipt = run.read(ROOT / 'runs/actor-single-action-dev2-20261106-v1/audit.json')
    assert all(run.file_hash(Path(p)) == h for p, h in source_receipt['protected_sha256'].items())
    graph = run.Graph.load(Path(states[REST]['current_graph']))
    assert len(graph.nodes) == 60 and set(run.read(graph.directory / 'index.json')['ids']) == set(graph.nodes)
    initial = run.Graph.load(FIRST / 'initial-graph'); assert len(initial.nodes) == 60
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(), planned=10, attempted=10, official_scored=10,
        successes=sum(r['official_score'] == 1 for r in rows), complete_handoffs=9,
        success_rate=sum(r['official_score'] == 1 for r in rows)/10,
        scope='Ten exposed developmental tasks; identical Agent/model source across two frozen queues. No controlled graph-size or prompt-effect comparison.',
        boundary='730 budget exhaustion and original needs_attention status retained. Its incomplete cards are not inherited. Only409/651/644 continue from the complete31 handoff. No task retried or replaced.',
        rows=rows, usage=usage, roles={k: aggregate(v) for k,v in by_role.items()}, actor_format_retries=retry_rows,
        queue_seconds=sum(s['total_seconds'] for s in states.values()),
        task_seconds=sum(r['task_seconds'] for r in rows), task_and_maintenance_seconds=sum(r['total_seconds'] for r in rows),
        maintenance_seconds=sum(r['maintenance_seconds'] or 0 for r in rows),
        guards=guards, prior_pause_and_prompt_guards_unchanged=True, original_graph_preserved=True,
        initial_graph_nodes=60, final_graph_nodes=len(graph.nodes), final_graph_edges=len(graph.data['edges']),
        final_index_pages=len(run.read(graph.directory / 'index.json')['ids']),
        paid_preflight_requests=0, separate_preparation_requests=0, prices_estimated=False)
    run.write_json(FIRST / 'combined-audit.json', summary)
    esc = html.escape
    table = ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in [
        f'<a href="{esc(r["trace"],quote=True)}">{r["task_id"]}</a>', int(r['official_score']), r['steps'],
        f'{r["task_seconds"]:.1f}', f'{r["maintenance_seconds"]:.1f}', r['usage']['physical_requests'],
        r['usage']['format_retries'], '额度耗尽，未交接' if r['task_id']==730 else '完成']) + '</tr>' for r in rows)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reddit Hard10 · 60节点 · 最终汇总</title><style>body{{font:17px/1.65 system-ui;max-width:1150px;margin:36px auto;padding:0 20px;color:#17202a}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #d9dfe5;text-align:left}}a{{color:#155f9d}}.note{{background:#f2f5f8;padding:16px;border-radius:8px}}.scroll{{overflow-x:auto}}</style>
<h1>Reddit Hard10 · 60节点 · 最终汇总</h1>
<p>全部 Qwen3.7-flash，百炼直连。10题均有一次新尝试和官方评分，<b>{summary['successes']}/10（{summary['success_rate']:.0%}）通过</b>；9题完整交接，730保留预算耗尽失败。</p>
<div class="note">730执行28步后耗尽31次Actor调用（AX搜索也消耗调用，但不算浏览器步）。现有代码对所有model_error禁止继续制卡，留下待更新/不可用卡片；原始失败、官方0分与全部成本保留。未重跑730。尚未开始的409、651、644在独立队列从31的完整图和经验接续，方法代码与额度不变。不是单一不中断队列，也不是挑选最高分。</div>
<h2>逐题结果</h2><div class="scroll"><table><tr><th>题目/实际轨迹</th><th>官方分</th><th>步数</th><th>执行秒</th><th>维护秒</th><th>物理请求</th><th>格式重生成</th><th>结尾</th></tr>{table}</table></div>
<h2>格式与维护</h2><p>Actor格式重生成{len(retry_rows)}次，已核验均保持同一输入且纠错期间不执行浏览器动作。本批这些Actor输出均为缺少ACTION标题的单个stop命令，没有观察到多动作输出；多动作拦截能力由先前离线测试覆盖。后台描述的JSON重生成另计{summary['roles']['description']['format_retries']}次，卡片另计{summary['roles']['cards_framework']['format_retries']}次。所有格式纠错最终恢复，未发生格式错误耗尽重生成次数而中断。十题均未触发根因诊断，不能据此评估自动返回或经验学习效果。</p>
<p>此次仍有两个可改进点：明确的裸单动作可以使用无损兼容，减少重复请求；正常调用额度耗尽不宜与永久模型错误共用禁止制卡的收尾分支。本次没有改动这些执行规则或提高额度。每题裁剪后的图均为60节点；730资产不作为交接图。最终有效图为60节点、{len(graph.data['edges'])}边、60页索引，原图和发布指针保留。</p>
<h2>本批用量与时间</h2><p>共{usage['physical_requests']}次物理请求，输入{usage['input_tokens']:,}、输出{usage['output_tokens']:,} token；未知用量{usage['unknown_usage']}，传输重试{usage['transport_retries']}。包括730失败和全部格式尝试，不含此前实验或建图；没有新增付费预检、准备制卡，也未估算价格。</p>
<p>两个队列记录合计{summary['queue_seconds']/60:.2f}分钟；各题执行合计{summary['task_seconds']/60:.2f}分钟、维护合计{summary['maintenance_seconds']/60:.2f}分钟。队列时间包含启动、复制、环境重置和核验，人工检查与接续登记间隔不含在内；并行请求时长不能相加当作墙钟时间。</p>
<p>这些任务与知识曾被开发使用，且中途有明确接续边界。没有同版225节点或旧提示词对照，不能据此证明60节点更优、提示词带来成功率提升或诊断的因果收益。</p>
<p><a href="combined-audit.json">完整分角色用量与审计</a> · <a href="report.html">前段原始报告</a> · <a href="../{REST.name}/report.html">剩余三题报告</a> · <a href="budget-exhaustion-diagnosis.json">730原因记录</a> · <a href="initial-graph/viewer.html">保留的初始60节点图</a> · <a href="../{REST.name}/644/execution-graph/viewer.html">最终图</a></p></html>'''
    (FIRST / 'analysis.html').write_text(page, encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('successes','usage','queue_seconds','task_seconds','maintenance_seconds','final_graph_nodes','final_graph_edges')}))


if __name__ == '__main__':
    main()
