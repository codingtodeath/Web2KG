"""Qwen baseline and the twenty unstarted original-method tasks, in one owned queue."""
import argparse
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pending(protocol, state, active_key=None):
    schedule = protocol['schedule']
    keys = [f"{r['arm']}/{r['task_id']}" for r in schedule]
    if len(keys) != len(set(keys)) or len(keys) != 62:
        raise ValueError('Expected 42 Qwen and 20 unstarted ours tasks')
    for key, row in state['tasks'].items():
        if row['status'] != 'completed' and not (key == active_key and row['status'] == 'started'):
            raise ValueError('Started entry unresolved; never restart: ' + key)
    return [r for r in schedule if f"{r['arm']}/{r['task_id']}" not in state['tasks']]


def verify(batch, active_key=None):
    for path, expected in read(batch / 'registration-integrity.json').items():
        if digest(path) != expected:
            raise ValueError('Registration changed: ' + path)
    protocol, state = read(batch / 'protocol.json'), read(batch / 'state.json')
    if any(Path(p).exists() for p in protocol.get('absent_inputs', [])):
        raise ValueError('Previously absent protected input appeared')
    for path, expected in protocol['guards'].items():
        if digest(path) != expected:
            raise ValueError('Protected input changed: ' + path)
    if digest(Path(state['current_graph']) / 'graph.json') != state['current_graph_sha256']:
        raise ValueError('Cumulative graph changed outside queue')
    return protocol, state, pending(protocol, state, active_key)


def pinned(runtime):
    sys.path.insert(0, str(runtime))
    from scripts import run_hard_comparison
    return run_hard_comparison


def load_transport(path):
    spec = importlib.util.spec_from_file_location('registered_qwen_transport', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(batch, arm, task_id):
    protocol, _, _ = verify(batch, f'{arm}/{task_id}')
    runtime = Path(protocol['runtimes'][arm])
    original = pinned(runtime)
    config = read(batch / arm / str(task_id) / 'input.json')
    original.check_inputs(config)
    if arm == 'webnavigator':
        load_transport(protocol['transport']).install(protocol['credential_file'], batch / 'routing-state.json')
    original.run_worker(batch, arm, task_id)
    folder = batch / arm / str(task_id)
    official = original.score(config['verified_root'], batch / arm, task_id, config['logical_origin'])
    write(folder / 'official.json', official)
    if official.get('score') is None:
        raise RuntimeError('Official score missing')
    result = read(folder / 'result.json')
    events = [e for name in ('events.jsonl', 'diagnosis-calls/events.jsonl',
                            'graph-annotations/events.jsonl', 'graph-cards/events.jsonl')
              for e in original.read_events(folder / name)]
    recovered_routes = {e.get('request_id') for e in events if e['event'] == 'provider_fallback'}
    if result.get('status') == 'runtime_error' or any(
        e['event'] == 'responses_failed' and e.get('request_id') not in recovered_routes and
        (e.get('failure_kind') == 'cancelled' or e.get('http_status') in {400, 401, 402, 403, 404})
        for e in events):
        raise RuntimeError('Engineering or permanent service error; retain score and stop')
    handoff = {'official_score': official['score']}
    if arm == 'ours':
        graph = original.advance_graph(folder)
        learner = original.ResponsesClient(runtime / 'auth/llm-crawl.env', original.EventLog(folder / 'diagnosis-calls'),
            config['diagnosis_calls'], 2048, model=config['diagnosis_model'],
            retry_connections=True, stop_file=config['stop_file'])
        memory = original.ExperienceStore(config['experience_store'],
            original.memory_namespace(original.Graph.load(Path(config['graph']))))
        original.learn_experience(folder, learner, memory, success=official['score'] == 1,
                                  verifier='WebArena-Verified official score=1')
        handoff.update(current_graph=str(graph), current_graph_sha256=digest(graph / 'graph.json'))
    write(folder / 'handoff.json', handoff)


def export_local_trajectories(directory, exporter):
    """Export this stage's records; inherited trajectories stay at their original paths."""
    original_read = exporter.__globals__['read']

    def local_read(path, *args):
        value = original_read(path, *args)
        if path == directory / 'summary.json':
            value = {**value, 'tasks': [row for row in value['tasks']
                if Path(row['evidence_directory']).resolve() == (directory / str(row['task_id'])).resolve()]}
        return value

    # Scope the summary view to the offline exporter without modifying the frozen module.
    local_export = types.FunctionType(exporter.__code__, {**exporter.__globals__, 'read': local_read},
                                     exporter.__name__, exporter.__defaults__, exporter.__closure__)
    return local_export([directory], directory / 'trajectories')


def report(batch, original):
    protocol, state = read(batch / 'protocol.json'), read(batch / 'state.json')
    original.report(batch)
    summary = read(batch / 'summary.json')
    previous = read(Path(protocol['previous_batch']) / 'summary.json')['arms']['webnavigator']
    groups = [('本项目方法 · 原 Terra/Luna（保留29工程失败）', summary['arms']['ours']),
              ('WebNavigator · 原 Terra（已完成）', previous),
              ('WebNavigator · Qwen / OpenRouter优先、百炼后备', summary['arms']['webnavigator'])]
    target = Path(protocol['report_path'])
    parts = ['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="90">',
             '<title>Reddit Hard 三组比较</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:32px auto}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}pre{white-space:pre-wrap}</style>',
             '<h1>Reddit Hard · 本项目方法 / WebNavigator Terra / WebNavigator Qwen</h1>',
             '<p>42题固定分母。Qwen独立新增42次；本项目方法只续原20道未开始题，旧22题原样引用。29基础设施失败仍为0；其0动作、0模型调用的中断图不交接，继续上一份已验证累计图。三组运行时间不同，属于已暴露任务上的整体开发比较，不是盲测、泛化或模块因果证明。</p>',
             '<p>Qwen优先OpenRouter免费入口；每日额度耗尽或明确免费版本不可用时使用百炼。免费不可用时每30分钟检查目录；每日额度在UTC新一天重新尝试。临时限流/连接错误相同输入持续重试；永久错误停止。每次服务商、模型、请求、响应、tokens和未知用量保留。</p>',
             '<p>新队列状态：' + html.escape(state['status']) + '</p>']
    for label, group in groups:
        parts.append(f'<h2>{label}</h2><p>成功 {group["official_successes"]}/42；已评分 {group["official_scored"]}；未运行 {group["unrun"]}；已开始未评分 {group["started_unscored"]}。</p>')
        totals = {k: sum(m.get(k, 0) for r in group['tasks'] for m in r['models'].values()) for k in ('input_tokens', 'output_tokens')}
        parts.append('<p>已知tokens（输入/输出）：' + str(totals['input_tokens']) + '/' + str(totals['output_tokens']) +
                     '；描述性Wilson 95%区间（仅已评分）：' + html.escape(str(group.get('descriptive_wilson95_scored'))) + '</p>')
        parts.append('<table><tr><th>任务</th><th>状态</th><th>官方分</th><th>步数</th><th>秒</th><th>物理请求/重试/未知</th><th>已知输入/输出tokens</th><th>诊断/返回</th></tr>')
        for row in group['tasks']:
            folder = Path(row['evidence_directory'])
            trace = folder.parent / 'trajectories' / f'task-{row["task_id"]}.html'
            if not trace.exists():
                trace = folder / 'result.json'
            task_label = str(row['task_id'])
            if trace.exists():
                task_label = '<a href="' + html.escape(Path(os.path.relpath(trace, target.parent)).as_posix()) + '">' + task_label + '</a>'
            tokens = [sum(m.get(k, 0) for m in row['models'].values()) for k in ('input_tokens', 'output_tokens')]
            vals = [task_label, html.escape(row['status']), str(row['official_score']), str(row['steps']),
                    str(row['total_seconds']), f'{row["physical_requests"]}/{row["retry_attempts"]}/{row["unknown_usage"]}', f'{tokens[0]}/{tokens[1]}', html.escape(json.dumps(row.get('recovery', {}), ensure_ascii=False))]
            parts.append('<tr>' + ''.join('<td>' + v + '</td>' for v in vals) + '</tr>')
        parts.append('</table>')
    providers = {}
    for task_id in protocol['task_ids']:
        events = original.read_events(batch / 'webnavigator' / str(task_id) / 'events.jsonl')
        responses = {e['request_id']: e for e in events if e['event'] == 'responses_received'}
        for event in events:
            if event['event'] != 'provider_request':
                continue
            group = providers.setdefault(event['provider'], dict(requests=0, unknown_usage=0, input_tokens=0, output_tokens=0, reported_cost_sum=None, cost_reported_responses=0))
            group['requests'] += 1
            usage = responses.get(event['request_id'], {}).get('usage', {})
            group['unknown_usage'] += int('input_tokens' not in usage or 'output_tokens' not in usage)
            for k in ('input_tokens', 'output_tokens'):
                group[k] += usage.get(k, 0)
            if isinstance(usage.get('cost'), (int, float)):
                group['reported_cost_sum'] = (group['reported_cost_sum'] or 0) + usage['cost']
                group['cost_reported_responses'] += 1
    write(batch / 'provider-summary.json', providers)
    parts.append('<h2>Qwen实际服务商用量（预检另列）</h2><pre>' + html.escape(json.dumps(providers, ensure_ascii=False, indent=2)) + '</pre><p>API未返回费用的请求不能视为免费；不编造人民币总费用。旧作废基线费用仍独立保留于原收尾记录。</p>')
    preflight = original.read_events(batch / 'preflight/qwen/events.jsonl')
    preflight_stats = {'physical_requests': sum(e['event'] == 'responses_started' for e in preflight),
        'responses': sum(e['event'] == 'responses_received' for e in preflight),
        'known_input_tokens': sum((e.get('usage') or {}).get('input_tokens', 0) for e in preflight if e['event'] == 'responses_received'),
        'known_output_tokens': sum((e.get('usage') or {}).get('output_tokens', 0) for e in preflight if e['event'] == 'responses_received')}
    parts.append('<h3>独立预检用量</h3><pre>' + html.escape(json.dumps(preflight_stats, ensure_ascii=False, indent=2)) + '</pre>')
    parts.append('<p><a href="webnavigator-closeout.json">原Terra基线收尾：包含作废旧20题的独立成本</a></p>')
    parts.append('<h2>本方法与Qwen已评分配对</h2><pre>' + html.escape(json.dumps(summary['paired'], ensure_ascii=False, indent=2)) + '</pre>')
    target.write_text(''.join(parts), encoding='utf-8')


def queue(batch, live=False):
    protocol, state, todo = verify(batch)
    original = pinned(Path(protocol['runtimes']['webnavigator']))
    # All registration guards are checked here; each worker also checks its own exact source hash.
    print(json.dumps({'guards': len(protocol['guards']), 'pending': len(todo), 'status': state['status']}), flush=True)
    if not live:
        report(batch, original)
        return
    import msvcrt
    with Path(protocol['shared_lock']).open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        protocol, state, todo = verify(batch)
        state.update(status='running', queue_pid=os.getpid())
        write(batch / 'state.json', state)
        encoder = original.LocalEncoder(original.Graph.load(Path(protocol['graph'])),
            Path(protocol['runtimes']['webnavigator']), protocol['encoder_python'])
        try:
            encoder.ensure()
            for item in todo:
                protocol, state, _ = verify(batch)
                if Path(protocol['stop_file']).exists():
                    state['status'] = 'user_stopped'
                    break
                arm, task_id = item['arm'], item['task_id']
                folder = batch / arm / str(task_id)
                config = read(batch / arm / 'protocol.json')
                if arm == 'ours':
                    graph = Path(state['current_graph'])
                    config.update(graph=str(graph), cards=str(original.cards_path(graph)))
                folder.mkdir(exist_ok=False)
                write(folder / 'input.json', config)
                key = f'{arm}/{task_id}'
                row = {'status': 'started', 'time': time.time(), 'graph_before': config['graph'] if arm == 'ours' else str(batch / arm / 'released-graph')}
                state['tasks'][key] = row
                write(batch / 'state.json', state)
                started = time.monotonic()
                print(key + ' started', flush=True)
                try:
                    row['container_id'] = original.reset_copy(config['container'], config['image_id'], config['backend'], 10598, config['logical_origin'], config['owner'])
                    with (folder / 'console.log').open('w', encoding='utf-8') as out:
                        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--directory', str(batch), '--worker', arm, str(task_id)],
                            cwd=protocol['runtimes'][arm], stdout=out, stderr=out,
                            creationflags=subprocess.CREATE_NO_WINDOW,
                            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
                        row['worker_pid'] = child.pid
                        write(batch / 'state.json', state)
                        if child.wait():
                            raise RuntimeError('Worker failed; never restart task')
                    handoff = read(folder / 'handoff.json')
                    if arm == 'ours':
                        state.update({k: handoff[k] for k in ('current_graph', 'current_graph_sha256')})
                    row.update(status='completed', seconds=round(time.monotonic() - started, 3))
                    write(batch / 'state.json', state)
                    original.report(batch)
                    export_local_trajectories(batch / arm, original.export_report)
                    report(batch, original)
                    print(key + ': official=' + str(handoff['official_score']), flush=True)
                except Exception as error:
                    error_text = type(error).__name__ + ': ' + str(error)
                    if row['status'] == 'completed':
                        row['postprocessing_error'] = error_text
                    else:
                        row.update(status='needs_attention', seconds=round(time.monotonic() - started, 3), error=error_text)
                    state['status'] = 'needs_attention'
                    write(batch / 'state.json', state)
                    raise
            else:
                verify(batch)
                state['status'] = 'completed'
                write(batch / 'integrity-closeout.json', {'guards_verified': len(protocol['guards']), 'time': time.time()})
            write(batch / 'state.json', state)
        except Exception:
            state['status'] = 'needs_attention'
            write(batch / 'state.json', state)
            raise
        finally:
            encoder.close()
            report(batch, original)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--worker', nargs=2)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.directory.resolve(), args.worker[0], int(args.worker[1]))
    else:
        queue(args.directory.resolve(), args.live)
