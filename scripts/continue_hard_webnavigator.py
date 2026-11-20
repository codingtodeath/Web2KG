"""Continue only the registered baseline, using its original source snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pending_tasks(protocol, state, allowed):
    order = [x['task_id'] for x in protocol['schedule'] if x['arm'] == 'webnavigator']
    if len(order) != len(set(order)) or not set(allowed) <= set(order):
        raise ValueError('Invalid baseline continuation registration')
    for key, row in state['tasks'].items():
        if key.startswith('webnavigator/') and row['status'] != 'completed':
            raise ValueError('Started baseline task unresolved; never restart')
    pending = [t for t in order if f'webnavigator/{t}' not in state['tasks']]
    if not set(pending) <= set(allowed):
        raise ValueError('Unregistered task start')
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    batch = args.directory.resolve()
    registration = read(batch / 'webnavigator-continuation.json')
    runtime = Path(registration['runtime'])
    sys.path.insert(0, str(runtime))
    from scripts import run_hard_comparison as original
    from web2kg.common.storage import write_json

    def verify():
        for path, expected in registration['guards'].items():
            if digest(path) != expected:
                raise ValueError('Continuation input or runtime changed: ' + path)
        protocol = read(batch / 'protocol.json')
        original.check_inputs(protocol)
        state = read(batch / 'state.json')
        ours = {k: v for k, v in state['tasks'].items() if k.startswith('ours/')}
        if ours != registration['ours_state']:
            raise ValueError('Ours task evidence changed')
        if any(state[k] != registration[k] for k in ('current_graph', 'current_graph_sha256')):
            raise ValueError('Cumulative graph handoff changed')
        if digest(Path(state['current_graph']) / 'graph.json') != state['current_graph_sha256']:
            raise ValueError('Cumulative graph changed')
        return protocol, state, pending_tasks(protocol, state, registration['remaining_tasks'])

    def report():
        original.report(batch)
        path = batch / 'report.html'
        note = ('<p><strong>2026-10-15：按用户要求仅续跑 WebNavigator 剩余20题，'
                '使用原v2源码与输入。ours/29基础设施失败及剩余20题保持原状；'
                '整体比较仍未完成。续跑状态见 webnavigator-continuation-state.json。</strong></p>')
        path.write_text(path.read_text(encoding='utf-8') + note, encoding='utf-8')

    protocol, state, pending = verify()
    print(json.dumps({'source_verified': True, 'protected_inputs': len(protocol['protected_inputs']),
                      'pending_webnavigator': pending, 'ours_unchanged': True}), flush=True)
    if not args.live:
        return
    import msvcrt
    with (batch / 'queue.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        protocol, state, pending = verify()
        progress_path = batch / 'webnavigator-continuation-state.json'
        progress = {'status': 'running', 'queue_pid': os.getpid(), 'time': time.time()}
        write_json(progress_path, progress)
        encoder = original.LocalEncoder(original.Graph.load(Path(protocol['graph'])),
                                        runtime, protocol['encoder_python'])
        try:
            encoder.ensure()
            for task_id in pending:
                protocol, state, _ = verify()
                if Path(protocol['stop_file']).exists():
                    progress['status'] = 'user_stopped'
                    break
                config = read(batch / 'webnavigator/protocol.json')
                original.check_inputs(config)
                folder = batch / 'webnavigator' / str(task_id)
                folder.mkdir(exist_ok=False)
                write_json(folder / 'input.json', config)
                row = {'status': 'started', 'time': time.time(),
                       'graph_before': str(batch / 'webnavigator/released-graph'),
                       'continuation': 'webnavigator-only-20261015'}
                key = f'webnavigator/{task_id}'
                state['tasks'][key] = row
                write_json(batch / 'state.json', state)
                progress['task_id'] = task_id
                write_json(progress_path, progress)
                started = time.monotonic()
                print(key + ' started', flush=True)
                try:
                    row['container_id'] = original.reset_copy(config['container'], config['image_id'],
                        config['backend'], 10598, config['logical_origin'], config['owner'])
                    with (folder / 'console.log').open('w', encoding='utf-8') as out:
                        child = subprocess.Popen([sys.executable, '-B',
                            str(runtime / 'scripts/run_hard_comparison.py'), '--directory', str(batch),
                            '--worker', 'webnavigator', str(task_id)], cwd=runtime,
                            stdout=out, stderr=out, creationflags=subprocess.CREATE_NO_WINDOW,
                            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
                        row['worker_pid'] = child.pid
                        write_json(batch / 'state.json', state)
                        if child.wait():
                            raise RuntimeError('Worker failed; preserve evidence, never restart task')
                    official = original.score(config['verified_root'], batch / 'webnavigator',
                                              task_id, config['logical_origin'])
                    write_json(folder / 'official.json', official)
                    if official.get('score') is None:
                        raise RuntimeError('Official score missing')
                    result = read(folder / 'result.json')
                    events = original.read_events(folder / 'events.jsonl')
                    if any(e['event'] == 'responses_failed' and
                           (e.get('failure_kind') == 'cancelled' or
                            (e.get('http_status') and e['http_status'] not in {408,429,500,502,503,504}))
                           for e in events) or result['status'] == 'runtime_error':
                        raise RuntimeError('Permanent service or runtime error; score retained')
                    row.update(status='completed', seconds=round(time.monotonic() - started, 3))
                    write_json(batch / 'state.json', state)
                    report()
                    original.export_report([batch / 'webnavigator'], batch / 'webnavigator/trajectories')
                    print(key + ': official=' + str(official['score']), flush=True)
                except Exception as error:
                    row.update(status='needs_attention', error=type(error).__name__ + ': ' + str(error),
                               seconds=round(time.monotonic() - started, 3))
                    write_json(batch / 'state.json', state)
                    raise
            else:
                verify()
                progress['status'] = 'completed'
                progress['protected_inputs_verified'] = len(protocol['protected_inputs'])
        except Exception as error:
            progress.update(status='needs_attention', error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            progress['updated_at'] = time.time()
            write_json(progress_path, progress)
            encoder.close()
            report()


if __name__ == '__main__':
    main()
