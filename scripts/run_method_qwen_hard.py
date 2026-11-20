"""Registered full Hard run of the current method with Qwen and cumulative knowledge."""
import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import run_verified_template as base
from scripts.run_hard_comparison import cards_path, advance_graph, task_stats, wilson
from web2kg.common.activity import read_events
from web2kg.common.jina_runtime import LocalEncoder
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.evaluation.verified import reset_copy, score
from web2kg.stage1.function_cards import file_hash
from web2kg.stage2.recovery import ExperienceStore, memory_namespace, learn_experience
from web2kg.visualization.trajectory_report import export_report


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pending(config, state, active=None):
    ids = config['task_ids']
    if len(ids) != 42 or len(set(ids)) != 42 or config['max_new_task_starts'] != 42:
        raise ValueError('Expected exactly 42 distinct registered starts')
    if not set(state['tasks']).issubset({str(i) for i in ids}):
        raise ValueError('Unregistered task in state')
    for key, row in state['tasks'].items():
        if row['status'] != 'completed' and not (key == active and row['status'] == 'started'):
            raise ValueError('Unresolved started task; never restart: ' + key)
    return [i for i in ids if str(i) not in state['tasks']]


def verify(batch, active=None):
    for path, digest in read(batch / 'registration-integrity.json').items():
        if file_hash(path) != digest:
            raise ValueError('Registration changed: ' + path)
    config, state = read(batch / 'protocol.json'), read(batch / 'state.json')
    remaining = pending(config, state, active)
    base.check_inputs(config)
    manifest = Path(state['knowledge_manifest'])
    if file_hash(manifest) != state['knowledge_manifest_sha256']:
        raise ValueError('Knowledge handoff manifest changed')
    for path, digest in read(manifest).items():
        if file_hash(path) != digest:
            raise ValueError('Cumulative knowledge changed: ' + path)
    return config, state, remaining


def service_error(result, events):
    recovered = {e.get('request_id') for e in events if e['event'] == 'provider_fallback'}
    return (result.get('status') == 'runtime_error' or
            result.get('provider_failure', {}).get('kind') in {'cancelled', 'http', 'transport'} or
            any(e['event'] == 'responses_failed' and e.get('request_id') not in recovered and
                (e.get('failure_kind') == 'cancelled' or e.get('http_status') in {400, 401, 402, 403, 404})
                for e in events))


def knowledge_manifest(graph, experience, path):
    files = [p for p in Path(graph).rglob('*') if p.is_file()]
    files.append(Path(experience))
    write_json(path, {str(p.resolve()): file_hash(p) for p in sorted(files)})
    return str(path.resolve()), file_hash(path)


def worker(batch, task_id):
    protocol, state, _ = verify(batch, str(task_id))
    folder = batch / str(task_id)
    if file_hash(folder / 'input.json') != state['tasks'][str(task_id)]['input_sha256']:
        raise ValueError('Task input changed')
    config = read(folder / 'input.json')
    base.worker(batch, config, task_id)
    official = score(config['verified_root'], batch, task_id, config['logical_origin'])
    write_json(folder / 'official.json', official)
    if official.get('score') is None:
        raise RuntimeError('Official score missing')
    result = read(folder / 'result.json')
    events = [e for name in ('events.jsonl', 'diagnosis-calls/events.jsonl',
                             'graph-annotations/events.jsonl', 'graph-cards/events.jsonl')
              for e in read_events(folder / name)]
    if service_error(result, events):
        raise RuntimeError('Engineering or permanent service failure; retain score and stop')
    graph = advance_graph(folder)
    _, diagnosis_class = base.execution_clients(batch, config)
    learner = diagnosis_class(ROOT / 'auth/llm-crawl.env', EventLog(folder / 'diagnosis-calls'),
        config['diagnosis_calls'], 2048, model=config['diagnosis_model'],
        retry_connections=True, stop_file=config['stop_file'])
    memory = ExperienceStore(config['experience_store'], memory_namespace(Graph.load(Path(config['graph']))))
    learn_experience(folder, learner, memory, success=official['score'] == 1,
                     verifier='WebArena-Verified official score=1')
    manifest, digest = knowledge_manifest(graph, config['experience_store'], folder / 'knowledge-handoff.json')
    write_json(folder / 'handoff.json', dict(current_graph=str(graph), knowledge_manifest=manifest,
        knowledge_manifest_sha256=digest, official_score=official['score']))


def export_existing(batch):
    original_read = export_report.__globals__['read']

    def completed_read(path, *args):
        value = original_read(path, *args)
        if path == batch / 'summary.json':
            value = {**value, 'tasks': [r for r in value['tasks']
                                      if (batch / str(r['task_id']) / 'result.json').exists()]}
        return value

    exporter = types.FunctionType(export_report.__code__, {**export_report.__globals__, 'read': completed_read},
        export_report.__name__, export_report.__defaults__, export_report.__closure__)
    return exporter([batch], batch / 'trajectories')


def report(batch):
    config, state = read(batch / 'protocol.json'), read(batch / 'state.json')
    rows = [task_stats(batch / str(i), i, state['tasks'].get(str(i), {})) for i in config['task_ids']]
    scored = sum(r['official_score'] is not None for r in rows)
    successes = sum(r['official_score'] == 1 for r in rows)
    summary = dict(planned=42, official_scored=scored, official_successes=successes,
        unrun=42-len(state['tasks']), started_unscored=len(state['tasks'])-scored,
        status=state['status'], tasks=rows, descriptive_wilson95_scored=wilson(successes, scored),
        scope=config['exposure'])
    write_json(batch / 'summary.json', summary)
    export_existing(batch)
    target = Path(config['report_path'])
    link = lambda path: html.escape(Path(os.path.relpath(path, target.parent)).as_posix(), quote=True)
    totals = {k: sum(r.get(k, 0) for r in rows) for k in ('physical_requests', 'unknown_usage', 'retry_attempts', 'connection_retry_seconds')}
    parts = ['<section id="method-qwen-hard-v1"><h2>本方法 · Qwen 3.8-27B · 当前累计知识开发重跑</h2>',
        '<p>独立42次新开始，包含596；旧单题成功不顶替本轮结果。以348节点/422边/45卡累计图及1条596页面经验起跑，逐题积累图谱和成功经验。任务已全部暴露，已有图谱含旧任务观察；不是盲测、泛化或单组件因果比较。</p>',
        '<p>Qwen负责Actor、selector、诊断和学习；Luna描述/摘要/卡片。AX初始800，关键词窗口800，最多3次/2500行。OpenRouter免费优先，已授权免费不可用或每日额度耗尽时转百炼；实际服务商逐请求记录。价格未知。</p>',
        f'<p>状态：{html.escape(state["status"])}；官方成功 {successes}/42；已评分 {scored}/42；未运行 {summary["unrun"]}；已开始未评分 {summary["started_unscored"]}。</p>',
        f'<p>物理请求 {totals["physical_requests"]}；未知usage {totals["unknown_usage"]}；重试 {totals["retry_attempts"]}；重试及等待 {totals["connection_retry_seconds"]:.3f}秒。区间仅描述已评分任务：{html.escape(str(summary["descriptive_wilson95_scored"]))}。</p>',
        '<table><tr><th>任务</th><th>状态 / 官方分</th><th>步数</th><th>请求 / 未知</th><th>输入 / 输出token</th><th>诊断 / 返回 / 学习</th></tr>']
    for row in rows:
        trace = batch / 'trajectories' / f'task-{row["task_id"]}.html'
        label = f'<a href="{link(trace)}">{row["task_id"]}</a>' if trace.exists() else str(row['task_id'])
        inp = sum(m['input_tokens'] for m in row['models'].values())
        out = sum(m['output_tokens'] for m in row['models'].values())
        recovery = row.get('recovery') or {}
        recovery_text = str(recovery.get('diagnoses', 0)) + ' / ' + str((recovery.get('return') or {}).get('success')) + ' / ' + str((row.get('learning') or {}).get('status'))
        parts.append('<tr>' + ''.join(f'<td>{v}</td>' for v in [label, html.escape(f'{row["status"]} / {row["official_score"]}'),
            row.get('steps'), f'{row["physical_requests"]} / {row["unknown_usage"]}', f'{inp} / {out}', html.escape(recovery_text)]) + '</tr>')
    parts.append(f'</table><p><a href="{link(batch / "summary.json")}">逐模型用量与逐题汇总</a> · <a href="{link(batch / "run_material_passport.json")}">来源与实验边界</a></p></section>')
    section = ''.join(parts)
    prefix = (batch / 'comparison-report-before.html').read_text(encoding='utf-8')
    existing = target.read_text(encoding='utf-8')
    if not existing.startswith(prefix):
        raise ValueError('Prior report changed outside this queue; preserve and inspect')
    temp = target.with_suffix('.tmp')
    temp.write_text(prefix + section, encoding='utf-8')
    temp.replace(target)
    (batch / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Qwen method Hard</title>' +
        f'<p><a href="{html.escape(Path(os.path.relpath(target, batch)).as_posix())}#method-qwen-hard-v1">打开完整比较报告</a></p>', encoding='utf-8')
    return summary


def queue(batch):
    import msvcrt
    config = read(batch / 'protocol.json')
    with Path(config['shared_lock']).open('a+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        config, state, remaining = verify(batch)
        if state['status'] == 'completed':
            return
        encoder = LocalEncoder(Graph.load(Path(state['current_graph'])), ROOT, config['encoder_python'])
        try:
            encoder.ensure()
            for task_id in remaining:
                if Path(config['stop_file']).exists():
                    state['status'] = 'user_stopped'; break
                config, state, _ = verify(batch)
                graph = Path(state['current_graph'])
                folder = batch / str(task_id); folder.mkdir(exist_ok=False)
                inputs = {**config, 'graph': str(graph), 'cards': str(cards_path(graph))}
                write_json(folder / 'input.json', inputs)
                row = dict(status='started', time=time.time(), graph_before=str(graph),
                           input_sha256=file_hash(folder / 'input.json'))
                state['tasks'][str(task_id)] = row; state['status'] = 'running'
                write_json(batch / 'state.json', state)
                started = time.monotonic()
                print(f'{task_id} started ({len(state["tasks"])}/42)', flush=True)
                try:
                    row['container_id'] = reset_copy(config['container'], config['image_id'], config['backend'],
                        10598, config['logical_origin'], config['owner'])
                    with (folder / 'console.log').open('w', encoding='utf-8') as out:
                        child = subprocess.Popen([sys.executable, str(Path(config['frozen_runtime']) / 'scripts/run_method_qwen_hard.py'),
                            '--directory', str(batch), '--worker', str(task_id)], cwd=ROOT, stdout=out, stderr=out,
                            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'}, creationflags=subprocess.CREATE_NO_WINDOW)
                        row['worker_pid'] = child.pid; write_json(batch / 'state.json', state)
                        code = child.wait()
                    if code:
                        raise RuntimeError(f'Worker exited {code}; never restart started task')
                    handoff = read(folder / 'handoff.json')
                    state.update({k: handoff[k] for k in ('current_graph', 'knowledge_manifest', 'knowledge_manifest_sha256')})
                    row.update(status='completed', seconds=round(time.monotonic()-started, 3))
                    write_json(batch / 'state.json', state)
                    report(batch)
                    print(f'{task_id}: official={handoff["official_score"]}', flush=True)
                except Exception as error:
                    row.update(status='needs_attention', error=f'{type(error).__name__}: {error}', seconds=round(time.monotonic()-started, 3))
                    state['status'] = 'needs_attention'; write_json(batch / 'state.json', state)
                    raise
            else:
                verify(batch)
                state['status'] = 'completed'
                write_json(batch / 'integrity-closeout.json', dict(source_verified=True,
                    protected_files=len(config['protected_inputs']), all_42_completed=True, time=time.time()))
            write_json(batch / 'state.json', state)
        finally:
            encoder.close()
            report(batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); batch = args.directory.resolve()
    if args.worker is not None:
        worker(batch, args.worker)
    elif args.live:
        queue(batch)
    elif args.check:
        _, _, remaining = verify(batch); print(f'Guards passed; {len(remaining)} unstarted tasks')
    else:
        report(batch)


if __name__ == '__main__':
    main()
