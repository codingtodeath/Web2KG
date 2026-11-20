"""Independent full Reddit Hard run of the retained original WebNavigator."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from scripts.run_midterm_qwen import freeze, queue_lock
from scripts.run_reddit_hard10_225 import read, hashes, selected_tasks
from scripts.qwen_method_transport import clients
from web2kg.common.activity import read_events
from web2kg.common.storage import write_json
from web2kg.stage1.function_cards import file_hash
from web2kg.evaluation.verified import reset_copy, score
from web2kg.visualization.trajectory_report import export_report

MODEL = 'qwen3.7-flash'
STOP_PROMPT = Path('webNavigator/prompts/navigation_specifications/stop.txt')
STOP_PROMPT_VERSION = 'verified-json-stop-v1'


def install_stop_prompt(batch, runtime):
    """Override only the stop spec in a newly copied upstream, never its source."""
    from web2kg.evaluation.verified_answer import STOP_SPEC
    source = runtime / 'third_party/webnavigator' / STOP_PROMPT
    if source.read_text(encoding='utf-8').strip() != STOP_SPEC:
        raise ValueError('Frozen stop prompt differs from the registered contract')
    target = batch / 'upstream' / STOP_PROMPT
    before = file_hash(target)
    temporary = target.with_suffix('.txt.tmp')
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    return dict(version=STOP_PROMPT_VERSION, path=str(STOP_PROMPT.as_posix()),
                previous_sha256=before, sha256=file_hash(target))


def check_continuation_prompt(protocol, expected_sha256):
    if (not protocol.get('unified_verified_stop')
            or protocol.get('stop_prompt_sha256') != expected_sha256):
        raise ValueError('Stop prompt changed; use a new experiment, not continuation')


def check_hashes(mapping):
    for path, expected in mapping.items():
        if file_hash(path) != expected:
            raise ValueError('Protected material changed: ' + path)


def verify(batch, full=False):
    check_hashes(read(batch / 'registration-integrity.json'))
    p = read(batch / 'protocol.json')
    runtime = Path(p['runtime'])
    check_hashes({str(runtime / k): v for k, v in read(runtime / 'manifest.json').items()})
    check_hashes(read(batch / 'runtime-inputs.json'))
    if full:
        check_hashes(read(batch / 'protected-inputs.json'))
    return p


def fresh_state(state, inherited=None):
    if state['status'] != 'registered' or state['tasks'] != (inherited or {}):
        raise ValueError('Never restart a started or unresolved batch')


def outcome(result, events):
    """Normal limits may finish a case; all other model/runtime failures stop."""
    failures = [e for e in events if e['event'] == 'responses_failed']
    if any(e.get('failure_kind') == 'cancelled' for e in failures):
        return 'user_stopped'
    if result.get('status') in {'agent_stopped', 'step_limit'}:
        return 'completed'
    if (result.get('status') == 'model_error'
            and result.get('error') == 'ResponsesError: Responses call budget exhausted'
            and not failures):
        return 'completed'
    return 'needs_attention'


def repeat_lineage(previous, protocol, summary, source):
    if (previous['status'] != 'completed' or summary.get('resolved') != 42
            or any(row['status'] not in {'completed', 'failed'} for row in previous['tasks'].values())
            or len(previous['tasks']) != 42):
        raise ValueError('Repeat requires all42 previous attempts resolved')
    return dict(previous=str(source), previous_usage=summary['totals'],
        authorization='User authorized rerunning the previous tasks after Verified JSON prompt adaptation.',
        change='Verified stop prompt and its explicit entry wiring; original Agent/navigation/model/budgets retained.',
        previous_prompt=protocol.get('stop_prompt_version', 'legacy-appended-verified-instruction'),
        restart='All42 new attempts; no inherited scores; previous results and costs remain separate.')


def register(batch, checks, repair_from=None, continue_from=None, repeat_from=None, continue_task_errors=False):
    if batch.exists() or not batch.is_relative_to(ROOT / 'runs'):
        raise ValueError('Use a fresh directory within runs')
    receipt = read(checks)
    if not receipt.get('passed'):
        raise ValueError('Offline validation required')
    check_hashes(receipt['source_sha256'])
    previous_guards, inherited, inherited_sources = {}, {}, {}
    prior_batch = continue_from or repair_from or repeat_from
    repeat = None
    if prior_batch:
        if continue_from:
            check_continuation_prompt(read(prior_batch / 'protocol.json'),
                file_hash(ROOT / 'third_party/webnavigator' / STOP_PROMPT))
        verify(prior_batch, full=True)
        previous = read(prior_batch / 'state.json')
        if repeat_from:
            repeat = repeat_lineage(previous, read(prior_batch/'protocol.json'),
                                    read(prior_batch/'summary.json'), prior_batch)
        else:
            diagnosis = read(prior_batch / 'failure-diagnosis.json')
            if diagnosis.get('status') != 'confirmed' or previous['status'] == 'running':
                raise ValueError('Repair requires a stopped diagnosed version')
        previous_guards = {path: sha for path, sha in hashes(prior_batch).items()
                           if 'auth' not in Path(path).parts}
        if continue_from:
            for key, row in previous['tasks'].items():
                if not (prior_batch/key/'result.json').is_file() or not (prior_batch/key/'official.json').is_file():
                    raise ValueError('Cannot inherit an unresolved/missing-result attempt')
                inherited[key] = {**row, 'status':'completed' if row['status']=='completed' else 'failed',
                                  'inherited_from':str(prior_batch/key), 'original_status':row['status']}
                inherited_sources[key] = str(prior_batch/key)
    retained = ROOT / 'runs/hard-comparison-retained-20261029/webnavigator-qwen/assets/reddit-hard-qwen-v1/webnavigator'
    prior = ROOT / 'runs/reddit-books3-tiers-flash-20261107-v2/case-66'
    old = read(prior / 'protocol.json')
    tasks = selected_tasks(ROOT, 'http://127.0.0.1:10599', seed=20261104, limit=1000)
    if len(tasks) != 42 or len({t['task_id'] for t in tasks}) != 42:
        raise ValueError('Expected all42 single-site Reddit Hard tasks')
    batch.mkdir()
    runtime = freeze(batch)
    for name in ('upstream', 'import-stubs', 'released-graph'):
        shutil.copytree(retained / name, batch / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    prompt_adaptation = install_stop_prompt(batch, runtime)
    encoder = batch / 'encoder-runtime/scripts'; encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py', 'jina_gpu.py', 'jina_vision_fp32.py'):
        shutil.copy2(Path(old['encoder_runtime']) / 'scripts' / name, encoder / name)
    import yaml
    cfg = yaml.safe_load((retained / 'config.yml').read_text('utf-8'))
    cfg['agent']['actor'].update(model=MODEL, model_family='gpt', domain_paths={'reddit': str(batch / 'released-graph')})
    cfg['agent']['actor']['selector']['model'] = MODEL
    cfg['env'].update(ax_search=False, max_browser_rows=1500)
    cfg['agent']['others'].update(logdir=str(batch / 'upstream-logs'))
    cfg['logdir'] = str(batch / 'upstream-logs')
    (batch / 'config.yml').write_text(yaml.safe_dump(cfg, allow_unicode=True), 'utf-8')
    write_json(batch / 'public-inputs.json', tasks)
    p = dict(version='original-webnavigator-flash-hard42-v4', created_at=datetime.now(timezone.utc).isoformat(),
        authorization='User requested Qwen3.7-flash original WebNavigator on ALL Reddit Hard tasks; report anomalies.',
        runtime=str(runtime), workspace=str(runtime), model=MODEL, provider='bailian',
        credential_file=old['credential_file'], role_models={'actor': MODEL, 'selector': MODEL},
        task_ids=[t['task_id'] for t in tasks], seed=20261104,
        site='reddit', backend=old['backend'], logical_origin=old['backend'], graph_origin=old['backend'],
        container=old['container'], owner=old['owner'], image_id=old['image_id'],
        verified_root=old['verified_root'], max_steps=30, max_calls=31, output_tokens_per_call=4096,
        actor_compatibility=True, retry_connections=True, ax_search=False, ax_line_limit=1500,
        section_parser='anchored-heading-v1',
        unified_verified_stop=True, stop_prompt_version=STOP_PROMPT_VERSION,
        stop_prompt_sha256=prompt_adaptation['sha256'], prompt_adaptation=prompt_adaptation,
        continue_task_errors=bool(continue_from or continue_task_errors), inherited_tasks=inherited, inherited_sources=inherited_sources,
        stop_file=str(batch / 'STOP'), shared_lock=str(ROOT / 'runs/reddit-hard-comparison-v2/queue.lock'),
        encoder_endpoint='http://127.0.0.1:8797/v1/embeddings', encoder_expected=old['encoder_expected'],
        encoder_python=old['encoder_python'], encoder_runtime=str(encoder.parent), gpu_memory='4GiB',
        source_assets=str(retained), nodes=225, indexed_images=201, paid_preflight=0,
        compatibility=['Retained original Agent/navigation; fixed AX1500; only stop prompt adapted to Verified JSON.',
            'Explicit minimal parser repair: whole-line headings, REASON FOR ACTION alias, ASCII/fullwidth colon.',
            'Existing lossless output syntax adapter; unified Verified stop prompt replaces prose/N/A examples. No added output retries.',
            'Bailian chat transport; persistent identical-input transient retries2..60sec; no browser retry.',
            'Original format regeneration consumes31 logical-call budget. No method single-action prompt/recovery/cards.',
            'Released image vectors retained; upstream image encoder attestation absent; local Jina query adapter.'],
        stop_policy=('User authorized task errors to be recorded as failures and continue remaining unstarted tasks. '
                     'User STOP still honored. Input integrity/shared encoder or environment setup failure cannot be bypassed.'
                     if continue_from or continue_task_errors else 'Ordinary official failures and step/call limit continue; engineering/model-output/permanent-service/asset errors stop. No restart.'),
        limits='Exposed development tasks; one new attempt per task, no score replacement; not a blinded generalization comparison.',
        checks=str(checks))
    write_json(batch / 'protocol.json', p)
    if continue_from:
        old_protocol = read(continue_from/'protocol.json')
        for key in ('task_ids','model','max_steps','max_calls','output_tokens_per_call','section_parser','source_assets','encoder_expected'):
            if p[key] != old_protocol[key]: raise ValueError('Continuation changes method/input: '+key)
        write_json(batch/'continuation.json', dict(source=str(continue_from), inherited_tasks=inherited,
            new_task_ids=[t for t in p['task_ids'] if str(t) not in inherited],
            authorization='User: run all WebNavigator tasks; errors must not stop the whole experiment.',
            policy='No repeated started task; retain598 failed attempt and646 result. Only queue/error/report behavior changes.'))
    write_json(batch / 'run_material_passport.json', dict(mode='run', planned=42, source=str(retained),
        authorization=p['authorization'], roles=p['role_models'], compatibility=p['compatibility'], limits=p['limits']))
    inputs = {}
    for directory in (batch / 'upstream', batch / 'import-stubs', batch / 'released-graph', encoder.parent):
        inputs.update(hashes(directory))
    for path in (batch / 'config.yml', batch / 'public-inputs.json', checks,
                 ROOT / 'auth/qwen-hard.env', ROOT / 'auth/llm-crawl.env'):
        inputs[str(path)] = file_hash(path)
    write_json(batch / 'runtime-inputs.json', inputs)
    protected = {**previous_guards, **inputs}
    if repeat:
        old_protocol = read(repeat_from/'protocol.json')
        for key in ('task_ids','model','provider','max_steps','max_calls','output_tokens_per_call',
                    'section_parser','source_assets','encoder_expected','ax_line_limit'):
            if p[key] != old_protocol[key]:
                raise ValueError('Prompt rerun changes other conditions: '+key)
        write_json(batch/'repeat-lineage.json', repeat)
        protected[str(batch/'repeat-lineage.json')] = file_hash(batch/'repeat-lineage.json')
    if continue_from:
        protected[str(batch/'continuation.json')] = file_hash(batch/'continuation.json')
    if repair_from:
        lineage = dict(previous=str(repair_from), previous_tasks=read(repair_from / 'state.json')['tasks'],
            previous_usage=read(repair_from / 'summary.json')['totals'],
            authorization='User explicitly authorized minimal heading-parser repair and rerun.',
            change='Opt-in heading extraction and explicit Verified stop prompt adaptation; Agent/navigation unchanged.',
            restart='All42 from original released graph, no inherited scores, prior attempt costs separate.')
        write_json(batch / 'repair-lineage.json', lineage)
        protected[str(batch / 'repair-lineage.json')] = file_hash(batch / 'repair-lineage.json')
    for name in ('upstream', 'import-stubs', 'released-graph'):
        protected.update(hashes(retained / name))
    for path in (ROOT / 'runs/active-graph.json', ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt'):
        protected[str(path)] = file_hash(path)
    write_json(batch / 'protected-inputs.json', protected)
    write_json(batch / 'registration-integrity.json', {str(path): file_hash(path) for path in [
        batch / 'protocol.json', batch / 'run_material_passport.json', batch / 'runtime-inputs.json',
        batch / 'protected-inputs.json', runtime / 'manifest.json']})
    write_json(batch / 'state.json', dict(status='registered', tasks=inherited))
    verify(batch, full=True)
    report(batch)
    print(json.dumps(dict(registered=str(batch), tasks=p['task_ids'], protected=len(protected))), flush=True)


def report(batch):
    p, state = read(batch / 'protocol.json'), read(batch / 'state.json')
    rows = []
    for task in p['task_ids']:
        folder = Path(p.get('inherited_sources', {}).get(str(task), batch / str(task)))
        result, official = read(folder / 'result.json', {}), read(folder / 'official.json', {})
        events = read_events(folder / 'events.jsonl')
        starts = [e for e in events if e['event'] == 'responses_started']
        receipts = {e['request_id']: e for e in events if e['event'] == 'responses_received'}
        rows.append(dict(task_id=task, status=state['tasks'].get(str(task), {}).get('status', 'not_started'),
            agent_status=result.get('status'), official_score=official.get('score'), steps=result.get('steps'),
            seconds=result.get('seconds'), error=result.get('error') or state['tasks'].get(str(task),{}).get('error'), physical_requests=len(starts),
            input_tokens=sum(e.get('usage', {}).get('input_tokens', 0) for e in receipts.values()),
            output_tokens=sum(e.get('usage', {}).get('output_tokens', 0) for e in receipts.values()),
            unknown_usage=sum(not receipts.get(e['request_id'], {}).get('usage') for e in starts),
            roles=dict(Counter(e.get('purpose') for e in starts)),
            actual_models=sorted({str(e.get('actual_model')) for e in receipts.values()}),
            retries=sum(e['event'] == 'responses_retry' for e in events),
            inherited=str(task) in p.get('inherited_sources', {}), source=str(folder),
            trace=(Path(os.path.relpath(folder.parent/'trajectories'/f'task-{task}.html', batch)).as_posix()
                   if str(task) in p.get('inherited_sources', {}) else f'trajectories/task-{task}.html') if result else None))
    finished = sum(r['status'] == 'completed' for r in rows)
    resolved = sum(r['status'] in {'completed','failed'} for r in rows)
    successes = sum(r['official_score'] == 1 for r in rows)
    scored = sum(r['official_score'] is not None for r in rows)
    summary = dict(planned=42, completed=finished, resolved=resolved, errors=sum(r['status']=='failed' for r in rows), official_scored=scored,
        official_successes=successes, success_rate=successes / 42 if resolved == scored == 42 else None,
        status=state['status'], model=p['model'], tasks=rows,
        totals={k: sum(r[k] for r in rows) for k in ('physical_requests','input_tokens','output_tokens','unknown_usage','retries')})
    write_json(batch / 'summary.json', summary)
    table = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in
        ('task_id','status','official_score','steps','seconds','physical_requests')) +
        (f'<td><a href="{r["trace"]}">轨迹</a></td>' if r['trace'] else '<td>—</td>') + '</tr>' for r in rows)
    page = ('<!doctype html><meta charset="utf-8"><title>WebNavigator · Flash · Reddit Hard42</title>'
        '<style>body{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 16px}td,th{border:1px solid #ddd;padding:6px}table{border-collapse:collapse}</style>'
        f'<h1>原版 WebNavigator × Qwen3.7-flash</h1><p>已结束 {resolved}/42（正常完成{finished}，异常{resolved-finished}）；官方成功 {successes}。状态：{state["status"]}。</p>'
        '<p>原发布225状态/201截图向量；Actor+selector共31逻辑调用、30步、4096输出、固定AX1500。继承条目按协议引用原始结果；独立重跑全部另计，不挑分替换旧结果。未全部评分时不发布全套成功率。</p>'
        '<p>保留原 Agent/导航；stop按登记协议适配Verified JSON，栏目解析支持独立标题、REASON FOR ACTION别名及中英文冒号，另有传输、输出格式和评分接口兼容。无本方法诊断、卡片或任务增图。原始图片编码器无身份凭证，本地Jina只编码查询；开发暴露任务不能作为盲测或组件因果证据。</p>'
        '<table><tr><th>任务</th><th>状态</th><th>官方分</th><th>步</th><th>秒</th><th>物理请求</th><th>证据</th></tr>' + table + '</table>'
        '<p><a href="summary.json">用量与结果</a> · <a href="protocol.json">协议</a></p>')
    (batch / 'report.html').write_text(page, 'utf-8')
    local_rows = [r for r in rows if not r['inherited']]
    if any(r['trace'] for r in local_rows):
        # Export only local evidence; inherited trajectories retain original links.
        # The exporter reads numeric folders from its batch. Use a temporary
        # selected summary in the real batch, restored even if a partial trace fails.
        write_json(batch/'summary.json', {**summary, 'tasks':[r for r in local_rows if r['trace']]})
        try:
            export_report([batch], batch/'trajectories')
        except Exception as error:
            write_json(batch/'trace-export-error.json', dict(error=str(error)))
        finally:
            write_json(batch/'summary.json', summary)
    return summary


def worker(batch, task_id):
    p = verify(batch)
    folder = batch / str(task_id)
    row = read(batch / 'state.json')['tasks'][str(task_id)]
    if row['status'] != 'started' or file_hash(folder / 'input.json') != row['input_sha256']:
        raise ValueError('Worker input/start mismatch')
    from web2kg.evaluation import webnavigator
    webnavigator.OriginalResponses = clients(p['credential_file'], batch / 'routing-state.json',
                                              provider='bailian', model_override=MODEL)
    task = next(t for t in read(batch / 'public-inputs.json') if t['task_id'] == task_id)
    result = webnavigator.run_original(batch, p, task)
    official = score(p['verified_root'], batch, task_id, p['logical_origin'])
    write_json(folder / 'official.json', official)
    status = outcome(result, read_events(folder / 'events.jsonl'))
    if official.get('score') is None:
        status = 'needs_attention'
    write_json(folder / 'closeout.json', dict(status=status, official_score=official.get('score')))
    if status != 'completed':
        raise RuntimeError('Task anomaly: ' + str(result.get('error') or status))


def live(batch):
    p = verify(batch, full=True)
    if Path(p['runtime']) != ROOT:
        raise ValueError('Use registered frozen entry')
    state = read(batch / 'state.json'); fresh_state(state, p.get('inherited_tasks'))
    Path(p['shared_lock']).parent.mkdir(exist_ok=True)
    from scripts import build_qwen_graph
    build_qwen_graph.ROOT = Path(p['encoder_runtime'])
    with queue_lock(p['shared_lock']):
        state.update(status='running', queue_pid=os.getpid(), started_at=time.time(), phase='encoder')
        write_json(batch / 'state.json', state)
        try:
            with build_qwen_graph.local_encoder(batch, 8797, p['encoder_python'],
                    gpu_memory=p['gpu_memory'], expected=p['encoder_expected']):
                for task_id in p['task_ids']:
                    if str(task_id) in p.get('inherited_tasks', {}): continue
                    if Path(p['stop_file']).exists():
                        state['status'] = 'user_stopped'; break
                    verify(batch)
                    state.update(phase='reset', current_task=task_id); write_json(batch / 'state.json', state)
                    container = reset_copy(p['container'], p['image_id'], p['backend'], 10598, p['logical_origin'], p['owner'])
                    folder = batch / str(task_id); folder.mkdir(exist_ok=False)
                    write_json(folder / 'input.json', p)
                    row = dict(status='started', started_at=time.time(), container_id=container,
                               input_sha256=file_hash(folder / 'input.json'))
                    state['tasks'][str(task_id)] = row; state['phase'] = 'task'
                    write_json(batch / 'state.json', state)
                    with (folder / 'console.log').open('w', encoding='utf-8') as output:
                        process = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(Path(__file__)),
                            '--directory', str(batch), '--worker', str(task_id)], cwd=ROOT,
                            stdout=output, stderr=output, creationflags=subprocess.CREATE_NO_WINDOW,
                            env={**os.environ, 'PYTHONUTF8':'1', 'PYTHONDONTWRITEBYTECODE':'1'})
                        row['worker_pid'] = process.pid; write_json(batch / 'state.json', state)
                        code = process.wait()
                    closeout = read(folder / 'closeout.json', {})
                    row.update(status=closeout.get('status', 'needs_attention'), seconds=time.time()-row['started_at'])
                    if (code or row['status'] != 'completed') and p.get('continue_task_errors') and not Path(p['stop_file']).exists():
                        row.update(status='failed', worker_exit_code=code,
                                   error=read(folder/'result.json', {}).get('error') or 'Worker/evaluator failed')
                        if not (folder/'official.json').is_file():
                            try:
                                write_json(folder/'official.json', score(p['verified_root'],batch,task_id,p['logical_origin']))
                            except Exception as error:
                                write_json(folder/'official.json', dict(score=None,status='evaluator_error',error=str(error)))
                    write_json(batch / 'state.json', state); report(batch)
                    if code or row['status'] != 'completed':
                        if row['status'] != 'failed':
                            raise RuntimeError(f'Task{task_id} stopped; inspect retained evidence')
                    print(json.dumps(dict(task=task_id, status=row['status'], score=read(folder/'official.json',{}).get('score'), resolved=len(state['tasks']))), flush=True)
                else:
                    state['status'] = 'completed'
        except Exception as error:
            state.update(status='user_stopped' if Path(p['stop_file']).exists() else 'needs_attention',
                         error=type(error).__name__ + ': ' + str(error))
        finally:
            state['seconds'] = time.time() - state['started_at']
            write_json(batch / 'state.json', state)
            try:
                verify(batch, full=True)
                write_json(batch / 'final-audit.json', dict(status=state['status'], guards_passed=True,
                    sources=len(read(ROOT / 'manifest.json')), protected=len(read(batch / 'protected-inputs.json'))))
            except Exception as error:
                state.update(status='needs_attention', audit_error=str(error)); write_json(batch / 'state.json', state)
                raise
            finally:
                report(batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--checks', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--repair-from', type=Path)
    parser.add_argument('--continue-from', type=Path)
    parser.add_argument('--repeat-from', type=Path)
    parser.add_argument('--continue-task-errors', action='store_true')
    args = parser.parse_args(); batch = args.directory.resolve()
    if args.worker is not None: worker(batch, args.worker)
    elif args.live: live(batch)
    else:
        if sum(bool(x) for x in (args.repair_from,args.continue_from,args.repeat_from)) > 1:
            parser.error('Choose repair, continuation or repeat, not several')
        register(batch, args.checks.resolve(), args.repair_from.resolve() if args.repair_from else None,
                 args.continue_from.resolve() if args.continue_from else None,
                 args.repeat_from.resolve() if args.repeat_from else None, args.continue_task_errors)
