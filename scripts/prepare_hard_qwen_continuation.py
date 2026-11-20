"""Register the user-authorized Qwen run and original-method continuation without starting tasks."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from dotenv import dotenv_values
import yaml

from run_hard_qwen_continuation import digest, read, write


ROOT = Path(__file__).resolve().parents[1]


def prepare():
    old = ROOT / 'runs/reddit-hard-comparison-v2'
    batch = ROOT / 'runs/reddit-hard-qwen-v1'
    original, state = read(old / 'protocol.json'), read(old / 'state.json')
    ids = original['task_ids']
    remaining = [t for t in ids if f'ours/{t}' not in state['tasks']]
    assert len(ids) == 42 and len(remaining) == 20
    assert all(r['status'] == 'completed' or k == 'ours/29' for k, r in state['tasks'].items())
    assert read(old / 'ours/29/official.json')['score'] == 0
    failed = read(old / 'ours/29/result.json')
    assert failed['steps'] == failed['atomic_actions'] == 0
    assert not any(json.loads(line)['event'] == 'responses_started' for line in
                   (old / 'ours/29/events.jsonl').read_text('utf-8').splitlines())
    guards = dict(original['protected_inputs'])
    guards.update(read(old / 'webnavigator-continuation.json')['guards'])
    for path, expected in guards.items():
        assert digest(path) == expected, path
    if (batch / 'protocol.json').exists() or (batch / 'state.json').exists():
        raise ValueError('Batch already registered; never prepare again')
    batch.mkdir(exist_ok=True)
    runtime = batch / 'frozen-ours-runtime'
    source = old / 'frozen-ours-runtime'
    manifest = read(source / 'manifest.json')
    for name, expected in manifest.items():
        assert digest(source / name) == expected
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    write(runtime / 'manifest.json', manifest)
    candidate = old / 'maintenance-candidate'
    resources = read(candidate / 'manifest.json')
    for resource in resources['files']:
        path = candidate / 'static-resources' / resource['path']
        assert digest(path) == resource['sha256']
        shutil.copy2(path, runtime / resource['path'])
    if not (runtime / 'auth').exists():
        subprocess.run(['powershell.exe', '-NoProfile', '-Command',
            "New-Item -ItemType Junction -Path '" + str(runtime / 'auth') + "' -Target '" + str(ROOT / 'auth') + "' | Out-Null"],
            check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    baseline_runtime = old / 'frozen-webnavigator-runtime'
    base = batch / 'webnavigator'
    for name in ('upstream', 'import-stubs', 'released-graph'):
        shutil.copytree(old / 'webnavigator' / name, base / name, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    config = yaml.safe_load((old / 'webnavigator/config.yml').read_text('utf-8'))
    config['logdir'] = config['agent']['others']['logdir'] = str(base / 'upstream-logs')
    config['agent']['actor']['domain_paths']['reddit'] = str(base / 'released-graph')
    (base / 'config.yml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    for arm in ('ours', 'webnavigator'):
        (batch / arm).mkdir(exist_ok=True)
        shutil.copy2(old / arm / 'public-inputs.json', batch / arm / 'public-inputs.json')
    memory = old / 'ours-experiences.json'
    if memory.exists():
        shutil.copy2(memory, batch / 'ours-experiences.json')
    # Explicit user request authorizes replacing the old ORCA entries in the current .env only.
    credentials = dotenv_values('D:/Agent/MyCode/web2kg/.env')
    private = ROOT / 'auth/qwen-hard.env'
    private.write_text('\n'.join(k + '=' + credentials[k] for prefix in ('OPENROUTER', 'BAILIAN')
        for k in (prefix + '_API_KEY', prefix + '_BASE_URL', prefix + '_MODEL')) + '\n', encoding='utf-8')
    scripts = [ROOT / 'scripts' / name for name in ('run_hard_qwen_continuation.py', 'qwen_hard_transport.py', 'prepare_hard_qwen_continuation.py')]
    for root in (runtime, baseline_runtime, base / 'upstream', base / 'import-stubs', base / 'released-graph'):
        for path in root.rglob('*'):
            if root in (runtime, baseline_runtime) and path.relative_to(root).parts[0] == 'runs':
                continue
            if path.is_file() and 'auth' not in path.relative_to(root).parts and path.suffix != '.pyc' and '__pycache__' not in path.parts:
                guards[str(path)] = digest(path)
    for arm in ('ours', 'webnavigator'):
        guards[str(batch / arm / 'public-inputs.json')] = digest(batch / arm / 'public-inputs.json')
    for path in [*scripts, private, ROOT / 'auth/llm-crawl.env', old / 'state.json', old / 'summary.json',
                 *([memory] if memory.exists() else []), old / 'webnavigator-closeout.json', base / 'config.yml']:
        guards[str(path)] = digest(path)
    for key, row in state['tasks'].items():
        folder = Path(row.get('evidence_directory', old / key))
        for name in ('result.json', 'official.json', 'events.jsonl'):
            guards[str(folder / name)] = digest(folder / name)
    current_graph = Path(state['current_graph'])
    # The failed task performed no actions; resume the last committed graph and memory.
    before, after = read(current_graph / 'graph.json'), read(old / 'ours/29/execution-graph/graph.json')
    assert before['nodes'] == after['nodes'] and before['edges'] == after['edges']
    for path in current_graph.rglob('*'):
        if path.is_file():
            guards[str(path)] = digest(path)
    write(batch / 'maintenance-audit.json', {'authorization': '2026-10-15 user requested continuing interrupted original-method experiment',
        'old_task': str(old / 'ours/29'), 'official_score_retained': 0, 'original_state_untouched': True,
        'old_runtime_untouched': True, 'new_runtime': str(runtime), 'static_resources': resources,
        'python_source_sha256': original['frozen_ours_source_sha256'], 'task_actions': 0, 'model_requests': 0,
        'failed_task_nodes_edges_equal_previous': True, 'resume_graph': str(current_graph),
        'handoff_resolution': 'Discard incomplete zero-action handoff; retain last committed graph and memory; do not rerun29 or learn from failure.'})
    inherited = {}
    for key, row in state['tasks'].items():
        if key.startswith('ours/'):
            inherited[key] = {**row, 'evidence_directory': row.get('evidence_directory', str(old / key)), 'inherited_from': str(old)}
            if key == 'ours/29':
                inherited[key].update(status='completed', original_status='needs_attention',
                    classification='infrastructure_failure_official_zero', maintenance_resolution=str(batch / 'maintenance-audit.json'))
    schedule = []
    for i, task_id in enumerate(ids):
        schedule.append({'arm': 'webnavigator', 'task_id': task_id})
        if i < len(remaining):
            schedule.append({'arm': 'ours', 'task_id': remaining[i]})
    protocol = {**original, 'guards': guards, 'absent_inputs': [] if memory.exists() else [str(memory)], 'previous_batch': str(old), 'task_ids': ids,
        'schedule': schedule, 'max_new_task_starts': 62, 'stop_file': str(batch / 'STOP'),
        'report_path': str(old / 'report.html'), 'shared_lock': str(old / 'queue.lock'),
        'credential_file': str(private), 'transport': str(ROOT / 'scripts/qwen_hard_transport.py'),
        'runtimes': {'ours': str(runtime), 'webnavigator': str(baseline_runtime)},
        'authorization': '2026-10-15 user: OpenRouter qwen/qwen3.8-27b:free baseline; daily quota exhaustion -> Bailian. Explicit followup allows Bailian while free model unavailable; prefer OpenRouter upon recovery. Resume unstarted original-method tasks with original models. Show Qwen in existing v2 report.',
        'comparison_limits': original['comparison_limits'] + ['Qwen changes model/provider and uses Chat Completions transport; provider fallback is recorded per physical request. Original-method code is original snapshot with six same-commit missing resources restored in new copy. No dev YAML/context changes.'],
        'routing_policy': {'primary': 'openrouter:qwen/qwen3.8-27b:free', 'fallback': 'bailian:qwen3.8-27b',
                           'fallback_reasons': ['daily_quota', 'free_unavailable'], 'catalog_probe_seconds': 1800,
                           'quota_reset': 'UTC day', 'temporary_errors': 'retry same provider/input indefinitely 2..60seconds',
                           'reasoning': 'provider defaults; no hidden reasoning policy changes'}}
    write(batch / 'protocol.json', protocol)
    for arm in ('ours', 'webnavigator'):
        config = read(old / arm / 'protocol.json')
        config.update(stop_file=protocol['stop_file'], protected_inputs=guards)
        if arm == 'ours':
            config['experience_store'] = str(batch / 'ours-experiences.json')
        else:
            config['model'] = 'qwen/qwen3.8-27b:free'
        write(batch / arm / 'protocol.json', config)
    write(batch / 'state.json', {'tasks': inherited, 'current_graph': state['current_graph'],
        'current_graph_sha256': state['current_graph_sha256'], 'status': 'registered'})
    # Separate immutable protocol hashes avoid circular hashes inside the registration.
    write(batch / 'registration-integrity.json', {str(p): digest(p) for p in
        [batch / 'protocol.json', batch / 'ours/protocol.json', base / 'protocol.json', batch / 'maintenance-audit.json']})
    print(json.dumps({'new_qwen_tasks': 42, 'remaining_ours': remaining, 'guards': len(guards), 'directory': str(batch)}))


if __name__ == '__main__':
    prepare()
