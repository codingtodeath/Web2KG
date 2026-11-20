"""Register a user-authorized continuation without changing paused evidence."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from scripts import run_midterm_qwen as run
from web2kg.common.storage import write_json


def continuation_plan(protocol, state, pause, result):
    """Only an explicit user-stop interruption may get a new attempt here."""
    interrupted = pause['interrupted_tasks']
    if len(interrupted) != 1 or not pause.get('queue_and_workers_exited'):
        raise ValueError('Expected one verified user interruption')
    key = interrupted[0]
    if (result.get('provider_failure', {}).get('kind') != 'cancelled'
            or 'User stop file present' not in result.get('error', '')
            or state['tasks'].get(key, {}).get('status') != 'needs_attention'):
        raise ValueError('Not a user-stop cancellation; diagnose instead of restarting')
    if state['current_graph'] != pause['last_committed_graph'] or state['tasks'][key]['graph_before'] != state['current_graph']:
        raise ValueError('Interrupted graph cannot be used as committed knowledge')
    completed = {k: deepcopy(v) for k, v in state['tasks'].items() if v['status'] == 'completed'}
    if set(state['tasks']) != set(completed) | {key} or len(completed) != pause['completed_starts']:
        raise ValueError('Other unresolved starts exist')
    keys = [f"{r['arm']}/{r['task_id']}" for r in protocol['schedule']]
    if len(keys) != len(set(keys)) or keys[:len(completed)] != list(completed) or keys[len(completed)] != key:
        raise ValueError('Completed entries must be the unchanged schedule prefix')
    if state['tasks'][key].get('official_score') is not None:
        raise ValueError('Interrupted entry was unexpectedly committed')
    return completed, key


def register(previous, suite):
    old_batch = previous / 'reddit'
    old, state = run.read(old_batch / 'protocol.json'), run.read(old_batch / 'state.json')
    pause = run.read(previous / 'user-pause.json')
    if not (previous / 'STOP').exists():
        raise ValueError('Preserve the paused STOP before registering a continuation')
    interrupted = pause['interrupted_tasks'][0]
    completed, interrupted = continuation_plan(old, state, pause, run.read(old_batch / interrupted / 'result.json'))
    previous_root = run.base.ROOT
    try:
        run.base.ROOT = Path(old['frozen_runtime'])
        run.verify(old_batch, interrupted)
    finally:
        run.base.ROOT = previous_root
    with run.queue_lock(old['shared_lock']):
        suite.mkdir(exist_ok=False)
        runtime = suite / 'frozen-runtime'
        old_runtime = Path(old['frozen_runtime'])
        old_manifest = run.read(old_runtime / 'manifest.json')
        for relative, digest in old_manifest.items():
            source, target = old_runtime / relative, runtime / relative
            if run.file_hash(source) != digest:
                raise ValueError('Frozen source manifest changed: ' + relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        # Agent/prompts and all other implementation files remain byte-identical.
        shutil.copy2(ROOT / 'scripts/run_midterm_qwen.py', runtime / 'scripts/run_midterm_qwen.py')
        manifest = {relative: run.file_hash(runtime / relative) for relative in old_manifest}
        changed = [p for p in manifest if manifest[p] != old_manifest[p]]
        if changed != ['scripts/run_midterm_qwen.py']:
            raise ValueError('Unexpected method source changes: ' + repr(changed))
        write_json(runtime / 'manifest.json', manifest)
        subprocess.run(['powershell.exe', '-NoProfile', '-Command',
            f"New-Item -ItemType Junction -Path '{runtime / 'auth'}' -Target '{ROOT / 'auth'}' | Out-Null"],
            check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        for source in run.encoder_material(previous / 'encoder-runtime'):
            target = suite / 'encoder-runtime' / source.relative_to(previous / 'encoder-runtime')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        batch = suite / 'reddit'
        for arm in run.ARMS:
            (batch / arm).mkdir(parents=True)
            shutil.copy2(old_batch / arm / 'public-inputs.json', batch / arm / 'public-inputs.json')
            routing = old_batch / arm / 'routing-state.json'
            if routing.exists():
                shutil.copy2(routing, batch / arm / routing.name)
        if (old_batch / 'routing-state.json').exists():
            shutil.copy2(old_batch / 'routing-state.json', batch / 'routing-state.json')
        for name in ('upstream', 'import-stubs'):
            shutil.copytree(old_batch / 'webnavigator' / name, batch / 'webnavigator' / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        released = (old_batch / 'webnavigator/released-graph').resolve()
        subprocess.run(['powershell.exe', '-NoProfile', '-Command',
            f"New-Item -ItemType Junction -Path '{batch / 'webnavigator/released-graph'}' -Target '{released}' | Out-Null"],
            check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        # Change only the baseline asset directory in its saved config.
        cfg = (old_batch / 'webnavigator/config.yml').read_text('utf-8')
        for old_path, new_path in [(str(old_batch / 'webnavigator'), str(batch / 'webnavigator')),
                                   ((old_batch / 'webnavigator').as_posix(), (batch / 'webnavigator').as_posix())]:
            cfg = cfg.replace(old_path, new_path)
        (batch / 'webnavigator/config.yml').write_text(cfg, encoding='utf-8')
        memory = batch / 'ours/experiences.json'
        shutil.copy2(old['experience_store'], memory)
        shutil.copy2(memory, memory.with_name('initial-experiences.json'))
        guards = dict(old['protected_inputs'])
        # Protect prior task outputs, scores, traces and their knowledge receipts.
        for key in state['tasks']:
            folder = old_batch / key
            for path in folder.rglob('*'):
                if path.is_file() and path.suffix != '.pyc':
                    guards[str(path.resolve())] = run.file_hash(path)
            trajectory = folder.parent / 'trajectories' / f'task-{folder.name}.html'
            if trajectory.exists():
                guards[str(trajectory.resolve())] = run.file_hash(trajectory)
        for name in ('suite.json', 'STOP', 'user-pause.json', 'report.html', 'summary.json', 'repair-lineage.json'):
            path = previous / name
            guards[str(path)] = run.file_hash(path)
        for path in [old_batch / 'state.json', old_batch / 'ours/experiences.json',
                     previous / 'encoder-check.json', ROOT / 'runs/active-graph.json']:
            guards[str(path)] = run.file_hash(path)
        for directory in (runtime, suite / 'encoder-runtime', batch / 'webnavigator/upstream', batch / 'webnavigator/import-stubs'):
            paths = ([directory / p for p in manifest] + [directory / 'manifest.json'] if directory == runtime
                     else run.encoder_material(directory) if directory == suite / 'encoder-runtime'
                     else directory.rglob('*'))
            for path in paths:
                if path.is_file() and path.suffix != '.pyc':
                    guards[str(path.resolve())] = run.file_hash(path)
        for path in [batch / 'webnavigator/config.yml', *(batch / a / 'public-inputs.json' for a in run.ARMS), memory.with_name('initial-experiences.json')]:
            guards[str(path)] = run.file_hash(path)
        original_root = run.base.ROOT
        try:
            run.base.ROOT = runtime
            source_sha = run.base.source_hash()
        finally:
            run.base.ROOT = original_root
        evidence = {key: str(old_batch / key) for key in completed}
        config = {**old, 'workspace': str(runtime), 'frozen_runtime': str(runtime), 'source_sha256': source_sha,
                  'encoder_runtime': str(suite / 'encoder-runtime'), 'protected_inputs': guards,
                  'graph': state['current_graph'], 'cards': str(run.cards_path(Path(state['current_graph']))),
                  'stop_file': str(suite / 'STOP'), 'experience_store': str(memory),
                  'inherited_completed': evidence, 'max_new_task_starts': len(old['schedule']) - len(completed)}
        manifest_path, digest = run.knowledge_manifest(Path(state['current_graph']), memory, batch / 'initial-handoff.json')
        now = datetime.now(timezone.utc).isoformat()
        lineage = dict(previous=str(previous), authorization='2026-10-28 user explicitly requests continue experiment after pause',
            created=now, inherited_completed=len(completed), remaining_new_starts={'reddit': config['max_new_task_starts'], 'gitlab': 114},
            interrupted_attempts=[dict(site='reddit', arm=interrupted.split('/')[0], task_id=int(interrupted.split('/')[1]),
                evidence_directory=str(old_batch / interrupted), reason='User pause; excluded from current score, usage retained separately')],
            inherited_preflight=run.read(previous / 'repair-lineage.json')['inherited_preflight'],
            original_method_source_sha256=old['source_sha256'], continuation_source_sha256=source_sha,
            changed_runtime_files=changed, source_policy='Only scheduler inheritance/report support changed; all Agent/model/prompt files byte-identical.',
            knowledge_source=state['current_graph'], old_stop_retained=True, old_evidence_immutable=True)
        write_json(suite / 'continuation.json', lineage)
        guards[str(suite / 'continuation.json')] = run.file_hash(suite / 'continuation.json')
        write_json(batch / 'protocol.json', config)
        write_json(batch / 'state.json', dict(status='registered', tasks=completed, current_graph=state['current_graph'],
            knowledge_manifest=manifest_path, knowledge_manifest_sha256=digest))
        passport = {**run.read(old_batch / 'run_material_passport.json'), 'authorization': lineage['authorization'],
                    'source_sha256': source_sha, 'continuation': str(suite / 'continuation.json')}
        write_json(batch / 'run_material_passport.json', passport)
        write_json(batch / 'registration-integrity.json', {str(p): run.file_hash(p) for p in
            (batch / 'protocol.json', batch / 'run_material_passport.json', batch / 'initial-handoff.json', suite / 'continuation.json')})
        status = run.read(previous / 'suite.json')
        status.update(status='registered', created=now, continuation_of=str(previous),
                      planned_new={'reddit': config['max_new_task_starts'], 'gitlab': 114})
        status.pop('error', None)
        write_json(suite / 'suite.json', status)
        run.report(suite)
        print(json.dumps(dict(status='registered', inherited=len(completed), new_reddit=config['max_new_task_starts'],
                              protected_files=len(guards), source_sha256=source_sha)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True, type=Path)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    register(args.previous.resolve(), args.directory.resolve())
