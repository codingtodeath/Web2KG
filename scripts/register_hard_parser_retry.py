"""Register the authorized 644 retry and interrupted 735 in a new frozen batch."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from scripts import run_reddit_hard10_225 as run
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.stage1.site_framework import load_framework


def retry_inputs(previous):
    config, state = run.verify(previous)
    diagnosis = run.read(previous / 'failure-diagnosis.json')
    if diagnosis.get('discovered_in_task') == 644:
        expected = {'409': 'completed', '651': 'completed', '644': 'completed', '735': 'needs_attention'}
        retry, donor, task_ids = 644, 651, [644, 735]
    elif diagnosis.get('bug') == 'identical_action_rejected' and diagnosis.get('discovered_in_task') == 735:
        expected = {'644': 'completed', '735': 'needs_attention'}
        retry, donor, task_ids = 735, 644, [735]
    else:
        raise ValueError('Require a confirmed parser diagnosis')
    if state['status'] != 'user_stopped' or {k: v['status'] for k, v in state['tasks'].items()} != expected:
        raise ValueError('Require the stopped diagnosed batch; no additional starts')
    if diagnosis.get('kind') != 'engineering_stop':
        raise ValueError('Require a confirmed parser diagnosis')
    seed = previous / f'{donor}/execution-graph'
    original_input = run.read(previous / f'{retry}/input.json')
    if Path(original_input['graph']) != seed or Path(state['current_graph']) != previous / '644/execution-graph':
        raise ValueError(f'Retry{retry} from its original{donor} input, never from its own output')
    tasks = [t for t in run.read(previous / 'public-inputs.json') if t['task_id'] in task_ids]
    if [t['task_id'] for t in tasks] != task_ids:
        raise ValueError('Only the authorized retry sequence is allowed')
    return config, state, seed, tasks


def register(batch, previous, checks_path):
    if batch.exists() or not batch.is_relative_to(ROOT / 'runs'):
        raise ValueError('Use a fresh workspace run directory')
    config, old_state, seed, tasks = retry_inputs(previous)
    checks = run.read(checks_path)
    if not checks.get('passed'):
        raise ValueError('Offline parser and registrar checks must pass')
    for rel, expected in checks['source_sha256'].items():
        if run.file_hash(ROOT / rel) != expected:
            raise ValueError('Tested source changed: ' + rel)
    old_manifest = run.read(Path(config['runtime']) / 'manifest.json')
    changes = {}
    for rel, expected in old_manifest.items():
        actual = run.file_hash(ROOT / rel)
        if actual != expected:
            if rel not in {'web2kg/stage2/action_notes.py', 'scripts/register_hard_parser_retry.py'}:
                raise ValueError('Unexpected runtime change: ' + rel)
            changes[rel] = {'before': expected, 'after': actual}
    if 'web2kg/stage2/action_notes.py' not in changes:
        raise ValueError('Require exactly the tested Actor parser fix')
    # Never inherit the failed task's own graph or later experience.
    handoff = run.read(seed.parent / 'handoff.json')
    if run.file_hash(handoff['knowledge_manifest']) != handoff['knowledge_manifest_sha256']:
        raise ValueError('Original input handoff manifest changed')
    knowledge = run.read(handoff['knowledge_manifest'])
    for path, expected in knowledge.items():
        if run.file_hash(path) != expected:
            raise ValueError('Original input graph or experience changed: ' + path)
    batch.mkdir()
    shutil.copytree(config['initial_graph'], batch / 'initial-graph')
    shutil.copytree(seed, batch / 'seed-graph')
    shutil.copy2(config['experience_store'], batch / 'initial-experience.json')
    graph = run.Graph.load(batch / 'seed-graph')
    TemplateKnowledge(graph, run.cards_path(graph.directory), run.EventLog(batch / 'registration-events'))
    if len(graph.nodes) != 225 or not load_framework(graph) or set(run.read(graph.directory / 'index.json')['ids']) != set(graph.nodes):
        raise ValueError('Require complete225 graph, cards, framework and index')
    encoder = batch / 'encoder-runtime/scripts'
    encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py', 'jina_gpu.py', 'jina_vision_fp32.py'):
        shutil.copy2(Path(config['encoder_runtime']) / 'scripts' / name, encoder / name)
    runtime = run.freeze(batch)
    task_ids = [t['task_id'] for t in tasks]
    donor = int(seed.parent.name)
    protocol = {**config, 'version': 'action-parser-repair-v2',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'authorization': '2026-11-05 user explicitly requested parser repair, retry affected644 and remaining735.',
        'runtime': str(runtime), 'task_ids': task_ids,
        'report_title': 'Reddit Hard · ACTION解析修复 · ' + '/'.join(map(str, task_ids)),
        'previous_version': str(previous), 'source_graph': str(seed),
        'initial_graph': str(batch / 'initial-graph'), 'prepared_graph': str(batch / 'seed-graph'),
        'graph': str(batch / 'seed-graph'), 'cards': str(run.cards_path(batch / 'seed-graph')),
        'experience_store': str(batch / 'experience.json'), 'initial_experience': str(batch / 'initial-experience.json'),
        'stop_file': str(batch / 'STOP'), 'encoder_runtime': str(encoder.parent),
        'inherited_results': [], 'excluded_attempts': [],
        'new_card_preparation': f'Reuse exact completed{donor} cards/framework and experience; zero new preparation model calls.',
        'restart_policy': f'Exactly{len(tasks)} new starts. Never restart any started entry; stop and diagnose engineering/permanent-service/asset failures.',
        'knowledge_policy': f'Retry{task_ids[0]} from its original{donor} input; subsequent tasks use new completed handoffs. No failed task output inherited.',
        'limits': 'Authorized exposed-task developmental retries, not a blind or causal test. Old409/651 completed scores and old644/735 attempts/costs remain separate. No best-score selection.'}
    run.write_json(batch / 'protocol.json', protocol)
    run.write_json(batch / 'public-inputs.json', tasks)
    run.write_json(batch / 'repair-lineage.json', dict(previous=str(previous), task_ids=task_ids,
        input_graph=str(seed), code_changes=changes, checks=str(checks_path),
        diagnosis=str(previous / 'failure-diagnosis.json'), prior_results_preserved=list(map(int, old_state['tasks'])),
        old644_result_not_selected_by_score=True, old735_partial_graph_not_inherited=True))
    run.write_json(batch / 'run_material_passport.json', dict(mode='run', status='registered', new_starts=len(tasks),
        task_ids=task_ids, model_roles='All Qwen3.7-flash, Bailian direct', paid_preflight_calls=0,
        preparation_calls=0, original225_preserved=True, source_graph=str(seed), limitations=protocol['limits']))
    protected = {**run.read(previous / 'protected-inputs.json'), **knowledge,
        **run.hashes(batch / 'initial-graph'), **run.hashes(batch / 'seed-graph'), **run.hashes(encoder.parent)}
    for rel, expected in old_manifest.items():
        protected[str(Path(config['runtime']) / rel)] = expected
    for path in [checks_path, Path(old_state['knowledge_manifest']), Path(config['experience_store']),
                 Path(handoff['knowledge_manifest'])]:
        protected[str(path)] = run.file_hash(path)
    for parent, names in [(batch, ['protocol.json', 'public-inputs.json', 'repair-lineage.json', 'run_material_passport.json', 'initial-experience.json']),
                          (previous, ['state.json', 'summary.json', 'STOP', 'final-audit.json', 'failure-diagnosis.json', 'operator-stop.json'])]:
        for name in names:
            protected[str(parent / name)] = run.file_hash(parent / name)
    for tid in map(int, old_state['tasks']):
        for name in ['result.json', 'official.json', 'input.json']:
            path = previous / str(tid) / name
            protected[str(path)] = run.file_hash(path)
    run.write_json(batch / 'protected-inputs.json', protected)
    run.write_json(batch / 'state.json', dict(status='registered', tasks={}, current_graph=str(batch / 'seed-graph')))
    run.report(batch)
    print(dict(registered=str(batch), task_ids=task_ids, source_files=len(run.read(runtime / 'manifest.json')),
               protected_files=len(protected)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--checks', type=Path, required=True)
    args = parser.parse_args()
    register(args.directory.resolve(), args.previous.resolve(), args.checks.resolve())
