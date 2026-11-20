"""Register four newly authorized tasks with the tested background execution pipeline."""
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


def next_inputs(previous):
    config, state = run.verify(previous)
    completed = [k for k, value in state['tasks'].items() if value['status'] == 'completed']
    if state['status'] != 'needs_attention' or completed != ['625', '596', '721', '31']:
        raise ValueError('Expected the paused v4 completed prefix')
    if set(state['tasks']) != {*completed, '730'} or state['tasks']['730']['status'] != 'needs_attention':
        raise ValueError('Unexpected starts; never retry unresolved tasks')
    seed = Path(state['current_graph'])
    if seed != previous / '31/execution-graph':
        raise ValueError('Use the completed31 graph, never the failed730 graph')
    ordered = run.selected_tasks(ROOT, config['logical_origin'], config['seed'], limit=None)
    tasks = ordered[8:12]
    if [t['task_id'] for t in tasks] != [409, 651, 644, 735]:
        raise ValueError('The authorized next four must follow the original seeded order')
    return config, state, seed, tasks


def register(batch, previous):
    if batch.exists() or not batch.is_relative_to(ROOT / 'runs'):
        raise ValueError('Use a fresh workspace run directory')
    config, old_state, seed, tasks = next_inputs(previous)
    checks_path = ROOT / 'runs/execution-background-dev2-20261104-v1/checks.json'
    checks = run.read(checks_path)
    for rel, expected in checks['source_sha256'].items():
        if run.file_hash(ROOT / rel) != expected:
            raise ValueError('Tested background implementation changed: ' + rel)
    old_manifest = run.read(Path(config['runtime']) / 'manifest.json')
    allowed = {'scripts/run_reddit_hard10_225.py', 'web2kg/common/replay_policy.py',
               'web2kg/stage2/agent.py', 'web2kg/stage2/graph_update.py', 'web2kg/stage2/navigation.py'}
    changes = {}
    for rel, expected in old_manifest.items():
        actual = run.file_hash(ROOT / rel)
        if actual != expected:
            if rel not in allowed:
                raise ValueError('Unexpected method change: ' + rel)
            changes[rel] = {'before': expected, 'after': actual}
    batch.mkdir()
    shutil.copytree(config['initial_graph'], batch / 'initial-graph')
    shutil.copytree(seed, batch / 'seed-graph')
    shutil.copy2(config['experience_store'], batch / 'initial-experience.json')
    graph = run.Graph.load(batch / 'seed-graph')
    TemplateKnowledge(graph, run.cards_path(graph.directory), run.EventLog(batch / 'registration-events'))
    if len(graph.nodes) != 225 or not load_framework(graph) or set(run.read(graph.directory / 'index.json')['ids']) != set(graph.nodes):
        raise ValueError('Require a complete225-node graph, cards, framework and all-node index')
    encoder = batch / 'encoder-runtime/scripts'
    encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py', 'jina_gpu.py', 'jina_vision_fp32.py'):
        shutil.copy2(Path(config['encoder_runtime']) / 'scripts' / name, encoder / name)
    runtime = run.freeze(batch)
    protocol = {**config, 'version': 'hard4-background-execution-v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User requested four tests after optimization, stop and diagnose bugs; explicitly chose409/651/644 plus next seeded task735, no730 retry.',
        'runtime': str(runtime), 'task_ids': [t['task_id'] for t in tasks],
        'report_title': 'Reddit Hard4 · 后台执行优化 · 225节点',
        'previous_version': str(previous), 'source_graph': str(seed),
        'initial_graph': str(batch / 'initial-graph'), 'prepared_graph': str(batch / 'seed-graph'),
        'graph': str(batch / 'seed-graph'), 'cards': str(run.cards_path(batch / 'seed-graph')),
        'experience_store': str(batch / 'experience.json'), 'initial_experience': str(batch / 'initial-experience.json'),
        'stop_file': str(batch / 'STOP'), 'encoder_runtime': str(encoder.parent),
        'inherited_results': [], 'excluded_attempts': [],
        'description_pipeline': {'mode': 'background-compact-v1', 'workers': 2, 'max_pending': 4,
            'pending_edges_replayable': False, 'summary_reuse': True, 'short_evidence': 'B1/A1'},
        'finalization': 'Drain descriptions; prune; overlap fixed-snapshot index and cards; framework follows cards.',
        'new_card_preparation': 'Reuse complete31 cards/framework; zero new preparation model calls.',
        'restart_policy': 'Exactly four new starts. Stop on engineering/permanent-service/asset failure. Never restart any new started entry.',
        'knowledge_policy': 'Independent completed31 graph and verified experience copy. Failed730 graph not inherited. Original225 preserved.',
        'limits': 'Four exposed developmental tasks, new implementation with inherited knowledge; no causal speedup or generalization claim. Prior attempts/costs kept separately, no selected-score replacement.'}
    protocol.pop('failed_version', None)
    run.write_json(batch / 'protocol.json', protocol)
    run.write_json(batch / 'public-inputs.json', tasks)
    run.write_json(batch / 'continuation.json', {'previous': str(previous), 'new_tasks': protocol['task_ids'],
        'not_retried': 730, 'input_graph': str(seed), 'code_changes': changes,
        'added_runtime': 'web2kg/stage2/execution_descriptions.py', 'offline_checks': str(checks_path),
        'prior_attempts': 'Preserve all previous scores/state/STOP and costs; four-task report contains only this new version.'})
    run.write_json(batch / 'run_material_passport.json', {'mode': 'run', 'status': 'registered', 'new_starts': 4,
        'task_ids': protocol['task_ids'], 'model_roles': 'All Qwen3.7-flash, Bailian direct',
        'paid_preflight_calls': 0, 'preparation_calls': 0, 'source_graph': str(seed),
        'original225_preserved': True, 'limits': protocol['limits']})
    protected = {**run.read(previous / 'protected-inputs.json'), **run.hashes(seed),
        **run.hashes(batch / 'initial-graph'), **run.hashes(batch / 'seed-graph'), **run.hashes(encoder.parent)}
    for rel, expected in old_manifest.items():
        protected[str(Path(config['runtime']) / rel)] = expected
    for path in [checks_path, Path(old_state['knowledge_manifest']), Path(config['experience_store'])]:
        protected[str(path)] = run.file_hash(path)
    for parent, names in [(batch, ['protocol.json', 'public-inputs.json', 'continuation.json', 'run_material_passport.json', 'initial-experience.json']),
                          (previous, ['state.json', 'summary.json', 'STOP', 'pause-audit.json', 'pause-after-task.json',
                                      '31/result.json', '31/official.json', '31/handoff.json', '730/result.json', '730/official.json'])]:
        for name in names:
            protected[str(parent / name)] = run.file_hash(parent / name)
    run.write_json(batch / 'protected-inputs.json', protected)
    run.write_json(batch / 'state.json', {'status': 'registered', 'tasks': {}, 'current_graph': str(batch / 'seed-graph')})
    run.report(batch)
    print({'registered': str(batch), 'new_tasks': protocol['task_ids'], 'protected_files': len(protected)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    args = parser.parse_args()
    register(args.directory.resolve(), args.previous.resolve())
