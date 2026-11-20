"""Register only unstarted tasks after an audited ordinary call-budget exhaustion.

The stopped attempt and its status are preserved. Resume from the last completed
knowledge handoff, without changing Agent code or increasing any task budget.
"""
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


def remaining_ids(config, state, result, events):
    ids = config['task_ids']
    started = list(state['tasks'])
    if state['status'] != 'needs_attention' or not started or started != [str(t) for t in ids[:len(started)]]:
        raise ValueError('Require a stopped, ordered prefix')
    if any(state['tasks'][k]['status'] != 'completed' for k in started[:-1]):
        raise ValueError('Earlier tasks must have complete handoffs')
    if state['tasks'][started[-1]]['status'] != 'needs_attention':
        raise ValueError('Preserve the unresolved budget attempt')
    if result.get('status') != 'model_error' or result.get('error') != 'ResponsesError: Responses call budget exhausted':
        raise ValueError('Only ordinary logical-call exhaustion is eligible')
    # The Agent deliberately disables card generation for every model_error,
    # including ordinary call exhaustion. This is why its graph is NOT inherited.
    # Permit only that unattempted maintenance omission, never a failed card call.
    update = dict(result.get('execution_graph') or {})
    cards = update.get('cards') or {}
    if cards.get('stopped') and cards.get('calls') == 0:
        update['cards'] = {**cards, 'stopped': False}
        refresh = update.get('card_refresh') or {}
        update['card_refresh'] = {**refresh, 'pending': [], 'unavailable': []}
    if run.stop_for_error({**result, 'status': 'budget_exhausted', 'execution_graph': update}, events):
        raise ValueError('Other model/service/asset failures prohibit continuation')
    if len(started) < 2 or len(started) >= len(ids):
        raise ValueError('Require a complete predecessor and unstarted tasks')
    return ids[len(started):]


def register(batch, previous):
    if batch.exists() or not batch.is_relative_to(ROOT / 'runs'):
        raise ValueError('Require a fresh workspace output')
    config, state = run.verify(previous)
    failed = list(state['tasks'])[-1]
    folder = previous / failed
    result = run.read(folder / 'result.json')
    events = [e for name in ('events.jsonl', 'graph-annotations/events.jsonl',
                            'graph-cards/events.jsonl', 'diagnosis-calls/events.jsonl')
              for e in run.read_events(folder / name)]
    ids = remaining_ids(config, state, result, events)
    if run.read(folder / 'official.json').get('score') is None:
        raise ValueError('Retain the actual failed-attempt score')
    seed = Path(state['current_graph'])
    predecessor = list(state['tasks'])[-2]
    if seed != previous / predecessor / 'execution-graph':
        raise ValueError('Require the last completed handoff, not the budget-failed graph')
    old_manifest = run.read(Path(config['runtime']) / 'manifest.json')
    for rel, expected in old_manifest.items():
        if run.file_hash(ROOT / rel) != expected:
            raise ValueError('No method changes allowed for remaining tasks: ' + rel)
    batch.mkdir()
    shutil.copytree(config['initial_graph'], batch / 'initial-graph')
    shutil.copytree(seed, batch / 'seed-graph')
    shutil.copy2(config['experience_store'], batch / 'initial-experience.json')
    graph = run.Graph.load(batch / 'seed-graph')
    TemplateKnowledge(graph, run.cards_path(graph.directory), run.EventLog(batch / 'registration-events'))
    if set(run.read(graph.directory / 'index.json')['ids']) != set(graph.nodes) or not load_framework(graph):
        raise ValueError('Require complete index, cards and framework')
    encoder = batch / 'encoder-runtime/scripts'; encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py', 'jina_gpu.py', 'jina_vision_fp32.py'):
        shutil.copy2(Path(config['encoder_runtime']) / 'scripts' / name, encoder / name)
    runtime = run.freeze(batch)
    tasks = [t for t in run.read(previous / 'public-inputs.json') if t['task_id'] in ids]
    protocol = {**config, 'created_at': datetime.now(timezone.utc).isoformat(),
        'version': 'hard10-60-budget-remainder-v1', 'runtime': str(runtime), 'task_ids': ids,
        'authorization': 'Continue only unstarted tasks within the user-authorized ten; no retry, replacement or increased budget.',
        'previous_version': str(previous), 'source_graph': str(seed),
        'initial_graph': str(batch / 'initial-graph'), 'prepared_graph': str(batch / 'seed-graph'),
        'graph': str(batch / 'seed-graph'), 'cards': str(run.cards_path(batch / 'seed-graph')),
        'experience_store': str(batch / 'experience.json'), 'initial_experience': str(batch / 'initial-experience.json'),
        'stop_file': str(batch / 'STOP'), 'encoder_runtime': str(encoder.parent),
        'inherited_results': [], 'excluded_attempts': [],
        'report_title': 'Reddit Hard10 · 60节点 · 剩余未开始任务',
        'knowledge_policy': f'Independent completed{predecessor} handoff and experience. Budget-failed{failed} graph is not inherited; attempt, score and all costs remain in previous run.',
        'restart_policy': 'Only the listed unstarted entries, once each; existing failure-stop rules unchanged.',
        'new_card_preparation': 'Reuse the last completed handoff; no paid preparation or preflight.',
        'limits': config['limits'] + ' Split queue after ordinary budget exhaustion; report the failed attempt and knowledge boundary explicitly.'}
    run.write_json(batch / 'protocol.json', protocol)
    run.write_json(batch / 'public-inputs.json', tasks)
    run.write_json(batch / 'continuation.json', dict(previous=str(previous), new_tasks=ids,
        failed_not_retried=int(failed), seed=str(seed), method_changes={}, all_prior_states_preserved=True))
    run.write_json(batch / 'run_material_passport.json', dict(mode='run', status='registered',
        new_starts=len(ids), models='All Qwen3.7-flash; Bailian direct', paid_preflight_calls=0,
        preparation_calls=0, limits=protocol['limits']))
    inputs = {**run.hashes(batch / 'initial-graph'), **run.hashes(batch / 'seed-graph'), **run.hashes(encoder.parent)}
    for name in ('protocol.json', 'public-inputs.json', 'continuation.json', 'run_material_passport.json', 'initial-experience.json'):
        inputs[str(batch / name)] = run.file_hash(batch / name)
    run.write_json(batch / 'runtime-inputs.json', inputs)
    protected = {**run.read(previous / 'protected-inputs.json'), **run.hashes(seed), **inputs}
    for rel, expected in old_manifest.items():
        protected[str(Path(config['runtime']) / rel)] = expected
    for p in [batch / 'runtime-inputs.json', previous / 'state.json', previous / 'summary.json',
              folder / 'result.json', folder / 'official.json', Path(state['knowledge_manifest']),
              Path(config['experience_store'])]:
        protected[str(p)] = run.file_hash(p)
    run.write_json(batch / 'protected-inputs.json', protected)
    run.write_json(batch / 'state.json', dict(status='registered', tasks={}, current_graph=str(batch / 'seed-graph')))
    run.verify(batch); run.report(batch)
    print(dict(registered=str(batch), tasks=ids, failed_not_retried=failed, seed_task=predecessor))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    a = p.parse_args()
    register(a.directory.resolve(), a.previous.resolve())
