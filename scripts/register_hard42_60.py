"""Register authorized Reddit Hard tasks from an untouched 60-state graph."""
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

SOURCE = ROOT / 'runs/reddit-complete-index-20261104-v1/reddit-jev-flash-expand60-20261104-v1/graph'
REFERENCE = ROOT / 'runs/reddit-hard1-numeric-flash-20261105-v1'
HARD10_REFERENCE = ROOT / 'runs/reddit-hard10-225-flash-20261104-v4/protocol.json'


def task_set(reference, selection='all'):
    tasks = run.selected_tasks(ROOT, reference['logical_origin'], reference['seed'], limit=None)
    if len(tasks) != 42 or len({t['task_id'] for t in tasks}) != 42:
        raise ValueError('Require all42 single-site Reddit Hard tasks')
    if selection == 'all':
        return tasks
    if selection != 'prior-hard10':
        raise ValueError('Unknown task selection')
    prior = run.read(HARD10_REFERENCE)
    ids = [r['task_id'] for r in prior['inherited_results']] + prior['task_ids']
    by_id = {t['task_id']: t for t in tasks}
    if len(ids) != len(set(ids)) or len(ids) != 10 or not set(ids) <= by_id.keys():
        raise ValueError('Require the previous selected ten unique Reddit Hard tasks')
    return [by_id[t] for t in ids]


def settings(batch, reference, tasks):
    config = dict(reference)
    for key in ('initial_experience', 'previous_version', 'comparison_version', 'failed_version'):
        config.pop(key, None)
    count = len(tasks)
    config.update(version=f'hard{count}-60-single-action-v2', created_at=datetime.now(timezone.utc).isoformat(),
        authorization=f'2026-11-06 user authorized {count} Reddit Hard tasks on the retained60 graph with current prompt and same-observation single-action regeneration.',
        runtime=str(batch / 'frozen-runtime'), task_ids=[t['task_id'] for t in tasks],
        source_graph=str(SOURCE), initial_graph=str(batch / 'initial-graph'),
        prepared_graph=str(batch / 'initial-graph'), graph=str(batch / 'initial-graph'),
        cards=str(run.cards_path(SOURCE)).replace(str(SOURCE), str(batch / 'initial-graph')),
        experience_store=str(batch / 'experience.json'), stop_file=str(batch / 'STOP'),
        encoder_runtime=str(batch / 'encoder-runtime'), max_graph_nodes=60,
        report_title=f'Reddit Hard{count} · 60节点 · 单动作提示词验证', inherited_results=[], excluded_attempts=[],
        actor_prompt='single-action-v2', max_output_retries=3, validate_verified_stop=True, unified_verified_stop=True,
        new_card_preparation='Reuse original60 Flash cards/framework and complete60 screenshot index; no paid preparation.',
        knowledge_policy='Original60 graph copied independently. Empty initial task experience; completed task graphs and experience accumulate in fixed order. Published graph unchanged.',
        restart_policy=f'{count} fresh starts, one per task. Stop on engineering/permanent-service/asset failure; no automatic restart.',
        limits='Previously exposed developmental tasks, no control group or causal/generalization claim. New-node protection may temporarily exceed60. Prior results/costs remain separate.',
        recovery_policy='Current-state route first; logical-home fallback; goto included in12-action cap; pending route judgments awaited; exact target verified.')
    return config


def register(batch, checks_path, selection='all'):
    if batch.exists() or not batch.is_relative_to(ROOT / 'runs'):
        raise ValueError('Require a fresh workspace run directory')
    checks = run.read(checks_path)
    if not checks.get('passed'):
        raise ValueError('Offline checks must pass')
    for rel, expected in checks['source_sha256'].items():
        if run.file_hash(ROOT / rel) != expected:
            raise ValueError('Tested source changed: ' + rel)
    reference = run.read(REFERENCE / 'protocol.json')
    tasks = task_set(reference, selection)
    source_hashes = run.hashes(SOURCE)
    batch.mkdir()
    shutil.copytree(SOURCE, batch / 'initial-graph')
    graph = run.Graph.load(batch / 'initial-graph')
    index = run.read(graph.directory / 'index.json')
    TemplateKnowledge(graph, run.cards_path(graph.directory), run.EventLog(batch / 'registration-events'))
    if len(graph.nodes) != 60 or set(index['ids']) != set(graph.nodes) or not load_framework(graph):
        raise ValueError('Require complete60 graph, cards, framework and all-node index')
    if index['encoder'] != reference['encoder_expected']:
        raise ValueError('Encoder identity differs from source index')
    encoder = batch / 'encoder-runtime/scripts'
    encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py', 'jina_gpu.py', 'jina_vision_fp32.py'):
        shutil.copy2(Path(reference['encoder_runtime']) / 'scripts' / name, encoder / name)
    run.freeze(batch)
    config = settings(batch, reference, tasks)
    config['task_selection'] = selection
    if selection == 'prior-hard10':
        config['task_selection_reference'] = str(HARD10_REFERENCE)
        config['task_selection_note'] = 'User confirmed the previous replacement set: omit724, include644. Inherit task IDs only, no old scores or task knowledge.'
    run.write_json(batch / 'protocol.json', config)
    run.write_json(batch / 'public-inputs.json', tasks)
    run.write_json(batch / 'run_material_passport.json', dict(mode='run', status='registered',
        new_starts=len(tasks), models='All Qwen3.7-flash; Bailian direct', initial_states=60, initial_edges=85,
        paid_preflight_calls=0, preparation_calls=0, checks=str(checks_path), limits=config['limits'],
        diagnosis_review=['trigger', 'root step and reason', 'alternative', 'actual return route and failure',
                          'post-diagnosis actions', 'official score', 'learning eligibility', 'role usage and timings']))
    inputs = {**run.hashes(batch / 'initial-graph'), **run.hashes(encoder.parent)}
    for p in [batch / n for n in ('protocol.json', 'public-inputs.json', 'run_material_passport.json')]:
        inputs[str(p)] = run.file_hash(p)
    run.write_json(batch / 'runtime-inputs.json', inputs)
    protected = {**source_hashes, **inputs}
    if selection == 'prior-hard10':
        protected[str(HARD10_REFERENCE)] = run.file_hash(HARD10_REFERENCE)
    for p in [checks_path, batch / 'runtime-inputs.json', ROOT / 'runs/active-graph.json',
              ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt']:
        protected[str(p.resolve())] = run.file_hash(p)
    run.write_json(batch / 'protected-inputs.json', protected)
    run.write_json(batch / 'state.json', dict(status='registered', tasks={}, current_graph=str(batch / 'initial-graph')))
    run.verify(batch)
    run.report(batch)
    print(dict(registered=str(batch), tasks=config['task_ids'], protected_files=len(protected)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--checks', type=Path, required=True)
    p.add_argument('--task-set', choices=['all', 'prior-hard10'], default='all')
    a = p.parse_args()
    register(a.directory.resolve(), a.checks.resolve(), a.task_set)
