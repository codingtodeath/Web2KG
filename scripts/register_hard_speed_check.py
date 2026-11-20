"""Register one authorized performance smoke test with the previous task's exact input."""
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


def selected_input(previous):
    config, state = run.verify(previous)
    if state['status'] != 'completed' or config['task_ids'] != [735] or state['tasks']['735']['status'] != 'completed':
        raise ValueError('Require the completed parser-v2 single735 reference')
    # The completed output and learned experience of735 must not leak into its replay.
    seed = Path(config['prepared_graph'])
    original = run.read(previous / '735/input.json')
    expected = run.hashes(Path(original['graph']))
    copied = run.hashes(seed)
    by_relative = lambda rows, base: {str(Path(p).relative_to(base)): h for p, h in rows.items()}
    if by_relative(expected, Path(original['graph'])) != by_relative(copied, seed):
        raise ValueError('Prepared graph differs from the original task input')
    tasks = [r for r in run.read(previous / 'public-inputs.json') if r['task_id'] == 735]
    if len(tasks) != 1 or set(tasks[0]) != {'task_id', 'sites', 'intent', 'start_urls'}:
        raise ValueError('Require exactly the public735 input')
    return config, seed, tasks


def register(batch, previous, checks_path, *, numeric_check=False):
    if batch.exists() or not batch.is_relative_to(ROOT / 'runs'):
        raise ValueError('Use a fresh workspace directory')
    checks = run.read(checks_path)
    if not checks.get('passed'):
        raise ValueError('Offline checks must pass')
    for path, expected in checks['source_sha256'].items():
        if run.file_hash(ROOT / path) != expected:
            raise ValueError('Tested source changed: ' + path)
    config, seed, tasks = selected_input(previous)
    batch.mkdir()
    shutil.copytree(config['initial_graph'], batch / 'initial-graph')
    shutil.copytree(seed, batch / 'seed-graph')
    shutil.copy2(config['initial_experience'], batch / 'initial-experience.json')
    graph = run.Graph.load(batch / 'seed-graph')
    TemplateKnowledge(graph, run.cards_path(graph.directory), run.EventLog(batch / 'registration-events'))
    if len(graph.nodes) != 225 or set(run.read(graph.directory / 'index.json')['ids']) != set(graph.nodes) or not load_framework(graph):
        raise ValueError('Require the complete225 input graph/index/cards/framework')
    encoder = batch / 'encoder-runtime/scripts'; encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py', 'jina_gpu.py', 'jina_vision_fp32.py'):
        shutil.copy2(Path(config['encoder_runtime']) / 'scripts' / name, encoder / name)
    runtime = run.freeze(batch)
    old_source = run.read(Path(config['runtime']) / 'manifest.json')
    new_source = run.read(runtime / 'manifest.json')
    changes = {p: {'before': old_source.get(p), 'after': h} for p, h in new_source.items() if old_source.get(p) != h}
    comparison = ROOT / 'runs/reddit-hard1-speed-flash-20261105-v1' if numeric_check else previous
    comparison_source = run.read(comparison / 'frozen-runtime/manifest.json')
    comparison_changes = {p: {'before': comparison_source.get(p), 'after': h}
                          for p, h in new_source.items() if comparison_source.get(p) != h}
    protocol = {**config, 'version': 'execution-speed-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'authorization': '2026-11-05 user requested timing optimizations without changing core workflow and one Hard rerun.',
        'integrity_policy': 'boundary-v1', 'runtime': str(runtime), 'task_ids': [735],
        'report_title': 'Reddit Hard735 · 时间优化检查', 'previous_version': str(previous),
        'source_graph': str(seed), 'initial_graph': str(batch / 'initial-graph'),
        'prepared_graph': str(batch / 'seed-graph'), 'graph': str(batch / 'seed-graph'),
        'cards': str(run.cards_path(batch / 'seed-graph')), 'experience_store': str(batch / 'experience.json'),
        'initial_experience': str(batch / 'initial-experience.json'), 'stop_file': str(batch / 'STOP'),
        'encoder_runtime': str(encoder.parent), 'inherited_results': [], 'excluded_attempts': [],
        'new_card_preparation': 'Reuse identical pre735 graph/cards/framework and experience; zero model preparation calls.',
        'restart_policy': 'Exactly one new start; engineering failure stops without automatically restarting.',
        'limits': 'One exposed-task developmental timing check. Model trajectory and system load can differ; no causal speedup or generalization claim.'}
    if numeric_check:
        protocol.update(version='numeric-observation-v1', numeric_text_policy='standalone-value-v1',
            authorization='2026-11-05 user explicitly requested a live experiment after numeric observation repair.',
            report_title='Reddit Hard735 · 数值观察修复复测', comparison_version=str(comparison),
            limits='One exposed-task developmental numeric-observation repair check. No generalization or causal speed/success claim.')
    run.write_json(batch / 'protocol.json', protocol)
    run.write_json(batch / 'public-inputs.json', tasks)
    run.write_json(batch / 'experiment-plan.json', dict(reference=str(previous), task_ids=[735],
        changes=changes, comparison_version=str(comparison), changes_since_comparison=comparison_changes,
        observation_check='Confirm independent numeric values reach Actor and normal duplicate labels remain compact.' if numeric_check else None,
        checks=str(checks_path), graph_source=str(seed), experience_source=config['initial_experience'],
        unchanged=['models/prompts/budgets', 'replay permissions', 'clustering and refresh thresholds', 'official evaluator'],
        timing_policy='Report execution, maintenance, queue phase times and all API usage separately.',
        success_criteria=['official score recorded', 'no engineering/model/asset errors', 'no pending jobs', 'complete handoff/index/cards', 'all guards pass']))
    run.write_json(batch / 'run_material_passport.json', dict(mode='run', status='registered', new_starts=1,
        task_ids=[735], models='All Qwen3.7-flash, Bailian direct', paid_preflight_calls=0, preparation_calls=0,
        original225_preserved=True, limits=protocol['limits']))
    current = {**run.hashes(batch / 'initial-graph'), **run.hashes(batch / 'seed-graph'), **run.hashes(encoder.parent)}
    for name in ['protocol.json', 'public-inputs.json', 'experiment-plan.json', 'run_material_passport.json', 'initial-experience.json']:
        current[str((batch / name).resolve())] = run.file_hash(batch / name)
    for path in [ROOT / 'runs/active-graph.json', ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt']:
        current[str(path)] = run.file_hash(path)
    run.write_json(batch / 'runtime-inputs.json', current)
    protected = {**run.read(previous / 'protected-inputs.json'), **current,
                 **{str(Path(config['runtime']) / p): h for p, h in old_source.items()},
                 **run.hashes(Path(config['prepared_graph']))}
    for path in [checks_path, Path(config['initial_experience']), batch / 'runtime-inputs.json']:
        protected[str(path.resolve())] = run.file_hash(path)
    for name in ['state.json', 'summary.json', 'analysis.json', 'final-audit.json', '735/result.json', '735/official.json', '735/input.json']:
        protected[str((previous / name).resolve())] = run.file_hash(previous / name)
    if numeric_check:
        for name in ['protocol.json', 'state.json', 'summary.json', 'analysis.json', 'final-audit.json',
                     'failure-diagnosis.json', '735/result.json', '735/official.json', '735/input.json']:
            protected[str((comparison / name).resolve())] = run.file_hash(comparison / name)
        protected.update({str(comparison / 'frozen-runtime' / p): h for p, h in comparison_source.items()})
    run.write_json(batch / 'protected-inputs.json', protected)
    run.write_json(batch / 'state.json', dict(status='registered', tasks={}, current_graph=str(batch / 'seed-graph')))
    run.report(batch)
    print(dict(registered=str(batch), source_files=len(new_source), full_guards=len(protected), task_guards=len(current), tasks=[735]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['directory', 'previous', 'checks']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--numeric-check', action='store_true')
    args = parser.parse_args()
    register(args.directory.resolve(), args.previous.resolve(), args.checks.resolve(), numeric_check=args.numeric_check)
