"""Independent, exposed failure-case reruns; confidence is shadow-only during calibration."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scripts import run_reddit_hard10_225 as run
from web2kg.stage2.confidence import DEFAULTS, POLICY
from web2kg.stage2.recovery import ExperienceStore, memory_namespace
from web2kg.stage1.site_framework import load_framework
from web2kg.stage2.template_context import TemplateKnowledge

CASES = [(601, 60, 'regional_forum'), (635, 60, 'specialist_forum'),
         (406, 225, 'premature_absence'), (722, 60, 'search_scope'),
         (638, 225, 'specialist_forum'), (715, 225, 'empty_home_scope'),
         (66, 225, 'incomplete_reading'), (68, 225, 'diagnosis_false_positive')]


def register(batch, checks_path):
    if batch.exists() or not batch.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a new run directory')
    checks = run.read(checks_path)
    if not checks.get('passed'):
        raise ValueError('Offline checks must pass')
    for rel, sha in checks['source_sha256'].items():
        if run.file_hash(ROOT/rel) != sha:
            raise ValueError('Tested source changed: '+rel)
    previous = ROOT/'runs/reddit-verified-comparison-flash-20261112-v1'
    latest = run.read(ROOT/'runs/reddit-verified-comparison-flash-20261112-v6/summary.json')
    hard_audit = run.read(ROOT/'runs/experiment-retained-20261112/ours-flash/original-combined-audit.json')
    batch.mkdir()
    runtime = run.freeze(batch)
    seeds, configs = {}, {}
    guard = {}
    for size in (60, 225):
        old = previous/f'ours{size}'
        configs[size] = run.read(old/'protocol.json')
        source = old/'initial-graph'
        seed = batch/f'seed-{size}'
        shutil.copytree(source, seed)
        seeds[size] = seed
        g = run.Graph.load(seed)
        index = run.read(seed/'index.json')
        TemplateKnowledge(g, run.cards_path(seed), run.EventLog(batch/'registration-events'))
        if len(g.nodes) != size or set(g.nodes) != set(index['ids']) or not load_framework(g):
            raise ValueError('Require complete initial graph, cards, framework and index')
        if index['encoder'] != configs[size]['encoder_expected']:
            raise ValueError('Encoder mismatch')
        guard.update(run.hashes(source))
    encoder = batch/'encoder-runtime/scripts'; encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py', 'jina_gpu.py', 'jina_vision_fp32.py'):
        shutil.copy2(Path(configs[225]['encoder_runtime'])/'scripts'/name, encoder/name)
    scope = []
    for task, size, family in CASES:
        prior = next((r for r in latest['arms'][f'ours{size}']['tasks'] if r['task_id'] == task), None)
        if prior is not None and prior.get('official_score') == 0:
            original = Path(prior['source_directory'])
        else:
            prior = next(r for r in hard_audit['rows'] if r['task_id'] == task)
            original = ROOT/'runs'/prior['source_run']/str(task)
        if prior['official_score'] != 0:
            raise ValueError('Selection requires a recorded failed attempt')
        # Read-only original evidence, no hidden evaluator answer enters model inputs.
        for name in ('input.json', 'trajectory.json', 'result.json', 'official.json'):
            path = original/name
            if path.exists(): guard[str(path)] = run.file_hash(path)
        scope.append(dict(task_id=task, graph_capacity=size, family=family,
                          original_attempt=str(original), original_score=0))
    protocol = dict(version='confidence-calibration-v1', created_at=datetime.now(timezone.utc).isoformat(),
        runtime=str(runtime), cases=scope, tasks=[r[0] for r in CASES],
        shared_lock=str(ROOT/'runs/reddit-hard-comparison-v2/queue.lock'), checks=str(checks_path),
        authorization='User authorized rerunning Hard and non-Hard failure cases to calibrate route-confidence triggers; watchdog every10 executed Actor decisions.',
        direction_confidence={**DEFAULTS, 'mode':'shadow'}, paid_preflight=0,
        selection='Eight representative recorded failures spanning wrong forum, premature absence, incomplete reading and erroneous diagnosis. Not an exhaustive failure rerun.',
        exclusions='Provider/format/length errors, correct writes with wrong stop schema, known evaluator placeholder problems and pure budget exhaustion excluded. Similar cases are represented by one or two tasks; no selection based on new results.',
        knowledge_policy='Independent preserved initial60/225 graphs; empty task experience per case. Original per-task Hard graph handoffs were removed in authorized cleanup. These are developmental reruns, not exact replays. No cross-case learning.',
        analysis_plan=dict(labels='Review observed route evidence without confidence scores. Mark each proposed action reasonable, review_needed, or ambiguous. Final failure alone is not a per-step label.',
            immediate_grid=[.2,.3,.4,.5,.6,.7], window_mean_grid=[.4,.5,.6,.7,.8],
            window=3, low_count=2, cooldown=3, watchdog=10,
            validation='Report normal-progress false alarms and first-review delay; hold out task families. No threshold claimed calibrated if confidence is uniformly high, labels ambiguous or too few independent families.'),
        stop_policy='First engineering/permanent/asset error stops; never repeat started cases. Transient identical-input transport retries retain all physical and unknown usage.',
        limits='Exposed selected failures, changed compact Actor context, score prompt and initial knowledge differ from historical attempts. Success does not prove diagnosis/confidence causality; no blind generalization claim.')
    run.write_json(batch/'protocol.json', protocol)
    run.write_json(batch/'state.json', dict(status='registered', tasks={}))
    for path in (ROOT/'runs/active-graph.json', ROOT/'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt'):
        guard[str(path)] = run.file_hash(path)
    for row in scope:
        task, size = row['task_id'], row['graph_capacity']
        child = batch/f'case-{task}'; child.mkdir()
        seed = seeds[size]
        memory = child/'initial-experience.json'
        run.write_json(memory, ExperienceStore(child/'unused-memory.json', memory_namespace(run.Graph.load(seed))).load())
        config = {**configs[size], 'version':'confidence-calibration-v1', 'runtime':str(runtime),
            'workspace':str(runtime), 'task_ids':[task], 'inherited_results':[], 'inherited_tasks':{},
            'inherited_sources':[], 'excluded_attempts':[], 'created_at':protocol['created_at'],
            'source_graph':str(previous/f'ours{size}/initial-graph'), 'initial_graph':str(seed),
            'prepared_graph':str(seed), 'graph':str(seed), 'cards':str(run.cards_path(seed)),
            'initial_experience':str(memory), 'experience_store':str(child/'experience.json'),
            'stop_file':str(batch/'STOP'), 'encoder_runtime':str(encoder.parent),
            'direction_confidence':protocol['direction_confidence'], 'integrity_policy':'boundary-v1',
            'knowledge_policy':protocol['knowledge_policy'], 'limits':protocol['limits'],
            'report_title':f'Reddit {task} · 方向置信度采样 · {size}节点',
            'authorization':protocol['authorization'], 'paid_preflight':0}
        for key in ('model', 'auxiliary_model', 'diagnosis_model'):
            if config[key] != 'qwen3.7-flash': raise ValueError('All new roles must use Flash')
        run.write_json(child/'protocol.json', config)
        public = [r for r in run.read(previous/f'ours{size}/public-inputs.json') if r['task_id'] == task]
        if len(public) != 1: raise ValueError('Missing public task')
        run.write_json(child/'public-inputs.json', public)
        run.write_json(child/'run_material_passport.json', dict(**row, models='All Qwen3.7-flash / Bailian',
            role_limits=dict(actor=31, browser_steps=30, diagnosis_learning=5, diagnoses=3, returns=1),
            direction_confidence=protocol['direction_confidence'], source_graph=config['source_graph'],
            knowledge_policy=protocol['knowledge_policy'], limits=protocol['limits']))
        inputs = {**run.hashes(seed), **run.hashes(encoder.parent)}
        for path in (batch/'protocol.json', checks_path, memory, child/'protocol.json',
                     child/'public-inputs.json', child/'run_material_passport.json'):
            inputs[str(path)] = run.file_hash(path)
        run.write_json(child/'runtime-inputs.json', inputs)
        run.write_json(child/'protected-inputs.json', {**guard, **inputs,
            str(child/'runtime-inputs.json'):run.file_hash(child/'runtime-inputs.json')})
        run.write_json(child/'state.json', dict(status='registered', tasks={}, current_graph=str(seed)))
        run.verify(child); run.report(child)
    print(dict(registered=str(batch), cases=len(scope), mode='shadow', watchdog=10))


def live(batch):
    protocol, state = run.read(batch/'protocol.json'), run.read(batch/'state.json')
    if ROOT != Path(protocol['runtime']) or state['status'] != 'registered' or state['tasks']:
        raise ValueError('Use the frozen entry; no restart of started calibration')
    with run.queue_lock(protocol['shared_lock']), run.queue_lock(batch/'queue.lock'):
        state.update(status='running', started_at=time.time(), pid=os.getpid())
        run.write_json(batch/'state.json', state)
        try:
            for task in protocol['tasks']:
                if (batch/'STOP').exists():
                    state['status']='user_stopped'; break
                child = batch/f'case-{task}'
                state['tasks'][str(task)]='started'; run.write_json(batch/'state.json', state)
                run.queue(child)
                state['tasks'][str(task)] = run.read(child/'state.json')['status']
                run.write_json(batch/'state.json', state)
                if state['tasks'][str(task)] != 'completed':
                    state['status']='needs_attention'; break
            else:
                state['status']='completed'
        except Exception as error:
            state.update(status='needs_attention', error=type(error).__name__+': '+str(error))
            raise
        finally:
            state['seconds']=time.time()-state['started_at']
            run.write_json(batch/'state.json', state)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--checks', type=Path)
    p.add_argument('--live', action='store_true')
    args=p.parse_args()
    if args.live: live(args.directory.resolve())
    else: register(args.directory.resolve(), args.checks.resolve())
