"""Register the authorized skip of task724 and its next seeded replacement."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scripts import run_reddit_hard10_225 as run
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.stage1.site_framework import load_framework


def substitution_inputs(previous):
    config, state = run.verify(previous)
    ordered = run.selected_tasks(ROOT, config['logical_origin'], config['seed'], limit=None)
    original = [row['task_id'] for row in ordered[:10]]
    inherited = config.get('inherited_results', [])
    if (state['status'] != 'needs_attention' or config['task_ids'] != original[1:]
            or [row['task_id'] for row in inherited] != original[:1]
            or original[:3] != [646, 598, 724]):
        raise ValueError('Expected the stopped v3 schedule and original seeded order')
    if (set(state['tasks']) != {'598', '724'} or state['tasks']['598']['status'] != 'completed'
            or state['tasks']['724']['status'] != 'needs_attention'):
        raise ValueError('Unexpected starts or incomplete inherited result')
    failed = run.read(previous/'724/result.json')
    receipts = [run.read(path) for path in (previous/'724/llm').glob('*/http_error.json')]
    if (failed.get('provider_failure') != {'kind': 'http', 'http_status': 400}
            or not any(r.get('http_status') == 400 and r.get('code') == 'data_inspection_failed' for r in receipts)):
        raise ValueError('Authorization concerns the recorded content-inspection rejection only')
    graph = Path(state['current_graph'])
    if graph != previous/'598/execution-graph':
        raise ValueError('Never inherit the failed724 graph')
    for row in [*inherited, dict(task_id=598, directory=str(previous/'598'), state=state['tasks']['598'])]:
        if row['state']['status'] != 'completed' or run.read(Path(row['directory'])/'official.json').get('score') not in (0, 1):
            raise ValueError('Inherited result must be completed with an official score')
    # Choose the successor without reading task answers or historical task scores.
    tasks = ordered[3:11]
    return config, state, graph, tasks, [row['task_id'] for row in ordered]


def register(batch, previous):
    if batch.exists() or not batch.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a fresh workspace run directory')
    config, old_state, seed, tasks, ordered = substitution_inputs(previous)
    manifest = run.read(Path(config['runtime'])/'manifest.json')
    changed = 'scripts/run_reddit_hard10_225.py'
    for rel, expected in manifest.items():
        if rel != changed and run.file_hash(ROOT/rel) != expected:
            raise ValueError('Unexpected method change: '+rel)
    if len(run.Graph.load(seed).nodes) != 225:
        raise ValueError('Expected complete225-node handoff')
    batch.mkdir()
    shutil.copytree(config['initial_graph'], batch/'initial-graph')
    shutil.copytree(seed, batch/'seed-graph')
    shutil.copy2(config['experience_store'], batch/'initial-experience.json')
    graph = run.Graph.load(batch/'seed-graph')
    TemplateKnowledge(graph, run.cards_path(graph.directory), run.EventLog(batch/'registration-events'))
    if not load_framework(graph) or set(run.read(graph.directory/'index.json')['ids']) != set(graph.nodes):
        raise ValueError('Complete framework and all-node index required')
    encoder = batch/'encoder-runtime/scripts'; encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py', 'jina_gpu.py', 'jina_vision_fp32.py'):
        shutil.copy2(Path(config['encoder_runtime'])/'scripts'/name, encoder/name)
    runtime = run.freeze(batch)
    inherited = [*config['inherited_results'], dict(task_id=598, directory=str(previous/'598'), state=old_state['tasks']['598'])]
    excluded = [dict(task_id=724, directory=str(previous/'724'), state=old_state['tasks']['724'],
                     reason='百炼内容审核拦截；用户明确授权跳过并按原种子顺延一题')]
    protocol = {**config, 'version':'hard10-225-authorized-substitution-v1',
        'created_at':datetime.now(timezone.utc).isoformat(),
        'authorization':'User explicitly requested skipping724 and substituting the next seeded task; retain646/598 and continue seven unstarted tasks.',
        'runtime':str(runtime), 'task_ids':[row['task_id'] for row in tasks], 'previous_version':str(previous),
        'source_graph':str(seed), 'initial_graph':str(batch/'initial-graph'), 'prepared_graph':str(batch/'seed-graph'),
        'graph':str(batch/'seed-graph'), 'cards':str(run.cards_path(batch/'seed-graph')),
        'experience_store':str(batch/'experience.json'), 'initial_experience':str(batch/'initial-experience.json'),
        'stop_file':str(batch/'STOP'), 'encoder_runtime':str(encoder.parent),
        'inherited_results':inherited, 'excluded_attempts':excluded,
        'new_card_preparation':'Reuse complete598 cards/framework; no new preparation model calls.',
        'restart_policy':'Exactly eight new starts, seven unstarted tasks then next seeded task644. Never restart an unresolved new entry.',
        'knowledge_policy':'Independent complete598 graph/experience copy; no failed724 graph or knowledge. Original225 retained.',
        'limits':'User-authorized multi-version development sample with explicit content-rejection substitution. All excluded attempts/costs remain visible; no best-score selection or generalization claim.'}
    run.write_json(batch/'protocol.json', protocol)
    run.write_json(batch/'public-inputs.json', tasks)
    run.write_json(batch/'substitution.json', {'previous':str(previous), 'skip':724, 'replacement':tasks[-1]['task_id'],
        'seed':config['seed'], 'full_seeded_hard_order':ordered, 'new_tasks':protocol['task_ids'],
        'inherited_completed':[646,598], 'input_graph':str(seed),
        'method_source_unchanged':True, 'scheduler_change':{'file':changed, 'before':manifest[changed], 'after':run.file_hash(ROOT/changed)},
        'past_attempts_and_costs':'Preserve all prior attempts including724 official0; exclusion only affects selected ten-task view.'})
    run.write_json(batch/'run_material_passport.json', {'mode':'run', 'new_starts':8, 'inherited_results':[646,598],
        'excluded_attempt':724, 'replacement_task':tasks[-1]['task_id'], 'model_roles':'All Qwen3.7-flash, Bailian direct',
        'paid_preflight_calls':0, 'preparation_calls':0, 'source_graph':str(seed), 'original225_preserved':True,
        'limitations':protocol['limits']})
    protected = {**run.read(previous/'protected-inputs.json'), **run.hashes(seed),
        **run.hashes(batch/'initial-graph'), **run.hashes(batch/'seed-graph'), **run.hashes(encoder.parent)}
    for rel, expected in manifest.items(): protected[str(Path(config['runtime'])/rel)] = expected
    for parent, names in [(batch,['protocol.json','public-inputs.json','substitution.json','run_material_passport.json','initial-experience.json']),
                          (previous,['state.json','summary.json','final-audit.json','failure-diagnosis.json','598/result.json','598/official.json','598/handoff.json','724/result.json','724/official.json'])]:
        for name in names: protected[str(parent/name)] = run.file_hash(parent/name)
    for path in (ROOT/'experiments/webarena-verified/assets/dataset/webarena-verified.json',
                 ROOT/'experiments/webarena-verified/assets/dataset/subsets/webarena-verified-hard.json',
                 Path(config['runtime'])/'manifest.json'):
        protected[str(path)] = run.file_hash(path)
    run.write_json(batch/'protected-inputs.json', protected)
    run.write_json(batch/'state.json', {'status':'registered','tasks':{},'current_graph':str(batch/'seed-graph')})
    run.report(batch)
    print({'registered':str(batch),'new_tasks':protocol['task_ids'],'inherited':[646,598], 'skipped':724,
           'replacement':tasks[-1]['task_id'], 'protected_files':len(protected)})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True); parser.add_argument('--previous',type=Path,required=True)
    args=parser.parse_args(); register(args.directory.resolve(),args.previous.resolve())
