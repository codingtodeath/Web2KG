"""Register the remaining31 tasks and reference the user's prior225-node Hard10."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from scripts import run_reddit_hard10_225 as run
from scripts.run_midterm_qwen import protect_graph
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.stage1.site_framework import load_framework

REFERENCE = ROOT / 'runs/reddit-hard1-numeric-flash-20261105-v1'
SEED_RUN = ROOT / 'runs/reddit-hard2-parser-flash-20261105-v1'
SEED = SEED_RUN / '644/execution-graph'
ORIGINAL = ROOT / 'runs/reddit-complete-index-20261104-v1/reddit-jev-flash-expand225-20261104-v2/graph'
SOURCES = [
    ('reddit-hard10-225-flash-20261104-v2', [646]),
    ('reddit-hard10-225-flash-20261104-v3', [598]),
    ('reddit-hard10-225-flash-20261104-v4', [625,596,721,31,730]),
    ('reddit-hard4-background-flash-20261104-v1', [409,651]),
    ('reddit-hard2-parser-flash-20261105-v1', [644]),
]


def selected_inputs(reference):
    inherited = []
    for name, ids in SOURCES:
        prior = ROOT / 'runs' / name
        config, state = run.read(prior / 'protocol.json'), run.read(prior / 'state.json')
        if config['max_graph_nodes'] != 225 or config['model'] != 'qwen3.7-flash':
            raise ValueError('Require the prior225 Flash lineage')
        for task in ids:
            folder = prior / str(task)
            if run.read(folder / 'official.json').get('score') not in (0,1):
                raise ValueError('Inherited official score missing')
            inherited.append(dict(task_id=task, directory=str(folder), state=state['tasks'][str(task)]))
    old_ids = [r['task_id'] for r in inherited]
    if old_ids != [646,598,625,596,721,31,730,409,651,644]:
        raise ValueError('Prior replacement ten must stay fixed independently of scores')
    full = run.selected_tasks(ROOT, reference['logical_origin'], reference['seed'], limit=None)
    tasks = [t for t in full if t['task_id'] not in set(old_ids) | {724}]
    assert len(full) == 42 and len(tasks) == 31 and len(set(old_ids) | {t['task_id'] for t in tasks}) == 41
    return inherited, tasks


def register(batch, checks_path):
    if batch.exists() or not batch.is_relative_to(ROOT / 'runs'):
        raise ValueError('Require a fresh run directory')
    checks = run.read(checks_path)
    if not checks.get('passed'):
        raise ValueError('Offline validation required')
    for p,h in checks['source_sha256'].items():
        if run.file_hash(ROOT / p) != h:raise ValueError('Tested source changed: '+p)
    reference = run.read(REFERENCE / 'protocol.json')
    inherited,tasks = selected_inputs(reference)
    old_handoff = run.read(SEED_RUN / '644/handoff.json')
    if run.file_hash(Path(old_handoff['knowledge_manifest'])) != old_handoff['knowledge_manifest_sha256']:
        raise ValueError('Seed handoff manifest changed')
    seed_guards = run.read(Path(old_handoff['knowledge_manifest']))
    for p,h in seed_guards.items():
        if run.file_hash(Path(p)) != h:raise ValueError('Seed knowledge changed: '+p)
    batch.mkdir()
    shutil.copytree(ORIGINAL, batch / 'initial-graph')
    shutil.copytree(SEED, batch / 'seed-graph')
    shutil.copy2(SEED_RUN / 'experience.json', batch / 'initial-experience.json')
    graph = run.Graph.load(batch / 'seed-graph')
    index = run.read(graph.directory / 'index.json')
    TemplateKnowledge(graph,run.cards_path(graph.directory),run.EventLog(batch / 'registration-events'))
    if len(graph.nodes) != 225 or set(index['ids']) != set(graph.nodes) or not load_framework(graph):
        raise ValueError('Require complete225 graph, all-node index, cards and framework')
    if index['encoder'] != reference['encoder_expected']:raise ValueError('Encoder identity mismatch')
    encoder = batch / 'encoder-runtime/scripts'; encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py','jina_gpu.py','jina_vision_fp32.py'):
        shutil.copy2(Path(reference['encoder_runtime']) / 'scripts' / name, encoder / name)
    runtime = run.freeze(batch)
    excluded_run = ROOT / 'runs/reddit-hard10-225-flash-20261104-v3'
    excluded = dict(task_id=724,directory=str(excluded_run / '724'),
        state=run.read(excluded_run / 'state.json')['tasks']['724'],
        reason='此前用户授权跳过内容审核失败的724，当前继续保留排除与原失败成本')
    config = {**reference, 'version':'hard31-225-single-action-v1',
        'created_at':datetime.now(timezone.utc).isoformat(),
        'authorization':'2026-11-06 user requested remaining approximately31 Hard tasks using225-node method and confirmed combining with prior225-node ten, with source-version differences disclosed.',
        'runtime':str(runtime),'task_ids':[t['task_id'] for t in tasks],
        'source_graph':str(SEED),'initial_graph':str(batch / 'initial-graph'),
        'prepared_graph':str(batch / 'seed-graph'),'graph':str(batch / 'seed-graph'),
        'cards':str(run.cards_path(batch / 'seed-graph')),
        'initial_experience':str(batch / 'initial-experience.json'),'experience_store':str(batch / 'experience.json'),
        'stop_file':str(batch / 'STOP'),'encoder_runtime':str(encoder.parent),'max_graph_nodes':225,
        'actor_prompt':'single-action-v2','max_output_retries':3,'validate_verified_stop':True,'unified_verified_stop':True,
        'inherited_results':inherited,'excluded_attempts':[excluded],
        'report_title':'Reddit Hard41 · 225节点 · 历史10题＋新增31题',
        'budget_exhaustion_policy':'retain_score_keep_previous_handoff',
        'integrity_policy':'boundary-v1',
        'new_card_preparation':'Reuse complete644 handoff; no paid preflight or preparation, original225 independently preserved.',
        'knowledge_policy':'Continue from the last complete selected-ten handoff644. Prior735 test graphs and all60-node task graphs excluded. Successful handoffs accumulate inside this batch; ordinary call exhaustion keeps previous complete graph.',
        'restart_policy':'31 new attempts only, no retries of started tasks; ordinary call-budget exhaustion retains official score and incomplete graph separately, then moves to next unstarted task. Engineering/permanent-service/asset errors stop.',
        'limits':'User-selected multiversion developmental aggregate41, not one frozen-version trial. All selected rows use225-node cap and Flash; code versions and knowledge sources differ. 724 excluded with original evidence/costs retained. 735 was previously tested separately and is rerun here without inheriting its graph/results. No best-score selection, causal or generalization claim.'}
    for key in ('previous_version','comparison_version','failed_version'):config.pop(key,None)
    run.write_json(batch / 'protocol.json',config)
    run.write_json(batch / 'public-inputs.json',tasks)
    run.write_json(batch / 'result-lineage.json',dict(selected_old_ten=inherited,excluded=[excluded],
        seed=str(SEED),new_tasks=config['task_ids'],selection='Fixed by previous authorized lineage, not by highest score',
        not_inherited=['All60-node results and graphs','Separate735 test results and graphs']))
    run.write_json(batch / 'run_material_passport.json',dict(mode='run',status='registered',
        new_starts=31,inherited_rows=10,excluded_rows=1,models='All Qwen3.7-flash; Bailian direct',
        initial_nodes=225,paid_preflight_calls=0,preparation_calls=0,checks=str(checks_path),limits=config['limits']))
    inputs={**run.hashes(batch / 'initial-graph'),**run.hashes(batch / 'seed-graph'),**run.hashes(encoder.parent)}
    for name in ('protocol.json','public-inputs.json','result-lineage.json','run_material_passport.json','initial-experience.json'):
        inputs[str(batch / name)] = run.file_hash(batch / name)
    run.write_json(batch / 'runtime-inputs.json',inputs)
    protected={**run.hashes(ORIGINAL),**protect_graph(SEED),**seed_guards,**inputs}
    for row in inherited+[excluded]:
        folder=Path(row['directory']);parent=folder.parent
        for name in ('protocol.json','state.json','frozen-runtime/manifest.json'):
            p=parent / name;protected[str(p)]=run.file_hash(p)
        for name in ('result.json','official.json','events.jsonl'):
            p=folder / name;protected[str(p)]=run.file_hash(p)
        p=parent / 'trajectories' / f'task-{row["task_id"]}.html';protected[str(p)]=run.file_hash(p)
    for p in (checks_path,batch / 'runtime-inputs.json',ROOT / 'runs/active-graph.json',
              ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt'):
        protected[str(p.resolve())]=run.file_hash(p)
    run.write_json(batch / 'protected-inputs.json',protected)
    run.write_json(batch / 'state.json',dict(status='registered',tasks={},current_graph=str(batch / 'seed-graph')))
    run.verify(batch);run.report(batch)
    print(dict(registered=str(batch),new_tasks=config['task_ids'],inherited=10,excluded=724,protected_files=len(protected)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--checks',type=Path,required=True)
    a=p.parse_args();register(a.directory.resolve(),a.checks.resolve())
