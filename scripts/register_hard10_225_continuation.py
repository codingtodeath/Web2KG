"""Register the explicitly authorized 598 retry and eight unstarted tasks; never resume old workers."""
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


def continuation_inputs(previous):
    config, state = run.verify(previous)
    if state['status'] != 'needs_attention' or config['task_ids'] != [646,598,724,625,596,721,31,730,409,651]:
        raise ValueError('Expected the stopped, registered Hard10 version')
    if set(state['tasks']) != {'646','598'} or state['tasks']['646']['status'] != 'completed':
        raise ValueError('Unexpected task starts or incomplete inherited task')
    if run.read(previous/'646/official.json')['score'] != 1:
        raise ValueError('Inherited task must match the user-authorized completed result')
    failed = run.read(previous/'598/result.json')
    if failed.get('provider_failure',{}).get('kind') != 'output:verified_stop_invalid':
        raise ValueError('This authorization concerns the diagnosed stop-format failure only')
    graph = Path(state['current_graph'])
    if graph != previous/'646/execution-graph':
        raise ValueError('Do not inherit the failed 598 task graph')
    return config, state, graph


def register(batch, previous):
    if batch.exists() or not batch.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a fresh workspace run directory')
    config, old_state, seed = continuation_inputs(previous)
    changed = {'web2kg/evaluation/webnavigator.py','web2kg/evaluation/verified_answer.py',
               'scripts/run_reddit_hard10_225.py'}
    manifest = run.read(Path(config['runtime'])/'manifest.json')
    for rel, expected in manifest.items():
        if rel not in changed and run.file_hash(ROOT/rel) != expected:
            raise ValueError('Unexpected method change: '+rel)
    if len(run.Graph.load(seed).nodes) != 225:
        raise ValueError('Expected the complete 225-node handoff')
    batch.mkdir()
    shutil.copytree(config['initial_graph'], batch/'initial-graph')
    shutil.copytree(seed,batch/'seed-graph')
    shutil.copy2(config['experience_store'],batch/'initial-experience.json')
    graph = run.Graph.load(batch/'seed-graph')
    TemplateKnowledge(graph,run.cards_path(graph.directory),run.EventLog(batch/'registration-events'))
    if not load_framework(graph):
        raise ValueError('Complete framework required')
    index=run.read(graph.directory/'index.json')
    if set(index['ids']) != set(graph.nodes):
        raise ValueError('All seed nodes must be indexed')
    encoder=batch/'encoder-runtime/scripts';encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py','jina_gpu.py','jina_vision_fp32.py'):
        shutil.copy2(Path(config['encoder_runtime'])/'scripts'/name,encoder/name)
    runtime=run.freeze(batch)
    protocol={**config,'version':'hard10-225-unified-stop-continuation-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'authorization':'User explicitly requested fixing stop instructions, retrying598 and running the remaining eight tasks; preserve646.',
        'runtime':str(runtime),'task_ids':config['task_ids'][1:],'source_graph':str(seed),'previous_version':str(previous),
        'initial_graph':str(batch/'initial-graph'),'prepared_graph':str(batch/'seed-graph'),
        'graph':str(batch/'seed-graph'),'cards':str(run.cards_path(batch/'seed-graph')),
        'experience_store':str(batch/'experience.json'),'initial_experience':str(batch/'initial-experience.json'),
        'stop_file':str(batch/'STOP'),'encoder_runtime':str(encoder.parent),
        'validate_verified_stop':True,'unified_verified_stop':True,
        'inherited_results':[dict(task_id=646,directory=str(previous/'646'),state=old_state['tasks']['646'])],
        'new_card_preparation':'Reuse verified646 handoff cards/framework; no extra preparation model calls.',
        'restart_policy':'Exactly nine new starts,598 then eight unstarted tasks; never restart any new unresolved entry.',
        'knowledge_policy':'Independent copy of completed646 graph/experience; never use failed598 assets. Original225 graph retained separately.',
        'limits':'User-authorized multi-version development view: retained646 plus nine new attempts, not a single-version blind evaluation. No best-score selection.'}
    run.write_json(batch/'protocol.json',protocol)
    public=run.read(previous/'public-inputs.json')
    run.write_json(batch/'public-inputs.json',[x for x in public if x['task_id'] in protocol['task_ids']])
    run.write_json(batch/'continuation.json',{'previous':str(previous),'retry':598,'new_tasks':protocol['task_ids'],
        'inherited_completed':646,'input_graph':str(seed),'original_graph':config['initial_graph'],
        'code_changes':{p:{'before':manifest[p],'after':run.file_hash(ROOT/p)} for p in sorted(changed)},
        'past_attempts_and_costs':'All v1/v2 records retained separately; no old score/state/STOP is edited.'})
    run.write_json(batch/'run_material_passport.json',{'mode':'run','new_starts':9,'inherited_result':646,
        'model_roles':'All Qwen3.7-flash, Bailian direct','paid_preflight_calls':0,'preparation_calls':0,
        'source_graph':str(seed),'original225_preserved':True,'limitations':protocol['limits']})
    protected={**run.read(previous/'protected-inputs.json'),**run.hashes(seed),
        **run.hashes(batch/'initial-graph'),**run.hashes(batch/'seed-graph'),**run.hashes(encoder.parent)}
    for parent,names in [(batch,['protocol.json','public-inputs.json','continuation.json','run_material_passport.json','initial-experience.json']),
                         (previous,['state.json','summary.json','final-audit.json','646/result.json','646/official.json','646/handoff.json','598/result.json','598/official.json'])]:
        for name in names:protected[str(parent/name)]=run.file_hash(parent/name)
    run.write_json(batch/'protected-inputs.json',protected)
    run.write_json(batch/'state.json',{'status':'registered','tasks':{},'current_graph':str(batch/'seed-graph')})
    run.report(batch)
    print({'registered':str(batch),'new_tasks':protocol['task_ids'],'inherited':646,'protected_files':len(protected)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();register(a.directory.resolve(),a.previous.resolve())
