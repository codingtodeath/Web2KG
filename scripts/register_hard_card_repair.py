"""Register the authorized624 card-validation repair and seven unstarted tasks."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts import run_reddit_hard10_225 as run
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.stage1.site_framework import load_framework


def continuation_inputs(previous, config, state):
    started=list(state['tasks']);ids=config['task_ids']
    if state['status']!='needs_attention' or started != [str(t) for t in ids[:15]] or started[-1]!='624':
        raise ValueError('Require the stopped624 prefix')
    if any(state['tasks'][key]['status']!='completed' for key in started[:-1]):
        raise ValueError('Prior entries need complete handoffs')
    if state['tasks']['624']['status']!='needs_attention':raise ValueError('Preserve the original failed624')
    diagnosis=run.read(previous/'failure-diagnosis.json')
    if (diagnosis.get('status')!='confirmed_schema_and_business_validation_mismatch'
            or diagnosis.get('task_id')!=624 or not diagnosis.get('schema_passed_offline')
            or not diagnosis.get('post_validator_failed_offline')):
        raise ValueError('Unexpected failure; this authorization repairs card validation')
    seed=Path(state['current_graph'])
    if seed != previous/'28/execution-graph':raise ValueError('Never inherit the failed624 graph')
    if Path(run.read(previous/'624/input.json')['graph']) != seed:
        raise ValueError('Retry must use the original624 input')
    inherited=list(config['inherited_results'])
    for key in started[:-1]:
        if run.read(previous/key/'official.json').get('score') not in (0,1):raise ValueError('Missing prior score')
        inherited.append(dict(task_id=int(key),directory=str(previous/key),state=state['tasks'][key]))
    remaining=ids[14:]
    if len(inherited)!=33 or len(remaining)!=8 or set(remaining)&{r['task_id'] for r in inherited}:
        raise ValueError('Require33 inherited and8 new attempts')
    return seed,inherited,remaining


def register(batch,previous,checks_path):
    if batch.exists() or not batch.is_relative_to(ROOT/'runs'):raise ValueError('Use a fresh workspace directory')
    checks=run.read(checks_path)
    if not checks.get('passed'):raise ValueError('Offline checks required')
    for rel,expected in checks['source_sha256'].items():
        if run.file_hash(ROOT/rel)!=expected:raise ValueError('Tested source changed: '+rel)
    config,state=run.verify(previous,record=False)
    seed,inherited,ids=continuation_inputs(previous,config,state)
    batch.mkdir()
    shutil.copytree(config['initial_graph'],batch/'initial-graph')
    shutil.copytree(seed,batch/'seed-graph')
    shutil.copy2(config['experience_store'],batch/'initial-experience.json')
    graph=run.Graph.load(batch/'seed-graph');index=run.read(graph.directory/'index.json')
    TemplateKnowledge(graph,run.cards_path(graph.directory),run.EventLog(batch/'registration-events'))
    if len(graph.nodes)!=225 or set(index['ids'])!=set(graph.nodes) or not load_framework(graph):
        raise ValueError('Require complete225 handoff')
    if index['encoder']!=config['encoder_expected']:raise ValueError('Encoder identity mismatch')
    encoder=batch/'encoder-runtime/scripts';encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py','jina_gpu.py','jina_vision_fp32.py'):
        shutil.copy2(Path(config['encoder_runtime'])/'scripts'/name,encoder/name)
    runtime=run.freeze(batch)
    excluded=list(config.get('excluded_attempts',[]))+[dict(task_id=624,directory=str(previous/'624'),
        state=state['tasks']['624'],reason='用户授权修复制卡校验后重跑624；原官方1分及收尾失败全部保留、成本单列，不按最高分选择')]
    protocol={**config,'version':'hard41-225-card-validation-v3','created_at':datetime.now(timezone.utc).isoformat(),
        'authorization':'2026-11-06 user explicitly authorized aligning card validation and bounded feedback, retry624 from its original complete input, then continue seven unstarted tasks.',
        'runtime':str(runtime),'task_ids':ids,'previous_version':str(previous),
        'source_graph':str(seed),'initial_graph':str(batch/'initial-graph'),'prepared_graph':str(batch/'seed-graph'),
        'graph':str(batch/'seed-graph'),'cards':str(run.cards_path(batch/'seed-graph')),
        'initial_experience':str(batch/'initial-experience.json'),'experience_store':str(batch/'experience.json'),
        'stop_file':str(batch/'STOP'),'encoder_runtime':str(encoder.parent),
        'output_retry_feedback':'previous-response-v1','card_output_validation':'validated-card-v1','inherited_results':inherited,'excluded_attempts':excluded,
        'report_title':'Reddit Hard41 · 225节点 · 制卡校验修复版',
        'knowledge_policy':'Restart624 from its original complete28 input, never inherit failed624. Subsequent tasks inherit only complete handoffs.',
        'restart_policy':'One authorized retry624 followed by7 previously unstarted tasks. Engineering/permanent/asset errors stop. Existing budgets and ordinary-exhaustion policy unchanged.',
        'new_card_preparation':'Reuse complete28 knowledge; zero additional paid preflight or preparation.',
        'limits':config['limits']+' Current selected624 is the authorized post-repair attempt regardless of score; prior624 official1 and maintenance failure/costs separately retained. Cards use aligned schema/domain validation and feedback within the same three-regeneration budget.'}
    tasks=[t for t in run.read(previous/'public-inputs.json') if t['task_id'] in ids]
    run.write_json(batch/'protocol.json',protocol);run.write_json(batch/'public-inputs.json',tasks)
    run.write_json(batch/'repair-lineage.json',dict(previous=str(previous),seed=str(seed),retry=624,unstarted=ids[1:],
        code_change='Align card schema with domain constraints; bounded validation feedback includes the rejected card and exact error, with no guessed evidence or budget change.',
        selection='Fixed authorized replacement regardless of outcome; retain previous failed attempt separately',checks=str(checks_path)))
    run.write_json(batch/'run_material_passport.json',dict(mode='run',status='registered',new_starts=8,inherited_rows=33,
        graph_nodes=225,models='All Qwen3.7-flash; Bailian direct',paid_preflight_calls=0,preparation_calls=0,limits=protocol['limits']))
    inputs={**run.hashes(batch/'initial-graph'),**run.hashes(batch/'seed-graph'),**run.hashes(encoder.parent)}
    for name in ('protocol.json','public-inputs.json','repair-lineage.json','run_material_passport.json','initial-experience.json'):
        inputs[str(batch/name)]=run.file_hash(batch/name)
    run.write_json(batch/'runtime-inputs.json',inputs)
    protected={**run.read(previous/'protected-inputs.json'),**run.hashes(seed),**inputs,
               **run.read(Path(state['knowledge_manifest'])),**run.hashes(previous/'624')}
    for rel,expected in run.read(previous/'frozen-runtime/manifest.json').items():
        protected[str(previous/'frozen-runtime'/rel)]=expected
    for name in ('state.json','summary.json','failure-diagnosis.json','combined-audit.json'):
        protected[str(previous/name)]=run.file_hash(previous/name)
    for key in state['tasks']:
        for name in ('result.json','official.json','events.jsonl'):
            protected[str(previous/key/name)]=run.file_hash(previous/key/name)
        p=previous/'trajectories'/f'task-{key}.html';protected[str(p)]=run.file_hash(p)
    for p in (checks_path,batch/'runtime-inputs.json',Path(state['knowledge_manifest'])):
        protected[str(p)]=run.file_hash(p)
    run.write_json(batch/'protected-inputs.json',protected)
    run.write_json(batch/'state.json',dict(status='registered',tasks={},current_graph=str(batch/'seed-graph')))
    run.verify(batch);run.report(batch)
    print(dict(registered=str(batch),new_tasks=ids,inherited=33,protected_files=len(protected)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--checks',type=Path,required=True);a=p.parse_args()
    register(a.directory.resolve(),a.previous.resolve(),a.checks.resolve())
