"""Register the authorized66 retry and21 unstarted tasks with previous-output feedback."""
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
    if state['status']!='needs_attention' or started != [str(t) for t in ids[:10]] or started[-1]!='66':
        raise ValueError('Require the stopped66 prefix')
    if any(state['tasks'][key]['status']!='completed' for key in started[:-1]):
        raise ValueError('Prior entries need complete handoffs')
    if state['tasks']['66']['status']!='needs_attention':raise ValueError('Preserve the original failed66')
    result=run.read(previous/'66/result.json')
    if result.get('error')!='ResponsesError: Expected exactly one ACTION section; multiple sections were returned':
        raise ValueError('Unexpected failure; this authorization repairs response feedback')
    seed=Path(state['current_graph'])
    if seed != previous/'407/execution-graph':raise ValueError('Never inherit the failed66 graph')
    inherited=list(config['inherited_results'])
    for key in started[:-1]:
        if run.read(previous/key/'official.json').get('score') not in (0,1):raise ValueError('Missing prior score')
        inherited.append(dict(task_id=int(key),directory=str(previous/key),state=state['tasks'][key]))
    remaining=ids[9:]
    if len(inherited)!=19 or len(remaining)!=22 or set(remaining)&{r['task_id'] for r in inherited}:
        raise ValueError('Require19 inherited and22 new attempts')
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
    excluded=list(config.get('excluded_attempts',[]))+[dict(task_id=66,directory=str(previous/'66'),
        state=state['tasks']['66'],reason='用户授权修复纠错反馈后重跑66；此为修复前失败，原0分与全部成本单列，不按最高分选择')]
    protocol={**config,'version':'hard31-225-rejected-response-feedback-v2','created_at':datetime.now(timezone.utc).isoformat(),
        'authorization':'2026-11-06 user explicitly requested returning the last malformed response to Agent, retry66, then continue remaining tasks if no engineering error.',
        'runtime':str(runtime),'task_ids':ids,'previous_version':str(previous),
        'source_graph':str(seed),'initial_graph':str(batch/'initial-graph'),'prepared_graph':str(batch/'seed-graph'),
        'graph':str(batch/'seed-graph'),'cards':str(run.cards_path(batch/'seed-graph')),
        'initial_experience':str(batch/'initial-experience.json'),'experience_store':str(batch/'experience.json'),
        'stop_file':str(batch/'STOP'),'encoder_runtime':str(encoder.parent),
        'output_retry_feedback':'previous-response-v1','inherited_results':inherited,'excluded_attempts':excluded,
        'report_title':'Reddit Hard41 · 225节点 · 纠错反馈修复版',
        'knowledge_policy':'Restart66 from its original complete407 input, never inherit failed66. Subsequent tasks inherit only complete handoffs.',
        'restart_policy':'One authorized retry66 followed by21 previously unstarted tasks. Engineering/permanent/asset errors stop. Existing budgets and ordinary-exhaustion policy unchanged.',
        'new_card_preparation':'Reuse complete407 knowledge; zero additional paid preflight or preparation.',
        'limits':config['limits']+' Current selected66 is the authorized post-repair attempt regardless of score; old66 failure/costs separately retained. Latest rejected visible answer is fed back, observation unchanged; strict single action validation retained.'}
    tasks=[t for t in run.read(previous/'public-inputs.json') if t['task_id'] in ids]
    run.write_json(batch/'protocol.json',protocol);run.write_json(batch/'public-inputs.json',tasks)
    run.write_json(batch/'repair-lineage.json',dict(previous=str(previous),seed=str(seed),retry=66,unstarted=ids[1:],
        code_change='Latest rejected visible assistant response plus explicit user feedback; no action deduplication or budget change.',
        selection='Fixed authorized replacement regardless of outcome; retain previous failed attempt separately',checks=str(checks_path)))
    run.write_json(batch/'run_material_passport.json',dict(mode='run',status='registered',new_starts=22,inherited_rows=19,
        graph_nodes=225,models='All Qwen3.7-flash; Bailian direct',paid_preflight_calls=0,preparation_calls=0,limits=protocol['limits']))
    inputs={**run.hashes(batch/'initial-graph'),**run.hashes(batch/'seed-graph'),**run.hashes(encoder.parent)}
    for name in ('protocol.json','public-inputs.json','repair-lineage.json','run_material_passport.json','initial-experience.json'):
        inputs[str(batch/name)]=run.file_hash(batch/name)
    run.write_json(batch/'runtime-inputs.json',inputs)
    protected={**run.read(previous/'protected-inputs.json'),**run.hashes(seed),**inputs,
               **run.read(Path(state['knowledge_manifest'])),**run.hashes(previous/'66')}
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
    print(dict(registered=str(batch),new_tasks=ids,inherited=19,protected_files=len(protected)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--checks',type=Path,required=True);a=p.parse_args()
    register(a.directory.resolve(),a.previous.resolve(),a.checks.resolve())
