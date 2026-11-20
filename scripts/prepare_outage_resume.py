"""One-time registration for the explicitly authorized outage continuation."""
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

from resume_hard_after_outage import read, write, digest, manifest

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    old = ROOT / 'runs/reddit-hard-method-qwen-v1'
    batch = ROOT / 'runs/reddit-hard-method-qwen-resume-v1'
    if batch.exists(): raise ValueError('Already prepared; never overwrite registration')
    p, s = read(old / 'protocol.json'), read(old / 'state.json')
    assert s['status'] == 'needs_attention' and s['tasks']['603']['status'] == 'needs_attention'
    inherited = {k:v for k,v in s['tasks'].items() if v['status']=='completed'}
    assert len(inherited)==16 and len(s['tasks'])==17
    remaining = [i for i in p['task_ids'] if str(i) not in s['tasks']]
    assert len(remaining)==25
    assert read(old / '603/result.json')['error']=='ResponsesError: User stop file present'
    guards = dict(p['protected_inputs'])
    for path, expected in read(old / 'registration-integrity.json').items():
        assert digest(path)==expected, path
        guards[path]=expected
    assert digest(s['knowledge_manifest'])==s['knowledge_manifest_sha256']
    for path, expected in read(s['knowledge_manifest']).items():
        assert digest(path)==expected, path
        guards[path]=expected
    # Protect every old task artifact, including interrupted603 and all costs.
    for tid in s['tasks']:
        for f in (old / tid).rglob('*'):
            if f.is_file(): guards[str(f)]=digest(f)
    for name in ('state.json','summary.json','STOP','experiences.json','routing-state.json','registration-integrity.json'):
        f=old/name
        if f.exists():guards[str(f)]=digest(f)
    terra=read(ROOT/'runs/reddit-hard-comparison-v2/ours/29/input.json')
    guards.update(terra['protected_inputs'])
    terra_runtime=ROOT/'runs/reddit-hard-qwen-v1/frozen-ours-runtime'
    # The repaired frozen runtime retains original Python/prompts plus the six missing static resources.
    prior=read(ROOT/'runs/reddit-hard-qwen-v1/protocol.json')
    for path,expected in prior['guards'].items():
        if str(terra_runtime) in path:guards[path]=expected
    for f in Path(terra['graph']).rglob('*'):
        if f.is_file():guards[str(f)]=digest(f)
    for name in ('resume_hard_after_outage.py','prepare_outage_resume.py'):
        f=ROOT/'scripts'/name;guards[str(f)]=digest(f)
    for path,expected in guards.items():
        if digest(path)!=expected:raise ValueError('Protected file changed: '+path)
    batch.mkdir()
    for arm in ('qwen','terra'):
        (batch/arm).mkdir()
        shutil.copy2(old/'public-inputs.json',batch/arm/'public-inputs.json')
        guards[str(batch/arm/'public-inputs.json')]=digest(batch/arm/'public-inputs.json')
    shutil.copy2(old/'experiences.json',batch/'qwen/experiences.json')
    shutil.copy2(old/'experiences.json',batch/'qwen/initial-experiences.json')
    shutil.copy2(old/'routing-state.json',batch/'qwen/routing-state.json')
    # No learned experience existed at the original29 boundary (the source store is absent).
    assert not Path(terra['experience_store']).exists()
    runtime=Path(p['frozen_runtime']);sys.path[:0]=[str(runtime),str(runtime/'scripts')]
    from web2kg.common.storage import Graph
    from web2kg.stage2.recovery import memory_namespace
    write(batch/'terra/experiences.json',dict(version=1,namespace=memory_namespace(Graph.load(Path(terra['graph']))),pages=[],templates={}))
    shutil.copy2(batch/'terra/experiences.json',batch/'terra/initial-experiences.json')
    schedule=[dict(arm='terra',task_id=29),dict(arm='qwen',task_id=603)]+[dict(arm='qwen',task_id=i) for i in remaining]
    for arm,cfg,ids in [('qwen',p,[603,*remaining]),('terra',terra,[29])]:
        config={**cfg,'task_ids':ids,'max_new_task_starts':len(ids),'stop_file':str(batch/'STOP'),
            'experience_store':str(batch/arm/'experiences.json'),
            'authorization':'2026-10-16 user explicitly requests continuation, affected task rerun, and original Terra29 supplement.'}
        if arm=='qwen':config['graph']=s['current_graph']
        write(batch/arm/'protocol.json',config)
    for arm in ('qwen','terra'):
        f=batch/arm/'initial-experiences.json';guards[str(f)]=digest(f)
    target=Path(p['report_path']);shutil.copy2(target,batch/'comparison-report-before.html')
    audit=dict(time=datetime.now(timezone.utc).isoformat(),
        authorization='User: 现在已经恢复了，请继续实验并重跑之前受影响的实验。...Terra...29...补上',
        affected_task=603,reason='27 Luna intermediary HTTP503 failures, then user STOP; no committed handoff.',
        excluded_output_limit_failures=[28,730],output_limit_evidence='Bailian chat-response finish_reason=length, successful HTTP response; not intermediary outage.',
        recovered_transport_tasks=[734,66,399],normal_failures_not_retried=[31,68,715],
        old_603=str(old/'603'),old_29=str(ROOT/'runs/reddit-hard-comparison-v2/ours/29'),
        inherited_tasks=list(inherited),qwen_graph=s['current_graph'],terra_graph=terra['graph'],
        terra_source=terra['source_sha256'],qwen_source=p['source_sha256'],
        old_results_states_stop_files_unchanged=True,selection='Latest authorized attempt, never max-score selection',
        new_starts=27,initial_knowledge_from_interrupted603=False,terra_knowledge_isolated=True)
    write(batch/'impact-audit.json',audit)
    protocol=dict(workspace=str(ROOT),prior_qwen=str(old),task_ids=p['task_ids'],schedule=schedule,
        max_new_starts=27,inherited=inherited,guards=guards,
        runtimes={'qwen':p['frozen_runtime'],'terra':str(terra_runtime)},
        stop_file=str(batch/'STOP'),shared_lock=p['shared_lock'],report_path=str(target),
        exposure=p['exposure'],authorization=audit['authorization'])
    write(batch/'protocol.json',protocol)
    knowledge={arm:manifest(read(batch/arm/'protocol.json')['graph'],batch/arm/'experiences.json',batch/f'{arm}-initial-knowledge.json') for arm in ('qwen','terra')}
    write(batch/'state.json',dict(status='registered',tasks={},knowledge=knowledge))
    write(batch/'run_material_passport.json',dict(study_id=batch.name,status='registered',mode='run',max_starts=27,
        sources={'qwen':p['source_sha256'],'terra':terra['source_sha256']},protected_files=len(guards),
        schedule=schedule,prior_exposure=p['exposure'],authorization=audit['authorization'],
        preflight='Inherited real Qwen596/full-run and original Terra full-run; no extra API preflight',
        hard_timeout=None,monitor_interval_minutes=30))
    registration=[batch/name for name in ('protocol.json','impact-audit.json','comparison-report-before.html','run_material_passport.json')]
    registration += [batch/arm/'protocol.json' for arm in ('qwen','terra')]
    write(batch/'registration-integrity.json',{str(f):digest(f) for f in registration})
    print(json.dumps({'new_starts':27,'inherited':16,'guards':len(guards),'batch':str(batch)}))


if __name__=='__main__':prepare()
