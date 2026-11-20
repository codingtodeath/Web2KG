import hashlib
import json
from pathlib import Path

import pytest

from scripts.continue_agentoccam_flash_hard import pause_selection, retained_hashes
from scripts.run_agentoccam_flash_hard import report, fresh_state


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def paused(tmp_path):
    b=tmp_path/'previous'; b.mkdir(); (b/'STOP').write_text('USER PAUSE',encoding='utf-8')
    put(b/'protocol.json',{'task_ids':[1,2,3],'model':'qwen3.7-flash'})
    put(b/'state.json',{'status':'user_stopped','current_task':2,
        'tasks':{'1':{'status':'completed'},'2':{'status':'needs_attention'}}})
    put(b/'user-pause.json',{'queue_and_workers_exited':True,'completed_tasks':1,'interrupted_task':2,
        'stop_sha256':hashlib.sha256((b/'STOP').read_bytes()).hexdigest()})
    for t in (1,2):
        put(b/str(t)/'result.json',{'status':'agent_stopped' if t==1 else 'model_error',
            'error':None if t==1 else 'ResponsesError: User stop file present','steps':t,'seconds':10})
        put(b/str(t)/'official.json',{'score':0})
        (b/'trajectories').mkdir(exist_ok=True)
        (b/f'trajectories/task-{t}.html').write_text('trace',encoding='utf-8')
    put(b/'summary.json',{'tasks':[{'task_id':1},{'task_id':2,'physical_requests':4,'input_tokens':123,'output_tokens':20}]})
    return b


def test_resume_inherits_failure_and_retries_only_cancelled_task(paused):
    before=retained_hashes(paused); s=pause_selection(paused)
    assert s['run_task_ids']==[2,3]
    assert s['inherited_tasks']=={'1':str(paused/'1')}
    assert s['interrupted_attempts'][0]['input_tokens']==123
    assert retained_hashes(paused)==before


@pytest.mark.parametrize('change',['engineering','unresolved','stop_changed','unregistered'])
def test_resume_rejects_unsafe_sources(paused,change):
    if change=='engineering':
        put(paused/'user-pause.json',{})
    elif change=='unresolved':
        put(paused/'2/result.json',{'status':'model_error','error':'Unexpected parser failure'})
    elif change=='stop_changed':
        (paused/'STOP').write_text('changed',encoding='utf-8')
    else:
        (paused/'3').mkdir()
    with pytest.raises(ValueError): pause_selection(paused)


def test_report_keeps_inherited_source_and_separate_interruption(paused,tmp_path):
    b=tmp_path/'new'; s=pause_selection(paused)
    put(b/'protocol.json',{'task_ids':[1,2,3],'model':'qwen3.7-flash',**s})
    put(b/'state.json',{'status':'registered','tasks':{}})
    before=retained_hashes(paused); summary=report(b)
    assert summary['resolved']==1 and summary['official_successes']==0
    assert summary['tasks'][0]['inherited'] is True
    assert (b/summary['tasks'][0]['trace']).resolve()==paused/'trajectories/task-1.html'
    assert summary['tasks'][1]['official_score'] is None
    assert summary['totals']['input_tokens']==0
    assert summary['interrupted_attempts'][0]['input_tokens']==123
    assert not (b/'1').exists() and retained_hashes(paused)==before
    with pytest.raises(ValueError):fresh_state({'status':'user_stopped','tasks':{}})


def test_copy_registration_preserves_method_bytes(paused,tmp_path,monkeypatch):
    import scripts.continue_agentoccam_flash_hard as mod
    root=tmp_path/'project'; root.mkdir(); b=root/'runs/new'
    source=paused/'frozen-runtime'
    for name,value in {'web2kg/method.py':b'exact\r\nbytes',
                       'scripts/run_agentoccam_flash_hard.py':b'old scheduler'}.items():
        f=source/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(value)
    put(source/'manifest.json',{str(f.relative_to(source)).replace('\\','/'):mod.file_hash(f)
        for f in source.rglob('*.py')})
    scripts=root/'scripts';scripts.mkdir()
    for name in ('run_agentoccam_flash_hard.py','continue_agentoccam_flash_hard.py'):
        (scripts/name).write_text('new scheduler',encoding='utf-8')
    (paused/'upstream').mkdir();(paused/'upstream/native.py').write_bytes(b'native\r\n')
    (paused/'config.yml').write_text('unchanged',encoding='utf-8')
    put(paused/'public-inputs.json',[]);put(paused/'protected-inputs.json',{})
    put(paused/'run_material_passport.json',{})
    auth=root/'auth';auth.mkdir();(auth/'llm-crawl.env').write_text('fixture',encoding='utf-8')
    p=json.loads((paused/'protocol.json').read_text('utf-8'))
    p.update(runtime=str(source),credential_file=str(auth/'llm-crawl.env'))
    put(paused/'protocol.json',p)
    receipt=root/'checks.json';put(receipt,{'passed':True,'source_sha256':{}})
    monkeypatch.setattr(mod,'ROOT',root)
    monkeypatch.setattr(mod,'verify',lambda folder,**kwargs:mod.read(folder/'protocol.json'))
    monkeypatch.setattr(mod.subprocess,'run',lambda *args,**kwargs:None)
    before=retained_hashes(paused)
    mod.register_pause_continuation(b,paused,receipt)
    assert (b/'frozen-runtime/web2kg/method.py').read_bytes()==b'exact\r\nbytes'
    assert (b/'upstream/native.py').read_bytes()==b'native\r\n'
    assert retained_hashes(paused)==before
    assert mod.read(b/'protocol.json')['run_task_ids']==[2,3]
    assert mod.read(b/'state.json')=={'status':'registered','tasks':{}}
    with pytest.raises(ValueError):mod.register_pause_continuation(b,paused,receipt)


@pytest.mark.parametrize('first_fails',[False,True])
def test_queue_runs_only_remaining_tasks_and_keeps_failure(paused,tmp_path,monkeypatch,first_fails):
    import contextlib
    import scripts.run_agentoccam_flash_hard as runner
    b=tmp_path/'run';selection=pause_selection(paused)
    p=dict(task_ids=[1,2,3],model='qwen3.7-flash',runtime=str(b/'frozen-runtime'),
           shared_lock='fixture',stop_file=str(b/'STOP'),container='owned',image_id='image',
           backend='http://127.0.0.1:1',logical_origin='http://127.0.0.1:1',owner='fixture',**selection)
    put(b/'protocol.json',p);put(b/'state.json',{'status':'registered','tasks':{}})
    put(b/'protected-inputs.json',{});put(b/'frozen-runtime/manifest.json',{})
    monkeypatch.setattr(runner,'ROOT',b/'frozen-runtime')
    monkeypatch.setattr(runner,'verify',lambda *args,**kwargs:p)
    monkeypatch.setattr(runner,'queue_lock',lambda *args:contextlib.nullcontext())
    monkeypatch.setattr(runner,'reset_copy',lambda *args:'mock-container')
    started=[]
    class Process:
        pid=123
        def __init__(self,args,**kwargs):
            self.task=int(args[-1]);started.append(self.task)
        def wait(self):
            f=b/str(self.task);failed=first_fails and self.task==2
            put(f/'result.json',{'status':'model_error' if failed else 'agent_stopped',
                'error':'fixture error' if failed else None,'steps':1,'seconds':1})
            put(f/'official.json',{'score':0 if failed else 1})
            put(f/'closeout.json',{'status':'needs_attention' if failed else 'completed'})
            return int(failed)
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    before=retained_hashes(paused);runner.live(b)
    assert started==[2,3]
    state=runner.read(b/'state.json')
    assert state['status']=='completed' and set(state['tasks'])=={'2','3'}
    assert state['tasks']['2']['status']==('failed' if first_fails else 'completed')
    assert runner.read(b/'summary.json')['resolved']==3
    assert retained_hashes(paused)==before
