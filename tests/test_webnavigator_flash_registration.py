import pytest
from scripts.run_webnavigator_flash_hard import fresh_state, outcome, selected_tasks, ROOT, MODEL


def test_prompt_repeat_preserves_costs_and_rejects_unresolved_batch():
    from scripts.run_webnavigator_flash_hard import repeat_lineage
    state = dict(status='completed', tasks={str(n):{'status':'failed' if n==1 else 'completed'} for n in range(42)})
    summary = dict(resolved=42, totals={'physical_requests':430})
    lineage = repeat_lineage(state, {}, summary, 'previous')
    assert lineage['previous_usage'] == summary['totals']
    assert 'no inherited scores' in lineage['restart']
    state['tasks']['1']['status'] = 'started'
    with pytest.raises(ValueError, match='resolved'):
        repeat_lineage(state, {}, summary, 'previous')


def test_full_reddit_hard_no_previous_exclusions():
    tasks = selected_tasks(ROOT, 'http://127.0.0.1:10599', seed=20261104, limit=1000)
    assert len(tasks) == len({t['task_id'] for t in tasks}) == 42
    assert 724 in {t['task_id'] for t in tasks}
    assert all(set(t) == {'task_id', 'sites', 'intent', 'start_urls'} for t in tasks)
    assert all(t['sites'] == ['reddit'] for t in tasks)
    assert MODEL == 'qwen3.7-flash'


@pytest.mark.parametrize('state', [
    {'status':'running','tasks':{}}, {'status':'completed','tasks':{}},
    {'status':'registered','tasks':{'66':{'status':'started'}}}])
def test_started_batches_cannot_restart(state):
    with pytest.raises(ValueError): fresh_state(state)


def test_limits_and_failures_are_distinct():
    fresh_state({'status':'registered','tasks':{}})
    assert outcome({'status':'agent_stopped'}, []) == 'completed'
    assert outcome({'status':'step_limit'}, []) == 'completed'
    limit = {'status':'model_error','error':'ResponsesError: Responses call budget exhausted'}
    assert outcome(limit, []) == 'completed'
    assert outcome(limit, [{'event':'responses_failed','http_status':403}]) == 'needs_attention'
    assert outcome({'status':'model_error','error':'ResponsesError: Incomplete response'}, []) == 'needs_attention'
    assert outcome({'status':'runtime_error'}, []) == 'needs_attention'
    assert outcome({}, [{'event':'responses_failed','failure_kind':'cancelled'}]) == 'user_stopped'


def test_inherited_failure_is_terminal_but_not_success():
    inherited={'598':{'status':'failed','inherited_from':'old/598'}}
    fresh_state({'status':'registered','tasks':inherited}, inherited)
    with pytest.raises(ValueError):
        fresh_state({'status':'registered','tasks':{**inherited,'724':{'status':'started'}}},inherited)


@pytest.mark.parametrize('stop_after_error',[False, True])
def test_queue_continues_after_worker_error_but_honors_user_stop(tmp_path,monkeypatch,stop_after_error):
    import contextlib
    from scripts import run_webnavigator_flash_hard as run, build_qwen_graph
    p=dict(runtime=str(run.ROOT),inherited_tasks={'1':{'status':'failed'}}, task_ids=[1,2,3],
           shared_lock=str(tmp_path/'lock'),encoder_runtime=str(tmp_path),encoder_python='unused',
           gpu_memory='4GiB',encoder_expected={},stop_file=str(tmp_path/'STOP'),
           container='fixture',image_id='fixture',backend='fixture',logical_origin='fixture',
           owner='fixture',continue_task_errors=True)
    run.write_json(tmp_path/'state.json',dict(status='registered',tasks=p['inherited_tasks']))
    monkeypatch.setattr(run,'verify',lambda *a,**kw:p)
    monkeypatch.setattr(run,'queue_lock',lambda *a:contextlib.nullcontext())
    monkeypatch.setattr(build_qwen_graph,'local_encoder',lambda *a,**kw:contextlib.nullcontext())
    monkeypatch.setattr(run,'reset_copy',lambda *a:'container')
    monkeypatch.setattr(run,'report',lambda *a:None)
    original_read=run.read
    monkeypatch.setattr(run,'read',lambda path,*a: {} if str(path).endswith(('manifest.json','protected-inputs.json')) else original_read(path,*a))
    monkeypatch.setattr(run.subprocess,'CREATE_NO_WINDOW',0,raising=False)
    started=[]
    class Child:
        pid=123
        def __init__(self,args,**kwargs): self.task=int(args[-1]);started.append(self.task)
        def wait(self):
            folder=tmp_path/str(self.task)
            failed=self.task==2
            run.write_json(folder/'closeout.json',dict(status='needs_attention' if failed else 'completed',official_score=0 if failed else 1))
            run.write_json(folder/'official.json',dict(score=0 if failed else 1))
            run.write_json(folder/'result.json',dict(status='model_error' if failed else 'agent_stopped',error='Incomplete response' if failed else None))
            if failed and stop_after_error: (tmp_path/'STOP').write_text('user pause')
            return 1 if failed else 0
    monkeypatch.setattr(run.subprocess,'Popen',Child)
    run.live(tmp_path)
    state=original_read(tmp_path/'state.json')
    assert started == ([2] if stop_after_error else [2,3])
    assert state['status']==('user_stopped' if stop_after_error else 'completed')
    if not stop_after_error:
        assert state['tasks']['2']['status']=='failed'
        assert state['tasks']['3']['status']=='completed'
