import copy
import json
from pathlib import Path

import pytest
from scripts import run_reddit_hard10_225 as run


def test_seeded_ten_hard_reddit_public_fields_only():
    rows = run.selected_tasks(run.ROOT, 'http://localhost:10599')
    assert rows == run.selected_tasks(run.ROOT, 'http://localhost:10599')
    assert len(rows) == len({r['task_id'] for r in rows}) == 10
    hard = set(run.read(run.ROOT / 'experiments/webarena-verified/assets/dataset/subsets/webarena-verified-hard.json')['task_ids'])
    assert all(r['task_id'] in hard and r['sites'] == ['reddit'] and
               set(r) == {'task_id','sites','intent','start_urls'} for r in rows)
    assert all(u.startswith('http://localhost:10599') for r in rows for u in r['start_urls'])
    full = run.selected_tasks(run.ROOT, 'http://localhost:10599', limit=None)
    assert len(full) == 42 and full[:10] == rows and full[10]['task_id'] == 644


@pytest.mark.parametrize('field,value', [('description_stopped',True),('index_status','failed'),
    ('card_refresh',{'pending':['T1']}),('template_global',{'status':'failed'}),
    ('framework_refresh',{'status':'refresh_failed'})])
def test_engineering_gate_preserves_ordinary_failures_but_stops_bad_handoffs(field, value):
    result = {'status':'agent_stopped','execution_graph':{'index_status':'completed',
              'card_refresh':{'deferred':['T1']}}}
    assert not run.stop_for_error(result, [])
    damaged = copy.deepcopy(result);damaged['execution_graph'][field] = value
    assert run.stop_for_error(damaged, [])
    assert run.stop_for_error({**result,'status':'runtime_error'}, [])


def test_integrity_guards_original_and_handoff_without_rewriting(tmp_path):
    runtime = tmp_path / 'source';runtime.mkdir()
    source = runtime / 'runner.py';source.write_text('source',encoding='utf-8')
    initial = tmp_path / 'initial.json';initial.write_text('{}',encoding='utf-8')
    knowledge = tmp_path / 'knowledge.json';knowledge.write_text('{}',encoding='utf-8')
    handoff = tmp_path / 'handoff.json'
    run.write_json(runtime / 'manifest.json', {'runner.py':run.file_hash(source)})
    run.write_json(handoff,{str(knowledge):run.file_hash(knowledge)})
    run.write_json(tmp_path / 'protocol.json',{'runtime':str(runtime)})
    run.write_json(tmp_path / 'protected-inputs.json',{str(initial):run.file_hash(initial)})
    run.write_json(tmp_path / 'state.json',{'knowledge_manifest':str(handoff),
        'knowledge_manifest_sha256':run.file_hash(handoff),'tasks':{}})
    assert run.verify(tmp_path)
    knowledge.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='Knowledge handoff changed'): run.verify(tmp_path)
    knowledge.write_text('{}',encoding='utf-8')
    initial.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='Protected input changed'): run.verify(tmp_path)


def test_registered_model_and_output_retry_policy(monkeypatch,tmp_path):
    captured = {}
    class Fake:
        def __init__(self,*args,**kwargs): pass
        def configure_output_retries(self,n): self.output_retries=n
    def factory(*args,**kwargs):
        captured.update(kwargs);return Fake
    monkeypatch.setattr(run,'clients',factory)
    config = {'credential_file':'private-path','request_read_timeout':300,'max_output_retries':3,
              'validate_verified_stop':True,'unified_verified_stop':True}
    client = run.model_class(tmp_path,config,'qwen3.7-flash')()
    assert captured == {'provider':'bailian','model_override':'qwen3.7-flash'}
    assert client.request_timeout == (10,300) and client.output_retries == 3
    assert client.validate_verified_stop and client.unified_verified_stop


def test_continuation_only_uses_completed_handoff(monkeypatch,tmp_path):
    from scripts import register_hard10_225_continuation as register
    previous=tmp_path.resolve()
    config={'task_ids':[646,598,724,625,596,721,31,730,409,651]}
    state={'status':'needs_attention','tasks':{'646':{'status':'completed'},'598':{'status':'needs_attention'}},
           'current_graph':str(previous/'646/execution-graph')}
    monkeypatch.setattr(run,'verify',lambda p:(config,state))
    run.write_json(previous/'646/official.json',{'score':1})
    run.write_json(previous/'598/result.json',{'provider_failure':{'kind':'output:verified_stop_invalid'}})
    assert register.continuation_inputs(previous)[2]==previous/'646/execution-graph'
    state['current_graph']=str(previous/'598/execution-graph')
    with pytest.raises(ValueError,match='failed 598'):register.continuation_inputs(previous)
    state['current_graph']=str(previous/'646/execution-graph')
    state['tasks']['724']={'status':'started'}
    with pytest.raises(ValueError,match='Unexpected task starts'):register.continuation_inputs(previous)


def test_continuation_report_keeps_one_prior_result_separate(monkeypatch,tmp_path):
    old=tmp_path/'prior/646'
    run.write_json(tmp_path/'protocol.json',{'task_ids':[598,724,625,596,721,31,730,409,651],
        'inherited_results':[{'task_id':646,'directory':str(old),'state':{'status':'completed'}}],
        'model':'qwen3.7-flash','limits':'development','initial_graph':'initial'})
    run.write_json(tmp_path/'state.json',{'status':'registered','tasks':{},'current_graph':'seed'})
    def stats(folder,task,state):
        return {'task_id':task,'status':state.get('status','not_started'),'official_score':1 if task==646 else None,
                'steps':8 if task==646 else None,'total_seconds':None,'physical_requests':0,'unknown_usage':0}
    monkeypatch.setattr(run,'task_stats',stats)
    summary=run.report(tmp_path)
    assert summary['planned']==10 and summary['new_planned']==9
    assert summary['completed']==summary['official_successes']==summary['official_scored']==1
    assert sum(row['inherited'] for row in summary['tasks'])==1
    assert summary['tasks'][0]['task_id']==646


@pytest.mark.parametrize('worker_code,budget_first', [(0,False), (1,False), (0,True)])
def test_queue_preserves_initial_and_never_restarts_started_attempt(monkeypatch, tmp_path, worker_code,budget_first):
    from contextlib import nullcontext
    from scripts import build_qwen_graph
    initial = tmp_path / 'initial-graph'
    graph = run.Graph(initial)
    graph.data.update(root='r', nodes={'r': {'url': 'http://localhost:10599/'}})
    graph.save()
    original = (initial / 'graph.json').read_bytes()
    config = dict(runtime=str(run.ROOT), initial_graph=str(initial), encoder_runtime=str(tmp_path),
        encoder_python='unused', gpu_memory='4GiB', encoder_expected={}, auxiliary_model='fixture',
        stop_file=str(tmp_path/'STOP'), experience_store=str(tmp_path/'experience.json'),
        task_ids=[1,2], container='owned', image_id='fixture', backend='local', logical_origin='local', owner='owned')
    if budget_first: config['budget_exhaustion_policy']='retain_score_keep_previous_handoff'
    run.write_json(tmp_path/'state.json', {'status':'registered','tasks':{},'current_graph':str(initial)})
    monkeypatch.setattr(run,'verify',lambda batch, **kwargs: (config,run.read(batch/'state.json')))
    monkeypatch.setattr(run,'queue_lock',lambda path: nullcontext())
    monkeypatch.setattr(build_qwen_graph,'local_encoder',lambda *a,**k: nullcontext())
    monkeypatch.setattr(run,'model_class',lambda *a: lambda *a,**k: object())
    monkeypatch.setattr(run,'load_layer',lambda graph: (initial/'layer.json',{}))
    monkeypatch.setattr(run,'build_cards',lambda *a: {'cards':{str(i):{'status':'completed'} for i in range(8)}})
    monkeypatch.setattr(run,'build_framework',lambda *a,**k: None)
    monkeypatch.setattr(run,'cards_path',lambda graph: graph/'cards.json')
    monkeypatch.setattr(run,'report',lambda *a: None)
    resets=[]
    monkeypatch.setattr(run,'reset_copy',lambda *a: resets.append(a))
    class Process:
        pid=123
        def __init__(self,args,**kwargs):
            task=args[-1]
            state=run.read(tmp_path/'state.json')
            assert state['tasks'][task]['status']=='started'
            if budget_first and task=='1':
                run.write_json(tmp_path/task/'outcome.json',dict(status='budget_exhausted',graph_advanced=False))
                run.write_json(tmp_path/task/'handoff.json',{'current_graph':'incomplete-should-not-inherit'})
            else:
                assert state['current_graph']==str(tmp_path/'starting-graph')
                run.write_json(tmp_path/task/'handoff.json',{'current_graph':state['current_graph']})
        def wait(self): return worker_code
    monkeypatch.setattr(run.subprocess,'Popen',Process)
    if worker_code:
        with pytest.raises(RuntimeError,match='no task restart'): run.queue(tmp_path)
    else:
        run.queue(tmp_path)
    state=run.read(tmp_path/'state.json')
    assert len(resets)==(1 if worker_code else 2)
    assert state['status']==('needs_attention' if worker_code else 'completed')
    if budget_first:
        assert state['tasks']['1']['status']=='budget_exhausted'
        assert state['tasks']['2']['status']=='completed'
    assert (initial/'graph.json').read_bytes()==original
    with pytest.raises(ValueError,match='Never restart'): run.queue(tmp_path)


def test_authorized_substitution_uses_next_seeded_task_and_completed_graph(monkeypatch,tmp_path):
    from scripts import register_hard10_225_substitution as register
    previous=tmp_path.resolve();prior=previous/'prior646'
    config={'logical_origin':'http://localhost:10599','seed':20261104,
        'task_ids':[598,724,625,596,721,31,730,409,651],
        'inherited_results':[{'task_id':646,'directory':str(prior),'state':{'status':'completed'}}]}
    state={'status':'needs_attention','tasks':{'598':{'status':'completed'},'724':{'status':'needs_attention'}},
        'current_graph':str(previous/'598/execution-graph')}
    monkeypatch.setattr(run,'verify',lambda p:(config,state))
    run.write_json(prior/'official.json',{'score':1})
    run.write_json(previous/'598/official.json',{'score':1})
    run.write_json(previous/'724/result.json',{'provider_failure':{'kind':'http','http_status':400}})
    receipt=previous/'724/llm/failed/http_error.json'
    run.write_json(receipt,{'http_status':400,'code':'data_inspection_failed'})
    result=register.substitution_inputs(previous)
    assert result[2]==previous/'598/execution-graph'
    assert [t['task_id'] for t in result[3]]==[625,596,721,31,730,409,651,644]
    run.write_json(previous/'598/official.json',{'score':0})
    assert register.substitution_inputs(previous)[3]==result[3]  # not score-based selection
    state['current_graph']=str(previous/'724/execution-graph')
    with pytest.raises(ValueError,match='failed724'):register.substitution_inputs(previous)
    state['current_graph']=str(previous/'598/execution-graph')
    state['tasks']['625']={'status':'started'}
    with pytest.raises(ValueError,match='Unexpected starts'):register.substitution_inputs(previous)
    del state['tasks']['625']
    run.write_json(receipt,{'http_status':400,'code':'invalid_parameter'})
    with pytest.raises(ValueError,match='content-inspection'):register.substitution_inputs(previous)
    run.write_json(receipt,{'http_status':400,'code':'data_inspection_failed'})
    config['seed']=20261014
    with pytest.raises(ValueError,match='seeded order'):register.substitution_inputs(previous)


def test_substitution_report_keeps_two_results_in_order_and_exclusion_separate(monkeypatch,tmp_path):
    prior=tmp_path/'prior'
    inherited=[{'task_id':t,'directory':str(prior/str(t)),'state':{'status':'completed'}} for t in (646,598)]
    run.write_json(tmp_path/'protocol.json',{'task_ids':[625,596,721,31,730,409,651,644],
        'inherited_results':inherited,'excluded_attempts':[{'task_id':724,'directory':str(prior/'724'),
            'state':{'status':'needs_attention'},'reason':'user-authorized content rejection skip'}],
        'model':'qwen3.7-flash','limits':'development','initial_graph':'initial'})
    run.write_json(tmp_path/'state.json',{'status':'registered','tasks':{},'current_graph':'seed'})
    def stats(folder,task,state):
        return {'task_id':task,'status':state.get('status','not_started'),
            'official_score':1 if task in (646,598) else (0 if task==724 else None),
            'steps':None,'total_seconds':None,'physical_requests':9 if task==724 else 0,'unknown_usage':1 if task==724 else 0}
    monkeypatch.setattr(run,'task_stats',stats)
    summary=run.report(tmp_path)
    assert summary['planned']==10 and summary['new_planned']==8
    assert summary['completed']==summary['official_successes']==summary['official_scored']==2
    assert [r['task_id'] for r in summary['tasks'][:2]]==[646,598]
    assert 724 not in {r['task_id'] for r in summary['tasks']}
    excluded=summary['excluded_attempts'][0]
    assert excluded['task_id']==724 and excluded['official_score']==0
    assert excluded['physical_requests']==9 and excluded['unknown_usage']==1
    assert 'user-authorized content rejection skip' in (tmp_path/'report.html').read_text(encoding='utf-8')
