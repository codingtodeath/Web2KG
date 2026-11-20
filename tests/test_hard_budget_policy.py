import copy
import pytest
from scripts import run_reddit_hard10_225 as run


def inputs():
    cfg={'budget_exhaustion_policy':'retain_score_keep_previous_handoff','max_calls':2,'retry_connections':True}
    result={'status':'model_error','error':'ResponsesError: Responses call budget exhausted',
        'execution_graph':{'index_status':'completed','cards':{'calls':0,'stopped':True},
                           'card_refresh':{'pending':['T1'],'unavailable':['T2']}}}
    events=[{'event':'responses_started'} for _ in range(2)]
    return cfg,result,events


def test_only_exact_registered_budget_and_unchanged_result():
    cfg,r,e=inputs();before=copy.deepcopy(r)
    assert run.ordinary_call_exhaustion(cfg,r,e,e)
    assert r==before
    e += [{'event':'responses_started','output_retry_number':1},
          {'event':'responses_started'}, {'event':'responses_retry','persistent':True}]
    assert run.ordinary_call_exhaustion(cfg,r,e,e)
    assert not run.ordinary_call_exhaustion({**cfg,'budget_exhaustion_policy':None},r,e,e)


@pytest.mark.parametrize('fault',['format','wrong_count','service','description','card','index','framework'])
def test_real_errors_are_not_discounted(fault):
    cfg,r,e=inputs()
    if fault=='format':r['error']='ResponsesError: invalid ACTION'
    if fault=='wrong_count':e.pop()
    if fault=='service':e.append({'event':'responses_failed','http_status':403})
    if fault=='description':r['execution_graph']['description_stopped']=True
    if fault=='card':r['execution_graph']['cards']['calls']=1
    if fault=='index':r['execution_graph']['index_status']='failed'
    if fault=='framework':r['execution_graph']['framework_refresh']={'status':'refresh_failed'}
    assert not run.ordinary_call_exhaustion(cfg,r,e,e)


def test_budget_terminal_state_never_uses_partial_handoff(tmp_path):
    cfg,r,e=inputs()
    state={'tasks':{'1':{'status':'started'}},'current_graph':'previous-complete',
           'knowledge_manifest':'previous-manifest'}
    run.write_json(tmp_path/'outcome.json',dict(status='budget_exhausted',graph_advanced=False))
    run.write_json(tmp_path/'handoff.json',dict(current_graph='partial-graph'))
    run.accept_worker_outcome(tmp_path,cfg,state,'1')
    assert state['tasks']['1']['status']=='budget_exhausted'
    assert state['current_graph']=='previous-complete' and state['knowledge_manifest']=='previous-manifest'
    with pytest.raises(ValueError):run.accept_worker_outcome(tmp_path,{},state,'1')


def test_action_budget_exhaustion_uses_action_units_not_physical_calls():
    from web2kg.stage2.action_budget import POLICY
    cfg, result, _ = inputs()
    events = [dict(event='responses_started', purpose=role, action_budget_policy=POLICY, action_budget_id=key)
              for role, key in [('original_actor','search'), ('original_actor','one'),
                                ('page_selector','selector'), ('original_actor','two'),
                                ('original_actor','two')]]
    events.append(dict(event='action_budget_search', action_budget_policy=POLICY, action_budget_id='search'))
    assert run.ordinary_call_exhaustion(cfg, result, events, events)
    assert not run.ordinary_call_exhaustion({**cfg, 'max_calls':3}, result, events, events)
