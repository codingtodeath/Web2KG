import pytest
from scripts.register_hard_budget_remainder import remaining_ids


def fixture():
    return ({'task_ids':[1,2,3,4]}, {'status':'needs_attention','tasks':{
        '1':{'status':'completed'},'2':{'status':'needs_attention'}}},
        {'status':'model_error','error':'ResponsesError: Responses call budget exhausted',
         'execution_graph':{'index_status':'completed'}})


def test_only_unstarted_tasks_preserving_failed_attempt():
    cfg,state,result=fixture()
    assert remaining_ids(cfg,state,result,[])==[3,4]
    assert state['tasks']['2']['status']=='needs_attention'


def test_budget_disabled_cards_are_not_inherited_or_reclassified():
    cfg,state,result=fixture()
    result['execution_graph'].update(cards={'stopped':True,'calls':0},
        card_refresh={'pending':['T1'],'unavailable':['T2']})
    assert remaining_ids(cfg,state,result,[])==[3,4]
    assert result['execution_graph']['cards']['stopped']
    assert result['execution_graph']['card_refresh']['pending']==['T1']
    result['execution_graph']['cards']['calls']=1
    with pytest.raises(ValueError):remaining_ids(cfg,state,result,[])


@pytest.mark.parametrize('mutation', ['format','asset','service','earlier_failure','out_of_order','running'])
def test_other_failures_and_unresolved_prefix_are_rejected(mutation):
    cfg,state,result=fixture();events=[]
    if mutation=='format':result['error']='ResponsesError: invalid ACTION'
    if mutation=='asset':result['execution_graph']['index_status']='failed'
    if mutation=='service':events=[{'event':'responses_failed','http_status':403}]
    if mutation=='earlier_failure':state['tasks']['1']['status']='needs_attention'
    if mutation=='out_of_order':cfg['task_ids']=[2,1,3,4]
    if mutation=='running':state['status']='running'
    with pytest.raises(ValueError):remaining_ids(cfg,state,result,events)
