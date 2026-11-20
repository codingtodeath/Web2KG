import copy
from pathlib import Path

import pytest

from scripts import register_hard_card_repair as reg


def inputs():
    previous=(Path.cwd()/'runs/reddit-hard22-feedback-flash-20261106-v1').resolve()
    return previous,reg.run.read(previous/'protocol.json'),reg.run.read(previous/'state.json')


def test_retry624_uses_original28_input_and_retains_33_fixed_rows():
    previous,config,state=inputs();before=copy.deepcopy(state)
    seed,old,ids=reg.continuation_inputs(previous,config,state)
    assert seed==previous/'28/execution-graph'
    assert ids==[624,581,29,645,618,610,580,734]
    assert len(old)==33 and len({r['task_id'] for r in old}|set(ids))==41
    assert 724 not in ids and 624 not in {r['task_id'] for r in old}
    assert state==before and reg.run.read(previous/'624/official.json')['score']==1


def test_failed_graph_or_unresolved_prefix_cannot_be_inherited():
    previous,config,state=inputs()
    state['current_graph']=str(previous/'624/execution-graph')
    with pytest.raises(ValueError,match='failed624'):reg.continuation_inputs(previous,config,state)
    previous,config,state=inputs();state['tasks']['28']['status']='started'
    with pytest.raises(ValueError,match='complete handoffs'):reg.continuation_inputs(previous,config,state)


def test_unrelated_failure_cannot_use_card_repair_authorization(monkeypatch):
    previous,config,state=inputs();read=reg.run.read
    monkeypatch.setattr(reg.run,'read',lambda p,*a: {'status':'other_failure'} if Path(p).name=='failure-diagnosis.json' else read(p,*a))
    with pytest.raises(ValueError,match='card validation'):reg.continuation_inputs(previous,config,state)
