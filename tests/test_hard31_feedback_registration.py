import copy
from pathlib import Path
import pytest
from scripts import register_hard31_feedback_repair as reg


def test_retry66_keeps_prior19_and_original407_input():
    previous=(Path.cwd()/'runs/reddit-hard31-225-flash-20261106-v1').resolve()
    config=reg.run.read(previous/'protocol.json');state=reg.run.read(previous/'state.json')
    saved=copy.deepcopy(state)
    seed,old,ids=reg.continuation_inputs(previous,config,state)
    assert seed==previous/'407/execution-graph' and ids[0]==66 and len(ids)==22
    assert len(old)==19 and len({r['task_id'] for r in old}|set(ids))==41
    assert 724 not in ids and state==saved
    state['current_graph']=str(previous/'66/execution-graph')
    with pytest.raises(ValueError,match='failed66'):reg.continuation_inputs(previous,config,state)


def test_unresolved_earlier_task_is_not_inherited():
    previous=(Path.cwd()/'runs/reddit-hard31-225-flash-20261106-v1').resolve()
    config=reg.run.read(previous/'protocol.json');state=reg.run.read(previous/'state.json')
    state['tasks']['407']['status']='started'
    with pytest.raises(ValueError,match='complete handoffs'):reg.continuation_inputs(previous,config,state)


def test_read_only_guard_check_does_not_append_to_old_logs(tmp_path):
    runtime=tmp_path/'runtime';runtime.mkdir()
    reg.run.write_json(runtime/'manifest.json',{})
    reg.run.write_json(tmp_path/'protocol.json',{'runtime':str(runtime),'integrity_policy':'boundary-v1'})
    reg.run.write_json(tmp_path/'protected-inputs.json',{})
    reg.run.write_json(tmp_path/'state.json',{'tasks':{}})
    reg.run.verify(tmp_path,record=False)
    assert not (tmp_path/'timings').exists()
