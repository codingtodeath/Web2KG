from pathlib import Path
import pytest
from scripts import run_hard_recovery_tiers as reg


@pytest.mark.parametrize('task,parent', [(66, None), (68, '597'), (67, '640')])
def test_books_original_inputs_and_models(task, parent):
    previous, folder, config, graph = reg.retry_inputs(task)
    assert graph == (previous/parent/'execution-graph' if parent else previous/'starting-graph')
    assert config['max_graph_nodes'] == 225 and config['model'] == 'qwen3.7-flash'
    assert reg.run.read(folder/'official.json')['score'] == 0
    assert not reg.run.read(config['experience_store'])['pages']


def test_non_diagnosis_failure_is_not_an_authorized_books_retry(monkeypatch):
    original = reg.run.read
    monkeypatch.setattr(reg.run, 'read', lambda p,*a: {'attempted': False}
        if Path(p).name == 'recovery.json' else original(p,*a))
    with pytest.raises(ValueError, match='Books failures only'): reg.retry_inputs(66)


def test_started_controller_never_restarts(tmp_path, monkeypatch):
    reg.run.write_json(tmp_path/'protocol.json', dict(runtime=str(reg.ROOT), tasks=list(reg.PREVIOUS)))
    reg.run.write_json(tmp_path/'state.json', dict(status='needs_attention', tasks={'66':'started'}))
    monkeypatch.setattr(reg.run, 'queue', lambda _:pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='Never restart'): reg.live(tmp_path)
