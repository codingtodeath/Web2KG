from pathlib import Path

import pytest

from scripts import run_hard_domain_repair as reg


@pytest.mark.parametrize('task,parent', [(615,'68'), (617,'400'), (618,'645')])
def test_each_retry_uses_its_own_original_complete_input(task, parent):
    previous, folder, config, graph = reg.retry_inputs(task)
    assert folder == previous/str(task)
    assert graph == previous/parent/'execution-graph'
    assert config['max_graph_nodes'] == 225 and config['model'] == 'qwen3.7-flash'
    assert reg.run.read(folder/'official.json')['score'] == 0
    assert not reg.run.read(config['experience_store'])['pages']


def test_other_failures_not_covered_by_registration(monkeypatch):
    original = reg.run.read
    monkeypatch.setattr(reg.run, 'read', lambda p,*a: {'error':'other_failure'}
        if Path(p).name == 'result.json' else original(p,*a))
    with pytest.raises(ValueError, match='navigation-domain failure only'):
        reg.retry_inputs(615)


def test_changed_original_memory_cannot_silently_be_used(monkeypatch):
    original = reg.run.file_hash
    monkeypatch.setattr(reg.run, 'file_hash', lambda p: 'changed'
        if Path(p).name == 'experience.json' else original(p))
    with pytest.raises(ValueError, match='Original task input/experience changed'):
        reg.retry_inputs(618)


def test_started_controller_cannot_resume_or_start_a_worker(tmp_path, monkeypatch):
    reg.run.write_json(tmp_path/'protocol.json',dict(runtime=str(reg.ROOT),tasks=list(reg.PREVIOUS)))
    reg.run.write_json(tmp_path/'state.json',dict(status='needs_attention',tasks={'615':'started'}))
    monkeypatch.setattr(reg.run,'queue',lambda _:pytest.fail('must not launch'))
    with pytest.raises(ValueError,match='Never restart'):
        reg.live(tmp_path)
