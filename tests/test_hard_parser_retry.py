from pathlib import Path
import pytest
from scripts import register_hard_parser_retry as register
from scripts import run_reddit_hard10_225 as run


@pytest.fixture
def previous(tmp_path, monkeypatch):
    previous = tmp_path.resolve()
    config = {}
    state = dict(status='user_stopped', tasks={str(k): {'status': s} for k, s in
        [(409, 'completed'), (651, 'completed'), (644, 'completed'), (735, 'needs_attention')]},
        current_graph=str(previous / '644/execution-graph'))
    monkeypatch.setattr(run, 'verify', lambda p: (config, state))
    run.write_json(previous / 'failure-diagnosis.json', dict(kind='engineering_stop', discovered_in_task=644))
    run.write_json(previous / '644/input.json', dict(graph=str(previous / '651/execution-graph')))
    run.write_json(previous / 'public-inputs.json', [dict(task_id=k) for k in [409, 651, 644, 735]])
    return previous, state


def test_retry_uses_pre644_knowledge_and_exact_two_tasks(previous):
    path, _ = previous
    _, _, seed, tasks = register.retry_inputs(path)
    assert seed == path / '651/execution-graph'
    assert [t['task_id'] for t in tasks] == [644, 735]


@pytest.mark.parametrize('bad_graph', ['644', '735'])
def test_cannot_seed_retry_from_old_same_task_or_interrupted_graph(previous, bad_graph):
    path, _ = previous
    run.write_json(path / '644/input.json', dict(graph=str(path / bad_graph / 'execution-graph')))
    with pytest.raises(ValueError, match='original651'):
        register.retry_inputs(path)


def test_unexpected_start_or_task_order_is_rejected(previous):
    path, state = previous
    state['tasks']['999'] = {'status': 'started'}
    with pytest.raises(ValueError, match='no additional starts'):
        register.retry_inputs(path)
    del state['tasks']['999']
    run.write_json(path / 'public-inputs.json', [dict(task_id=k) for k in [735, 644]])
    with pytest.raises(ValueError, match='sequence'):
        register.retry_inputs(path)


def test_second_repair_retries_only735_from_completed644(previous):
    path, state = previous
    state['tasks'] = {'644': {'status': 'completed'}, '735': {'status': 'needs_attention'}}
    run.write_json(path / 'failure-diagnosis.json', dict(kind='engineering_stop',
        discovered_in_task=735, bug='identical_action_rejected'))
    run.write_json(path / '735/input.json', dict(graph=str(path / '644/execution-graph')))
    run.write_json(path / 'public-inputs.json', [dict(task_id=k) for k in [644,735]])
    _, _, seed, tasks = register.retry_inputs(path)
    assert seed == path / '644/execution-graph'
    assert [t['task_id'] for t in tasks] == [735]
    run.write_json(path / '735/input.json', dict(graph=str(path / '735/execution-graph')))
    with pytest.raises(ValueError, match='original644'):
        register.retry_inputs(path)
