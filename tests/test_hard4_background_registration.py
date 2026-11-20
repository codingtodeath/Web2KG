from pathlib import Path
import pytest
from scripts import run_reddit_hard10_225 as run
from scripts import register_hard4_background as register


def test_next_four_never_inherit_failed_attempt_or_select_by_score(tmp_path, monkeypatch):
    previous = tmp_path.resolve()
    config = {'seed': 20261104, 'logical_origin': 'http://localhost:10599'}
    state = {'status': 'needs_attention', 'tasks': {str(k): {'status': 'completed'} for k in [625, 596, 721, 31]},
             'current_graph': str(previous / '31/execution-graph')}
    state['tasks']['730'] = {'status': 'needs_attention'}
    monkeypatch.setattr(run, 'verify', lambda p: (config, state))
    assert [t['task_id'] for t in register.next_inputs(previous)[3]] == [409, 651, 644, 735]
    state['current_graph'] = str(previous / '730/execution-graph')
    with pytest.raises(ValueError, match='failed730'):
        register.next_inputs(previous)
    state['current_graph'] = str(previous / '31/execution-graph')
    state['tasks']['409'] = {'status': 'started'}
    with pytest.raises(ValueError, match='Unexpected starts'):
        register.next_inputs(previous)


def test_report_counts_only_four_new_tasks_and_uses_registered_title(tmp_path):
    run.write_json(tmp_path / 'protocol.json', {'task_ids': [409, 651, 644, 735],
        'model': 'qwen3.7-flash', 'report_title': 'Hard4 background', 'limits': 'development', 'initial_graph': 'initial'})
    run.write_json(tmp_path / 'state.json', {'status': 'registered', 'tasks': {}, 'current_graph': 'seed'})
    summary = run.report(tmp_path)
    assert summary['new_planned'] == summary['planned'] == 4
    assert summary['completed'] == summary['official_scored'] == 0
    assert '<h1>Hard4 background</h1>' in (tmp_path / 'report.html').read_text(encoding='utf-8')
