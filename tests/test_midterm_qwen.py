import json
from unittest.mock import Mock

import pytest

from scripts import run_midterm_qwen as run


def test_full_single_site_subsets_and_public_fields():
    for site, count in [('reddit', 42), ('gitlab', 57)]:
        tasks = run.public_tasks(site)
        assert len(tasks) == count == len({t['task_id'] for t in tasks})
        assert tasks == run.public_tasks(site)
        assert all(t['sites'] == [site] and set(t) == {'task_id', 'sites', 'intent', 'start_urls'} for t in tasks)
        assert all('__' not in u for t in tasks for u in t['start_urls'])


def test_regression_alarm_does_not_treat_missing_as_failures():
    previous = {str(i): 1 for i in range(10)}
    rows = [dict(arm='ours', task_id=i, official_score=None) for i in range(10)]
    assert run.regression(rows, previous) is None
    for row in rows[:3]:
        row['official_score'] = 0
    assert run.regression(rows, previous) == 'three_consecutive_historical_successes_now_failed'
    rows = [dict(arm='ours', task_id=i, official_score=i % 2) for i in range(6)]
    assert run.regression(rows, previous) == 'historical_prefix_drop_at_least_25pp_and_three_tasks'
    rows += [dict(arm='webnavigator', task_id=i, official_score=1) for i in range(6)]
    assert run.regression(rows) == 'paired_baseline_gap_at_least_30pp_and_three_tasks'


def test_never_restart_unresolved_task_and_duplicate_start():
    protocol = dict(schedule=[dict(arm='ours', task_id=1), dict(arm='webnavigator', task_id=1)], max_new_task_starts=2)
    state = dict(tasks={'ours/1': dict(status='started')})
    with pytest.raises(ValueError, match='Unresolved'):
        run.pending(protocol, state)
    assert run.pending(protocol, state, 'ours/1') == [protocol['schedule'][1]]
    protocol['schedule'][1] = protocol['schedule'][0]
    with pytest.raises(ValueError, match='Duplicate'):
        run.pending(protocol, state)


@pytest.mark.parametrize('site', ['reddit', 'gitlab'])
def test_all_new_method_roles_use_qwen_and_registered_output_caps(tmp_path, monkeypatch, site):
    from scripts import run_verified_template as runner
    qwen, luna = Mock(), Mock()
    monkeypatch.setattr(runner, 'check_inputs', lambda _: None)
    monkeypatch.setattr(runner, 'execution_clients', lambda *args: (qwen, qwen))
    monkeypatch.setattr(runner, 'ResponsesClient', luna)
    monkeypatch.setattr(runner.Graph, 'load', lambda _: object())
    monkeypatch.setattr(runner, 'TemplateKnowledge', Mock())
    execute = Mock(return_value={'answer': 'answer'})
    monkeypatch.setattr(runner, 'run_task', execute)
    (tmp_path / 'public-inputs.json').write_text(json.dumps([dict(task_id=1, sites=[site], intent='fixture', start_urls=['http://fixture/'])]))
    config = dict(graph=str(tmp_path), cards='fixture', site=site, model=run.MODEL, max_calls=31,
        max_steps=30, top_k=5, backend='http://fixture', update_graph=True,
        auxiliary_provider='qwen', auxiliary_model=run.MODEL, diagnosis_model=run.MODEL, diagnosis_calls=5,
        auxiliary_output_tokens=4096, diagnosis_output_tokens=4096, output_tokens_per_call=4096,
        encoder_endpoint='http://127.0.0.1:8793/v1/embeddings', storage_state='private-session.json')
    runner.worker(tmp_path, config, 1)
    assert qwen.call_count == 4 and luna.call_count == 0
    assert all(call.args[3] == 4096 and call.kwargs['model'] == run.MODEL for call in qwen.call_args_list)
    assert execute.call_args.kwargs['site'] == site
    assert execute.call_args.kwargs['storage_state'] == 'private-session.json'
    assert bool(execute.call_args.kwargs['headers']) == (site == 'reddit')


def test_official_scoring_uses_registered_site_placeholder(tmp_path, monkeypatch):
    from web2kg.evaluation import verified
    monkeypatch.setattr(verified.subprocess, 'run', Mock(return_value=Mock(stdout='', stderr='', returncode=0)))
    (tmp_path / '1').mkdir()
    for site in ['gitlab', 'reddit']:
        verified.score('fixture', tmp_path, 1, 'http://local', site)
        assert set(json.loads((tmp_path / 'evaluator-config.json').read_text())['environments']) == {'__' + site.upper() + '__'}


def test_gitlab_reset_does_not_touch_user_container(monkeypatch):
    foreign = dict(Config={'Labels': {'web2kg.midterm': 'another-owner'}})
    call = Mock(return_value=Mock(returncode=0, stdout=json.dumps([foreign])))
    monkeypatch.setattr(run.subprocess, 'run', call)
    with pytest.raises(ValueError, match='owner mismatch'):
        run.reset_gitlab({'image_id': 'local'})
    assert call.call_count == 1


def test_queue_uses_index_matching_encoder_runtime(tmp_path, monkeypatch):
    from contextlib import nullcontext
    config = dict(shared_lock='unused', encoder_runtime=str(tmp_path / 'exact-encoder'),
                  encoder_python='local-python', stop_file=str(tmp_path / 'STOP'))
    state = dict(status='registered', tasks={}, current_graph='original-graph')
    (tmp_path / 'reddit').mkdir()
    (tmp_path / 'reddit/protocol.json').write_text(json.dumps(config))
    (tmp_path / 'STOP').write_text('stop before task')
    monkeypatch.setattr(run, 'queue_lock', lambda _: nullcontext())
    monkeypatch.setattr(run, 'verify', lambda _: (config, state, [dict(arm='ours', task_id=1)]))
    monkeypatch.setattr(run.Graph, 'load', lambda _: 'graph-object')
    monkeypatch.setattr(run, 'report', Mock())
    encoder = Mock()
    monkeypatch.setattr(run, 'LocalEncoder', encoder)
    run.queue(tmp_path, 'reddit')
    assert encoder.call_args.args == ('graph-object', tmp_path / 'exact-encoder', 'local-python')
    encoder.return_value.ensure.assert_called_once()
    encoder.return_value.close.assert_called_once()
    assert state['tasks'] == {}
