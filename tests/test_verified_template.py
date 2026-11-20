import json
from unittest.mock import Mock

import pytest

from web2kg.common.storage import EventLog
from web2kg.evaluation.verified import VerifiedBrowser, reset_copy


def test_isolated_transport_preserves_visible_urls_and_har(site, tmp_path):
    # No server exists on the visible address: all page traffic must use the backend.
    logical = 'http://127.0.0.1:65521'
    with VerifiedBrowser(logical, EventLog(tmp_path), backend=site, allow_writes=True, settle_ms=20) as browser:
        assert browser.page.url == logical + '/'
        browser.page.get_by_role('link', name='Catalog', exact=True).click()
        assert browser.page.url == logical + '/catalog'
        assert browser.page.locator('h1').inner_text() == 'Catalog'
        assert browser.page.evaluate("fetch('/change',{method:'POST',body:'test=fixture'}).then(r=>r.text())") == 'mutation'
    entries = json.loads((tmp_path / 'network.har').read_text())['log']['entries']
    assert any(e['request']['url'] == logical + '/change' and e['request']['method'] == 'POST' for e in entries)
    assert all(not e['request']['url'].startswith(site) for e in entries)
    raw = json.loads((tmp_path / 'network-transport.har').read_text())['log']['entries']
    assert len(raw) > len(entries) and all(e in raw for e in entries)


def test_wrong_container_cannot_be_reset():
    with pytest.raises(ValueError, match='dedicated'):
        reset_copy('web2kg-dev-reddit', 'fixture', 'http://127.0.0.1:10599', 10598, 'http://127.0.0.1:10499', 'fixture')


@pytest.mark.parametrize('flag', [None, False, True])
def test_worker_requires_explicit_finalization_registration(tmp_path, monkeypatch, flag):
    from scripts import run_verified_template as runner
    monkeypatch.setattr(runner, 'check_inputs', lambda _: None)
    monkeypatch.setattr(runner.Graph, 'load', lambda _: object())
    monkeypatch.setattr(runner, 'OriginalResponses', Mock())
    monkeypatch.setattr(runner, 'TemplateKnowledge', Mock())
    execute = Mock(return_value={'answer': 'saved answer'})
    monkeypatch.setattr(runner, 'run_task', execute)
    (tmp_path / 'public-inputs.json').write_text(json.dumps([{
        'task_id': 1, 'sites': ['reddit'], 'intent': 'fixture', 'start_urls': ['http://test/']}]))
    config = {'graph': str(tmp_path), 'cards': 'fixture', 'model': 'fixture', 'max_calls': 1,
              'max_steps': 1, 'top_k': 1, 'backend': 'http://test', 'update_graph': True,
              'encoder_endpoint': 'http://127.0.0.1:8793/v1/embeddings'}
    if flag is not None:
        config['finalize_graph_assets'] = flag
    runner.worker(tmp_path, config, 1)
    assert execute.call_args.kwargs['finalize_graph_assets'] is (flag is True)


def test_new_batch_report_keeps_previous_report(tmp_path):
    from scripts.run_verified_template import summarize
    graph = tmp_path / 'graph'
    (graph / 'reports').mkdir(parents=True)
    old = graph / 'reports/template-agent-hard10-v1.html'
    old.write_text('Original results', encoding='utf-8')
    batch = tmp_path / 'template-agent-hard10-v2'
    batch.mkdir()
    summarize(batch, {'task_ids': [625], 'graph': str(graph)}, {})
    assert old.read_text(encoding='utf-8') == 'Original results'
    assert (graph / 'reports/template-agent-hard10-v2.html').exists()


def test_report_counts_annotation_and_card_requests_including_failed_calls(tmp_path):
    from scripts.run_verified_template import summarize
    graph, batch = tmp_path / 'graph', tmp_path / 'batch'
    graph.mkdir(); batch.mkdir()
    for directory, model, calls, tokens in (
        (batch / '1', 'gpt-5.6-terra', 1, 10),
        (batch / '1/graph-annotations', 'gpt-6-luna', 2, 20),
        (batch / '1/graph-cards', 'gpt-6-luna', 2, 30)):
        log = EventLog(directory)
        for _ in range(calls):
            log.emit('responses_started')
        log.emit('responses_received', actual_model=model, usage={'input_tokens': tokens, 'output_tokens': 5})
    result = summarize(batch, {'task_ids': [1], 'graph': str(graph), 'transition_calls': 7}, {})
    row = result['tasks'][0]
    assert (row['calls'], row['annotation_calls'], row['card_calls'], row['responses']) == (5, 2, 2, 3)
    assert (row['input_tokens'], row['output_tokens']) == (60, 15)
    assert row['actual_models'] == ['gpt-5.6-terra', 'gpt-6-luna']
    assert '另限7次Luna描述' in (batch / 'report.html').read_text(encoding='utf-8')


@pytest.mark.parametrize('with_comparison', [False, True])
def test_original_report_does_not_depend_on_archived_batch(tmp_path, with_comparison):
    from scripts.run_verified_webnavigator import summarize
    batch, graph = tmp_path / 'new-run', tmp_path / 'graph'
    (batch / '1').mkdir(parents=True)
    (graph / 'reports').mkdir(parents=True)
    (batch / '1/official.json').write_text('{"score":1}', encoding='utf-8')
    config = {'task_ids': [1], 'graph': str(graph)}
    if with_comparison:
        comparison = tmp_path / 'comparison.json'
        comparison.write_text('{"tasks":[{"task_id":1,"official_score":0}]}', encoding='utf-8')
        config['comparison_summary'] = str(comparison)
    result = summarize(batch, config, {'1': {'status': 'completed'}})
    assert result['official_scored'] == result['official_successes'] == 1
    assert result['paired_scored'] == result['original_only'] == int(with_comparison)
    assert result['tasks'][0]['template_score'] == (0 if with_comparison else None)


@pytest.mark.parametrize('status', [400, 401, 402, 403, 429, 500, 502, 503, 504])
def test_provider_http_failure_gates_batch(status):
    from scripts.run_verified_template import provider_gate
    assert provider_gate({'error': f'ResponsesError: Responses HTTP {status}'})
    assert provider_gate({'provider_failure': {'kind': 'http', 'http_status': status}})


def test_transport_gates_but_output_and_task_errors_do_not():
    from scripts.run_verified_template import provider_gate
    assert provider_gate({'provider_failure': {'kind': 'transport'}})
    for error in ['Missing or conflicting final messages', 'Snapshot has no usable visible structure', 'budget exhausted']:
        assert not provider_gate({'error': error})


def test_first_503_stops_queue_without_starting_next_task(tmp_path, monkeypatch):
    from scripts import run_verified_template as runner
    monkeypatch.setattr(runner, 'check_inputs', lambda _: None)
    monkeypatch.setattr(runner.Graph, 'load', lambda _: object())
    monkeypatch.setattr(runner, 'LocalEncoder', Mock())
    reset = Mock(return_value='owned-container')
    monkeypatch.setattr(runner, 'reset_copy', reset)
    def worker(*a, **kw):
        (tmp_path/'1/result.json').write_text(json.dumps({'status': 'model_error',
            'error': 'Responses HTTP 503', 'provider_failure': {'kind': 'http', 'http_status': 503}}))
        return Mock(wait=Mock(return_value=0))
    spawn = Mock(side_effect=worker)
    monkeypatch.setattr(runner.subprocess, 'Popen', spawn)
    monkeypatch.setattr(runner, 'score', lambda *a: {'score': 0})
    config = dict(graph=str(tmp_path), encoder_python='unused', task_ids=[1, 2], container='owned',
                  image_id='local', backend='http://local', logical_origin='http://logical', owner='test',
                  task_timeout_seconds=10, verified_root='unused')
    summarize = lambda *a: {'tasks': [{'calls': 1}]}
    runner.queue(tmp_path, config, summarizer=summarize)
    state = json.loads((tmp_path/'state.json').read_text())
    assert state['1']['status'] == 'provider_error' and '2' not in state
    assert spawn.call_count == reset.call_count == 1
    with pytest.raises(RuntimeError, match='unresolved'):
        runner.queue(tmp_path, config, summarizer=summarize)
    assert spawn.call_count == 1
