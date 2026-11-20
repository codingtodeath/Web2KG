import copy
import json

import pytest

from scripts.qwen_hard_transport import chat_request, daily_exhausted, free_unavailable
from scripts.run_hard_qwen_continuation import pending, export_local_trajectories


def test_export_keeps_inherited_paths_and_does_not_mutate_frozen_reader(tmp_path):
    directory = tmp_path / 'ours'
    summary = {'tasks': [
        {'task_id': 624, 'evidence_directory': str(tmp_path / 'old/ours/624')},
        {'task_id': 400, 'evidence_directory': str(directory / '400')}]}
    original_summary = copy.deepcopy(summary)
    def reader(path, default=None):
        return summary if path == directory / 'summary.json' else default
    namespace = {'read': reader}
    exec('def exporter(batches, output):\n    return read(batches[0] / "summary.json")', namespace)
    exporter = namespace['exporter']
    result = export_local_trajectories(directory, exporter)
    assert [r['task_id'] for r in result['tasks']] == [400]
    assert summary == original_summary
    assert exporter.__globals__['read'] is reader


def test_daily_limit_is_not_temporary_throttling():
    assert daily_exhausted(429, {'error': {'message': 'Rate limit exceeded: free-models-per-day'}})
    assert not daily_exhausted(429, {'error': {'message': 'free-models-per-min'}})
    assert not daily_exhausted(429, {'error': {'message': 'temporarily rate-limited upstream'}})
    assert not daily_exhausted(401, {'error': {'message': 'daily limit'}})
    assert free_unavailable(404, {'error': {'message': 'This model is unavailable for free.'}})
    assert not free_unavailable(404, {'error': {'message': 'Unknown endpoint'}})
    assert not free_unavailable(402, {'error': {'message': 'Insufficient credits'}})


def test_wire_conversion_preserves_exact_multimodal_content():
    request = {'model': 'qwen/qwen3.8-27b:free', 'instructions': 'Exact original instructions',
               'max_output_tokens': 2048,
               'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'line1\nline2'},
                         {'type': 'input_image', 'image_url': 'data:image/png;base64,AAAA'}]}]}
    old = copy.deepcopy(request)
    wire = chat_request(request)
    assert request == old
    assert wire['messages'][0]['content'] == request['instructions']
    assert wire['messages'][1]['content'][0]['text'] == 'line1\nline2'
    assert wire['messages'][1]['content'][1]['image_url']['url'] == 'data:image/png;base64,AAAA'
    assert wire['max_tokens'] == 2048


def test_completed_tasks_never_repeat_and_unresolved_blocks():
    schedule = [{'arm': 'webnavigator', 'task_id': i} for i in range(42)] + [
        {'arm': 'ours', 'task_id': i} for i in range(22, 42)]
    protocol = {'schedule': schedule}
    state = {'tasks': {'ours/29': {'status': 'completed', 'original_status': 'needs_attention'},
                       'webnavigator/0': {'status': 'completed'}}}
    result = pending(protocol, state)
    assert len(result) == 60
    assert {'arm': 'ours', 'task_id': 29} not in result
    state['tasks']['ours/30'] = {'status': 'started'}
    with pytest.raises(ValueError, match='unresolved'):
        pending(protocol, state)
    assert len(pending(protocol, state, 'ours/30')) == 59
    with pytest.raises(ValueError, match='unresolved'):
        pending(protocol, state, 'ours/31')
    protocol['schedule'][0] = protocol['schedule'][1]
    with pytest.raises(ValueError, match='Expected'):
        pending(protocol, state)


@pytest.mark.parametrize('status,message,expected_provider', [
    (429, 'Rate limit exceeded: free-models-per-day', 'bailian'),
    (404, 'This model is unavailable for free.', 'bailian'),
    (429, 'temporarily rate-limited upstream', 'openrouter'),
    (503, 'Service unavailable', 'openrouter'),
])
def test_routing_with_one_logical_call(tmp_path, monkeypatch, status, message, expected_provider):
    import requests
    from scripts import qwen_hard_transport as transport
    from web2kg.evaluation import webnavigator
    from web2kg.common.storage import EventLog
    from web2kg.common.activity import read_events
    monkeypatch.setattr(webnavigator, 'OriginalResponses', webnavigator.OriginalResponses)
    monkeypatch.setattr('time.sleep', lambda _: None)
    env = tmp_path / 'private.env'
    env.write_text('CRAWL_API_KEY=test\nCRAWL_BASE_URL=https://unused.test\nCRAWL_MODEL=test\n'
                   'OPENROUTER_API_KEY=test\nOPENROUTER_BASE_URL=https://openrouter.ai/api/v1\nOPENROUTER_MODEL=free-model\n'
                   'BAILIAN_API_KEY=test\nBAILIAN_BASE_URL=https://bailian.test/v1\nBAILIAN_MODEL=paid-model\n')
    calls = []
    def post(self, url, **kwargs):
        calls.append((url, kwargs['json']))
        response = requests.Response()
        response.status_code = status if len(calls) == 1 else 200
        data = {'error': {'message': message}} if len(calls) == 1 else {
            'id': 'test', 'model': kwargs['json']['model'],
            'choices': [{'message': {'content': 'OK', 'reasoning_content': 'private reasoning'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 4}}
        response._content = json.dumps(data).encode()
        return response
    monkeypatch.setattr(requests.Session, 'post', post)
    cls = transport.install(env, tmp_path / 'routing.json')
    client = cls(env, EventLog(tmp_path / 'task'), 1, 2048, verified_protocol=False, retry_connections=True)
    assert client.call([{'role': 'user', 'content': 'Exact user content'}]) == 'OK'
    assert len(calls) == 2 and client.provider == expected_provider and client.budget_used() == 1
    assert calls[0][1]['messages'] == calls[1][1]['messages']
    events = read_events(tmp_path / 'task/events.jsonl')
    assert sum(e['event'] == 'responses_started' for e in events) == 2
    assert sum(e['event'] == 'responses_received' for e in events) == 1
    assert any(e['event'] == 'provider_fallback' for e in events) == (expected_provider == 'bailian')
