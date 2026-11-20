import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from web2kg.common.storage import EventLog
from web2kg.evaluation.webnavigator import OriginalResponses, final_text
from web2kg.common.responses import ResponsesError


def message(text, phase=None):
    return {'role': 'assistant', 'status': 'completed', 'phase': phase,
            'content': [{'type': 'output_text', 'text': text}]}


def test_visible_final_boundaries():
    assert final_text([message('working', 'commentary'), message('ACTION: stop [done]', 'final_answer')]) == 'ACTION: stop [done]'
    assert final_text([message('ACTION: click [1]'), message('ACTION: click [1]')]) == 'ACTION: click [1]'
    with pytest.raises(ValueError):
        final_text([message('ACTION: click [1]'), message('ACTION: click [2]')])
    with pytest.raises(ValueError):
        final_text([{'content': [{'type': 'refusal', 'refusal': 'No'}]}])


def test_empty_final_uses_complete_commentary_action():
    text = 'REASON:\nOpen the sorting menu.\nACTION:\nclick [18539]\nOBSERVATION HIGHLIGHT:\n18539'
    used = []
    assert final_text([message(text, 'commentary'), message('  ', 'final_answer')],
                      on_fallback=lambda: used.append(True)) == text
    assert used == [True]
    assert final_text([message(text, 'commentary'), message('ACTION: stop [done]', 'final_answer')]) == 'ACTION: stop [done]'


@pytest.mark.parametrize('texts', [
    ['I will open the menu.'], ['ACTION:\n'],
    ['ACTION: click [1]\nACTION: click [2]'],
    ['ACTION: click [1]', 'ACTION: click [2]'],
])
def test_empty_final_does_not_guess_commentary(texts):
    with pytest.raises(ValueError):
        final_text([*[message(t, 'commentary') for t in texts], message('', 'final_answer')])


@pytest.mark.parametrize('bad_part', ['incomplete_commentary', 'incomplete_final', 'refusal'])
def test_empty_final_fallback_preserves_completion_and_refusal_checks(bad_part):
    messages = [message('ACTION: click [1]', 'commentary'), message('', 'final_answer')]
    if bad_part == 'refusal':
        messages[1]['content'] = [{'type': 'refusal', 'refusal': 'No'}]
    else:
        messages[0 if bad_part == 'incomplete_commentary' else 1]['status'] = 'in_progress'
    with pytest.raises(ValueError):
        final_text(messages)


@pytest.mark.parametrize('verified', [True, False])
def test_responses_plain_text_budget_and_usage(tmp_path, verified):
    env = tmp_path / 'test.env'
    env.write_text('CRAWL_API_KEY=dummy\nCRAWL_BASE_URL=https://invalid.example/v1\nCRAWL_MODEL=test')
    client = OriginalResponses(env, EventLog(tmp_path), 1, 2048, verified_protocol=verified)
    seen = []
    def post(url, **kwargs):
        seen.append(kwargs['json'])
        return SimpleNamespace(ok=True, json=lambda: {'id': 'test', 'model': 'test', 'status': 'completed',
            'usage': {'input_tokens': 12, 'output_tokens': 5}, 'output': [{'type': 'message', **message('ACTION: stop [done]', 'final_answer')}]})
    client.session.post = post
    image = base64.b64encode(b'\x89PNG\r\n\x1a\nfixture').decode()
    assert client.call([{'role': 'user', 'content': [{'type': 'text', 'text': 'synthetic task'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + image}}]}],
        system_prompt='Original prompt') == 'ACTION: stop [done]'
    assert seen[0]['text']['format'] == {'type': 'text'}
    assert seen[0]['input'][0]['content'][1]['image_url'] == 'data:image/png;base64,' + image
    assert seen[0]['instructions'].startswith('Original prompt')
    assert ('retrieved_data' in seen[0]['instructions']) == verified
    with pytest.raises(ResponsesError, match='budget'):
        client.call([{'role': 'user', 'content': 'another'}])
    assert len(seen) == 1
    event = [json.loads(s) for s in (tmp_path / 'events.jsonl').read_text().splitlines() if 'responses_received' in s][0]
    assert event['usage']['input_tokens'] == 12


@pytest.mark.parametrize('status', [403, 502, 503])
def test_http_failure_never_retries(tmp_path, status):
    env = tmp_path / 'test.env'
    env.write_text('CRAWL_API_KEY=dummy\nCRAWL_BASE_URL=https://invalid.example/v1\nCRAWL_MODEL=test')
    client = OriginalResponses(env, EventLog(tmp_path), 31)
    calls = []
    payload = {'error': {'code': 'channel_unavailable', 'type': 'gateway_error', 'message': 'private-secret'}}
    client.session.post = lambda *a, **k: calls.append(1) or SimpleNamespace(ok=False, status_code=status,
        content=json.dumps(payload).encode(), json=lambda: payload)
    with pytest.raises(ResponsesError, match=str(status)) as raised:
        client.call([{'role': 'user', 'content': 'test'}])
    assert raised.value.http_status == status and raised.value.kind == 'http'
    assert calls == [1]
    detail = json.loads(next((tmp_path/'llm').glob('*/http_error.json')).read_text())
    assert detail['code'] == 'channel_unavailable' and detail['http_status'] == status
    assert 'private-secret' not in json.dumps(detail)
    assert 'private-secret' not in (tmp_path/'events.jsonl').read_text()


@pytest.mark.parametrize('failure', ['http', 'transport', 'empty'])
def test_opt_in_retry_records_every_attempt(tmp_path, monkeypatch, failure):
    import requests
    env = tmp_path / 'test.env'
    env.write_text('CRAWL_API_KEY=dummy\nCRAWL_BASE_URL=https://invalid.example/v1\nCRAWL_MODEL=test')
    client = OriginalResponses(env, EventLog(tmp_path), 3, max_retries=2)
    delays = []
    monkeypatch.setattr('web2kg.evaluation.webnavigator.time.sleep', delays.append)
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs['json'])
        if len(calls) < 3:
            if failure == 'transport':
                raise requests.ConnectionError('private transport detail')
            if failure == 'http':
                return SimpleNamespace(ok=False, status_code=502, content=b'{}', json=lambda: {})
            text = ''
        else:
            text = 'ACTION: click [1]'
        return SimpleNamespace(ok=True, json=lambda: {'status': 'completed', 'model': 'test',
            'usage': {'input_tokens': 10, 'output_tokens': 2},
            'output': [{'type': 'message', **message(text, 'final_answer')}]})
    client.session.post = post
    assert client.call([{'role': 'user', 'content': 'test'}]) == 'ACTION: click [1]'
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    assert delays == [2, 5]
    events = [json.loads(x) for x in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert sum(e['event'] == 'responses_started' for e in events) == 3
    assert sum(e['event'] == 'responses_retry' for e in events) == 2
    received = [e for e in events if e['event'] == 'responses_received']
    assert sum(e['usage']['input_tokens'] for e in received) == (30 if failure == 'empty' else 10)
    assert 'private transport detail' not in (tmp_path/'events.jsonl').read_text()


@pytest.mark.parametrize('status,budget,expected', [(403, 31, 1), (502, 2, 2), (503, 31, 3)])
def test_retry_still_obeys_auth_budget_and_attempt_limits(tmp_path, monkeypatch, status, budget, expected):
    env = tmp_path / 'test.env'
    env.write_text('CRAWL_API_KEY=dummy\nCRAWL_BASE_URL=https://invalid.example/v1\nCRAWL_MODEL=test')
    client = OriginalResponses(env, EventLog(tmp_path), budget, max_retries=2)
    monkeypatch.setattr('web2kg.evaluation.webnavigator.time.sleep', lambda _: None)
    calls = []
    client.session.post = lambda *a, **k: calls.append(1) or SimpleNamespace(
        ok=False, status_code=status, content=b'{}', json=lambda: {})
    with pytest.raises(ResponsesError):
        client.call([{'role': 'user', 'content': 'test'}])
    assert len(calls) == expected


def test_conflicting_output_not_retried(tmp_path, monkeypatch):
    env = tmp_path / 'test.env'
    env.write_text('CRAWL_API_KEY=dummy\nCRAWL_BASE_URL=https://invalid.example/v1\nCRAWL_MODEL=test')
    client = OriginalResponses(env, EventLog(tmp_path), 31, max_retries=2)
    calls = []
    client.session.post = lambda *a, **k: calls.append(1) or SimpleNamespace(ok=True,
        json=lambda: {'status': 'completed', 'output': [
            {'type': 'message', **message('ACTION: click [1]')},
            {'type': 'message', **message('ACTION: click [2]')}]})
    with pytest.raises(ResponsesError, match='conflicting'):
        client.call([{'role': 'user', 'content': 'test'}])
    assert len(calls) == 1


@pytest.mark.parametrize('selector',[False,True])
def test_opt_in_action_adapter_preserves_selector_and_logs_consumed_text(tmp_path,selector):
    env=tmp_path/'test.env'
    env.write_text('CRAWL_API_KEY=dummy\nCRAWL_BASE_URL=https://invalid.example/v1\nCRAWL_MODEL=test')
    client=OriginalResponses(env,EventLog(tmp_path),1,actor_compatibility=True)
    raw='{"target_page":"None"}' if selector else 'type [15] [exact] [press_enter_after=0]'
    client.session.post=lambda *a,**k:SimpleNamespace(ok=True,json=lambda:{'status':'completed',
        'output':[{'type':'message',**message(raw,'final_answer')}]})
    result=client.call([{'role':'user','content':'fixture'}],system_prompt='target_page' if selector else 'actor')
    assert result==raw if selector else result.strip()=='ACTION:\ntype [15] [exact] [0]'
    assert next((tmp_path/'llm').glob('*/parsed.txt')).read_text()==result
    assert ('actor_output_adapted' in client.log.path.read_text()) is (not selector)
