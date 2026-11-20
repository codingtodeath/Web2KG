import json
from types import SimpleNamespace

import pytest

from web2kg.common.activity import read_events
from web2kg.common.responses import ResponsesClient, ResponsesError, object_schema
from web2kg.common.storage import EventLog


def client(tmp_path, outputs, *, retries=3):
    env = tmp_path / 'fixture.env'
    env.write_text('CRAWL_API_KEY=private\nCRAWL_BASE_URL=https://unused.test\nCRAWL_MODEL=fixture\n')
    c = ResponsesClient(env, EventLog(tmp_path / 'log'), max_calls=1)
    c.configure_output_retries(retries)
    sent = []
    def post(*args, **kwargs):
        sent.append(kwargs['json'])
        content = outputs[min(len(sent)-1, len(outputs)-1)]
        if content in ('401', '503'):
            return SimpleNamespace(ok=False, status_code=int(content))
        parts = [{'type': 'refusal', 'refusal': 'declined'}] if content == 'refusal' else [{'type': 'output_text', 'text': content}]
        return SimpleNamespace(ok=True, json=lambda: {'status': 'completed', 'model': 'fixture',
            'usage': {'input_tokens': 10, 'output_tokens': 5},
            'output': [{'type': 'message', 'role': 'assistant', 'content': parts}]})
    c.session.post = post
    return c, sent


SCHEMA = object_schema({'ok': {'type': 'boolean'}})


def test_invalid_then_schema_then_success_preserves_evidence_and_logical_budget(tmp_path):
    c, sent = client(tmp_path, ['{"ok": bad}', '{"ok":"yes"}', '{"ok":true}'])
    assert c.json('describe', {'fact': 'same'}, SCHEMA, 'description') == {'ok': True}
    assert len(sent) == 3 and c.budget_used() == 1
    assert all(r['input'] == sent[0]['input'] and r['text'] == sent[0]['text'] for r in sent)
    events = read_events(c.log.path)
    assert sum(e['event'] == 'responses_output_retry' for e in events) == 2
    assert sum(e['event'] == 'responses_received' for e in events) == 3
    assert not any(e['event'] == 'responses_retry' for e in events)
    saved = [json.loads(p.read_text()) for p in (c.log.directory / 'llm').glob('*/response.json')]
    assert sorted(r['parse']['status'] for r in saved) == ['accepted', 'rejected', 'rejected']
    with pytest.raises(ResponsesError, match='budget'):
        c.json('describe', {}, SCHEMA, 'description')
    assert len(sent) == 3
    assert c.fork().max_output_retries == 3


@pytest.mark.parametrize('output,expected', [('bad',4), ('401',1), ('refusal',1)])
def test_bounded_retry_does_not_retry_auth_or_refusal(tmp_path, output, expected):
    c, sent = client(tmp_path, [output])
    with pytest.raises(ResponsesError):
        c.json('describe', {}, SCHEMA, 'description')
    assert len(sent) == expected


def test_stop_between_attempts_prevents_new_request(tmp_path):
    c, sent = client(tmp_path, ['bad', '{"ok":true}'])
    original = c.log.emit
    stop = tmp_path / 'STOP'; c.stop_file = stop
    def emit(event, **data):
        original(event, **data)
        if event == 'responses_output_retry':
            stop.write_text('user pause')
    c.log.emit = emit
    with pytest.raises(ResponsesError):
        c.json('describe', {}, SCHEMA, 'description')
    assert len(sent) == 1
    assert c.budget_used() == 1


def test_output_retry_then_transport_retry_are_counted_once_each(tmp_path, monkeypatch):
    c, sent = client(tmp_path, ['bad', '503', '{"ok":true}'])
    c.retry_connections = True
    monkeypatch.setattr('time.sleep', lambda _: None)
    assert c.json('describe', {}, SCHEMA, 'description') == {'ok':True}
    events = read_events(c.log.path)
    assert len(sent) == 3 and c.budget_used() == 1
    assert sum(e['event'] == 'responses_retry' for e in events) == 1
    assert sum(e['event'] == 'responses_started' and bool(e.get('output_retry_number')) for e in events) == 1
