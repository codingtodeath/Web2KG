import json

import pytest
import requests

from scripts.qwen_method_transport import clients
from web2kg.common.activity import read_events
from web2kg.common.responses import ResponsesError, object_schema
from web2kg.common.storage import EventLog
from web2kg.evaluation import webnavigator


@pytest.mark.parametrize('override', [None, 'qwen3.7-flash'])
def test_fixed_bailian_uses_direct_transport_and_truthful_identity(tmp_path, monkeypatch, override):
    env = tmp_path / 'test.env'
    env.write_text('CRAWL_API_KEY=test\nCRAWL_BASE_URL=https://unused.test\nCRAWL_MODEL=unused\n'
        'OPENROUTER_API_KEY=test\nOPENROUTER_BASE_URL=https://openrouter.test/v1\nOPENROUTER_MODEL=qwen-free\n'
        'BAILIAN_API_KEY=test\nBAILIAN_BASE_URL=https://bailian.test/v1\nBAILIAN_MODEL=qwen-paid\n')
    cls = clients(env, tmp_path / 'routing.json', provider='bailian', model_override=override)
    client = cls(env, EventLog(tmp_path / 'task'), 1, 4096)
    assert client.identity['model'] == client.model == (override or 'qwen-paid')
    assert client.identity['endpoint'] == 'https://bailian.test/v1/chat/completions'
    assert client.session.http.trust_env is False
    called = []
    def post(self, url, **kwargs):
        assert kwargs['json']['model'] == (override or 'qwen-paid')
        called.append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': '{"ok":true}'}}]}).encode()
        return response
    monkeypatch.setattr(requests.Session, 'post', post)
    assert client.json('test', {}, object_schema({'ok': {'type': 'boolean'}}), 'check') == {'ok': True}
    assert called == ['https://bailian.test/v1/chat/completions']


def test_qwen_initializes_real_actor_without_changing_api_model(tmp_path):
    from web2kg.stage2.agent import configuration, load_runtime, namespace
    runtime = load_runtime()
    from webNavigator.prompts import webnavigator_prompt
    settings = configuration(EventLog(tmp_path), model='qwen/qwen3.8-27b:free')
    agent = runtime.webNavigator(namespace(settings['agent']), {
        k: v for k, v in vars(webnavigator_prompt).items() if isinstance(v, dict)})
    agent.sites = ['reddit']
    agent.objective = 'Inspect a catalog item'
    agent.online_url = 'http://localhost/'
    agent.init_actor()
    assert agent.actor.model_family == 'gpt'
    assert agent.actor.config.model == agent.actor.config.selector.model == 'qwen/qwen3.8-27b:free'


def test_qwen_action_budget_survives_chat_conversion(tmp_path, monkeypatch):
    from web2kg.stage2.action_budget import POLICY, action_budget_used
    from web2kg.stage2.actor_prompt import PROTOCOL
    env = tmp_path / 'test.env'
    env.write_text('CRAWL_API_KEY=test\nCRAWL_BASE_URL=https://unused.test\nCRAWL_MODEL=unused\n'
        'OPENROUTER_API_KEY=test\nOPENROUTER_BASE_URL=https://unused.test/v1\nOPENROUTER_MODEL=unused\n'
        'BAILIAN_API_KEY=test\nBAILIAN_BASE_URL=https://bailian.test/v1\nBAILIAN_MODEL=qwen3.7-flash\n')
    cls = clients(env, tmp_path/'routing.json', provider='bailian', model_override='qwen3.7-flash')
    c = cls(env, EventLog(tmp_path/'task'), 1, 4096, verified_protocol=False)
    outputs = iter(['ACTION: search_ax [target]', '{"target_page":"None"}', 'ACTION: stop [done]'])
    sent = []
    def post(self, url, **kwargs):
        sent.append(kwargs['json'])
        response = requests.Response(); response.status_code = 200
        response._content = json.dumps({'model':'qwen3.7-flash', 'choices':[{'finish_reason':'stop',
            'message':{'content':next(outputs)}}], 'usage':{'prompt_tokens':10,'completion_tokens':5}}).encode()
        return response
    monkeypatch.setattr(requests.Session, 'post', post)
    c.call([], system_prompt=PROTOCOL, action_budget_policy=POLICY, ax_search_available=True)
    c.call([], system_prompt='Select target_page', action_budget_policy=POLICY)
    assert c.call([], system_prompt=PROTOCOL, action_budget_policy=POLICY) == 'ACTION: stop [done]'
    assert action_budget_used(read_events(c.log.path)) == 1 and len(sent) == 3
    assert all(set(wire)=={'model','messages','max_tokens','stream'} for wire in sent)


def test_flash_output_retry_uses_fresh_request_same_evidence(tmp_path, monkeypatch):
    env = tmp_path / 'test.env'
    env.write_text('CRAWL_API_KEY=test\nCRAWL_BASE_URL=https://unused.test\nCRAWL_MODEL=unused\n'
        'OPENROUTER_API_KEY=test\nOPENROUTER_BASE_URL=https://free.test/v1\nOPENROUTER_MODEL=unused\n'
        'BAILIAN_API_KEY=test\nBAILIAN_BASE_URL=https://bailian.test/v1\nBAILIAN_MODEL=unused\n')
    cls = clients(env, tmp_path / 'routing.json', provider='bailian', model_override='qwen3.7-flash')
    c = cls(env, EventLog(tmp_path / 'calls'), 1, 4096)
    c.configure_output_retries(3)
    sent = []
    def post(self, url, **kwargs):
        sent.append(kwargs['json'])
        response = requests.Response(); response.status_code = 200
        output = '{"summary":"title "quoted" inside"}' if len(sent) == 1 else '{"summary":"observed"}'
        response._content = json.dumps({'model':'qwen3.7-flash', 'choices':[{'finish_reason':'stop',
            'message':{'content':output}}], 'usage':{'prompt_tokens':10,'completion_tokens':5}}).encode()
        return response
    monkeypatch.setattr(requests.Session, 'post', post)
    assert c.json('describe', {'same':'page'}, object_schema({'summary':{'type':'string'}}), 'description') == {'summary':'observed'}
    assert len(sent) == 2 and sent[0]['messages'][1:] == sent[1]['messages'][1:]
    events = read_events(c.log.path)
    assert sum(e['event'] == 'responses_output_retry' for e in events) == 1
    assert c.budget_used() == 1


@pytest.mark.parametrize('status,message,provider', [
    (404, 'This model is unavailable for free.', 'bailian'),
    (429, 'Rate limit exceeded: free-models-per-day', 'bailian'),
    (429, 'Temporary rate limit', 'openrouter'),
    (503, 'Service unavailable', 'openrouter'),
    (401, 'Unauthorized', None),
    (200, 'malformed', None),
])
def test_structured_qwen_routing_and_strict_output(tmp_path, monkeypatch, status, message, provider):
    env = tmp_path / 'test.env'
    env.write_text('CRAWL_API_KEY=test\nCRAWL_BASE_URL=https://unused.test\nCRAWL_MODEL=unused\n'
        'OPENROUTER_API_KEY=test\nOPENROUTER_BASE_URL=https://openrouter.test/v1\nOPENROUTER_MODEL=qwen-free\n'
        'BAILIAN_API_KEY=test\nBAILIAN_BASE_URL=https://bailian.test/v1\nBAILIAN_MODEL=qwen-paid\n')
    sent = []
    def post(self, url, **kwargs):
        sent.append((url, kwargs['json']))
        response = requests.Response()
        response.status_code = status if len(sent) == 1 else 200
        data = {'error': {'message': message}} if response.status_code != 200 else {
            'model': kwargs['json']['model'], 'choices': [{'finish_reason': 'stop', 'message': {
                'content': '{"summary":"observed"}' if message != 'malformed' else 'not JSON'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        response._content = json.dumps(data).encode()
        return response
    monkeypatch.setattr(requests.Session, 'post', post)
    monkeypatch.setattr('time.sleep', lambda _: None)
    original = webnavigator.OriginalResponses
    cls = clients(env, tmp_path / 'routing.json')
    assert webnavigator.OriginalResponses is original
    client = cls(env, EventLog(tmp_path / 'task'), 1, 2048, retry_connections=True)
    if provider is None:
        with pytest.raises(ResponsesError):
            client.json('Summarize recorded evidence', {'page': 'test'}, object_schema({'summary': {'type': 'string'}}), 'diagnosis')
        assert len(sent) == 1
    else:
        assert client.json('Summarize recorded evidence', {'page': 'test'}, object_schema({'summary': {'type': 'string'}}), 'diagnosis') == {'summary': 'observed'}
        assert len(sent) == 2 and client.provider == provider and client.budget_used() == 1
        assert sent[0][1]['messages'] == sent[1][1]['messages']
    assert 'properties:\n' in sent[0][1]['messages'][0]['content']
    assert 'page: test' in sent[0][1]['messages'][1]['content'][0]['text']
    events = read_events(client.log.path)
    assert sum(e['event'] == 'provider_request' for e in events) == len(sent)
