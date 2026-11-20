import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from web2kg.common.jina import JinaClient
from web2kg.common.storage import EventLog
from web2kg.evaluation.actor_output import actor_output, single_action
from web2kg.evaluation.webnavigator import final_text
from test_webnavigator_adapter import message


@pytest.mark.parametrize('action', ['click [42]', 'go_back', 'note [remember this]',
    'type [15] [literal [nested] content] [0]', 'stop [{"result":["yes"]}]',
    'optimal_navigate [reason] [site] [query]'])
def test_bare_commands_keep_exact_content(action):
    value, changes = actor_output([message(action, 'final_answer')], final_text)
    assert value == 'ACTION:\n' + action
    assert changes == ['bare_action_heading']


@pytest.mark.parametrize('text', ['click [1]\nclick [2]', 'click [1] then submit',
    'I should click [1]', 'goto [https://example.com]', 'type [1] [text] [2]',
    'optimal_navigate [reason] [site]', 'click [1', 'click [1]]'])
def test_does_not_infer_or_expand_action_space(text):
    assert not single_action(text)
    value, changes = actor_output([message(text, 'final_answer')], final_text)
    assert value == text and changes == []


def test_fragmented_commentary_only_accepts_adjacent_heading_and_single_command():
    prefix = message('REASON:\nUse visible link.\nACTION:\n', 'commentary')
    tail = message('click [42]', 'commentary')
    empty = message('', 'final_answer')
    value, changes = actor_output([prefix, tail, empty], final_text)
    assert value.endswith('ACTION:\nclick [42]')
    assert changes == ['adjacent_action_continuation']
    for messages in ([tail, prefix, empty], [prefix, empty, tail],
                     [message('ACTION: click [1]', 'commentary'), tail, empty]):
        with pytest.raises(ValueError):
            actor_output(messages, final_text)
    tail['status'] = 'in_progress'
    with pytest.raises(ValueError):
        actor_output([prefix, tail, empty], final_text)


@pytest.mark.parametrize('flag', [0, 1])
def test_named_flag_reaches_original_action_parser_without_corrupting_text(flag):
    from web2kg.stage2.agent import load_runtime
    load_runtime()
    actions = importlib.import_module('browser_env.actions')
    content = 'Price $1000; literal [press_enter_after=1] stays'
    raw = f'REASON: enter text\nACTION:\ntype [15] [{content}] [press_enter_after={flag}]\nOBSERVATION HIGHLIGHT:\n15'
    text, changes = actor_output([message(raw, 'final_answer')], final_text)
    command = text.split('ACTION:\n')[1].split('\nOBSERVATION')[0].strip()
    expected = actions.create_id_based_actions(f'type [15] [{content}] [{flag}]')[0]
    actual = actions.create_id_based_actions(command)[0]
    assert actual['flag'] is bool(flag) and actual['element_id'] == '15'
    assert np.array_equal(actual['text'], expected['text'])
    assert changes == ['named_enter_flag']


def test_unmodified_output_and_conflicting_final_remain_strict():
    text = 'REASON: observed\nACTION:\nclick [1]\nNOTE: keep this'
    assert actor_output([message(text, 'final_answer')], final_text) == (text, [])
    with pytest.raises(ValueError):
        actor_output([message('click [1]', 'final_answer'), message('click [2]', 'final_answer')], final_text)


def test_local_encoder_timeout_retries_only_encoding_with_same_identity(tmp_path, monkeypatch):
    identity = {'fingerprint': 'fixture'}
    log = EventLog(tmp_path)
    client = JinaClient(log=log, timeout=600, retry_connections=True)
    calls = []
    monkeypatch.setattr('web2kg.common.jina.time.sleep', lambda _: None)
    def post(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise requests.ReadTimeout('fixture')
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'encoder': identity, 'data': [{'embeddings': [[1.] * 128]}], 'request_id': 'fixture'})
    client.http.post = post
    assert client.encode('same text', 'text', identity).shape == (1, 128)
    assert calls[0] == calls[1] and calls[0]['timeout'] == (10, 600)
    assert 'jina_retry' in log.path.read_text()
    with pytest.raises(ValueError, match='differs'):
        client.encode('same text', 'text', {'fingerprint': 'other'})
    assert len(calls) == 3


def test_local_encoder_permanent_error_and_stop_do_not_retry(tmp_path):
    client = JinaClient(retry_connections=True, stop_file=tmp_path/'STOP')
    calls = []
    def post(*a, **k):
        calls.append(1)
        raise requests.HTTPError(response=SimpleNamespace(status_code=400))
    client.http.post = post
    with pytest.raises(requests.HTTPError):
        client.encode('text', 'text', {})
    (tmp_path/'STOP').touch()
    with pytest.raises(RuntimeError, match='cancelled'):
        client.encode('text', 'text', {})
    assert calls == [1]


def test_named_flag_types_exact_text_in_real_browser_without_submitting(site,tmp_path):
    import json
    from web2kg.stage2.agent import load_runtime, configuration, namespace
    load_runtime()
    from webNavigator.env import WebArenaEnvironmentWrapper
    config=tmp_path/'browser.json'
    config.write_text(json.dumps({'task_id':1,'sites':['reddit'],'intent':'Synthetic form check',
        'start_url':site+'/catalog','storage_state':None,'allowed_origin':site}))
    env=WebArenaEnvironmentWrapper(config_file=str(config),observation_type='accessibility_tree',
        current_viewport_only=False,headless=True,global_config=namespace(configuration(EventLog(tmp_path))))
    try:
        import re
        field=re.search(r'\[(\d+)\] searchbox',env.obs['text'][0])[1]
        content='literal $1000 and [brackets]'
        value,_=actor_output([message(f'type [{field}] [{content}] [press_enter_after=0]','final_answer')],final_text)
        env.step(value.split('ACTION:\n')[1])
        assert env.webarena_env.page.get_by_role('searchbox').input_value()==content
        assert env.webarena_env.page.url==site+'/catalog'
    finally:
        env.close()
