import copy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from test_verified_stop_retry import client, VALID
from web2kg.common.activity import read_events
from web2kg.common.responses import ResponsesError
from web2kg.common.storage import Graph, EventLog
from web2kg.evaluation.actor_output import validate_single_action
from web2kg.evaluation.verified_answer import stop_instruction
from web2kg.stage2.actor_prompt import PROTOCOL, instruction
from web2kg.stage2.agent import COMMANDS, run_task


@pytest.mark.parametrize('text', [
    'ACTION: click [7]\nACTION: stop [done]',
    'ACTION: click [7]\nNOTE: waiting\nACTION: click [8]',
    'ACTION: click [7]\nACTION: click [7]',
    'ACTION: click [7]\nstop [done]',
    'ACTION: click [7]; stop [done]',
    'ACTION: click [7] stop [done]',
    'ACTION: type [8] [hello] [0]\nclick [7]',
    'REASON FOR ACTION: click [7]',
    'ACTION: note [memo]',
])
def test_multiple_or_missing_actions_rejected_before_agent(text):
    with pytest.raises(ResponsesError) as e:
        validate_single_action(text)
    assert e.value.kind == 'output:actor_action_invalid'


@pytest.mark.parametrize('action', [
    'click [7]', 'go_back', 'search_ax [submit]',
    'type [8] [first\nACTION: literal text\nNOTE: more text] [0]',
    'stop [{"task_type":"RETRIEVE","status":"SUCCESS","retrieved_data":[{"text":"click [7]"}]}]',
    'optimal_navigate [Find form] [reddit] [Submission form]',
])
def test_nested_action_text_and_short_evidence_are_not_extra_commands(action):
    validate_single_action('REASON: Next.\nACTION: ' + action + '\nNOTE: Observed fact.')


def test_prompt_has_one_contract_and_keeps_user_type_spec():
    actor = SimpleNamespace(config=SimpleNamespace(navigation_command=COMMANDS,
        output=['repeat final object', 'mode analysis', 'reason', 'action', 'observation highlight']))
    p = Path('third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt')
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    prompt = stop_instruction(instruction(actor))
    assert PROTOCOL in prompt and prompt.count('\nACTION:') == 1
    assert 'without any markdown formatting, headers' not in prompt and 'punish' not in prompt
    assert 'at least once' not in prompt and 'min 50 words' not in prompt
    assert 'after an operation, wait for the next real observation' in prompt
    assert p.read_text(encoding='utf-8').strip() in prompt
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('bad', ['ACTION: click [7]\nACTION: stop [done]', 'ACTION: click [7]\nclick [8]'])
def test_same_request_regeneration_keeps_budget_and_all_usage(tmp_path, bad):
    c, sent = client(tmp_path, [bad, 'ACTION: click [7]'])
    obs = [{'role': 'user', 'content': 'same observation id7'}]
    original = copy.deepcopy(obs)
    assert c.call(obs, system_prompt=PROTOCOL) == 'ACTION: click [7]'
    assert len(sent) == 2 and sent[1]['input'][:-2] == sent[0]['input'] and obs == original
    assert sent[1]['input'][-2] == {'role':'assistant','content':[{'type':'input_text','text':bad}]}
    assert 'No action from it was executed' in sent[1]['input'][-1]['content'][0]['text']
    assert 'exactly ONE ACTION' in sent[1]['instructions'] and 'No action from that response was executed' in sent[1]['instructions']
    assert c.budget_used() == 1
    events = read_events(c.log.path)
    assert len([e for e in events if e['event'] == 'responses_received']) == 2
    assert len([e for e in events if e['event'] == 'responses_output_parsed']) == 1
    assert sum(e.get('usage', {}).get('input_tokens', 0) for e in events if e['event']=='responses_received') == 20


def test_action_retry_does_not_force_stop_but_stop_retry_stays_stop(tmp_path):
    c, sent = client(tmp_path, ['ACTION: click [7]\nACTION: stop [done]', 'ACTION: stop [done]', 'ACTION: click [7]', VALID])
    assert c.call([{'role':'user','content':'same'}], system_prompt=PROTOCOL) == VALID
    assert len(sent) == 4 and c.budget_used() == 1
    assert all(p['input'][:-2] == sent[0]['input'] for p in sent[1:])
    assert [p['input'][-2]['content'][0]['text'] for p in sent[1:]] == [
        'ACTION: click [7]\nACTION: stop [done]', 'ACTION: stop [done]', 'ACTION: click [7]']


def test_exhaustion_and_stop_never_spend_browser_actions(tmp_path):
    c, sent = client(tmp_path, ['ACTION: click [7]\nACTION: click [8]'])
    with pytest.raises(ResponsesError) as e:
        c.call([{'role':'user','content':'same'}], system_prompt=PROTOCOL)
    assert e.value.kind == 'output:actor_action_invalid' and len(sent) == 4 and c.budget_used() == 1
    c.stop_file = tmp_path / 'STOP'; c.stop_file.write_text('pause')
    with pytest.raises(ResponsesError):
        c.call([{'role':'user','content':'same'}], system_prompt=PROTOCOL)
    assert len(sent) == 4


def test_legacy_actor_and_selector_do_not_opt_into_new_validation(tmp_path):
    multiple = 'ACTION: click [7]\nACTION: stop [done]'
    c, sent = client(tmp_path, [multiple])
    assert c.call([{'role':'user','content':'same'}], system_prompt='Legacy Actor') == multiple
    assert len(sent) == 1
    c, sent = client(tmp_path/'selector', ['{"target_page":"r"}'])
    assert c.call([{'role':'user','content':'query'}], system_prompt='target_page') == '{"target_page":"r"}'
    assert len(sent) == 1


def test_transport_retry_is_separate_from_action_regeneration(tmp_path, monkeypatch):
    monkeypatch.setattr('time.sleep', lambda _: None)
    c, sent = client(tmp_path, ['ACTION: click [7]\nACTION: click [8]', '503', 'ACTION: click [7]'])
    assert c.call([{'role':'user','content':'same'}], system_prompt=PROTOCOL) == 'ACTION: click [7]'
    events = read_events(c.log.path)
    assert len(sent) == 3 and c.budget_used() == 1
    assert sum(e['event'] == 'responses_retry' for e in events) == 1
    assert sum(e['event'] == 'responses_output_retry' for e in events) == 1
    assert sent[1] == sent[2]  # Connection retry resends the exact corrected request.


def test_conflicting_messages_are_format_errors_not_transport_errors(tmp_path):
    c, sent = client(tmp_path, ['unused'])
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs['json'])
        texts = ['ACTION: click [7]', VALID] if len(calls) == 1 else ['ACTION: click [7]']
        return SimpleNamespace(ok=True, json=lambda: dict(status='completed',model='fixture',
            usage={'input_tokens':10,'output_tokens':5},output=[{'type':'message','role':'assistant',
            'content':[{'type':'output_text','text':text}]} for text in texts]))
    c.session.post = post
    assert c.call([{'role':'user','content':'same'}], system_prompt=PROTOCOL) == 'ACTION: click [7]'
    assert len(calls) == 2 and c.budget_used() == 1


@pytest.mark.parametrize('always_bad', [False, True])
def test_real_actor_invalid_response_does_not_advance_observation_or_steps(site, tmp_path, always_bad):
    c, unused = client(tmp_path, [])
    c.max_calls = 2
    c.log = EventLog(tmp_path / 'run')
    sent = []
    def post(*args, **kwargs):
        request = copy.deepcopy(kwargs['json']); sent.append(request)
        if len(sent) <= 2 or always_bad:
            control = re.search(r"link \[(\d+)\] 'Catalog'", json.dumps(request['input']))[1]
            value = f'ACTION: click [{control}]'
            if len(sent) == 1 or always_bad:
                value += '\nNOTE: Ready to click.\n' + VALID
        else:
            value = VALID
        return SimpleNamespace(ok=True, json=lambda: dict(status='completed',model='fixture',
            usage={'input_tokens':10,'output_tokens':5},output=[{'type':'message','role':'assistant',
            'content':[{'type':'output_text','text':value}]}]))
    c.session.post = post
    result = run_task(Graph(tmp_path/'graph'), 'Open catalog', c.log, c, None, use_graph=False,
        start_url=site, max_steps=2, final_check=lambda b: b.page.url == site + '/catalog')
    actions = [e for e in read_events(c.log.path) if e['event'] == 'agent_action']
    if always_bad:
        assert result['status'] == 'model_error' and result['steps'] == 0 and not actions, result
        assert len(sent) == 4 and c.budget_used() == 1
        assert all(p['input'][:-2] == sent[0]['input'] for p in sent[1:])
        return
    assert result['status'] == 'agent_stopped' and result['local_success'], result
    assert result['steps'] == 2 and len(sent) == 3 and c.budget_used() == 2
    assert sent[0]['input'] == sent[1]['input'][:-2]
    assert len(actions) == 2 and actions[0]['action'].startswith('click') and actions[1]['action'].startswith('stop')
