import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_verified_stop_retry import client
from scripts.qwen_hard_transport import chat_request
from web2kg.common.activity import read_events
from web2kg.common.responses import ResponsesError
from web2kg.evaluation.actor_output import validate_navigation_domain
from web2kg.stage2.actor_prompt import PROTOCOL, instruction


def rejected_outputs():
    for batch, task in [('reddit-hard22-feedback-flash-20261106-v1', 615),
                        ('reddit-hard22-feedback-flash-20261106-v1', 617),
                        ('reddit-hard8-cardfix-flash-20261106-v1', 618)]:
        for path in sorted((Path('runs')/batch/str(task)/'llm').glob('*/parsed.txt')):
            text = path.read_text(encoding='utf-8')
            if 'optimal_navigate [' in text:
                yield pytest.param(text, id=f'{task}-{path.parent.name[:8]}')


@pytest.mark.parametrize('rejected', list(rejected_outputs()))
def test_saved_domain_failures_receive_feedback_before_dispatch(tmp_path, rejected):
    valid = 'ACTION: optimal_navigate [Need a forum page] [reddit] [relevant forum]'
    c, sent = client(tmp_path, [rejected, valid])
    obs = [{'role': 'user', 'content': 'Original observation and current IDs'}]
    before = copy.deepcopy(obs)
    assert c.call(obs, system_prompt=PROTOCOL, navigation_domains=('reddit',)) == valid
    assert obs == before and sent[1]['input'][:-2] == sent[0]['input']
    wire = chat_request(sent[1])
    assert wire['messages'][-2]['content'][0]['text'] == rejected
    feedback = wire['messages'][-1]['content'][0]['text']
    assert "Available navigation site IDs: ['reddit']" in feedback
    assert 'not a hostname or URL' in feedback and 'No action from it was executed' in feedback
    events = read_events(c.log.path)
    assert sum(e['event'] == 'responses_output_parsed' for e in events) == 1
    assert c.budget_used() == 1 and len(sent) == 2


def test_domain_limit_latest_feedback_and_next_observation_isolation(tmp_path):
    bad = 'ACTION: optimal_navigate [page] [localhost:10599] [forum]'
    other = bad.replace('localhost:10599', 'https://wrong.invalid')
    c, sent = client(tmp_path, [bad, other, 'ACTION: click [7]', 'ACTION: go_back'])
    c.max_calls = 2
    obs = [{'role': 'user', 'content': 'page'}]
    assert c.call(obs, system_prompt=PROTOCOL, navigation_domains=('reddit',)) == 'ACTION: click [7]'
    assert sent[2]['input'][-2]['content'][0]['text'] == other
    assert len(sent[2]['input']) == len(sent[0]['input']) + 2
    assert c.call(obs, system_prompt=PROTOCOL, navigation_domains=('reddit',)) == 'ACTION: go_back'
    assert sent[3]['input'] == sent[0]['input'] and c.budget_used() == 2
    c, sent = client(tmp_path/'cap', [bad])
    with pytest.raises(ResponsesError, match='Available navigation site IDs'):
        c.call(obs, system_prompt=PROTOCOL, navigation_domains=('reddit',))
    assert len(sent) == 3 and c.budget_used() == 1
    assert not any(e['event'] == 'responses_output_parsed' for e in read_events(c.log.path))


def test_format_domain_share_budget_and_transient_keeps_same_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr('time.sleep', lambda _: None)
    bad = 'ACTION: optimal_navigate [page] [host] [forum]'
    c, sent = client(tmp_path, ['ACTION: go_back\nACTION: go_back', bad, '503', 'ACTION: click [1]'])
    assert c.call([], system_prompt=PROTOCOL, navigation_domains=('reddit',)) == 'ACTION: click [1]'
    assert sent[2]['input'] == sent[3]['input']
    assert c.budget_used() == 1 and len(sent) == 4
    c, sent = client(tmp_path/'cap', [bad]); c.configure_output_retries(0)
    with pytest.raises(ResponsesError): c.call([], system_prompt=PROTOCOL, navigation_domains=('reddit',))
    assert len(sent) == 1


def test_registry_changes_empty_registry_and_baseline_selector_unchanged(tmp_path):
    valid = 'ACTION: optimal_navigate [page] [gitlab] [issues]'
    assert validate_navigation_domain(valid, ('gitlab', 'reddit')) is None
    with pytest.raises(ResponsesError, match='Available navigation site IDs: \\[\\]'):
        validate_navigation_domain(valid, ())
    for value in ['ACTION: optimal_navigate [] [reddit] [page]',
                  'ACTION: optimal_navigate [why] [] [page]']:
        with pytest.raises(ResponsesError): validate_navigation_domain(value, ('reddit',))
    c, sent = client(tmp_path, [valid])
    assert c.call([], system_prompt='original baseline', navigation_domains=('reddit',)) == valid
    assert len(sent) == 1
    c, sent = client(tmp_path/'selector', ['target_page: r'])
    assert c.call([], system_prompt='target_page', navigation_domains=()) == 'target_page: r'


def test_prompt_uses_actual_live_registry_not_task_specific_aliases():
    actor = SimpleNamespace(config=SimpleNamespace(navigation_command=['optimal_navigate'], output=['action']),
                            domain_navigators={'reddit': object(), 'gitlab': object(), 'disabled': None})
    prompt = instruction(actor)
    assert "Available navigation site IDs: ['gitlab', 'reddit']" in prompt
    assert 'not a hostname or URL' in prompt and 'disabled' not in prompt
