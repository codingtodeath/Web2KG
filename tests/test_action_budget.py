import pytest

from test_verified_stop_retry import client, VALID
from web2kg.common.activity import read_events
from web2kg.common.responses import ResponsesError
from web2kg.stage2.action_budget import POLICY, action_budget_used
from web2kg.stage2.actor_prompt import PROTOCOL


def call(c, *, search=True):
    return c.call([{'role': 'user', 'content': 'same observation'}], system_prompt=PROTOCOL,
                  action_budget_policy=POLICY, ax_search_available=search)


def test_search_and_selector_do_not_steal_final_action_slot(tmp_path):
    outputs = ['ACTION: click [7]'] * 28 + ['ACTION: search_ax [target]'] * 3 + [
        'ACTION: optimal_navigate [find target] [reddit] [target]', '{"target_page":"None"}',
        'ACTION: click [8]', VALID]
    c, sent = client(tmp_path, outputs)
    c.max_calls = 31
    for _ in range(28):
        call(c)
    for _ in range(3):
        call(c)
    assert action_budget_used(read_events(c.log.path)) == 28
    call(c)
    c.call([], system_prompt='Select target_page', action_budget_policy=POLICY)
    call(c)
    assert call(c) == VALID
    assert len(sent) == 35 and action_budget_used(read_events(c.log.path)) == 31
    events = read_events(c.log.path)
    assert sum(e['event'] == 'responses_received' for e in events) == 35
    assert sum(e.get('usage', {}).get('input_tokens', 0) for e in events if e['event']=='responses_received') == 350
    with pytest.raises(ResponsesError, match='call budget exhausted'):
        call(c)
    assert len(sent) == 35


def test_search_after_format_correction_credits_only_one_logical_request(tmp_path):
    c, sent = client(tmp_path, ['ACTION: click [7]\nACTION: click [8]',
                                'ACTION: search_ax [target]', VALID])
    call(c)
    assert len(sent) == 2 and action_budget_used(read_events(c.log.path)) == 0
    assert call(c) == VALID
    assert action_budget_used(read_events(c.log.path)) == 1


def test_exhausted_search_is_bounded_regeneration_not_free_infinite_requests(tmp_path):
    c, sent = client(tmp_path, ['ACTION: search_ax [target]'])
    with pytest.raises(ResponsesError, match='AX search is exhausted'):
        call(c, search=False)
    assert len(sent) == 4
    assert action_budget_used(read_events(c.log.path)) == 1
    assert not any(e['event']=='action_budget_search' for e in read_events(c.log.path))


def test_legacy_baseline_keeps_its_shared_call_limit(tmp_path):
    c, sent = client(tmp_path, ['ACTION: search_ax [target]', VALID])
    c.call([], system_prompt=PROTOCOL)
    with pytest.raises(ResponsesError, match='call budget exhausted'):
        c.call([], system_prompt=PROTOCOL)
    assert len(sent) == 1
    assert not any('action_budget_policy' in e for e in read_events(c.log.path))


@pytest.mark.parametrize('persistent', [False, True])
def test_transport_retry_and_search_credit_keep_physical_usage(tmp_path, monkeypatch, persistent):
    c, sent = client(tmp_path, ['503', 'ACTION: search_ax [target]', VALID])
    c.retry_connections = persistent
    c.max_retries = 1
    monkeypatch.setattr('time.sleep', lambda _: None)
    call(c)
    assert action_budget_used(read_events(c.log.path)) == 0
    assert call(c) == VALID
    events = read_events(c.log.path)
    assert len(sent) == 3 and sum(e['event']=='responses_failed' for e in events) == 1
    assert action_budget_used(events) == 1
