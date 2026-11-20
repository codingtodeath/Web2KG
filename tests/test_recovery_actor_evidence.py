import copy
import json

import pytest

from test_recovery import decision, setup_recovery


def add_transition(recovery, page, attempt, annotation, **extra):
    row = dict(attempt=attempt, source=page['id'], target=page['id'],
               action=copy.deepcopy(page['actions'][0]), annotation=annotation, **extra)
    recovery.graph.data.setdefault('transition_records', []).append(row)
    return row


def test_repeated_page_keeps_each_actor_summary_and_exact_transition_effect(tmp_path):
    recovery, page, model, _ = setup_recovery(tmp_path, [decision(False)])
    for attempt in (1, 2):
        add_transition(recovery, page, attempt, dict(status='completed', effect=f'Effect {attempt}',
                                                    uncertainties=['Not independently verified']))
        with recovery.actor_step(page['id'], f'Actor observation {attempt}'):
            recovery.record(page, page['actions'][0], page, execution_step=attempt, transition_attempt=attempt)
    assert not recovery.check(page)
    payload = model.calls[0][1]
    for i, row in enumerate(payload['trajectory'], 1):
        keys = list(row)
        assert keys.index('source_page') == keys.index('source') + 1
        assert keys.index('transition_effect') == keys.index('action') + 1
        assert row['source_page']['summary'] == 'Recorded item directory.'
        assert row['source_page']['actor_observation']['text'] == f'Actor observation {i}'
        assert row['source_page']['actor_observation']['timing'] == 'before_action'
        assert row['transition_effect']['effect'] == f'Effect {i}'
        assert row['transition_effect']['kind'] == 'model_interpretation'
        assert row['transition_effect']['uncertainties'] == ['Not independently verified']
        assert 'control' not in row and 'highlight' not in json.dumps(row)
    assert json.loads((recovery.directory/'diagnosis-1-input.json').read_text(encoding='utf-8')) == payload


@pytest.mark.parametrize('status', ['pending', 'failed', 'not_requested'])
def test_unavailable_effect_does_not_leak_stale_or_historical_explanation(tmp_path, status):
    recovery, page, _, _nav = setup_recovery(tmp_path, [])
    add_transition(recovery, page, 1, dict(status='completed', effect='Earlier visit'))
    add_transition(recovery, page, 2, dict(status=status, effect='Do not use this', reason='fixture'))
    recovery.record(page, page['actions'][0], page, transition_attempt=2)
    effect = recovery.diagnosis_trajectory()[0]['transition_effect']
    assert effect['status'] == status and 'effect' not in effect


def test_ready_jobs_are_collected_before_payload_without_requesting_a_wait(tmp_path):
    recovery, page, model, _ = setup_recovery(tmp_path, [decision(False)])
    row = add_transition(recovery, page, 1, dict(status='pending'))
    polled = []
    def poll():
        polled.append(True)
        row['annotation'] = dict(status='completed', effect='Newly received result')
    recovery.prepare_evidence = poll
    for _ in range(2):
        recovery.record(page, page['actions'][0], page, transition_attempt=1)
    assert not recovery.check(page)
    assert polled == [True]
    assert model.calls[0][1]['trajectory'][0]['transition_effect']['effect'] == 'Newly received result'


def test_navigation_summary_only_belongs_to_start_and_never_automatic_return(tmp_path):
    recovery, page, _, _nav = setup_recovery(tmp_path, [])
    other = {**page, 'id': 'b', 'url': 'http://test/other'}
    with recovery.actor_step(page['id'], 'At the navigation start'):
        recovery.record(page, page['actions'][0], other, phase='replay')
        recovery.record(other, page['actions'][0], page, phase='replay')
        recovery.record(page, page['actions'][0], other, phase='replay')
    recovery.replaying = True
    recovery.record(other, page['actions'][0], page, phase='replay')
    assert recovery.trace[0]['actor_observation'] == 'At the navigation start'
    assert all('actor_observation' not in row for row in recovery.trace[1:])
    assert all(row['transition_effect']['status'] == 'not_recorded' for row in recovery.diagnosis_trajectory())


def test_scope_clears_on_error_and_mismatched_source_cannot_inherit_summary(tmp_path):
    recovery, page, _, _nav = setup_recovery(tmp_path, [])
    with pytest.raises(RuntimeError):
        with recovery.actor_step(page['id'], 'Failed turn'):
            raise RuntimeError('fixture')
    recovery.record(page, page['actions'][0], page)
    with recovery.actor_step('different-state', 'Wrong state summary'):
        recovery.record(page, page['actions'][0], page)
    assert all('actor_observation' not in row for row in recovery.trace)


def test_effect_reuse_is_explicit_and_invalid_link_is_not_matched_by_state(tmp_path):
    recovery, page, _, _nav = setup_recovery(tmp_path, [])
    record = add_transition(recovery, page, 1, dict(status='completed', effect='Known transition'),
                            description_reused={'request_id': 'previous-request'})
    recovery.record(page, page['actions'][0], page, transition_attempt=1)
    assert recovery.diagnosis_trajectory()[0]['transition_effect']['reused'] is True
    record['action']['name'] = 'Different control'
    assert recovery.diagnosis_trajectory()[0]['transition_effect'] == {'status': 'not_recorded'}


def test_prompt_projection_is_bounded_without_overwriting_saved_actor_text(tmp_path):
    recovery, page, _, _nav = setup_recovery(tmp_path, [])
    add_transition(recovery, page, 1, dict(status='completed', effect='e' * 1100))
    with recovery.actor_step(page['id'], 'a' * 1100):
        recovery.record(page, page['actions'][0], page, transition_attempt=1)
    row = recovery.diagnosis_trajectory()[0]
    assert row['source_page']['actor_observation']['truncated']
    assert len(row['source_page']['actor_observation']['text']) == 1000
    assert row['transition_effect']['truncated'] and len(row['transition_effect']['effect']) == 1000
    assert len(recovery.trace[0]['actor_observation']) == 1100
