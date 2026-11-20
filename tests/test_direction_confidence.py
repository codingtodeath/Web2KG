import json
import re

import pytest

from web2kg.stage2.confidence import DirectionConfidence, parse_confidence
from web2kg.stage2.actor_prompt import OUTPUT_FIELDS, output_specifications
from web2kg.stage2.action_notes import parse_sections
from web2kg.stage2.agent import run_task
from web2kg.stage2.recovery import Recovery
from web2kg.common.storage import EventLog
from test_execution_routes import graph
from test_recovery import setup_recovery, decision


@pytest.mark.parametrize('raw', ['', None, True, 'nan', 'inf', '-0.1', '1.1', '80%', '.8 because'])
def test_invalid_confidence_is_unknown(raw):
    assert parse_confidence(raw) is None


def fields(score, action='go_back'):
    return {'direction confidence': score, 'action': action}


def test_optional_field_does_not_change_default_actor_or_action_parser():
    assert len(OUTPUT_FIELDS) == 4
    keys = [*OUTPUT_FIELDS, 'direction confidence']
    assert 'DIRECTION CONFIDENCE:' in output_specifications(keys)
    parsed = parse_sections('ACTION: stop [done]\nDIRECTION CONFIDENCE: 0.7', keys)
    assert parsed['action'] == 'stop [done]' and parse_confidence(parsed['direction confidence']) == .7


def test_four_cases_reset_and_cooldown_without_product_of_entire_history():
    p = DirectionConfidence(dict(mode='active'))
    for _ in range(30):
        assert not p.propose(fields(.9))['would_trigger']
        p.commit(fields(.9))
    assert p.propose(fields(.3))['would_trigger'] == ['direction_confidence_low']
    p.mark_diagnosis()
    for _ in range(3):
        assert not p.propose(fields(.2))['would_trigger']
        p.commit(fields(.2))
    assert p.propose(fields(.2))['would_trigger']
    p.mark_diagnosis()
    for v in [.8, .55, .55]:
        p.commit(fields(v))
    signal = p.propose(fields(.55))
    assert signal['would_trigger'] == ['direction_confidence_window']
    assert p.propose(fields(.9))['would_trigger'] == []


def test_unknown_breaks_three_consecutive_score_window_and_periodic_counts_only_commits():
    p = DirectionConfidence(dict(mode='shadow'))
    for v in [.5, None]:
        p.commit(fields(v))
    assert p.propose(fields(.5))['would_trigger'] == []
    for _ in range(50):
        p.propose(fields(.2))  # Retries/discarded proposals don't count as executed actions.
    assert p.completed == 2 and p.periodic_reasons() == []
    for _ in range(8):
        p.commit(fields(.9))
    assert p.periodic_reasons() == ['ten_actions_without_diagnosis']
    p.mark_diagnosis()
    assert p.periodic_reasons() == [] and p.segment == []


def test_extra_trigger_cannot_bypass_diagnosis_cap_or_turn_pending_action_into_history(tmp_path):
    r, page, model, _ = setup_recovery(tmp_path, [decision(False)])
    r.triggers = lambda _: []
    pending = fields(.2, 'click [7]')
    assert not r.check(page, extra_triggers=['direction_confidence_low'], pending_action=pending)
    payload = model.calls[0][1]
    assert payload['trajectory'] == [] and payload['eligible_root_steps'] == []
    assert payload['pending_actor_proposal']['action'] == 'click [7]'
    assert 'NOT executed' in payload['trigger_notice']
    r.cap = 1
    r.check(page, extra_triggers=['direction_confidence_low'])
    assert len(model.calls) == 1


@pytest.mark.parametrize('mode,verdict,expected_calls,expected_steps', [
    ('off', 'on_track', 11, 11),
    ('shadow', 'on_track', 11, 11),
    ('active', 'on_track', 1, 1),
    ('active', 'uncertain', 2, 1)])
def test_live_policy_periodic_and_preaction_advice(graph, tmp_path, monkeypatch,
                                                mode, verdict, expected_calls, expected_steps):
    monkeypatch.setattr(Recovery, 'triggers', lambda self, observed: [])
    monkeypatch.setattr(Recovery, 'prepare_pages', lambda self: None)
    class Diagnosis:
        calls = []
        def json(self, prompt, payload, schema, role):
            self.calls.append(payload)
            return dict(verdict=verdict, root_step=0, reason='No observed contradiction.',
                        alternative='Check current evidence before stopping.')
    diagnosis = Diagnosis()
    class Actor:
        calls = 0
        def call(self, messages, **kwargs):
            self.calls += 1
            action = 'go_back' if mode in ('off', 'shadow') and self.calls <= 10 else 'stop [No result]'
            return ('OBSERVATION DESCRIPTION: Current page is visible.\nREASON: Check the task.\n'
                    f'ACTION: {action}\nDIRECTION CONFIDENCE: 0.2')
    actor = Actor()
    log = EventLog(tmp_path/'run')
    result = run_task(graph, 'Inspect items', log, actor, None,
        max_steps=12, update_graph=True, finalize_graph_assets=False,
        max_descriptions=0, max_cards=0, diagnosis_client=diagnosis,
        direction_confidence=dict(mode=mode) if mode != 'off' else None)
    assert result['status'] == 'agent_stopped', result
    assert actor.calls == expected_calls and result['steps'] == expected_steps
    assert len(diagnosis.calls) == 1
    events = [json.loads(s) for s in log.path.read_text(encoding='utf-8').splitlines()]
    committed = [e for e in events if e['event'] == 'direction_confidence_committed']
    assert len(committed) == (0 if mode == 'off' else expected_steps)
    if mode in ('off', 'shadow'):
        assert diagnosis.calls[0]['triggers'] == ['ten_actions_without_diagnosis']
        if mode == 'off':
            assert sum(e['event'] == 'diagnosis_action_count' for e in events) == 11
            cfg = json.loads((log.directory/'execution-config.json').read_text(encoding='utf-8'))
            assert 'direction confidence' not in cfg['agent']['actor']['output']
            assert cfg['diagnosis_watchdog_actions'] == 10
    else:
        assert 'pending_actor_proposal' in diagnosis.calls[0]
        assert not diagnosis.calls[0]['trajectory']
        assert any(e['event'] == 'confidence_proposal_discarded' for e in events) == (verdict == 'uncertain')


def test_preaction_return_refreshes_observation_without_executing_old_proposal(graph, tmp_path, monkeypatch):
    monkeypatch.setattr(Recovery, 'triggers', lambda self, observed: [])
    monkeypatch.setattr(Recovery, 'prepare_pages', lambda self: None)
    class Diagnosis:
        def json(self, prompt, payload, schema, role):
            assert payload['pending_actor_proposal']['action'].startswith('stop')
            return decision(True, 1)
    class Actor:
        calls = 0
        def call(self, messages, **kwargs):
            self.calls += 1
            text = json.dumps(messages)
            if self.calls == 1:
                target = re.search(r"link \[(\d+)\] 'Red mug'", text)[1]
                return f'ACTION: click [{target}]\nDIRECTION CONFIDENCE: 0.9'
            if self.calls == 2:
                return 'ACTION: stop [Discard this proposed answer]\nDIRECTION CONFIDENCE: 0.2'
            assert 'Green mug' in text  # Real returned catalog, not old detail observation.
            return 'ACTION: stop [After review]\nDIRECTION CONFIDENCE: 0.8'
    actor = Actor()
    log = EventLog(tmp_path/'run')
    result = run_task(graph, 'Inspect the green item', log, actor, None, max_steps=4,
        update_graph=True, max_descriptions=0, max_cards=0, finalize_graph_assets=False,
        diagnosis_client=Diagnosis(), direction_confidence=dict(mode='active'))
    assert result['status'] == 'agent_stopped' and result['answer'] == 'After review', result
    assert actor.calls == 3 and result['steps'] == 2 and result['refunded_steps'] == 1
    events = [json.loads(s) for s in log.path.read_text(encoding='utf-8').splitlines()]
    assert len([e for e in events if e['event'] == 'direction_confidence_committed']) == 2
    trace = json.loads((log.directory/'trajectory.json').read_text(encoding='utf-8'))
    assert all('Discard this proposed answer' not in r.get('action', '') for r in trace)


def test_shadow_without_diagnosis_records_low_scores_past_watchdog_without_intervening(graph, tmp_path, monkeypatch):
    def unexpected_recovery(*args, **kwargs):
        raise AssertionError('Shadow-only sampling must not create a diagnosis controller')
    monkeypatch.setattr(Recovery, '__init__', unexpected_recovery)
    class Actor:
        calls = 0
        def call(self, messages, **kwargs):
            self.calls += 1
            action = 'go_back' if self.calls <= 10 else 'stop [Finished sampling]'
            return f'ACTION: {action}\nDIRECTION CONFIDENCE: 0.2'
    actor = Actor()
    log = EventLog(tmp_path/'run')
    result = run_task(graph, 'Inspect items', log, actor, None, max_steps=12,
        update_graph=True, max_descriptions=0, max_cards=0, finalize_graph_assets=False,
        diagnosis_client=None, direction_confidence=dict(mode='shadow'))
    assert result['status'] == 'agent_stopped' and result['steps'] == actor.calls == 11
    events = [json.loads(s) for s in log.path.read_text(encoding='utf-8').splitlines()]
    signals = [e for e in events if e['event'] == 'direction_confidence']
    committed = [e for e in events if e['event'] == 'direction_confidence_committed']
    assert len(signals) == len(committed) == 11
    assert all(e['score'] == .2 and e['would_trigger'] for e in signals)
    assert [e['actor_action'] for e in committed] == list(range(1, 12))
    assert not any(e['event'] in ('confidence_proposal_discarded', 'direction_confidence_reset') for e in events)
    cfg = json.loads((log.directory/'execution-config.json').read_text(encoding='utf-8'))
    assert 'direction confidence' in cfg['agent']['actor']['output']
    assert 'diagnosis_watchdog_actions' not in cfg


def test_active_without_diagnosis_rejected_before_loading_runtime(monkeypatch, tmp_path):
    def unexpected_runtime():
        raise AssertionError('Invalid policy must fail before runtime side effects')
    monkeypatch.setattr('web2kg.stage2.agent.load_runtime', unexpected_runtime)
    with pytest.raises(ValueError, match='Active direction confidence requires diagnosis'):
        run_task(None, 'Inspect items', EventLog(tmp_path/'run'), None, None,
                 direction_confidence=dict(mode='active'))
