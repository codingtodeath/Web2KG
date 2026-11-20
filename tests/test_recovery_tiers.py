import copy
import json
import re

import pytest
import yaml

from test_recovery import setup_recovery, decision, Diagnostic
from test_function_cards import saved
from test_llm_exploration import FakeClient
from web2kg.common.storage import EventLog
from web2kg.stage2.agent import run_task
from web2kg.stage2.prompt_context import context_messages
from web2kg.stage2.template_context import TemplateKnowledge


def advice(verdict):
    return dict(verdict=verdict, root_step=0, reason='Current sort value is not recorded.',
                alternative='Check the visible sort value before changing it.')


@pytest.mark.parametrize('verdict', ['on_track', 'uncertain'])
def test_advice_reaches_actor_without_ban_return_refund_or_history_mutation(tmp_path, verdict):
    r, page, model, nav = setup_recovery(tmp_path, [advice(verdict)])
    for _ in range(2):
        r.record(page, page['actions'][0], page, execution_step=1)
    assert not r.check(page)
    assert not r.attempted and r.ban is None and r.root is None and r.return_result is None
    assert not nav.return_to_observed.called and not r.guard(page, page['actions'][0])
    assert r.refund_steps(2) == 0
    messages = [{'role': 'user', 'content': 'Current observation'}]
    original = copy.deepcopy(messages)
    sent = context_messages(messages, global_evidence=r.context(page))
    ctx = yaml.safe_load(sent[-1]['content'].split('\n', 1)[1])
    assert ctx['diagnosis_advice']['verdict'] == verdict
    assert ctx['diagnosis_advice']['alternative'] == advice(verdict)['alternative']
    assert 'counterfactual' not in ctx and messages == original
    assert not r.check(page) and len(model.calls) == 1
    data = json.loads((r.directory/'recovery.json').read_text())
    assert data['decision_policy'] == 'three-tier-v1'
    assert data['decisions'][0]['decision']['deviated'] is False


def test_latest_advice_replaces_older_and_definite_verdict_uses_original_return_budget(tmp_path):
    r, page, model, nav = setup_recovery(tmp_path,
        [advice('uncertain'), {**advice('on_track'), 'alternative': 'Continue inspecting items.'}, decision()])
    r.record(page, page['actions'][0], page)
    r.record(page, page['actions'][0], page)
    assert not r.check(page)
    r.record(page, page['actions'][0], page)
    assert not r.check(page)
    assert r.context(page)['diagnosis_advice']['verdict'] == 'on_track'
    assert 'sort' not in r.context(page)['diagnosis_advice']['alternative']
    r.record(page, page['actions'][0], page)
    assert r.check(page) and r.attempted
    assert nav.return_to_observed.call_count == 1 and r.guard(page, page['actions'][0])
    assert 'diagnosis_advice' not in r.context(page)
    assert not r.check(page) and len(model.calls) == 3


@pytest.mark.parametrize('bad', [
    {**advice('uncertain'), 'root_step': 1},
    {**advice('on_track'), 'root_step': 1},
    {**advice('deviated'), 'root_step': 0},
    {**advice('uncertain'), 'alternative': ''},
    {**advice('deviated'), 'root_step': 1, 'reason': ''},
    advice('maybe'),
    dict(deviated=True, root_step=1, reason='Legacy binary response', alternative='Other'),
])
def test_invalid_or_legacy_binary_decision_cannot_ban_or_replay(tmp_path, bad):
    r, page, _, nav = setup_recovery(tmp_path, [bad])
    for _ in range(2): r.record(page, page['actions'][0], page)
    assert not r.check(page) and r.stopped
    assert not r.attempted and r.ban is None and not nav.return_to_observed.called


@pytest.mark.parametrize('verdict', ['on_track', 'uncertain'])
def test_real_actor_can_use_same_control_after_nondefinite_diagnosis(saved, site, tmp_path, verdict):
    graph, _, _, cards, _ = saved
    log = EventLog(tmp_path/'task')
    diagnostic = Diagnostic([advice(verdict)])
    calls = []
    class Actor:
        def call(self, messages, **kwargs):
            calls.append(messages)
            if len(calls) == 4:
                ctx_text = next(m['content'] for m in messages
                    if isinstance(m['content'], str) and m['content'].startswith('Website framework and conditional experience'))
                ctx = yaml.safe_load(ctx_text.split('\n', 1)[1])
                assert ctx['diagnosis_advice']['verdict'] == verdict
                assert 'counterfactual' not in ctx
            if len(calls) == 5:
                return 'REASON: Finished inspecting.\nACTION: stop [done]'
            # Re-enter the exact same detail link after diagnosis; it must remain usable.
            name = 'Red mug' if len(calls) in (1, 4) else 'Catalog'
            ids = re.findall(r"link \[(\d+)\] '"+name+"'", json.dumps(messages))
            assert ids
            return f'REASON: Inspect observed item.\nACTION: click [{ids[-1]}]'
    result = run_task(graph, 'Inspect items', log, Actor(), None, start_url=site+'/catalog',
        knowledge=TemplateKnowledge(graph, cards, log), update_graph=True, finalize_graph_assets=False,
        description_client=FakeClient(EventLog(log.directory/'annotations')), max_descriptions=30,
        diagnosis_client=diagnostic, max_diagnoses=1, max_steps=8,
        final_check=lambda browser: browser.page.url == site+'/item/2')
    assert result['local_success'] and not result['recovery']['attempted']
    events = [json.loads(s) for s in log.path.read_text().splitlines()]
    assert not any(e['event'] in ('counterfactual_return', 'counterfactual_control_blocked',
                                 'counterfactual_step_refund') for e in events)

