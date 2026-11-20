import copy
import hashlib
import json
import threading

import pytest

from test_function_cards import saved
from test_llm_exploration import FakeClient
from web2kg.common.browser import Browser
from web2kg.common.responses import ResponsesError
from web2kg.common.storage import EventLog, Graph
from web2kg.stage2.execution_descriptions import prepare
from web2kg.stage2.graph_update import ExecutionGraph
from web2kg.stage2.navigation import RouteNavigator


class Delayed(FakeClient):
    model = 'fixture'

    def __init__(self, log, *, fail=False, replayable=True):
        super().__init__(log, fail)
        self.entered, self.release = threading.Event(), threading.Event()
        self.threads = []
        self.replayable = replayable

    def json(self, system, payload, schema, purpose, images=()):
        self.threads.append(threading.get_ident())
        self.calls.append((purpose, copy.deepcopy(payload)))
        self.entered.set()
        assert self.release.wait(30), 'test did not release model job'
        if self.fail:
            raise ResponsesError('fixture failure', kind='http', http_status=401)
        assert len(images) == 2 and all(p.exists() for p in images)
        return dict(replayable=self.replayable, effect='Observed change', uncertainties=[],
                    control_evidence=['B1'], page_descriptions={
                        side: 'Summary ' + side for side in payload['page_descriptions_requested']})


def step_to_new(updater, browser):
    before = browser.observe(); updater.observe(before); updater.capture_before(before)
    action = next(a for a in before['actions'] if a['name'] == 'Green mug')
    browser.act(action); after = browser.observe(); updater.record(before, action, after)
    return before, action, after


def test_actor_can_continue_and_pending_route_waits_for_only_model_decision(saved, site, tmp_path):
    base = saved[0]; owner = threading.get_ident()
    original = (base.directory / 'graph.json').read_bytes()
    log = EventLog(tmp_path / 'task'); model = Delayed(EventLog(tmp_path / 'model'))
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(base, browser, log, description_client=model, finalize_assets=False)
        nav = RouteNavigator(browser, base, log); nav.recorder = updater; updater.navigator = nav
        try:
            before, action, after = step_to_new(updater, browser)
            assert model.entered.wait(2)
            first = updater.graph.data['transition_records'][0]
            assert first['annotation']['status'] == 'pending'
            assert updater.graph.path(before['id'], after['id']) is None
            assert nav.graph.path(before['id'], after['id']) is None
            disk = Graph.load(updater.graph.directory)
            assert after['id'] in disk.nodes and not disk.data['edges'][-1]['replayable']
            # A second actual browser action and submission proceed while the first model is blocked.
            updater.capture_before(after)
            browser.page.go_back(); browser.settle(); returned = browser.observe()
            back = dict(id='browser-back', kind='go_back', role='browser', name='go_back')
            updater.record(after, back, returned)
            assert len(updater.description_jobs.pending) == 2
            assert updater.graph.data['transition_records'][1]['page_description_requests'] == {}
            assert all(t != owner for t in model.threads)
            assert all((updater.graph.directory / p).exists() for p in first['images'])
            # Facts alone did not authorize replay; finishing the judgment does.
            model.release.set()
            updater.wait_for_route([before['id']], after['id'])
            assert updater.graph.path(before['id'], after['id'])
            assert nav.graph.path(before['id'], after['id'])
            assert updater.graph.nodes[after['id']]['page_description']['text'] == 'Summary after'
            assert first['annotation']['control_evidence'][0]['ref'] == 'before:' + action['id']
            assert first['annotation']['control_evidence'][0]['short_ref'] == 'B1'
            updater.finish()
        finally:
            model.release.set(); updater.description_jobs.close()
    assert (base.directory / 'graph.json').read_bytes() == original


@pytest.mark.parametrize('failure', [False, True])
def test_denied_or_failed_description_never_opens_edge_and_finish_drains(saved, site, tmp_path, failure):
    log = EventLog(tmp_path / 'task'); model = Delayed(EventLog(tmp_path / 'model'), fail=failure, replayable=False)
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(saved[0], browser, log, description_client=model, finalize_assets=False)
        try:
            before, _, after = step_to_new(updater, browser)
            model.release.set()
            if failure:
                with pytest.raises(ResponsesError):
                    updater.poll_descriptions(all_pending=True)
                # A navigation adapter catching the first exception must not clear the failure.
                with pytest.raises(ResponsesError):
                    updater.poll_descriptions()
            else:
                updater.poll_descriptions(all_pending=True)
            assert updater.graph.path(before['id'], after['id']) is None
        finally:
            model.release.set()
            result = updater.finish()
        assert result['description_stopped'] is failure
        assert updater.description_jobs.closed and not updater.description_jobs.pending
        assert len(model.calls) == 1


def test_short_evidence_covers_all_supplied_controls_and_does_not_request_existing_summaries():
    before = dict(id='node-long-id', title='Before', url='http://local/a', text='', actions=[
        dict(id=f'long-hash-{i}', kind='click', role='link', name=f'Link {i}') for i in range(70)])
    after = {**before, 'id': 'other-node', 'title': 'After'}
    record = dict(outcome='transition', phase='agent', action_attempted=True, error=None)
    payload, schema, mapping = prepare(before, before['actions'][40], after, record, {},
                                       {'before': 'Saved B', 'after': 'Saved A'})
    assert mapping['B1'] == 'before:long-hash-40' and mapping['A70'] == 'after:long-hash-69'
    assert 'long-hash' not in json.dumps(payload)
    assert 'node-long-id' not in json.dumps(payload)
    assert set(mapping) == set(schema['properties']['control_evidence']['items']['enum'])
    assert {r['ref'] for r in payload['provided_evidence']} == set(mapping)
    assert payload['page_descriptions_requested'] == []
    assert schema['properties']['page_descriptions']['properties'] == {}


def test_repeat_pending_transition_has_one_job_and_shared_decision(saved, site, tmp_path):
    log = EventLog(tmp_path / 'task'); model = Delayed(EventLog(tmp_path / 'model'))
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(saved[0], browser, log, description_client=model, finalize_assets=False)
        try:
            before, action, after = step_to_new(updater, browser)
            updater.record(before, {**action, 'ax_node_id': 'another-live-id'}, after)
            updater.record(before, action, after, error='fixture_click_failure')
            assert len(updater.description_jobs.pending) == 1 and updater.description_calls == 1
            model.release.set(); updater.poll_descriptions(all_pending=True)
            assert all(r['annotation']['status'] == 'completed' for r in updater.graph.data['transition_records'])
            assert not updater.graph.data['transition_records'][-1]['replayable']
            assert updater.graph.data['transition_records'][-1]['outcome'] == 'failed'
            assert len(model.calls) == 1
            # Recovery consumes the completed page summary without generating another one.
            assert updater.summarize_recovery_page(after)['summary'] == 'Summary after'
            assert updater.recovery_summary_calls == 0
        finally:
            model.release.set(); updater.finish()


def test_final_index_overlaps_cards_using_immutable_node_snapshot(saved, tmp_path, monkeypatch):
    updater = ExecutionGraph(saved[0], None, EventLog(tmp_path / 'task'))
    entered, cards_done = threading.Event(), threading.Event()
    owner = threading.get_ident()
    def index():
        assert threading.get_ident() != owner
        assert updater._index_snapshot.data is not updater.graph.data
        entered.set()
        assert cards_done.wait(5)
        assert 'test_owner_change' not in updater._index_snapshot.data
        return {'status': 'completed'}
    def cards(decision, allow_calls):
        assert threading.get_ident() == owner and entered.wait(5)
        updater.graph.data['test_owner_change'] = True
        cards_done.set()
        return {'status': 'disabled'}, {'status': 'disabled'}, 0
    monkeypatch.setattr(updater, 'finish_index', index)
    monkeypatch.setattr(updater, 'finish_knowledge', cards)
    result = updater.finish()
    assert result['index_status'] == 'completed'
    assert 'test_owner_change' in Graph.load(updater.graph.directory).data


def test_pending_input_is_blocked_then_becomes_actor_handoff(saved, site, tmp_path):
    log = EventLog(tmp_path / 'task'); model = Delayed(EventLog(tmp_path / 'model'), replayable=False)
    with Browser(site, log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(saved[0], browser, log, description_client=model, finalize_assets=False)
        nav = RouteNavigator(browser, saved[0], log); nav.recorder = updater; updater.navigator = nav
        try:
            browser.page.set_content('<main><h1>Editor</h1><label>Text<input></label></main>')
            before = browser.observe(); updater.observe(before); updater.capture_before(before)
            field = next(a for a in before['actions'] if a['role'] == 'textbox')
            action = {**field, 'kind': 'type', 'press_enter': False}
            browser.page.get_by_role('textbox').fill('fixture value')
            after = browser.observe(); updater.record(before, action, after)
            edge = updater.graph.data['edges'][-1]
            assert not edge['replayable'] and not updater.graph.edge_replayable(edge)
            browser.page.get_by_role('textbox').fill('')
            assert nav.replay([edge])['reason'] == 'transition_not_replayable'
            model.release.set(); updater.poll_descriptions(all_pending=True)
            assert updater.graph.edge_replayable(edge)
            assert nav.replay([edge])['status'] == 'requires_input'
            assert browser.page.get_by_role('textbox').input_value() == ''
        finally:
            model.release.set(); updater.finish()


def test_backpressure_failure_keeps_current_facts_without_dangling_job(saved, site, tmp_path, monkeypatch):
    log = EventLog(tmp_path / 'task'); model = Delayed(EventLog(tmp_path / 'model'), fail=True)
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(saved[0], browser, log, description_client=model,
            description_workers=1, max_pending_descriptions=1, finalize_assets=False)
        try:
            before, _, after = step_to_new(updater, browser)
            updater.capture_before(after)
            browser.page.go_back(); browser.settle(); returned = browser.observe()
            original_save = updater.save
            def saved_facts(**kwargs):
                original_save(**kwargs)
                if updater.count == 2:
                    row = json.loads((updater.graph.directory / 'transitions/00002.json').read_text())
                    assert row['action_attempted'] and row['outcome'] == 'transition'
                    assert all((updater.graph.directory / p).exists() for p in row['images'])
                    model.release.set()
            monkeypatch.setattr(updater, 'save', saved_facts)
            with pytest.raises(ResponsesError):
                updater.record(after, dict(id='back', kind='go_back', role='browser', name='go_back'), returned)
            assert updater.description_calls == 1
            assert not updater.description_jobs.pending
            assert all(r['annotation']['status'] == 'failed' for r in updater.graph.data['transition_records'])
            assert not updater.graph.path(after['id'], before['id'])
            assert len(model.calls) == 1
        finally:
            model.release.set(); updater.finish()


def test_queued_descriptions_reserve_budget_and_recovery_cannot_oversubscribe(saved, site, tmp_path):
    log = EventLog(tmp_path / 'task'); model = Delayed(EventLog(tmp_path / 'model'))
    model.max_calls = 1
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(saved[0], browser, log, description_client=model, finalize_assets=False)
        try:
            before, _, after = step_to_new(updater, browser)
            unrelated = next(k for k in updater.graph.nodes if k not in (before['id'], after['id']))
            with pytest.raises(ValueError, match='budget exhausted'):
                updater.summarize_recovery_page({'id': unrelated})
            updater.capture_before(after)
            browser.page.go_back(); browser.settle()
            updater.record(after, dict(id='back', kind='go_back', role='browser', name='go_back'), browser.observe())
            assert updater.graph.data['transition_records'][-1]['annotation']['reason'] == 'call_budget'
            assert updater.description_calls == 1
        finally:
            model.release.set(); updater.finish()


def test_multiple_completed_jobs_have_one_graph_checkpoint(saved, site, tmp_path, monkeypatch):
    log = EventLog(tmp_path / 'task'); model = Delayed(EventLog(tmp_path / 'model'))
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(saved[0], browser, log, description_client=model, finalize_assets=False)
        try:
            before, _, after = step_to_new(updater, browser)
            updater.capture_before(after); browser.page.go_back(); browser.settle()
            updater.record(after, dict(id='back', kind='go_back', role='browser', name='go_back'), browser.observe())
            checkpoints = []
            original_save = updater.save
            def count_save(**kwargs):
                checkpoints.append(1); original_save(**kwargs)
            monkeypatch.setattr(updater, 'save', count_save)
            model.release.set(); updater.poll_descriptions(all_pending=True)
            assert len(checkpoints) == 1
            assert len(model.calls) == 2
        finally:
            model.release.set(); updater.finish()


def test_stop_skips_queued_request_but_preserves_inflight_result(saved, site, tmp_path):
    class Stoppable(Delayed):
        def stopped(self):
            return self.stop.is_set()
    log = EventLog(tmp_path / 'task'); model = Stoppable(EventLog(tmp_path / 'model'))
    model.stop = threading.Event()
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(saved[0], browser, log, description_client=model,
            description_workers=1, max_pending_descriptions=2, finalize_assets=False)
        try:
            before, _, after = step_to_new(updater, browser)
            assert model.entered.wait(2)
            updater.capture_before(after); browser.page.go_back(); browser.settle()
            updater.record(after, dict(id='back', kind='go_back', role='browser', name='go_back'), browser.observe())
            model.stop.set(); model.release.set()
        finally:
            model.release.set(); result = updater.finish()
        rows = updater.graph.data['transition_records']
        assert [r['annotation']['status'] for r in rows] == ['completed', 'failed']
        assert len(model.calls) == 1 and result['description_stopped']
        assert not updater.description_jobs.pending


def test_real_actor_second_decision_does_not_wait_for_description(saved, site, tmp_path):
    import re
    from test_execution_assets import indexed_copy
    from web2kg.stage2.agent import run_task
    from web2kg.stage2.template_context import TemplateKnowledge
    base, cards, jina = indexed_copy(saved, tmp_path)
    log = EventLog(tmp_path / 'task'); model = Delayed(EventLog(tmp_path / 'model'))
    knowledge = TemplateKnowledge(base, cards, log)
    class Actor:
        calls = 0
        def call(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                key = re.search(r"link \[(\d+)\] 'Green mug'", json.dumps(messages))[1]
                return f'REASON: Inspect item.\nACTION: click [{key}]'
            assert model.entered.wait(2) and not model.release.is_set()
            model.release.set()
            return 'REASON: Finished.\nACTION: stop [done]'
    try:
        result = run_task(base, 'Inspect item', log, Actor(), jina, knowledge=knowledge,
            start_url=site + '/catalog', update_graph=True, description_client=model)
        assert result['status'] == 'agent_stopped' and result['answer'] == 'done'
        assert not result.get('execution_graph_error'), result
        assert not result['execution_graph']['description_stopped']
        graph = Graph.load(log.directory / 'execution-graph')
        assert graph.data['transition_records'][0]['annotation']['status'] == 'completed'
    finally:
        model.release.set()
