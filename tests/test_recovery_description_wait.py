from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest

from test_execution_background import Delayed, step_to_new
from test_function_cards import saved
from test_llm_exploration import FakeClient
from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog
from web2kg.stage2.graph_update import ExecutionGraph
from web2kg.stage2.recovery import Recovery
from web2kg.stage2.navigation import RouteNavigator


@pytest.mark.parametrize('historical_summary', [False, True])
def test_diagnosis_waits_only_when_page_summary_depends_on_pending_transition(
        saved, site, tmp_path, historical_summary):
    log = EventLog(tmp_path / 'task')
    model = Delayed(EventLog(tmp_path / 'model'))
    waiting = threading.Event()
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(saved[0], browser, log, description_client=model, finalize_assets=False)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            before, action, after = step_to_new(updater, browser)
            assert model.entered.wait(2) and not model.release.is_set()
            records = [{'source': before['id'], 'target': after['id'], 'annotation': {
                'status': 'completed', 'before_page_description': 'Historical before',
                'after_page_description': 'Historical after'}}] if historical_summary else []
            original_poll = updater.poll_descriptions
            def poll(*, wait_for=(), **kwargs):
                if wait_for:
                    waiting.set()
                return original_poll(wait_for=wait_for, **kwargs)
            updater.poll_descriptions = poll
            def diagnose(prompt, payload, schema, purpose):
                assert purpose == 'root_cause_diagnosis'
                assert all(page['summary'] for page in payload['pages'])
                if historical_summary:
                    assert not waiting.is_set() and not model.release.is_set()
                    assert updater.description_jobs.pending
                    assert payload['trajectory'][0]['transition_effect']['status'] == 'pending'
                    assert 'effect' not in payload['trajectory'][0]['transition_effect']
                else:
                    assert waiting.is_set() and model.release.is_set()
                    assert not updater.description_jobs.pending
                    assert payload['trajectory'][0]['transition_effect']['effect'] == 'Observed change'
                return dict(verdict='uncertain', root_step=0, reason='Insufficient evidence', alternative='Inspect the current page.')
            diagnostic = SimpleNamespace(json=Mock(side_effect=diagnose))
            recovery = Recovery('Inspect item', updater.graph, None, log, diagnostic,
                historical_records=records, page_summarizer=updater.summarize_recovery_page,
                prepare_evidence=updater.poll_descriptions)
            recovery.start(before)
            recovery.record(before, action, after, transition_attempt=updater.count)
            # Model the repeated unchanged action trigger without submitting more descriptions.
            for _ in range(2):
                recovery.record(after, action, after)
            if not historical_summary:
                def release_on_dependency():
                    assert waiting.wait(5), 'diagnosis never awaited its page-summary dependency'
                    model.release.set()
                release = pool.submit(release_on_dependency)
            assert not recovery.check(after)
            diagnostic.json.assert_called_once()
            assert updater.recovery_summary_calls == 0
            if not historical_summary:
                release.result(timeout=5)
        finally:
            model.release.set()
            pool.shutdown(wait=True)
            updater.finish()


def test_real_diagnosis_returns_from_dead_end_via_home_and_records_actual_jump(saved, site, tmp_path):
    log = EventLog(tmp_path / 'task')
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(saved[0], browser, log, description_client=FakeClient(log), finalize_assets=False)
        nav = RouteNavigator(browser, saved[0], log)
        nav.recorder = updater
        updater.navigator = nav
        diagnostic = SimpleNamespace(json=Mock(return_value=dict(
            verdict='deviated', root_step=1, reason='The chosen item has no controls.',
            alternative='Inspect another item from the catalog.')))
        recovery = Recovery('Inspect another item', updater.graph, nav, log, diagnostic,
                            page_summarizer=updater.summarize_recovery_page)
        def record(before, action, after, **kwargs):
            recovery.record(before, action, after, execution_step=1, **kwargs)
        updater.on_transition = nav.on_url_jump = record
        updater.action_guard = nav.action_guard = recovery.guard
        try:
            before = browser.observe()
            recovery.start(before)
            updater.observe(before)
            updater.capture_before(before)
            action = next(a for a in before['actions'] if a['name'] == 'Red mug')
            browser.page.route(site + '/item/2', lambda route: route.fulfill(
                status=200, content_type='text/html', body='<h1>Unavailable item</h1>'))
            browser.act(action)
            dead_end = browser.observe()
            updater.record(before, action, dead_end)
            assert nav.graph.path(dead_end['id'], before['id']) is None
            assert recovery.check(dead_end)
            assert recovery.return_result == {'success': True, 'steps': 2, 'route': 'root_entry'}
            assert browser.observe()['id'] == before['id']
            assert recovery.refund_steps(1) == 1
            returned = [row for row in recovery.trace if row['branch'] == 'return']
            assert len(returned) == 2
            assert returned[0]['source'] == dead_end['id']
            assert returned[0]['target'] == saved[0].data['root']
            assert returned[0]['action']['kind'] == 'url_jump'
            assert returned[1]['target'] == before['id']
            assert recovery.guard(before, action)
            assert not recovery.check(before)
            diagnostic.json.assert_called_once()
        finally:
            updater.finish()
