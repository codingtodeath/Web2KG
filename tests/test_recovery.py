import copy
import json
import yaml
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_function_cards import saved
from test_llm_exploration import FakeClient
from web2kg.stage1.function_cards import file_hash
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.stage1.site_framework import build_framework, framework_input, load_framework
from web2kg.stage2.recovery import Recovery, ExperienceStore, learn_experience, template_key, memory_namespace
from web2kg.stage2.agent import run_task
from web2kg.stage2.navigation import RouteNavigator
from web2kg.common.browser import Browser
from web2kg.common.responses import ResponsesError
from web2kg.stage2.graph_update import ExecutionGraph
from web2kg.stage2.template_context import TemplateKnowledge


class Diagnostic:
    model = 'fixture-sol'
    def __init__(self, decisions):
        self.decisions, self.calls = iter(decisions), []
    def json(self, prompt, payload, schema, purpose):
        self.calls.append((purpose, copy.deepcopy(payload)))
        if purpose == 'root_cause_diagnosis':
            return next(self.decisions)
        return {'condition': 'When searching for an item in a category', 'lesson': 'Check the visible item name before opening it.',
                'limitations': 'One successful local observation, not a universal rule.'}


def decision(deviated=True, step=1):
    return {'verdict': 'deviated' if deviated else 'on_track', 'root_step': step if deviated else 0,
            'reason': 'The original item is not the requested item.', 'alternative': 'Inspect another visible item.'}


def setup_recovery(tmp_path, decisions):
    graph = Graph(tmp_path / 'graph')
    page = {'id': 'a', 'url': 'http://test/', 'title': 'Directory', 'headings': ['Items'],
            'actions': [{'kind': 'click', 'role': 'link', 'name': 'Wrong item', 'xpath': '/a', 'href': 'http://test/wrong'}]}
    graph.add_node(page, [])
    model = Diagnostic(decisions)
    nav = SimpleNamespace(return_to_observed=Mock(return_value={'success': True, 'steps': 1}))
    recovery = Recovery('Find right item', graph, nav, EventLog(tmp_path / 'task'), model,
        page_summarizer=Mock(return_value={'summary': 'Recorded item directory.',
            'summary_source': {'kind': 'recovery_page_summary'}}))
    recovery.start(page)
    return recovery, page, model, nav


def test_arrivals_not_reads_and_negative_can_retrigger(tmp_path):
    recovery, page, model, nav = setup_recovery(tmp_path, [decision(False), decision()])
    for _ in range(5):
        recovery.context(page); recovery.check(page)
    assert not model.calls
    for _ in range(2):
        recovery.record(page, page['actions'][0], page)
    assert set(recovery.triggers(page)) == {'same_state_three_arrivals', 'same_action_two_no_changes'}
    assert not recovery.check(page)
    assert not recovery.check(page)
    assert len(model.calls) == 1
    recovery.record(page, page['actions'][0], page)
    assert recovery.check(page)
    assert nav.return_to_observed.call_args.args == ('a', 12)
    for _ in range(4):
        recovery.record(page, page['actions'][0], page); recovery.check(page)
    assert len(model.calls) == 2


@pytest.mark.parametrize('success', [True, False])
def test_refund_uses_execution_turn_not_atomic_trace_and_only_once(tmp_path, success):
    r, page, _, nav = setup_recovery(tmp_path, [decision(step=3)])
    nav.return_to_observed.return_value = {'success': success, 'steps': 2}
    for turn in (1, 1, 2, 3):
        r.record(page, page['actions'][0], page, execution_step=turn)
    assert r.check(page)
    # Returning adds physical actions, not Actor turns, and cannot earn another refund.
    r.replaying = True
    r.record(page, page['actions'][0], page, phase='replay', execution_step=3)
    r.replaying = False
    assert r.refund_steps(3) == (2 if success else 0)
    assert r.refund_steps(3) == 0
    data = json.loads((r.directory / 'recovery.json').read_text())
    assert len(data['trajectory']) == 5
    if success:
        assert data['step_refund']['budget_steps'] == 1
        assert data['step_refund']['actual_steps'] == 3
    else:
        assert data['step_refund'] is None


def test_no_controls_invalid_root_and_failed_return_no_retry(tmp_path):
    r, page, model, nav = setup_recovery(tmp_path, [decision(step=99)])
    empty = {**page, 'id': 'b', 'actions': []}
    r.record(page, page['actions'][0], empty)
    assert r.triggers(empty) == ['no_interactive_controls']
    assert not r.check(empty) and r.stopped
    assert not nav.return_to_observed.called
    r, page, model, nav = setup_recovery(tmp_path / 'next', [decision()])
    nav.return_to_observed.return_value = {'success': False, 'reason': 'no_safe_route_within_cap'}
    for _ in range(2):
        r.record(page, page['actions'][0], page)
    assert r.check(page) and r.attempted
    assert not r.check(page) and len(model.calls) == 1


def test_ban_ignores_ax_ids_but_not_other_controls(tmp_path):
    r, page, _, _ = setup_recovery(tmp_path, [decision()])
    for _ in range(2):
        r.record(page, page['actions'][0], page)
    r.check(page)
    assert r.guard(page, {**page['actions'][0], 'ax_node_id': '999', 'id': 'new'})
    assert not r.guard(page, {**page['actions'][0], 'name': 'Other item', 'href': 'http://test/other'})
    assert not r.guard({**page, 'url': 'http://test/other'}, page['actions'][0])


def test_summary_is_seeded_from_real_transition_annotation(tmp_path):
    r, page, _, _ = setup_recovery(tmp_path, [])
    r.pages.clear()
    r.graph.data['transition_records'] = [{'source': 'a', 'annotation': {'before_page_description': 'Actual historical summary'}}]
    assert r.page(page)['summary'] is None
    r.prepare_pages()
    assert r.page(page)['summary'] == 'Actual historical summary'
    assert r.page(page)['summary_source']['field'] == 'before_page_description'
    r.page_summarizer.assert_not_called()


def test_diagnosis_refreshes_late_transition_and_completes_missing_page_first(tmp_path):
    r, page, model, _ = setup_recovery(tmp_path, [decision(False), decision(False)])
    empty = {**page, 'id': 'b', 'actions': [], 'text': 'An empty detail view'}
    r.record(page, page['actions'][0], empty)
    r.page_summarizer.assert_not_called()
    # The record arrived after the initial page was cached.
    r.graph.data['transition_records'] = [{'source': 'a', 'annotation': {
        'status': 'completed', 'before_page_description': 'Late actual source summary'}}]
    def summarize(observed):
        assert not model.calls
        assert observed['id'] == 'b' and observed['text'] == 'An empty detail view'
        return {'summary': 'Empty detail view.', 'summary_source': {'kind': 'recovery_page_summary'}}
    r.page_summarizer.side_effect = summarize
    assert not r.check(empty)
    payload = model.calls[0][1]
    assert payload['pages'][0]['summary'] == 'Late actual source summary'
    assert payload['current']['summary'] == 'Empty detail view.'
    saved_input = json.loads((r.directory / 'diagnosis-1-input.json').read_text(encoding='utf-8'))
    assert saved_input == payload
    r.record(empty, {'kind': 'scroll', 'role': 'browser'}, empty)
    r.check(empty)
    assert r.page_summarizer.call_count == 1 and len(model.calls) == 2


@pytest.mark.parametrize('failure', [None, ValueError('budget exhausted'), RuntimeError('malformed summary')])
def test_incomplete_page_evidence_never_falls_back_to_headings(tmp_path, failure):
    r, page, model, _ = setup_recovery(tmp_path, [decision()])
    if failure is None:
        r.page_summarizer = None
    else:
        r.page_summarizer.side_effect = failure
    for _ in range(2):
        r.record(page, page['actions'][0], page)
    assert not r.check(page) and r.stopped
    assert not model.calls and r.pages['a']['summary'] is None
    events = [json.loads(s) for s in r.log.path.read_text(encoding='utf-8').splitlines()]
    assert events[-1]['reason'] == 'page_evidence_incomplete'


@pytest.mark.parametrize('outcome', ['valid', 'invalid', 'permanent_error'])
def test_page_backfill_uses_saved_evidence_and_shared_description_budget(saved, tmp_path, outcome):
    base, _, _, _, (_, catalog, _) = saved
    log = EventLog(tmp_path / 'task')
    descriptions = FakeClient(EventLog(log.directory / 'annotations'))
    descriptions.model = 'fixture-luna'
    def summarize(prompt, payload, schema, purpose):
        descriptions.calls.append((purpose, payload))
        assert purpose == 'recovery_page_summary'
        assert payload['ax_tree'] and payload['candidates']
        assert 'runtime_result' not in payload  # No fabricated Transition.
        if outcome == 'permanent_error':
            raise ResponsesError('Unauthorized', kind='http', http_status=401)
        return {'summary': 'A catalog of items with observed links.' if outcome == 'valid' else ''}
    descriptions.json = summarize
    updater = ExecutionGraph(base, None, log, description_client=descriptions, max_descriptions=1,
                             finalize_assets=False)
    model = Diagnostic([decision(False)])
    r = Recovery('Inspect item', updater.graph, None, log, model,
                 page_summarizer=updater.summarize_recovery_page)
    r.start(catalog)
    action = catalog['actions'][0]
    for _ in range(2):
        r.record(catalog, action, catalog)
    assert not descriptions.calls
    if outcome == 'permanent_error':
        with pytest.raises(ResponsesError):
            r.check(catalog)
    else:
        assert not r.check(catalog)
    assert updater.description_calls == updater.recovery_summary_calls == 1
    folder = r.directory / 'page-summaries' / catalog['id']
    assert (folder / 'input.json').is_file() and (folder / 'result.json').is_file()
    if outcome == 'valid':
        assert len(model.calls) == 1
        assert model.calls[0][1]['current']['summary_source']['request_id'] == 'fixture-request'
        # A later Transition cannot spend an extra description call.
        updater.record(catalog, action, catalog)
        assert updater.graph.data['transition_records'][-1]['annotation']['reason'] == 'facts_only_no_change'
        assert len(descriptions.calls) == 1
    else:
        assert not model.calls and r.stopped and updater.description_stopped
        assert json.loads((folder / 'result.json').read_text(encoding='utf-8'))['status'] == 'failed'


def test_success_gate_three_distinct_experiences_and_reuse(tmp_path):
    store = ExperienceStore(tmp_path / 'memory.json', 'same-graph')
    client = Diagnostic([])
    for i in range(3):
        folder = tmp_path / f'task{i}'
        r, page, _, _ = setup_recovery(folder, [decision()])
        r.task = f'Find distinct item {i}'
        for _ in range(2):
            r.record(page, page['actions'][0], page)
        r.check(page)
        r.record(page, {**page['actions'][0], 'name': 'Right item'}, {**page, 'id': 'correct'})
        path = r.directory / 'recovery.json'
        data = json.loads(path.read_text()); data['template_key'] = 'shared-structure'; write_json(path, data)
        out = learn_experience(r.log.directory, client, store, success=True, verifier='fixture DOM assertion')
        assert out['status'] == 'completed'
        assert learn_experience(r.log.directory, client, store, success=True, verifier='fixture DOM assertion') == out
    assert [p for p, _ in client.calls].count('page_experience') == 3
    assert [p for p, _ in client.calls].count('template_experience') == 1
    assert store.context('a', 'shared-structure')['template']
    assert store.context('a', 'shared-structure')['page_total'] == 3
    denied = learn_experience(tmp_path / 'unscored', client, store, success=False, verifier='')
    assert denied['reason'] == 'no_independent_success'
    assert len(client.calls) == 4
    with pytest.raises(ValueError, match='namespace'):
        ExperienceStore(store.path, 'other-graph').load()


def test_framework_has_all_templates_compact_edges_and_provenance(saved, tmp_path):
    base, path, cards, _, _ = saved
    layer = json.loads(path.read_text())
    payload = framework_input(base, layer, cards)
    assert len(payload['templates']) == len(layer['templates'])
    assert 'evidence_nodes' not in json.dumps(payload)
    assert all(set(r) == {'source', 'target', 'kind', 'role'} for r in payload['transitions'])
    model = SimpleNamespace(model='fixture-luna', json=Mock(return_value={
        'overview': 'Catalog and items', 'areas': [{'name': 'Catalog', 'purpose': 'Find items', 'templates': [layer['templates'][0]['id']]}],
        'navigation': ['Follow actual links'], 'limitations': ['Partial observation']}))
    # Publish a fresh local graph copy, not the shared module fixture.
    import shutil
    shutil.copytree(base.directory, tmp_path / 'graph')
    graph = Graph.load(tmp_path / 'graph')
    cp = graph.directory / 'cards/cards.json'; write_json(cp, cards)
    build_framework(graph, cp, model, graph.directory / 'site-framework/test', publish=True)
    assert load_framework(graph)['framework']['overview'] == 'Catalog and items'
    child = Graph(tmp_path / 'child')
    shutil.copytree(graph.directory, child.directory)
    child.data = copy.deepcopy(graph.data)
    child.data['execution_update'] = {'source_graph': str(graph.directory), 'source_sha256': file_hash(graph.directory / 'graph.json')}
    child.save()
    child_layer = json.loads((child.directory / 'templates/test/templates.json').read_text())
    from web2kg.common.storage import digest
    child_layer['graph_digest'] = digest(child.data)
    write_json(child.directory / 'templates/test/templates.json', child_layer)
    write_json(child.directory / 'templates/active.json', {'directory': 'templates/test'})
    # Execution graphs save their initial summary separately and have no newly published framework.
    write_json(child.directory / 'initial-website-framework.json', load_framework(graph))
    (child.directory / 'site-framework/active.json').unlink()
    assert load_framework(child)['inherited_initial_view']
    write_json(cp, {**cards, 'changed': True})
    # Advisory summaries no longer reread historical cards to validate their ancestry.
    assert load_framework(graph)['framework']['overview'] == 'Catalog and items'
    assert load_framework(child)['framework']['overview'] == 'Catalog and items'


def test_real_agent_recovers_and_refreshes_ax_ids(saved, site, tmp_path):
    graph, _, _, cards, (_, catalog, product) = saved
    log = EventLog(tmp_path / 'run')
    model = Diagnostic([decision(step=1)])
    calls = []
    class Actor:
        def call(self, messages, **kwargs):
            text = json.dumps(messages)
            contexts = [m['content'] for m in messages if isinstance(m['content'], str)
                        and m['content'].startswith('Website framework and conditional experience')]
            context = yaml.safe_load(contexts[0].split('\n', 1)[1]) if contexts else {}
            calls.append(context)
            if context.get('counterfactual'):
                assert context['counterfactual']['return']['success']
                assert 'Wrong' not in context['counterfactual']['forbidden_control']['name']
                # A deliberately repeated banned control must be blocked by code.
                if len(calls) == 4:
                    name = 'Red mug'
                elif len(calls) == 5:
                    name = 'Green mug'
                else:
                    return 'REASON: Correct item inspected.\nACTION: stop [done]'
            else:
                name = 'Red mug' if len(calls) == 1 else 'Catalog'
            match = re.findall(r"link \[(\d+)\] '" + name + "'", text)
            assert match, (name, text[-1000:])
            return (f'OBSERVATION DESCRIPTION: Seen before decision {len(calls)}.\n'
                    f'REASON: Local recovery regression.\nACTION: click [{match[-1]}]')
    # Create a loop: catalog -> red -> catalog -> catalog (third arrival).
    result = run_task(graph, 'Inspect Green mug', log, Actor(), None, start_url=site + '/catalog',
        knowledge=TemplateKnowledge(graph, cards, log), update_graph=True, finalize_graph_assets=False,
        description_client=FakeClient(EventLog(log.directory / 'annotations')), max_descriptions=30,
        diagnosis_client=model, final_check=lambda b: b.page.url == site + '/item/3', max_steps=3)
    assert result['status'] == 'agent_stopped', result
    assert result['local_success'], result
    assert result['recovery']['attempted'] and result['recovery']['return']['success']
    assert result['steps'] == 6 and result['budget_steps'] == result['step_budget'] == 3
    assert result['refunded_steps'] == 3 and len(calls) == 6
    assert all((log.directory / f'step-{i}.png').is_file() for i in range(7))
    events = [json.loads(s) for s in log.path.read_text().splitlines()]
    assert any(e['event'] == 'counterfactual_control_blocked' for e in events)
    assert len([e for e in events if e['event'] == 'counterfactual_step_refund']) == 1
    assert len(model.calls) == 1
    diagnostic_trace = model.calls[0][1]['trajectory']
    assert [r['source_page']['actor_observation']['text'] for r in diagnostic_trace] == [
        f'Seen before decision {i}.' for i in range(1, 4)]
    assert all(r['transition_attempt'] for r in diagnostic_trace)
    trace = json.loads((log.directory / 'recovery/recovery.json').read_text())
    assert trace['root']['id'] == catalog['id']
    assert all('actor_observation' not in r for r in trace['trajectory'] if r['branch'] == 'return')
    assert any(r['branch'] == 'counterfactual' and r['action'].get('name') == 'Green mug' for r in trace['trajectory'])


def test_safe_return_supports_new_observed_state_and_action_cap(saved, site, tmp_path):
    from test_llm_exploration import FakeClient
    graph, _, _, _, _ = saved
    with Browser(site, EventLog(tmp_path), identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(graph, browser, browser.log, finalize_assets=False,
                                 description_client=FakeClient(browser.log), max_descriptions=6)
        nav = RouteNavigator(browser, graph, browser.log)
        nav.recorder = updater; updater.navigator = nav
        nav.connect()
        nav.navigate(graph.data['root'])
        source = browser.observe(); updater.observe(source)
        catalog = next(a for a in source['actions'] if a['name'] == 'Catalog')
        browser.act(catalog); middle = browser.observe(); updater.record(source, catalog, middle)
        green = next(a for a in middle['actions'] if a['name'] == 'Green mug')
        browser.act(green); target = browser.observe(); updater.record(middle, green, target)
        assert target['id'] not in graph.nodes
        # An isolated page can return through the registered homepage in the same session.
        result = nav.return_to_observed(source['id'])
        assert result['success'] and result['route'] == 'root_entry' and result['steps'] == 1
        assert nav.return_to_observed(target['id'], max_actions=2)['success']
        for name in ('Catalog', 'Home'):
            before = browser.observe()
            back = next(a for a in before['actions'] if a['name'] == name)
            browser.act(back)
            updater.record(before, back, browser.observe())
        nav.home()
        assert not nav.return_to_observed(target['id'], max_actions=1)['success']
        assert nav.return_to_observed(target['id'], max_actions=2)['success']
        assert browser.page.url == site + '/item/3'
        nav.home()
        nav.action_guard = lambda page, action: action['name'] == 'Green mug'
        assert not nav.return_to_observed(target['id'], max_actions=2)['success']


def test_negative_diagnosis_cap_and_success_without_alternative_does_not_learn(tmp_path):
    r, page, model, nav = setup_recovery(tmp_path, [decision(False)] * 3)
    for _ in range(8):
        r.record(page, page['actions'][0], page); r.check(page)
    assert len(model.calls) == 3 and not nav.return_to_observed.called
    r, page, model, _ = setup_recovery(tmp_path / 'other', [decision()])
    for _ in range(2):
        r.record(page, page['actions'][0], page)
    r.check(page)
    result = learn_experience(r.log.directory, model, ExperienceStore(tmp_path / 'memory.json', 'test'),
                              success=True, verifier='Independent check')
    assert result['reason'] == 'no_alternative_action_at_root' and len(model.calls) == 1


def test_real_home_resets_count_but_never_offer_a_browser_reset_as_root(tmp_path):
    r, page, model, _ = setup_recovery(tmp_path, [decision(False)])
    for _ in range(2):
        r.reset(page, page)
    assert r.visits[page['id']] == 3
    r.check(page)
    assert model.calls[0][1]['eligible_root_steps'] == []
    assert all(row['phase'] == 'home_reset' for row in r.trace)
    assert r.graph.data['edges'] == []
