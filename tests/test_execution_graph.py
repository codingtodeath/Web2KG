import copy
import hashlib
import json
import yaml
import re

import pytest

from test_function_cards import saved, CardModel
from test_llm_exploration import FakeClient
from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog, Graph
from web2kg.stage2.agent import run_task
from web2kg.stage2.graph_update import ExecutionGraph
from web2kg.stage2.navigation import RouteNavigator
from web2kg.stage2.template_context import TemplateKnowledge


def hashes(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


def test_new_entity_reuses_template_and_later_actor_sees_it(saved, site, tmp_path):
    base, _, _, cards, (_, catalog, product) = saved
    original = hashes(base.directory)
    log = EventLog(tmp_path)
    knowledge = TemplateKnowledge(base, cards, log)
    original_assignments = copy.deepcopy(knowledge.layer['assignments'])
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        model = FakeClient(EventLog(tmp_path / 'annotations'))
        updater = ExecutionGraph(base, browser, log, knowledge, model, 1, async_descriptions=False)
        nav = RouteNavigator(browser, base, log)
        nav.recorder = updater; updater.navigator = nav
        before = browser.observe(); updater.observe(before)
        action = next(a for a in before['actions'] if a['name'] == 'Green mug')
        browser.act(action); after = browser.observe()
        updater.record(before, action, after)
        assert after['id'] not in base.nodes
        assert knowledge.observe(after, 'Inspect')['match']['template_id'] == knowledge.layer['assignments'][product['id']]['template']
        assert knowledge.observe(after, 'Inspect')['function_card']
        assert all(updater.layer['assignments'][k] == v for k, v in original_assignments.items())
        assert not updater.observe(after)
        assert updater.graph.path(catalog['id'], after['id'])
        assert len(model.calls) == 1 and model.calls[0][0] == 'transition_description'
        record = updater.graph.data['transition_records'][0]
        assert record['annotation']['status'] == 'completed'
        assert record['before_observation'] == 'live'
        to_catalog = next(a for a in after['actions'] if a['name'] == 'Catalog')
        browser.act(to_catalog); returned = browser.observe()
        updater.record(after, to_catalog, returned)
        browser.page.go_back(); browser.settle()
        updater.record(returned, {'id': 'back', 'kind': 'go_back', 'role': 'browser', 'name': 'go_back'}, browser.observe())
        assert updater.graph.path(after['id'], catalog['id']) is None  # Unreviewed return stays blocked.
        assert nav.navigate(product['id'])['success']  # The current URL-root policy permits the known root route.
        navigation = [json.loads(s) for s in log.path.read_text(encoding='utf-8').splitlines()]
        assert any(e['event'] == 'navigation_root_goto' for e in navigation)
        assert [e for e in navigation if e['event'] == 'navigation'][-1]['route'] == 'recorded'
        initial_edges = {e['id'] for e in base.data['edges']}
        assert set(updater.graph.data['execution_update']['new_edges']) == {
            e['id'] for e in updater.graph.data['edges'] if e['id'] not in initial_edges}
        updater.finish()
        preview = browser.context.browser.new_page()
        preview.goto((updater.graph.directory / 'viewer.html').as_uri() + '#templates')
        frame = preview.frame_locator('#templateFrame')
        assert frame.locator('#groups button').count() == len(updater.layer['templates'])
        assert (updater.graph.directory / 'snapshots' / after['id'] / 'page.png').exists()
        preview.close()
    assert hashes(base.directory) == original


@pytest.mark.parametrize('fail', [False, True])
def test_description_cap_or_failure_never_retries_and_evidence_survives(saved, site, tmp_path, fail):
    base, _, _, _, _ = saved
    log = EventLog(tmp_path)
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        model = FakeClient(EventLog(tmp_path / 'annotations'), fail=fail)
        updater = ExecutionGraph(base, browser, log, description_client=model, max_descriptions=1, async_descriptions=False)
        before = browser.observe(); updater.observe(before)
        action = next(a for a in before['actions'] if a['name'] == 'Green mug')
        browser.act(action); after = browser.observe()
        updater.record(before, action, after)
        browser.page.go_back(); browser.settle()
        native = {'id': 'native-back', 'kind': 'go_back', 'role': 'browser', 'name': 'go_back'}
        updater.record(after, native, browser.observe())
        assert len(model.calls) == 1
        records = updater.graph.data['transition_records']
        assert records[0]['annotation']['status'] == ('failed' if fail else 'completed')
        assert records[0]['replayable'] is (not fail)
        assert records[1]['annotation']['reason'] == ('previous_failure' if fail else 'call_budget')
        assert records[1]['replayable'] is False
        assert updater.graph.path(after['id'], before['id']) is None
        assert after['id'] in updater.graph.nodes


def test_transition_description_reused_despite_changed_ax_id(saved, site, tmp_path):
    base, _, _, _, _ = saved
    log = EventLog(tmp_path)
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        model = FakeClient(EventLog(tmp_path / 'annotations'))
        updater = ExecutionGraph(base, browser, log, description_client=model, async_descriptions=False)
        assert updater.description_limit is None
        nav = RouteNavigator(browser, base, log)
        updater.navigator = nav
        before = browser.observe(); updater.observe(before)
        action = next(a for a in before['actions'] if a['name'] == 'Green mug')
        browser.act(action); after = browser.observe()
        updater.record(before, action, after)
        assert nav.graph.path(before['id'], after['id'])
        edge_count, revision = len(updater.graph.data['edges']), nav.graph.replay_revision
        original = model.json
        model.json = lambda *args: {**original(*args), 'replayable': False, 'replay_reason': 'Not reliably repeatable'}
        browser.page.go_back(); browser.settle()
        before = browser.observe(); browser.act(action); after = browser.observe()
        updater.record(before, {**action, 'ax_node_id': 'new-live-id'}, after)
        assert len(model.calls) == 1
        assert len(updater.graph.data['edges']) == edge_count
        assert nav.graph.replay_revision == revision
        assert updater.graph.path(before['id'], after['id'])
        assert nav.graph.path(before['id'], after['id'])
        edge = next(e for e in nav.graph.data['edges'] if e['id'] == updater.graph.data['transition_records'][-1]['edge'])
        assert edge['replayable'] is True
        assert updater.graph.data['transition_records'][-1]['description_reused']
        inherited_model = FakeClient(EventLog(tmp_path / 'next-model'))
        inherited = ExecutionGraph(updater.graph, browser, EventLog(tmp_path / 'next-task'),
                                   description_client=inherited_model, async_descriptions=False)
        inherited.record(before, {**action, 'ax_node_id': 'third-live-id'}, after)
        assert not inherited_model.calls
        assert inherited.graph.data['transition_records'][-1]['description_reused']
        inherited.record(before, action, after, error='observation_error')
        assert not inherited_model.calls
        assert not inherited.graph.data['transition_records'][-1]['replayable']
        assert inherited.graph.path(before['id'], after['id'])


def test_menu_inherits_input_is_boundary_and_unreviewed_write_stays_blocked(saved, site, tmp_path):
    base, _, _, _, _ = saved
    log = EventLog(tmp_path)
    with Browser(site, log, identity_mode='interaction', allow_writes=True, settle_ms=20) as browser:
        updater = ExecutionGraph(base, browser, log)
        browser.page.set_content('<main><h1>Editor</h1><button aria-expanded="false" onclick="this.setAttribute(\'aria-expanded\',\'true\');document.querySelector(\'form\').hidden=false">More</button><form hidden><label>Text<input></label><button>Save</button></form></main>')
        before = browser.observe(); updater.observe(before)
        action = next(a for a in before['actions'] if a['name'] == 'More')
        browser.act(action); after = browser.observe()
        updater.record(before, action, after)
        assignments = updater.layer['assignments']
        assert assignments[after['id']]['template'] == assignments[before['id']]['template']
        assert assignments[after['id']]['reason'] == 'observed_local_state_variant'
        field = next(a for a in after['actions'] if a['role'] == 'textbox')
        browser.page.get_by_role('textbox').fill('A private test value')
        filled = browser.observe()
        updater.record(after, {**field, 'kind': 'type', 'press_enter': False}, filled)
        assert assignments[filled['id']]['template'] == assignments[after['id']]['template']
        assert assignments[filled['id']]['local_state_rule'] == 'observed-form-edit-v1'
        assert updater.graph.path(after['id'], filled['id'])
        browser.page.goto(site); browser.settle()
        source = browser.observe(); updater.observe(source)
        write = next(a for a in source['actions'] if a['name'] == 'Continue')
        browser.act(write); target = browser.observe()
        updater.record(source, write, target)
        record = updater.graph.data['transition_records'][-1]
        assert record['outcome'] == 'transition' and not record['replayable']
        assert updater.graph.path(source['id'], target['id']) is None
        assigned = updater.layer['assignments'][target['id']]
        assert assigned['reason'] == 'isolated_empty_structure'
        assert next(g for g in updater.layer['templates'] if g['id'] == assigned['template'])['isolated']
        assert any(updater.layer['assignments'][k]['reason'] == 'new_template'
                   for k in updater.graph.data['execution_update']['new_states'] if k in updater.layer['assignments'])
        updater.finish()
        blocked = next(e for e in updater.graph.data['edges'] if e['id'] == record['edge'])
        browser.page.goto(source['url']); browser.settle()
        assert RouteNavigator(browser, updater.graph, log).replay([blocked])['reason'] == 'transition_not_replayable'


def test_failed_action_has_no_success_edge_and_duplicate_state_is_not_rewritten(saved, site, tmp_path):
    base, _, _, _, _ = saved
    with Browser(site, EventLog(tmp_path), identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(base, browser, browser.log)
        before = browser.observe(); updater.observe(before)
        original = hashes(updater.graph.directory / 'snapshots')
        edges = len(updater.graph.data['edges'])
        action = next(a for a in before['actions'] if a['name'] == 'Catalog')
        updater.record(before, action, None, error='timeout')
        updater.record(before, action, before)
        assert len(updater.graph.data['edges']) == edges
        assert hashes(updater.graph.directory / 'snapshots') == original
        assert [r['outcome'] for r in updater.graph.data['transition_records']] == ['failed', 'no_change']


def test_original_agent_updates_new_state_before_next_decision(saved, site, tmp_path):
    base, _, _, cards, _ = saved
    log = EventLog(tmp_path)
    knowledge = TemplateKnowledge(base, cards, log)
    seen = []
    class Actor:
        def call(self, messages, **kwargs):
            context = yaml.safe_load(messages[-1]['content'].split('\n', 1)[1])
            seen.append(context)
            if len(seen) == 1:
                match = re.search(r"link \[(\d+)\] 'Green mug'", json.dumps(messages))
                assert match
                return f'REASON: Inspect a new item.\nACTION: click [{match[1]}]'
            assert context['template_id']
            assert context['function_card']
            return 'REASON: New item recorded.\nACTION: stop [done]'
    result = run_task(base, 'Inspect Green mug', log, Actor(), None, knowledge=knowledge,
                      start_url=site + '/catalog', update_graph=True)
    assert result['status'] == 'agent_stopped', result
    assert 'execution_graph_error' not in result
    expanded = Graph.load(tmp_path / 'execution-graph')
    assert any(n['url'].endswith('/item/3') for n in expanded.nodes.values())
    assert result['execution_graph']['new_states']
    assert len(seen) == 2
    events = [json.loads(line) for line in log.path.read_text(encoding='utf-8').splitlines()]
    assert any(e['event'] == 'template_match' and e['reason'] == 'saved_state_identity'
               for e in events)


def test_bridge_and_replay_updates_record_actual_steps_without_reset_edges(saved, site, tmp_path):
    base, _, _, _, (_, catalog, product) = saved
    with Browser(site, EventLog(tmp_path), identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(base, browser, browser.log)
        nav = RouteNavigator(browser, base, browser.log)
        nav.recorder = updater; updater.navigator = nav
        nav.connect()
        assert updater.count == 0
        assert nav.navigate(product['id'])['success']
        assert updater.count == 2
        assert all(r['phase'] == 'replay' for r in updater.graph.data['transition_records'])
        assert all(r['outcome'] == 'transition' for r in updater.graph.data['transition_records'])
        assert updater.graph.path(base.data['root'], product['id']) is None  # No Luna client: replay is unreviewed.


def test_browser_hook_records_each_input_and_a_failed_action(saved, site, tmp_path):
    from types import SimpleNamespace
    from web2kg.stage2.agent import load_runtime
    load_runtime()
    from browser_env import ActionTypes
    base, _, _, _, _ = saved
    with Browser(site + '/catalog', EventLog(tmp_path), identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(base, browser, browser.log)
        field = next(a for a in browser.observe()['actions'] if a['role'] == 'searchbox')
        def execute(command):
            if command['action_type'] == ActionTypes.CLICK:
                raise RuntimeError('fixture action failed')
            browser.page.get_by_role('searchbox').fill(command['text'])
            return ({}, 0, False, False, {'fail_error': ''})
        env = SimpleNamespace(step=execute)
        updater.wrap_step(env)
        for value in ['first input', 'second input']:
            env.step({'action_type': ActionTypes.TYPE, 'element_id': field['ax_node_id'], 'text': value, 'flag': False})
        with pytest.raises(RuntimeError, match='fixture'):
            env.step({'action_type': ActionTypes.CLICK, 'element_id': field['ax_node_id']})
        records = updater.graph.data['transition_records']
        assert len(records) == 3
        assert records[0]['target'] == records[1]['source']
        assert records[1]['target'] == records[2]['source']
        assert all(r['replayable'] for r in records[:2])
        assert not records[-1]['replayable']
        assert records[-1]['outcome'] == 'failed' and records[-1]['edge'] is None
        assert all('text' not in r['action'] for r in records)
        assert all(r['action']['press_enter'] is False for r in records[:2])
        assigned = updater.layer['assignments']
        assert len({assigned[k]['template'] for r in records[:2] for k in (r['source'], r['target'])}) == 1


def test_form_edits_keep_template_but_not_state_and_reject_replacement(saved, site, tmp_path):
    from web2kg.stage1.templates import build_templates, local_variant, structure_profile
    base, _, _, _, _ = saved
    with Browser(site, EventLog(tmp_path), identity_mode='interaction', settle_ms=20) as browser:
        browser.page.set_content('<main><h1>Editor</h1><form method="post"><label>Title<input></label>'
            '<label>Body<textarea></textarea></label><button>Save</button><div id="validation"></div></form></main>')
        updater = ExecutionGraph(base, browser, browser.log)
        first = before = browser.observe(); updater.observe(first)
        for role, value in [('input', 'A title'), ('textarea', 'A paragraph\nA second paragraph')]:
            field = next(a for a in before['actions'] if f'/{role}[' in a['xpath'])
            browser.page.locator(role).fill(value)
            browser.page.locator('#validation').evaluate('(el) => el.innerHTML = "<p>Draft changed</p>"')
            after = browser.observe()
            action = {**field, 'kind': 'type', 'press_enter': False}
            updater.record(before, action, after)
            assert after['id'] != before['id']
            assignments = updater.layer['assignments']
            assert assignments[after['id']]['template'] == assignments[first['id']]['template']
            assert assignments[after['id']]['local_state_rule'] == 'observed-form-edit-v1'
            a, b = (structure_profile(n['ax'], n['dom']) for n in (before, after))
            assert not local_variant(before, after, {**action, 'press_enter': True}, a, b)
            assert not local_variant(before, after, {**action, 'press_enter': None}, a, b)
            assert not local_variant(before, {**after, 'url': site + '/other'}, action, a, b)
            before = after
        form_graph = Graph(updater.graph.directory)
        form_graph.data = copy.deepcopy(updater.graph.data)
        form_graph.data['nodes'] = {k: n for k, n in form_graph.nodes.items()
                                    if k in updater.graph.data['execution_update']['new_states']}
        form_graph.data['root'] = first['id']
        form_graph.data['edges'] = [e for e in form_graph.data['edges'] if e['source'] in form_graph.nodes]
        offline = build_templates(form_graph, weights=(.5, 1.5, 1))['assignments']
        assert offline[first['id']]['template'] == offline[after['id']]['template']
        # Identical markup at the same URL is insufficient if the actual field was replaced.
        browser.page.locator('textarea').evaluate('(el) => el.outerHTML = "<textarea></textarea>"')
        replaced = browser.observe()
        assert not local_variant(before, replaced, action, b, structure_profile(replaced['ax'], replaced['dom']))


def test_input_cannot_inherit_when_results_or_form_controls_change(saved, site, tmp_path):
    from web2kg.stage1.templates import local_variant, structure_profile
    with Browser(site, EventLog(tmp_path), identity_mode='interaction', settle_ms=20) as browser:
        browser.page.set_content('<main><h1>Search</h1><form><input type="search"><button>Find</button></form></main>')
        before = browser.observe()
        field = next(a for a in before['actions'] if a['role'] == 'searchbox')
        action = {**field, 'kind': 'type', 'press_enter': False}
        a = structure_profile(before['ax'], before['dom'])
        browser.page.locator('input').fill('Query')
        browser.page.locator('main').evaluate('(el) => el.insertAdjacentHTML("beforeend", "<article><h2>Result</h2><a href=/item>Open</a></article>")')
        after = browser.observe()
        assert not local_variant(before, after, action, a, structure_profile(after['ax'], after['dom']))
        browser.page.locator('article').evaluate('(el) => el.remove()')
        browser.page.locator('form').evaluate('(el) => el.insertAdjacentHTML("beforeend", "<input type=checkbox>")')
        changed = browser.observe()
        assert not local_variant(before, changed, action, a, structure_profile(changed['ax'], changed['dom']))


def test_new_template_card_is_available_before_next_decision_and_preserves_descriptions(saved, site, tmp_path):
    base, _, _, cards, _ = saved
    log = EventLog(tmp_path)
    knowledge = TemplateKnowledge(base, cards, log)
    original = hashes(base.directory)
    with Browser(site, log, identity_mode='interaction', settle_ms=20) as browser:
        model = CardModel(tmp_path / 'card-calls')
        descriptions = FakeClient(EventLog(tmp_path / 'description-calls'))
        updater = ExecutionGraph(base, browser, log, knowledge, descriptions, 2, model, 2, async_descriptions=False)
        before = browser.observe(); updater.observe(before)
        action = next(a for a in before['actions'] if a['name'] == 'Catalog')
        browser.page.set_content('<main><h1>Edit</h1><form><label>Title<input></label><button>Save</button></form></main>')
        after = browser.observe(); updater.record(before, action, after)
        context = knowledge.observe(after, 'Edit')
        assert context['function_card'] and len(model.calls) == 1
        assert set(model.calls[0]) == {'template_id', 'members', 'representatives'}
        assert model.calls[0]['representatives'][0]['evidence_ref'] == 'P1'
        assert updater.graph.data['transition_records'][0]['annotation']['status'] == 'completed'
        field = next(a for a in after['actions'] if a['role'] == 'textbox')
        browser.page.get_by_role('textbox').fill('Draft')
        updater.record(after, {**field, 'kind': 'type', 'press_enter': False}, browser.observe())
        assert len(model.calls) == 1  # Same template: retain sampled card, no per-keystroke API.
        summary = updater.finish()
        assert summary['cards']['completed_new'] == 1 and summary['description_calls'] == 2
        generated = updater.live_cards.directory / 'cards.json'
        checked = TemplateKnowledge(updater.graph, generated, EventLog(tmp_path / 'checked'))
        assert context['match']['template_id'] in checked.cards
    assert hashes(base.directory) == original


def test_orphan_representative_changes_only_at_finish(saved, site, tmp_path):
    base, _, _, cards, (_, catalog, _) = saved
    log = EventLog(tmp_path / 'run')
    knowledge = TemplateKnowledge(base, cards, log)
    original = hashes(base.directory)
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(base, browser, log, knowledge)
        source = browser.observe(); updater.observe(source)
        # A new entry has no recorded incoming action.
        html = '<main><h1>New entry</h1><form><input type="date"><input type="range"><button>Save</button></form></main>'
        browser.page.set_content(html)
        root = browser.observe(); updater.observe(root)
        template = updater.layer['assignments'][root['id']]['template']
        group = next(g for g in updater.layer['templates'] if g['id'] == template)
        assert group['representative'] == root['id']
        # Another observed arrival has the same page structure and a real incoming action.
        browser.page.set_content(html.replace('New entry', 'Another entry'))
        reached = browser.observe()
        assert reached['id'] != root['id']
        action = next(a for a in source['actions'] if a['name'] == 'Green mug')
        updater.record(source, action, reached)
        assert updater.layer['assignments'][reached['id']]['template'] == template
        assert group['representative'] == root['id']
        assignments = copy.deepcopy(updater.layer['assignments'])
        result = updater.finish()
        assert group['representative'] == reached['id']
        assert updater.layer['assignments'] == assignments
        assert any(row['template'] == template for row in result['representative_updates'])
        assert result['card_refresh']['reason'] == 'no_card_client'
    assert hashes(base.directory) == original


@pytest.mark.parametrize('fail', [False, True])
def test_card_cap_and_failure_leave_graph_and_actor_running(saved, site, tmp_path, fail):
    from unittest.mock import Mock
    from web2kg.common.responses import ResponsesError
    base, _, _, cards, _ = saved
    knowledge = TemplateKnowledge(base, cards, EventLog(tmp_path))
    with Browser(site, knowledge.log, identity_mode='interaction', settle_ms=20) as browser:
        model = CardModel(tmp_path / 'card-calls')
        if fail:
            model.json = Mock(side_effect=ResponsesError('HTTP 503'))
        updater = ExecutionGraph(base, browser, knowledge.log, knowledge, card_client=model, max_cards=1)
        for content in ['<main><form><input><button>Save</button></form></main>',
                        '<main><table><tr><td>Report</td></tr></table><input type="range"></main>']:
            browser.page.set_content(content)
            observed = browser.observe(); updater.observe(observed)
            assert not updater.observe(observed)
        assert updater.live_cards.calls == 1
        rows = updater.live_cards.data['cards']
        new = [r for k, r in rows.items() if k not in updater.live_cards.initial]
        assert [r['status'] for r in new] == ['failed' if fail else 'completed', 'not_generated']
        assert new[-1]['reason'] == ('previous_failure' if fail else 'call_budget')
        assert updater.finish()['new_states']
