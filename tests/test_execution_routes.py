import copy
import json

import pytest

from web2kg.common.browser import Browser
from web2kg.common.storage import Graph, EventLog
from web2kg.stage2.navigation import RouteNavigator, state_matches
from web2kg.stage2.agent import run_task, configuration, COMMANDS
from web2kg.stage1.indexing import build_index


@pytest.fixture(scope='module')
def graph(site, tmp_path_factory):
    graph = Graph(tmp_path_factory.mktemp('route-graph'))
    with Browser(site + '/catalog', EventLog(graph.directory), identity_mode='interaction', settle_ms=20) as browser:
        root = browser.observe(graph.directory)
        graph.data.update(root=root['id'], start_url=root['url'], identity_mode='interaction')
        graph.add_node(root, [])
        for name in ['Red mug', 'Green mug']:
            browser.page.goto(root['url']); browser.settle()
            action = next(a for a in browser.observe()['actions'] if a['name'] == name)
            browser.act(action)
            target = browser.observe(graph.directory)
            graph.add_node(target, [action]); graph.add_edge(root['id'], target['id'], action)
    graph.save(); build_index(graph)
    return graph


@pytest.mark.parametrize('strategy', ['role', 'xpath'])
def test_entry_url_link_and_interior_fallback_preserve_session(graph, site, tmp_path, strategy):
    original = (graph.directory / 'graph.json').read_bytes()
    index = (graph.directory / 'index.json').read_bytes()
    log = EventLog(tmp_path)
    with Browser(site, log, identity_mode='interaction', settle_ms=20) as browser:
        browser.context.add_cookies([{'name': 'session-test', 'value': 'retained', 'url': site}])
        nav = RouteNavigator(browser, graph, log, strategy=strategy)
        assert nav.connect()
        home = nav.graph.data['navigation_home']
        assert nav.graph.path(home, graph.data['root'])
        targets = [k for k in graph.nodes if k != graph.data['root']]
        assert nav.navigate(targets[0])['success']
        assert nav.graph.path(targets[0], targets[1]) is None
        result = nav.navigate(targets[1])
        assert result['success'] and result['steps'] == 2
        assert browser.page.url == graph.nodes[targets[1]]['url']
        assert any(c['name'] == 'session-test' for c in browser.context.cookies())
        assert nav.used == 0
    events = [json.loads(s) for s in log.path.read_text(encoding='utf-8').splitlines()]
    assert any(e['event'] == 'navigation' and e['route'] == 'recorded' and e['success'] for e in events)
    assert (graph.directory / 'graph.json').read_bytes() == original
    assert (graph.directory / 'index.json').read_bytes() == index
    assert (tmp_path / 'navigation/graph.json').is_file()


def test_url_entry_registration_does_not_explore(graph, site, tmp_path):
    with Browser(site, EventLog(tmp_path), identity_mode='interaction', settle_ms=20) as browser:
        nav = RouteNavigator(browser, graph, browser.log, max_bridge_actions=0)
        assert nav.connect()
        new = nav.graph.data['edges'][len(graph.data['edges']):]
        assert len(new) == 2 and all(e['action']['kind'] == 'url_jump' for e in new)
        assert nav.used == 0
        assert nav.navigate(graph.data['root'])['success']


def test_only_post_csrf_value_is_ignored():
    before = {'id': 'old', 'url': 'http://local/', 'headings': ['Home'], 'actions': [
        {'role': 'button', 'name': 'Log out', 'method': 'post', 'form_action': 'http://local/logout?_token=old', 'xpath': '/button'}]}
    after = copy.deepcopy(before); after['id'] = 'new'
    after['actions'][0]['form_action'] = 'http://local/logout?_token=new'
    assert state_matches(before, after)
    for key, value in [('form_action', 'http://local/delete?_token=new'), ('name', 'Delete'), ('disabled', True)]:
        changed = copy.deepcopy(after); changed['actions'][0][key] = value
        assert not state_matches(before, changed)
    after['headings'] = ['Different tab']
    assert not state_matches(before, after)


def test_upstream_stop_and_action_configuration(graph, site, tmp_path):
    calls = []
    class Client:
        def call(self, messages, **kwargs):
            calls.append(messages)
            return 'REASON: Local test complete.\nACTION: stop [done]'
    cfg = configuration(EventLog(tmp_path))
    assert cfg['agent']['actor']['navigation_command'] == COMMANDS
    assert set(COMMANDS) == {'click', 'type', 'stop', 'go_back', 'optimal_navigate'}
    assert cfg['agent']['actor']['note_mode'] == 'attached_field_v1'
    result = run_task(graph, 'Inspect home', EventLog(tmp_path), Client(), None, start_url=site, use_graph=False)
    assert result['status'] == 'agent_stopped', result
    assert result['answer'] == 'done'
    assert len(calls) == 1
    assert (tmp_path / 'step-0.png').is_file()


def test_original_selector_navigates_then_original_type_submits(graph, site, tmp_path):
    import re
    calls = []
    class Client:
        def call(self, messages, system_prompt='', **kwargs):
            calls.append(system_prompt)
            if 'target_page' in system_prompt:
                return json.dumps({'target_page': graph.data['root'] + '.png', 'reasoning': 'Catalog'})
            actor_step = sum('target_page' not in p for p in calls)
            if actor_step == 1:
                return 'REASON: Find the catalog.\nACTION: optimal_navigate [Find product search] [reddit] [Catalog]'
            if actor_step == 2:
                text = json.dumps(messages, ensure_ascii=False)
                field = re.search(r"(?:searchbox|textbox) \[(\d+)\] 'Search products'", text)
                assert field, text[-3000:]
                return f'REASON: Search.\nACTION: type [{field[1]}] [blue mug] [1]'
            return 'REASON: Results visible.\nACTION: stop [done]'
    result = run_task(graph, 'Search blue mug', EventLog(tmp_path), Client(), None, start_url=site,
                      final_check=lambda b: b.page.locator('h1').inner_text() == 'Search results')
    assert result['status'] == 'agent_stopped', result
    assert result['navigation']['success'], result
    assert result['local_success']
    assert len(calls) == 4


def test_model_failure_does_not_trigger_another_call(graph, site, tmp_path):
    from web2kg.common.responses import ResponsesError
    calls = []
    class Client:
        def call(self, *args, **kwargs):
            calls.append(1)
            raise ResponsesError('HTTP 403 fixture')
    result = run_task(graph, 'Inspect', EventLog(tmp_path), Client(), None, start_url=site, use_graph=False)
    assert result['status'] == 'model_error'
    assert calls == [1]


def test_attached_note_is_visible_next_turn_without_extra_browser_action(graph, site, tmp_path):
    import re
    calls = []
    class Client:
        def call(self, messages, system_prompt='', **kwargs):
            assert 'Optional: one short factual memory' in system_prompt
            assert 'use the `note` action' not in system_prompt
            assert '- note [content]' not in system_prompt
            calls.append(messages)
            if len(calls) == 1:
                link = re.search(r"link \[(\d+)\] 'Catalog'", json.dumps(messages))
                assert link
                return f'REASON: Open directory.\nACTION: click [{link[1]}]\nNOTE: The requested item is a blue mug; stock is unverified.'
            assert 'The requested item is a blue mug; stock is unverified.' in json.dumps(messages)
            return 'REASON: Check complete.\nACTION: stop [done]'
    result = run_task(graph, 'Inspect catalog', EventLog(tmp_path), Client(), None, start_url=site, use_graph=False)
    assert result['status'] == 'agent_stopped', result
    assert result['steps'] == result['atomic_actions'] == len(calls) == 2
    events = [json.loads(s) for s in (tmp_path / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    notes = [e for e in events if e['event'] == 'agent_note']
    assert len(notes) == 1 and notes[0]['step'] == 1
    actions = [e for e in events if e['event'] == 'agent_action']
    assert actions[0]['note'] == notes[0]['note'] and actions[1]['note'] == ''
    trace = json.loads((tmp_path / 'trajectory.json').read_text(encoding='utf-8'))
    assert trace[0]['note'] == notes[0]['note']
    assert all(not a['action'].startswith('note') for a in actions)


def test_note_only_reply_does_not_execute_or_pollute_memory(graph, site, tmp_path):
    calls = []
    class Client:
        def call(self, messages, **kwargs):
            calls.append(messages)
            if len(calls) == 1:
                return 'REASON: Remember.\nACTION: note [obsolete command]\nNOTE: Discard this rejected response.'
            return 'REASON: Done.\nACTION: stop [done]\nNOTE: Only this accepted response is retained.'
    result = run_task(graph, 'Inspect', EventLog(tmp_path), Client(), None, start_url=site, use_graph=False)
    assert result['status'] == 'agent_stopped', result
    assert len(calls) == 2  # Existing invalid-action regeneration; no browser round for note.
    assert result['steps'] == result['atomic_actions'] == 1
    events = [json.loads(s) for s in (tmp_path / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [e['note'] for e in events if e['event'] == 'agent_note'] == ['Only this accepted response is retained.']


def test_failed_route_is_not_reported_as_arrival(graph, site, tmp_path, monkeypatch):
    monkeypatch.setattr(RouteNavigator, 'navigate', lambda *a: {'success': False, 'steps': 0, 'reason': 'source_mismatch'})
    calls = []
    class Client:
        def call(self, messages, system_prompt='', **kwargs):
            if 'target_page' in system_prompt:
                return json.dumps({'target_page': graph.data['root'] + '.png'})
            calls.append(messages)
            if len(calls) == 1:
                return 'REASON: Catalog.\nACTION: optimal_navigate [Catalog] [reddit] [Catalog]'
            text = json.dumps(messages)
            assert 'arrival_confirmed: false' in text and 'source_mismatch' in text
            return 'REASON: Route failed.\nACTION: stop [Unable to navigate]'
    result = run_task(graph, 'Catalog', EventLog(tmp_path), Client(), None, start_url=site)
    assert result['status'] == 'agent_stopped', result
    assert not result['navigation']['success']
    actions = json.loads((tmp_path / 'actions.json').read_text(encoding='utf-8'))
    assert actions[0]['requested_target_page'] and actions[0]['target_page'] is None


def test_original_parser_has_no_custom_press_fill_select_back():
    from types import SimpleNamespace
    from web2kg.stage2.agent import load_runtime
    actor = load_runtime().Actor.__new__(load_runtime().Actor)
    actor.config = SimpleNamespace(navigation_command=COMMANDS, planning_command=[])
    actor.online_interaction = {'observation': 'textbox [12] Search products'}
    assert actor.are_valid_actions('type [12] [blue mug] [1]')
    assert actor.are_valid_actions('go_back')
    for action in ['press [12] [Enter]', 'fill [12] [text]', 'select [12] [value]', 'back', 'scroll [down]']:
        assert not actor.are_valid_actions(action)


def test_no_selected_target_cannot_reuse_previous_success(graph, site, tmp_path):
    calls, selections = [], []
    class Client:
        def call(self, messages, system_prompt='', **kwargs):
            if 'target_page' in system_prompt:
                selections.append(1)
                return json.dumps({'target_page': graph.data['root'] + '.png' if len(selections) == 1 else 'None'})
            calls.append(messages)
            if len(calls) <= 2:
                return 'ACTION: optimal_navigate [Catalog] [reddit] [Catalog]'
            text = json.dumps(messages)
            assert text.count('arrival_confirmed: true') == 1
            assert text.count('arrival_confirmed: false') == 1
            assert 'no_target_selected' in text
            return 'ACTION: stop [Second destination not found]'
    result = run_task(graph, 'Inspect catalog', EventLog(tmp_path), Client(), None, start_url=site, max_steps=3)
    assert result['status'] == 'agent_stopped', result
    assert len(calls) == result['steps'] == 3 and len(selections) == 2
    assert not result['navigation']['success'] and result['navigation']['reason'] == 'no_target_selected'


@pytest.mark.parametrize('allow_writes', [False, True])
def test_upstream_transport_removes_legacy_read_only_gate(graph, site, tmp_path, allow_writes):
    class Client:
        def call(self, *a, **kw):
            return 'REASON: Check completed.\nACTION: stop [done]'
    def check(browser):
        write = browser.page.evaluate("fetch('/change',{method:'POST'}).then(r=>r.ok).catch(()=>false)")
        return write
    result = run_task(graph, 'Transport check', EventLog(tmp_path), Client(), None,
        use_graph=False, start_url='http://127.0.0.1:65521', backend=site, allow_writes=allow_writes, final_check=check)
    assert result['status'] == 'agent_stopped', result
    assert result['local_success']
    assert result['url'].startswith('http://127.0.0.1:65521')
