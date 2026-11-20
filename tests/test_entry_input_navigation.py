import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_replay_eligibility import graph_at
from web2kg.common.storage import EventLog, write_json
from web2kg.stage2.navigation import RouteNavigator
from web2kg.stage2.retrieval import template_order, retrieve
from web2kg.stage1.indexing import build_index
from web2kg.stage1.visual_index import build_visual_index
from test_visual_navigation import FakeJina


def test_url_edges_require_both_entries_and_check_actual_destination(tmp_path):
    graph = graph_at(tmp_path / 'graph')
    graph.add_node({'id': 'd', 'url': 'http://other/entry', 'title': 'D',
                    'headings': [], 'text': '', 'actions': []}, [])
    graph.initialize_url_entries()
    graph.register_url_entry('d', 'fixture_task_initial_state')
    assert graph.reachable() == {'a', 'b', 'c', 'd'}
    assert {e['source'] for e in graph.data['edges'] if e['action']['kind'] == 'url_jump'} == {'a', 'd'}
    key = graph.add_edge('b', 'd', {'kind': 'url_jump', 'url': graph.nodes['d']['url']})
    assert not graph.edge_replayable(next(e for e in graph.data['edges'] if e['id'] == key))
    browser = SimpleNamespace(start_url=graph.data['start_url'], act=Mock(), settle=Mock(),
        page=SimpleNamespace(goto=Mock()), observe=Mock(side_effect=[graph.nodes['d'], graph.nodes['d'], graph.nodes['b']]))
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    result = nav.navigate('a')
    assert result['reason'] == 'destination_mismatch' and result['steps'] == 1
    assert result['status'] == 'requires_local_action' and not result['success']
    assert browser.page.goto.call_count == 1
    browser.act.assert_not_called()


@pytest.mark.parametrize('phase', ['source', 'destination'])
def test_mismatched_route_hands_off_without_running_suffix(tmp_path, phase):
    graph = graph_at(tmp_path / 'graph')
    changed = copy.deepcopy(graph.nodes['b'])
    changed['headings'] = ['Different page']
    observations = [changed] if phase == 'source' else [graph.nodes['a'], changed]
    browser = SimpleNamespace(start_url=graph.data['start_url'], act=Mock(),
                              observe=Mock(side_effect=observations))
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    result = nav.replay(graph.path('a', 'c'))
    assert result['status'] == 'requires_local_action' and not result['success']
    assert result['reason'] == phase + '_mismatch'
    assert result['requested_target'] == 'c'
    assert browser.act.call_count == result['steps'] == (phase == 'destination')
    browser.observe = Mock(return_value=changed)
    assert nav.feedback()['current_url'] == changed['url']
    assert nav.navigate('c')['reason'] == 'teleport_requires_local_action'
    assert browser.act.call_count == (phase == 'destination')
    # Changing the live page releases the guard without claiming the saved target.
    browser.observe.return_value = graph.nodes['b']
    assert not nav.awaiting_local_action()


@pytest.mark.parametrize('repeat_teleport', [False, True])
def test_real_actor_continues_from_mismatched_intermediate_page(site, tmp_path, repeat_teleport):
    import re
    from web2kg.common.browser import Browser
    from web2kg.common.storage import Graph
    from web2kg.stage2.agent import run_task
    graph = Graph(tmp_path / 'graph')
    with Browser(site, EventLog(graph.directory), identity_mode='interaction', settle_ms=20) as browser:
        root = browser.observe(graph.directory)
        graph.data.update(root=root['id'], start_url=root['url'], identity_mode='interaction')
        graph.add_node(root, [])
        for name in ['Catalog', 'Red mug']:
            before = browser.observe()
            action = next(a for a in before['actions'] if a['name'] == name)
            browser.act(action); after = browser.observe(graph.directory)
            graph.add_node(after, []); graph.add_edge(before['id'], after['id'], action)
            if name == 'Catalog':
                graph.nodes[after['id']]['headings'] = ['Old catalog heading']
    graph.save(); build_index(graph)
    calls, selections = [], []
    class Client:
        def call(self, messages, system_prompt='', **kwargs):
            if 'target_page' in system_prompt:
                selections.append(1)
                return json.dumps({'target_page': after['id'] + '.png'})
            calls.append(messages)
            if len(calls) == 1:
                return 'ACTION: optimal_navigate [Red mug] [reddit] [Product]'
            if len(calls) == 2:
                text = json.dumps(messages)
                assert 'destination_mismatch' in text and 'status: requires_local_action' in text
                assert 'arrival_confirmed: false' in text
                assert 'NAVIGATION SUCCESS' not in text and 'NAVIGATION FAILED' not in text
                assert 'actual_url' in text and site + '/catalog' in text
                if repeat_teleport:
                    return 'ACTION: optimal_navigate [Red mug] [reddit] [Product]'
                control = re.search(r"link \[(\d+)\] 'Red mug'", text)[1]
                return f'ACTION: click [{control}]'
            return 'ACTION: stop [Red mug]'
    result = run_task(graph, 'Open Red mug', EventLog(tmp_path/'run'), Client(), None,
                      max_steps=5, final_check=lambda b: b.page.url == site + '/item/2')
    assert selections == [1]
    assert result['navigation']['steps'] == 1 and result['navigation']['status'] == 'requires_local_action'
    if repeat_teleport:
        assert len(calls) == 2 and result['steps'] == 1
        assert result['url'] == site + '/catalog' and 'local action' in result['error']
    else:
        assert result['status'] == 'agent_stopped' and result['local_success'], result
        assert len(calls) == result['steps'] == 3
    events = [json.loads(line) for line in (tmp_path/'run/events.jsonl').read_text().splitlines()]
    handoff = next(e for e in events if e['event'] == 'replay_state_handoff')
    assert handoff['steps'] == 1 and handoff['current_url'] == site + '/catalog'


@pytest.mark.parametrize('kind', ['type', 'fill'])
def test_input_boundary_is_reachable_but_never_executed_or_retried(tmp_path, kind):
    graph = graph_at(tmp_path / 'graph')
    graph.data['edges'][1]['action'].update(kind=kind, value='old task secret')
    graph.data['edges'][1]['replayable'] = False  # Explicit input override, even for old graphs.
    browser = SimpleNamespace(start_url=graph.data['start_url'], act=Mock(),
        observe=Mock(side_effect=[graph.nodes['a'], graph.nodes['a'], graph.nodes['b'], graph.nodes['b']]))
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    result = nav.navigate('c')
    assert result['status'] == 'requires_input' and result['steps'] == 1
    assert browser.act.call_count == 1 and browser.act.call_args.args[0]['kind'] == 'click'
    assert 'old task secret' not in json.dumps(result)
    browser.observe = Mock(return_value=graph.nodes['b'])
    assert nav.navigate('c')['reason'] == 'teleport_requires_local_action'
    assert browser.act.call_count == 1
    assert nav.feedback()['status'] == 'requires_input'
    # Input state change resumes eligibility; no synthetic arrival at the saved target.
    changed = copy.deepcopy(graph.nodes['b']); changed['id'] = 'new-input-state'
    changed['headings'] = ['Actual changed input state']
    browser.observe = Mock(return_value=changed)
    assert not nav.awaiting_input()
    browser.page = SimpleNamespace(goto=Mock())
    browser.settle = Mock()
    browser.observe = Mock(side_effect=[changed, graph.nodes['a'], graph.nodes['a'],
                                        graph.nodes['b'], graph.nodes['b']])
    result = nav.navigate('c')
    assert result['status'] == 'requires_input' and result['steps'] == 2
    browser.observe = Mock(return_value=graph.nodes['b'])
    assert nav.feedback()['steps'] == 2
    assert browser.page.goto.call_args.args[0] == graph.nodes['a']['url']
    assert browser.act.call_count == 2  # Only the two separate root clicks; no stored input.


@pytest.mark.parametrize('visual', [False, True])
def test_template_dedup_uses_best_page_and_reaches_beyond_topk(tmp_path, visual):
    graph = graph_at(tmp_path)
    write_json(tmp_path / 'templates/active.json', {'directory': 'templates/test'})
    write_json(tmp_path / 'templates/test/templates.json', {'assignments': {
        'a': {'template': 'T1'}, 'b': {'template': 'T1'}, 'c': {'template': 'T2'}}})
    assert template_order(graph, ['a', 'b', 'c'], [.8, .9, .7], 2) == [1, 2]
    client = FakeJina()
    build_visual_index(graph, client) if visual else build_index(graph)
    rows = retrieve(graph, 'b', client if visual else None, 2)
    assert [r['id'] for r in rows] == ['b', 'c']


def test_return_can_goto_registered_root_from_nonentry(tmp_path):
    graph = graph_at(tmp_path / 'graph')
    browser = Mock(start_url=graph.data['start_url'])
    browser.observe.return_value = graph.nodes['a']
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    nav.connect()
    browser.observe.return_value = graph.nodes['b']
    browser.page.goto.side_effect = lambda *args, **kwargs: setattr(browser.observe, 'return_value', graph.nodes['a'])
    result = nav.return_to_observed('a')
    assert result['success'] and result['route'] == 'root_entry' and result['steps'] == 1
    browser.page.goto.assert_called_once()
    browser.act.assert_not_called()


def test_disconnected_node_is_not_automatically_an_initial_entry(tmp_path):
    graph = graph_at(tmp_path)
    graph.data['edges'].clear()
    graph.initialize_url_entries()
    assert set(graph.data['url_entries']) == {'a'}
    assert graph.reachable() == {'a'}


def test_same_structural_id_cannot_hide_input_change(tmp_path):
    from web2kg.stage2.navigation import state_matches
    graph = graph_at(tmp_path)
    before = copy.deepcopy(graph.nodes['a'])
    before['actions'] = [{'role': 'textbox', 'name': 'Search', 'value_digest': 'empty'}]
    after = copy.deepcopy(before); after['actions'][0]['value_digest'] = 'filled'
    assert not state_matches(before, after)


@pytest.mark.parametrize('repeat_teleport', [False, True])
def test_real_actor_receives_input_handoff_and_cannot_loop(site, tmp_path, repeat_teleport):
    import re
    from web2kg.common.browser import Browser
    from web2kg.common.storage import Graph
    from web2kg.stage2.agent import run_task
    graph = Graph(tmp_path / 'graph')
    with Browser(site + '/catalog', EventLog(graph.directory), identity_mode='interaction', settle_ms=20) as browser:
        before = browser.observe(graph.directory)
        graph.data.update(root=before['id'], start_url=before['url'], identity_mode='interaction')
        graph.add_node(before, [])
        field = next(a for a in before['actions'] if a['role'] == 'searchbox')
        browser.page.get_by_role('searchbox').fill('old-task-value')
        browser.page.get_by_role('searchbox').press('Enter'); browser.settle()
        after = browser.observe(graph.directory)
        graph.add_node(after, [])
        graph.add_edge(before['id'], after['id'], {**field, 'kind': 'type', 'value': 'old-task-value'})
    graph.save(); build_index(graph)
    actor_calls, selector_calls = [], []
    class Client:
        def call(self, messages, system_prompt='', **kwargs):
            if 'target_page' in system_prompt:
                selector_calls.append(1)
                return json.dumps({'target_page': after['id'] + '.png'})
            actor_calls.append(messages)
            if len(actor_calls) == 1:
                return 'REASON: Find search.\nACTION: optimal_navigate [Search] [reddit] [Search results]'
            text = json.dumps(messages)
            if len(actor_calls) == 2:
                assert 'actor_input_required' in text and 'Paused BEFORE input' in text
                assert 'arrival_confirmed: false' in text and 'status: requires_input' in text
                assert 'NAVIGATION SUCCESS' not in text
                if repeat_teleport:
                    return 'ACTION: optimal_navigate [Search] [reddit] [Search results]'
                field_id = re.search(r"searchbox \[(\d+)\] 'Search products'", text)[1]
                return f'ACTION: type [{field_id}] [blue mug] [1]'
            return 'ACTION: stop [done]'
    result = run_task(graph, 'Search for blue mug', EventLog(tmp_path / 'run'), Client(), None, max_steps=5,
                      final_check=lambda b: 'blue' in b.page.url)
    assert selector_calls == [1]
    if repeat_teleport:
        assert len(actor_calls) == 2 and result['steps'] == 1
        assert 'local action' in result['error']
    else:
        assert result['status'] == 'agent_stopped' and result['local_success'], result
        assert len(actor_calls) == result['steps'] == 3
    assert result['navigation']['status'] == 'requires_input'
