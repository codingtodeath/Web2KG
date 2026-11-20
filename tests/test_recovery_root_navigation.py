import copy
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from test_replay_eligibility import graph_at
from web2kg.common.storage import EventLog
from web2kg.stage2.graph_update import ExecutionGraph
from web2kg.stage2.navigation import RouteNavigator


class Browser:
    """Deterministic live state; graph traversal still uses the real navigator."""
    def __init__(self, graph, current='c', goto_state='a'):
        self.start_url = graph.data['start_url']
        self.current = copy.deepcopy(graph.nodes[current])
        self.goto_state = copy.deepcopy(graph.nodes[goto_state])
        self.moves = {e['action']['id']: copy.deepcopy(graph.nodes[e['target']])
                      for e in graph.data['edges']}
        self.page = SimpleNamespace(goto=Mock(side_effect=self.goto))
        self.act = Mock(side_effect=self.move)
        self.settle = self.save_observation = Mock()

    def observe(self, *args):
        return copy.deepcopy(self.current)

    def goto(self, *args, **kwargs):
        self.current = copy.deepcopy(self.goto_state)

    def move(self, action, strategy):
        self.current = copy.deepcopy(self.moves[action['id']])


def extra_node(graph, key):
    graph.add_node(dict(id=key, url='http://test/' + key, title=key, headings=[], actions=[]), [])


@pytest.mark.parametrize('current,steps', [('b', 1), ('c', 0)])
def test_return_prefers_direct_route_or_already_arrived(tmp_path, current, steps):
    graph = graph_at(tmp_path / 'graph')
    browser = Browser(graph, current)
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    result = nav.return_to_observed('c')
    assert result == {'success': True, 'steps': steps, 'route': 'recorded'}
    browser.page.goto.assert_not_called()
    assert browser.act.call_count == steps


@pytest.mark.parametrize('cap,success', [(1, False), (2, False), (3, True)])
def test_return_root_route_counts_goto_and_preserves_before_after(tmp_path, cap, success):
    graph = graph_at(tmp_path / 'graph')
    extra_node(graph, 'd')
    original = copy.deepcopy(graph.data)
    browser = Browser(graph, 'd')
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    nav.on_url_jump = Mock()
    result = nav.return_to_observed('c', cap)
    assert result['success'] is success
    assert graph.data == original
    if success:
        assert result['route'] == 'root_entry' and result['steps'] == 3
        assert browser.current['id'] == 'c'
        browser.page.goto.assert_called_once_with(graph.nodes['a']['url'], wait_until='domcontentloaded', timeout=20000)
        before, action, after = nav.on_url_jump.call_args.args
        assert before['id'] == 'd' and after['id'] == 'a' and action['kind'] == 'url_jump'
        assert nav.on_url_jump.call_args.kwargs == {'phase': 'root_entry'}
    else:
        assert result['steps'] == 0 and result['reason'] == 'no_recorded_route_within_cap'
        browser.page.goto.assert_not_called()
        browser.act.assert_not_called()


def test_over_budget_current_route_can_use_shorter_home_route(tmp_path):
    graph = graph_at(tmp_path / 'graph')
    for key in ('d', 'e'):
        extra_node(graph, key)
    for source, target in [('c', 'd'), ('d', 'e'), ('e', 'b')]:
        graph.add_edge(source, target, dict(id=source + target, kind='click', role='button', name='Open'))
    browser = Browser(graph)
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    result = nav.return_to_observed('b', 2)
    assert result == {'success': True, 'steps': 2, 'route': 'root_entry'}
    assert browser.act.call_count == 1


@pytest.mark.parametrize('blocked', ['unapproved', 'input', 'guard', 'destination', 'goto'])
def test_home_fallback_preserves_replay_checks_and_no_action_retry(tmp_path, blocked):
    graph = graph_at(tmp_path / 'graph')
    extra_node(graph, 'd')
    if blocked == 'unapproved':
        graph.data['edges'][0].update(replayable=False, replay_policy='execution-description-v2')
    elif blocked == 'input':
        graph.data['edges'][0]['action'].update(kind='type', value='old task input')
    browser = Browser(graph, 'd')
    if blocked == 'destination':
        browser.moves['ab'] = graph.nodes['d']
    elif blocked == 'goto':
        browser.goto_state = graph.nodes['d']
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    nav.action_guard = lambda *_: blocked == 'guard'
    result = nav.return_to_observed('c')
    assert not result['success']
    assert browser.page.goto.call_count == (blocked != 'unapproved')
    assert browser.act.call_count == (blocked == 'destination')
    if blocked == 'input':
        assert result['status'] == 'requires_input' and result['steps'] == 1
        assert nav.feedback()['steps'] == 1
    elif blocked == 'destination':
        assert result['status'] == 'requires_local_action' and result['steps'] == 2
        assert nav.feedback()['steps'] == 2
    elif blocked == 'goto':
        assert result['reason'] == 'root_entry_failed' and result['steps'] == 1


def test_failed_direct_replay_does_not_retry_via_home(tmp_path):
    graph = graph_at(tmp_path / 'graph')
    browser = Browser(graph, 'b')
    browser.act.side_effect = ValueError('click failed after attempt')
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    assert nav.return_to_observed('c')['reason'] == 'replay_failed'
    browser.act.assert_called_once()
    browser.page.goto.assert_not_called()


def test_changed_home_is_not_false_recovery_to_old_home_state(tmp_path):
    graph = graph_at(tmp_path / 'graph')
    browser = Browser(graph)
    browser.goto_state.update(id='new-root', headings=['New homepage'])
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    result = nav.return_to_observed('a')
    assert not result['success'] and result['reason'] == 'return_target_mismatch'
    assert result['steps'] == 1 and result['observed_state'] == 'new-root'
    assert nav.graph.data['root_entry']['current_state'] == 'new-root'
    browser.act.assert_not_called()


def test_actual_root_route_is_replanned_and_rechecked_against_remaining_cap(tmp_path):
    graph = graph_at(tmp_path / 'graph')
    # Historical open-menu root has a shorter route than the actual closed-menu root.
    graph.nodes['b']['url'] = graph.nodes['a']['url']
    graph.data['root_entry'] = {'current_state': 'b', 'states': ['a', 'b']}
    extra_node(graph, 'd')
    browser = Browser(graph, 'd')
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    result = nav.return_to_observed('c', 2)
    assert not result['success'] and result['steps'] == 1
    assert result['reason'] == 'no_recorded_route_within_cap'
    browser.act.assert_not_called()


@pytest.mark.parametrize('approved', [False, True])
def test_home_fallback_waits_for_pending_route_decision(tmp_path, approved):
    graph = graph_at(tmp_path / 'graph')
    extra_node(graph, 'd')
    browser = Browser(graph, 'd')
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    # Exercise the real dependency planner, resolving only the selected future.
    updater = ExecutionGraph.__new__(ExecutionGraph)
    updater.graph = nav.graph
    edge = nav.graph.data['edges'][0]
    edge.update(replayable=False, replay_policy='execution-description-v2')
    updater.description_jobs = SimpleNamespace(pending={1: {'record': {'edge': edge['id']}}})
    waits = []
    def poll(*, wait_for=()):
        if wait_for:
            waits.extend(wait_for)
            assert browser.page.goto.call_count == 0
            updater.description_jobs.pending.clear()
            edge['replayable'] = approved
            edge['annotation'] = {'status': 'completed'}
            updater.graph.invalidate_reachability()
    updater.poll_descriptions = poll
    # Root binding itself is not under test in this lightweight recorder.
    updater.observe = updater.save = updater.capture_before = Mock()
    updater.record = Mock()
    nav.recorder = updater
    result = nav.return_to_observed('c', 3)
    assert waits == [1] and result['success'] is approved, result
    assert browser.page.goto.call_count == int(approved)


def test_pending_route_wait_respects_cap_and_ignores_unrelated_jobs(tmp_path):
    graph = graph_at(tmp_path / 'graph')
    edge = graph.data['edges'][0]
    edge.update(replayable=False, replay_policy='execution-description-v2')
    updater = ExecutionGraph.__new__(ExecutionGraph)
    updater.graph = graph
    updater.description_jobs = SimpleNamespace(pending={1: {'record': {'edge': edge['id']}}})
    updater.poll_descriptions = Mock()
    updater.wait_for_route(['a'], 'c', max_actions=1)
    updater.wait_for_route(['c'], 'a', max_actions=12)
    assert updater.poll_descriptions.call_args_list == [call(), call()]


def test_over_budget_ready_path_does_not_hide_pending_shortcut(tmp_path):
    graph = graph_at(tmp_path / 'graph')
    key = graph.add_edge('a', 'c', dict(id='shortcut', kind='click', role='button', name='Shortcut'))
    edge = next(e for e in graph.data['edges'] if e['id'] == key)
    edge.update(replayable=False, replay_policy='execution-description-v2')
    updater = ExecutionGraph.__new__(ExecutionGraph)
    updater.graph = graph
    updater.description_jobs = SimpleNamespace(pending={1: {'record': {'edge': key}}})
    waits = []
    def poll(*, wait_for=()):
        if wait_for:
            waits.extend(wait_for)
            updater.description_jobs.pending.clear()
            edge.update(replayable=True, annotation={'status': 'completed'})
            graph.invalidate_reachability()
    updater.poll_descriptions = poll
    assert len(graph.path('a', 'c')) == 2
    updater.wait_for_route(['a'], 'c', max_actions=1)
    assert waits == [1] and len(graph.path('a', 'c')) == 1
