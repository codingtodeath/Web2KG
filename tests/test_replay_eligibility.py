import copy
import json
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from test_llm_exploration import FakeClient
from test_visual_navigation import FakeJina
from web2kg.common.replay import execute_path
from web2kg.common.storage import EventLog, Graph
from web2kg.stage1.indexing import build_index
from web2kg.stage1.incremental_index import extend_visual_index
from web2kg.stage1.llm_exploration import LLMExplorer
from web2kg.stage1.visual_index import build_visual_index
from web2kg.stage2.navigation import RouteNavigator
from web2kg.stage2.retrieval import retrieve


def graph_at(directory):
    graph = Graph(directory)
    graph.data.update(root='a', start_url='http://test/a')
    for key in ('a', 'b', 'c'):
        graph.add_node({'id': key, 'url': 'http://test/' + key, 'title': key,
                        'headings': [], 'text': key, 'actions': []}, [])
        path = directory / 'snapshots' / key / 'page.png'
        path.parent.mkdir(parents=True)
        Image.new('RGB', (20, 80), 'white').save(path)
    for source, target in [('a', 'b'), ('b', 'c')]:
        graph.add_edge(source, target, {'id': source + target, 'kind': 'click', 'role': 'button', 'name': 'Reply'})
        graph.data['edges'][-1].update(replayable=True, replay_policy='luna-v1')
    return graph


@pytest.mark.parametrize('decision', [True, False, 'true', None])
def test_luna_decision_overrides_keyword_rules_and_fails_closed(tmp_path, decision):
    graph = graph_at(tmp_path)
    client = FakeClient(EventLog(tmp_path))
    original = client.json
    def respond(*args):
        result = original(*args)
        if decision is None:
            del result['replayable']
        else:
            result['replayable'] = decision
        return result
    client.json = respond
    edge = graph.data['edges'][0]
    explorer = LLMExplorer(client, tmp_path)
    args = (graph, graph.nodes['a'], edge['action'], graph.nodes['b'], 1, 'transition', 'agent', True, edge['id'])
    if type(decision) is not bool:
        with pytest.raises(RuntimeError):
            explorer.describe(*args, before=graph.nodes['a'])
    else:
        explorer.describe(*args, before=graph.nodes['a'])
    assert edge['replayable'] is (decision is True)
    assert graph.data['transition_records'][-1]['replayable'] is (decision is True)
    assert graph.reachable() == ({'a', 'b', 'c'} if decision is True else {'a'})


def test_luna_approved_unknown_label_replays_in_both_executors(tmp_path):
    graph = graph_at(tmp_path)
    edge = graph.data['edges'][0]
    browser = SimpleNamespace(observe=Mock(side_effect=[graph.nodes['a'], graph.nodes['b']]), act=Mock())
    assert execute_path(browser, graph, [edge], EventLog(tmp_path))['success']
    browser.observe = Mock(side_effect=[graph.nodes['a'], graph.nodes['b']])
    browser.start_url = graph.data['start_url']
    navigator = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    assert navigator.replay([edge])['success']
    assert browser.act.call_count == 2


@pytest.mark.parametrize('visual', [False, True])
def test_unreachable_descendants_excluded_before_topk_even_with_stale_index(tmp_path, visual):
    graph = graph_at(tmp_path)
    client = FakeJina()
    (build_visual_index(graph, client) if visual else build_index(graph))
    graph.data['edges'][0]['replayable'] = False
    graph.invalidate_reachability()
    assert [row['id'] for row in retrieve(graph, 'b', client if visual else None, 1)] == ['a']
    client.calls.clear()
    data = build_visual_index(graph, client) if visual else build_index(graph)
    assert data['ids'] == ['a', 'b', 'c']
    assert data['node_scope'] == 'all_graph_nodes'
    assert [row['id'] for row in retrieve(graph, 'b', client if visual else None, 1,
            allowed_ids=graph.reachable('b'))] == ['b']
    assert retrieve(graph, 'b', client if visual else None, 1, allowed_ids=set()) == []
    assert [row['id'] for row in retrieve(graph, 'b', client if visual else None, 1)] == ['a']
    graph.add_edge('a', 'c', {'id': 'ac', 'kind': 'click', 'role': 'link', 'name': 'Comments', 'href': 'http://test/c'})
    assert graph.reachable() == {'a', 'c'}


@pytest.mark.parametrize('tiled', [False, True])
def test_incremental_keeps_vectors_when_root_path_becomes_blocked(tmp_path, tiled):
    base = graph_at(tmp_path / 'base')
    client = FakeJina()
    build_visual_index(base, client, tiles=tiled, tile_height=40, overlap=8)
    task = Graph(tmp_path / 'task')
    task.data = copy.deepcopy(base.data)
    shutil.copytree(base.directory / 'snapshots', task.directory / 'snapshots')
    task.data['edges'][0]['replayable'] = False
    task.invalidate_reachability()
    client.calls.clear()
    filtered = extend_visual_index(base, task, client, EventLog(task.directory))
    assert filtered['ids'] == ['a', 'b', 'c'] and not client.calls
    with np.load(task.directory / filtered['vectors']) as vectors:
        assert {'a', 'b', 'c'} <= set(vectors.files)
    third = Graph(tmp_path / 'third')
    third.data = copy.deepcopy(task.data)
    shutil.copytree(task.directory / 'snapshots', third.directory / 'snapshots')
    third.data['edges'][0]['replayable'] = True
    third.invalidate_reachability()
    restored = extend_visual_index(task, third, client, EventLog(third.directory))
    assert restored['incremental']['encoded_ids'] == [] and not client.calls
    assert set(restored['ids']) == {'a', 'b', 'c'}


def test_navigation_rejects_ineligible_target_without_browser_actions(tmp_path):
    graph = graph_at(tmp_path)
    graph.data['edges'][0]['replayable'] = False
    graph.invalidate_reachability()
    browser = Mock(start_url=graph.data['start_url'])
    browser.observe.return_value = graph.nodes['a']
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    assert nav.navigate('c')['reason'] == 'no_recorded_route_from_current_or_root'
    browser.act.assert_not_called()
    browser.observe.assert_called_once()


def test_live_navigation_decision_filters_immutable_source_index(tmp_path):
    graph = graph_at(tmp_path)
    build_index(graph)
    nav = Graph(tmp_path / 'navigation')
    nav.data = copy.deepcopy(graph.data)
    nav.data['edges'][0]['replayable'] = False
    nav.invalidate_reachability()
    assert [r['id'] for r in retrieve(graph, 'b', top_k=1, allowed_ids=nav.reachable())] == ['a']
    assert graph.data['edges'][0]['replayable'] is True


def test_runtime_constraint_does_not_turn_luna_approval_into_unsupported_replay(tmp_path):
    graph = graph_at(tmp_path)
    edge = graph.data['edges'][0]
    edge['action']['kind'] = 'type'  # Browser command lacks a saved replay value.
    client = FakeClient(EventLog(tmp_path))
    LLMExplorer(client, tmp_path).describe(graph, graph.nodes['a'], edge['action'], graph.nodes['b'],
        1, 'transition', 'agent', True, edge['id'], before=graph.nodes['a'])
    assert edge['description']['annotation']['replayable'] is True
    assert edge['replayable'] is True
    assert edge['replay_policy'] == 'actor-input-boundary-v1'
