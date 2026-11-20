import copy
import json
import re
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from test_replay_eligibility import graph_at
from test_visual_navigation import FakeJina
from web2kg.common.browser import Browser
from web2kg.common.jina import sha256
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.stage1.incremental_index import extend_visual_index
from web2kg.stage1.indexing import build_index
from web2kg.stage1.visual_index import build_visual_index
from web2kg.stage2.agent import run_task
from web2kg.stage2.navigation import RouteNavigator
from web2kg.stage2.retrieval import retrieve


@pytest.mark.parametrize('kind', ['click', 'type'])
def test_disconnected_branch_uses_current_route_and_preserves_input_boundary(tmp_path, kind):
    graph = graph_at(tmp_path / 'graph')
    graph.data['edges'][0]['replayable'] = False
    graph.data['edges'][1]['action'].update(kind=kind, value='historical input')
    graph.invalidate_reachability()
    browser = SimpleNamespace(start_url=graph.data['start_url'], act=Mock(),
        observe=Mock(side_effect=[graph.nodes['b'], graph.nodes['b'], graph.nodes['c']]))
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    assert nav.graph.reachable() == {'a'}
    result = nav.navigate('c')
    if kind == 'click':
        assert result['success'] and result['steps'] == 1
        browser.act.assert_called_once_with(graph.data['edges'][1]['action'], 'role')
    else:
        assert not result['success'] and result['status'] == 'requires_input' and result['steps'] == 0
        browser.act.assert_not_called()


def test_navigation_rechecks_actual_state_after_retrieval(tmp_path):
    graph = graph_at(tmp_path / 'graph')
    graph.data['edges'][0]['replayable'] = False
    graph.invalidate_reachability()
    build_index(graph)
    browser = SimpleNamespace(start_url=graph.data['start_url'], act=Mock(),
                              observe=Mock(return_value=graph.nodes['b']))
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    assert 'c' in {r['id'] for r in retrieve(graph, 'c', allowed_ids=nav.graph.reachable('b'))}
    browser.observe.return_value = graph.nodes['a']
    assert nav.navigate('c')['reason'] == 'no_recorded_route_from_current_or_root'
    browser.act.assert_not_called()


def legacy_root_only_index(graph, client, tiled):
    """Save a valid old-format subset without using the new indexing policy."""
    full = build_visual_index(graph, client, tiles=tiled, tile_height=40, overlap=8)
    old = copy.deepcopy(full)
    old.pop('node_scope')
    old.update(ids=['a'], texts=full['texts'][:1], images_sha256={'a': full['images_sha256']['a']})
    if tiled:
        old['tiling']['pages'] = {'a': old['tiling']['pages']['a']}
    with np.load(graph.directory / full['vectors']) as vectors:
        kept = {k: vectors[k] for k in vectors.files if k == 'a' or k.startswith('a__tile_')}
    old['vectors'] = 'legacy-vectors.npz'
    np.savez(graph.directory / old['vectors'], **kept)
    old['vectors_sha256'] = sha256(graph.directory / old['vectors'])
    write_json(graph.directory / 'index.json', old)
    if not tiled:
        write_json(graph.directory / 'index-whole.json', old)
    client.calls.clear()
    return old


@pytest.mark.parametrize('tiled', [False, True])
def test_old_partial_index_is_extended_without_changing_source_or_reencoding(tmp_path, tiled):
    base = graph_at(tmp_path / 'base')
    base.data['edges'][0]['replayable'] = False
    base.invalidate_reachability()
    client = FakeJina()
    old = legacy_root_only_index(base, client, tiled)
    original = {p.relative_to(base.directory): sha256(p) for p in base.directory.rglob('*') if p.is_file()}
    task = Graph(tmp_path / 'task')
    task.data = copy.deepcopy(base.data)
    shutil.copytree(base.directory / 'snapshots', task.directory / 'snapshots')
    result = extend_visual_index(base, task, client, EventLog(task.directory))
    assert result['ids'] == ['a', 'b', 'c'] and result['node_scope'] == 'all_graph_nodes'
    assert result['incremental']['reused_ids'] == ['a']
    assert result['incremental']['encoded_ids'] == ['b', 'c']
    assert len(client.calls) == (8 if tiled else 2)
    with np.load(base.directory / old['vectors']) as before, np.load(task.directory / result['vectors']) as after:
        assert all(np.array_equal(before[k], after[k]) for k in before.files)
    assert [r['id'] for r in retrieve(task, 'b', client, 1, allowed_ids=task.reachable('b'))] == ['b']
    assert [r['id'] for r in retrieve(task, 'b', client, 1)] == ['a']
    assert original == {p.relative_to(base.directory): sha256(p) for p in base.directory.rglob('*') if p.is_file()}


@pytest.mark.parametrize('tiled', [False, True])
def test_full_rebuild_can_reuse_partial_whole_index(tmp_path, tiled):
    graph, client = graph_at(tmp_path), FakeJina()
    graph.data['edges'][0]['replayable'] = False
    graph.invalidate_reachability()
    old = legacy_root_only_index(graph, client, False)
    result = build_visual_index(graph, client, tiles=tiled, tile_height=40, overlap=8, reuse_whole=True)
    assert result['ids'] == ['a', 'b', 'c']
    whole = json.loads((tmp_path / 'index-whole.json').read_text())
    assert whole['ids'] == ['a', 'b', 'c'] and 'tiling' not in whole
    assert not any(path.name == 'page.png' and path.parent.name == 'a' for path, _ in client.calls)
    assert len(client.calls) == (11 if tiled else 2)
    with np.load(tmp_path / old['vectors']) as before, np.load(tmp_path / whole['vectors']) as after:
        assert np.array_equal(before['a'], after['a'])


@pytest.mark.parametrize('unknown_current', [False, True])
def test_real_actor_retrieval_uses_current_or_root_routes(site, tmp_path, unknown_current):
    graph = Graph(tmp_path / 'graph')
    with Browser(site, EventLog(graph.directory), identity_mode='interaction', settle_ms=20) as browser:
        root = browser.observe(graph.directory)
        graph.data.update(root=root['id'], start_url=root['url'], identity_mode='interaction')
        graph.add_node(root, [])
        for name in ['Catalog', 'Red mug']:
            before = browser.observe()
            action = next(a for a in before['actions'] if a['name'] == name)
            browser.act(action)
            after = browser.observe(graph.directory)
            graph.add_node(after, [])
            graph.add_edge(before['id'], after['id'], action)
            if name == 'Catalog':
                branch = after['id']
                graph.data['edges'][-1]['replayable'] = False
        if unknown_current:
            graph.nodes[branch]['headings'] = ['Outdated saved catalog']
    graph.invalidate_reachability()
    assert graph.reachable() == {root['id']}
    graph.save()
    build_index(graph)
    calls, selections = [], []
    class Client:
        def call(self, messages, system_prompt='', **kwargs):
            if 'target_page' in system_prompt:
                selections.append(1)
                if unknown_current:
                    return json.dumps({'target_page': root['id'] + '.png'})
                return json.dumps({'target_page': after['id'] + '.png'})
            calls.append(messages)
            if len(calls) == 1:
                control = re.search(r"link \[(\d+)\] 'Catalog'", json.dumps(messages))[1]
                return f'ACTION: click [{control}]'
            if len(calls) == 2:
                return 'ACTION: optimal_navigate [Red mug] [reddit] [Product]'
            return 'ACTION: stop [done]'
    result = run_task(graph, 'Open Red mug', EventLog(tmp_path / 'run'), Client(), None,
                      max_steps=4, final_check=lambda b: b.page.url == site + '/item/2')
    events = [json.loads(line) for line in (tmp_path / 'run/events.jsonl').read_text().splitlines()]
    retrieval = next(e for e in events if e['event'] == 'retrieval')
    if unknown_current:
        assert retrieval['source'] is None and {r['id'] for r in retrieval['candidates']} == {root['id']}
        assert selections == [1] and not result['local_success']
        assert result['status'] == 'agent_stopped', result
        assert result['url'] == site + '/'
        assert result['navigation']['success'] and result['navigation']['steps'] == 1
    else:
        assert retrieval['source'] == branch
        assert {r['id'] for r in retrieval['candidates']} == {root['id'], branch, after['id']}
        assert selections == [1] and result['local_success'], result
        assert result['navigation']['success'] and result['navigation']['steps'] == 1
