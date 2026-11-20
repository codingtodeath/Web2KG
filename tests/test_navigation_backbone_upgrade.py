import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_replay_eligibility import graph_at
from web2kg.common.backbone import shortest_backbone
from web2kg.common.browser import Browser
from web2kg.common.storage import Graph, EventLog
from web2kg.stage2.navigation import RouteNavigator
from web2kg.stage1.function_cards import representatives, card_input, request_card


@pytest.mark.parametrize('outer', ['<main aria-label="Content">{}</main>', '{}'])
def test_observation_and_click_use_same_nearest_region(outer):
    log = SimpleNamespace(emit=lambda *args, **kwargs: None)
    with Browser('about:blank', log, identity_mode='interaction', settle_ms=0) as browser:
        browser.page.set_content(outer.format('<a href="#outer">Open</a><nav aria-label="Links"><a href="#inner">Open</a></nav>'))
        action = next(a for a in browser.observe()['actions'] if a['href'].endswith('#outer'))
        assert action['match_count'] == 1
        browser.act(action)
        assert browser.page.url.endswith('#outer')


def test_bfs_backbone_ignores_cross_links_and_persists_random_shortest_parent(tmp_path):
    graph = graph_at(tmp_path)
    graph.add_edge('a', 'c', {'kind': 'click', 'role': 'link', 'name': 'C'})
    graph.add_edge('c', 'b', {'kind': 'click', 'role': 'link', 'name': 'B'})
    first = shortest_backbone(graph)
    assert first['depths'] == {'a': 0, 'b': 1, 'c': 1}
    assert first['parents']['b']['node'] == first['parents']['c']['node'] == 'a'
    graph.add_node({'id': 'd', 'url': 'http://test/d', 'title': 'D', 'headings': [], 'actions': []}, [])
    for source in ['b', 'c']:
        graph.add_edge(source, 'd', {'kind': 'click', 'role': 'link', 'name': 'D'})
    graph.data['backbone'] = first = shortest_backbone(graph)
    graph.data['edges'].reverse()
    assert shortest_backbone(graph) == first
    assert first['depths']['d'] == 2 and first['parents']['d']['node'] in {'b', 'c'}
    assert all(p is None or first['depths'][p['node']] + 1 == first['depths'][k]
               for k, p in first['parents'].items())


def test_retrieval_union_includes_root_and_current_branch(tmp_path):
    graph = graph_at(tmp_path / 'graph')
    graph.data['edges'][0]['replayable'] = False
    graph.invalidate_reachability()
    browser = SimpleNamespace(start_url=graph.data['start_url'])
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    current, allowed = nav.retrieval_scope(graph.nodes['b'])
    assert current == 'b' and allowed == {'a', 'b', 'c'}
    current, allowed = nav.retrieval_scope({'id': 'unknown', 'url': 'http://other/', 'headings': [], 'actions': []})
    assert current is None and allowed == {'a'}


@pytest.mark.parametrize('missing', [False, True, 'ambiguous'])
@pytest.mark.parametrize('operation', ['navigate', 'return_to_observed'])
def test_root_goto_accepts_new_state_and_rebinds_actual_control(site, tmp_path, missing, operation):
    graph = Graph(tmp_path / 'graph')
    with Browser(site, EventLog(graph.directory), identity_mode='interaction', settle_ms=0) as browser:
        root = browser.observe(graph.directory)
        graph.data.update(root=root['id'], start_url=root['url'])
        graph.add_node(root, [])
        action = next(a for a in root['actions'] if a['name'] == 'Catalog')
        browser.act(action)
        target = browser.observe(graph.directory)
        graph.add_node(target, [action])
        graph.add_edge(root['id'], target['id'], action)
        graph.save()
        original = copy.deepcopy(graph.data)
        nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
        if not missing:
            from web2kg.stage2.graph_update import ExecutionGraph
            nav.recorder = ExecutionGraph(graph, browser, EventLog(tmp_path / 'task'), template_clustering='off')
            nav.recorder.navigator = nav
        browser.page.goto(site + '/item/2')
        body = '<h1>Changed account homepage</h1>'
        if not missing:
            body += '<a href="/catalog">Catalog</a>'
            body += '<button onclick="document.querySelector(\'h1\').textContent=\'Latest homepage\'">Refresh view</button>'
        elif missing == 'ambiguous':
            body += '<a href="/catalog">Catalog</a><a href="/catalog">Catalog</a>'
        browser.page.route(site + '/', lambda route: route.fulfill(status=200, content_type='text/html', body=body))
        result = getattr(nav, operation)(target['id'])
        assert graph.data == original
        entry = nav.graph.data['root_entry']
        assert entry['reference_state'] == root['id'] and entry['current_state'] != root['id']
        assert entry['current_state'] in nav.graph.nodes
        if missing:
            assert not result['success'] and result['status'] == 'requires_local_action', result
            assert result['reason'] == 'root_control_unavailable' and result['steps'] == 1
            assert browser.page.url == site + '/'
        else:
            assert result['success'] and result['steps'] == 2, result
            assert browser.page.url == site + '/catalog'
            browser.page.goto(site + '/')
            before = browser.observe()
            nav.recorder.observe(before)
            action = next(a for a in before['actions'] if a['name'] == 'Refresh view')
            browser.act(action)
            after = browser.observe()
            nav.recorder.record(before, action, after)
            entry = nav.graph.data['root_entry']
            assert entry['current_state'] == after['id'] != before['id']
            nav.recorder.finish()
            updated = Graph.load(nav.recorder.graph.directory)
            assert updated.data['root_entry'] == entry
            assert updated.data['backbone']['root'] == entry['current_state']
            assert entry['current_state'] in nav.recorder.layer['assignments']
            assert any(e['source'] == before['id'] and e['target'] == target['id']
                       for e in updated.data['edges'])
            assert graph.data == original


def test_current_root_backbone_connects_historical_entry_without_browser_edges(tmp_path):
    graph = graph_at(tmp_path)
    graph.add_node({'id': 'new-root', 'url': graph.nodes['a']['url'], 'title': 'New',
                    'headings': [], 'actions': []}, [])
    graph.data['root_entry'] = {'current_state': 'new-root', 'states': ['a', 'new-root']}
    before = copy.deepcopy(graph.data)
    tree = shortest_backbone(graph)
    assert tree['root'] == 'new-root' and tree['root_unreachable'] == []
    assert tree['parents']['a'] == {'node': 'new-root', 'edge': 'root-entry:a'}
    assert tree['depths']['new-root'] == 0 and tree['depths']['a'] == 1
    assert graph.data == before and graph.path('new-root', 'a') is None


def test_root_goto_replans_from_actual_version_instead_of_old_open_menu(tmp_path):
    graph = graph_at(tmp_path / 'graph')
    graph.nodes['b']['url'] = graph.nodes['a']['url']
    graph.data['root_entry'] = {'current_state': 'b', 'states': ['a', 'b']}
    outside = {'id': 'outside', 'url': 'http://test/outside', 'headings': [], 'actions': []}
    browser = SimpleNamespace(start_url=graph.data['start_url'], page=SimpleNamespace(goto=Mock()),
        settle=Mock(), act=Mock(), observe=Mock(side_effect=[outside, graph.nodes['a'],
            graph.nodes['a'], graph.nodes['b'], graph.nodes['b'], graph.nodes['c']]))
    nav = RouteNavigator(browser, graph, EventLog(tmp_path / 'nav'))
    result = nav.navigate('c')
    assert result['success'] and result['steps'] == 3
    assert browser.act.call_count == 2
    assert [r.args[0]['name'] for r in browser.act.call_args_list] == [
        graph.data['edges'][0]['action']['name'], graph.data['edges'][1]['action']['name']]


def test_five_distant_visual_samples_have_only_compact_model_context(tmp_path):
    graph = Graph(tmp_path)
    for i in range(7):
        key = str(i)
        graph.data['nodes'][key] = {'id': key, 'title': f'Page {i}', 'headings': [str(i)],
            'actions': [], 'page_description': {'text': f'Description {i}'}}
        folder = tmp_path / 'snapshots' / key
        folder.mkdir(parents=True)
        (folder / 'page.png').write_bytes(b'fixture image')
    group = {'id': 'T1', 'members': list(graph.nodes), 'representative': '0'}
    selected = representatives(graph, group)
    assert len(selected) == len(set(selected)) == 5
    assert representatives(graph, {**group, 'members': ['0', '1']}) == ['0', '1']
    evidence = card_input(graph, {}, group)
    def respond(prompt, payload, schema, purpose, images):
        assert len(images) == 5 and all(p.is_file() for p in images)
        assert all(set(p) == {'title', 'page_description', 'evidence_ref'} for p in payload['representatives'])
        assert 'incoming' not in payload and 'node_id' not in json.dumps(payload)
        return {'name': 'Pages', 'summary': 'Summary', 'common_functions': [], 'importance': .5,
            'variants': [{'when': 'Varies', 'functions': [], 'evidence_nodes': ['P1', 'P5']}], 'uncertainties': []}
    result = request_card(evidence, SimpleNamespace(json=respond))
    assert result['variants'][0]['evidence_nodes'] == [selected[0], selected[4]]
