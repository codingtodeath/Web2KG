import json
from types import SimpleNamespace

from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog, Graph, write_json


def test_compact_graph_preserves_data_and_atomic_replacement(tmp_path):
    graph = Graph(tmp_path)
    graph.data['nodes']['example'] = {'text': '中文\n"quoted"', 'score': 0.125,
                                     'state': [True, False, None], 'nested': {'x': [1, 2]}}
    pretty = tmp_path / 'pretty.json'
    write_json(pretty, graph.data)
    graph.save()
    assert Graph.load(tmp_path).data == json.loads(pretty.read_text(encoding='utf-8'))
    assert (tmp_path / 'graph.json').stat().st_size < pretty.stat().st_size
    original = (tmp_path / 'graph.json').read_bytes()
    graph.data['invalid'] = object()
    try:
        graph.save()
    except TypeError:
        pass
    else:
        raise AssertionError('Unserializable graph must fail')
    assert (tmp_path / 'graph.json').read_bytes() == original


def test_observation_peer_index_preserves_identity_and_refreshes(tmp_path):
    def ax_node(i, role, parent=None, name=''):
        return {'nodeId': str(i), 'backendDOMNodeId': i, 'parentId': str(parent),
                'role': {'value': role}, 'name': {'value': name}}

    def dom_node(i, tag, children=(), attrs=()):
        return {'nodeType': 1, 'nodeName': tag, 'backendNodeId': i,
                'children': list(children), 'attributes': list(attrs)}

    ax = [ax_node(1, 'RootWebArea'), ax_node(2, 'navigation', 1, 'Menu'),
          ax_node(3, 'link', 2, 'Open'), ax_node(4, 'link', 2, 'Open'),
          ax_node(5, 'navigation', 1, 'Menu'), ax_node(6, 'link', 5, 'Open')]
    dom = dom_node(1, 'HTML', [
        dom_node(2, 'NAV', [dom_node(3, 'A', attrs=['href', '/one']),
                            dom_node(4, 'A', attrs=['href', '/two'])]),
        dom_node(5, 'NAV', [dom_node(6, 'A', attrs=['href', '/three'])])])
    calls = []

    def send(method, *args):
        calls.append(method)
        return {'nodes': ax} if method.startswith('Accessibility') else {'root': dom}

    browser = object.__new__(Browser)
    browser.cdp = SimpleNamespace(send=send)
    browser.page = SimpleNamespace(url='http://fixture.local/', title=lambda: 'Fixture',
        evaluate=lambda script, paths=None: [False] * len(paths) if paths is not None else 'content')
    browser.identity_mode, browser.document_response = 'semantic', None
    browser.log = EventLog(tmp_path)
    first = browser.observe()
    assert [(a['number'], a['occurrence'], a['match_count']) for a in first['actions']] == [
        (1, 0, 2), (2, 1, 2), (3, 0, 1)]
    assert [a['region']['occurrence'] for a in first['actions']] == [0, 0, 1]
    assert all(a['region']['match_count'] == 2 for a in first['actions'])
    assert [a['href'] for a in first['actions']] == [
        'http://fixture.local/one', 'http://fixture.local/two', 'http://fixture.local/three']
    # An asynchronous DOM change must be observed even without a browser action.
    dom['children'][0]['children'][0]['attributes'][1] = '/changed'
    second = browser.observe()
    assert second['id'] != first['id']
    assert second['actions'][0]['href'].endswith('/changed')
    assert len(calls) == 4
    events = [json.loads(line) for line in browser.log.path.read_text(encoding='utf-8').splitlines()]
    assert len(events) == 2 and all(e['operation'] == 'browser_observe' for e in events)
