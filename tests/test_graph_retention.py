import os
import copy

import pytest

from web2kg.common.storage import Graph, EventLog
from web2kg.stage1.templates import next_template_id
from web2kg.stage2.retention import record_retrieval, trim_graph


def small_graph(tmp_path, edges, groups):
    graph = Graph(tmp_path)
    graph.data['root'] = 'root'
    graph.data['start_url'] = 'https://fixture.test'
    layer = {'templates': [], 'assignments': {}, 'profiles': {}, 'snapshot_sha256': {},
             'transitions': [], 'decisions': []}
    for template, members in groups.items():
        layer['templates'].append({'id': template, 'members': members[:], 'representative': members[0]})
        for key in members:
            graph.nodes[key] = {'id': key, 'url': graph.data['start_url'], 'headings': []}
            layer['assignments'][key] = {'template': template, 'parent_node': None, 'parent_template': None}
            layer['profiles'][key] = {}
            folder = tmp_path / 'snapshots' / key
            folder.mkdir(parents=True)
            (folder / 'page.png').write_bytes(key.encode())
    for a, b in edges:
        edge = graph.add_edge(a, b, {'kind': 'click'})
        layer['transitions'].append({'source': layer['assignments'][a]['template'],
            'target': layer['assignments'][b]['template'], 'evidence_edges': [edge]})
    return graph, layer


def test_retrieval_rounds_importance_redundancy_and_idle_control_eviction(tmp_path):
    graph, layer = small_graph(tmp_path, [('root', k) for k in ('old', 'used', 'important')],
                              {'T001': ['root'], 'T002': ['old', 'used'], 'T003': ['important']})
    for _ in range(40):
        record_retrieval(graph, ['used'])
    # Important singleton survives equal idle time; frequently retrieved redundant page also survives.
    result = trim_graph(graph, layer, {'T002': {'importance': .5}, 'T003': {'importance': 1}}, max_nodes=3)
    assert [r['node'] for r in result['removed']] == ['old']
    assert graph.nodes['used']['last_retrieved_round'] == 40
    assert layer['templates'][1]['members'] == ['used']
    assert layer['templates'][1]['representative'] == 'used'
    assert all('old' not in (e['source'], e['target']) for e in graph.data['edges'])
    assert not (tmp_path / 'snapshots/old').exists()


def test_leaf_recomputation_retirement_and_fresh_protection(tmp_path):
    graph, layer = small_graph(tmp_path, [('root', 'parent'), ('parent', 'leaf'), ('root', 'new')],
                              {'T001': ['root'], 'T002': ['parent'], 'T003': ['leaf'], 'T004': ['new']})
    result = trim_graph(graph, layer, {}, max_nodes=1, protected=['new'])
    assert [r['node'] for r in result['removed']] == ['leaf', 'parent']
    assert result['status'] == 'limited_by_protection_or_no_leaves'
    assert set(graph.nodes) == {'root', 'new'}
    assert layer['retired_templates'] == {'T002': None, 'T003': None}
    # Protection is supplied by this task only; the next boundary can delete the former newcomer.
    result = trim_graph(graph, layer, {}, max_nodes=1)
    assert set(graph.nodes) == {'root'} and result['after'] == 1
    assert next_template_id(layer) == 'T005'


def test_cycle_prunes_by_backbone_and_local_hardlink_unlink_preserves_source(tmp_path):
    graph, layer = small_graph(tmp_path / 'task', [('root', 'a'), ('a', 'b'), ('b', 'a'), ('root', 'leaf')],
                              {'T001': ['root'], 'T002': ['a', 'b'], 'T003': ['leaf']})
    original = tmp_path / 'original.png'
    os.link(graph.directory / 'snapshots/leaf/page.png', original)
    result = trim_graph(graph, layer, {}, max_nodes=1)
    assert set(graph.nodes) == {'root'}
    assert result['remaining_excess'] == 0
    assert original.read_bytes() == b'leaf'
    unchanged = copy.deepcopy(layer)
    graph_revision = graph.replay_revision
    trim_graph(graph, layer, {}, max_nodes=1)
    assert layer == unchanged and graph.replay_revision == graph_revision


def test_card_requests_importance_and_accepts_legacy_cards(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from web2kg.stage1.function_cards import request_card, validate_card
    from web2kg.stage1.function_cards import file_hash
    image = tmp_path / 'page.png'; image.write_bytes(b'fixture')
    evidence = {'template_id': 'T1', 'members': 1, 'representatives': [{'node_id': 'a', 'title': 'A',
        'page_description': 'Page A', 'screenshot': {'path': str(image), 'sha256': file_hash(image)}}]}
    answer = {'name': 'Entry', 'summary': 'Primary site function', 'common_functions': [],
              'variants': [], 'uncertainties': [], 'importance': .9}
    client = SimpleNamespace(json=Mock(return_value=answer))
    assert request_card(evidence, client)['importance'] == .9
    assert 'importance' in client.json.call_args.args[2]['required']
    validate_card({k: v for k, v in answer.items() if k != 'importance'}, evidence)
    with pytest.raises(ValueError, match='importance'):
        validate_card({**answer, 'importance': 1.5}, evidence)


from test_function_cards import saved
from test_execution_assets import indexed_copy
from test_execution_graph import hashes
from web2kg.stage2.graph_update import ExecutionGraph
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.stage2.retrieval import retrieve
from web2kg.common.browser import Browser


def test_finish_prunes_assets_without_model_calls_and_next_task_loads(saved, tmp_path):
    base, cards, encoder = indexed_copy(saved, tmp_path)
    original = hashes(base.directory)
    log = EventLog(tmp_path / 'task')
    knowledge = TemplateKnowledge(base, cards, log)
    updater = ExecutionGraph(base, None, log, knowledge, index_client=encoder, max_graph_nodes=2)
    product = saved[-1][-1]['id']
    retired = updater.layer['assignments'][product]['template']
    updater.retrieved([product])
    assert len(updater.graph.nodes) == 3  # Never evict while a task is running.
    result = updater.finish()
    assert result['retention']['after'] == 2 and product not in updater.graph.nodes
    assert result['index']['encoded'] == 0
    assert not encoder.calls
    assert retired not in updater.live_cards.data['cards']
    assert retired in updater.layer['retired_templates']
    assert all(product != r['id'] for r in retrieve(updater.graph, 'content', encoder, 5))
    next_log = EventLog(tmp_path / 'next')
    context = TemplateKnowledge(updater.graph, updater.live_cards.directory / 'cards.json', next_log)
    next_task = ExecutionGraph(updater.graph, None, next_log, context, index_client=encoder)
    assert len(next_task.graph.nodes) == 2
    assert not (next_task.graph.directory / 'snapshots' / product).exists()
    assert updater.finish() == result
    assert hashes(base.directory) == original
    with pytest.raises(ValueError, match='finalization'):
        updater.retrieved([])


def test_reappearance_after_pruning_keeps_history_and_uses_new_template_id(saved, tmp_path):
    base, cards, encoder = indexed_copy(saved, tmp_path)
    original = hashes(base.directory)
    log = EventLog(tmp_path / 'task')
    knowledge = TemplateKnowledge(base, cards, log)
    updater = ExecutionGraph(base, None, log, knowledge, index_client=encoder, max_graph_nodes=2)
    product = saved[-1][-1]
    retired = updater.layer['assignments'][product['id']]['template']
    updater.finish()
    next_log = EventLog(tmp_path / 'next')
    context = TemplateKnowledge(updater.graph, updater.live_cards.directory / 'cards.json', next_log)
    with Browser(product['url'], next_log, identity_mode='interaction', settle_ms=20) as browser:
        next_task = ExecutionGraph(updater.graph, browser, next_log, context, index_client=encoder, max_graph_nodes=2)
        observed = browser.observe()
        assert observed['id'] == product['id']
        assert next_task.observe(observed)
        assert next_task.layer['assignments'][product['id']]['template'] != retired
        next_task.finish()
        assert product['id'] in next_task.graph.nodes  # Reintroduced node is protected for this task.
    assert hashes(base.directory) == original


def test_one_finish_removes_multiple_layers_and_next_task_inherits_consistent_assets(saved, tmp_path):
    base, cards, encoder = indexed_copy(saved, tmp_path)
    original = hashes(base.directory)
    log = EventLog(tmp_path / 'task')
    knowledge = TemplateKnowledge(base, cards, log)
    updater = ExecutionGraph(base, None, log, knowledge, index_client=encoder, max_graph_nodes=1)
    result = updater.finish()
    assert len(result['retention']['removed']) == 2
    assert result['retention']['removed'][0]['exposed_leaves']
    assert set(updater.graph.nodes) == {base.data['root']}
    assert len(updater.layer['templates']) == 1 and not updater.layer['transitions']
    assert len(updater.layer['retired_templates']) == 2
    assert result['index']['pages'] == 1 and result['index']['encoded'] == 0
    assert not encoder.calls
    assert len(updater.live_cards.data['cards']) == 1
    next_log = EventLog(tmp_path / 'next')
    context = TemplateKnowledge(updater.graph, updater.live_cards.directory / 'cards.json', next_log)
    next_task = ExecutionGraph(updater.graph, None, next_log, context, index_client=encoder, max_graph_nodes=1)
    assert next_task.finish()['retention']['status'] == 'within_limit'
    assert hashes(base.directory) == original
