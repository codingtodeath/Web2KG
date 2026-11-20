import copy
import json
import shutil
from unittest.mock import Mock

import pytest

from test_function_cards import saved, CardModel
from test_execution_graph import hashes
from test_visual_navigation import FakeJina
from web2kg.common.browser import Browser
from web2kg.common.storage import Graph, EventLog, digest, write_json
from web2kg.stage1.function_cards import build_cards, file_hash, load_layer
from web2kg.stage1.site_framework import build_framework, load_framework
from web2kg.stage1.templates import build_templates, next_template_id
from web2kg.stage1.visual_index import build_visual_index
from web2kg.stage2.graph_update import ExecutionGraph
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.stage2.recovery import ExperienceStore, template_key
from web2kg.stage2.retrieval import retrieve


def fragmented_graph(saved, site, tmp_path):
    original, _, _, _, (_, catalog, product) = saved
    shutil.copytree(original.directory, tmp_path / 'base')
    graph = Graph.load(tmp_path / 'base')
    with Browser(site + '/catalog', EventLog(tmp_path / 'capture'), identity_mode='interaction', settle_ms=20) as browser:
        before = browser.observe()
        action = next(a for a in before['actions'] if a['name'] == 'Green mug')
        browser.act(action)
        after = browser.observe(graph.directory)
        graph.add_node(after, graph.nodes[catalog['id']]['path'] + [action])
        graph.add_edge(catalog['id'], after['id'], action)
    graph.save()
    layer = build_templates(graph)
    old = layer['assignments'][product['id']]['template']
    assert layer['assignments'][after['id']]['template'] == old
    new = next_template_id(layer)
    group = next(g for g in layer['templates'] if g['id'] == old)
    group['members'].remove(after['id'])
    group['representative'] = product['id']
    layer['templates'].append({'id': new, 'members': [after['id']], 'representative': after['id']})
    layer['assignments'][after['id']]['template'] = new
    for row in layer['transitions'][:]:
        target_edges = [e['id'] for e in graph.data['edges'] if e['target'] == after['id'] and e['id'] in row['evidence_edges']]
        if target_edges:
            row['evidence_edges'] = [k for k in row['evidence_edges'] if k not in target_edges]
            layer['transitions'].append({**row, 'id': digest([row['source'], new, row['control']]),
                                         'target': new, 'evidence_edges': target_edges})
    layer['transitions'] = [r for r in layer['transitions'] if r['evidence_edges']]
    path = graph.directory / 'templates/split/templates.json'
    write_json(path, layer)
    write_json(graph.directory / 'templates/active.json', {'directory': 'templates/split'})
    cards = graph.directory / 'function-cards/split'
    build_cards(graph, path, cards, CardModel(tmp_path / 'seed-cards'))
    model = Mock(model='fixture-luna', last_request_id='fixture')
    model.json.return_value = {'overview': 'Historical product groups', 'areas': [
        {'name': 'Products', 'purpose': 'Read products', 'templates': [old, new]}], 'navigation': [], 'limitations': []}
    build_framework(graph, cards / 'cards.json', model, graph.directory / 'site-framework/test', publish=True)
    build_visual_index(graph, FakeJina())
    return graph, cards / 'cards.json', old, new, product['id']


@pytest.mark.parametrize('allow_cards', [True, False])
def test_merge_end_to_end_preserves_evidence_and_can_start_next_task(saved, site, tmp_path, allow_cards):
    base, cards, old, split, node = fragmented_graph(saved, site, tmp_path)
    originals = hashes(base.directory)
    _, base_layer = load_layer(base)
    memory = ExperienceStore(tmp_path / 'memory.json', 'fixture')
    old_key = template_key(base_layer, old)
    write_json(memory.path, {'version': 1, 'namespace': 'fixture', 'pages': [
        {'node': node, 'experience': {'lesson': 'Existing page lesson'}}],
        'templates': {old_key: {'experience': {'lesson': 'Old group lesson'}}}})
    memory_before = memory.path.read_bytes()
    log = EventLog(tmp_path / 'run')
    knowledge = TemplateKnowledge(base, cards, log)
    model, encoder = CardModel(tmp_path / 'refresh'), FakeJina()
    with Browser(site, log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(base, browser, log, knowledge, card_client=model, index_client=encoder,
                                 template_clustering='legacy')
        write_json(updater.graph.directory / 'initial-website-framework.json', load_framework(base))
        old_layer = updater.graph.directory / 'templates/execution/templates.json'
        result = updater.finish(allow_card_calls=allow_cards)
        merged = result['template_reconciliation']['mapping'][old]
        assert merged == result['template_reconciliation']['mapping'][split] and merged not in {old, split}
        assert result['template_reconciliation']['templates_after'] == result['template_reconciliation']['templates_before'] - 1
        active = json.loads((updater.graph.directory / 'templates/active.json').read_text())
        assert active['directory'] == 'templates/finalized'
        historical = json.loads((updater.graph.directory / 'function-cards/execution/cards.json').read_text(encoding='utf-8'))
        assert historical['source']['layer_sha256'] == file_hash(old_layer)
        assert old in historical['cards'] and split in historical['cards']
        final_cards = updater.graph.directory / 'function-cards/finalized/cards.json'
        final = json.loads(final_cards.read_text(encoding='utf-8'))
        assert old not in final['cards'] and split not in final['cards']
        assert final['cards'][merged]['status'] == ('completed' if allow_cards else 'not_generated')
        assert 'previous_card' not in final['cards'][merged]
        assert set(final['cards'][merged]['merged_template_sources']) == {old, split}
        checked = TemplateKnowledge(updater.graph, final_cards, log)
        assert (merged in checked.cards) is allow_cards
        framework = load_framework(updater.graph)
        assert framework['framework']['areas'][0]['templates'] == [merged]
        assert framework['historical_framework']['areas'][0]['templates'] == [old, split]
        assert not memory.context(node, template_key(updater.layer, merged))['template']
        assert memory.context(node, None)['page'] and memory.path.read_bytes() == memory_before
        assert result['index']['encoded'] == 0 and result['index']['reused'] == len(base.nodes)
        assert len(retrieve(updater.graph, 'product', encoder, 10)) == len(updater.layer['templates'])
        assert updater.finish() == result  # No second merge or paid card calls.
        # A subsequent task inherits caches and projects the local framework.
        next_log = EventLog(tmp_path / 'next')
        next_model = CardModel(tmp_path / 'next-refresh')
        next_updater = ExecutionGraph(updater.graph, browser, next_log, checked, index_client=encoder,
                                     card_client=next_model, template_clustering='legacy')
        write_json(next_updater.graph.directory / 'initial-website-framework.json', framework)
        next_result = next_updater.finish()
        assert not next_result['template_reconciliation']['merges']
        assert next_result['template_reconciliation']['page_comparisons'] == 0
        assert not next_model.calls
        assert load_framework(next_updater.graph)['framework']['areas'][0]['templates'] == [merged]
        manifest = updater.graph.directory / 'templates/reconciliation.json'
        manifest.write_text('{}')
        # Runtime framework loading uses current retired-template IDs, not merge manifests.
        assert load_framework(updater.graph)['framework']['areas'][0]['templates'] == [merged]
    assert hashes(base.directory) == originals
