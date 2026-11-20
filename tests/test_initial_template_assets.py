import json
import shutil

import pytest

from test_function_cards import saved
from test_global_template_assets import FullModel
from test_execution_assets import hashes
from web2kg.common.storage import Graph, EventLog, write_json
from web2kg.stage1.template_assets import initialize_assets, rebuild_assets
from web2kg.stage1.function_cards import load_layer
from web2kg.stage1.site_framework import load_framework
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.stage2.graph_update import ExecutionGraph


@pytest.mark.parametrize('fail', [None, 'site_framework'])
def test_first_build_clusters_and_publishes_only_complete_bundle(saved, tmp_path, fail):
    base = saved[0]
    original = hashes(base.directory)
    shutil.copytree(base.directory, tmp_path / 'graph')
    graph = Graph.load(tmp_path / 'graph')
    for pointer in ('templates/active.json', 'function-cards/active.json', 'site-framework/active.json'):
        (graph.directory / pointer).unlink(missing_ok=True)
    model = FullModel(tmp_path / 'model', fail)
    if fail:
        with pytest.raises(RuntimeError, match='fixture failed'):
            initialize_assets(graph, model)
        assert not (graph.directory / 'templates/active.json').exists()
        assert not (graph.directory / 'function-cards/active.json').exists()
        assert not (graph.directory / 'site-framework/active.json').exists()
    else:
        result = initialize_assets(graph, model)
        assert result['audit']['initial'] and result['audit']['method'] == 'average'
        assert result['audit']['selection']['tolerance'] == .05
        _, layer = load_layer(graph)
        assert not layer['global_maintenance']['pending_templates']
        assert len(model.calls) == len(layer['templates']) + 1
        knowledge = TemplateKnowledge(graph, graph.directory / 'function-cards/initial/cards.json', EventLog(tmp_path/'task'))
        updater = ExecutionGraph(graph, None, knowledge.log, knowledge, card_client=FullModel(tmp_path/'next-model'))
        write_json(updater.graph.directory / 'initial-website-framework.json', load_framework(graph))
        next_result = updater.finish()
        assert next_result['template_global']['status'] == 'pending'
        assert not updater.live_cards.client.calls
        with pytest.raises(ValueError, match='without published'):
            initialize_assets(graph, model)
    assert hashes(base.directory) == original


@pytest.mark.parametrize('fail', [None, 'site_framework'])
def test_expansion_rebuild_keeps_old_assets_until_complete(saved, tmp_path, fail):
    base = saved[0]
    original = hashes(base.directory)
    shutil.copytree(base.directory, tmp_path / 'copy')
    graph = Graph.load(tmp_path / 'copy')
    pointers = {p: (graph.directory / p).read_bytes() if (graph.directory / p).exists() else None
                for p in ('templates/active.json', 'function-cards/active.json', 'site-framework/active.json')}
    model = FullModel(tmp_path / 'calls', fail)
    if fail:
        with pytest.raises(RuntimeError, match='fixture failed'):
            rebuild_assets(graph, model)
        for p, data in pointers.items():
            assert ((graph.directory / p).read_bytes() if (graph.directory / p).exists() else None) == data
    else:
        result = rebuild_assets(graph, model)
        assert not result['audit']['initial']
        assert json.loads((graph.directory / 'templates/active.json').read_text())['directory'] == 'templates/expanded'
        assert load_framework(graph) is not None
        knowledge = TemplateKnowledge(graph, graph.directory / 'function-cards/expanded/cards.json', EventLog(tmp_path / 'validate'))
        assert set(knowledge.cards) == set(knowledge.groups)
    assert hashes(base.directory) == original
