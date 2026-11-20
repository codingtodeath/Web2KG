import copy
import json

import pytest

from test_function_cards import saved, CardModel
from test_reconciliation_assets import fragmented_graph
from test_execution_assets import hashes
from test_visual_navigation import FakeJina
from web2kg.common.storage import EventLog, write_json
from web2kg.stage1.function_cards import load_layer, file_hash
from web2kg.stage1.site_framework import load_framework
from web2kg.stage2.graph_update import ExecutionGraph
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.stage2.retrieval import retrieve


class FullModel(CardModel):
    def __init__(self, directory, fail=None):
        super().__init__(directory)
        self.fail = fail

    def json(self, system, payload, schema, purpose, images=(), *, validate=None):
        if self.fail == purpose:
            raise RuntimeError('fixture failed generation')
        if purpose == 'site_framework':
            self.calls.append(payload)
            return dict(overview='Updated fixture', areas=[dict(name='All', purpose='Browse',
                templates=[t['id'] for t in payload['templates']])], navigation=[], limitations=[])
        return super().json(system, payload, schema, purpose, images=images, validate=validate)


@pytest.mark.parametrize('fail', [None, 'site_framework', 'template_card'])
def test_global_assets_stage_before_publication_and_next_task_can_inherit(saved, site, tmp_path, fail):
    base, path, _, _, _ = fragmented_graph(saved, site, tmp_path)
    before = hashes(base.directory)
    log = EventLog(tmp_path / 'task')
    knowledge = TemplateKnowledge(base, path, log)
    model = FullModel(tmp_path / 'model', fail)
    updater = ExecutionGraph(base, None, log, knowledge, card_client=model, index_client=FakeJina(),
                             template_refresh_min=1, template_refresh_fraction=0)
    write_json(updater.graph.directory / 'initial-website-framework.json', load_framework(base))
    old = copy.deepcopy(updater.layer['assignments'])
    updater.layer['global_maintenance']['pending_templates'] = [next(iter(knowledge.groups))]
    result = updater.finish()
    audit = result['template_global']
    assert audit['status'] == ('failed' if fail else 'published')
    assert updater.finish() == result
    if fail:
        assert updater.layer['assignments'] == old
        assert updater.layer['global_maintenance']['pending_templates']
        assert updater.live_cards.stopped and audit['calls'] >= 1
        assert all(r['status'] == 'completed' for r in updater.live_cards.data['cards'].values())
    else:
        assert audit['calls'] == len(updater.layer['templates']) + 1
        assert not updater.layer['global_maintenance']['pending_templates']
        assert result['card_refresh']['status'] == 'global_rebuild'
        assert result['framework_refresh']['status'] == 'global_rebuild'
        assert all(r['refresh_state']['pending_changes'] == 0 for r in updater.live_cards.data['cards'].values())
        generated = json.loads((updater.live_cards.directory / 'generated-cards.json').read_text(encoding='utf-8'))
        assert generated['source']['graph_sha256'] == file_hash(updater.graph.directory / 'templates/global/input-graph.json')
        assert generated['source']['layer_sha256'] == file_hash(updater.graph.directory / 'templates/global/staged-templates.json')
        framework = json.loads((updater.graph.directory / 'site-framework/global/framework.json').read_text(encoding='utf-8'))
        assert framework['generation_source']['cards_sha256'] == file_hash(updater.live_cards.directory / 'generated-cards.json')
    checked = TemplateKnowledge(updater.graph, updater.live_cards.directory / 'cards.json', log)
    assert set(checked.cards) == set(checked.groups)
    _, persisted = load_layer(updater.graph)
    assert persisted == updater.layer
    assert result['index']['encoded'] == 0
    assert len(retrieve(updater.graph, 'product', FakeJina(), 10)) == len(updater.layer['templates'])
    frame = load_framework(updater.graph)
    following = ExecutionGraph(updater.graph, None, EventLog(tmp_path / 'next'), checked,
        card_client=FullModel(tmp_path / 'next-model'), index_client=FakeJina(), template_refresh_min=0)
    write_json(following.graph.directory / 'initial-website-framework.json', frame)
    following.finish()
    assert hashes(base.directory) == before


def test_insufficient_budget_defers_whole_rebuild_without_partial_paid_calls(saved, site, tmp_path):
    base, path, _, _, _ = fragmented_graph(saved, site, tmp_path)
    log = EventLog(tmp_path / 'task')
    model = FullModel(tmp_path / 'model')
    updater = ExecutionGraph(base, None, log, TemplateKnowledge(base, path, log), card_client=model,
        max_cards=1, index_client=FakeJina(), template_refresh_min=1, template_refresh_fraction=0)
    old = copy.deepcopy(updater.layer['assignments'])
    updater.layer['global_maintenance']['pending_templates'] = [updater.layer['templates'][0]['id']]
    result = updater.finish()
    assert result['template_global']['status'] == 'deferred'
    assert not model.calls and updater.layer['assignments'] == old
