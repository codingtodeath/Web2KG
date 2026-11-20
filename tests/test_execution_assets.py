import json
import shutil

import numpy as np
import pytest

from test_execution_graph import hashes
from test_function_cards import saved, CardModel
from test_visual_navigation import FakeJina
from test_llm_exploration import FakeClient
from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog, Graph
from web2kg.stage1.visual_index import build_visual_index
from web2kg.stage2.graph_update import ExecutionGraph
from web2kg.stage2.retrieval import retrieve
from web2kg.stage2.template_context import TemplateKnowledge


def indexed_copy(saved, tmp_path):
    base, _, _, cards, _ = saved
    directory = tmp_path / 'base'
    shutil.copytree(base.directory, directory)
    base = Graph.load(directory)
    class Encoder(FakeJina):
        def encode(self, value, kind, expected):
            super().encode(value, kind, expected)
            i = int(kind == 'text' or value.parent.name not in base.nodes)
            return np.eye(128, dtype=np.float32)[i:i + 1]
    client = Encoder()
    build_visual_index(base, client)
    client.calls.clear()
    return base, cards, client


def test_task_end_refreshes_existing_card_and_indexes_new_member(saved, site, tmp_path):
    base, cards, client = indexed_copy(saved, tmp_path)
    original = hashes(base.directory)
    log = EventLog(tmp_path / 'run')
    knowledge = TemplateKnowledge(base, cards, log)
    model = CardModel(tmp_path / 'luna')
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(base, browser, log, knowledge, card_client=model, index_client=client,
                                 description_client=FakeClient(EventLog(tmp_path / 'descriptions')), max_descriptions=1)
        assert updater.live_cards.limit is None
        assert not updater.affected_templates
        before = browser.observe(); updater.observe(before)
        action = next(a for a in before['actions'] if a['name'] == 'Green mug')
        browser.act(action); after = browser.observe()
        updater.record(before, action, after)
        group = updater.layer['assignments'][after['id']]['template']
        assert group in updater.live_cards.initial
        assert not model.calls and not client.calls  # Existing cards and index stay stable during execution.
        live = updater.live_cards.directory
        original_live = (live / 'cards.json').read_bytes()
        result = updater.finish()
        assert result['index'] == {'status': 'completed', 'pages': 4, 'encoded': 1, 'reused': 3, 'file': 'index.json'}
        assert group in result['card_refresh']['refreshed']
        payload = next(p for p in model.calls if p['template_id'] == group)
        assert payload['members'] == 2
        assert after['id'] in [p['node_id'] for p in json.loads(
            (updater.live_cards.directory / 'inputs' / f'{group}.json').read_text(encoding='utf-8'))['representatives']]
        assert len(model.calls) == 1 and len(updater.affected_templates) == 2
        # The source template only gained an edge; its member count did not change.
        # Final save may advance graph provenance; the finalizer preserves that live bundle thereafter.
        retained = (live / 'cards.json').read_bytes()
        assert json.loads(retained)['cards'] == json.loads(original_live)['cards']
        calls = (len(client.calls), len(model.calls))
        assert updater.finish() == result and calls == (len(client.calls), len(model.calls))
        assert (live / 'cards.json').read_bytes() == retained
        active = json.loads((updater.graph.directory / 'function-cards/active.json').read_text())
        assert active['cards'] == 'function-cards/finalized/cards.json'
        next_context = TemplateKnowledge(updater.graph, updater.graph.directory / active['cards'], log)
        assert group in next_context.cards
        assert retrieve(updater.graph, 'new member', client, 1)[0]['id'] == after['id']
    assert hashes(base.directory) == original


@pytest.mark.parametrize('blocked', ['budget', 'provider', 'index', 'card'])
def test_finalization_limits_and_failures_are_independent(saved, site, tmp_path, blocked):
    from unittest.mock import Mock
    from web2kg.common.responses import ResponsesError
    base, cards, client = indexed_copy(saved, tmp_path)
    log = EventLog(tmp_path / 'run')
    model = CardModel(tmp_path / 'luna')
    knowledge = TemplateKnowledge(base, cards, log)
    if blocked == 'index':
        client.encode = Mock(side_effect=ConnectionError('local encoder stopped'))
    if blocked == 'card':
        model.json = Mock(side_effect=ResponsesError('HTTP 503'))
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(base, browser, log, knowledge, card_client=model, max_cards=1, index_client=client,
                                 description_client=FakeClient(EventLog(tmp_path / 'descriptions')), max_descriptions=1)
        before = browser.observe(); updater.observe(before)
        action = next(a for a in before['actions'] if a['name'] == 'Green mug')
        browser.act(action); after = browser.observe()
        updater.record(before, action, after)
        target = updater.layer['assignments'][after['id']]['template']
        result = updater.finish(allow_card_calls=blocked != 'provider')
        assert result['index_status'] == ('failed' if blocked == 'index' else 'completed')
        assert len(model.calls) == (0 if blocked in {'provider', 'card'} else 1)
        if blocked not in {'provider', 'card'}:
            assert result['card_refresh']['refreshed'] == [target]
            assert not result['card_refresh']['pending']
        else:
            assert not result['card_refresh']['refreshed']
            assert result['card_refresh']['pending'] == [target]
            row = updater.live_cards.data['cards'][target]
            assert row['status'] == 'completed' and row['refresh_state']['pending_changes'] == 1
            assert row['refresh_attempt']['status'] == ('failed' if blocked == 'card' else 'not_generated')
        assert not result['card_refresh']['unavailable']
        if blocked == 'index':
            assert not (updater.graph.directory / 'index.json').exists()
            assert client.encode.call_count == 1
        if blocked == 'card':
            assert model.json.call_count == 1 and updater.live_cards.stopped


def test_disabled_finalization_keeps_historical_protocol_behavior(saved, site, tmp_path):
    base, cards, client = indexed_copy(saved, tmp_path)
    log = EventLog(tmp_path / 'run')
    model = CardModel(tmp_path / 'luna')
    knowledge = TemplateKnowledge(base, cards, log)
    with Browser(site + '/catalog', log, identity_mode='interaction', settle_ms=20) as browser:
        updater = ExecutionGraph(base, browser, log, knowledge, card_client=model, max_cards=4,
                                 index_client=client, finalize_assets=False)
        before = browser.observe(); updater.observe(before)
        action = next(a for a in before['actions'] if a['name'] == 'Green mug')
        browser.act(action); updater.record(before, action, browser.observe())
        result = updater.finish()
        assert result['index_status'] == 'disabled' and result['card_refresh'] is None
        assert not model.calls and not client.calls
        assert not (updater.graph.directory / 'index.json').exists()


def test_agent_saves_trajectory_and_closes_browser_before_finalizing(saved, site, tmp_path, monkeypatch):
    import re
    from web2kg.stage2.agent import run_task
    base, cards, client = indexed_copy(saved, tmp_path)
    log = EventLog(tmp_path / 'run')
    knowledge = TemplateKnowledge(base, cards, log)
    model = CardModel(tmp_path / 'luna')
    seen = []
    class Actor:
        def call(self, messages, **kwargs):
            seen.append(messages)
            if len(seen) == 1:
                key = re.search(r"link \[(\d+)\] 'Green mug'", json.dumps(messages))[1]
                return f'REASON: Inspect new item.\nACTION: click [{key}]'
            return 'REASON: Finished.\nACTION: stop [done]'
    finish = ExecutionGraph.finish
    def checked_finish(self, **kwargs):
        assert self.browser.page.is_closed()
        assert (log.directory / 'result.json').exists() and (log.directory / 'trajectory.json').exists()
        return finish(self, **kwargs)
    monkeypatch.setattr(ExecutionGraph, 'finish', checked_finish)
    result = run_task(base, 'Inspect Green mug', log, Actor(), client, knowledge=knowledge,
                      start_url=site + '/catalog', update_graph=True, card_client=model, max_cards=4,
                      description_client=FakeClient(EventLog(tmp_path / 'descriptions')), max_descriptions=2)
    assert result['status'] == 'agent_stopped' and result['answer'] == 'done'
    assert 'execution_graph_error' not in result, result
    assert result['execution_graph']['index']['encoded'] == 1
    expanded = Graph.load(log.directory / 'execution-graph')
    changed = set(result['execution_graph']['new_states'])
    affected = {expanded.nodes[k]['template_id'] for k in changed}
    assert set(result['execution_graph']['card_refresh']['refreshed']) == affected
    found = retrieve(expanded, 'new item', client, 1)[0]['id']
    assert expanded.nodes[found]['url'].endswith('/item/3')
    assert result['total_seconds'] >= result['seconds']
