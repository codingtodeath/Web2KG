import json
from unittest.mock import Mock
import pytest
import numpy as np
from test_replay_eligibility import graph_at
from test_visual_navigation import FakeJina
from web2kg.stage1.visual_index import build_visual_index
from web2kg.common.storage import Graph
from web2kg.stage2.graph_update import transition_key
from web2kg.stage2.retrieval import retrieve
from scripts.update_replay_graph import prune_index, review_batch
import copy
from types import SimpleNamespace


def test_transition_identity_ignores_live_ids_but_keeps_values():
    action = {'id': '1', 'ax_node_id': '7', 'role': 'button', 'name': 'Open', 'kind': 'click'}
    assert transition_key('a', action, 'b') == transition_key('a', {**action, 'id': '2', 'ax_node_id': '9'}, 'b')
    assert transition_key('a', {**action, 'value': 'one'}, 'b') != transition_key('a', {**action, 'value': 'two'}, 'b')
    assert transition_key('a', action, 'b') != transition_key('a', action, 'c')


@pytest.mark.parametrize('delete_node', [False, True])
def test_offline_prune_preserves_kept_arrays_and_source(tmp_path, delete_node):
    base = graph_at(tmp_path / 'base')
    data = build_visual_index(base, FakeJina())
    old = (base.directory / 'index.json').read_bytes()
    graph = Graph(tmp_path / 'updated')
    graph.directory.mkdir()
    graph.data = copy.deepcopy(base.data)
    graph.data['edges'][0]['replayable'] = False
    if delete_node:
        del graph.nodes['c']
        graph.data['edges'] = graph.data['edges'][:1]
    result = prune_index(base, graph)
    assert result['before'] == 3 and result['after'] == (2 if delete_node else 3)
    assert result['removed'] == (['c'] if delete_node else [])
    from web2kg.common.page_text import fingerprint
    saved = json.loads((graph.directory / 'index.json').read_text())
    assert saved['fingerprint'] == fingerprint(graph)
    with np.load(graph.directory / 'replay-vectors.npz') as new, np.load(base.directory / data['vectors']) as original:
        assert set(new.files) == set(graph.nodes)
        assert all(np.array_equal(new[k], original[k]) for k in new.files)
    assert (base.directory / 'index.json').read_bytes() == old


def test_candidate_reachability_computed_only_once(tmp_path, monkeypatch):
    graph = graph_at(tmp_path)
    from web2kg.stage1.indexing import build_index
    build_index(graph)
    call = Mock(wraps=graph.reachable)
    monkeypatch.setattr(graph, 'reachable', call)
    retrieve(graph, 'a'); retrieve(graph, 'b')
    assert call.call_count == 1


def test_review_batch_validates_complete_distinct_ids(tmp_path):
    client = Mock(last_request_id='fixture')
    client.json.return_value = {'decisions': [{'id': 'a', 'replayable': True, 'reason': 'Read only'}]}
    assert review_batch([{'id': 'a'}], client, tmp_path / 'good')['a']['replayable']
    with pytest.raises(ValueError):
        review_batch([{'id': 'a'}, {'id': 'b'}], client, tmp_path / 'bad')
    assert json.loads((tmp_path / 'bad/result.json').read_text())['status'] == 'failed'


def test_unlimited_responses_passes_old_description_cap_and_keeps_usage(tmp_path):
    from web2kg.common.responses import ResponsesClient, object_schema
    from web2kg.common.storage import EventLog
    env = tmp_path / 'fixture.env'
    env.write_text('CRAWL_API_KEY=fixture\nCRAWL_BASE_URL=https://example.org/v1\nCRAWL_MODEL=fixture\n')
    log = EventLog(tmp_path / 'calls')
    client = ResponsesClient(env, log, max_calls=None)
    client.session.post = Mock(return_value=SimpleNamespace(ok=True, json=lambda: {
        'status': 'completed', 'usage': {'input_tokens': 1, 'output_tokens': 1},
        'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"ok":true}'}]}]}))
    for _ in range(32):
        assert client.json('fixture', {}, object_schema({'ok': {'type': 'boolean'}}), 'test')['ok']
    assert client.budget_used() == 32
    assert client.session.post.call_count == 32
