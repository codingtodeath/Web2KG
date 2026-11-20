import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from test_function_cards import CardModel
from web2kg.common.storage import Graph, digest, write_json
from web2kg.stage1.function_cards import LiveCards, build_cards, file_hash
from web2kg.stage1.templates import VERSION
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.stage2.retention import trim_graph


def persist(graph, layer):
    # Current visual cards require a real image even in this membership-only fixture.
    from PIL import Image
    for key in graph.nodes:
        image = graph.directory / 'snapshots' / key / 'page.png'
        if not image.exists():
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (2, 2), 'white').save(image)
    graph.save()
    layer['graph_digest'] = digest(graph.data)
    path = graph.directory / 'templates/test/templates.json'
    write_json(path, layer)
    write_json(graph.directory / 'templates/active.json', {'directory': 'templates/test'})
    return path


def members(graph, layer, ids):
    graph.data['nodes'] = {k: {'id': k, 'url': 'https://fixture.test/' + k, 'title': k,
                              'actions': [], 'headings': []} for k in ids}
    graph.data['root'] = ids[0]
    layer['templates'] = [{'id': 'T001', 'members': list(ids), 'representative': ids[0]}]
    layer['assignments'] = {k: {'template': 'T001', 'parent_template': None} for k in ids}


def initial(tmp_path, count):
    graph = Graph(tmp_path / 'base')
    layer = {'version': VERSION, 'snapshot_sha256': {}, 'profiles': {}, 'transitions': []}
    members(graph, layer, [f'n{i:02}' for i in range(count)])
    path = persist(graph, layer)
    model = CardModel(tmp_path / 'seed-calls')
    build_cards(graph, path, graph.directory / 'function-cards/seed', model)
    return graph, layer, graph.directory / 'function-cards/seed/cards.json'


def boundary(previous, ids, folder, model=None, allow_calls=True, limit=None, affected=('T001',), prune_to=None):
    base, layer, seed = previous
    graph = Graph(folder)
    graph.data = copy.deepcopy(base.data)
    graph.data['execution_update'] = {'source_graph': str(base.directory),
                                      'source_sha256': file_hash(base.directory / 'graph.json')}
    layer = copy.deepcopy(layer)
    persist(graph, layer)
    model = model or CardModel(folder / 'calls')
    cards = LiveCards(graph, layer, model, limit, seed)
    old_members = set(graph.nodes)
    members(graph, layer, ids)
    if prune_to is not None:
        trim_graph(graph, layer, {}, max_nodes=prune_to)
    persist(graph, layer)
    cards.save()
    result = cards.finalize(set(affected), sorted(set(ids) - old_members), allow_calls)
    saved = cards.directory / 'cards.json'
    knowledge = TemplateKnowledge(graph, saved, model.log)
    assert 'T001' in knowledge.cards
    return (graph, layer, saved), cards, result, model


@pytest.mark.parametrize('baseline,delta,expected', [(20, 7, False), (20, 8, True),
    (4, 3, False), (4, 4, True), (1, 1, True), (12, -7, False), (12, -8, True)])
def test_strict_absolute_or_relative_threshold_including_deletions(tmp_path, baseline, delta, expected):
    start = initial(tmp_path, baseline)
    ids = list(start[0].nodes)
    ids = ids + [f'new{i}' for i in range(delta)] if delta > 0 else ids[:delta]
    _, cards, result, model = boundary(start, ids, tmp_path / 'task')
    assert bool(result['refreshed']) is expected
    assert len(model.calls) == int(expected)
    state = cards.data['cards']['T001']['refresh_state']
    assert state['pending_changes'] == (0 if expected else abs(delta))
    assert state['baseline_members'] == (len(ids) if expected else baseline)


def test_cross_task_changes_accumulate_without_netting_and_reset_only_on_generation(tmp_path):
    start = initial(tmp_path, 10)
    ids = list(start[0].nodes)
    original = json.loads(start[2].read_text(encoding='utf-8'))['cards']['T001']
    ids += ['new1', 'new2', 'new3']
    task, cards, result, model = boundary(start, ids, tmp_path / 't1')
    assert result['deferred'] == ['T001'] and not model.calls
    ids = ids[2:]
    task, cards, result, model = boundary(task, ids, tmp_path / 't2')
    assert cards.data['cards']['T001']['refresh_state']['pending_changes'] == 5
    ids = ids[1:] + ['replacement']
    task, cards, result, model = boundary(task, ids, tmp_path / 't3')
    row = cards.data['cards']['T001']
    assert row['refresh_state']['pending_changes'] == 7 and not model.calls
    assert row['input_hash'] == original['input_hash'] and row['card'] == original['card']
    assert row['current_scope']['members'] == sorted(ids)
    task, cards, result, model = boundary(task, ids + ['new4'], tmp_path / 't4')
    assert result['refreshed'] == ['T001'] and len(model.calls) == 1
    assert cards.data['cards']['T001']['refresh_state'] == {
        'baseline_members': len(ids) + 1, 'pending_changes': 0, 'pending_members': []}
    _, cards, result, model = boundary(task, ids + ['new4'], tmp_path / 't5')
    assert result['up_to_date'] == ['T001'] and not model.calls


@pytest.mark.parametrize('blocked', ['budget', 'provider', 'failure'])
def test_failed_or_deferred_refresh_keeps_counter_card_and_input_across_tasks(tmp_path, blocked):
    start = initial(tmp_path, 4)
    ids = list(start[0].nodes) + [f'new{i}' for i in range(4)]
    original = (start[2].parent / 'inputs/T001.json').read_bytes()
    model = CardModel(tmp_path / 'failure-calls')
    if blocked == 'failure':
        model.json = Mock(side_effect=RuntimeError('fixture unavailable'))
    task, cards, result, _ = boundary(start, ids, tmp_path / 't1', model,
        limit=0 if blocked == 'budget' else None, allow_calls=blocked != 'provider')
    row = cards.data['cards']['T001']
    assert row['status'] == 'completed' and row['refresh_state']['pending_changes'] == 4
    assert result['pending'] == ['T001'] and not result['unavailable']
    attempt = copy.deepcopy(row['refresh_attempt'])
    assert attempt['status'] == ('failed' if blocked == 'failure' else 'not_generated')
    assert Path(attempt['input_file']).is_file()
    assert (cards.directory / 'inputs/T001.json').read_bytes() == original
    _, cards, result, model = boundary(task, ids, tmp_path / 't2', affected=())
    assert result['refreshed'] == ['T001'] and len(model.calls) == 1
    assert cards.data['cards']['T001']['refresh_state']['pending_changes'] == 0
    assert Path(attempt['input_file']).is_file()


def test_legacy_adoption_and_representative_only_changes_do_not_force_calls(tmp_path):
    start = initial(tmp_path, 4)
    bundle = json.loads(start[2].read_text(encoding='utf-8'))
    del bundle['cards']['T001']['refresh_state']
    bundle['cards']['T001']['current_input_hash'] = 'old-policy-hash'
    write_json(start[2], bundle)
    original_input = (start[2].parent / 'inputs/T001.json').read_bytes()
    ids = list(start[0].nodes)[1:]
    task, cards, result, model = boundary(start, ids, tmp_path / 't1')
    assert result['deferred'] == ['T001'] and not model.calls
    assert cards.data['cards']['T001']['refresh_state']['baseline_members'] == 4
    assert 'current_input_hash' not in cards.data['cards']['T001']
    assert (cards.directory / 'inputs/T001.json').read_bytes() == original_input
    _, cards, result, model = boundary(task, ids[::-1], tmp_path / 't2')
    assert not model.calls and cards.data['cards']['T001']['refresh_state']['pending_changes'] == 1


@pytest.mark.parametrize('capacity,refresh', [(5, False), (4, True)])
def test_applied_pruning_counts_removed_members_before_card_finalization(tmp_path, capacity, refresh):
    start = initial(tmp_path, 12)
    graph_bytes = (start[0].directory / 'graph.json').read_bytes()
    task, cards, result, model = boundary(start, list(start[0].nodes), tmp_path / 'task', prune_to=capacity)
    assert len(task[0].nodes) == capacity
    assert bool(result['refreshed']) is refresh and len(model.calls) == int(refresh)
    assert result['decisions']['T001']['pending_changes'] == 12 - capacity
    assert (start[0].directory / 'graph.json').read_bytes() == graph_bytes
