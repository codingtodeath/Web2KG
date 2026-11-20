import copy
import math
import random

import pytest

from web2kg.common.storage import Graph
from web2kg.stage2.retention import plan_pruning, retention_score


def fixture_graph(tmp_path, members, edges, importance, idle=None):
    graph = Graph(tmp_path)
    graph.data.update(root='root', retention={'retrieval_round': 100})
    graph.data['nodes'] = {k: {'id': k, 'last_retrieved_round': 100 - (idle or {}).get(k, 0)}
                           for nodes in members.values() for k in nodes}
    graph.data['edges'] = [{'id': str(i), 'source': a, 'target': b} for i, (a, b) in enumerate(edges)]
    layer = {'assignments': {k: {'template': t} for t, nodes in members.items() for k in nodes}}
    cards = {t: {'importance': value} for t, value in importance.items()}
    return graph, layer, cards


def scan_reference(graph, layer, cards, limit, protected=(), half_life=20):
    """Deliberately simple exhaustive oracle, independent of the two-level heap."""
    from web2kg.common.backbone import shortest_backbone
    tree = shortest_backbone(graph)
    edges = [(p['node'], k) for k, p in tree['parents'].items() if p]
    remaining = set(graph.nodes)
    protected = set(protected) | {graph.data['root']}
    removed = []
    now = graph.data['retention']['retrieval_round']
    while len(remaining) > limit:
        leaves = [k for k in remaining - protected if not any(
            source == k and target in remaining for source, target in edges)]
        if not leaves:
            break
        def score(key):
            template = layer['assignments'][key]['template']
            count = sum(layer['assignments'][k]['template'] == template for k in remaining)
            idle = max(0, now - graph.nodes[key]['last_retrieved_round'])
            return math.log2((cards[template]['importance'] + 1 / count) / 2) - idle / half_life, key
        key = min(leaves, key=score)
        removed.append(key)
        remaining.remove(key)
    return removed


def test_batch_reorders_after_template_count_changes(tmp_path):
    args = fixture_graph(tmp_path, {'R': ['root'], 'A': ['a1', 'a2', 'a3'], 'B': ['b1', 'b2']},
                         [('root', k) for k in ('a1', 'a2', 'a3', 'b1', 'b2')], {'R': 1, 'A': .2, 'B': .1})
    original = copy.deepcopy([args[0].data, args[1], args[2]])
    plan = plan_pruning(*args, max_nodes=4)
    assert [r['node'] for r in plan['initial_ranking'][:2]] == ['a1', 'a2']
    assert [r['node'] for r in plan['removed']] == ['a1', 'b1']
    assert plan['removed'][0]['template_members'] == 3
    assert plan['removed'][1]['template_members'] == 2
    assert plan['removed'][0]['retention_score'] == pytest.approx((.2 + 1 / 3) / 2)
    assert original == [args[0].data, args[1], args[2]]


def test_newly_exposed_parent_competes_with_current_leaves(tmp_path):
    args = fixture_graph(tmp_path, {'R': ['root'], 'A': ['a', 'parent'], 'B': ['b']},
                         [('root', 'parent'), ('parent', 'a'), ('root', 'b')], {'R': 1, 'A': 0, 'B': 1})
    plan = plan_pruning(*args, max_nodes=2)
    assert [r['node'] for r in plan['removed']] == ['a', 'parent']
    assert plan['removed'][0]['exposed_leaves'] == ['parent']


def test_three_factors_and_long_idle_ordering():
    score = lambda f, n, idle: retention_score(f, n, idle, 20)['log2_retention']
    assert score(.9, 2, 0) > score(.1, 2, 0)
    assert score(.5, 1, 0) > score(.5, 5, 0)
    assert score(.5, 2, 0) > score(.5, 2, 20)
    assert retention_score(.5, 2, 20, 20)['retention_score'] == pytest.approx(.25)
    assert score(.5, 2, 100000) > score(.5, 2, 100001)


def test_dynamic_heap_matches_exhaustive_ranking_with_cycles_ties_and_protection(tmp_path):
    for seed in range(50):
        randomizer = random.Random(seed)
        members = {'R': ['root'], **{f'T{i}': [] for i in range(5)}}
        nodes = [f'n{i:02}' for i in range(25)]
        for node in nodes:
            members[f'T{randomizer.randrange(5)}'].append(node)
        edges = [('root', k) for k in nodes]
        edges += [(a, b) for a in nodes for b in nodes if randomizer.random() < .018]
        args = fixture_graph(tmp_path, members, edges, {t: randomizer.choice([0, .5, 1]) for t in members},
                             {k: randomizer.choice([0, 5, 20, 40, 100000]) for k in nodes})
        protected = randomizer.sample(nodes, 3)
        expected = scan_reference(*args, 10, protected)
        first = plan_pruning(*args, max_nodes=10, protected=protected)
        assert [r['node'] for r in first['removed']] == expected
        args[0].data['nodes'] = dict(reversed(list(args[0].nodes.items())))
        randomizer.shuffle(args[0].data['edges'])
        second = plan_pruning(*args, max_nodes=10, protected=protected)
        assert first['removed'] == second['removed']


def test_unknown_nodes_are_not_one_shared_template_and_noop_does_not_score(tmp_path):
    args = fixture_graph(tmp_path, {'R': ['root'], 'A': ['a', 'b']}, [('root', 'a'), ('root', 'b')], {'R': 1, 'A': .5})
    args[1]['assignments'] = {'root': args[1]['assignments']['root']}
    plan = plan_pruning(*args, max_nodes=2)
    assert plan['removed'][0]['structural_importance'] == 1
    assert plan_pruning(*args, max_nodes=3)['score_evaluations'] == 0
    assert plan_pruning(*args, max_nodes=0)['remaining_excess'] == 0
