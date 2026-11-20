import copy
import json
from itertools import permutations
from types import SimpleNamespace

import pytest

from test_template_entries import layer, member, profile
from web2kg.stage1.template_reconciliation import reconcile, apply_graph_templates
from web2kg.stage1.templates import next_template_id
from web2kg.stage2.recovery import template_key


def test_consolidation_remaps_groups_edges_and_keeps_history_and_page_evidence():
    data = layer()
    a = member(data, 'T001', 'a', profile('same'))
    b = member(data, 'T003', 'b', profile('same'))
    member(data, 'T003', 'variant', profile('different'), parent='b', variant=True)
    member(data, 'T008', 'child', profile('other'), parent='a')
    data['transitions'] = [dict(id='old', source='T001', target='T003', control={'kind': 'click'}, evidence_edges=['e'])]
    data['decisions'] = [{'node': 'a', 'template': 'T001', 'reason': 'new_template'}]
    before = copy.deepcopy(data)
    result, audit = reconcile(data)
    assert data == before
    assert audit['templates_after'] == 2 and len(audit['merges']) == 1
    new = audit['mapping']['T001']
    assert new == audit['mapping']['T003'] == 'T009'
    merged = next(g for g in result['templates'] if g['id'] == new)
    assert set(merged['members']) == {'a', 'b', 'variant'}
    assert result['assignments']['child']['parent_template'] == new
    assert result['transitions'][0]['source'] == result['transitions'][0]['target'] == new
    assert result['transitions'][0]['evidence_edges'] == ['e']
    assert result['decisions'] == before['decisions']
    assert template_key(result, new) not in {template_key(before, a['id']), template_key(before, b['id'])}
    assert set(merged['experience_sources']) == {template_key(before, a['id']), template_key(before, b['id'])}
    assert next_template_id(result) == 'T010'
    graph = SimpleNamespace(nodes={k: {'id': k, 'path': []} for k in data['assignments']},
        data={'edges': [dict(id='e', source='a', target='b', action={'kind': 'click'},
                            replayable=False, source_template='T001', target_template='T003')]})
    apply_graph_templates(graph, result)
    assert graph.data['edges'][0]['replayable'] is False
    assert graph.data['edges'][0]['action'] == {'kind': 'click'}
    assert graph.data['edges'][0]['source_template'] == new


def test_bridge_cannot_merge_incompatible_endpoints():
    pages = {'a': profile('1', '2', '3', '4'), 'b': profile('1', '2', '3', '4', '5'),
             'c': profile('2', '3', '4', '5')}
    partitions = []
    for order in permutations(pages):
        data = layer()
        data['parameters']['threshold'] = .75
        for i, key in enumerate(order):
            member(data, f'T{i+1:03d}', key, pages[key])
        result, audit = reconcile(data)
        assert len(audit['merges']) == 1
        assert result['assignments']['a']['template'] != result['assignments']['c']['template']
        partitions.append(sorted(sorted(g['members']) for g in result['templates']))
    assert all(p == partitions[0] for p in partitions)


def test_cache_invalidates_on_member_structure_rule_and_threshold_changes():
    data = layer()
    member(data, 'T001', 'a', profile('x'))
    member(data, 'T002', 'b', profile('y'))
    result, _ = reconcile(data)
    warm, audit = reconcile(json.loads(json.dumps(result)))
    assert audit['pair_evaluations'] == audit['page_comparisons'] == 0 and audit['cache_hits'] == 1
    member(warm, 'T002', 'b2', profile('x'))
    _, audit = reconcile(warm)
    assert audit['pair_evaluations'] == 1
    warm['profiles']['b'] = profile('x')
    merged, audit = reconcile(warm)
    assert len(audit['merges']) == 1
    assert not reconcile(merged)[1]['merges']
    for mutate in (lambda x: x['parameters'].update(threshold=.9),
                   lambda x: x['parameters'].update(page_features={'structure': 1.}),
                   lambda x: x['reconciliation_cache'].update(rule_key='old-code')):
        changed = copy.deepcopy(result)
        mutate(changed)
        assert reconcile(changed)[1]['pair_evaluations'] == 1


def test_any_core_counterexample_blocks_merge_but_variants_do_not():
    data = layer()
    member(data, 'T001', 'a', profile('x'))
    member(data, 'T002', 'b', profile('x'))
    member(data, 'T002', 'c', profile('y'))
    assert not reconcile(data)[1]['merges']
    data['assignments']['c']['reason'] = 'observed_local_state_variant'
    assert len(reconcile(data)[1]['merges']) == 1


def test_legacy_consolidation_disabled():
    data = layer()
    data['version'] = 'context-structure-v7'
    original = copy.deepcopy(data)
    assert reconcile(data)[1]['status'] == 'disabled_legacy'
    assert data == original
