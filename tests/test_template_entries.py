"""Missing entry evidence, representative promotion and reproducible ambiguous matching."""
import copy
from itertools import permutations
from types import SimpleNamespace

import pytest

from web2kg.common.storage import EventLog
from web2kg.stage1.templates import (VERSION, PAGE_RULE, MATCHING_POLICY, adopt_matching_policy,
    assign_template, optional_entry_candidates, refresh_representatives, structure_fingerprint)
from web2kg.stage2.template_context import TemplateKnowledge


def profile(*features):
    return {'region_features': {'main': {'structure': list(features), 'combinations': []}},
            'evidence': {'sufficient': True}, 'structure': list(features)}


def layer():
    return {'version': VERSION, 'parameters': {'matching_policy': MATCHING_POLICY, 'match_rule': PAGE_RULE,
        'threshold': .85, 'raw_weights': dict(parent=.5, incoming=.5, page=1),
        'page_features': dict(structure=.5, combinations=.5)},
        'profiles': {}, 'assignments': {}, 'templates': [], 'decisions': []}


def member(data, template, key, structure, *, parent=None, variant=False):
    data['profiles'][key] = structure
    data['assignments'][key] = {'template': template, 'parent_node': parent,
        'parent_template': data['assignments'][parent]['template'] if parent else None,
        'incoming': {'kind': 'click', 'role': 'link', 'available': False} if parent else {'root': True},
        'reason': 'observed_local_state_variant' if variant else 'new_template'}
    group = next((g for g in data['templates'] if g['id'] == template), None)
    if group is None:
        group = {'id': template, 'representative': key, 'members': []}
        data['templates'].append(group)
    group['members'].append(key)
    return group


@pytest.mark.parametrize('root_first', [True, False])
def test_orphan_and_entered_state_can_join_in_both_directions(root_first):
    data = layer()
    member(data, 'T001', 'parent', profile('different'))
    group = member(data, 'T002', 'first', profile('article', 'link'), parent=None if root_first else 'parent')
    data['profiles']['new'] = profile('article', 'link')
    graph = SimpleNamespace(nodes={'parent': {'url': '/list'}, 'new': {'url': '/new'}})
    assigned = assign_template(graph, data, 'new', 'parent' if root_first else None,
        {'kind': 'click', 'available': False} if root_first else {'root': True}, {'kind': 'click'})
    assert assigned['template'] == 'T002'
    assert assigned['selected_match']['score'] == 1
    assert assigned['selected_match']['ranking_basis'] == 'page_only'
    assert group['representative'] == 'first'  # Never change it mid-task.
    refresh_representatives(data)
    assert group['representative'] == ('new' if root_first else 'first')
    assert len(group['entry_contexts']) == 1


def test_orphan_below_page_threshold_stays_separate():
    data = layer()
    member(data, 'T001', 'first', profile('article', 'link'))
    data['profiles']['new'] = profile('form', 'button')
    assigned = assign_template(SimpleNamespace(nodes={'new': {'url': '/new'}}), data, 'new', None, {'root': True})
    assert assigned['reason'] == 'new_template' and assigned['template'] == 'T002'


def test_same_score_uses_core_mean_before_fingerprint():
    data = layer()
    member(data, 'T001', 'a', profile('article', 'link'))
    member(data, 'T001', 'a2', profile('article', 'link', 'extra'))
    member(data, 'T002', 'b', profile('article', 'link'))
    member(data, 'T002', 'b2', profile('article', 'link'))
    # A low-scoring observed local variant is not a separate core vote.
    member(data, 'T002', 'b3', profile('form'), variant=True)
    result = optional_entry_candidates(data, profile('article', 'link'), None, {'root': True})
    assert result[0]['template'] == 'T002'
    assert result[0]['core_mean'] == 1 and result[1]['core_mean'] < 1
    assert result[0]['tied_templates'] == ['T001', 'T002']


@pytest.mark.parametrize('order', list(permutations(['a', 'b', 'c'])))
def test_tie_is_stable_under_template_order_and_number(order):
    data = layer()
    for i, key in enumerate(order):
        member(data, f'T{i + 1:03d}', key, profile('same'))
    candidates = optional_entry_candidates(data, profile('same'), None, {'root': True})
    chosen = next(g for g in data['templates'] if g['id'] == candidates[0]['template'])
    assert chosen['representative'] == 'a'
    assert candidates[0]['tie_rule'] == 'core_mean_then_structure_then_member'


def test_structural_fingerprint_ignores_names_counts_and_collection_order():
    a, b = profile('x', 'y'), profile('y', 'x', 'x')
    a.update(title='First', dom_fallback_nodes=5)
    b.update(title='Second', dom_fallback_nodes=90)
    assert structure_fingerprint(a, {'structure': .5, 'combinations': .5}) == structure_fingerprint(b, {'structure': .5, 'combinations': .5})


def test_equal_core_means_use_structure_before_member_id():
    data = layer()
    data['parameters']['threshold'] = .5
    member(data, 'T001', 'z', profile('a', 'b'))
    member(data, 'T002', 'a', profile('a', 'c'))
    results = optional_entry_candidates(data, profile('a'), None, {'root': True})
    assert results[0]['score'] == results[1]['score'] == .5
    assert results[0]['core_mean'] == results[1]['core_mean']
    assert results[0]['structural_fingerprint'] < results[1]['structural_fingerprint']
    data['templates'].reverse()
    assert optional_entry_candidates(data, profile('a'), None, {'root': True})[0]['template'] == results[0]['template']


def test_available_context_keeps_weighted_ranking():
    data = layer()
    member(data, 'T001', 'parent', profile('parent'))
    member(data, 'T002', 'child', profile('child'), parent='parent')
    incoming = dict(kind='click', role='link', available=True, parent_structure=['link'],
                    siblings=[], relative_depth=1, anchor_role='main')
    data['assignments']['child']['incoming'] = incoming
    candidates = optional_entry_candidates(data, profile('child'), 'T001', {**incoming, 'kind': 'type'})
    child = next(c for c in candidates if c['template'] == 'T002')
    assert child['context_available'] and child['ranking_basis'] == 'weighted_context'
    assert child['scores'] == dict(parent=1, incoming=0, page=1)
    assert child['score'] == .75 and child['eligible']


def test_medoid_prefers_real_entry_and_keeps_all_entry_evidence():
    data = layer()
    member(data, 'T001', 'parent', profile('parent'))
    group = member(data, 'T002', 'root', profile('a', 'b', 'c'))
    member(data, 'T002', 'outlier', profile('a'), parent='parent')
    member(data, 'T002', 'central', profile('a', 'b', 'c'), parent='parent')
    member(data, 'T002', 'core2', profile('a', 'b', 'c'))
    before = copy.deepcopy(data['assignments'])
    changes = refresh_representatives(data, {'T002'})
    assert group['representative'] == 'central'
    assert {e['node'] for e in group['entry_contexts']} == {'outlier', 'central'}
    assert changes[0]['previous'] == 'root' and data['assignments'] == before
    assert not refresh_representatives(data, {'T002'})


def test_url_jump_does_not_count_as_real_entry():
    data = layer()
    member(data, 'T001', 'parent', profile('parent'))
    group = member(data, 'T002', 'jump', profile('x'), parent='parent')
    data['assignments']['jump']['incoming']['kind'] = 'url_jump'
    member(data, 'T002', 'click', profile('x'), parent='parent')
    refresh_representatives(data)
    assert group['representative'] == 'click'
    assert [e['node'] for e in group['entry_contexts']] == ['click']


def test_live_without_parent_matches_root_and_same_candidates_as_offline(tmp_path):
    data = layer()
    member(data, 'T001', 'root', profile('a'))
    member(data, 'T002', 'other', profile('b'))
    knowledge = TemplateKnowledge.__new__(TemplateKnowledge)
    knowledge.graph = SimpleNamespace(nodes={})
    knowledge.layer, knowledge.groups = data, {g['id']: g for g in data['templates']}
    knowledge.previous = knowledge.previous_match = knowledge.pending_action = None
    knowledge.profile = lambda observation: profile('a')
    knowledge.log = EventLog(tmp_path)
    matched = knowledge.match({'id': 'unseen', 'url': '/new'})
    offline = optional_entry_candidates(data, profile('a'), None, {'root': True})[0]
    assert matched['template_id'] == offline['template'] == 'T001'
    assert matched['scores'] == offline['scores'] and matched['ranking_basis'] == 'page_only'


def test_private_layer_adoption_preserves_existing_assignments():
    from web2kg.stage2.recovery import template_key
    old = layer()
    member(old, 'T001', 'root', profile('x'))
    old['version'] = 'context-structure-v7'
    old['parameters'].pop('matching_policy')
    upgraded = copy.deepcopy(old)
    memory_key = template_key(old, 'T001')
    adopt_matching_policy(upgraded)
    assert old['version'] == 'context-structure-v7'
    assert upgraded['version'] == VERSION and upgraded['assignments'] == old['assignments']
    assert upgraded['matching_policy_source']['version'] == old['version']
    assert template_key(upgraded, 'T001') == memory_key
    again = copy.deepcopy(upgraded)
    adopt_matching_policy(upgraded)
    assert upgraded == again


def test_representative_promotion_retains_template_experience_identity():
    from web2kg.stage2.recovery import template_key
    data = layer()
    member(data, 'T001', 'parent', profile('parent'))
    group = member(data, 'T002', 'root', profile('a', 'b', 'c'))
    member(data, 'T002', 'arrived', profile('a', 'b'), parent='parent')
    memory_key = template_key(data, 'T002')
    refresh_representatives(data)
    assert group['representative'] == 'arrived'
    assert template_key(data, 'T002') == memory_key
