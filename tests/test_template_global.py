import copy
from types import SimpleNamespace

from web2kg.stage1.template_global import account_changes, initialize_maintenance, regroup, reset_maintenance
from web2kg.stage1.templates import VERSION, MATCHING_POLICY
from web2kg.stage1.page_structure import PAGE_FEATURES


def layer_fixture():
    structures = {'a': ['x'], 'b': ['x'], 'c': ['y'], 'd': ['y'], 'empty': []}
    profiles = {k: {'region_features': {'main': {'structure': f, 'combinations': []}},
                    'isolated': not bool(f), 'evidence': {'sufficient': bool(f)}} for k, f in structures.items()}
    assignments = {k: dict(template='T001' if k in {'a', 'c'} else 'T002',
        reason='new_template', parent_node=None, parent_template=None, incoming={'root': True}) for k in structures}
    assignments['empty']['template'] = 'T003'
    assignments['c'].update(parent_node='a', parent_template='T001', incoming={'kind': 'click'})
    groups = [dict(id='T001', members=['a', 'c'], representative='a', experience_key='old1'),
              dict(id='T002', members=['b', 'd'], representative='b', experience_key='old2'),
              dict(id='T003', members=['empty'], representative='empty', isolated=True)]
    layer = dict(version=VERSION, profiles=profiles, assignments=assignments, templates=groups,
                 transitions=[dict(id='edgegroup', source='T001', target='T001', control={'kind': 'click'}, evidence_edges=['e1', 'e2'])],
                 parameters=dict(matching_policy=MATCHING_POLICY, page_features=PAGE_FEATURES))
    graph = SimpleNamespace(data={'edges': [dict(id='e1', source='a', target='c', action={'kind': 'click'}),
                                           dict(id='e2', source='a', target='b', action={'kind': 'click'})]})
    return graph, layer


def test_trigger_counts_distinct_changed_templates_not_member_multiplicity():
    _, layer = layer_fixture()
    initialize_maintenance(layer)
    layer['profiles']['copy'] = copy.deepcopy(layer['profiles']['a'])
    layer['templates'][0]['members'].append('copy')
    assert not account_changes(layer, minimum=1, fraction=0)['due']
    layer['profiles']['a']['region_features']['main']['structure'].append('new')
    first = account_changes(layer, minimum=2, fraction=0)
    assert first['count'] == 1 and not first['due']
    layer['profiles']['a']['region_features']['main']['structure'].append('again')
    assert account_changes(layer, minimum=2, fraction=0)['count'] == 1
    layer['templates'].pop()  # retirement
    assert account_changes(layer, minimum=2, fraction=.5)['due']
    assert not account_changes(layer, minimum=0)['due']
    reset_maintenance(layer)
    assert account_changes(layer, minimum=1, fraction=0)['count'] == 0


def test_global_split_merge_remaps_each_evidence_edge_without_promoting_experience():
    graph, layer = layer_fixture()
    original = copy.deepcopy(layer)
    new, audit = regroup(graph, layer)
    assert layer == original
    assert audit['changed'] and set(audit['splits']) == {'T001', 'T002'}
    assignments = new['assignments']
    assert assignments['a']['template'] == assignments['b']['template']
    assert assignments['c']['template'] == assignments['d']['template'] != assignments['a']['template']
    assert assignments['empty']['template'] == 'T003'
    assert assignments['c']['parent_template'] == assignments['a']['template']
    for row in new['transitions']:
        for eid in row['evidence_edges']:
            edge = next(e for e in graph.data['edges'] if e['id'] == eid)
            assert row['source'] == assignments[edge['source']]['template']
            assert row['target'] == assignments[edge['target']]['template']
    assert len(new['transitions']) == 2
    assert new['retired_templates'] == {'T001': None, 'T002': None}
    assert not {g.get('experience_key') for g in new['templates']} & {'old1', 'old2'}
    again, second = regroup(graph, new)
    assert second['status'] == 'unchanged' and again is new


def test_verified_local_variant_follows_its_anchor_even_if_structure_differs():
    graph, layer = layer_fixture()
    layer['assignments']['c']['reason'] = 'observed_local_state_variant'
    new, audit = regroup(graph, layer)
    assert audit['anchors']['c'] == 'a'
    assert new['assignments']['c']['template'] == new['assignments']['a']['template']
