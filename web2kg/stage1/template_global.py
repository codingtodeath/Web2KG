"""Batched structural-template maintenance; no model calls or graph mutation here."""
import copy
import math

from ..common.storage import digest
from .templates import (control_profile, next_template_id, refresh_representatives,
                        structure_fingerprint, uses_optional_entry)

RULE = 'unique-structure-silhouette-v1'
DEFAULT_TOLERANCE = .05


def template_state(layer):
    features = layer['parameters']['page_features']
    return {g['id']: digest([bool(g.get('isolated')), sorted({
        structure_fingerprint(layer['profiles'][k], features) for k in g['members']})])
        for g in layer['templates']}


def initialize_maintenance(layer):
    layer.setdefault('global_maintenance', {
        'template_state': template_state(layer), 'baseline_templates': len(layer['templates']),
        'pending_templates': []})


def account_changes(layer, *, minimum=5, fraction=.2):
    if type(minimum) is not int or minimum < 0 or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid global template-change threshold')
    initialize_maintenance(layer)
    state = template_state(layer)
    previous = layer['global_maintenance']
    old = previous['template_state']
    changed = sorted(k for k in old.keys() | state.keys() if old.get(k) != state.get(k))
    pending = sorted(set(previous['pending_templates']) | set(changed))
    previous.update(template_state=state, pending_templates=pending)
    count = len(pending)
    return dict(changed_templates=changed, pending_templates=pending, count=count,
                baseline_templates=previous['baseline_templates'], minimum=minimum, fraction=fraction,
                due=bool(minimum and count >= minimum and count >= previous['baseline_templates'] * fraction))


def reset_maintenance(layer):
    layer['global_maintenance'] = dict(template_state=template_state(layer),
        baseline_templates=len(layer['templates']), pending_templates=[])


def regroup(graph, layer, *, method='average', tolerance=DEFAULT_TOLERANCE):
    """Split/merge all structural groups, retaining exact partitions and verified local variants."""
    if not uses_optional_entry(layer):
        return layer, dict(status='disabled_legacy', changed=False)
    from .silhouette_selection import prepare, candidates, select
    old = layer['assignments']
    profiles = layer['profiles']

    def anchor(key):
        seen = set()
        while (key not in seen and old[key]['reason'] == 'observed_local_state_variant'
               and not profiles[key].get('isolated')):
            seen.add(key)
            parent = old[key].get('parent_node')
            if parent not in old or profiles[parent].get('isolated'):
                break
            key = parent
        return key

    anchors = {key: anchor(key) for key in sorted(old)}
    keys = sorted(set(anchors.values()))
    arrays = prepare([profiles[k] for k in keys], deduplicate=True, weights=layer['parameters']['page_features'])
    selection = select(arrays, candidates(arrays, method), tolerance=tolerance)
    predicted = dict(zip(keys, selection['predicted']))
    isolated = {k for k, eligible in zip(keys, arrays['eligible']) if not eligible}
    partitions = {}
    for key, root in anchors.items():
        partitions.setdefault(predicted[root], []).append(key)
    groups = sorted((sorted(members) for members in partitions.values()), key=lambda members: members[0])
    previous = {tuple(sorted(g['members'])): g for g in layer['templates']}
    audit = dict(rule=RULE, method=method, selection=selection,
                 anchors=anchors, templates_before=len(previous), templates_after=len(groups),
                 changed=set(previous) != {tuple(g) for g in groups})
    if not audit['changed']:
        return layer, {**audit, 'status': 'unchanged'}
    result = copy.deepcopy(layer)
    result.pop('reconciliation_cache', None)
    result['parameters']['global_clustering'] = dict(rule=RULE, method=method, tolerance=tolerance)
    # Reserve old IDs until every new partition has been allocated.
    reserved = copy.deepcopy(layer)
    new_groups, assigned = [], {}
    for members in groups:
        if tuple(members) in previous:
            group = copy.deepcopy(previous[tuple(members)])
        else:
            key = next_template_id(reserved)
            origins = sorted({old[k]['template'] for k in members})
            group = dict(id=key, members=members, representative=members[0],
                         regrouped_from=origins, experience_key=digest([RULE, key, members, origins]))
            if all(anchors[k] in isolated for k in members):
                group['isolated'] = True
            reserved['templates'].append(group)
        new_groups.append(group)
        assigned.update({k: group['id'] for k in members})
    result['templates'] = new_groups
    mapping = {g['id']: sorted({assigned[k] for k in g['members']}) for g in layer['templates']}
    active = {g['id'] for g in new_groups}
    for key, targets in mapping.items():
        if key not in active:
            result.setdefault('retired_templates', {})[key] = targets[0] if len(targets) == 1 else None
    for key, row in result['assignments'].items():
        row.setdefault('discovery_template', row['template'])
        row['template'] = assigned[key]
        row['parent_template'] = assigned.get(row.get('parent_node'))
        row['global_assignment_rule'] = RULE
    # Split transition evidence by its real endpoint nodes; many-to-one mapping is insufficient.
    controls = {edge: row['control'] for row in layer['transitions'] for edge in row['evidence_edges']}
    transitions = {}
    for edge in graph.data['edges']:
        source, target = assigned[edge['source']], assigned[edge['target']]
        control = controls.get(edge['id'])
        if control is None:
            control = control_profile(edge['action'])
        key = digest([source, target, control])
        row = transitions.setdefault(key, dict(id=key, source=source, target=target,
            control=copy.deepcopy(control), evidence_edges=[]))
        row['evidence_edges'].append(edge['id'])
    result['transitions'] = list(transitions.values())
    refresh_representatives(result)
    reset_maintenance(result)
    audit.update(status='candidate', mapping=mapping,
                 splits=[k for k, targets in mapping.items() if len(targets) > 1],
                 merges=[g['id'] for g in new_groups if len(g.get('regrouped_from', [])) > 1])
    return result, audit
