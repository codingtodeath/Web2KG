"""Cached complete-link template consolidation at task boundaries; no model calls."""
import copy
import math
import time
from itertools import combinations

from ..common.storage import digest
from .page_structure import page_similarity
from .templates import core_members, next_template_id, refresh_representatives, structure_fingerprint, uses_optional_entry

RULE = 'core-complete-link-v1'


def reconcile(layer):
    """Return a private layer and audit; never mutate caller's assignments or cache."""
    started = time.perf_counter()
    audit = {'rule': RULE, 'merges': [], 'mapping': {}, 'affected': [],
             'page_comparisons': 0, 'cache_hits': 0, 'pair_evaluations': 0}
    if not uses_optional_entry(layer):
        return layer, {**audit, 'status': 'disabled_legacy', 'seconds': 0.}
    # Profiles and discovery decisions are immutable here; avoid copying their large evidence payloads.
    result = {**layer, **{k: copy.deepcopy(layer[k]) for k in
              ('templates', 'assignments', 'retired_templates', 'representative_updates') if k in layer}}
    profiles = result['profiles']
    features, threshold = result['parameters']['page_features'], result['parameters']['threshold']
    rule_key = digest([RULE, features, threshold])
    prior = result.get('reconciliation_cache', {})
    cache = copy.deepcopy(prior.get('pairs', {})) if prior.get('rule_key') == rule_key else {}
    fingerprints = {k: structure_fingerprint(p, features) for k, p in profiles.items()}
    scores = {}

    def score(a, b):
        key = tuple(sorted((fingerprints[a], fingerprints[b])))
        if key not in scores:
            scores[key] = page_similarity(profiles[a], profiles[b], features)
            audit['page_comparisons'] += 1
        return scores[key]

    def describe(group):
        core = sorted(core_members(result, group), key=lambda k: (fingerprints[k], k))
        # Multiplicity and membership matter, even when two pages have identical structure.
        identity = digest([(k, fingerprints[k]) for k in core])
        return {'group': group, 'core': core, 'identity': identity,
                'stable': (digest(sorted(set(fingerprints[k] for k in core))), min(group['members']))}

    def compare(a, b):
        key = digest(sorted((a['identity'], b['identity'])))
        if key in cache:
            audit['cache_hits'] += 1
            return key, cache[key]
        audit['pair_evaluations'] += 1
        # Use a core page, never a form/expansion variant representative, as the cheap gate.
        first = score(a['core'][0], b['core'][0])
        row = {'eligible': False, 'seed_score': first, 'cross_pairs': len(a['core']) * len(b['core'])}
        if first >= threshold:
            values = [score(x, y) for x in a['core'] for y in b['core']]
            row.update(minimum=min(values), mean=math.fsum(sorted(values)) / len(values),
                       eligible=min(values) >= threshold)
        cache[key] = row
        return key, row

    original_groups = {g['id']: g for g in layer['templates']}
    origins = {k: [k] for k in original_groups}
    final_keys = set()
    while True:
        descriptions = [describe(g) for g in result['templates'] if not g.get('isolated')]
        candidates, final_keys = [], set()
        for a, b in combinations(descriptions, 2):
            key, row = compare(a, b)
            final_keys.add(key)
            if row['eligible']:
                candidates.append((a, b, row))
        if not candidates:
            break
        a, b, comparison = min(candidates, key=lambda c: (
            -round(c[2]['minimum'], 12), -round(c[2]['mean'], 12),
            sorted((c[0]['stable'], c[1]['stable']))))
        old_ids = {a['group']['id'], b['group']['id']}
        key = next_template_id(result)
        members = sorted(set(a['group']['members']) | set(b['group']['members']))
        source_keys = sorted({g.get('experience_key') or digest([result['parameters'], profiles[g['representative']]])
                              for g in (a['group'], b['group'])})
        group = {'id': key, 'members': members, 'representative': members[0],
                 'merged_from': sorted(old_ids), 'experience_sources': source_keys,
                 'experience_key': digest([RULE, source_keys, members])}
        result['templates'] = [g for g in result['templates'] if g['id'] not in old_ids] + [group]
        origins[key] = sorted({old for parent in old_ids for old in origins.pop(parent)})
        for old in old_ids:
            result.setdefault('retired_templates', {})[old] = key
        for old, target in list(result.get('retired_templates', {}).items()):
            if target in old_ids:
                result['retired_templates'][old] = key
        for assigned in result['assignments'].values():
            if assigned['template'] in old_ids:
                assigned.setdefault('discovery_template', assigned['template'])
                assigned['template'] = key
            if assigned['parent_template'] in old_ids:
                assigned['parent_template'] = key
        refresh_representatives(result, {key})
        audit['merges'].append({'sources': sorted(old_ids), 'target': key, **comparison,
                                'members': len(members)})
    mapping = {old: key for key, originals in origins.items() for old in originals}
    changed = {old for old, new in mapping.items() if old != new}
    affected = {mapping[old] for old in changed}
    transitions = {}
    for old in layer.get('transitions', []):
        source, target = mapping[old['source']], mapping[old['target']]
        key = digest([source, target, old['control']])
        row = transitions.setdefault(key, {**copy.deepcopy(old), 'id': key, 'source': source,
                                          'target': target, 'evidence_edges': []})
        row['evidence_edges'] = sorted(set(row['evidence_edges']) | set(old['evidence_edges']))
        if old['source'] in changed or old['target'] in changed:
            affected.update((source, target))
    if changed:
        result['transitions'] = list(transitions.values())
        for old in layer['assignments'].values():
            if old['parent_template'] in changed:
                affected.add(mapping[old['template']])
        # Entry contexts refer to current groups; discovery decisions retain their original IDs.
        refresh_representatives(result, affected)
    result['reconciliation_cache'] = {'rule_key': rule_key, 'pairs': {k: cache[k] for k in final_keys}}
    audit.update(status='completed', mapping=mapping, affected=sorted(affected),
                 templates_before=len(layer['templates']), templates_after=len(result['templates']),
                 seconds=round(time.perf_counter() - started, 6))
    return result, audit


def apply_graph_templates(graph, layer):
    """Update annotations only. Node/edge IDs, actions, paths and replay permissions stay intact."""
    assignments = layer['assignments']
    for key, assigned in assignments.items():
        graph.nodes[key]['template_id'] = assigned['template']
    for edge in graph.data['edges']:
        for side in ('source', 'target'):
            if edge[side] in assignments and side + '_template' in edge:
                edge[side + '_template'] = assignments[edge[side]]['template']


def checked_mapping(graph, parent_layer, current):
    """Verify a task-boundary remap before inheriting a historical website framework."""
    import json
    from .function_cards import file_hash
    record = graph.data.get('template_reconciliation')
    if not record:
        return {}
    path = (graph.directory / record['file']).resolve()
    if (not path.is_relative_to(graph.directory.resolve()) or file_hash(path) != record['sha256']):
        raise ValueError('Template reconciliation manifest changed')
    audit = json.loads(path.read_text(encoding='utf-8'))
    before_path = (graph.directory / audit['before_layer']).resolve()
    if (not before_path.is_relative_to(graph.directory.resolve()) or file_hash(before_path) != audit['before_sha256']):
        raise ValueError('Template reconciliation source changed')
    before = json.loads(before_path.read_text(encoding='utf-8'))
    mapping = audit['mapping']
    if (audit['rule'] != RULE or set(mapping) != {g['id'] for g in before['templates']}
            or audit['after_assignments_digest'] != digest(current['assignments'])):
        raise ValueError('Template reconciliation result changed')
    if any(before['assignments'].get(k, {}).get('template') != a['template']
           for k, a in parent_layer['assignments'].items()):
        raise ValueError('Template reconciliation ancestry differs')
    if any(current['assignments'].get(k, {}).get('template') != mapping[a['template']]
           for k, a in before['assignments'].items()):
        raise ValueError('Unrecorded template reassignment')
    return mapping
