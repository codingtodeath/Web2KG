"""Offline, evidence-linked page templates. Never changes state IDs or replay rules."""
import hashlib
import copy
import json
import math
from collections import Counter

from ..common.storage import digest
from .page_admission import observation_profile
from .control_structure import ControlContext, control_comparison, control_similarity

from .page_structure import (structure_profile, page_similarity, page_comparison, jaccard,
                             PAGE_FEATURES, LEGACY_PAGE_FEATURES)

VERSION = "context-structure-v8"
SUPPORTED_VERSIONS = {VERSION, 'context-structure-v7', 'context-structure-v6', 'context-structure-v5'}
PREVIOUS_VERSION = "context-structure-v6"
LEGACY_VERSION = "context-structure-v5"
PAGE_RULE = "page_structure_threshold"
LEGACY_RULE = "weighted_score_only"
DEFAULT_THRESHOLD = .85
DEFAULT_WEIGHTS = (.5, .5, 1.)
LOCAL_STATE_RULE = "observed-form-edit-v1"
MATCHING_POLICY = 'optional-entry-medoid-v1'


def uses_optional_entry(layer):
    return layer.get('version') == VERSION and layer['parameters'].get('matching_policy') == MATCHING_POLICY


def adopt_matching_policy(layer):
    """Upgrade a task's private layer, retaining all existing assignments and evidence."""
    if layer['parameters'].get('match_rule') != PAGE_RULE or uses_optional_entry(layer):
        return
    layer.setdefault('matching_policy_source', {'version': layer['version'], 'layer_digest': digest(layer)})
    for group in layer['templates']:
        group.setdefault('experience_key', digest([layer['parameters'], layer['profiles'][group['representative']]]))
    layer['version'] = VERSION
    layer['parameters']['matching_policy'] = MATCHING_POLICY


def structure_fingerprint(profile, features):
    # Hash only scored, name-free structure. Diagnostic counts and node IDs are excluded.
    return digest({region: {f: sorted(set(values.get(f, []))) for f in features}
                   for region, values in profile['region_features'].items()})


def core_members(layer, group):
    return [k for k in group['members']
            if layer['assignments'][k]['reason'] != 'observed_local_state_variant'] or list(group['members'])


def next_template_id(layer):
    used = {g['id'] for g in layer['templates']} | set(layer.get('retired_templates', {}))
    number = max((int(k[1:]) for k in used if k.startswith('T') and k[1:].isdigit()), default=0) + 1
    return f'T{number:03d}'


def real_entry(assignment):
    incoming = assignment['incoming']
    return (assignment.get('observed_parent', assignment.get('parent_node')) is not None
            and not incoming.get('root') and incoming.get('kind') not in {None, 'url_jump'})


def rank_candidates(layer, current, candidates):
    """Resolve equal ranking scores by core agreement, then content, never template order."""
    groups = {g['id']: g for g in layer['templates']}
    profiles, features = layer['profiles'], layer['parameters']['page_features']
    buckets = {}
    for candidate in candidates:
        if candidate['eligible']:
            buckets.setdefault(round(candidate['score'], 12), []).append(candidate)
    for tied in buckets.values():
        if len(tied) < 2:
            continue
        for candidate in tied:
            group = groups[candidate['template']]
            members = core_members(layer, group)
            candidate['core_mean'] = math.fsum(sorted(page_similarity(current, profiles[k], features)
                                                       for k in members)) / len(members)
            fingerprints = sorted(set(structure_fingerprint(profiles[k], features) for k in members))
            candidate['structural_fingerprint'] = digest(fingerprints)
            # Identical structure can remain in distinct historical groups. State IDs
            # are only a final reproducibility tie-break, never similarity evidence.
            candidate['stable_member_key'] = min(group['members'])
            candidate['tied_templates'] = sorted(c['template'] for c in tied)
            candidate['tie_rule'] = 'core_mean_then_structure_then_member'
    candidates.sort(key=lambda c: (-round(c['score'], 12), -round(c.get('core_mean', 0), 12),
                                   c.get('structural_fingerprint', ''), c.get('stable_member_key', '')))
    return candidates


def optional_entry_candidates(layer, current, parent_template, incoming):
    """Shared offline/online candidate comparison; missing entry context is not a penalty."""
    if current.get('isolated'):
        return []
    groups, assignments, profiles = layer['templates'], layer['assignments'], layer['profiles']
    features, parameters = layer['parameters']['page_features'], layer['parameters']
    by_id = {g['id']: g for g in groups}
    candidates = []
    for group in groups:
        if group.get('isolated'):
            continue
        anchor = group['representative']
        reference = assignments[anchor]
        other_parent = reference['parent_template']
        entry = control_comparison(incoming, reference['incoming'])
        comparable = (parent_template in by_id and other_parent in by_id
                      and incoming.get('available', False) and reference['incoming'].get('available', False)
                      and incoming.get('kind') != 'url_jump' and reference['incoming'].get('kind') != 'url_jump')
        parent_score = page_similarity(profiles[by_id[parent_template]['representative']],
                                      profiles[by_id[other_parent]['representative']], features) if comparable else 0.
        page_match = page_comparison(current, profiles[anchor], features)
        scores = {'parent': parent_score, 'incoming': entry['score'] if comparable else 0., 'page': page_match['score']}
        match = match_candidate(scores, parameters)
        if not comparable:
            match['score'] = scores['page']
        candidates.append({'template': group['id'], 'scores': scores, 'incoming_comparison': entry,
                           'page_comparison': page_match, 'context_available': bool(comparable),
                           'ranking_basis': 'weighted_context' if comparable else 'page_only', **match})
    return rank_candidates(layer, current, candidates)


def refresh_representatives(layer, only=None):
    """At an offline/task boundary, choose a real-entry medoid without moving members."""
    if not uses_optional_entry(layer):
        return []
    assignments, profiles = layer['assignments'], layer['profiles']
    features, changes = layer['parameters']['page_features'], []
    for group in layer['templates']:
        if only is not None and group['id'] not in only:
            continue
        if group.get('isolated'):
            group.setdefault('experience_key', digest(['isolated', group['representative']]))
            group['entry_contexts'] = []
            continue
        core = core_members(layer, group)
        entries = [k for k in group['members'] if real_entry(assignments[k])]
        pool = entries or core
        means = {k: math.fsum(sorted(page_similarity(profiles[k], profiles[m], features)
                                     for m in core)) / len(core) for k in pool}
        selected = min(pool, key=lambda k: (-round(means[k], 12), structure_fingerprint(profiles[k], features), k))
        previous = group['representative']
        # Choosing a representative does not create a new template identity.
        if 'experience_key' not in group:
            group['experience_key'] = digest([layer['parameters'], profiles.get(previous, profiles[selected])])
        group['representative'] = selected
        group['representative_selection'] = {'rule': MATCHING_POLICY, 'real_entry_preferred': bool(entries),
                                             'core_mean': means[selected], 'core_members': sorted(core)}
        group['entry_contexts'] = [{'node': k, **{f: assignments[k].get(f) for f in
                                   ('parent_node', 'observed_parent', 'parent_template', 'incoming')}}
                                  for k in sorted(entries)]
        if previous != selected:
            change = {'template': group['id'], 'previous': previous, 'representative': selected,
                      'core_mean': means[selected]}
            changes.append(change)
            layer.setdefault('representative_updates', []).append(change)
    return changes


def control_profile(action, context=None):
    if not action:
        return {"root": True}
    if context is None:
        return {"kind": action.get("kind"), "role": action.get("role"), "available": False,
                "unavailable_reason": "source_snapshot_missing"}
    return context.profile(action)


def state_evidence(node):
    return [{"control": a.get("id"), "role": a.get("role"), "state": a.get("state", {})}
            for a in node.get("actions", []) if a.get("state")]


def match_candidate(scores, parameters):
    """Page structure admits candidates; context ranks them. Keep saved v5 semantics."""
    weights = parameters['raw_weights']
    score = sum(scores[k] * w for k, w in weights.items()) / sum(weights.values())
    rule = parameters.get('match_rule', LEGACY_RULE)
    if rule not in {PAGE_RULE, LEGACY_RULE}:
        raise ValueError(f'Unknown template match rule: {rule}')
    eligible = (scores['page'] if rule == PAGE_RULE else score) >= parameters['threshold']
    return {'score': score, 'eligible': eligible, 'eligibility_reason': rule,
            'rejection_reason': None if eligible else
                ('page_below_threshold' if rule == PAGE_RULE else 'score_below_threshold')}


def form_edit_evidence(observation, action):
    """Name/value-free evidence that the same live form and field survived editing."""
    if not observation.get('dom') or not observation.get('ax'):
        return None
    path = action.get('xpath')
    fields = [a for a in observation.get('actions', []) if a.get('xpath') == path
              and a.get('role') == action.get('role') and not a.get('disabled')]
    if len(fields) != 1 or action.get('role') not in {'textbox', 'searchbox', 'combobox', 'spinbutton'}:
        return None
    dom = copy.deepcopy(observation['dom'])
    found = []

    def walk(node, current='', form=None):
        attrs = dict(zip(node.get('attributes', [])[::2], node.get('attributes', [])[1::2]))
        if node.get('nodeName') == 'FORM' or attrs.get('role') == 'form':
            form = (node, current, attrs)
        if current == path and form:
            found.append((node, form))
        counts = Counter()
        for child in node.get('children', []):
            if child.get('nodeType') == 1:
                tag = child['nodeName'].lower()
                counts[tag] += 1
                walk(child, f'{current}/{tag}[{counts[tag]}]', form)

    walk(dom)
    if len(found) != 1:
        return None
    field, (form, form_path, attrs) = found[0]
    evidence = {'document': dom['backendNodeId'], 'form': form['backendNodeId'],
                'field': field['backendNodeId'], 'form_path': form_path,
                'method': attrs.get('method', 'get').lower(),
                'action_digest': digest(attrs.get('action', '')),
                'controls': sorted((a['xpath'], a['role'], a.get('type', ''))
                    for a in observation['actions'] if a.get('xpath', '').startswith(form_path + '/'))}
    # Values, validation text and previews inside this form are state, not a new entry.
    form['children'] = []
    evidence['outside'] = structure_profile(observation['ax'], dom)['region_features']
    return evidence


def local_variant(source, target, action, a, b):
    """Only observed local UI changes may inherit a template; URL alone is insufficient."""
    if source["url"] != target["url"]:
        return False
    if action.get('kind') == 'type':
        # An Enter/unknown flag can submit the form; never infer an edit from URL alone.
        if action.get('press_enter') is not False:
            return False
        before = form_edit_evidence(source, action)
        return before is not None and before == form_edit_evidence(target, action)
    if action.get('kind') != 'click':
        return False
    base = set(a["structure"])
    retention = len(base & set(b["structure"])) / len(base) if base else 0
    added_form = any("form:" in f for f in set(b["structure"]) - base)
    expanded = "expanded" in action.get("state", {}) and any(
        a.get("id") == action.get("id") and "expanded" in a.get("state", {})
        and a["state"]["expanded"] != action["state"]["expanded"]
        for a in target.get("actions", []))
    return retention >= .90 and (expanded or added_form)


def assign_template(graph, layer, key, parent, incoming, action=None, observations=None):
    """Append one discovery without reclassifying any existing template."""
    groups, assignments, profiles = layer['templates'], layer['assignments'], layer['profiles']
    decisions = layer['decisions']
    node = graph.nodes[key]
    if profiles[key].get('isolated'):
        template = next_template_id(layer)
        groups.append({'id': template, 'representative': key, 'members': [key], 'isolated': True,
                       'experience_key': digest(['isolated', key])})
        assignments[key] = {'template': template, 'parent_node': parent,
            'parent_template': assignments[parent]['template'] if parent else None,
            'incoming': incoming, 'reason': 'isolated_empty_structure', 'state': state_evidence(node),
            'selected_match': None}
        decisions.append({'node': key, 'template': template, 'reason': 'isolated_empty_structure', 'candidates': []})
        return assignments[key]
    weight_map = layer['parameters']['raw_weights']
    weights = list(weight_map.values())
    equal = len(set(weights)) == 1
    page_features = layer['parameters']['page_features']
    context = {"parent": assignments[parent]["template"] if parent else None, "incoming": incoming}
    candidates = optional_entry_candidates(layer, profiles[key], context['parent'], incoming) if uses_optional_entry(layer) else []
    for group in ([] if uses_optional_entry(layer) else groups):
        if group.get('isolated'):
            continue
        anchor = group["representative"]
        reference = assignments[anchor]
        if (parent is None) != (reference["parent_node"] is None):
            continue
        # Compare parent templates by their fixed representative structure, not raw IDs.
        parent_score = page_similarity(profiles[groups_by_id(groups, context["parent"])["representative"]],
                                       profiles[groups_by_id(groups, reference["parent_template"])["representative"]],
                                       page_features) if parent else 1.
        entry = control_comparison(incoming, reference["incoming"])
        page_match = page_comparison(profiles[key], profiles[anchor], page_features)
        scores = {"parent": parent_score, "incoming": entry['score'],
                  "page": page_match["score"]}
        candidates.append({"template": group["id"], "scores": scores,
                           "incoming_comparison": entry, "page_comparison": page_match,
                           **match_candidate(scores, layer['parameters'])})
    if not uses_optional_entry(layer):
        candidates.sort(key=lambda c: (-c["score"], c["template"]))
    best = next((c for c in candidates if c["eligible"]), None)
    if parent and action and action.get('kind') == 'type' and observations is None:
        observations = []
        for state in (parent, key):
            folder = graph.directory / 'snapshots' / state
            observations.append({**graph.nodes[state], **{name: json.loads((folder / f'{name}.json').read_text(encoding='utf-8'))
                                                          for name in ('ax', 'dom')}})
    before, after = observations if observations else (graph.nodes.get(parent), node)
    variant = parent and action and not profiles[parent].get('isolated') and local_variant(before, after, action, profiles[parent], profiles[key])
    if variant:
        template = assignments[parent]["template"]
        reason = "observed_local_state_variant"
    elif best:
        template, reason = best["template"], "equal_weight_match" if equal else "weighted_match"
    else:
        template, reason = next_template_id(layer), "new_template"
        groups.append({"id": template, "representative": key, "members": []})
    groups_by_id(groups, template)["members"].append(key)
    assignments[key] = {"template": template, "parent_node": parent, "parent_template": context["parent"],
                        "incoming": incoming, "reason": reason, "state": state_evidence(node),
                        "selected_match": best if reason in {"equal_weight_match", "weighted_match"} else None}
    if variant and action.get('kind') == 'type':
        assignments[key]['local_state_rule'] = LOCAL_STATE_RULE
    decisions.append({"node": key, "template": template, "reason": reason, "candidates": candidates})
    return assignments[key]


def build_templates(graph, threshold=DEFAULT_THRESHOLD, weights=DEFAULT_WEIGHTS, match_rule=PAGE_RULE):
    if not 0 <= threshold <= 1:
        raise ValueError("Similarity threshold must be in [0, 1]")
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Parent/incoming/page weights must be finite, nonnegative and have a positive sum")
    if match_rule not in {PAGE_RULE, LEGACY_RULE}:
        raise ValueError(f'Unknown template match rule: {match_rule}')
    weight_map = dict(zip(("parent", "incoming", "page"), weights))
    page_features = dict(LEGACY_PAGE_FEATURES if match_rule == LEGACY_RULE else PAGE_FEATURES)
    profiles, snapshot_hashes, contexts = {}, {}, {}
    for key in graph.nodes:
        folder = graph.directory / "snapshots" / key
        inputs = {}
        for name in ("ax.json", "dom.json"):
            raw = (folder / name).read_bytes()
            snapshot_hashes[f"{key}/{name}"] = hashlib.sha256(raw).hexdigest()
            inputs[name] = json.loads(raw)
        profiles[key] = observation_profile({'ax': inputs['ax.json'], 'dom': inputs['dom.json']})
        contexts[key] = ControlContext(inputs["ax.json"], inputs["dom.json"])
    groups, assignments, decisions = [], {}, []
    parameters = {"raw_weights": weight_map, "threshold": threshold, "match_rule": match_rule,
                  "weights": [w / sum(weights) for w in weights],
                  "page_regions": {"main": .85, "auxiliary": .15}, "page_features": page_features,
                  "page_evidence": "two_control_patterns_or_content_unit_diagnostic_only"}
    if match_rule == PAGE_RULE:
        parameters['matching_policy'] = MATCHING_POLICY
    layer = {"version": VERSION if match_rule == PAGE_RULE else LEGACY_VERSION,
             "templates": groups, "assignments": assignments, "profiles": profiles,
             "decisions": decisions, "parameters": parameters}
    # Depth and insertion order preserve the recorded discovery order, not URL order.
    for key in sorted(graph.nodes, key=lambda k: graph.nodes[k]["depth"]):
        node = graph.nodes[key]
        # Classification needs a known incoming edge, not a validated root-to-page route.
        # A live page after an excluded loading state can legitimately have no parent.
        path = node.get('path', [])
        entries = [e for e in graph.data['edges'] if path and e['target'] == key
                   and e['action'] == path[-1] and e['source'] in assignments]
        edge = entries[0] if len(entries) == 1 else None
        parent = edge["source"] if edge else None
        incoming = control_profile(edge["action"] if edge else None, contexts.get(parent))
        assign_template(graph, layer, key, parent, incoming, edge["action"] if edge else None)
    refresh_representatives(layer)
    transitions = {}
    for edge in graph.data["edges"]:
        source, target = (assignments[edge[k]]["template"] for k in ("source", "target"))
        control = control_profile(edge["action"], contexts[edge["source"]])
        key = digest([source, target, control])
        row = transitions.setdefault(key, {"id": key, "source": source, "target": target,
                                          "control": control, "evidence_edges": []})
        row["evidence_edges"].append(edge["id"])
    return {**layer,
            "graph_digest": digest(graph.data), "snapshot_sha256": snapshot_hashes,
            "transitions": list(transitions.values()),
            "summary": {"nodes": len(graph.nodes), "templates": len(groups),
                        "low_evidence_nodes": sum(not p["evidence"]["sufficient"] for p in profiles.values()),
                        "singleton_templates": sum(len(g["members"]) == 1 for g in groups),
                        "cross_url_templates": sum(len({graph.nodes[k]["url"] for k in g["members"]}) > 1 for g in groups),
                        "local_variants": sum(a["reason"] == "observed_local_state_variant" for a in assignments.values()),
                        "observed_edges": len(graph.data["edges"]), "template_transitions": len(transitions)}}


def groups_by_id(groups, key):
    return next(group for group in groups if group["id"] == key)
