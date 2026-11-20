"""Task-boundary leaf eviction. Retrieval ranking is deliberately independent."""
import math
import heapq
import time
import shutil
from collections import Counter
from ..common.backbone import shortest_backbone

RULE = 'template-idle-backbone-leaf-v3'


def record_retrieval(graph, ids):
    if 'retention' not in graph.data:
        for node in graph.nodes.values():
            node.setdefault('last_retrieved_round', 0)
    state = graph.data.setdefault('retention', {'retrieval_round': 0})
    state['retrieval_round'] += 1
    for key in ids:
        if key in graph.nodes:
            graph.nodes[key]['last_retrieved_round'] = state['retrieval_round']


def retention_score(importance, members, idle_rounds, half_life):
    """Higher means retain longer. Log scores avoid underflow in long experiments."""
    if type(importance) not in (int, float) or not math.isfinite(importance) or not 0 <= importance <= 1:
        importance = .5
    structural = 1 / max(1, members)
    idle_rounds = max(0, idle_rounds)
    log_score = math.log2((importance + structural) / 2) - idle_rounds / half_life
    return {'functional_importance': importance, 'template_members': members,
            'structural_importance': structural, 'idle_rounds': idle_rounds,
            'recency': 2 ** (-idle_rounds / half_life),
            'retention_score': 2 ** log_score, 'log2_retention': log_score}


def plan_pruning(graph, layer, cards, *, max_nodes, protected=(), half_life=20):
    """Read-only, deterministic batch plan with dynamic template and leaf priorities.

    Within one template only idle rounds differ, so its oldest eligible leaf always
    comes first. A second heap orders these template leaders. Each deletion updates
    its own template and templates of newly exposed parents, not every candidate.
    """
    started = time.perf_counter()
    if type(max_nodes) is not int or max_nodes < 0 or not math.isfinite(half_life) or half_life <= 0:
        raise ValueError('Invalid graph capacity or retrieval half-life')
    before = len(graph.nodes)
    backbone = shortest_backbone(graph)
    audit = {'rule': RULE, 'before': before, 'limit': max_nodes, 'removed': [],
             'backbone': backbone,
             'retired_templates': [], 'affected': [], 'half_life_retrievals': half_life,
             'initial_ranking': [], 'score_evaluations': 0,
             'tie_rule': 'log2_retention_then_node_id', 'remaining_excess': max(0, before - max_nodes)}
    if not max_nodes or before <= max_nodes:
        return {**audit, 'after': before, 'status': 'disabled' if not max_nodes else 'within_limit',
                'remaining_excess': 0, 'planning_seconds': time.perf_counter() - started}
    keep = set(protected) | {graph.data.get(k) for k in ('root', 'task_start', 'navigation_home')}
    keep.update(graph.data.get('url_entries', {}))
    keep.add(graph.data.get('root_entry', {}).get('current_state'))
    remaining = set(graph.nodes)
    keep &= remaining
    assignments = layer['assignments']
    templates = {k: assignments.get(k, {}).get('template') for k in remaining}
    # Unknown states are individual units, never one shared fictitious template.
    groups = {k: ('template', templates[k]) if templates[k] is not None else ('node', k) for k in remaining}
    counts = Counter(groups.values())
    round_number = graph.data.get('retention', {}).get('retrieval_round', 0)
    last_used = {k: min(round_number, graph.nodes[k].get('last_retrieved_round', round_number)) for k in remaining}
    outgoing = {k: set() for k in remaining}
    parents = {k: set() for k in remaining}
    for child, parent in backbone['parents'].items():
        if parent is not None:
            outgoing[parent['node']].add(child)
            parents[child].add(parent['node'])
    leaves = {k for k in remaining - keep if not outgoing[k]}

    def score(key):
        audit['score_evaluations'] += 1
        return {'node': key, 'template': templates[key],
                **retention_score(cards.get(templates[key], {}).get('importance', .5),
                                  counts[groups[key]], round_number - last_used[key], half_life)}

    audit['initial_ranking'] = sorted((score(k) for k in leaves), key=lambda r: (r['log2_retention'], r['node']))
    queues, versions, leaders = {}, Counter(), []
    for key in sorted(leaves):
        heapq.heappush(queues.setdefault(groups[key], []), (last_used[key], key))

    def push_leader(group):
        versions[group] += 1
        queue = queues.get(group, [])
        while queue and queue[0][1] not in remaining:
            heapq.heappop(queue)
        if queue:
            key = queue[0][1]
            row = score(key)
            heapq.heappush(leaders, (row['log2_retention'], key, versions[group], row))

    for group in sorted(queues):
        push_leader(group)
    while len(remaining) > max_nodes and leaders:
        _, key, revision, row = heapq.heappop(leaders)
        if revision != versions[groups[key]] or key not in remaining:
            continue
        row.update(step=len(audit['removed']) + 1, eligible_leaves=len(leaves),
                   remaining_before=len(remaining))
        audit['removed'].append(row)
        leaves.remove(key)
        remaining.remove(key)
        counts[groups[key]] -= 1
        affected_groups = {groups[key]}
        exposed = []
        for parent in parents[key] & remaining:
            outgoing[parent].discard(key)
            if not outgoing[parent] and parent not in keep and parent not in leaves:
                leaves.add(parent)
                exposed.append(parent)
                heapq.heappush(queues.setdefault(groups[parent], []), (last_used[parent], parent))
                affected_groups.add(groups[parent])
        row['exposed_leaves'] = sorted(exposed)
        for group in sorted(affected_groups):
            push_leader(group)

    audit.update(after=len(remaining), remaining_excess=max(0, len(remaining) - max_nodes),
                 protected_nodes=sorted(keep), blocked_nonleaves=len(remaining - keep - leaves),
                 status='trimmed' if len(remaining) <= max_nodes else 'limited_by_protection_or_no_leaves',
                 planning_seconds=time.perf_counter() - started)
    return audit


def trim_graph(graph, layer, cards, *, max_nodes, protected=(), half_life=20):
    """Plan and apply one completed-task batch; ordinary finalization publishes its assets."""
    audit = plan_pruning(graph, layer, cards, max_nodes=max_nodes, protected=protected, half_life=half_life)
    graph.data['backbone'] = audit['backbone']
    removed = {row['node'] for row in audit['removed']}
    if not removed:
        return audit
    assignments, edges = layer['assignments'], graph.data['edges']

    # Check every destination before mutating the graph or unlinking local snapshot names.
    root = (graph.directory / 'snapshots').resolve()
    folders = [(graph.directory / 'snapshots' / key).resolve() for key in removed]
    if any(p == root or not p.is_relative_to(root) for p in folders):
        raise ValueError('Snapshot deletion must stay inside the task snapshot directory')
    affected = {assignments[k]['template'] for k in removed if k in assignments}
    removed_edges = {e['id'] for e in edges if e['source'] in removed or e['target'] in removed}
    affected.update(assignments[k]['template'] for e in edges if e['id'] in removed_edges
                    for k in (e['source'], e['target']) if k in assignments)
    graph.data['edges'] = [e for e in edges if e['id'] not in removed_edges]
    for key in removed:
        graph.nodes.pop(key)
        for field in ('assignments', 'profiles'):
            layer[field].pop(key, None)
    # These are active lookup tables; transition_records/new_states remain historical evidence.
    for field in ('url_entries', 'rankings'):
        if field in graph.data:
            graph.data[field] = {k: v for k, v in graph.data[field].items() if k not in removed}
    if 'root_entry' in graph.data:
        entry = graph.data['root_entry']
        entry['states'] = [k for k in entry.get('states', []) if k not in removed]
    layer['snapshot_sha256'] = {k: v for k, v in layer['snapshot_sha256'].items()
                                if k.split('/', 1)[0] not in removed}
    layer['decisions'] = [r for r in layer.get('decisions', []) if r['node'] not in removed]
    for group in layer['templates']:
        group['members'] = [k for k in group['members'] if k not in removed]
    retired = {g['id'] for g in layer['templates'] if not g['members']}
    layer['templates'] = [g for g in layer['templates'] if g['members']]
    mapping = layer.setdefault('retired_templates', {})
    mapping.update({k: None for k in retired})
    for key, target in list(mapping.items()):
        if target in retired:
            mapping[key] = None
    for group in layer['templates']:
        if group['representative'] in removed:
            group['representative'] = min(group['members'])
        group['entry_contexts'] = [r for r in group.get('entry_contexts', []) if r['node'] not in removed]
        if group['id'] in affected:
            group.pop('representative_selection', None)
    for assignment in assignments.values():
        lost_parent = any(assignment.get(f) in removed for f in ('parent_node', 'observed_parent'))
        for field in ('parent_node', 'observed_parent'):
            if assignment.get(field) in removed:
                assignment[field] = None
        if lost_parent or assignment.get('parent_template') in retired:
            affected.add(assignment['template'])
            assignment['parent_template'] = None
    transitions = []
    for row in layer.get('transitions', []):
        row['evidence_edges'] = [k for k in row['evidence_edges'] if k not in removed_edges]
        if row['source'] not in retired and row['target'] not in retired and row['evidence_edges']:
            transitions.append(row)
    layer['transitions'] = transitions
    layer.pop('reconciliation_cache', None)
    graph.invalidate_reachability()
    graph.data['backbone'] = shortest_backbone(graph)
    for folder in folders:
        if folder.exists():
            shutil.rmtree(folder)  # Unlinks this task's names, never writes shared hardlink contents.
    audit.update(affected=sorted(affected - retired), retired_templates=sorted(retired))
    return audit
