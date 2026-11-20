"""One saved shortest-path parent per node; cross-links stay in the real graph."""
from collections import defaultdict, deque
import random

from .storage import digest


def shortest_backbone(graph):
    entry = graph.data.get('root_entry', {})
    root = entry.get('current_state') if entry.get('current_state') in graph.nodes else graph.data.get('root')
    previous = graph.data.get('backbone', {})
    seed = previous.get('seed', int(digest(['backbone-v1', root])[:16], 16))
    rng = random.Random(seed)
    outgoing, incoming = defaultdict(list), defaultdict(list)
    # These are maintenance-only links between versions of the logical URL root,
    # never fabricated browser transitions or permission to replay an action.
    aliases = [{'id': 'root-entry:' + k, 'source': root, 'target': k}
               for k in entry.get('states', []) if k in graph.nodes and k != root]
    for edge in [*graph.data['edges'], *aliases]:
        a, b = edge['source'], edge['target']
        if a in graph.nodes and b in graph.nodes and a != b:
            outgoing[a].append(edge)
            incoming[b].append(edge)
    depths, components, parents = {}, {}, {}
    unseen = set(graph.nodes)
    roots = []
    while unseen:
        start = root if root in unseen else min(unseen)
        roots.append(start)
        depths[start], components[start], parents[start] = 0, start, None
        unseen.remove(start)
        queue = deque([start])
        while queue:
            source = queue.popleft()
            for edge in sorted(outgoing[source], key=lambda e: e['id']):
                target = edge['target']
                if target in unseen:
                    unseen.remove(target)
                    depths[target], components[target] = depths[source] + 1, start
                    queue.append(target)
        for node in sorted(k for k in depths if components[k] == start and k != start):
            options = sorted((e for e in incoming[node] if components.get(e['source']) == start
                              and depths[e['source']] + 1 == depths[node]), key=lambda e: e['id'])
            old = previous.get('parents', {}).get(node)
            chosen = next((e for e in options if old and e['id'] == old['edge']), None) or rng.choice(options)
            parents[node] = {'node': chosen['source'], 'edge': chosen['id']}
    return {'version': 'shortest-backbone-v1', 'seed': seed, 'root': root, 'parents': parents,
            'depths': depths, 'component_roots': roots,
            'root_unreachable': sorted(k for k in graph.nodes if components[k] != root),
            'entry_alias_edges': aliases,
            'edge_scope': 'recorded_edges_and_logical_entry_aliases_not_replay_permission'}
