"""Name-free incoming-control context from the saved source AX tree."""
import json
from collections import Counter, defaultdict

ANCHORS = {"listitem", "row", "article", "form", "main", "navigation", "region",
           "dialog", "search", "complementary", "banner", "contentinfo"}
TEXT = {"StaticText", "InlineTextBox", "ListMarker"}
TRANSPARENT = {"generic", "none", "Ignored", ""}


def overlap(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a or b else 0.


class ControlContext:
    def __init__(self, ax, dom):
        self.paths, self.nodes, self.parents = {}, {}, {}
        by_id = {n['nodeId']: n for n in ax}
        children = defaultdict(list)
        for n in ax:
            children[n.get('parentId')].append(n)

        def prune(raw):
            role = raw.get('role', {}).get('value', '')
            if role in TEXT:
                return []
            kids = [child for n in children[raw['nodeId']] for child in prune(n)]
            if raw.get('ignored') or role in TRANSPARENT:
                if len(kids) <= 1 or raw.get('ignored'):
                    return kids
                role = 'group'
            node = {'id': raw['nodeId'], 'role': role, 'children': kids}
            self.nodes[node['id']] = node
            for child in kids:
                self.parents[child['id']] = node
            return [node]

        for raw in ax:
            if raw.get('parentId') not in by_id:
                prune(raw)
        by_backend = {n.get('backendDOMNodeId'): n['nodeId'] for n in ax
                      if n['nodeId'] in self.nodes and not n.get('ignored')}

        def dom_walk(node, path=''):
            key = by_backend.get(node.get('backendNodeId'))
            if key:
                self.paths[path] = key
            counts = Counter()
            for child in node.get('children', []):
                if child.get('nodeType') == 1:
                    tag = child['nodeName'].lower()
                    counts[tag] += 1
                    dom_walk(child, f'{path}/{tag}[{counts[tag]}]')

        dom_walk(dom)

    def shape(self, node, depth=2, target=None):
        label = node['role'] + ('@target' if node['id'] == target else '')
        kids = [self.shape(c, depth - 1, target) for c in node['children']] if depth else []
        # Repeated sibling counts and entity position do not define the context.
        return (label, tuple(dict.fromkeys(kids)))

    def profile(self, action):
        if not action:
            return {'root': True}
        result = {'kind': action.get('kind'), 'role': action.get('role'), 'available': False}
        key = self.paths.get(action.get('xpath'))
        if key not in self.nodes or self.nodes[key]['role'] != action.get('role'):
            return {**result, 'unavailable_reason': 'control_not_resolved_in_source_ax'}
        node = self.nodes[key]
        parent = self.parents.get(key)
        if parent is None:
            return {**result, 'unavailable_reason': 'parent_not_observed'}
        ancestors, current = [], parent
        while current is not None:
            ancestors.append(current)
            current = self.parents.get(current['id'])
        anchor, distance = None, None
        for depth, ancestor in enumerate(ancestors, 1):
            outer = self.parents.get(ancestor['id'])
            repeated = outer and sum(self.shape(c) == self.shape(ancestor) for c in outer['children']) >= 2
            if ancestor['role'] in ANCHORS or repeated:
                # Semantic anchors must not change when another list item appears.
                anchor, distance = (ancestor['role'] if ancestor['role'] in ANCHORS else 'repeated_item'), depth
                break
        structure = set()

        def paths(n, depth, trail=()):
            trail += (n['role'] + ('@target' if n['id'] == key else ''),)
            structure.add('/'.join(trail))
            if depth:
                for child in n['children']:
                    paths(child, depth - 1, trail)

        # Immediate parent plus one outer parent preserve local nesting, never absolute depth.
        paths(ancestors[min(1, len(ancestors) - 1)], 3)
        siblings = sorted({json.dumps(self.shape(c), ensure_ascii=False)
                           for c in parent['children'] if c['id'] != key})
        return {**result, 'available': True, 'relative_depth': distance, 'anchor_role': anchor,
                'parent_structure': sorted(structure), 'siblings': siblings}


def control_comparison(a, b):
    if a.get('root') or b.get('root'):
        return {'score': float(a.get('root') == b.get('root')), 'reason': 'root', 'components': {}}
    if a.get('kind') != b.get('kind') or a.get('role') != b.get('role'):
        return {'score': 0., 'reason': 'action_or_role_mismatch', 'components': {}}
    if not a.get('available') or not b.get('available'):
        return {'score': 0., 'reason': 'missing_source_ax_context', 'components': {}}
    parts = {'parent_structure': overlap(a['parent_structure'], b['parent_structure'])}
    weights = {'parent_structure': .5, 'siblings': .3, 'relative_depth': .2}
    if a['siblings'] or b['siblings']:
        parts['siblings'] = overlap(a['siblings'], b['siblings'])
    if a['relative_depth'] is not None and b['relative_depth'] is not None:
        parts['relative_depth'] = (1 / (1 + abs(a['relative_depth'] - b['relative_depth']))
                                   if a['anchor_role'] == b['anchor_role'] else 0.)
    return {'score': sum(weights[k] * v for k, v in parts.items()) / sum(weights[k] for k in parts),
            'reason': 'structural_match', 'components': parts}


def control_similarity(a, b):
    return control_comparison(a, b)['score']
