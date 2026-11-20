"""Name-free, region-scoped page structure with explicit evidence sufficiency."""
import json

REGIONS = {'main', 'navigation', 'banner', 'complementary', 'contentinfo', 'search', 'dialog'}
CONTROLS = {'link', 'button', 'textbox', 'combobox', 'checkbox', 'radio', 'slider', 'spinbutton', 'tab'}
PAGE_FEATURES = {'structure': .5, 'combinations': .5}
LEGACY_PAGE_FEATURES = {'structure': .4, 'combinations': .4, 'controls': .2}


class EmptyStructureError(ValueError):
    """A valid snapshot contains no usable visible structure."""


TAG_ROLES = {'a': 'link', 'button': 'button', 'textarea': 'textbox', 'select': 'combobox',
             'main': 'main', 'nav': 'navigation', 'aside': 'complementary', 'form': 'form',
             'ul': 'list', 'ol': 'list', 'li': 'listitem', 'p': 'paragraph',
             'article': 'article', 'table': 'table', 'tr': 'row', 'img': 'image'}


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left or right else 0.


def role(tree):
    return tree[0].split(':', 1)[0]


def contains_control(tree):
    return role(tree) in CONTROLS or any(contains_control(c) for c in tree[1])


def structure_profile(ax, dom):
    by_dom = {n.get('backendDOMNodeId'): n for n in ax if not n.get('ignored')}
    fallback = 0

    def walk(node):
        nonlocal fallback
        tag = node.get('nodeName', '').lower()
        if node.get('nodeType') not in {1, 9}:
            return None
        attrs = dict(zip(node.get('attributes', [])[::2], node.get('attributes', [])[1::2]))
        if tag in {'head', 'script', 'style', 'noscript'} or 'hidden' in attrs or attrs.get('aria-hidden') == 'true':
            return None
        kids = tuple(c for n in node.get('children', []) if (c := walk(n)) is not None)
        semantic = by_dom.get(node.get('backendNodeId'))
        if not semantic and not kids:
            return None
        label = (semantic or {}).get('role', {}).get('value', '')
        properties = {p['name']: p['value'].get('value') for p in (semantic or {}).get('properties', [])}
        if label in {'generic', 'none', 'Ignored', ''}:
            label = TAG_ROLES.get(tag, 'group')
            fallback += 1
        if tag in {'form', 'p', 'article', 'ul', 'ol', 'li'}:
            label = TAG_ROLES[tag]
        if tag == 'input':
            label = {'checkbox': 'checkbox', 'radio': 'radio', 'submit': 'button',
                     'button': 'button', 'hidden': 'hidden'}.get(attrs.get('type', 'text'), 'textbox')
        if label in {'StaticText', 'InlineTextBox', 'ListMarker', 'text', 'hidden'}:
            return None
        if label in {'Section', 'section', 'RootWebArea', 'WebArea'}:
            label = 'group'
        if label in CONTROLS - {'link', 'button', 'tab'}:
            label += ':' + attrs.get('type', '')
        elif label == 'heading':
            label += ':' + str(properties.get('level', tag))
        elif label == 'form':
            label += ':' + attrs.get('method', 'get').lower()
        # Collapse repeated instances BEFORE transparent containers. One item and many
        # items retain the same item subtree; no count-triggered collection label.
        kids = tuple(dict.fromkeys(kids))
        if label in {'list', 'listitem', 'directory'} and any(contains_control(c) for c in kids):
            label = 'group'
        if label == 'group':
            if len(kids) == 1:
                return kids[0]
            if not kids:
                return None
        return label, kids

    tree = walk(dom)
    if tree is None:
        raise EmptyStructureError('Snapshot has no usable visible structure')
    buckets = {}

    def bucket(key):
        return buckets.setdefault(key, {k: set() for k in ('structure', 'controls', 'combinations', 'units')})

    def collect(branch, region='body', trail=()):
        label, children = branch
        r = role(branch)
        if r in REGIONS:
            region = r if region == 'body' else region + '/' + r
            trail = ()
        target = bucket(region)
        trail += (label,)
        target['structure'].add('/'.join(trail[-4:]))
        if r in CONTROLS:
            target['controls'].add('/'.join(trail[-4:]))
        # Local productions preserve which children belong to the SAME parent.
        local = tuple(c[0] for c in children if role(c) not in REGIONS)
        if local:
            target['combinations'].add(json.dumps((trail[-3:], local)))
        if r not in REGIONS and len(local) >= 2 and contains_control(branch):
            target['units'].add(json.dumps((label, local)))
        if r in {'list', 'table', 'article', 'form'} and children:
            target['units'].add(json.dumps((label, local)))
        for child in children:
            collect(child, region, trail)

    collect(tree)
    # Ignore empty document wrappers; retain every actual region separately.
    if len(buckets) > 1 and buckets.get('body', {}).get('structure') == {'group'}:
        del buckets['body']
    main = [k for k in buckets if k.split('/')[-1] == 'main']
    core = main or list(buckets)
    def union(field):
        return sorted({v for k in core for v in buckets[k][field]})
    controls, units = union('controls'), union('units')
    return {'structure': union('structure'), 'controls': controls, 'repeated': units,
            'regions': sorted(buckets), 'has_main': bool(main), 'dom_fallback_nodes': fallback,
            'region_features': {k: {f: sorted(v) for f, v in b.items()} for k, b in buckets.items()},
            'evidence': {'sufficient': len(controls) >= 2 or bool(units),
                         'control_patterns': len(controls), 'content_units': len(units)}}


def region_comparison(a, b, weights=None):
    # Control paths are a subset of structure paths, retained only for diagnostics/old layers.
    weights = PAGE_FEATURES if weights is None else weights
    parts = {k: jaccard(a.get(k, []), b.get(k, [])) for k in weights if a.get(k) or b.get(k)}
    return {'score': sum(weights[k] * v for k, v in parts.items()) / sum(weights[k] for k in parts)
            if parts else 0., 'components': parts}


def page_comparison(a, b, weights=None):
    if a.get('isolated') or b.get('isolated'):
        return {'score': 0., 'evidence_sufficient': False, 'reason': 'isolated_empty_structure',
                'regions': {}, 'core_score': 0.}
    left, right = a['region_features'], b['region_features']
    keys = sorted(left.keys() | right.keys())
    main = [k for k in keys if k.split('/')[-1] == 'main']
    details = {k: region_comparison(left.get(k, {}), right.get(k, {}), weights) for k in keys}
    def mean(names):
        return sum(details[k]['score'] for k in names) / len(names)
    auxiliary = [k for k in keys if k not in main]
    score = (.85 * mean(main) + .15 * mean(auxiliary)) if main and auxiliary else mean(keys)
    sufficient = a['evidence']['sufficient'] and b['evidence']['sufficient']
    return {'score': score, 'evidence_sufficient': sufficient,
            'reason': 'structural_evidence' if sufficient else 'insufficient_core_structure',
            'regions': details, 'core_score': mean(main or keys)}


def page_similarity(a, b, weights=None):
    return page_comparison(a, b, weights)['score']
