"""Graph-local query-slot equivalence; interaction-v2 and historical IDs stay intact."""
from copy import deepcopy
from urllib.parse import urlsplit, urlunsplit, unquote_plus

from .interaction import VERSION, state_id, state_signature
from .storage import digest

RULE = 'scoped-query-variation-v1'


def _parts(value):
    parts = urlsplit(value)
    return parts, [part.split('=', 1) for part in parts.query.split('&')] if parts.query else []


def small_difference(saved, observed):
    """Only query VALUE edits in <=2 slots/<=10% of controls; everything else equal."""
    if saved.get('identity_version') != VERSION or observed.get('identity_version') != VERSION:
        return None
    left, right = state_signature(saved), state_signature(observed)
    if left[:4] != right[:4] or left[5:] != right[5:] or saved.get('text', '') != observed.get('text', ''):
        return None
    if len(left[4]) != len(right[4]):
        return None
    slots, controls = [], set()
    for index, (old, new) in enumerate(zip(left[4], right[4])):
        if old == new:
            continue
        if set(old) != set(new):
            return None
        for field in old:
            if old[field] == new[field]:
                continue
            if field not in {'href', 'form_action'} or not isinstance(old[field], str) or not isinstance(new[field], str):
                return None
            a, aq = _parts(old[field]); b, bq = _parts(new[field])
            if a._replace(query='') != b._replace(query='') or len(aq) != len(bq):
                return None
            for qindex, (av, bv) in enumerate(zip(aq, bq)):
                if av == bv:
                    continue
                if len(av) != 2 or len(bv) != 2 or av[0] != bv[0]:
                    return None
                slots.append({'control': index, 'field': field, 'query_index': qindex,
                              'parameter': unquote_plus(av[0])})
                controls.add(index)
    if not 1 <= len(slots) <= 2 or len(controls) > max(1, len(left[4]) // 10):
        return None
    # Reject differences caused only by URL encoding/structure rather than the selected slots.
    return slots if scoped_fingerprint(saved, slots) == scoped_fingerprint(observed, slots) else None


def scoped_fingerprint(page, slots):
    signature = deepcopy(state_signature(page))
    try:
        for slot in slots:
            control = signature[4][slot['control']]
            parts, query = _parts(control[slot['field']])
            item = query[slot['query_index']]
            if len(item) != 2 or unquote_plus(item[0]) != slot['parameter']:
                return None
            item[1] = '<variable>'
            control[slot['field']] = urlunsplit(parts._replace(query='&'.join('='.join(v) for v in query)))
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return digest([signature, page.get('text', '')])


def judgment_payload(saved, observed, slots):
    """Complete stored page contexts plus addressable differences, with query values redacted."""
    before, after = state_signature(saved)[4], state_signature(observed)[4]
    symbols = {}
    def url_context(value):
        parts, query = _parts(value)
        rows = []
        for item in query:
            raw = item[1] if len(item) == 2 else None
            if raw is not None and raw not in symbols:
                symbols[raw] = f'VALUE_{len(symbols) + 1}'
            rows.append({'key': unquote_plus(item[0]), 'has_value': raw is not None,
                         'value': {'redacted': True, 'symbol': symbols.get(raw),
                                   'length': len(raw) if raw is not None else 0}})
        # Userinfo and fragments can contain credentials; neither is needed to locate the endpoint.
        return {'origin': parts.scheme + '://' + (parts.hostname or '') +
                (':' + str(parts.port) if parts.port else '') if parts.scheme else '',
                'path': parts.path, 'query': rows, 'fragment_present': bool(parts.fragment)}

    def action_context(action, index):
        value = deepcopy(action)
        for key in ('href', 'form_action'):
            if key in value and isinstance(value[key], str):
                value[key] = url_context(value[key])
        return {'ref': f'C{index:04d}', **value}

    def page_context(page, controls):
        return {'url': url_context(page['url']), 'headings': deepcopy(page.get('headings', [])),
                'regions': deepcopy(state_signature(page)[3]), 'visible_text': page.get('text', ''),
                'controls': [action_context(a, i) for i, a in enumerate(controls)]}

    left, right = page_context(saved, before), page_context(observed, after)
    differences = []
    for slot in slots:
        old, new = before[slot['control']], after[slot['control']]
        parts, aq = _parts(old[slot['field']]); _, bq = _parts(new[slot['field']])
        av, bv = aq[slot['query_index']][1], bq[slot['query_index']][1]
        pointer = f'/controls/{slot["control"]}/{slot["field"]}/query/{slot["query_index"]}/value'
        differences.append({**slot, 'control_ref': f'C{slot["control"]:04d}', 'context_pointer': pointer,
            'control_role': old.get('role'), 'control_name': old.get('name'),
            'method': old.get('method'), 'endpoint_path': parts.path,
            'query_keys': [unquote_plus(v[0]) for v in aq],
            'before_value': {'redacted': True, 'symbol': symbols[av], 'length': len(av)},
            'after_value': {'redacted': True, 'symbol': symbols[bv], 'length': len(bv)}})
    def context(action):
        return {key: action[key] for key in ('kind','role','name','type','method','state','disabled') if key in action}
    # Mask any repeated long query value also appearing in visible text/labels. Never dump raw AX/DOM.
    def redact(value):
        if isinstance(value, dict): return {k: redact(v) for k, v in value.items()}
        if isinstance(value, list): return [redact(v) for v in value]
        if isinstance(value, str):
            for raw in sorted(symbols, key=len, reverse=True):
                if len(raw) >= 8:
                    value = value.replace(raw, '<' + symbols[raw] + '>')
            return value
        return value
    return redact({'input_format': 'page-context-diff-v2', 'before': left, 'after': right,
            'control_count': len(before), 'all_other_identity_fields_equal': True,
            'visible_text_equal': True, 'differences': differences,
            'coverage': {'controls_omitted': 0, 'regions_omitted': 0,
                         'visible_text': 'All stored observation text; upstream extraction may truncate at 16000 characters.',
                         'raw_dom_and_cookies_included': False},
            'recorded_navigation_path': [context(a) for a in saved.get('path', [])]})


class StateResolver:
    def __init__(self, graph, judge=None, *, reserve=None, log=None):
        self.graph, self.judge, self.reserve = graph, judge, reserve
        self.log = log or getattr(judge, 'log', None)
        self.data = graph.data.setdefault('state_equivalence', {'rule': RULE, 'rules': [], 'decisions': []})
        if self.data['rule'] != RULE:
            raise ValueError('Unknown state equivalence rule')
        self.rejected = set()

    def emit(self, event, **data):
        if self.log:
            self.log.emit(event, **data)

    def _valid(self, rule, observed):
        saved = self.graph.nodes.get(rule['canonical'])
        return bool(saved and state_id(saved) == rule['canonical'] and
                    scoped_fingerprint(saved, rule['slots']) == rule['scope'] and
                    scoped_fingerprint(observed, rule['slots']) == rule['scope'])

    def matches(self, expected, observed):
        saved = self.graph.nodes.get(expected)
        if not saved or state_id(saved) != expected:
            return False
        return state_signature(saved) == state_signature(observed) or any(
            r['canonical'] == expected and self._valid(r, observed) for r in self.data['rules'])

    def resolve(self, observed, *, learn=False, expected=None):
        if observed.get('identity_version') != VERSION:
            return observed
        raw = state_id(observed)
        if raw in self.graph.nodes:
            return {**observed, 'id': raw}
        matches = {r['canonical'] for r in self.data['rules'] if self._valid(r, observed)}
        if len(matches) == 1:
            canonical = next(iter(matches))
            self.emit('state_equivalence_reused', raw_state_id=raw, canonical=canonical)
            return {**observed, 'id': canonical, 'raw_state_id': raw, 'equivalence_rule': RULE}
        if matches or not learn or self.judge is None:
            return {**observed, 'id': raw}
        keys = [expected] if expected else list(self.graph.nodes)
        candidates = [(key, small_difference(self.graph.nodes[key], observed)) for key in keys
                      if key in self.graph.nodes and state_id(self.graph.nodes[key]) == key]
        candidates = [(key, slots) for key, slots in candidates if slots]
        if len(candidates) != 1:
            return {**observed, 'id': raw}
        canonical, slots = candidates[0]
        saved = self.graph.nodes[canonical]
        scope = scoped_fingerprint(saved, slots)
        if any(key != canonical and scoped_fingerprint(node, slots) == scope
               for key, node in self.graph.nodes.items()):
            return {**observed, 'id': raw}  # Never silently merge already distinct graph nodes.
        payload = judgment_payload(saved, observed, slots)
        comparison = digest([canonical, payload])
        if comparison in self.rejected:
            return {**observed, 'id': raw}
        if self.reserve:
            self.reserve()
        result = self.judge.json('', payload, {}, 'state_equivalence')
        if result.get('decision') not in {'same', 'different', 'uncertain'}:
            raise ValueError('Invalid state equivalence decision')
        receipt = dict(canonical=canonical, raw_state_id=raw, slots=slots, judgment=result,
                       request_id=self.judge.last_request_id)
        self.data['decisions'].append(receipt)
        self.emit('state_equivalence_judged', **receipt)
        if result['decision'] == 'same':
            self.data['rules'].append({'canonical': canonical, 'slots': slots,
                'scope': scoped_fingerprint(saved, slots), 'request_id': self.judge.last_request_id})
        else:
            self.rejected.add(comparison)
        self.graph.save()
        return self.resolve(observed)
