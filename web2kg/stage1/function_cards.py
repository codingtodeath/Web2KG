"""One evidence-bounded model summary per template, without changing the graph."""
import hashlib
import copy
import html
import json
import shutil
from pathlib import Path

from ..common.responses import object_schema
from ..common.response_output import OutputError
from ..common.storage import digest, write_json
from .templates import SUPPORTED_VERSIONS
from .selection import group_key

TEXT = {"type": "string"}
def text_list(limit, minimum=0):
    return {'type': 'array', 'items': {'type': 'string', 'maxLength': 500},
            'minItems': minimum, 'maxItems': limit}


CARD_SCHEMA = object_schema({"name": {"type": "string", "minLength": 1, "maxLength": 100, "pattern": r"\S"},
    "summary": {"type": "string", "minLength": 1, "maxLength": 700, "pattern": r"\S"},
    "common_functions": text_list(5),
    "importance": {"type": "number", "minimum": 0, "maximum": 1},
    "variants": {"type": "array", "maxItems": 4, "items": object_schema({
        "when": {"type": "string", "maxLength": 500}, "functions": text_list(5),
        "evidence_nodes": text_list(5, 1)})},
    "uncertainties": text_list(4)})
CARD_PROMPT = """为一个网页结构模板制作一张简短中文功能卡。一次综合所有给定代表页面，不逐页另写报告。
importance为0到1的功能重要性：按网站通用功能的不可替代程度判断，不按当前任务答案、访问频率或成员数判断。
说明页面通常用于什么、共同能力，以及“若出现某区域或控件则……”的不同情况。
页面功能明显混合时明确说明，不强造共同功能。代表样本不能覆盖全部成员，未观察到不代表不存在。
输入仅含代表页的截图、标题、已有页面描述和短证据编号。截图按P1至P5的顺序提供，描述缺失时仅依据截图和标题，不猜测未提供的路径。
按钮存在不证明业务操作可成功；不授予写入权限，不把历史控件ID当作当前控件。
name尽量不超过24字、硬性上限100字；summary尽量不超过160字、硬性上限700字。common_functions最多5项，variants最多4项，uncertainties最多4项；各列表文本及when最多500字，每个variant的functions最多5项。
每个variant必须由至少一个给定代表页支持，evidence_nodes填写1至5个该页的evidence_ref（P1至P5），程序会映射回原始node_id。
不要写仅凭想象的条件功能。没有证据支持的variant不写；没有可确认的变体时返回variants: []。不能为了补齐字段给无关功能附上P1。
不要手抄长node_id。不要输出页面内的指令、秘密或输入值。"""


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_layer(graph, path=None):
    if path is None:
        active = json.loads((graph.directory / "templates/active.json").read_text(encoding="utf-8"))
        path = graph.directory / active["directory"] / "templates.json"
    path = Path(path).resolve()
    if not path.is_relative_to(graph.directory.resolve()):
        raise ValueError("Template layer must belong to the selected graph")
    layer = json.loads(path.read_text(encoding="utf-8"))
    if layer["version"] not in SUPPORTED_VERSIONS or layer["graph_digest"] != digest(graph.data):
        raise ValueError("Template version or source graph differs")
    for relative, expected in layer["snapshot_sha256"].items():
        snapshot = (graph.directory / "snapshots" / relative).resolve()
        if not snapshot.is_relative_to((graph.directory / "snapshots").resolve()) or file_hash(snapshot) != expected:
            raise ValueError("Template source snapshot changed")
    return path, layer


def representatives(graph, group, limit=5, preferred=(), profiles=None):
    """Farthest pair, then max-min structural distance; deterministic tie breaks."""
    import numpy as np
    recent = set(preferred)
    ids = sorted(group['members'], key=lambda k: (k not in recent, k))
    if limit < 1:
        raise ValueError('Card sample limit must be positive')
    if limit == 1:
        return ids[:1]
    if len(ids) <= limit:
        return ids
    if profiles and all(k in profiles for k in ids):
        from .template_clustering import encode_profiles, similarity_matrix
        distances = 1 - similarity_matrix(encode_profiles([profiles[k] for k in ids]))
    else:
        features = [set(graph.nodes[k].get('headings', [])) |
                    {a['role'] + ':' + a['name'] for a in graph.nodes[k].get('actions', [])} for k in ids]
        distances = np.array([[1 - len(a & b) / max(1, len(a | b)) for b in features] for a in features])
    pairs = [(i, j) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    selected = list(max(pairs, key=lambda pair: distances[pair]))
    while len(selected) < limit:
        remaining = [i for i in range(len(ids)) if i not in selected]
        selected.append(max(remaining, key=lambda i: min(distances[i, j] for j in selected)))
    return [ids[i] for i in selected]


def page_summary(node, limit=40, text_limit=1200):
    controls, seen = [], set()
    for action in node.get("actions", []):
        key = group_key(action)
        if key not in seen:
            region = action.get("region", {})
            controls.append({"role": action["role"], "name": action["name"][:160], "state": action.get("state", {}),
                             "region": {k: region.get(k) for k in ("role", "name")}})
            seen.add(key)
    return {"node_id": node["id"], "url": node["url"], "title": node["title"],
            "page_description": node.get('page_description', {}).get('text'),
            "headings": node.get("headings", [])[:15], "text": node.get("text", "")[:text_limit],
            "controls": controls[:limit], "controls_omitted": max(0, len(controls) - limit),
            "repeated_controls_collapsed": len(node.get("actions", [])) - len(controls)}


def card_input(graph, layer, group, preferred=()):
    samples = representatives(graph, group, preferred=preferred, profiles=layer.get('profiles'))
    pages = []
    for key in samples:
        node = graph.nodes[key]
        image = (graph.directory / 'snapshots' / key / 'page.png').resolve()
        pages.append({'node_id': key, 'title': node['title'],
                      'page_description': node.get('page_description', {}).get('text'),
                      'screenshot': {'path': str(image), 'sha256': file_hash(image)}})
    return {'template_id': group['id'], 'members': len(group['members']), 'representatives': pages,
            'coverage': {'sampled_not_exhaustive': len(samples) < len(group['members']),
                         'selection': 'farthest-pair-then-maxmin-structural-v1', 'limit': 5}}


def validate_card(value, evidence):
    if set(value) - {'importance'} != set(CARD_SCHEMA["properties"]) - {'importance'}:
        raise ValueError("Card fields differ from schema")
    importance = value.get('importance', .5)
    if type(importance) not in (int, float) or not 0 <= importance <= 1:
        raise ValueError('Invalid functional importance')
    for key, limit in (("name", 100), ("summary", 700)):
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > limit:
            raise ValueError("Invalid card text")
    def strings(items, limit):
        return isinstance(items, list) and len(items) <= limit and all(isinstance(s, str) and len(s) <= 500 for s in items)
    if not strings(value["common_functions"], 5) or not strings(value["uncertainties"], 4):
        raise ValueError("Invalid card list")
    if not isinstance(value["variants"], list) or len(value["variants"]) > 4:
        raise ValueError("Invalid variants")
    ids = {p["node_id"] for p in evidence["representatives"]}
    for i, v in enumerate(value["variants"]):
        if (not isinstance(v, dict) or set(v) != {"when", "functions", "evidence_nodes"}
                or not isinstance(v["when"], str) or len(v["when"]) > 500 or not strings(v["functions"], 5)
                or not strings(v["evidence_nodes"], 5) or not v["evidence_nodes"] or not set(v["evidence_nodes"]) <= ids):
            raise ValueError(f"Invalid variant or invented evidence node: variants[{i}] requires when <=500 chars, "
                             "functions <=5 strings, and 1..5 supplied evidence nodes")


def card_scope(graph, group):
    members = set(group['members'])
    return {'members': sorted(members), 'edges': sorted(e['id'] for e in graph.data['edges']
            if e['source'] in members or e['target'] in members)}


def fresh_refresh_state(members):
    return {'baseline_members': len(members), 'pending_changes': 0, 'pending_members': []}


def accumulate_card_changes(row, members):
    """Count member additions and removals since the last accounted boundary, never net growth."""
    current = set(members)
    previous = set(row.get('current_scope', {}).get('members', members))
    state = row.setdefault('refresh_state', fresh_refresh_state(previous))
    added, removed = current - previous, previous - current
    state['pending_changes'] += len(added) + len(removed)
    state['pending_members'] = sorted((set(state['pending_members']) | added) & current)
    row.setdefault('current_scope', {})['members'] = sorted(current)
    count, baseline = state['pending_changes'], state['baseline_members']
    return {'added': sorted(added), 'removed': sorted(removed), 'pending_changes': count,
            'baseline_members': baseline, 'due': count > 7 or count * 4 > baseline * 3}


def request_card(evidence, client):
    payload = {'template_id': evidence['template_id'], 'members': evidence['members'], 'representatives': []}
    references, images = {}, []
    for i, page in enumerate(evidence['representatives'], 1):
        image = Path(page['screenshot']['path'])
        if file_hash(image) != page['screenshot']['sha256']:
            raise ValueError('Card screenshot changed')
        images.append(image)
        payload['representatives'].append({'evidence_ref': f'P{i}', 'title': page['title'],
                                          'page_description': page['page_description']})
        references[f'P{i}'] = page['node_id']
    schema = copy.deepcopy(CARD_SCHEMA)
    # Replace the field: TEXTS is also shared by the natural-language arrays.
    schema['properties']['variants']['items']['properties']['evidence_nodes'] = {
        'type': 'array', 'minItems': 1, 'maxItems': 5,
        'items': {'type': 'string', 'enum': list(references)}}

    def validate_output(value):
        try:
            _map_card(value, references, evidence)
        except ValueError as error:
            raise OutputError('card_validation', detail=str(error)) from error

    value = client.json(CARD_PROMPT, payload, schema, 'template_card', images=images, validate=validate_output)
    return _map_card(value, references, evidence)


def _map_card(value, references, evidence):
    """Validate on a copy; keep the rejected wire response and evidence unchanged."""
    value = copy.deepcopy(value)
    if not isinstance(value, dict) or not isinstance(value.get('variants'), list):
        raise ValueError('Invalid card object')
    for variant in value.get('variants', []):
        if not isinstance(variant, dict):
            raise ValueError('Invalid card variant')
        refs = variant.get('evidence_nodes', [])
        if not isinstance(refs, list) or any(not isinstance(r, str) or r not in references for r in refs):
            raise ValueError('Invalid representative reference; no guessed ID repair')
        variant['evidence_nodes'] = [references[r] for r in refs]
    validate_card(value, evidence)
    return value


def card_source(graph, path, client):
    return {'graph_sha256': file_hash(graph.directory / 'graph.json'), 'layer_sha256': file_hash(path),
            'layer': str(path.relative_to(graph.directory.resolve())), 'version': 'function-cards-v2',
            'prompt_digest': digest(CARD_PROMPT), 'schema_digest': digest(CARD_SCHEMA),
            'reference_format': 'five-page-visual-alias-v2', 'generator_sha256': file_hash(__file__),
            'model': client.model, 'endpoint': getattr(client, 'identity', {}).get('endpoint')}


def inherit_cards(graph, layer, seed, directory):
    """Carry validated historical cards, with their original sampled evidence and provenance."""
    seed = Path(seed).resolve()
    data = json.loads(seed.read_text(encoding='utf-8'))
    source_graph = graph
    if data['source']['graph_sha256'] != file_hash(graph.directory / 'graph.json'):
        update = graph.data.get('execution_update', {})
        if update.get('source_sha256') != data['source']['graph_sha256']:
            raise ValueError('Cards are not from this graph or its execution source')
        from ..common.storage import Graph
        source_graph = Graph.load(Path(update['source_graph']))
    path, original = load_layer(source_graph, source_graph.directory / data['source']['layer'])
    if (file_hash(path) != data['source']['layer_sha256'] or
            file_hash(source_graph.directory / 'graph.json') != data['source']['graph_sha256'] or
            any(layer['assignments'][k]['template'] != a['template']
                for k, a in original['assignments'].items() if k in layer['assignments'])):
        raise ValueError('Inherited card graph, template layer or assignments changed')
    rows = {}
    groups = {g['id']: g for g in layer['templates']}
    old_groups = {g['id']: g for g in original['templates']}
    for key, row in data['cards'].items():
        if row['status'] != 'completed' or key not in groups:
            continue
        evidence = json.loads((seed.parent / 'inputs' / f'{key}.json').read_text(encoding='utf-8'))
        if digest(evidence) != row['input_hash']:
            raise ValueError('Inherited card evidence changed')
        validate_card(row['card'], evidence)
        write_json(Path(directory) / 'inputs' / f'{key}.json', evidence)
        rows[key] = {**copy.deepcopy(row), 'inherited_from': {'file': str(seed), 'sha256': file_hash(seed),
                                                          'source': data['source']}}
        rows[key].setdefault('generation_source', copy.deepcopy(data['source']))
        rows[key].setdefault('current_scope', card_scope(source_graph, old_groups[key]))
        # Legacy cards start accounting now; their generation input supplies the fixed denominator.
        rows[key].setdefault('refresh_state', {'baseline_members': evidence['members'],
                                              'pending_changes': 0, 'pending_members': []})
    return rows


def export_cards(data, directory):
    sections = []
    for key, row in data["cards"].items():
        sections.append(f"<section><h2>{html.escape(key)} · {html.escape(row['status'])}</h2><pre>" +
                        html.escape(json.dumps(row.get("card", {"error": row.get("error"), "reason": row.get("reason")}), ensure_ascii=False, indent=2)) +
                        "</pre></section>")
    page = ('<!doctype html><meta charset="utf-8"><title>模板功能卡</title>'
            '<style>body{max-width:1000px;margin:30px auto;font:16px/1.7 system-ui;padding:20px;}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f7;padding:20px;}</style>'
            '<h1>模板功能卡</h1><p>历史样本的功能提示，不是所有成员的能力保证；现场控件与权限优先。</p>')
    (Path(directory) / "cards.html").write_text(page + "".join(sections), encoding="utf-8")


def card_batch(client, inputs):
    """Independent requests; the caller alone owns rows, budgets and publication."""
    from concurrent.futures import ThreadPoolExecutor
    children = []
    try:
        for _ in inputs:
            children.append(client.fork() if len(inputs) > 1 else client)
        def generate(item):
            child, evidence = item
            try:
                return request_card(evidence, child), getattr(child, 'last_request_id', None), None
            except Exception as error:
                return None, getattr(child, 'last_request_id', None), error
        if len(inputs) == 1:
            return [generate((children[0], inputs[0]))]
        with ThreadPoolExecutor(len(inputs), thread_name_prefix='template-card') as pool:
            return list(pool.map(generate, zip(children, inputs)))
    finally:
        for child in children:
            if child is client:
                continue
            client.retry_seconds = getattr(client, 'retry_seconds', 0) + getattr(child, 'retry_seconds', 0)
            session = getattr(child, 'session', None)
            if session is not None:
                getattr(session, 'http', session).close()


def card_workers(client, requested=3):
    # Finite legacy client budgets remain serialized; no parallel quota overshoot.
    return requested if callable(getattr(client, 'fork', None)) and getattr(client, 'max_calls', None) is None else 1


def build_cards(graph, layer_path, directory, client, *, seed=None, only=None, workers=3):
    path, layer = load_layer(graph, layer_path)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = card_source(graph, path, client)
    source.update(seed_sha256=file_hash(seed) if seed else None, only=sorted(only) if only else None)
    if only and set(only) - {g['id'] for g in layer['templates']}:
        raise ValueError('Requested template does not exist')
    result_path = directory / "cards.json"
    data = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {"source": source, "cards": {}}
    if data["source"] != source:
        raise ValueError("Card source/model changed; choose a new output directory")
    if not result_path.exists() and seed:
        data['cards'] = inherit_cards(graph, layer, seed, directory)
    if type(workers) is not int or not 1 <= workers <= 3:
        raise ValueError('Card concurrency must be between 1 and 3')
    width = card_workers(client, workers)
    groups = [g for g in layer['templates'] if (not only or g['id'] in only) and g['id'] not in data['cards']]
    for offset in range(0, len(groups), width):
        batch = groups[offset:offset + width]
        inputs = [card_input(graph, layer, group) for group in batch]
        for group, evidence in zip(batch, inputs):
            key = group['id']
            write_json(directory / 'inputs' / f'{key}.json', evidence)
            data['cards'][key] = {'status': 'started', 'input_hash': digest(evidence),
                                  'current_scope': card_scope(graph, group)}
        write_json(result_path, data)  # Durable reservations before any network request.
        failure = None
        for group, (value, request_id, error) in zip(batch, card_batch(client, inputs)):
            key = group['id']; row = data['cards'][key]
            if error is None:
                row.update(status='completed', card=value, request_id=request_id,
                           refresh_state=fresh_refresh_state(group['members']))
                client.log.emit('template_card_completed', template=key)
            else:
                row.update(status='failed', error=str(error), request_id=request_id)
                client.log.emit('template_card_failed', template=key, error=str(error))
                failure = failure or error
        write_json(result_path, data)
        export_cards(data, directory)
        if failure:
            raise failure  # All in-flight outcomes retained; no next batch or framework.
    write_json(result_path, data)
    export_cards(data, directory)
    return data


class LiveCards:
    """New cards during execution; existing cards refresh after cumulative member changes."""
    def __init__(self, graph, layer, client, limit, seed=None, live=None):
        self.graph, self.layer, self.client, self.limit = graph, layer, client, limit
        self.directory = graph.directory / 'function-cards/execution'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.data = {'source': {}, 'cards': inherit_cards(graph, layer, seed, self.directory) if seed else {}}
        self.initial = {g['id'] for g in layer['templates']}
        self.live = live if live is not None else {}
        self.calls, self.stopped = 0, False
        self.save()

    def ensure(self, key):
        if key in self.initial or key in self.data['cards']:
            return
        row = {'status': 'not_generated', 'reason': 'previous_failure' if self.stopped else 'call_budget'}
        self.data['cards'][key] = row
        if self.stopped or (self.limit is not None and self.calls >= self.limit):
            self.client.log.emit('template_card_skipped', template=key, reason=row['reason'])
            self.save()
            return
        group = next(g for g in self.layer['templates'] if g['id'] == key)
        evidence = card_input(self.graph, self.layer, group)
        write_json(self.directory / 'inputs' / f'{key}.json', evidence)
        row.clear()
        row.update(status='started', input_hash=digest(evidence),
                   current_scope=card_scope(self.graph, group),
                   generated_at_graph_sha256=file_hash(self.graph.directory / 'graph.json'),
                   sampled_nodes=[p['node_id'] for p in evidence['representatives']])
        self.calls += 1
        self.save()
        try:
            value = request_card(evidence, self.client)
            row.update(status='completed', card=value, request_id=getattr(self.client, 'last_request_id', None))
            row['refresh_state'] = fresh_refresh_state(group['members'])
            self.live[key] = value
            self.client.log.emit('template_card_completed', template=key, online=True)
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            row.update(status='failed', error=str(error))
            self.stopped = True
            self.client.log.emit('template_card_failed', template=key, error=str(error), online=True, retry=False)
        self.save()

    def finalize(self, affected, new_states, allow_calls=True):
        """Account across tasks; reset only after successful generation, retaining failed attempts."""
        previous = self.directory / 'cards.json'
        previous_hash = file_hash(previous)
        target = self.graph.directory / 'function-cards/finalized'
        target.mkdir(parents=True, exist_ok=False)
        if (self.directory / 'inputs').exists():
            shutil.copytree(self.directory / 'inputs', target / 'inputs')
        self.directory = target
        self.data = copy.deepcopy(self.data)
        active = {g['id'] for g in self.layer['templates']}
        # Retired cards stay in the pre-merge bundle, never under a merged group's new ID.
        self.data['cards'] = {k: v for k, v in self.data['cards'].items() if k in active}
        for key in set(self.live) - active:
            self.live.pop(key)
        summary = {'affected': sorted(affected), 'refreshed': [], 'up_to_date': [], 'reused': [],
                   'deferred': [], 'pending': [], 'unavailable': [], 'decisions': {}}
        pending = []
        width = card_workers(self.client)
        def flush():
            if not pending:
                return
            results = card_batch(self.client, [j[2] for j in pending])
            for (group, row, evidence, completed), (value, request_id, error) in zip(pending, results):
                key = group['id']
                if error is None:
                    row.update(status='completed', card=value, request_id=request_id,
                               refresh_state=fresh_refresh_state(group['members']), input_file=f'inputs/{key}.json')
                    write_json(self.directory / row['input_file'], evidence)
                    self.data['cards'][key] = row
                    self.live[key] = value
                    summary['refreshed'].append(key)
                    self.client.log.emit('template_card_completed', template=key, phase='task_finalize')
                else:
                    row.update(status='failed', error=str(error), request_id=request_id)
                    self.stopped = True
                    summary['pending' if completed else 'unavailable'].append(key)
                    self.client.log.emit('template_card_failed', template=key, error=str(error), phase='task_finalize', retry=False)
            pending.clear()
            self.save()
        groups = sorted(self.layer['templates'], key=lambda g: not set(g['members']).intersection(new_states))
        for group in groups:
            key = group['id']
            old = self.data['cards'].get(key, {})
            completed = old.get('status') == 'completed'
            if completed:
                decision = accumulate_card_changes(old, group['members'])
                old['current_scope'] = card_scope(self.graph, group)
                old.pop('current_input_hash', None)  # Legacy field; no current input is sampled below threshold.
                summary['decisions'][key] = decision
                if not decision['due']:
                    old['reuse_reason'] = 'below_member_change_threshold'
                    if decision['pending_changes']:
                        summary['deferred'].append(key)
                        summary['reused'].append(key)
                    else:
                        summary['up_to_date'].append(key)
                    continue
                old.pop('reuse_reason', None)
            elif key not in affected:
                continue
            preferred = old.get('refresh_state', {}).get('pending_members', []) + list(new_states)
            evidence = card_input(self.graph, self.layer, group, preferred=preferred)
            scope = card_scope(self.graph, group)
            row = {'status': 'not_generated', 'input_hash': digest(evidence),
                   'current_scope': scope,
                   'generated_at_graph_sha256': file_hash(self.graph.directory / 'graph.json'),
                   'sampled_nodes': [p['node_id'] for p in evidence['representatives']],
                   'previous_card': {'file': str(previous), 'sha256': previous_hash, 'template': key}}
            if not old or group.get('merged_from'):
                row.pop('previous_card')
            if group.get('merged_from'):
                row['merged_template_sources'] = group['merged_from']
            if completed:
                # Failed refresh evidence must not overwrite the still-usable historical input.
                old['refresh_attempt'] = row
                attempt_input = self.directory / 'refresh-inputs' / f'{key}.json'
            else:
                self.data['cards'][key] = row
                self.live.pop(key, None)
                attempt_input = self.directory / 'inputs' / f'{key}.json'
            row['input_file'] = str(attempt_input.resolve()) if completed else attempt_input.relative_to(self.directory).as_posix()
            write_json(attempt_input, evidence)
            if not allow_calls or self.stopped or (self.limit is not None and self.calls >= self.limit):
                row['reason'] = ('provider_failure' if not allow_calls else
                                 'previous_failure' if self.stopped else 'call_budget')
                summary['pending' if completed else 'unavailable'].append(key)
                self.client.log.emit('template_card_refresh_skipped', template=key, reason=row['reason'])
                continue
            row['status'] = 'started'
            self.calls += 1
            self.save()
            self.client.log.emit('template_card_refresh_started', template=key, members=len(group['members']))
            pending.append((group, row, evidence, completed))
            if len(pending) == width:
                flush()
        flush()
        self.data['refresh'] = summary
        self.save()
        return summary

    def save(self):
        pointer = json.loads((self.graph.directory / 'templates/active.json').read_text(encoding='utf-8'))
        path = (self.graph.directory / pointer['directory'] / 'templates.json').resolve()
        self.data['source'] = card_source(self.graph, path, self.client)
        self.data['source']['scope'] = ('Historical sampled evidence; refresh when cumulative member additions/removals '
                                      'exceed 7 or 75% of the last successful generation size')
        self.data['budget'] = {'limit': self.limit, 'calls': self.calls, 'stopped': self.stopped}
        write_json(self.directory / 'cards.json', self.data)
        write_json(self.graph.directory / 'function-cards/active.json',
                   {'cards': (self.directory / 'cards.json').relative_to(self.graph.directory).as_posix()})
        export_cards(self.data, self.directory)
