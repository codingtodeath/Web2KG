import copy
import json
import yaml
from pathlib import Path
from unittest.mock import Mock

import pytest

from web2kg.common.browser import Browser
from web2kg.common.responses import ResponsesClient, ResponsesError
from web2kg.common.storage import Graph, EventLog, write_json
from web2kg.stage1.function_cards import build_cards, card_input, file_hash, validate_card, representatives
from web2kg.stage1.indexing import build_index
from web2kg.stage1.templates import build_templates
from web2kg.stage2.agent import run_task
from web2kg.stage2.template_context import TemplateKnowledge


class CardModel:
    model = 'fixture-luna'

    def __init__(self, directory):
        self.log, self.calls = EventLog(directory), []

    def json(self, system, payload, schema, purpose, images=(), *, validate=None):
        self.calls.append(payload)
        return {'name': '内容页面', 'summary': '根据现场控件浏览。', 'common_functions': ['查看内容'],
                'variants': [{'when': '出现相应控件时', 'functions': ['进入下一级'],
                              'evidence_nodes': [payload['representatives'][0]['evidence_ref']]}],
                'uncertainties': ['样本不是全部页面。']}


@pytest.fixture(scope='module')
def saved(site, tmp_path_factory):
    root = tmp_path_factory.mktemp('function-cards')
    graph = Graph(root / 'graph')
    with Browser(site, EventLog(graph.directory), identity_mode='interaction', settle_ms=20) as browser:
        first = browser.observe(graph.directory)
        graph.data.update(start_url=site, root=first['id'], identity_mode='interaction')
        graph.add_node(first, [])
        a = next(a for a in first['actions'] if a['name'] == 'Catalog')
        browser.act(a)
        catalog = browser.observe(graph.directory)
        graph.add_node(catalog, [a]); graph.add_edge(first['id'], catalog['id'], a)
        b = next(a for a in catalog['actions'] if a['name'] == 'Red mug')
        browser.act(b)
        product = browser.observe(graph.directory)
        graph.add_node(product, [a, b]); graph.add_edge(catalog['id'], product['id'], b)
    graph.save()
    layer = build_templates(graph, weights=(.5, 1.5, 1))
    path = graph.directory / 'templates/test/templates.json'
    write_json(path, layer)
    model = CardModel(root / 'cards')
    data = build_cards(graph, path, root / 'cards', model)
    build_index(graph)
    return graph, path, data, root / 'cards/cards.json', (first, catalog, product)


def test_one_card_per_template_bounded_samples_and_safe_resume(saved, tmp_path):
    graph, path, _, _, _ = saved
    before = file_hash(graph.directory / 'graph.json')
    model = CardModel(tmp_path)
    data = build_cards(graph, path, tmp_path, model)
    assert len(model.calls) == len(data['cards'])
    assert all(1 <= len(p['representatives']) <= 5 for p in model.calls)
    assert all('parent_templates' not in p and 'incoming' not in p for p in model.calls)
    build_cards(graph, path, tmp_path, model)
    assert len(model.calls) == len(data['cards'])
    assert file_hash(graph.directory / 'graph.json') == before
    assert (tmp_path / 'cards.html').is_file()


def test_failed_card_is_retained_and_not_retried(saved, tmp_path):
    graph, path, _, _, _ = saved
    model = CardModel(tmp_path)
    model.json = Mock(side_effect=ResponsesError('HTTP 503'))
    with pytest.raises(ResponsesError):
        build_cards(graph, path, tmp_path, model)
    data = json.loads((tmp_path / 'cards.json').read_text())
    failed = next(iter(data['cards']))
    assert data['cards'][failed]['status'] == 'failed'
    healthy = CardModel(tmp_path)
    completed = build_cards(graph, path, tmp_path, healthy)
    assert failed not in [p['template_id'] for p in healthy.calls]
    assert completed['cards'][failed]['status'] == 'failed'


def test_invented_evidence_rejected(saved):
    graph, path, data, _, _ = saved
    layer = json.loads(path.read_text())
    evidence = card_input(graph, layer, layer['templates'][0])
    card = copy.deepcopy(next(iter(data['cards'].values()))['card'])
    card['variants'][0]['evidence_nodes'] = ['invented']
    with pytest.raises(ValueError, match='evidence'):
        validate_card(card, evidence)


def test_refresh_samples_new_member_without_incoming_edge_payload(saved, tmp_path):
    base, path, _, _, (_, _, product) = saved
    graph = Graph(tmp_path)
    graph.data = copy.deepcopy(base.data)
    layer = json.loads(path.read_text())
    assignment = layer['assignments'][product['id']]
    group = next(g for g in layer['templates'] if g['id'] == assignment['template'])
    fresh = {**graph.nodes[product['id']], 'id': 'new-member', 'url': 'http://test/new'}
    graph.add_node(fresh, [])
    layer['assignments']['new-member'] = copy.deepcopy(assignment)
    group['members'].append('new-member')
    old_edge = next(e for e in graph.data['edges'] if e['target'] == product['id'])
    graph.data['edges'] += [{**old_edge, 'id': f'historical-{i}'} for i in range(15)]
    graph.data['edges'].append({**old_edge, 'id': 'new-transition', 'target': 'new-member'})
    import shutil
    shutil.copytree(base.directory / 'snapshots', graph.directory / 'snapshots')
    shutil.copytree(base.directory / 'snapshots' / product['id'], graph.directory / 'snapshots/new-member')
    evidence = card_input(graph, layer, group, preferred=['new-member'])
    assert 'new-member' in [r['node_id'] for r in evidence['representatives']]
    assert 'incoming' not in evidence and 'parent_templates' not in evidence


def test_short_references_are_exactly_mapped_without_guessing(saved, tmp_path):
    from web2kg.stage1.function_cards import request_card
    graph, path, _, _, _ = saved
    layer = json.loads(path.read_text())
    evidence = card_input(graph, layer, layer['templates'][0])
    model = CardModel(tmp_path)
    value = request_card(evidence, model)
    assert value['variants'][0]['evidence_nodes'] == [evidence['representatives'][0]['node_id']]
    assert 'evidence_ref' not in evidence['representatives'][0]
    bad = copy.deepcopy(value)
    bad['variants'][0]['evidence_nodes'] = [evidence['representatives'][0]['node_id'][:-2]]
    model.json = Mock(return_value=bad)
    with pytest.raises(ValueError, match='no guessed ID repair'):
        request_card(evidence, model)
    assert model.json.call_count == 1


def test_new_revision_retries_only_missing_card_and_keeps_old_failure(saved, tmp_path):
    graph, path, data, _, _ = saved
    old = tmp_path / 'old'; new = tmp_path / 'new'
    model = CardModel(old)
    built = build_cards(graph, path, old, model)
    key = next(iter(built['cards']))
    built['cards'][key] = {'status': 'failed', 'error': 'truncated reference'}
    write_json(old / 'cards.json', built)
    previous = file_hash(old / 'cards.json')
    healthy = CardModel(new)
    repaired = build_cards(graph, path, new, healthy, seed=old / 'cards.json', only=[key])
    assert len(healthy.calls) == 1 and healthy.calls[0]['template_id'] == key
    assert all(r['status'] == 'completed' for r in repaired['cards'].values())
    assert file_hash(old / 'cards.json') == previous
    TemplateKnowledge(graph, new / 'cards.json', EventLog(new))
    build_cards(graph, path, new, healthy, seed=old / 'cards.json', only=[key])
    assert len(healthy.calls) == 1


def test_reference_enum_does_not_constrain_function_text(saved, tmp_path):
    from web2kg.stage1.function_cards import CARD_SCHEMA, request_card
    graph, path, _, _, _ = saved
    layer = json.loads(path.read_text())
    evidence = card_input(graph, layer, layer['templates'][0])
    original = copy.deepcopy(CARD_SCHEMA)
    model = CardModel(tmp_path)
    generate = model.json
    def check_schema(system, payload, schema, purpose, images=(), *, validate=None):
        props = schema['properties']
        variant = props['variants']['items']['properties']
        assert variant['evidence_nodes']['items']['enum'] == [p['evidence_ref'] for p in payload['representatives']]
        for field in (props['common_functions'], props['uncertainties'], variant['functions']):
            assert field['items'] == {'type': 'string', 'maxLength': 500}
            assert 'enum' not in field['items']
        return generate(system, payload, schema, purpose)
    model.json = check_schema
    result = request_card(evidence, model)
    assert result['common_functions'] == ['查看内容']
    assert result['uncertainties'] == ['样本不是全部页面。']
    assert CARD_SCHEMA == original


def test_cards_escape_untrusted_markup_and_reject_stale_source(saved, tmp_path):
    from web2kg.stage1.function_cards import export_cards
    graph, _, data, _, _ = saved
    changed = copy.deepcopy(data)
    next(iter(changed['cards'].values()))['card']['summary'] = '<script>alert(1)</script>'
    export_cards(changed, tmp_path)
    assert '<script>' not in (tmp_path / 'cards.html').read_text(encoding='utf-8')
    changed['source']['layer_sha256'] = 'stale'
    write_json(tmp_path / 'cards.json', changed)
    with pytest.raises(ValueError, match='stale'):
        TemplateKnowledge(graph, tmp_path / 'cards.json', EventLog(tmp_path))


def test_live_unseen_entity_matches_structurally_and_unknown_remains_unknown(saved, site, tmp_path):
    graph, _, _, cards, (first, catalog, product) = saved
    knowledge = TemplateKnowledge(graph, cards, EventLog(tmp_path))
    with Browser(site, knowledge.log, identity_mode='interaction', settle_ms=20) as browser:
        assert knowledge.observe(browser.observe(), 'Find green mug')['match']['template_id']
        a = next(a for a in browser.observe()['actions'] if a['name'] == 'Catalog')
        browser.act(a); knowledge.after_action(a)
        context = knowledge.observe(browser.observe(), 'Find green mug')
        current_ids = {a['ax_node_id'] for a in browser.observe()['actions']}
        assert len(context['nearby']) <= 6
        assert all(set(h['current_control_ids']) <= current_ids for h in context['nearby'])
        b = next(a for a in browser.observe()['actions'] if a['name'] == 'Green mug')
        browser.act(b); knowledge.after_action(b)
        live = browser.observe()
        assert live['id'] not in graph.nodes
        result = knowledge.observe(live, 'Find green mug')
        assert result['match']['reason'] == 'weighted_match'
        assert result['match']['template_id'] == knowledge.layer['assignments'][product['id']]['template']
        browser.page.set_content('<main><table><tr><td>Unseen report</td></tr></table><input type="range"></main>')
        knowledge.after_action({'kind': 'back'})
        unknown = knowledge.observe(browser.observe(), 'Find an unknown report')
        assert unknown['match']['template_id'] is None
        assert unknown['function_card'] is None and unknown['nearby'] == []


def test_empty_structure_is_unknown_then_recovers(saved, tmp_path):
    graph, _, _, cards, (first, _, _) = saved
    knowledge = TemplateKnowledge(graph, cards, EventLog(tmp_path))
    empty = {'id': 'browser-error', 'url': 'chrome-error://chromewebdata/', 'ax': [],
             'dom': {'nodeType': 9, 'nodeName': '#document'}, 'actions': []}
    knowledge.observe(first, 'Inspect')
    for _ in range(2):
        result = knowledge.observe(empty, 'Inspect')
        assert result['match']['reason'] == 'no_visible_structure'
        assert result['match']['template_id'] is None
        assert result['function_card'] is None and result['nearby'] == []
    assert knowledge.observe(first, 'Inspect')['match']['template_id']


def test_page_gate_live_without_entry_and_legacy_card_loading(saved, tmp_path):
    from web2kg.stage1.templates import LEGACY_RULE, LEGACY_VERSION
    graph, path, data, cards, (_, _, product) = saved
    knowledge = TemplateKnowledge(graph, cards, EventLog(tmp_path))
    observed = {**product, 'id': 'unseen-product', 'url': product['url'] + '?new=1'}
    result = knowledge.match(observed)
    assert result['template_id'] == knowledge.layer['assignments'][product['id']]['template']
    assert result['scores'] == dict(parent=0, incoming=0, page=1)
    assert result['eligible'] and result['score'] == 1
    assert result['ranking_basis'] == 'page_only'
    # Saved old cards keep their exact old rule instead of silently using v6.
    old_layer = copy.deepcopy(knowledge.layer)
    old_layer['version'] = LEGACY_VERSION
    old_layer['parameters'].update(match_rule=LEGACY_RULE, threshold=.82)
    old_layer['parameters']['page_features'] = dict(structure=.4, combinations=.4, controls=.2)
    old_path = graph.directory / 'templates/legacy-test/templates.json'
    write_json(old_path, old_layer)
    old_cards = copy.deepcopy(data)
    old_cards['source'].update(layer=old_path.relative_to(graph.directory).as_posix(), layer_sha256=file_hash(old_path))
    write_json(tmp_path / 'legacy-cards.json', old_cards)
    legacy = TemplateKnowledge(graph, tmp_path / 'legacy-cards.json', EventLog(tmp_path / 'legacy'))
    assert legacy.match(observed)['template_id'] is None


@pytest.mark.parametrize('version', ['context-structure-v5', 'context-structure-v6', 'context-structure-v7'])
def test_live_matching_uses_loaded_page_feature_weights(saved, tmp_path, monkeypatch, version):
    import web2kg.stage2.template_context as matching
    from web2kg.stage1.page_structure import LEGACY_PAGE_FEATURES, PAGE_FEATURES
    graph, path, data, _, (_, catalog, product) = saved
    layer = json.loads(path.read_text())
    layer['version'] = version
    expected = dict(PAGE_FEATURES if version.endswith('v7') else LEGACY_PAGE_FEATURES)
    layer['parameters']['page_features'] = expected
    saved_path = graph.directory / f'templates/{version}-weights/templates.json'
    write_json(saved_path, layer)
    cards = copy.deepcopy(data)
    cards['source'].update(layer=saved_path.relative_to(graph.directory).as_posix(), layer_sha256=file_hash(saved_path))
    write_json(tmp_path / 'cards.json', cards)
    knowledge = TemplateKnowledge(graph, tmp_path / 'cards.json', EventLog(tmp_path))
    knowledge.previous = catalog
    knowledge.previous_match = {'template_id': layer['assignments'][catalog['id']]['template']}
    calls = []
    original = matching.page_similarity
    def compare(a, b, weights=None):
        calls.append(weights)
        return original(a, b, weights)
    monkeypatch.setattr(matching, 'page_similarity', compare)
    result = knowledge.match({**product, 'id': 'new-state'})
    assert result['template_id']
    assert len(calls) >= 2 and all(w == expected for w in calls)


def test_live_skips_ineligible_top_score_and_matches_next_candidate(saved, tmp_path, monkeypatch):
    import web2kg.stage2.template_context as matching
    graph, _, _, cards, (_, catalog, product) = saved
    knowledge = TemplateKnowledge(graph, cards, EventLog(tmp_path))
    knowledge.layer['version'] = 'context-structure-v7'  # Regression for the saved legacy ranking rule.
    assignments, profiles = knowledge.layer['assignments'], knowledge.layer['profiles']
    knowledge.previous = catalog
    knowledge.previous_match = {'template_id': assignments[catalog['id']]['template']}
    current = copy.deepcopy(profiles[product['id']])
    monkeypatch.setattr(knowledge, 'profile', lambda _: current)
    # Catalog has high context but page below threshold; product passes with no context.
    def similarity(a, b, weights=None):
        if a is current:
            return .84 if b is profiles[catalog['id']] else .9
        return 1. if b is profiles[graph.data['root']] else 0.
    monkeypatch.setattr(matching, 'page_similarity', similarity)
    monkeypatch.setattr(matching, 'control_comparison', lambda a, b: {'score': 1 if b is assignments[catalog['id']]['incoming'] else 0})
    result = knowledge.match({**product, 'id': 'changed-product'})
    assert result['closest'][0]['eligible'] is False
    assert result['template_id'] == assignments[product['id']]['template']
    assert result['score'] < result['closest'][0]['score']


def test_original_agent_receives_live_cards_and_selects_page(saved, site, tmp_path):
    graph, _, _, cards, (_, catalog, _) = saved
    log = EventLog(tmp_path)
    knowledge = TemplateKnowledge(graph, cards, log)
    calls = []
    class Client:
        def call(self, messages, system_prompt='', **kwargs):
            if 'target_page' in system_prompt:
                assert not any(m['content'].startswith('Historical template context') for m in messages
                               if isinstance(m['content'], str))
                return json.dumps({'target_page': catalog['id'] + '.png'})
            template_messages = [m for m in messages if isinstance(m['content'], str)
                                 and m['content'].startswith('Historical template context')]
            assert len(template_messages) == 1
            context = yaml.safe_load(template_messages[0]['content'].split('\n', 1)[1])
            assert context['template_id']
            assert 'match' not in context
            assert context['function_card']
            assert all(str(i).isdigit() for row in context['nearby'] for i in row['current_control_ids'])
            calls.append(context)
            return ('REASON: Catalog.\nACTION: optimal_navigate [Catalog] [reddit] [Catalog]' if len(calls) == 1
                    else 'REASON: Arrived.\nACTION: stop [done]')
    result = run_task(graph, 'Catalog', log, Client(), None, knowledge=knowledge, start_url=site)
    assert result['status'] == 'agent_stopped', result
    assert result['navigation']['success']
    assert len(calls) == 2
    assert calls[0]['template_id'] != calls[1]['template_id']


def test_replay_mismatch_keeps_actual_parent_and_incoming_context(saved, site, tmp_path):
    from web2kg.common.replay import execute_path
    graph, _, _, cards, (first, catalog, product) = saved
    log = EventLog(tmp_path)
    knowledge = TemplateKnowledge(graph, cards, log)
    path = copy.deepcopy(graph.path(first['id'], product['id']))
    path[-1]['action'] = next(a for a in catalog['actions'] if a['name'] == 'Green mug')
    with Browser(site, log, identity_mode='interaction', settle_ms=20) as browser:
        result = execute_path(browser, graph, path, log, on_action=knowledge.replay_action)
        assert not result['success'] and 'State mismatch' in result['error']
        context = knowledge.observe(browser.observe(), 'Find green mug')
        assert context['match']['reason'] == 'weighted_match'
        assert context['match']['template_id'] == knowledge.layer['assignments'][product['id']]['template']
        assert context['function_card']


def test_agent_omits_template_message_after_unmatched_page(saved, site, tmp_path, monkeypatch):
    import re
    from web2kg.stage2.prompt_context import TEMPLATE_HEADER
    graph, _, _, cards, _ = saved
    log = EventLog(tmp_path)
    knowledge = TemplateKnowledge(graph, cards, log)
    original_match = knowledge.match
    monkeypatch.setattr(knowledge, 'match', lambda observation:
        {'template_id': None, 'reason': 'fixture_unknown_page', 'score': None}
        if observation['url'].endswith('/item/3') else original_match(observation))
    calls = []
    class Client:
        def call(self, messages, **kwargs):
            contexts = [m['content'] for m in messages if isinstance(m['content'], str)
                        and m['content'].startswith(TEMPLATE_HEADER)]
            calls.append(contexts)
            if len(calls) == 1:
                assert len(contexts) == 1
                match = re.search(r"link \[(\d+)\] 'Green mug'", json.dumps(messages))
                assert match
                return f'REASON: Inspect a different page.\nACTION: click [{match[1]}]'
            assert contexts == []
            return 'REASON: Observed the new page.\nACTION: stop [done]'
    result = run_task(graph, 'Inspect Green mug', log, Client(), None, knowledge=knowledge,
                      start_url=site + '/catalog')
    assert result['status'] == 'agent_stopped' and len(calls) == 2, result
    evidence = [json.loads(line) for line in log.path.read_text(encoding='utf-8').splitlines()]
    raw_contexts = [e['context'] for e in evidence if e['event'] == 'template_context']
    assert raw_contexts[-1]['match']['template_id'] is None


def test_responses_model_override_images_and_budget(tmp_path):
    env = tmp_path / 'test.env'
    env.write_text('CRAWL_API_KEY=fixture\nCRAWL_BASE_URL=https://test.invalid/v1\nCRAWL_MODEL=luna\n')
    client = ResponsesClient(env, EventLog(tmp_path), 1, 200, model='terra')
    response = Mock(ok=True)
    response.json.return_value = {'status': 'completed', 'model': 'terra', 'usage': {'input_tokens': 5},
        'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"ok":true}'}]}]}
    client.session.post = Mock(return_value=response)
    png = tmp_path / 'page.png'; png.write_bytes(b'fixture-png')
    assert client.json('test', {}, {'type': 'object'}, 'test', images=[png]) == {'ok': True}
    request = client.session.post.call_args.kwargs['json']
    assert request['model'] == 'terra'
    assert request['input'][0]['content'][1]['image_url'].startswith('data:image/png;base64,')
    stored = json.loads(next((tmp_path / 'llm').glob('*/request.json')).read_text())
    assert 'data:image' not in json.dumps(stored) and stored['local_images'][0]['sha256']
    assert 'CRAWL_MODEL=luna' in env.read_text()
    with pytest.raises(ResponsesError, match='budget'):
        client.json('test', {}, {}, 'test')
    assert client.session.post.call_count == 1
