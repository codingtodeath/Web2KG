from types import SimpleNamespace
from unittest.mock import Mock

from test_function_cards import saved, CardModel
from web2kg.common.storage import Graph, write_json
from web2kg.stage1.site_framework import framework_state, load_framework, update_framework


def setup_summary(tmp_path):
    graph = Graph(tmp_path)
    layer = {'templates': [{'id': 'T001', 'members': ['a']}, {'id': 'T002', 'members': ['b']}],
             'transitions': []}
    cards = {'source': {'model': 'fixture'}, 'cards': {k: {'status': 'completed', 'card': {
        'name': k, 'summary': 'Original function', 'common_functions': [], 'variants': [], 'uncertainties': []}}
        for k in ('T001', 'T002')}}
    framework = {'overview': 'Original site', 'areas': [
        {'name': k, 'purpose': 'Browse', 'templates': [k]} for k in ('T001', 'T002')],
        'navigation': [], 'limitations': []}
    write_json(tmp_path / 'templates/active.json', {'directory': 'templates/current'})
    write_json(tmp_path / 'templates/current/templates.json', layer)
    data = {'version': 1, 'status': 'completed', 'source': {'graph_sha256': 'historical'},
            'framework': framework, 'maintenance': {
                'pending_updates': 0, 'template_state': framework_state(layer, cards)}}
    write_json(tmp_path / 'initial-website-framework.json', data)
    return graph, layer, cards, data


def test_local_load_projects_merges_and_retirements_without_ancestry_or_hashing(tmp_path, monkeypatch):
    from web2kg.stage1 import site_framework
    graph, layer, _, original = setup_summary(tmp_path)
    graph.data['execution_update'] = {'source_graph': 'missing/ancestor', 'source_sha256': 'old'}
    layer['templates'] = [{'id': 'T003', 'members': ['a']}]
    layer['retired_templates'] = {'T001': 'T003', 'T002': None}
    write_json(tmp_path / 'templates/current/templates.json', layer)
    monkeypatch.setattr(Graph, 'load', Mock(side_effect=AssertionError('No ancestor reads')))
    monkeypatch.setattr(site_framework, 'file_hash', Mock(side_effect=AssertionError('No hash scans')))
    monkeypatch.setattr(site_framework, 'load_layer', Mock(side_effect=AssertionError('No snapshot scans')))
    result = load_framework(graph)
    assert result['framework']['areas'] == [{'name': 'T001', 'purpose': 'Browse', 'templates': ['T003']}]
    assert result['source'] == original['source']
    assert result['historical_framework'] == original['framework']


def test_refresh_counts_functional_updates_across_tasks_not_reads_or_member_counts(tmp_path):
    graph, layer, cards, _ = setup_summary(tmp_path)
    answer = {'overview': 'Updated site', 'areas': [
        {'name': 'Remaining', 'purpose': 'Browse', 'templates': ['T001']}], 'navigation': [], 'limitations': []}
    client = SimpleNamespace(model='fixture', json=Mock(return_value=answer))
    layer['templates'][0]['members'] += ['another-page']
    assert update_framework(graph, layer, cards, client, refresh_after=2)['pending_updates'] == 0
    cards['cards']['T001']['card']['summary'] = 'A new function'
    assert update_framework(graph, layer, cards, client, refresh_after=2)['pending_updates'] == 1
    assert update_framework(graph, layer, cards, client, refresh_after=2)['pending_updates'] == 1
    client.json.assert_not_called()
    # Simulate the next task reading the persisted framework, with one template retired.
    layer['templates'].pop()
    result = update_framework(graph, layer, cards, client, refresh_after=2)
    assert result['status'] == 'refreshed' and result['pending_updates'] == 0
    assert client.json.call_count == 1
    payload = client.json.call_args.args[1]
    assert [t['id'] for t in payload['templates']] == ['T001']
    loaded = load_framework(graph, layer=layer)
    assert loaded['framework']['overview'] == 'Updated site'
    assert loaded['maintenance']['pending_updates'] == 0
    assert update_framework(graph, layer, cards, client, refresh_after=2)['calls'] == 0


def test_deferred_or_failed_refresh_retains_projection_and_pending_count(tmp_path):
    graph, layer, cards, original = setup_summary(tmp_path)
    cards['cards']['T001']['card']['summary'] = 'Changed function'
    client = SimpleNamespace(model='fixture', json=Mock(side_effect=RuntimeError('offline service failure')))
    result = update_framework(graph, layer, cards, client, refresh_after=1, allow_calls=False)
    assert result['status'] == 'deferred' and result['pending_updates'] == 1
    client.json.assert_not_called()
    result = update_framework(graph, layer, cards, client, refresh_after=1)
    assert result['status'] == 'refresh_failed' and result['calls'] == 1
    loaded = load_framework(graph, layer=layer)
    assert loaded['framework']['overview'] == original['framework']['overview']
    assert loaded['maintenance']['pending_updates'] == 1


def test_disabling_generation_still_removes_dead_template_references(tmp_path):
    graph, layer, cards, _ = setup_summary(tmp_path)
    layer['templates'].pop()
    client = SimpleNamespace(model='fixture', json=Mock())
    update_framework(graph, layer, cards, client, refresh_after=0)
    client.json.assert_not_called()
    assert all(a['templates'] == ['T001'] for a in load_framework(graph, layer=layer)['framework']['areas'])


def test_task_finalization_refreshes_framework_with_shared_call_budget(saved, tmp_path):
    from web2kg.common.storage import EventLog
    from web2kg.stage2.graph_update import ExecutionGraph
    from web2kg.stage2.template_context import TemplateKnowledge
    base, _, _, cards_path, _ = saved
    log = EventLog(tmp_path / 'run')
    knowledge = TemplateKnowledge(base, cards_path, log)
    model = CardModel(tmp_path / 'calls')
    original_json = model.json
    purposes = []
    def generate(system, payload, schema, purpose):
        purposes.append(purpose)
        if purpose == 'site_framework':
            return {'overview': 'Refreshed', 'areas': [], 'navigation': [], 'limitations': []}
        return original_json(system, payload, schema, purpose)
    model.json = generate
    updater = ExecutionGraph(base, None, log, knowledge, card_client=model, max_cards=1,
                             framework_refresh_updates=1, template_clustering='off')
    write_json(updater.graph.directory / 'initial-website-framework.json', {
        'status': 'completed', 'source': {}, 'framework': {
            'overview': 'Old', 'areas': [], 'navigation': [], 'limitations': []},
        'maintenance': {'pending_updates': 1, 'template_state': framework_state(updater.layer, updater.live_cards.data)}})
    result = updater.finish()
    assert result['framework_refresh']['status'] == 'refreshed'
    assert purposes == ['site_framework']
    assert result['cards']['calls'] == 1
    assert updater.live_cards.data['framework_calls'] == 1
    assert load_framework(updater.graph)['framework']['overview'] == 'Refreshed'
    assert updater.finish() == result and len(purposes) == 1
