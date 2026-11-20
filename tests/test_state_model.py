import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from web2kg.common.browser import Browser
from web2kg.common.interaction import VERSION, state_id, state_matches, regions_from_ax
from web2kg.common.replay import verify
from web2kg.common.storage import EventLog, Graph
from web2kg.stage2.navigation import state_matches as navigation_matches
from scripts.migrate_state_model import remap, navigation_depths


def page():
    return {'url': 'http://local/', 'headings': ['Home'], 'regions': [['main', '', {}]],
            'identity_version': VERSION, 'actions': [{'role': 'button', 'name': 'Submit',
            'method': 'post', 'form_action': 'http://local/action?_token=one&item=1',
            'value_digest': 'one', 'occurrence': 0, 'match_count': 1}]}


def test_shared_equivalence_and_raw_evidence():
    before = page()
    before['id'] = state_id(before)
    after = copy.deepcopy(before)
    after['actions'][0]['form_action'] = 'http://local/action?_token=two&item=1'
    after['id'] = 'untrusted-observation-id'
    assert state_id(before) == state_id(after)
    assert navigation_matches is state_matches
    assert state_matches(before, after)
    assert before['actions'][0]['form_action'].endswith('_token=one&item=1')
    browser = SimpleNamespace(observe=Mock(return_value=after), page=Mock())
    assert verify(browser, before['id'], before) is after
    browser.page.wait_for_timeout.assert_not_called()


@pytest.mark.parametrize('field,value', [
    ('form_action', 'http://local/action?_token=two&item=2'),
    ('form_action', 'http://other/action?_token=two&item=1'),
    ('form_action', 'http://local/other?_token=two&item=1'),
    ('value_digest', 'two'), ('occurrence', 1), ('match_count', 2),
    ('disabled', True), ('state', {'expanded': True}), ('name', 'Other')])
def test_business_and_control_changes_remain_distinct(field, value):
    before, after = page(), page()
    after['actions'][0][field] = value
    # Even an equal/stale ID cannot override actual content differences.
    before['id'] = after['id'] = 'same'
    assert state_id(before) != state_id(after)
    assert not state_matches(before, after)


@pytest.mark.parametrize('method,parameter', [('get', '_token'), ('post', 'token'),
                                              ('post', 'nonce'), ('post', 'id')])
def test_no_new_token_heuristics(method, parameter):
    before, after = page(), page()
    for node, value in ((before, 'one'), (after, 'two')):
        node['actions'][0].update(method=method, form_action=f'http://local/action?{parameter}={value}')
    assert state_id(before) != state_id(after)
    assert not state_matches(before, after)


def test_regions_order_and_multiplicity_are_part_of_both_models():
    before, after = page(), page()
    after['regions'].append(['dialog', 'Confirm', {}])
    assert state_id(before) != state_id(after) and not state_matches(before, after)
    after = page()
    after['actions'].append(copy.deepcopy(after['actions'][0]))
    assert not state_matches(before, after) and state_id(before) != state_id(after)


def test_actual_browser_csrf_rotation_does_not_add_graph_state(site, tmp_path):
    graph = Graph(tmp_path / 'graph')
    with Browser(site, EventLog(tmp_path), identity_mode='interaction', settle_ms=20) as browser:
        browser.page.set_content('<main><h1>Home</h1><form method="post" action="/action?_token=one">'
                                 '<button>Submit</button></form></main>')
        first = browser.observe()
        assert regions_from_ax(first['ax']) == first['regions']
        graph.add_node(first, [])
        browser.page.locator('form').evaluate("e => e.action='/action?_token=two'")
        second = browser.observe()
        assert first['id'] == second['id'] == state_id(second)
        assert state_matches(first, second)
        assert not graph.add_node(second, [])
        browser.page.locator('form').evaluate("e => e.action='/action?_token=two&item=2'")
        changed = browser.observe()
        assert not state_matches(second, changed)
        assert graph.add_node(changed, [])


def test_remap_exact_references_preserves_provenance_text_and_rejects_collisions():
    assert remap({'old': ['old', '/path/old', 'old explanation']}, {'old': 'new'}) == {
        'new': ['new', '/path/old', 'old explanation']}
    with pytest.raises(ValueError):
        remap({'a': 1, 'b': 2}, {'a': 'same', 'b': 'same'})


def test_depths_use_only_replayable_routes_and_preserve_discovery(tmp_path):
    from test_replay_eligibility import graph_at
    graph = graph_at(tmp_path)
    graph.nodes['b']['depth'] = 25
    graph.data['edges'][1]['replayable'] = False
    assert navigation_depths(graph) == {'a': 0, 'b': 1}
    assert graph.nodes['b']['depth'] == 25


def test_legacy_graph_migration_rekeys_assets_and_preserves_source(site, tmp_path):
    import json
    import numpy as np
    from test_function_cards import CardModel
    from test_visual_navigation import FakeJina
    from test_execution_graph import hashes
    from web2kg.common.interaction import CONTROL_FIELDS
    from web2kg.common.storage import digest, write_json
    from web2kg.stage1.templates import build_templates
    from web2kg.stage1.function_cards import build_cards
    from web2kg.stage1.visual_index import build_visual_index
    from web2kg.stage1.site_framework import build_framework, load_framework
    from web2kg.stage2.recovery import ExperienceStore
    from scripts.migrate_state_model import migrate

    graph = Graph(tmp_path / 'legacy')
    with Browser(site, EventLog(graph.directory), identity_mode='interaction', settle_ms=20) as browser:
        def capture():
            observed = browser.observe()
            observed['id'] = digest(['interaction-v1', observed['url'], observed['headings'],
                observed.pop('regions'), [{k: a[k] for k in CONTROL_FIELDS if k in a} for a in observed['actions']]])
            observed['identity_version'] = 'interaction-v1'
            browser.save_observation(observed, graph.directory)
            return observed
        first = capture()
        graph.data.update(root=first['id'], start_url=first['url'], identity_mode='interaction',
                          experience_namespace='fixture-namespace')
        graph.add_node(first, [])
        action = next(a for a in first['actions'] if a['name'] == 'Catalog')
        browser.act(action)
        second = capture()
        graph.add_node(second, [action])
        graph.add_edge(first['id'], second['id'], action)
        graph.data['edges'][-1]['replayable'] = False
        graph.initialize_url_entries()
    graph.save()
    layer = build_templates(graph)
    layer_path = graph.directory / 'templates/test/templates.json'
    write_json(layer_path, layer)
    write_json(graph.directory / 'templates/active.json', {'directory': 'templates/test'})
    build_cards(graph, layer_path, graph.directory / 'function-cards/test', CardModel(tmp_path / 'fake-luna'))
    write_json(graph.directory / 'function-cards/active.json', {'cards': 'function-cards/test/cards.json'})
    framework_model = Mock(model='fixture-luna', last_request_id='fixture')
    framework_model.json.return_value = {'overview': 'Fixture site', 'areas': [
        {'name': 'Site', 'purpose': 'Browse', 'templates': [g['id'] for g in layer['templates']]}],
        'navigation': [], 'limitations': []}
    build_framework(graph, graph.directory / 'function-cards/test/cards.json', framework_model,
                    graph.directory / 'site-framework/test', publish=True)
    write_json(graph.directory / 'experiences/memory.json', {'version': 1, 'namespace': 'fixture-namespace',
        'pages': [{'node': second['id'], 'experience': {'lesson': 'fixture'}}], 'templates': {}})
    index = build_visual_index(graph, FakeJina())
    originals = hashes(graph.directory)
    updated, report = migrate(graph, tmp_path / 'migrated')
    mapping = json.loads((updated.directory / 'state-id-map.json').read_text())['nodes']
    assert set(mapping) == set(graph.nodes) and not set(graph.nodes) & set(updated.nodes)
    assert report['nodes'] == report['indexed'] == 2 and report['new_model_calls'] == 0
    assert updated.reachable() == {updated.data['root']}
    edge = updated.data['edges'][0]
    assert edge['source'] == mapping[first['id']] and edge['check']['state_id'] == mapping[second['id']]
    memory = ExperienceStore(updated.directory / 'experiences/memory.json', 'fixture-namespace')
    assert memory.context(mapping[second['id']], None)['page'] == [{'lesson': 'fixture'}]
    with np.load(graph.directory / index['vectors']) as old, np.load(updated.directory / 'state-vectors.npz') as new:
        assert all(np.array_equal(old[k], new[mapping[k]]) for k in old.files)
    assert hashes(graph.directory) == originals
    assert load_framework(updated)['framework']['overview'] == 'Fixture site'
    # A later execution copy reads its local summary without traversing migration ancestry.
    import shutil
    from web2kg.common.jina import sha256
    child_dir = tmp_path / 'execution-copy'
    shutil.copytree(updated.directory, child_dir)
    child = Graph.load(child_dir)
    child.data['execution_update'] = {'source_graph': str(updated.directory),
                                    'source_sha256': sha256(updated.directory / 'graph.json')}
    child.save()
    child_layer_path = child.directory / 'templates/state-model/templates.json'
    child_layer = json.loads(child_layer_path.read_text(encoding='utf-8'))
    child_layer.update(graph_digest=digest(child.data), graph_sha256=sha256(child.directory / 'graph.json'))
    write_json(child_layer_path, child_layer)
    assert load_framework(child)['inherited_initial_view']
    mapping_path = updated.directory / 'state-id-map.json'
    original_mapping = mapping_path.read_bytes()
    mapping_path.write_text('{}')
    assert load_framework(updated)['framework']['overview'] == 'Fixture site'
    mapping_path.write_bytes(original_mapping)
    path = updated.directory / 'templates/state-model/templates.json'
    changed_layer = json.loads(path.read_text(encoding='utf-8'))
    changed_layer['assignments'][mapping[first['id']]]['template'] = 'invented-template'
    write_json(path, changed_layer)
    assert load_framework(updated)['framework']['overview'] == 'Fixture site'
