import copy
from functools import partial
import json
import hashlib

import pytest
from playwright.sync_api import sync_playwright

from web2kg.common.storage import Graph, write_json
from web2kg.stage1.templates import (build_templates, control_profile, control_similarity,
                                    local_variant, page_similarity, structure_profile)
from web2kg.visualization.template_report import export_template_report
from web2kg.visualization.viewer import export_viewer
from web2kg.stage1.control_structure import ControlContext, control_comparison
from web2kg.stage1.page_structure import page_comparison
from web2kg.stage1.templates import match_candidate, PAGE_RULE, LEGACY_RULE

legacy_templates = partial(build_templates, threshold=.82, weights=(1., 1., 1.), match_rule=LEGACY_RULE)


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        yield page
        browser.close()


def snapshot(page, content):
    page.set_content(content)
    client = page.context.new_cdp_session(page)
    ax = client.send("Accessibility.getFullAXTree")["nodes"]
    dom = client.send("DOM.getDocument", {"depth": -1})["root"]
    client.detach()
    return ax, dom


def cards(count, prefix="Item", semantic=True):
    outer, item = ("ul", "li") if semantic else ("div", "div")
    return '<main><h1>Catalog</h1><' + outer + '>' + ''.join(
        f'<{item}><h2><a href="/{prefix}/{i}">{prefix} {i}</a></h2>'
        f'<a href="/users/{i}">Author {i}</a><button>Open</button></{item}>'
        for i in range(count)) + f'</{outer}></main>'


def profile(page, content):
    return structure_profile(*snapshot(page, content))


def test_repeated_entities_counts_urls_and_wrapper_type_do_not_define_template(page):
    a = profile(page, cards(3))
    b = profile(page, cards(12, "Different"))
    c = profile(page, cards(6, "Products", semantic=False))
    assert a["repeated"] and b["repeated"] and c["repeated"]
    assert page_similarity(a, b) == pytest.approx(1)
    assert page_similarity(a, c) == pytest.approx(1)
    assert "Different" not in json.dumps(b)


def test_nested_controls_differ_from_flat_same_role_counts(page):
    a = profile(page, cards(3))
    b = profile(page, '<main><h1>Catalog</h1>' + '<a href="/x">Title</a>' * 6
                + '<button>Open</button>' * 3 + '</main>')
    assert page_similarity(a, b) < .72


def test_names_values_and_tokens_do_not_affect_page_profile(page):
    def form(label, value, token):
        return f'<main><form method="post" action="/submit?token={token}"><label>{label}<input value="{value}"></label><button>Save</button></form></main>'
    a = profile(page, form("Email", "a@b.test", "private-one"))
    b = profile(page, form("Email", "new@b.test", "private-two"))
    c = profile(page, form("Quantity", "5", "private-three"))
    assert page_similarity(a, b) == pytest.approx(1)
    assert page_similarity(a, c) == pytest.approx(1)
    d = profile(page, form("Email", "a@b.test", "private-one").replace('<input ', '<input type="checkbox" '))
    assert page_similarity(a, d) < 1
    assert "private-" not in json.dumps(a)
    assert "a@b" not in json.dumps(a)


def test_no_landmarks_and_hidden_controls(page):
    a = profile(page, cards(3, semantic=False).replace('<main>', '<div>').replace('</main>', '</div>'))
    b = profile(page, cards(3, semantic=False).replace('<main>', '<div>').replace('</main>', '</div>')
                + '<div hidden><button>Erase everything</button></div>')
    assert not a["has_main"] and a["repeated"]
    assert page_similarity(a, b) == pytest.approx(1)


def test_missing_context_does_not_award_points_for_shared_empty_fields():
    a = control_profile(dict(kind="click", role="link", name="A"))
    b = control_profile(dict(kind="click", role="link", name="B"))
    assert control_similarity(a, b) == 0


def test_same_url_alone_is_not_a_state_variant(page):
    a, b = profile(page, cards(3)), profile(page, '<main><form><input aria-label="Email"><button>Save</button></form></main>')
    node = {"url": "http://local.test/"}
    assert not local_variant(node, node, {"kind": "click", "role": "button"}, a, b)
    assert not local_variant(node, node, {"kind": "click", "state": {"expanded": False}}, a, b)


def fixture_graph(page, tmp_path):
    graph = Graph(tmp_path)
    root_action = dict(id="a", kind="click", role="link", name="First", xpath="/html[1]/body[1]/main[1]/ul[1]/li[1]/h2[1]/a[1]")
    other_action = {**root_action, "id": "b", "name": "Second", "xpath": "/html[1]/body[1]/main[1]/ul[1]/li[2]/h2[1]/a[1]"}
    toggle = dict(id="toggle", kind="click", role="button", name="Account", state={"expanded": False},
                  xpath="/html[1]/body[1]/main[1]/article[1]/button[1]")
    contents = {
        "root": ('http://local.test/list', [], cards(3)),
        "first": ('http://local.test/item/a', [root_action], '<main><h1>First</h1><article><p>Body A</p><button aria-expanded="false">Account</button></article></main>'),
        "second": ('http://local.test/item/b', [other_action], '<main><h1>Second</h1><article><p>Body B</p><button aria-expanded="false">Account</button></article></main>'),
        "expanded": ('http://local.test/item/a', [root_action, toggle], '<main><h1>First</h1><article><p>Body A</p><button aria-expanded="true">Account</button></article></main><nav><a href="/settings">Settings</a></nav>')}
    for key, (url, path, content) in contents.items():
        ax, dom = snapshot(page, content)
        current_toggle = {**toggle, "state": {"expanded": key == "expanded"}}
        graph.nodes[key] = dict(id=key, url=url, path=path, depth=len(path), title=key, headings=[], actions=[current_toggle])
        write_json(tmp_path / 'snapshots' / key / 'ax.json', ax)
        write_json(tmp_path / 'snapshots' / key / 'dom.json', dom)
    graph.data['root'] = 'root'
    for source, target, action in [('root', 'first', root_action), ('root', 'second', other_action), ('first', 'expanded', toggle)]:
        graph.add_edge(source, target, action)
    graph.save()
    return graph


def test_end_to_end_cross_url_context_state_and_evidence_without_mutation(page, tmp_path):
    graph = fixture_graph(page, tmp_path)
    before = copy.deepcopy(graph.data)
    result = legacy_templates(graph)
    a = result['assignments']
    assert a['first']['template'] == a['second']['template'] == a['expanded']['template']
    assert a['root']['template'] != a['first']['template']
    assert a['expanded']['reason'] == 'observed_local_state_variant'
    assert a['expanded']['state'] != a['first']['state']
    scores = a['second']['selected_match']['scores']
    assert a['second']['selected_match']['score'] == sum(scores.values()) / 3
    assert sum(len(t['evidence_edges']) for t in result['transitions']) == len(graph.data['edges'])
    assert graph.data == before
    assert legacy_templates(graph) == result
    (tmp_path / 'snapshots' / 'first' / 'ax.json').unlink()
    with pytest.raises(FileNotFoundError):
        legacy_templates(graph)


def test_template_report_filter_members_and_scores(page, tmp_path):
    graph = fixture_graph(page, tmp_path)
    result = legacy_templates(graph, weights=(.5, 1.5, 1))
    out = tmp_path / 'templates' / 'context-structure-v1'
    out.mkdir(parents=True)
    export_template_report(graph, result, out)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto((out / 'report.html').as_uri())
    assert page.locator('#groups button').count() == 2
    assert '0.5 / 1.5 / 1' in page.locator('#description').inner_text()
    page.locator('#groups button').nth(1).click()
    assert page.locator('.card').count() == 3
    page.locator('.card summary').first.click()
    assert page.locator('.card pre').first.is_visible()
    page.locator('#filter').fill('item/b')
    assert page.locator('#groups button').count() == 1
    assert not errors


def test_viewer_links_only_matching_template_graph(page, tmp_path):
    graph = fixture_graph(page, tmp_path)
    result = legacy_templates(graph)
    out = tmp_path / 'templates' / 'context-structure-v2-test'
    out.mkdir(parents=True)
    export_template_report(graph, result, out)
    write_json(out / 'templates.json', result)
    write_json(tmp_path / 'templates' / 'active.json', {'directory': out.relative_to(tmp_path).as_posix()})
    page.goto(export_viewer(graph).as_uri())
    assert page.locator('#templateReport').is_visible()
    page.locator('#templateTab').click()
    assert page.locator('#templatePanel').is_visible()
    assert page.locator('#detail').is_hidden()
    frame = page.frame_locator('#templateFrame')
    assert frame.locator('.template-node').count() == 2
    page.locator('#logTab').click()
    assert page.locator('#logPanel').is_visible()
    assert page.locator('#templatePanel').is_hidden()
    page.locator('#templateReport').click()
    assert page.locator('#groups button').count() == 2
    graph.nodes['root']['title'] = 'Changed graph'
    page.goto(export_viewer(graph).as_uri())
    assert not page.locator('#templateReport').is_visible()
    assert page.locator('#templateTab').is_hidden()


def test_template_map_cards_and_transition_evidence(page, tmp_path):
    graph = fixture_graph(page, tmp_path)
    result = legacy_templates(graph)
    out = tmp_path / 'templates' / 'active-test'
    write_json(out / 'templates.json', result)
    template = result['assignments']['first']['template']
    card = {'name': 'Detail template', 'summary': 'Saved sample functionality',
            'common_functions': ['Read'], 'variants': [], 'uncertainties': ['Sample only']}
    cards_path = tmp_path / 'cards.json'
    source = {'layer': 'templates/active-test/templates.json',
              'layer_sha256': hashlib.sha256((out / 'templates.json').read_bytes()).hexdigest(),
              'graph_sha256': hashlib.sha256((tmp_path / 'graph.json').read_bytes()).hexdigest()}
    write_json(cards_path, {'source': source, 'cards': {template: {'status': 'completed', 'card': card}}})
    before = (tmp_path / 'graph.json').read_bytes()
    export_template_report(graph, result, out, cards_path)
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto((out / 'report.html').as_uri())
    page.locator(f'[data-template="{template}"]').click()
    assert 'Detail template' in page.locator('#functionCard').inner_text()
    page.locator('#mapScope').select_option('all')
    assert page.locator('.template-edge').count() == len(result['transitions'])
    edge = page.locator('.template-edge-hit').first
    edge.scroll_into_view_if_needed()
    point = edge.evaluate('''p => {
        const middle = p.getPointAtLength(p.getTotalLength() / 2);
        const screen = new DOMPoint(middle.x, middle.y).matrixTransform(p.getScreenCTM());
        return {x: screen.x, y: screen.y};
    }''')
    page.mouse.click(point['x'], point['y'])
    assert '实际边' in page.locator('#edgeInfo').inner_text()
    assert '源截图' in page.locator('#edgeInfo').inner_text()
    assert len(page.locator('#edgeInfo li').all()) >= 1
    page.locator('[data-transition]').last.focus()
    page.keyboard.press('Enter')
    assert '实际边' in page.locator('#edgeInfo').inner_text()
    page.locator('#mapZoom').fill('140')
    page.locator('#mapFit').click()
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.set_viewport_size({'width': 1280, 'height': 720})
    assert not errors
    assert before == (tmp_path / 'graph.json').read_bytes()
    source['graph_sha256'] = 'stale'
    write_json(cards_path, {'source': source, 'cards': {}})
    with pytest.raises(ValueError, match='do not match'):
        export_template_report(graph, result, out, cards_path)


def test_names_never_enter_page_or_control_features(page):
    ax, dom = snapshot(page, '<main aria-label="Shop"><form><label>Secret field<input></label>'
                       '<button>Save</button></form></main>')
    changed = copy.deepcopy(ax)
    for node in changed:
        if 'name' in node:
            node['name']['value'] = 'Completely different name'
    assert structure_profile(ax, dom) == structure_profile(changed, dom)
    a, b = ControlContext(ax, dom), ControlContext(changed, dom)
    path = next(p for p in a.paths if p.endswith('/button[1]'))
    assert a.profile({'kind':'click','role':'button','xpath':path,'name':'Save'}) == b.profile(
        {'kind':'click','role':'button','xpath':path,'name':'Delete'})


def test_same_parent_and_entry_cannot_override_different_page_structure(page, tmp_path):
    graph = fixture_graph(page, tmp_path)
    ax, dom = snapshot(page, '<main><form method="post"><label>Email<input type="email"></label>'
                       '<label>Password<input type="password"></label><button>Register</button></form></main>')
    write_json(tmp_path / 'snapshots' / 'second' / 'ax.json', ax)
    write_json(tmp_path / 'snapshots' / 'second' / 'dom.json', dom)
    result = legacy_templates(graph)
    assert result['assignments']['first']['template'] != result['assignments']['second']['template']


def test_entry_weight_changes_assignment_without_changing_structure(page, tmp_path):
    graph = fixture_graph(page, tmp_path)
    action = graph.data['edges'][1]['action']
    # Enter via the author link in the same real list item, rather than its heading.
    action['xpath'] = '/html[1]/body[1]/main[1]/ul[1]/li[2]/a[1]'
    initial = legacy_templates(graph)
    candidate = next(c for d in initial['decisions'] if d['node'] == 'second' for c in d['candidates'])
    entry_score = candidate['scores']['incoming']
    assert 0 < entry_score < 1
    threshold = ((2 + entry_score) / 3 + (.5 + 1.5 * entry_score + 1) / 3) / 2
    equal = legacy_templates(graph, threshold=threshold)
    weighted = legacy_templates(graph, threshold=threshold, weights=(.5, 1.5, 1))
    assert equal['assignments']['first']['template'] == equal['assignments']['second']['template']
    assert weighted['assignments']['first']['template'] != weighted['assignments']['second']['template']
    assert equal['profiles'] == weighted['profiles']
    candidate = next(c for d in weighted['decisions'] if d['node'] == 'second' for c in d['candidates'])
    assert candidate['scores'] == pytest.approx({'parent': 1, 'incoming': entry_score, 'page': 1})
    assert candidate['score'] == pytest.approx((.5 + 1.5 * entry_score + 1) / 3)
    assert weighted['parameters']['raw_weights'] == {'parent': .5, 'incoming': 1.5, 'page': 1}


@pytest.mark.parametrize('weights', [(0, 0, 0), (-1, 1, 1), (float('nan'), 1, 1), (float('inf'), 1, 1), (1, 1)])
def test_invalid_weights_fail_before_reading_snapshots(tmp_path, weights):
    with pytest.raises(ValueError, match='weights'):
        legacy_templates(Graph(tmp_path), weights=weights)


def incoming(page, content, suffix):
    context = ControlContext(*snapshot(page, content))
    path = next(p for p in context.paths if p.endswith(suffix))
    role = context.nodes[context.paths[path]]['role']
    return context.profile(dict(kind='click', role=role, xpath=path, name='ignored'))


def test_control_names_list_count_and_outer_wrappers_are_ignored(page):
    a = incoming(page, cards(3), '/li[1]/h2[1]/a[1]')
    b = incoming(page, '<div><div>' + cards(8, 'Unrelated') + '</div></div>', '/li[7]/h2[1]/a[1]')
    assert a == b
    assert a['relative_depth'] == 2
    assert control_similarity(a, b) == pytest.approx(1)
    save = incoming(page, '<main><div><button>Save</button><a href="/a">Other</a></div></main>', '/button[1]')
    delete = incoming(page, '<main><div><button aria-label="Delete">Erase</button><a href="/b">Different</a></div></main>', '/button[1]')
    assert save == delete


def test_relative_depth_parent_and_sibling_structures_change_scores(page):
    a = incoming(page, cards(3), '/li[1]/h2[1]/a[1]')
    b = incoming(page, cards(3), '/li[1]/a[1]')
    assert a['relative_depth'] == 2 and b['relative_depth'] == 1
    comparison = control_comparison(a, b)
    assert 0 < comparison['score'] < 1
    assert comparison['components']['relative_depth'] == .5
    c = incoming(page, cards(3).replace('<button>Open</button>', '<input aria-label="Quantity">'), '/li[1]/h2[1]/a[1]')
    assert a['parent_structure'] != c['parent_structure']
    assert control_similarity(a, c) < 1
    d = incoming(page, '<main><div><button>Open</button><a href="/x">A</a></div></main>', '/a[1]')
    e = incoming(page, '<main><div><input aria-label="Field"><a href="/x">A</a></div></main>', '/a[1]')
    assert d['siblings'] != e['siblings']
    assert control_comparison(d, e)['components']['siblings'] == 0


def test_role_action_mismatch_is_zero_and_empty_siblings_add_no_points(page):
    a = incoming(page, '<main><a href="/x">A</a></main>', '/a[1]')
    assert control_similarity(a, {**a, 'kind': 'fill'}) == 0
    assert control_similarity(a, {**a, 'role': 'button'}) == 0
    assert 'siblings' not in control_comparison(a, a)['components']
    assert 'function' not in a and 'name' not in a


def test_one_and_many_cards_keep_same_structure(page):
    single = profile(page, cards(1, semantic=False))
    for n, semantic in ((3, False), (1, True), (8, True)):
        other = profile(page, cards(n, 'Changed', semantic=semantic))
        assert page_similarity(single, other) == pytest.approx(1)
        assert page_comparison(single, other)['evidence_sufficient']


def test_parent_child_combinations_survive_path_set_collision(page):
    a = profile(page, '<main><section><h2>A</h2><input><a href="/x">X</a></section>'
                '<section><h2>B</h2><input><button>Open</button></section></main>')
    b = profile(page, '<main><section><h2>A</h2><input><a href="/x">X</a><button>Open</button></section>'
                '<section><h2>B</h2><input></section></main>')
    assert a['structure'] == b['structure']
    assert page_similarity(a, b) < .72


def test_sidebar_cannot_contaminate_main_features(page):
    main = cards(3)
    a = profile(page, main + '<aside><ul><li><a href="/x">X</a></li></ul></aside>')
    b = profile(page, main + '<aside><ul><li><button>X</button></li></ul></aside>')
    assert a['region_features']['main'] == b['region_features']['main']
    assert a['repeated'] == b['repeated']
    result = page_comparison(a, b)
    assert result['core_score'] == pytest.approx(1)
    assert .85 <= result['score'] < 1


def test_missing_evidence_is_not_positive_match_and_text_list_is_retained(page):
    a = profile(page, '<main><h1>Page</h1><p>Body</p></main>')
    b = profile(page, '<main><h1>Page</h1><ul><li>A</li><li>B</li></ul></main>')
    result = page_comparison(a, b)
    assert 'controls' not in result['regions']['main']['components']
    assert result['score'] < .72 and not result['evidence_sufficient']
    assert any('listitem' in x for x in b['structure'])
    same = page_comparison(a, a)
    assert same['score'] == pytest.approx(1)
    assert same['reason'] == 'insufficient_core_structure'


def test_sparse_pages_merge_with_observed_agreeing_context(page, tmp_path):
    graph = fixture_graph(page, tmp_path)
    ax, dom = snapshot(page, '<main><h1>Empty</h1><p>No entries</p></main>')
    for key in ('first', 'second'):
        write_json(tmp_path / 'snapshots' / key / 'ax.json', ax)
        write_json(tmp_path / 'snapshots' / key / 'dom.json', dom)
    result = legacy_templates(graph, weights=(.5, 1.5, 1))
    assert result['assignments']['first']['template'] == result['assignments']['second']['template']
    candidate = next(c for d in result['decisions'] if d['node'] == 'second' for c in d['candidates'])
    assert candidate['score'] == pytest.approx(1)
    assert candidate['eligible']
    assert candidate['eligibility_reason'] == 'weighted_score_only'
    assert candidate['rejection_reason'] is None
    assert candidate['page_comparison']['reason'] == 'insufficient_core_structure'
    assert result['summary']['low_evidence_nodes'] >= 2
    out = tmp_path / 'report'
    out.mkdir()
    export_template_report(graph, result, out)
    page.goto((out / 'report.html').as_uri())
    page.locator('[data-group="' + result['assignments']['first']['template'] + '"]').click()
    assert '结构证据不足' in page.locator('#members').inner_text()
    assert '不参与否决' in page.locator('#description').inner_text()
    assert 'page_floor' not in result['parameters']


@pytest.mark.parametrize('dimension,value', [('parent', .1), ('incoming', .8), ('page', .55)])
def test_low_component_and_sparse_evidence_do_not_veto_weighted_match(page, tmp_path, monkeypatch, dimension, value):
    import web2kg.stage1.templates as templates
    graph = fixture_graph(page, tmp_path)
    original = templates.page_comparison
    def comparison(a, b, weights=None):
        return {**original(a, b, weights), 'score': value if dimension == 'page' else 1.,
                'evidence_sufficient': False, 'reason': 'insufficient_core_structure'}
    monkeypatch.setattr(templates, 'page_comparison', comparison)
    if dimension == 'parent':
        monkeypatch.setattr(templates, 'page_similarity', lambda a, b, weights=None: value)
    if dimension == 'incoming':
        original_control = templates.control_comparison
        monkeypatch.setattr(templates, 'control_comparison', lambda a, b: {**original_control(a, b), 'score': value})
    result = legacy_templates(graph, weights=(.5, 1.5, 1))
    a = result['assignments']
    assert a['second']['template'] == a['first']['template']
    assert a['second']['selected_match']['scores'][dimension] == value
    assert a['second']['selected_match']['rejection_reason'] is None


def test_total_threshold_still_rejects_sparse_match(page, tmp_path):
    graph = fixture_graph(page, tmp_path)
    ax, dom = snapshot(page, '<main><h1>Empty</h1><p>No entries</p></main>')
    for key in ('first', 'second'):
        write_json(tmp_path / 'snapshots' / key / 'ax.json', ax)
        write_json(tmp_path / 'snapshots' / key / 'dom.json', dom)
    graph.data['edges'][1]['action']['xpath'] = '/missing-control'
    result = legacy_templates(graph, weights=(.5, 1.5, 1))
    a = result['assignments']
    assert a['first']['template'] != a['second']['template']
    candidate = next(d for d in result['decisions'] if d['node'] == 'second')['candidates'][0]
    assert candidate['scores'] == dict(parent=1, incoming=0, page=1)
    assert candidate['score'] == .5
    assert candidate['rejection_reason'] == 'score_below_threshold'


@pytest.mark.parametrize('tag,outer', [('article', 'div'), ('li', 'ul')])
def test_semantic_entry_anchor_stable_for_single_and_repeated_items(page, tag, outer):
    def content(count):
        item = f'<{tag}><h2><a href="/item">Title</a></h2><a href="/user">Author</a><button>Open</button></{tag}>'
        return f'<main><{outer}>' + item * count + f'</{outer}></main>'
    a = incoming(page, content(1), f'/{tag}[1]/h2[1]/a[1]')
    b = incoming(page, content(4), f'/{tag}[3]/h2[1]/a[1]')
    assert a['anchor_role'] == b['anchor_role'] == ('listitem' if tag == 'li' else tag)
    assert a['relative_depth'] == b['relative_depth'] == 2
    assert control_similarity(a, b) == pytest.approx(1)


def test_page_gate_accepts_identical_page_with_no_parent_or_entry_score(page, tmp_path, monkeypatch):
    import web2kg.stage1.templates as templates
    graph = fixture_graph(page, tmp_path)
    graph.data['edges'][1]['action']['xpath'] = '/missing-control'
    monkeypatch.setattr(templates, 'page_similarity', lambda a, b, weights=None: 0.)
    result = build_templates(graph)
    assignment = result['assignments']['second']
    assert assignment['template'] == result['assignments']['first']['template']
    selected = assignment['selected_match']
    assert selected['scores'] == dict(parent=0, incoming=0, page=1)
    assert selected['score'] == 1 and selected['eligible']
    assert selected['ranking_basis'] == 'page_only'
    assert selected['eligibility_reason'] == PAGE_RULE
    assert result['parameters']['raw_weights'] == dict(parent=.5, incoming=.5, page=1)
    assert result['parameters']['threshold'] == .85
    out = tmp_path / 'report'
    out.mkdir()
    export_template_report(graph, result, out)
    page.goto((out / 'report.html').as_uri())
    assert '只要求页面结构分 ≥ 0.85' in page.locator('#description').inner_text()
    assert '加权总分仅排序' in page.locator('#description').inner_text()


@pytest.mark.parametrize('score,accepted', [(.849, False), (.85, True)])
def test_page_gate_cannot_be_compensated_by_perfect_context(score, accepted):
    parameters = dict(raw_weights=dict(parent=.5, incoming=.5, page=1), threshold=.85, match_rule=PAGE_RULE)
    result = match_candidate(dict(parent=1, incoming=1, page=score), parameters)
    assert result['score'] > .9
    assert result['eligible'] is accepted
    assert result['rejection_reason'] == (None if accepted else 'page_below_threshold')


def test_weights_rank_eligible_candidates_but_do_not_change_admission():
    scores = [dict(parent=1, incoming=0, page=.9), dict(parent=0, incoming=1, page=.9)]
    def compare(weights):
        return [match_candidate(s, dict(raw_weights=weights, threshold=.85, match_rule=PAGE_RULE)) for s in scores]
    parent = compare(dict(parent=1, incoming=.5, page=1))
    incoming = compare(dict(parent=.5, incoming=1, page=1))
    assert all(c['eligible'] for c in parent + incoming)
    assert parent[0]['score'] > parent[1]['score']
    assert incoming[0]['score'] < incoming[1]['score']


def test_preview_keeps_active_templates_cards_and_graph(page, tmp_path):
    from scripts.build_page_templates import run
    graph = fixture_graph(page, tmp_path)
    active = tmp_path / 'templates/active.json'
    cards = tmp_path / 'function-cards/active.json'
    write_json(active, {'directory': 'templates/original'})
    write_json(cards, {'cards': 'function-cards/original/cards.json'})
    before = {p: p.read_bytes() for p in (active, cards, tmp_path / 'graph.json')}
    result = run(tmp_path)
    assert result['parameters']['match_rule'] == PAGE_RULE
    assert all(p.read_bytes() == value for p, value in before.items())


def test_control_paths_are_subset_and_no_longer_scored_twice(page):
    from web2kg.stage1.page_structure import LEGACY_PAGE_FEATURES
    a = profile(page, '<main><h1>Form</h1><form><label>Value<input></label><button>Save</button></form></main>')
    b = profile(page, '<main><h1>Form</h1><form><label>Value<input type="checkbox"></label><button>Save</button></form></main>')
    for row in (a, b):
        for region in row['region_features'].values():
            assert set(region['controls']) <= set(region['structure'])
    new, old = page_comparison(a, b), page_comparison(a, b, LEGACY_PAGE_FEATURES)
    assert set(new['regions']['main']['components']) == {'structure', 'combinations'}
    assert 'controls' in old['regions']['main']['components']
    assert new['score'] != old['score']
    assert new['score'] < 1  # Control role/type differences still appear in the retained structure.
    without_control_list = copy.deepcopy(a)
    without_control_list['region_features']['main']['controls'] = []
    assert page_comparison(a, without_control_list)['score'] == pytest.approx(1)


def test_new_and_saved_scoring_metadata_are_used_offline(page, tmp_path):
    graph = fixture_graph(page, tmp_path)
    new, old = build_templates(graph), legacy_templates(graph)
    assert new['parameters']['page_features'] == dict(structure=.5, combinations=.5)
    assert old['parameters']['page_features'] == dict(structure=.4, combinations=.4, controls=.2)
    for layer, expected in ((new, {'structure', 'combinations'}), (old, {'structure', 'combinations', 'controls'})):
        comparison = layer['assignments']['second']['selected_match']['page_comparison']
        assert set(comparison['regions']['main']['components']) == expected
