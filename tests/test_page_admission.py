import os
import hashlib

from test_function_cards import saved
from web2kg.common.browser import Browser
from web2kg.common.interaction import state_id, state_matches
from web2kg.common.storage import EventLog
from web2kg.stage1.page_admission import admitted, observation_profile
from web2kg.stage1.template_reconciliation import reconcile
from web2kg.stage1.templates import build_templates
from web2kg.stage2.graph_update import ExecutionGraph
from web2kg.stage2.navigation import RouteNavigator
from web2kg.stage2.template_context import TemplateKnowledge


def test_blank_loading_and_visible_but_unextractable_pages(saved, site, tmp_path):
    base, _, _, cards, _ = saved
    log = EventLog(tmp_path / 'task')
    knowledge = TemplateKnowledge(base, cards, log)
    with Browser(site, log, identity_mode='interaction', settle_ms=10) as browser:
        updater = ExecutionGraph(base, browser, log, knowledge, max_cards=0)
        before = set(updater.graph.nodes)
        browser.page.set_content('<html><body></body></html>')
        blank = browser.observe()
        assert blank['page_content']['kind'] == 'blank'
        assert not updater.observe(blank) and set(updater.graph.nodes) == before
        nav = RouteNavigator(browser, base, log)
        nav.recorder = updater
        assert not nav.connect() and blank['id'] not in nav.graph.nodes
        browser.page.set_content('<div role="progressbar"><svg width="60" height="60"><circle cx="20" cy="20" r="10"/></svg></div>')
        loading = browser.observe()
        assert loading['page_content']['kind'] == 'loading_only'
        assert not updater.observe(loading)
        browser.page.set_content('<div aria-hidden="true"><canvas width="160" height="80"></canvas></div>')
        browser.page.evaluate("document.querySelector('canvas').getContext('2d').fillRect(10,10,90,40)")
        visual = browser.observe()
        assert admitted(visual) and observation_profile(visual)['isolated']
        assert state_id(visual) == visual['id']
        updater.record(loading, {'kind': 'wait', 'id': 'wait', 'role': 'browser', 'name': 'wait'}, visual)
        assert visual['id'] in updater.graph.nodes and loading['id'] not in updater.graph.nodes
        assert hashlib.sha256((updater.graph.directory / 'snapshots' / visual['id'] / 'page.png').read_bytes()).hexdigest() == visual['page_content']['visual_sha256']
        assert not any(loading['id'] in (e['source'], e['target']) for e in updater.graph.data['edges'])
        template = updater.layer['assignments'][visual['id']]['template']
        assert knowledge.match(visual)['template_id'] == template
        browser.page.evaluate("document.querySelector('canvas').getContext('2d').fillRect(110,10,20,40)")
        another = browser.observe()
        assert another['id'] != visual['id'] and not state_matches(visual, another)
        updater.observe(another, visual, {'kind': 'click', 'role': 'button', 'name': 'Next'})
        other = updater.layer['assignments'][another['id']]['template']
        assert other != template
        updated, audit = reconcile(updater.layer)
        assert not any({template, other} & set(m['sources']) for m in audit['merges'])
        assert all(g['members'] == [g['representative']] for g in updated['templates'] if g.get('isolated'))
        offline = build_templates(updater.graph)
        assert offline['assignments'][visual['id']]['reason'] == 'isolated_empty_structure'
        assert offline['assignments'][another['id']]['template'] != offline['assignments'][visual['id']]['template']
        updater.finish(allow_card_calls=False)
        assert visual['id'] in updater.graph.nodes and another['id'] in updater.graph.nodes


def test_screenshot_replacement_does_not_overwrite_inherited_hardlink(site, tmp_path):
    original = tmp_path / 'historical.png'
    original.write_bytes(b'historical image')
    target = tmp_path / 'current.png'
    os.link(original, target)
    with Browser(site, EventLog(tmp_path), settle_ms=10) as browser:
        browser.screenshot(target)
    assert original.read_bytes() == b'historical image'
    assert target.read_bytes().startswith(b'\x89PNG')


def test_missing_content_evidence_never_discards_an_empty_structure():
    observation = {'ax': [], 'dom': {'nodeType': 9, 'nodeName': '#document'}}
    assert admitted(observation) and observation_profile(observation)['isolated']


def test_initial_loading_page_does_not_become_a_crawl_root(site, tmp_path):
    from web2kg.stage1.crawl import crawl, CrawlBudget
    log = EventLog(tmp_path)
    with Browser(site, log, settle_ms=10) as browser:
        browser.page.set_content('<div aria-busy="true">Loading...</div>')
        graph = crawl(browser, tmp_path, log, CrawlBudget(max_states=2))
    assert graph.data['root'] is None and not graph.nodes and not graph.data['edges']
