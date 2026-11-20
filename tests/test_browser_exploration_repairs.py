import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from playwright.sync_api import TimeoutError as BrowserTimeout

from test_construction_pipeline import Model
from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog
from web2kg.stage1.crawl import CrawlBudget, crawl
from web2kg.stage1.llm_exploration import LLMExplorer
from web2kg.stage1.pipeline import PipelinedExplorer


@pytest.fixture
def repair_site():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                body = '<h1>Home</h1><a href="/menu">Account</a><a href="/ok">Independent</a>'
            elif self.path == '/menu':
                self.server.menus += 1
                body = ('<h1>Account</h1><a href="/a">First</a><a href="/b">Second</a><a href="/c">Third</a>'
                        f'<form method="post" action="/exit?token={self.server.menus}"><button>Exit</button></form>')
            elif self.path == '/ok':
                body = '<h1>Independent</h1><a href="/next">Next</a>'
            elif self.path == '/slow':
                self.server.slow_requests += 1
                time.sleep(.8)
                body = '<h1>Loaded once</h1>'
            else:
                body = '<h1>Destination</h1>'
            raw = ('<!doctype html><title>Fixture</title><main>' + body + '</main>').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.menus = server.slow_requests = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}/', server
    server.shutdown()
    server.server_close()
    thread.join()


def test_associated_labels_activate_once_and_do_not_bypass_unrelated_overlay(repair_site, tmp_path):
    url, _ = repair_site
    with Browser(url, EventLog(tmp_path), settle_ms=10) as browser:
        browser.page.set_content('''<style>label{display:block;position:relative;width:140px;height:32px}
        input{position:absolute;left:0;top:0;width:20px;height:20px}
        label span{position:absolute;inset:0;background:white}</style>
        <label><input id="check" type="checkbox" onchange="window.changes=(window.changes||0)+1"><span>Help</span></label>
        <label><input id="image" type="radio" name="format"><span>Image</span></label>
        <label><input id="url" type="radio" name="format"><span>URL</span></label>''')
        browser.context.set_default_timeout(300)
        controls = {a['name']:a for a in browser.observe()['actions']}
        browser.act(controls['Help'])
        assert browser.page.locator('#check').is_checked()
        assert browser.page.evaluate('window.changes') == 1
        browser.act(controls['Help'])
        assert not browser.page.locator('#check').is_checked()
        assert browser.page.evaluate('window.changes') == 2
        browser.act(controls['Image'])
        assert browser.page.locator('#image').is_checked()
        browser.act(controls['URL'])
        assert browser.page.locator('#url').is_checked() and not browser.page.locator('#image').is_checked()
        browser.page.evaluate('''() => {const e=document.createElement('div');
            e.style='position:fixed;inset:0;background:white;z-index:999';document.body.append(e)}''')
        with pytest.raises(BrowserTimeout):
            browser.act(controls['Help'])
        assert browser.page.evaluate('window.changes') == 2


def test_negative_position_links_filtered_but_below_fold_links_remain(repair_site, tmp_path):
    from web2kg.common.storage import Graph
    url, _ = repair_site
    with Browser(url, EventLog(tmp_path), settle_ms=10) as browser:
        browser.page.set_content('''<a style="position:absolute;left:-10000px" href="#main">Skip</a>
        <div style="height:1500px">Content</div><a href="/next">Below fold</a><main id="main">Main</main>''')
        node = browser.observe()
        skipped = next(a for a in node['actions'] if a['name'] == 'Skip')
        visible = next(a for a in node['actions'] if a['name'] == 'Below fold')
        assert skipped['pointer_unavailable'] == 'negative_position_link'
        assert not visible.get('pointer_unavailable')
        graph = Graph(tmp_path); graph.data['start_url'] = url; graph.add_node(node, [])
        model = Model(tmp_path)
        chosen = LLMExplorer(model, tmp_path).select(graph, node, 4)
        assert [a['name'] for a in chosen] == ['Below fold']
        with pytest.raises(ValueError, match='negative-position'):
            browser.act(skipped)
        browser.act(visible)
        assert browser.page.url.endswith('/next')


def test_plain_navigation_has_own_wait_budget_without_second_click(repair_site, tmp_path):
    url, server = repair_site
    with Browser(url, EventLog(tmp_path), settle_ms=10) as browser:
        browser.page.set_content('<a href="/slow">Slow destination</a>')
        action = browser.observe()['actions'][0]
        browser.context.set_default_timeout(300)
        browser.act(action)
        assert browser.page.locator('h1').inner_text() == 'Loaded once'
        assert server.slow_requests == 1
    events = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert sum(e['event'] == 'browser_click_dispatched' for e in events) == 1


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('learn', [False, True])
def test_unrestorable_source_skips_siblings_but_other_frontier_continues(repair_site, tmp_path, parallel, learn):
    url, server = repair_site

    class Ranking(Model):
        def json(self, system, payload, schema, purpose, images=()):
            self.fixture_calls = getattr(self, 'fixture_calls', 0) + 1
            self.last_request_id = 'fixture-' + str(self.fixture_calls)
            if purpose == 'candidate_ranking':
                rows = [a for a in payload['page']['candidates'] if a['name'] != 'Exit']
                return {'selected':[dict(id=a['id'],group=a['name'],reason='Fixture') for a in rows[:payload['K']]]}
            if 'page_descriptions_requested' not in payload:
                return dict(replayable=True,replay_reason='Fixture navigation', before_page_description='Source',
                    after_page_description='Destination',operation_description='Navigation',result_description='Opened',
                    control_evidence=[dict(ref=payload['provided_evidence'][0]['ref'],description='Control')],uncertainties=[])
            return super().json(system,payload,schema,purpose,images)

    model = Ranking(tmp_path)
    explorer = PipelinedExplorer(model, tmp_path) if parallel else LLMExplorer(model, tmp_path)
    from types import SimpleNamespace
    from unittest.mock import Mock
    judge = SimpleNamespace(json=Mock(return_value={'decision':'same','score':.99}),last_request_id='fixture') if learn else None
    with Browser(url, EventLog(tmp_path), identity_mode='interaction', settle_ms=10) as browser:
        graph = crawl(browser, tmp_path, model.log, CrawlBudget(max_states=10,max_depth=3,
                      actions_per_page=3,max_attempts=10), explorer=explorer, state_judge=judge)
    failures = [r for r in graph.data['transition_records'] if r['outcome'] == 'not_attempted']
    if learn:
        assert not failures and not graph.data.get('unavailable_sources')
        assert judge.json.call_count == 1
        assert len(graph.data['state_equivalence']['rules']) == 1
        assert len([n for n in graph.nodes.values() if n['url'].endswith('/menu')]) == 1
        assert all(any(r['action']['name'] == name and r['action_attempted'] for r in graph.data['transition_records'])
                   for name in ['First','Second','Third'])
        return
    assert len(failures) == 1
    assert 'State mismatch' in failures[0]['error']
    assert len(graph.data['unavailable_sources']) == 1
    assert len(graph.data['unavailable_sources'][0]['skipped_candidate_ids']) == 2
    assert server.menus == 2
    assert any(n['url'].endswith('/next') for n in graph.nodes.values())
    assert not any(r['action']['name'] in {'Second','Third'} for r in graph.data['transition_records'])
