"""Browser integration: a rejected first click never reaches the server."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import requests

from test_construction_pipeline import Model
from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog
from web2kg.stage1.crawl import CrawlBudget, crawl
from web2kg.stage1.jev_ranking import JevRankingClient
from web2kg.stage1.pipeline import PipelinedExplorer


def test_prefetched_jev_gate_blocks_first_write_and_keeps_navigation(tmp_path, monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = '<h1>Home</h1><a href="/forums">Forums</a>'
            if self.path == '/forums':
                body = ('<h1>Forums</h1><form method="post" action="/subscribe">'
                        '<button>Subscribe</button></form><a href="/next">Next</a>')
            elif self.path == '/next':
                body = '<h1>Next page</h1><p>Forum directory continued</p>'
            raw = ('<!doctype html><title>Fixture</title><main>' + body + '</main>').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            self.server.writes += 1
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    requests_seen = []

    def post(session, url, **kwargs):
        request = kwargs['json']
        rows = request['state']['exploration']['page']['candidates']
        requests_seen.append(request)
        answers = {}
        for i, row in enumerate(rows):
            # Deliberately give the forbidden action the HIGHEST opportunity score.
            denied = row['name'] == 'Subscribe'
            if denied:
                assert row['method'].lower() == 'post'
                assert row['form_action'] == '/subscribe'
            answers[f'c{i}'] = dict(type='noul', noul=.99 if denied else .6)
            answers[f'e{i}'] = dict(type='noul', noul=.01 if denied else .99)
        return SimpleNamespace(status_code=200, json=lambda: dict(model='fixture', answers=answers, usage={}))

    monkeypatch.setattr(requests.Session, 'post', post)
    env = tmp_path / '.env'
    env.write_text('JEV_API_KEY=fixture-only', encoding='utf8')
    ranker = JevRankingClient(env, EventLog(tmp_path / 'ranking'), max_calls=10)
    descriptions = Model(tmp_path / 'descriptions')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.writes = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log = EventLog(tmp_path / 'graph')
    try:
        explorer = PipelinedExplorer(descriptions, log.directory, ranking_client=ranker)
        with Browser(f'http://127.0.0.1:{server.server_port}/', log, identity_mode='interaction') as browser:
            graph = crawl(browser, log.directory, log, CrawlBudget(max_states=3, max_depth=3,
                          actions_per_page=3, max_attempts=8), explorer=explorer)
        assert server.writes == 0
        assert len(graph.nodes) == 3
        assert [r['action']['name'] for r in graph.data['transition_records']] == ['Forums', 'Next']
        forum = next(n for n in graph.nodes.values() if n['url'].endswith('/forums'))
        sub = next(a for a in forum['actions'] if a['name'] == 'Subscribe')
        assert not graph.data['rankings'][forum['id']]['first_exploration'][sub['id']]['allowed']
        assert any('parent_pending_candidates' in q['state']['exploration'] or
                   q['state']['exploration']['page']['url'].endswith('/forums') for q in requests_seen)
        assert all(c[0] == 'transition_description' for c in descriptions.calls)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
