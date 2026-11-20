import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import sync_playwright, Error

from web2kg.stage2.transport import LocalSiteProxy
from web2kg.common.storage import EventLog, Graph
from web2kg.stage2.agent import run_task


LOGICAL = 'http://127.0.0.1:65521'


@pytest.fixture
def redirect_site():
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def handle_request(self):
            body = self.rfile.read(int(self.headers.get('Content-Length', 0))).decode()
            seen.append((self.command, self.path, body, self.headers.get('Cookie', ''), self.headers.get('X-Fixture-Route')))
            locations = {'/submit': (302, '/middle'), '/middle': (303, '/finish'),
                         '/absolute': (302, 'http://' + self.headers['Host'] + '/finish'),
                         '/keep': (307, '/finish'), '/leave': (302, 'http://example.invalid/'),
                         '/unsafe': (302, '/delete')}
            if self.path in locations:
                status, target = locations[self.path]
                self.send_response(status)
                self.send_header('Location', target)
                self.send_header('Set-Cookie', 'first=one; Path=/; HttpOnly')
                self.send_header('Set-Cookie', 'second=two; Path=/')
            else:
                self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>Finished</h1>' if self.path == '/finish' else
                             b'<h1>Form</h1><form method="post" action="/submit"><input name="title" value="fixture"><button>Save</button></form>')
        do_GET = do_POST = handle_request
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    yield f'http://127.0.0.1:{server.server_port}', seen
    server.shutdown(); server.server_close(); thread.join()


def test_reproduces_original_route_redirect_escape(redirect_site):
    backend, seen = redirect_site
    called = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        def original_relay(route):
            called.append(urlsplit(route.request.url).path)
            response = route.fetch(url=backend + urlsplit(route.request.url).path, max_redirects=0)
            route.fulfill(response=response)
        context.route('**/*', original_relay)
        page = context.new_page(); page.goto(LOGICAL)
        page.get_by_role('button', name='Save').click()
        page.wait_for_timeout(200)
        assert page.url.startswith('chrome-error:')
        assert '/submit' in called and '/middle' not in called
        assert not any(row[1] == '/middle' for row in seen)
        browser.close()


def test_proxy_all_redirect_hops_cookies_body_scope_and_har(redirect_site, tmp_path):
    backend, seen = redirect_site
    proxy = LocalSiteProxy({'backend': backend, 'allowed_origin': LOGICAL, 'allow_writes': True})
    proxy.context = SimpleNamespace(web2kg_read_only=False)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(proxy={'server': proxy.url, 'bypass': '<-loopback>'},
                                        args=['--proxy-bypass-list=<-loopback>'])
            context = browser.new_context(record_har_path=str(tmp_path/'test.har'), service_workers='block')
            page = context.new_page(); page.goto(LOGICAL)
            page.get_by_role('button', name='Save').click()
            page.wait_for_url(LOGICAL + '/finish')
            assert page.locator('h1').inner_text() == 'Finished'
            assert ('POST', '/submit', 'title=fixture') in [r[:3] for r in seen]
            last = next(r for r in seen if r[1] == '/finish')
            assert last[0] == 'GET' and 'first=one' in last[3] and 'second=two' in last[3]
            assert page.evaluate("fetch('/keep',{method:'POST',body:'kept'}).then(r=>r.status)") == 200
            assert any(r[:3] == ('POST', '/finish', 'kept') for r in seen)
            page.goto(LOGICAL + '/absolute')
            assert page.url == LOGICAL + '/finish'
            assert page.locator('h1').inner_text() == 'Finished'
            # A different origin is relayed normally too (local fixture, no external network).
            page.goto(backend + '/finish')
            assert page.locator('h1').inner_text() == 'Finished'
            page.goto(LOGICAL + '/finish')
            proxy.context.web2kg_read_only = True
            assert page.evaluate("fetch('/submit',{method:'POST'}).then(r=>r.status)") == 200
            assert page.evaluate("fetch('/unsafe').then(r=>r.status)") == 200
            assert any(r[1] == '/delete' for r in seen)
            context.close(); browser.close()
    finally:
        proxy.close()
    entries = json.loads((tmp_path/'test.har').read_text(encoding='utf-8'))['log']['entries']
    assert any(e['request']['method'] == 'POST' and e['response']['status'] == 302 for e in entries)
    assert any(e['request']['url'] == LOGICAL + '/finish' and e['response']['status'] == 200 for e in entries)
    assert any(e['request']['url'].startswith(backend) for e in entries)


def test_upstream_uses_proxy_through_submit(redirect_site, tmp_path):
    backend, seen = redirect_site
    class Client:
        def call(self, *a, **kw):
            return 'REASON: Test.\nACTION: stop [done]'
    def check(browser):
        browser.page.get_by_role('button', name='Save').click()
        browser.page.wait_for_url(LOGICAL + '/finish')
        return browser.page.locator('h1').inner_text() == 'Finished'
    result = run_task(Graph(tmp_path/'graph'), 'Test', EventLog(tmp_path/'run'), Client(), None,
                      use_graph=False, start_url=LOGICAL, backend=backend, allow_writes=True, final_check=check)
    assert result['status'] == 'agent_stopped', result
    assert result['local_success'] and result['url'] == LOGICAL + '/finish'


def test_connect_and_external_http_relay_with_origin_scoped_headers(redirect_site):
    import http.client
    import requests
    backend, seen = redirect_site
    proxy = LocalSiteProxy({'backend': backend, 'allowed_origin': LOGICAL,
                            'extra_http_headers': {'X-Fixture-Route': 'local-only'}})
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.proxies = {'http': proxy.url}
            assert session.get(LOGICAL + '/logical-headers', timeout=5).status_code == 200
            assert session.get(backend + '/external-headers', timeout=5).status_code == 200
        assert next(r for r in seen if r[1] == '/logical-headers')[4] == 'local-only'
        assert next(r for r in seen if r[1] == '/external-headers')[4] is None
        connection = http.client.HTTPConnection(urlsplit(proxy.url).hostname, urlsplit(proxy.url).port, timeout=5)
        connection.set_tunnel(urlsplit(backend).hostname, urlsplit(backend).port)
        connection.request('GET', '/tunneled')
        assert connection.getresponse().status == 200
        connection.close()
        assert any(r[1] == '/tunneled' for r in seen)
    finally:
        proxy.close()
