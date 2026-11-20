import json
import importlib.util
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from playwright.sync_api import sync_playwright, BrowserType

from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog, Graph
from web2kg.stage1.indexing import build_index
from web2kg.visualization.control import ReviewController, create_control_server
from web2kg.visualization.viewer import export_viewer
from web2kg.visualization.manual import serve_manual


@contextmanager
def portal(root, site):
    graph = Graph(root / 'runs' / 'graph')
    with Browser(site, EventLog(graph.directory)) as browser:
        node = browser.observe(graph.directory)
    graph.data.update(root=node['id'], start_url=site)
    graph.add_node(node, [])
    graph.save()
    build_index(graph)
    export_viewer(graph)
    (root / 'docs').mkdir()
    (root / 'docs' / 'usage.md').write_text('# Usage\n\nCurrent project guide.', encoding='utf-8')
    (root / '.env').write_text('PRIVATE=not-served', encoding='utf-8')
    controller = ReviewController(graph, root, port=0, headless=True)
    server = create_control_server(graph, root, controller, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', controller
    finally:
        server.shutdown()
        server.server_close()
        controller.close()
        thread.join()


def get(url):
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def post(url, token, origin=None):
    headers = {'Content-Type': 'application/json', 'X-Control-Token': token}
    if origin:
        headers['Origin'] = origin
    with urlopen(Request(url, data=b'{}', headers=headers), timeout=5) as response:
        return json.load(response)


def wait_state(controller, expected):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = controller.snapshot()
        if state['state'] == expected and (expected != 'stopped' or state['pid'] is None):
            return state
        time.sleep(.1)
    pytest.fail(str(controller.snapshot()))


def test_portal_starts_one_worker_and_stops_browser(site, tmp_path):
    with portal(tmp_path, site) as (url, controller), sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 1050})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.locator('#start').click()
        started = controller.snapshot()
        controller.start()
        assert controller.snapshot()['pid'] == started['pid']
        page.wait_for_function("document.querySelector('#review-state').dataset.state === 'running'", timeout=30000)
        running = controller.snapshot()
        assert get(running['url'] + '/api/nodes')[0]['index_text']
        assert page.locator('#review-link').get_attribute('href') == running['url']
        for selector in ('#graph-link', '#logs-link', '#docs-link'):
            with urlopen(url + page.locator(selector).get_attribute('href'), timeout=5) as response:
                assert response.status == 200
        with urlopen(url + '/docs/usage.md') as response:
            assert response.headers.get_content_type() == 'text/html'
            assert b'<h1>Usage</h1>' in response.read()
        page.screenshot(path=str(tmp_path / 'portal.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.locator('#stop').click()
        stopped = wait_state(controller, 'stopped')
        assert controller.process.exitcode == 0
        assert (controller.root / stopped['directory'] / 'manual_review.json').exists()
        with pytest.raises(OSError):
            urlopen(running['url'] + '/api/state', timeout=2)
        assert not errors
        old_pid = running['pid']
        page.locator('#start').click()
        restarted = wait_state(controller, 'running')
        assert restarted['pid'] != old_pid
        assert restarted['directory'] != stopped['directory']
        assert get(restarted['url'] + '/api/nodes')
        controller.stop()
        wait_state(controller, 'stopped')
        browser.close()


def test_updated_portal_links_work_with_existing_server_state(site, tmp_path):
    with portal(tmp_path, site) as (url, controller), sync_playwright() as p:
        state = get(url + '/api/state')
        state['links'].pop('docs')
        state['links']['progress'] = '/docs/removed-journal.html'
        browser = p.chromium.launch()
        page = browser.new_page()
        page.route('**/api/state', lambda route: route.fulfill(json=state))
        page.goto(url)
        page.wait_for_function("document.querySelector('#connection').textContent === '本地服务已连接'")
        assert page.locator('#docs-link').get_attribute('href') == '/docs/usage.md'
        assert not page.locator('#progress-link').count()
        assert controller.snapshot()['state'] == 'idle'
        browser.close()


@pytest.mark.parametrize('window', ['target', 'debug'])
def test_closing_either_window_ends_session_and_preserves_portal(site, tmp_path, monkeypatch, window):
    instances = []
    launch = BrowserType.launch

    def capture(self, *args, **kwargs):
        browser = launch(self, *args, **kwargs)
        instances.append(browser)
        return browser

    with portal(tmp_path, site) as (url, controller):
        monkeypatch.setattr(BrowserType, 'launch', capture)
        review_url = []

        def close_window(address):
            review_url.append(address)
            assert get(address + '/api/nodes')
            instances[0 if window == 'target' else 1].contexts[0].pages[0].close()

        directory = tmp_path / 'window-close'
        serve_manual(controller.graph, directory, port=0, headless=False, on_ready=close_window)
        assert all(not browser.is_connected() for browser in instances)
        assert (directory / 'manual_review.json').is_file()
        assert 'manual_window_closed' in (directory / 'events.jsonl').read_text(encoding='utf-8')
        with pytest.raises(OSError):
            urlopen(review_url[0] + '/api/state', timeout=2)
        assert get(url + '/api/state')['graph']['nodes'] == 1


def test_double_launch_reopens_existing_portal_without_new_worker(site, tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/start_control.py'
    spec = importlib.util.spec_from_file_location('start_control', path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    with portal(tmp_path, site) as (url, controller):
        opened = []
        monkeypatch.setattr(launcher.webbrowser, 'open', opened.append)
        monkeypatch.setattr(sys, 'argv', ['start_control', '--graph', str(controller.graph.directory),
                                        '--port', url.rsplit(':', 1)[1]])
        launcher.main()
        assert opened == [url]
        assert controller.snapshot()['state'] == 'idle'
        index = controller.graph.directory / 'index.json'
        data = json.loads(index.read_text(encoding='utf-8'))
        data['model'] = 'new-local-index'
        index.write_text(json.dumps(data), encoding='utf-8')
        assert get(url + '/api/state')['graph']['index'] == 'new-local-index'


def test_portal_rejects_unsafe_requests_and_reports_start_failures(site, tmp_path):
    with portal(tmp_path, site) as (url, controller):
        token = get(url + '/api/state')['token']
        for route in ('/.env', '/auth/site.json', '/graph/%2e%2e/%2e%2e/.env', '/docs/%2e%2e/.env'):
            with pytest.raises(HTTPError) as error:
                urlopen(url + route)
            assert error.value.code == 404
        for key, origin in [('wrong', url), (token, 'https://example.com')]:
            with pytest.raises(HTTPError) as error:
                post(url + '/api/review/start', key, origin)
            assert error.value.code == 403
        controller.storage_state = str(tmp_path / 'missing-auth.json')
        with pytest.raises(HTTPError) as error:
            post(url + '/api/review/start', token)
        assert error.value.code == 400
        controller.storage_state = None
        controller.port = int(url.rsplit(':', 1)[1])
        post(url + '/api/review/start', token)
        failed = wait_state(controller, 'failed')
        assert failed['message']
        with urlopen(url + '/api/review-log') as response:
            assert b'OSError' in response.read()
        # An occupied review port must not stop the unrelated portal server.
        assert get(url + '/api/state')['graph']['nodes'] == 1
