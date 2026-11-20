import importlib
import json
import re
import threading
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from web2kg.common.storage import EventLog, Graph
from web2kg.evaluation.actor_output import actor_output, single_action
from web2kg.stage2.agent import load_runtime, run_task


@pytest.fixture
def search_class():
    load_runtime()
    return importlib.import_module('webNavigator.ax_search').AXSearch


def lines(n=5000):
    return [f"button [{i + 10000}] 'Control {i}'" for i in range(n)]


def test_windows_overlap_by_source_position_and_keep_ids(search_class):
    source = lines()
    source[900] = "button [10900] '目标'"
    source[1000] = source[900]  # Same text at another position is not the same source line.
    state = search_class('\n'.join(source))
    assert len(state.selected) == 800
    result = state.search('search_ax [目标]')
    assert result['match_line'] == 901
    assert len(state.selected) == 1300
    assert state.text == '\n'.join(source[:1300])
    assert state.text.count("button [10900] '目标'") == 2
    assert state.ranges() == [[1, 1300]]


def test_three_searches_and_hard_2500_line_cap(search_class):
    source = lines()
    for position, keyword in [(1500, 'first'), (3000, 'second'), (4500, 'third')]:
        source[position] = f"button [{position}] '{keyword}'"
    state = search_class('\n'.join(source))
    for keyword, count in [('first', 1600), ('second', 2400), ('third', 2500)]:
        assert state.search(f'search_ax [{keyword}]')['status'] == 'found'
        assert len(state.selected) == count
    assert 4500 in state.selected
    assert state.searches == 3 and not state.available
    before = state.text
    assert state.search('search_ax [Control]')['status'] == 'blocked'
    assert state.text == before and state.searches == 3


def test_misses_invalid_commands_and_literal_matching_consume_budget(search_class):
    state = search_class('\n'.join(lines()))
    for command, status in [('search_ax [.*]', 'not_found'),
                            ('search_ax []', 'invalid'),
                            ('search_ax [Control]\nclick [10000]', 'invalid')]:
        assert state.search(command)['status'] == status
    assert not state.available and len(state.selected) == 800


def test_repeated_keyword_reaches_next_unseen_hit(search_class):
    source = lines()
    source[100] = "button [10100] 'Subscribe'"
    source[1800] = "button [11800] 'SUBSCRIBE'"
    state = search_class('\n'.join(source))
    assert state.search('search_ax [subscribe]')['match_line'] == 1801
    assert "[11800]" in state.text


def test_short_page_and_edge_window(search_class):
    state = search_class("button [1] '最后'")
    assert state.search('search_ax [最后]')['added_lines'] == 0
    assert state.ranges() == [[1, 1]]
    large = search_class('\n'.join(lines(1900)))
    assert large.search('search_ax [Control 1899]')['window'] == [1101, 1900]
    assert len(large.selected) == 1600


def test_new_observation_resets_without_merging_old_ids(search_class):
    old = search_class('\n'.join(lines()))
    old.search('search_ax [Control 4000]')
    new = search_class("button [42] 'Fresh page'")
    assert new.searches == 0 and new.text == "button [42] 'Fresh page'"
    assert '14000' not in new.text


def test_bare_search_transport_adaptation():
    assert single_action('search_ax [Subscribe]')
    assert not single_action('search_ax [Subscribe]\nclick [1]')
    assert not single_action('search_ax []')
    text, changes = actor_output([], lambda _: 'search_ax [Subscribe]')
    assert text == 'ACTION:\nsearch_ax [Subscribe]' and changes == ['bare_action_heading']


@pytest.mark.parametrize('misses', [False, True])
@pytest.mark.parametrize('budgeted', [False, True])
def test_real_actor_search_is_not_browser_step_and_limits_survive_retries(tmp_path, misses, budgeted):
    html = ('<!doctype html><title>AX search fixture</title>' +
            ''.join(f'<button>Control {i}</button>' for i in range(1900)) +
            '<button onclick="this.textContent=\'Subscribed\'">Subscribe</button>').encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/'
    calls = []

    class Actor:
        def call(self, messages, system_prompt='', **kwargs):
            assert len(calls) < 6
            prompt = next(m['content'] for m in messages if 'CURRENT OBSERVATION:' in str(m['content']))
            if isinstance(prompt, list):
                prompt = next(p['text'] for p in prompt if p.get('type') == 'text')
            current = prompt.split('CURRENT OBSERVATION:\n', 1)[1].split('\nCURRENT VISUAL OBSERVATION:', 1)[0]
            ax, status = current.split('\n\nAX_SEARCH_STATUS (not AX nodes):\n', 1)
            state = yaml.safe_load(status)
            calls.append(state)
            assert len(ax.splitlines()) == state['shown_lines']
            if misses:
                if len(calls) <= 3:
                    return 'ACTION: search_ax [missing-keyword]'
                assert not state['search_available'] and state['searches_used'] == 3
                assert '- search_ax [keyword]' not in system_prompt
                if len(calls) == 4:
                    return 'ACTION: search_ax [Subscribe]'  # Must be rejected without budget reset.
                return 'ACTION: stop [not found]'
            if len(calls) == 1:
                assert state['shown_lines'] == 800 and not re.search(r"button \[\d+\] 'Subscribe'", ax)
                return 'ACTION: search_ax [Subscribe]'
            if len(calls) == 2:
                match = re.search(r"button \[(\d+)\] 'Subscribe'", ax)
                assert match and state['searches_used'] == 1 and state['shown_lines'] <= 1600
                return f'ACTION: click [{match[1]}]'
            assert state['searches_used'] == 0 and state['shown_lines'] == 800
            return 'ACTION: stop [done]'

    log = EventLog(tmp_path / 'run')
    model = Actor()
    if budgeted:
        from web2kg.evaluation.webnavigator import OriginalResponses
        env_file = tmp_path / 'fixture.env'
        env_file.write_text('CRAWL_API_KEY=fixture\nCRAWL_BASE_URL=https://unused.invalid\nCRAWL_MODEL=fixture\n')
        model = OriginalResponses(env_file, log, 2, verified_protocol=False)
        model.configure_output_retries(3)
        def post(*args, **kwargs):
            request = kwargs['json']
            messages = [{'role': m['role'], 'content': [
                {'type': 'text', 'text': x['text']} for x in m['content'] if x['type']=='input_text']}
                for m in request['input']]
            output = Actor().call(messages, system_prompt=request['instructions'])
            return SimpleNamespace(ok=True, json=lambda: {'status':'completed','model':'fixture',
                'usage':{'input_tokens':10,'output_tokens':2}, 'output':[{'type':'message',
                'status':'completed','content':[{'type':'output_text','text':output}]}]})
        model.session.post = post
    try:
        graph = Graph(tmp_path / 'graph')
        graph.data['start_url'] = url
        result = run_task(graph, 'Click Subscribe', log, model, None, start_url=url,
                          use_graph=False, framework=None, allow_writes=True, max_steps=3,
                          final_check=lambda browser: browser.page.get_by_role('button', name='Subscribed', exact=True).count() == 1)
        assert result['status'] == 'agent_stopped', result
        assert result['steps'] == result['atomic_actions'] == (1 if misses else 2)
        assert result['local_success'] is (not misses)
        events = [json.loads(line) for line in (log.directory / 'upstream-log/ax-search/events.jsonl').read_text(encoding='utf-8').splitlines()]
        searches = [e for e in events if e['event'] == 'search']
        assert len(searches) == ((3 if budgeted else 4) if misses else 1)
        if misses:
            if not budgeted:
                assert searches[-1]['result']['status'] == 'blocked'
            assert len({e['observation'] for e in searches}) == 1
        if budgeted:
            assert result['action_budget']['used'] == (1 if misses else 2)
            assert result['action_budget']['ax_search_requests'] == (3 if misses else 1)
        assert all(e['browser_step'] == 0 for e in searches)
        assert all((log.directory / 'upstream-log/ax-search' / e['view']).exists() for e in events)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
