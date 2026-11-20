import importlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from web2kg.common.storage import EventLog, Graph
from web2kg.stage2.agent import load_runtime, run_task
from web2kg.stage2.observation import install_numeric_text_guard, is_numeric_text


@pytest.mark.parametrize('text', ['4', '0', '-2', '+2', '4.5', '.5', '12%', '12 %',
    '1,200', '1,200.50', '￥99', '$-99.50', '-$99', '99€', '−2', ' １２％ ', '١٢'])
def test_standalone_values(text):
    assert is_numeric_text(text)


@pytest.mark.parametrize('text', ['', ' ', 't3_11wjn49', '4 years ago', 'Title',
    'Chapter 4', 'Ann', '1,20', '1.2.3', '--2', '$', '%', '12/34', 'v4.5'])
def test_other_text_keeps_original_policy(text):
    assert not is_numeric_text(text)


@pytest.fixture
def runtime():
    load_runtime()
    opt = importlib.import_module('webNavigator.obs_opt')
    Node = importlib.import_module('browser_env.processors').TreeNode
    return opt, Node


def make_tree(Node, value, sibling='t3_11wjn49', parent_name=''):
    parent = Node(1, 'article', parent_name, 0)
    for n in [Node(2, 'StaticText', sibling, 1), Node(3, 'button', 'Upvote', 1),
              Node(4, 'StaticText', value, 1), Node(5, 'button', 'Downvote', 1)]:
        parent.add_child(n)
    return parent


@pytest.mark.parametrize('value', ['4', '-2', '4.5', '12%', '1,200', '￥99', '+2', '−2'])
def test_full_pruning_keeps_numeric_text_and_original_symbols(runtime, value):
    opt, Node = runtime
    root = make_tree(Node, value, sibling='ID: ' + value, parent_name='Summary: ' + value)
    restore = install_numeric_text_guard()
    try:
        pruned = opt.prune_tree('', root, mode='node')
        target = pruned.search_node_by_id(4)
        assert target is not None and target.name == value
        assert f'text {value!r}' in opt.prune_tree('', root)
    finally:
        restore()
    assert root.children[2].visible and root.children[2].name == value


def test_sibling_and_parent_labels_still_deduplicate(runtime):
    opt, Node = runtime
    cases = [('Search query', 'Search query'), ('Body', 'Body'), ('URL', 'URL'),
             ('Title ', 'Title This field is required.')]
    restore = install_numeric_text_guard()
    try:
        for label, control_name in cases:
            root = Node(1, 'RootWebArea', '', 0)
            root.add_child(Node(2, 'StaticText', label, 1))
            root.add_child(Node(3, 'textbox', control_name, 1))
            output = opt.prune_tree('', root, mode='node')
            assert output.search_node_by_id(2) is None
            assert output.search_node_by_id(3) is not None
        button = Node(4, 'button', 'Submit', 0)
        button.add_child(Node(5, 'StaticText', 'Submit', 1))
        assert opt.prune_tree('', button, mode='node').children == []
    finally:
        restore()


def test_upstream_behavior_restored_even_after_exception(runtime):
    opt, Node = runtime
    clean, deduplicate = opt.action_remove_unwanted_characters, opt.action_remove_redundant_statictext_node
    assert opt.prune_tree('', make_tree(Node, '4'), mode='node').search_node_by_id(4) is None
    with pytest.raises(RuntimeError):
        restore = install_numeric_text_guard()
        try:
            assert opt.prune_tree('', make_tree(Node, '4'), mode='node').search_node_by_id(4) is not None
            raise RuntimeError('fixture failure')
        finally:
            restore()
    assert opt.action_remove_unwanted_characters is clean
    assert opt.action_remove_redundant_statictext_node is deduplicate
    assert opt.prune_tree('', make_tree(Node, '4'), mode='node').search_node_by_id(4) is None


@pytest.mark.parametrize('fail', [False, True])
def test_real_actor_receives_values_and_task_restores_upstream(tmp_path, runtime, fail):
    opt, _ = runtime
    clean, deduplicate = opt.action_remove_unwanted_characters, opt.action_remove_redundant_statictext_node
    values = ['4', '-2', '4.5', '12%', '1,200', '￥99']
    body = ('<!doctype html><title>Numeric observations</title>' + ''.join(
        f'<article><span>ID: {v}</span><button>Upvote</button><span>{v}</span><button>Downvote</button></article>'
        for v in values) + '<label for="body">Body</label><input id="body">').encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/'
    received = []

    class Client:
        def call(self, messages, **kwargs):
            prompt = next(m['content'] for m in messages if 'CURRENT OBSERVATION:' in str(m['content']))
            if isinstance(prompt, list):
                prompt = next(p['text'] for p in prompt if p.get('type') == 'text')
            current = prompt.split('CURRENT OBSERVATION:\n', 1)[1].split('\nCURRENT VISUAL OBSERVATION:', 1)[0]
            received.append(current)
            for value in values:
                assert f'text {value!r}' in current
            assert re.search(r"textbox \[\d+\] 'Body'", current)
            assert "text 'Body'" not in current
            if fail:
                raise RuntimeError('fixture task error')
            return 'ACTION: stop [done]'

    try:
        graph = Graph(tmp_path / 'graph')
        graph.data['start_url'] = url
        log = EventLog(tmp_path / 'run')
        result = run_task(graph, 'Read the numeric values', log, Client(), None,
                          start_url=url, use_graph=False, framework=None, max_steps=1)
        assert len(received) == 1
        assert result['status'] == ('runtime_error' if fail else 'agent_stopped'), result
        if fail:
            assert 'fixture task error' in result['error'], result
        cfg = json.loads((log.directory / 'execution-config.json').read_text(encoding='utf-8'))
        assert cfg['env']['numeric_text_policy'] == 'standalone-value-v1'
        assert opt.action_remove_unwanted_characters is clean
        assert opt.action_remove_redundant_statictext_node is deduplicate
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
