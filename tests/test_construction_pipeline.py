import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

from web2kg.common.activity import read_events
from web2kg.common.browser import Browser
from web2kg.common.replay import execute_path
from web2kg.common.responses import ResponsesError, object_schema
from web2kg.common.storage import EventLog, Graph
from web2kg.stage1.crawl import CrawlBudget, crawl
from web2kg.stage1.pipeline import PipelinedExplorer


class Model:
    identity = {'model': 'fixture', 'max_calls': 50}
    max_calls = 50

    def __init__(self, directory, delay=0, fail=False):
        self.log = EventLog(directory)
        self.calls, self.delay, self.fail = [], delay, fail
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.gates = {}

    def budget_used(self):
        return 0

    def stopped(self):
        return self.stop.is_set()

    def fork(self, stop_event=None):
        child = copy.copy(self)
        child.stop_event = stop_event
        return child

    def json(self, system, payload, schema, purpose, images=()):
        with self.lock:
            self.last_request_id = 'request-' + str(len(self.calls))
            self.calls.append((purpose, copy.deepcopy(payload), tuple(images), self.last_request_id))
        if purpose in self.gates:
            assert self.gates[purpose].wait(5)
        time.sleep(self.delay)
        if purpose == 'candidate_ranking':
            candidates = payload['page']['candidates']
            # Fixed fixture policy; does not inform production selection.
            priority = {'Catalog': 0, 'Help': 1, 'Shipping': 2, 'Returns': 3}
            ordered = sorted(candidates, key=lambda a: priority.get(a['name'], 10))
            return {'selected': [{'id': a['id'], 'group': a['name'], 'reason': 'Fixture'}
                for a in ordered[:payload['K']]]}
        if self.fail:
            raise ValueError('fixture annotation failure')
        return {'replayable': True, 'effect': 'Opened page', 'uncertainties': [],
                'page_descriptions': {side: side + ' page' for side in payload['page_descriptions_requested']}}


def fixture_graph(directory, count=3):
    graph = Graph(directory)
    graph.data.update(root='0', start_url='http://localhost/', budget={'max_depth': 3, 'actions_per_page': 2})
    action = {'id': 'open', 'kind': 'click', 'role': 'button', 'name': 'Open', 'nth': 0}
    for i in range(count):
        graph.add_node({'id': str(i), 'url': 'http://localhost/' + str(i), 'title': str(i),
                        'headings': [str(i)], 'actions': [action], 'text': 'Page content'}, [] if i == 0 else [action])
    images = []
    for name in ('before.png', 'after.png'):
        path = directory / name
        Image.new('RGB', (20, 20), 'white').save(path)
        images.append(path)
    return graph, action, images


def record(explorer, graph, action, images, target, attempt, new=True):
    edge = graph.add_edge('0', target, action)
    explorer.record(graph, graph.nodes['0'], action, graph.nodes[target], attempt, 'transition',
                    'capture_result', True, edge, new_state=new, images=images)
    return graph.data['edges'][-1]


def test_out_of_order_descriptions_claim_pages_once_and_main_thread_applies(tmp_path):
    graph, action, images = fixture_graph(tmp_path)
    client = Model(tmp_path)
    gate = threading.Event()
    client.gates['transition_description'] = gate
    explorer = PipelinedExplorer(client, tmp_path)
    try:
        first = record(explorer, graph, action, images, '1', 1)
        second = record(explorer, graph, action, images, '2', 2)
        assert first['replayable'] and graph.path('0', '1')
        assert graph.data['transition_records'][0]['page_description_requests'] == {'before': '0', 'after': '1'}
        assert graph.data['transition_records'][1]['page_description_requests'] == {'after': '2'}
        record(explorer, graph, action, images, '1', 3, new=False)
        assert first['replay_policy'] == 'construction-pending-v1'
        gate.set()
        for future, _, _ in explorer.pending.values():
            future.result(timeout=5)
        assert all('page_description' not in n for n in graph.nodes.values())
        explorer.poll(graph)
        assert all(n.get('page_description') for n in graph.nodes.values())
        assert first['description']['annotation']['status'] == second['description']['annotation']['status'] == 'completed'
        assert len(client.calls) == 2
        assert len({n['page_description']['request_id'] for n in graph.nodes.values()}) == 2
    finally:
        gate.set()
        explorer.close(graph)


@pytest.mark.parametrize('outcome', ['no_change', 'failed', 'not_attempted', 'transition'])
def test_known_or_unsuccessful_attempts_keep_facts_without_requests(tmp_path, outcome):
    graph, action, images = fixture_graph(tmp_path)
    client = Model(tmp_path)
    explorer = PipelinedExplorer(client, tmp_path)
    edge = graph.add_edge('0', '1', action) if outcome == 'transition' else None
    explorer.record(graph, graph.nodes['0'], action, graph.nodes['1'], 1, outcome, 'action',
                    outcome != 'not_attempted', edge, new_state=False)
    explorer.close(graph)
    assert not client.calls
    annotation = graph.data['transition_records'][0]['annotation']
    assert annotation == {'status': 'not_requested', 'reason': 'known_state' if edge else outcome}


def test_job_budget_and_missing_images_never_leave_provisional_edge(tmp_path):
    graph, action, images = fixture_graph(tmp_path)
    client = Model(tmp_path)
    client.max_calls = 1
    explorer = PipelinedExplorer(client, tmp_path)
    record(explorer, graph, action, images, '1', 1)
    with pytest.raises(ResponsesError, match='budget'):
        record(explorer, graph, action, images, '2', 2)
    explorer.close(graph, failed=True)
    assert not graph.data['edges'][-1]['replayable']
    assert len(client.calls) <= 1 and not explorer.claims


def test_missing_images_release_claims(tmp_path):
    graph, action, _ = fixture_graph(tmp_path)
    explorer = PipelinedExplorer(Model(tmp_path), tmp_path)
    with pytest.raises(RuntimeError, match='screenshots'):
        record(explorer, graph, action, [], '1', 1)
    explorer.close(graph, failed=True)
    assert not explorer.claims and not graph.data['edges'][0]['replayable']


def test_page_description_reused_by_cards_and_diagnosis_without_model(tmp_path):
    from web2kg.stage1.function_cards import page_summary
    from web2kg.stage2.recovery import Recovery
    from unittest.mock import Mock
    graph, action, images = fixture_graph(tmp_path)
    explorer = PipelinedExplorer(Model(tmp_path), tmp_path)
    record(explorer, graph, action, images, '1', 1)
    explorer.close(graph)
    summary = graph.nodes['0']['page_description']['text']
    assert page_summary(graph.nodes['0'])['page_description'] == summary
    summarizer = Mock(side_effect=AssertionError('Existing node description must be reused'))
    recovery = Recovery('Inspect page', graph, SimpleNamespace(), EventLog(tmp_path / 'recovery'),
                        SimpleNamespace(), page_summarizer=summarizer)
    recovery.start(graph.nodes['0'])
    recovery.prepare_pages()
    assert recovery.pages['0']['summary'] == summary
    summarizer.assert_not_called()


def test_ranking_client_is_independent_from_description_client(tmp_path):
    graph, action, images = fixture_graph(tmp_path)
    descriptions, ranking = Model(tmp_path / 'descriptions'), Model(tmp_path / 'ranking')
    ranking.identity = {'model': 'ranking-only', 'max_calls': 50}
    explorer = PipelinedExplorer(descriptions, tmp_path, ranking_client=ranking)
    explorer.select(graph, graph.nodes['0'], 1)
    record(explorer, graph, action, images, '1', 1)
    explorer.close(graph)
    assert [r[0] for r in ranking.calls] == ['candidate_ranking']
    assert [r[0] for r in descriptions.calls] == ['transition_description']
    assert explorer.identity['ranking_client']['model'] == 'ranking-only'
    assert explorer.reserved == 2


def test_jev_ranking_flows_through_real_pipeline(tmp_path, monkeypatch):
    from web2kg.stage1.jev_ranking import JevRankingClient
    graph, action, images = fixture_graph(tmp_path)
    env = tmp_path / '.env'
    env.write_text('api_key=fixture-secret\n', encoding='utf-8')
    descriptions = Model(tmp_path / 'descriptions')
    ranking = JevRankingClient(env, EventLog(tmp_path / 'ranking'))
    requests_sent = []

    def post(session, url, **kwargs):
        requests_sent.append(kwargs['json'])
        return SimpleNamespace(status_code=200, json=lambda: {'model': 'jev-test',
            'answers': {key: {'type': 'noul', 'noul': .8} for key in kwargs['json']['questions']},
            'usage': {'input_tokens': 12, 'output_tokens': 3}})

    monkeypatch.setattr(requests.Session, 'post', post)
    explorer = PipelinedExplorer(descriptions, tmp_path, ranking_client=ranking)
    selected = explorer.select(graph, graph.nodes['0'], 1)
    record(explorer, graph, action, images, '1', 1)
    explorer.close(graph)
    assert len(selected) == 1 and len(requests_sent) == 1
    assert [r[0] for r in descriptions.calls] == ['transition_description']
    assert graph.data['rankings']['0']['selected'][0]['group'].startswith('structure:')
    assert graph.nodes['1']['page_description']['text']
    assert explorer.reserved == 2


def test_late_rejection_checked_after_source_verification(tmp_path):
    graph, action, _ = fixture_graph(tmp_path)
    edge_id = graph.add_edge('0', '1', action)
    edge = graph.data['edges'][0]
    calls = []
    browser = SimpleNamespace(observe=lambda: graph.nodes['0'], act=lambda *args: calls.append(args))
    ticks = []
    def poll():
        ticks.append(True)
        if len(ticks) == 2:
            edge['replayable'] = False
    # Failure logging asks for live evidence; isolate it from the policy assertion.
    from unittest.mock import patch
    with patch('web2kg.common.replay.record_failure'):
        result = execute_path(browser, graph, [edge], EventLog(tmp_path), poll=poll)
    assert not result['success'] and not calls and edge['id'] == edge_id


def test_real_browser_prefetch_and_single_control_images(site, tmp_path):
    client = Model(tmp_path, delay=.15)
    explorer = PipelinedExplorer(client, tmp_path)
    with Browser(site, client.log, identity_mode='interaction', settle_ms=10) as browser:
        graph = crawl(browser, tmp_path, client.log, CrawlBudget(5, 3, 2, 1, 6), explorer=explorer)
    descriptions = [row for row in client.calls if row[0] == 'transition_description']
    assert descriptions
    assert all(len(images) == 2 and all(Path(p).is_file() for p in images) for _, _, images, _ in descriptions)
    for _, payload, images, _ in descriptions:
        metadata = json.loads((Path(images[0]).parent / 'annotations.json').read_text())
        assert len(metadata['items']) == 1
        assert payload['interaction_marker']['number'] == 1
        assert 'ax' not in payload and 'text' not in payload
    ranks = [row[1] for row in client.calls if row[0] == 'candidate_ranking']
    assert any(p.get('parent_pending_candidates') for p in ranks)
    assert graph.data['pipeline_summary']['pending_annotations'] == 0
    assert len(graph.data['transition_records']) == graph.data['attempts']
    events = read_events(client.log.path)
    child_start = next(e['time'] for e in events if e['event'] == 'pipeline_started' and
                       e['kind'] == 'ranking' and e['key'] != graph.data['root'])
    parent_last = max(e['time'] for e in events if e['event'] == 'exploration_action_started' and
                      e['source'] == graph.data['root'])
    assert child_start < parent_last


def test_final_drain_failure_keeps_attempt_and_graph(site, tmp_path):
    client = Model(tmp_path, delay=.05, fail=True)
    with Browser(site, client.log, identity_mode='interaction', settle_ms=10) as browser:
        with pytest.raises(RuntimeError, match='annotation failed'):
            crawl(browser, tmp_path, client.log, CrawlBudget(2, 2, 1, 1, 1),
                  explorer=PipelinedExplorer(client, tmp_path))
    graph = Graph.load(tmp_path)
    assert graph.data['attempts'] == 1
    assert graph.data['transition_records'][0]['annotation']['status'] == 'failed'
    assert not graph.data['edges'][0]['replayable']


def test_expand_completed_copy_skips_old_attempts_and_preserves_seed(site, tmp_path):
    import hashlib
    seed, out = tmp_path / 'seed', tmp_path / 'expanded'
    first = Model(seed)
    with Browser(site, first.log, identity_mode='interaction', settle_ms=10) as browser:
        old = crawl(browser, seed, first.log, CrawlBudget(2, 3, 2, 1, 5),
                    explorer=PipelinedExplorer(first, seed))
    assert len(old.nodes) == 2
    hashes = {p.relative_to(seed): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in seed.rglob('*') if p.is_file()}
    previous = {(r['source'], r['action']['id']) for r in old.data['transition_records']}
    descriptions = {k: n['page_description'] for k, n in old.nodes.items()}
    second = Model(out)
    with Browser(site, second.log, identity_mode='interaction', settle_ms=10) as browser:
        expanded = crawl(browser, out, second.log, CrawlBudget(4, 4, 3, 1, 10),
                         explorer=PipelinedExplorer(second, out), seed=seed)
    assert len(expanded.nodes) == 4 and set(old.nodes) < set(expanded.nodes)
    assert all(expanded.nodes[k]['page_description'] == value for k, value in descriptions.items())
    records = expanded.data['transition_records'][len(old.data['transition_records']):]
    assert records and all((r['source'], r['action']['id']) not in previous for r in records)
    assert min(r['attempt'] for r in records) > old.data['attempts']
    root_rank = next(payload for purpose, payload, _, _ in second.calls if purpose == 'candidate_ranking')
    assert not any((expanded.data['root'], a['id']) in previous for a in root_rank['page']['candidates'])
    assert hashes == {p.relative_to(seed): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in seed.rglob('*') if p.is_file()}
    assert read_events(out / 'events.jsonl')[-1]['reason'] == 'state_budget'


def test_expansion_rejects_original_directory(site, tmp_path):
    client = Model(tmp_path)
    with Browser(site, client.log, identity_mode='interaction', settle_ms=10) as browser:
        with pytest.raises(ValueError, match='independent new directory'):
            crawl(browser, tmp_path, client.log, seed=tmp_path)


def test_failed_annotation_recovered_from_saved_images_without_reclick(site, tmp_path):
    seed, out = tmp_path / 'failed', tmp_path / 'recover'
    first = Model(seed, fail=True)
    with Browser(site, first.log, identity_mode='interaction', settle_ms=10) as browser:
        with pytest.raises(RuntimeError, match='annotation failed'):
            crawl(browser, seed, first.log, CrawlBudget(2, 3, 1, 1, 1),
                  explorer=PipelinedExplorer(first, seed))
    before = (seed / 'graph.json').read_bytes()
    second = Model(out)
    with Browser(site, second.log, identity_mode='interaction', settle_ms=10) as browser:
        def forbidden(*args, **kwargs):
            pytest.fail('Recovered annotation must not repeat a browser action')
        browser.act = forbidden
        graph = crawl(browser, out, second.log, CrawlBudget(2, 3, 1, 1, 1),
            explorer=PipelinedExplorer(second, out), seed=seed, recover_annotations=True)
    assert graph.data['attempts'] == 1 and len(graph.data['transition_records']) == 1
    assert graph.data['transition_records'][0]['annotation']['status'] == 'failed'
    recovery = graph.data['annotation_recoveries'][0]
    assert recovery['annotation']['status'] == 'completed' and not recovery['browser_action_repeated']
    assert graph.data['edges'][0]['replayable']
    assert all(n.get('page_description') for n in graph.nodes.values())
    assert len(second.calls) == 1 and second.calls[0][0] == 'transition_description'
    assert (seed / 'graph.json').read_bytes() == before
    assert (out / 'transitions/recovery-00001.json').exists()


def test_changed_root_stops_after_first_failed_restore_with_evidence(site, tmp_path):
    from web2kg.common.replay import RootStateMismatch
    client = Model(tmp_path, delay=.05)
    with Browser(site, client.log, identity_mode='interaction', settle_ms=10) as browser:
        reset, act = browser.reset, browser.act
        changed = False

        def persistent_change(action, *args, **kwargs):
            nonlocal changed
            act(action, *args, **kwargs)
            changed = True

        def reset_changed_entry():
            reset()
            if changed:
                browser.page.evaluate("document.body.insertAdjacentHTML('beforeend', '<button>Persistent account state changed</button>')")

        browser.act, browser.reset = persistent_change, reset_changed_entry
        with pytest.raises(RootStateMismatch, match='State mismatch'):
            crawl(browser, tmp_path, client.log, CrawlBudget(10, 3, 3, 1, 10),
                  explorer=PipelinedExplorer(client, tmp_path))
    graph = Graph.load(tmp_path)
    assert graph.data['attempts'] == 2
    records = graph.data['transition_records']
    assert len(records) == 2 and records[0]['outcome'] == 'transition'
    assert records[0]['annotation']['status'] == 'completed'
    assert records[1]['outcome'] == 'not_attempted' and not records[1]['action_attempted']
    assert graph.data['pipeline_summary']['pending_annotations'] == 0
    events = read_events(client.log.path)
    assert sum(e['event'] == 'crawl_root_changed' for e in events) == 1
    assert not any(e['event'] == 'crawl_finished' for e in events)


def test_stop_prevents_queued_job_and_drains_inflight_evidence(tmp_path):
    graph, action, images = fixture_graph(tmp_path)
    client = Model(tmp_path)
    gate = threading.Event()
    client.gates['transition_description'] = gate
    explorer = PipelinedExplorer(client, tmp_path, description_workers=1)
    record(explorer, graph, action, images, '1', 1)
    deadline = time.monotonic() + 5
    while not client.calls and time.monotonic() < deadline:
        time.sleep(.01)
    record(explorer, graph, action, images, '2', 2)
    client.stop.set()
    gate.set()
    explorer.close(graph, failed=True)
    assert len(client.calls) == 1
    assert graph.data['transition_records'][0]['annotation']['status'] == 'completed'
    assert graph.data['transition_records'][1]['annotation']['status'] == 'failed'
    assert not explorer.pending


def test_qwen_forks_keep_http_request_ids_and_events_separate(tmp_path, monkeypatch):
    from scripts.qwen_method_transport import clients
    env = tmp_path / 'fixture.env'
    env.write_text('CRAWL_API_KEY=test\nCRAWL_BASE_URL=https://unused.test\nCRAWL_MODEL=unused\n'
        'OPENROUTER_API_KEY=test\nOPENROUTER_BASE_URL=https://free.test/v1\nOPENROUTER_MODEL=free\n'
        'BAILIAN_API_KEY=test\nBAILIAN_BASE_URL=https://paid.test/v1\nBAILIAN_MODEL=paid\n')
    parent = clients(env, tmp_path / 'route.json', provider='bailian')(env, EventLog(tmp_path / 'calls'), 2, 1024)
    children = [parent.fork() for _ in range(2)]
    assert children[0].session.http is not children[1].session.http
    assert all(c.session.client is c and c.session.http.trust_env is False for c in children)
    barrier = threading.Barrier(2)
    def post(self, url, **kwargs):
        barrier.wait(5)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({'model': 'paid', 'choices': [{'finish_reason': 'stop',
            'message': {'content': '```json\n{"ok":true}\n```'}}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 4}}).encode()
        return response
    monkeypatch.setattr(requests.Session, 'post', post)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda c: c.json('test', {}, object_schema({'ok': {'type': 'boolean'}}), 'fixture'), children))
    assert results == [{'ok': True}] * 2
    assert len({c.last_request_id for c in children}) == 2 and not hasattr(parent, 'last_request_id')
    assert len([e for e in read_events(parent.log.path) if e['event'] == 'responses_output_parsed']) == 2
    assert all(e['json_fence_removed_messages'] == [0] for e in read_events(parent.log.path)
               if e['event'] == 'responses_output_parsed')
    for c in children:
        c.session.http.close()
