import copy
import hashlib
import json
import threading
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_function_cards import saved, CardModel
from web2kg.common.storage import Graph, write_json, EventLog
from web2kg.stage1.function_cards import build_cards, load_layer, LiveCards
from scripts import run_reddit_hard10_225 as runner


class ConcurrentCards(CardModel):
    def __init__(self, folder, fail=False):
        super().__init__(folder)
        self.shared = SimpleNamespace(barrier=threading.Barrier(3), active=0, peak=0,
                                      lock=threading.Lock(), count=0, closed=0)
        self.fail = fail

    def fork(self):
        child = copy.copy(self)
        child.session = SimpleNamespace(close=lambda: self.close_child())
        child.last_request_id = str(id(child))
        return child

    def close_child(self):
        with self.shared.lock:
            self.shared.closed += 1

    def json(self, *args, **kwargs):
        if args[3] == 'site_framework':
            return dict(overview='Fixture', areas=[dict(name='All', purpose='Browse',
                templates=[t['id'] for t in args[1]['templates']])], navigation=[], limitations=[])
        with self.shared.lock:
            self.shared.active += 1
            self.shared.count += 1
            number = self.shared.count
            self.shared.peak = max(self.shared.peak, self.shared.active)
        try:
            self.shared.barrier.wait(timeout=5)
            if self.fail and number == 1:
                raise RuntimeError('one failed request')
            return super().json(*args, **kwargs)
        finally:
            with self.shared.lock:
                self.shared.active -= 1


@pytest.mark.parametrize('fail', [False, True])
def test_parallel_cards_preserve_outputs_and_drain_failure(saved, tmp_path, fail):
    graph, path, _, _, _ = saved
    _, layer = load_layer(graph, path)
    assert len(layer['templates']) == 3
    client = ConcurrentCards(tmp_path / 'client', fail)
    target = tmp_path / 'parallel'
    if fail:
        with pytest.raises(RuntimeError, match='one failed'):
            build_cards(graph, path, target, client)
    else:
        parallel = build_cards(graph, path, target, client)
        serial = build_cards(graph, path, tmp_path / 'serial', CardModel(tmp_path / 'serial-model'))
        assert {k: r['card'] for k, r in parallel['cards'].items()} == {
            k: r['card'] for k, r in serial['cards'].items()}
    data = json.loads((target / 'cards.json').read_text(encoding='utf-8'))
    assert client.shared.peak == 3 and client.shared.active == 0 and client.shared.closed == 3
    assert sum(r['status'] == 'completed' for r in data['cards'].values()) == (2 if fail else 3)
    assert len({r['request_id'] for r in data['cards'].values()}) == 3
    # Existing failed/started rows cannot silently submit again.
    build_cards(graph, path, target, client)
    assert client.shared.count == 3


def test_graph_saved_fingerprint_matches_bytes_and_preserves_insertion_order(tmp_path):
    graph = Graph(tmp_path)
    graph.data['nodes'] = {'z': {'text': '中文\n'}, 'a': {}}
    actual = graph.save()
    raw = (tmp_path / 'graph.json').read_bytes()
    assert actual == hashlib.sha256(raw).hexdigest()
    assert raw == json.dumps(graph.data, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    assert list(Graph.load(tmp_path).nodes) == ['z', 'a']


@pytest.mark.parametrize('fail,limit', [(False, None), (True, None), (False, 2)])
def test_parallel_small_refresh_and_budget_preserve_all_outcomes(saved, tmp_path, fail, limit):
    source, path, _, _, _ = saved
    shutil.copytree(source.directory, tmp_path / 'copy')
    graph = Graph.load(tmp_path / 'copy')
    relative = path.relative_to(source.directory)
    _, layer = load_layer(graph, graph.directory / relative)
    write_json(graph.directory / 'templates/active.json', {'directory': relative.parent.as_posix()})
    client = ConcurrentCards(tmp_path / 'client', fail) if limit is None else CardModel(tmp_path / 'client')
    live = LiveCards(graph, layer, client, limit)
    result = live.finalize({g['id'] for g in layer['templates']}, [], True)
    assert live.calls == (3 if limit is None else 2)
    assert len(result['refreshed']) == (2 if fail or limit else 3)
    assert all(r['status'] != 'started' for r in live.data['cards'].values())
    if limit is None:
        assert client.shared.closed == 3 and client.shared.peak == 3
    assert live.stopped == fail


def test_global_parallel_cards_count_once_then_publish_framework(saved, tmp_path, monkeypatch):
    from test_visual_navigation import FakeJina
    from web2kg.stage2 import graph_update
    from web2kg.stage2.template_context import TemplateKnowledge
    base, _, _, cards, _ = saved
    log = EventLog(tmp_path / 'task')
    client = ConcurrentCards(tmp_path / 'client')
    updater = graph_update.ExecutionGraph(base, None, log, TemplateKnowledge(base, cards, log),
        card_client=client, index_client=FakeJina(), template_refresh_min=1, template_refresh_fraction=0)
    updater.layer['global_maintenance']['pending_templates'] = [updater.layer['templates'][0]['id']]
    monkeypatch.setattr(graph_update, 'regroup', lambda graph, layer, **kwargs: (
        copy.deepcopy(layer), {'status': 'proposed', 'changed': True}))
    result = updater.finish()
    assert result['template_global']['status'] == 'published'
    assert result['template_global']['calls'] == updater.live_cards.calls == 4
    assert client.shared.count == 3 and client.shared.closed == 3
    assert result['framework_refresh']['calls'] == 1
    assert len(TemplateKnowledge(updater.graph, updater.live_cards.directory / 'cards.json', log).cards) == 3


def test_scoped_verification_still_checks_live_inputs_and_full_checks_history(tmp_path):
    runtime = tmp_path / 'runtime'; runtime.mkdir()
    source = runtime / 'source.py'; source.write_text('fixed')
    current = tmp_path / 'current'; current.write_text('current')
    history = tmp_path / 'history'; history.write_text('history')
    write_json(runtime / 'manifest.json', {'source.py': runner.file_hash(source)})
    write_json(tmp_path / 'protocol.json', {'runtime': str(runtime), 'integrity_policy': 'boundary-v1'})
    write_json(tmp_path / 'runtime-inputs.json', {str(current): runner.file_hash(current)})
    write_json(tmp_path / 'protected-inputs.json', {str(current): runner.file_hash(current), str(history): runner.file_hash(history),
        str((tmp_path / 'runtime-inputs.json').resolve()): runner.file_hash(tmp_path / 'runtime-inputs.json')})
    write_json(tmp_path / 'state.json', {'status': 'registered'})
    runner.verify(tmp_path)
    history.write_text('changed')
    runner.verify(tmp_path, full=False)
    with pytest.raises(ValueError, match='Protected input changed'):
        runner.verify(tmp_path)
    original_scope = (tmp_path / 'runtime-inputs.json').read_bytes()
    write_json(tmp_path / 'runtime-inputs.json', {})
    with pytest.raises(ValueError, match='Runtime input manifest changed'):
        runner.verify(tmp_path, full=False)
    (tmp_path / 'runtime-inputs.json').write_bytes(original_scope)
    current.write_text('changed')
    with pytest.raises(ValueError, match='Protected input changed'):
        runner.verify(tmp_path, full=False)
    config = runner.read(tmp_path / 'protocol.json'); config.pop('integrity_policy')
    write_json(tmp_path / 'protocol.json', config)
    with pytest.raises(ValueError, match='Protected input changed'):
        runner.verify(tmp_path, full=False)  # Legacy still means full verification.


def test_speed_registration_uses_pre_task_input_not_success_output(tmp_path, monkeypatch):
    from scripts import register_hard_speed_check as registration
    previous = tmp_path / 'previous'; previous.mkdir()
    seed = previous / 'seed'; seed.mkdir()
    start = previous / 'starting'; start.mkdir()
    for folder in (seed, start):
        (folder / 'graph.json').write_text('original graph')
    config = {'prepared_graph': str(seed), 'task_ids': [735]}
    state = {'status': 'completed', 'tasks': {'735': {'status': 'completed'}},
             'current_graph': 'must-not-use-final735'}
    monkeypatch.setattr(registration.run, 'verify', lambda p: (config, state))
    write_json(previous / '735/input.json', {'graph': str(start)})
    write_json(previous / 'public-inputs.json', [{'task_id': 735, 'sites': ['reddit'], 'intent': 'public task', 'start_urls': ['http://localhost:10599']}])
    assert registration.selected_input(previous)[1] == seed
    (seed / 'graph.json').write_text('different state')
    with pytest.raises(ValueError, match='original task input'):
        registration.selected_input(previous)
