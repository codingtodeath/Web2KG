import contextlib
from pathlib import Path

import pytest

from scripts import run_reddit_verified_comparison as run


def test_complete_task_order_uses_metadata_and_public_inputs():
    tasks = run.task_order()
    assert len(tasks) == len({t['task_id'] for t in tasks}) == 106
    assert [t['difficulty'] for t in tasks] == ['non-hard'] * 64 + ['hard'] * 42
    assert [t['task_id'] for t in tasks[:64]] == sorted(t['task_id'] for t in tasks[:64])
    assert [t['task_id'] for t in tasks[64:]] == sorted(t['task_id'] for t in tasks[64:])
    assert 724 in {t['task_id'] for t in tasks}
    public = run.public_tasks(tasks, 'http://localhost:10599')
    assert all(set(t) == {'task_id', 'sites', 'intent', 'start_urls'} for t in public)
    assert all('__REDDIT__' not in u for t in public for u in t['start_urls'])


@pytest.mark.parametrize('finish,score,kind,expected', [
    ('length', 0, None, True), ('length', 1, None, True),
    ('stop', 0, None, False), ('length', None, None, False),
    ('length', 0, 'transport', False)])
def test_only_received_scored_length_failure_can_continue(tmp_path, finish, score, kind, expected):
    run.write_json(tmp_path / 'result.json', dict(status='model_error', error='ResponsesError: Incomplete response'))
    run.write_json(tmp_path / 'official.json', dict(score=score))
    run.write_json(tmp_path / 'llm/request/chat-response.json', dict(choices=[dict(finish_reason=finish)]))
    run.EventLog(tmp_path).emit('responses_received', request_id='request')
    run.EventLog(tmp_path).emit('responses_failed', request_id='request', http_status=None, failure_kind=kind)
    assert run.model_output_failure(tmp_path) is expected


@pytest.mark.parametrize('index_status,calls,card_stopped,expected', [
    ('completed', 0, True, True), ('completed', 0, False, True),
    ('failed', 0, True, False), ('completed', 1, True, False)])
def test_method_length_failure_requires_no_independent_asset_error(tmp_path, index_status, calls, card_stopped, expected):
    run.write_json(tmp_path / 'result.json', dict(status='model_error', error='ResponsesError: Incomplete response',
        execution_graph=dict(index_status=index_status, cards=dict(calls=calls, stopped=card_stopped))))
    run.write_json(tmp_path / 'official.json', dict(score=0))
    run.write_json(tmp_path / 'llm/request/chat-response.json', dict(choices=[dict(finish_reason='length')]))
    run.EventLog(tmp_path).emit('responses_received', request_id='request')
    run.EventLog(tmp_path).emit('responses_failed', request_id='request', http_status=None, failure_kind=None)
    assert run.model_output_failure(tmp_path, method_assets=True) is expected


@pytest.mark.parametrize('kind,purpose,received,http_status,expected', [
    ('output:actor_action_invalid', 'original_actor', True, None, True),
    ('output:actor_action_invalid', 'original_actor', False, None, False),
    ('output:actor_action_invalid', 'original_actor', True, 401, False),
    ('transport', 'original_actor', True, None, False),
    ('output:card_invalid', 'template_card', True, None, False)])
def test_scored_typed_actor_output_failures_are_terminal(tmp_path, kind, purpose, received, http_status, expected):
    run.write_json(tmp_path / 'result.json', dict(status='model_error', error='ResponsesError: invalid output',
        provider_failure=dict(kind=kind)))
    run.write_json(tmp_path / 'official.json', dict(score=0))
    log = run.EventLog(tmp_path)
    if received:
        log.emit('responses_received', request_id='request')
    log.emit('responses_failed', request_id='request', purpose=purpose, http_status=http_status, failure_kind=kind)
    assert run.model_output_failure(tmp_path) is expected


@pytest.mark.parametrize('http_status,code,expected', [
    (400, 'data_inspection_failed', True), (401, 'data_inspection_failed', False),
    (400, 'invalid_api_key', False), (402, 'insufficient_quota', False)])
def test_inspection_rejection_is_scoped_to_one_scored_request(tmp_path, http_status, code, expected):
    run.write_json(tmp_path / 'result.json', dict(status='model_error', error='ResponsesError: Responses HTTP400'))
    run.write_json(tmp_path / 'official.json', dict(score=0))
    run.write_json(tmp_path / 'llm/request/http_error.json', dict(http_status=http_status, code=code))
    run.EventLog(tmp_path).emit('responses_failed', request_id='request', purpose='original_actor',
                               http_status=http_status, failure_kind='http')
    assert run.model_output_failure(tmp_path) is expected


def test_continuation_binds_current_inputs_receipts_and_selected_evidence(tmp_path, monkeypatch):
    previous = tmp_path / 'previous'
    folder = previous / 'ours60/1'
    run.write_json(folder / 'official.json', dict(score=0))
    run.write_json(folder / 'input.json', dict(model='fixed'))
    run.write_json(previous / 'final-audit.json', dict(guards_passed=True))
    run.write_json(previous / 'protected-inputs.json', {'archived/raw.json': 'sealed-digest'})
    run.write_json(folder / 'llm/old/response.json', dict(large='archived payload'))
    live = tmp_path / 'current-graph/graph.json'
    run.write_json(live, {})
    memory = previous / 'experience.json'
    run.write_json(memory, {})
    manifest = previous / 'handoff.json'
    run.write_json(manifest, {str(memory): run.method.file_hash(memory)})
    for arm in run.ARMS[:2]:
        run.write_json(previous / arm / 'state.json', dict(current_graph=str(live.parent), knowledge_manifest=str(manifest)))
    monkeypatch.setattr(run, 'protect_graph', lambda directory: {str(live): run.method.file_hash(live)})
    inherited = {'ours60/1': dict(source_directory=str(folder), input_sha256=run.method.file_hash(folder / 'input.json'))}
    guards = run.continuation_material(previous, inherited)
    assert str(live) in guards and str(memory) in guards
    assert str(folder / 'official.json') in guards
    assert str(previous / 'final-audit.json') in guards
    assert str(previous / 'protected-inputs.json') in guards
    assert str(folder / 'llm/old/response.json') not in guards
    run.write_json(folder / 'input.json', dict(model='changed'))
    with pytest.raises(ValueError, match='Inherited task input changed'):
        run.continuation_material(previous, inherited)


@pytest.mark.parametrize('index_status,extra_http,expected', [
    ('completed', None, True), ('failed', None, False), ('completed', 401, False)])
def test_method_inspection_rejection_cannot_hide_an_independent_failure(tmp_path, index_status, extra_http, expected):
    run.write_json(tmp_path / 'result.json', dict(status='model_error', error='ResponsesError: Responses HTTP400',
        provider_failure=dict(kind='http', http_status=400),
        execution_graph=dict(index_status=index_status, cards=dict(calls=0, stopped=True))))
    run.write_json(tmp_path / 'official.json', dict(score=0))
    run.write_json(tmp_path / 'llm/request/http_error.json', dict(http_status=400, code='data_inspection_failed'))
    run.EventLog(tmp_path).emit('responses_failed', request_id='request', purpose='original_actor',
                               http_status=400, failure_kind='http')
    if extra_http:
        run.EventLog(tmp_path / 'graph-annotations').emit('responses_failed', request_id='independent',
            purpose='transition_description', http_status=extra_http, failure_kind='http')
    before = (tmp_path / 'events.jsonl').read_bytes()
    assert run.model_output_failure(tmp_path, method_assets=True) is expected
    assert (tmp_path / 'events.jsonl').read_bytes() == before


@pytest.mark.parametrize('templates,reason,expected', [
    (['new'], 'provider_failure', True), (['new', 'unexplained'], 'provider_failure', False),
    (['new'], 'unknown_failure', False)])
def test_only_explicitly_skipped_unstarted_cards_follow_actor_failure(tmp_path, templates, reason, expected):
    run.write_json(tmp_path / 'result.json', dict(status='model_error', error='ResponsesError: Responses HTTP400',
        provider_failure=dict(kind='http', http_status=400), execution_graph=dict(index_status='completed',
            cards=dict(calls=0, stopped=True), card_refresh=dict(unavailable=templates))))
    run.write_json(tmp_path / 'official.json', dict(score=0))
    run.write_json(tmp_path / 'llm/request/http_error.json', dict(http_status=400, code='data_inspection_failed'))
    run.EventLog(tmp_path).emit('responses_failed', request_id='request', purpose='original_actor',
                               http_status=400, failure_kind='http')
    run.EventLog(tmp_path / 'graph-cards').emit('template_card_refresh_skipped', template='new', reason=reason)
    before = (tmp_path / 'result.json').read_bytes()
    assert run.model_output_failure(tmp_path, method_assets=True) is expected
    assert (tmp_path / 'result.json').read_bytes() == before


@pytest.mark.parametrize('arm', run.ARMS)
def test_independent_method_settings(tmp_path, arm, monkeypatch):
    monkeypatch.setattr(run.method, 'cards_path', lambda graph: graph / 'cards.json')
    old = dict(initial_experience='old-task-memory', previous_version='old-run',
               inherited_results=[1], inherited_tasks={'1': {}}, auxiliary_model='old-model')
    config = run.arm_settings(tmp_path, arm, old, [dict(task_id=27)])
    assert 'initial_experience' not in config and 'previous_version' not in config
    assert config['inherited_results'] == [] and config['inherited_tasks'] == {}
    assert config['runtime'] == str(tmp_path / 'frozen-runtime')
    assert config['stop_file'] == str(tmp_path / 'STOP')
    if arm != 'webnavigator':
        assert config['experience_store'] == str(tmp_path / arm / 'experience.json')
        assert config['max_graph_nodes'] == int(arm[4:])
        assert config['auxiliary_model'] == config['diagnosis_model'] == config['model']
        assert config['action_budget_policy'] == 'actor-action-v1'
    else:
        assert config['action_budget_policy'] == 'legacy-shared-calls'
        assert config['ax_line_limit'] == 1500


@pytest.mark.parametrize('worker_code', [0, 1])
def test_suite_advances_once_or_stops_with_unresolved_attempt(tmp_path, monkeypatch, worker_code):
    from scripts import build_qwen_graph
    p = dict(runtime=str(run.ROOT), schedule=[dict(arm='webnavigator', task_id=n) for n in (1, 2)])
    c = dict(shared_lock=str(tmp_path / 'lock'), encoder_runtime=str(tmp_path), encoder_python='unused',
             gpu_memory='4GiB', encoder_expected={}, container='owned', image_id='image', backend='local',
             logical_origin='local', owner='owner')
    run.write_json(tmp_path / 'suite.json', p)
    run.write_json(tmp_path / 'state.json', dict(status='registered', tasks={}))
    batch = tmp_path / 'webnavigator'
    batch.mkdir()
    run.write_json(batch / 'protocol.json', c)
    run.write_json(batch / 'state.json', dict(status='registered', tasks={}))
    (tmp_path / 'ours60').mkdir()
    run.write_json(tmp_path / 'ours60/protocol.json', c)
    # Avoid method initialisation in this baseline-only fixture.
    class Arms(tuple):
        def __getitem__(self, key):
            return () if isinstance(key, slice) else super().__getitem__(key)
    monkeypatch.setattr(run, 'ARMS', Arms(('webnavigator',)))
    monkeypatch.setattr(run, 'verify', lambda *args, **kwargs: p)
    monkeypatch.setattr(run, 'report', lambda *args: None)
    monkeypatch.setattr(run, 'queue_lock', lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(build_qwen_graph, 'local_encoder', lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(run, 'reset_copy', lambda *args: None)
    class Process:
        pid = 123
        def wait(self):
            return worker_code
    monkeypatch.setattr(run.subprocess, 'Popen', lambda *args, **kwargs: Process())
    run.live(tmp_path)
    state = run.read(tmp_path / 'state.json')
    assert state['status'] == ('completed' if worker_code == 0 else 'needs_attention')
    assert len(state['tasks']) == (2 if worker_code == 0 else 1)
    assert state['tasks']['webnavigator/1']['status'] == ('completed' if worker_code == 0 else 'needs_attention')
    with pytest.raises(ValueError, match='never-started'):
        run.live(tmp_path)
