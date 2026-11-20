from copy import deepcopy
from pathlib import Path

import pytest

from scripts import continue_midterm_qwen as resume
from scripts import run_midterm_qwen as run
from web2kg.common.storage import write_json


def fixture_plan():
    protocol = dict(schedule=[dict(arm='ours', task_id=1), dict(arm='ours', task_id=2)], max_new_task_starts=2)
    state = dict(current_graph='committed', tasks={'ours/1': dict(status='completed', official_score=1),
                                                'ours/2': dict(status='needs_attention', graph_before='committed')})
    pause = dict(interrupted_tasks=['ours/2'], queue_and_workers_exited=True,
                 completed_starts=1, last_committed_graph='committed')
    result = dict(provider_failure={'kind': 'cancelled'}, error='ResponsesError: User stop file present')
    return protocol, state, pause, result


def test_only_explicit_user_stop_can_be_continued_without_losing_prefix():
    protocol, state, pause, result = fixture_plan()
    original = deepcopy(state)
    completed, key = resume.continuation_plan(protocol, state, pause, result)
    assert key == 'ours/2' and list(completed) == ['ours/1'] and state == original
    continuation = {**protocol, 'max_new_task_starts': 1, 'inherited_completed': {'ours/1': 'old/ours/1'}}
    assert run.pending(continuation, {'tasks': completed}) == [protocol['schedule'][1]]
    with pytest.raises(ValueError, match='Unresolved'):
        run.pending(continuation, state)
    with pytest.raises(ValueError, match='Inherited'):
        run.pending(continuation, {'tasks': {}})


@pytest.mark.parametrize('change', ['transport', 'graph', 'another_unresolved', 'reordered'])
def test_refuse_nonpause_failures_uncommitted_knowledge_or_changed_order(change):
    protocol, state, pause, result = fixture_plan()
    if change == 'transport':
        result['provider_failure']['kind'] = 'transport'
    elif change == 'graph':
        state['current_graph'] = 'interrupted'
    elif change == 'another_unresolved':
        state['tasks']['ours/3'] = dict(status='needs_attention')
    else:
        protocol['schedule'].reverse()
    with pytest.raises(ValueError):
        resume.continuation_plan(protocol, state, pause, result)


def test_report_uses_inherited_evidence_and_keeps_cancelled_costs_separate(tmp_path, monkeypatch):
    old, suite = tmp_path / 'old', tmp_path / 'continuation'
    previous = old / 'reddit/ours/1'
    cancelled = old / 'reddit/ours/2'
    write_json(previous / 'result.json', dict(status='completed', steps=3))
    write_json(previous / 'official.json', {'score': 1})
    write_json(cancelled / 'result.json', dict(status='model_error', steps=5))
    write_json(cancelled / 'official.json', {'score': 0})
    write_json(cancelled / 'events.json', {})
    (cancelled / 'events.jsonl').write_text('{"event":"responses_started","request_id":"r","model":"qwen"}\n')
    trace = previous.parent / 'trajectories/task-1.html'
    trace.parent.mkdir()
    trace.write_text('saved original trace')
    hashes = {str(p): run.file_hash(p) for p in old.rglob('*') if p.is_file()}
    protocol = dict(task_ids=[1, 2], inherited_completed={'ours/1': str(previous)})
    write_json(suite / 'reddit/protocol.json', protocol)
    write_json(suite / 'reddit/state.json', dict(status='registered', tasks={'ours/1': {'status': 'completed'}}))
    write_json(suite / 'continuation.json', dict(previous=str(old), interrupted_attempts=[
        dict(evidence_directory=str(cancelled), task_id=2, arm='ours', site='reddit')]))
    monkeypatch.setattr(run, 'export_existing', lambda _: pytest.fail('Do not re-export inherited evidence'))
    summary = run.report(suite)
    group = summary['sites']['reddit']['groups']['ours']
    assert group['official_scored'] == group['successes'] == 1 and group['unrun'] == 1
    assert summary['interrupted_attempts'][0]['physical_requests'] == 1
    assert summary['interrupted_attempts'][0]['official_score_excluded_from_current_view']
    assert '../old/reddit/ours/trajectories/task-1.html' in (suite / 'report.html').read_text('utf-8')
    assert all(run.file_hash(p) == digest for p, digest in hashes.items())


def test_encoder_outputs_never_become_registered_inputs(tmp_path):
    root = tmp_path / 'encoder-runtime'
    script = root / 'scripts/serve_jina_gpu.py'
    script.parent.mkdir(parents=True)
    script.write_bytes(b'# exact source\n')
    write_json(root / 'manifest.json', {script.name: {'sha256': run.file_hash(script)}})
    log = root / 'runs/jina-gpu/control-encoder.log'
    log.parent.mkdir(parents=True)
    log.write_text('old startup\n')
    guards = {str(p): run.file_hash(p) for p in run.encoder_material(root)}
    log.write_text('old startup\nnew real encoder startup\n')
    assert set(guards) == {str(script), str(root / 'manifest.json')}
    assert all(run.file_hash(p) == digest for p, digest in guards.items())
    script.write_bytes(b'# changed source\n')
    with pytest.raises(ValueError, match='Encoder source'):
        run.encoder_material(root)


def test_repaired_full_registration_uses_original_graph_and_no_inherited_results(tmp_path, monkeypatch):
    original = tmp_path / 'original'
    original.mkdir()
    write_json(original / 'graph.json', {})
    prior = tmp_path / 'prior'
    suite = tmp_path / 'new'
    write_json(tmp_path / 'runs/active-graph.json', {'graph': str(original)})
    write_json(prior / 'reddit/protocol.json', {'graph': str(original),
        'protected_inputs': {str(original / 'graph.json'): run.file_hash(original / 'graph.json')}})
    write_json(prior / 'reddit/state.json', {'tasks': {'ours/1': {'status': 'completed'}}})
    write_json(prior / 'preflight/closeout.json', {'status': 'completed'})
    diagnosis = tmp_path / 'diagnosis.json'
    write_json(diagnosis, {'status': 'confirmed_bug', 'repair': 'encoder_static_inputs_only',
        'full_restart_from_original_inputs': True, 'reason': 'Runtime log wrongly protected'})
    old_encoder = tmp_path / 'runs/reddit-hard-method-qwen-v1/frozen-runtime/scripts'
    old_encoder.mkdir(parents=True)
    for name in ('jina_gpu.py', 'jina_vision_fp32.py', 'serve_jina_gpu.py'):
        (old_encoder / name).write_text('# exact source')
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'freeze', lambda _: None)
    calls = []
    monkeypatch.setattr(run, 'prepare_site', lambda *a: calls.append(a))
    run.prepare(suite, prior, diagnosis)
    assert calls[0][2] == original
    assert run.read(suite / 'suite.json')['planned'] == {'reddit': 84, 'gitlab': 114}
    assert run.read(suite / 'repair-lineage.json')['old_task_starts'] == 1
    assert not (suite / 'reddit/state.json').exists()  # no prior rows injected
    with pytest.raises(ValueError, match='zero-start'):
        run.prepare(tmp_path / 'invalid-restart', prior)
