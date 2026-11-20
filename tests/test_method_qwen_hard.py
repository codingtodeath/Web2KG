import json

import pytest

from scripts import run_method_qwen_hard as runner


def test_registration_never_restarts_or_adds_tasks():
    config = {'task_ids': list(range(42)), 'max_new_task_starts': 42}
    state = {'tasks': {'0': {'status': 'completed'}, '1': {'status': 'started'}}}
    with pytest.raises(ValueError, match='Unresolved'):
        runner.pending(config, state)
    assert runner.pending(config, state, '1') == list(range(2, 42))
    state['tasks']['1']['status'] = 'completed'
    assert runner.pending(config, state) == list(range(2, 42))
    config['task_ids'][-1] = 0
    with pytest.raises(ValueError, match='distinct'):
        runner.pending(config, state)


def test_service_gate_distinguishes_routing_and_malformed_output():
    failure = dict(event='responses_failed', request_id='r', http_status=404)
    assert runner.service_error({}, [failure])
    assert not runner.service_error({}, [failure, dict(event='provider_fallback', request_id='r')])
    assert runner.service_error({'status': 'runtime_error'}, [])
    assert not runner.service_error({'status': 'model_error'}, [dict(event='responses_failed', http_status=None)])
    assert runner.service_error({'provider_failure': {'kind': 'cancelled'}}, [])


def test_report_preserves_old_report_and_exports_only_existing(tmp_path):
    target = tmp_path / 'combined.html'; target.write_text('<h1>Old results</h1>')
    (tmp_path / 'comparison-report-before.html').write_text(target.read_text())
    (tmp_path / 'protocol.json').write_text(json.dumps({'task_ids': list(range(42)),
        'report_path': str(target), 'exposure': 'development'}))
    (tmp_path / 'state.json').write_text(json.dumps({'tasks': {}, 'status': 'registered'}))
    summary = runner.report(tmp_path)
    assert summary['planned'] == summary['unrun'] == 42
    assert summary['official_scored'] == 0
    assert target.read_text().startswith('<h1>Old results</h1>')
    first = target.read_text()
    runner.report(tmp_path)
    assert target.read_text() == first
    target.write_text('User changed the old report')
    with pytest.raises(ValueError, match='Prior report changed'):
        runner.report(tmp_path)
    assert target.read_text() == 'User changed the old report'


def test_knowledge_manifest_covers_graph_vectors_and_experience(tmp_path):
    graph = tmp_path / 'graph'; graph.mkdir()
    for name in ['graph.json', 'index.json', 'vectors.npy']:
        (graph / name).write_bytes(b'original')
    experience = tmp_path / 'experience.json'; experience.write_text('{}')
    output, digest = runner.knowledge_manifest(graph, experience, tmp_path / 'handoff.json')
    recorded = json.loads(open(output).read())
    assert len(recorded) == 4 and runner.file_hash(output) == digest
    (graph / 'vectors.npy').write_bytes(b'changed')
    assert runner.file_hash(graph / 'vectors.npy') != recorded[str((graph / 'vectors.npy').resolve())]
