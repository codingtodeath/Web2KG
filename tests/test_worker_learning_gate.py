"""Exercise the post-score worker path, beyond the Actor's no-diagnosis switch."""
import shutil

import pytest

from scripts import run_reddit_hard10_225 as runner
from test_function_cards import saved


@pytest.mark.parametrize('budget,official_score,learning_failure', [
    (0, 1, False), (0, 0, False),
    (None, 1, False), (None, 0, False),
    (5, 1, False), (5, 0, False), (5, 1, True)])
def test_scored_worker_handoff_obeys_diagnosis_budget(
        saved, tmp_path, monkeypatch, budget, official_score, learning_failure):
    source, _, _, cards, _ = saved
    folder = tmp_path / '27'
    graph = folder / 'execution-graph'
    shutil.copytree(source.directory, graph)
    shutil.copytree(cards.parent, graph / 'function-cards/fixture')
    runner.write_json(graph / 'function-cards/active.json',
                      {'cards': 'function-cards/fixture/cards.json'})
    memory = tmp_path / 'experience.json'
    runner.write_json(memory, runner.ExperienceStore(memory, runner.memory_namespace(source)).load())
    before = memory.read_bytes()
    config = dict(runtime=str(runner.ROOT), model='fixture', auxiliary_model='fixture',
                  graph=str(source.directory), verified_root='unused', logical_origin='http://fixture',
                  integrity_policy='boundary-v1', experience_store=str(memory), stop_file=str(tmp_path/'STOP'))
    if budget is not None:
        config['diagnosis_calls'] = budget
    if budget:
        config['diagnosis_model'] = 'fixture'
    runner.write_json(folder / 'input.json', config)
    state = {'tasks': {'27': {'status': 'started', 'input_sha256': runner.file_hash(folder/'input.json')}}}
    monkeypatch.setattr(runner, 'verify', lambda *a, **k: (config, state))
    # The expensive browser/model work and external scorer are isolated. The real
    # worker still performs its gates, asset validation, learning and handoff.
    def executed(*args):
        runner.write_json(folder/'result.json', {'status': 'agent_stopped',
            'execution_graph': {'index_status': 'completed', 'directory': str(graph)}})
    monkeypatch.setattr(runner.base, 'worker', executed)
    monkeypatch.setattr(runner.base, 'check_inputs', runner.base.check_inputs)
    monkeypatch.setattr(runner.base, 'execution_clients', runner.base.execution_clients)
    monkeypatch.setattr(runner, 'score', lambda *a: {'score': official_score})
    constructed = []
    class Auxiliary:
        def __init__(self, credential, log, limit, *args, **kwargs):
            assert budget and limit == budget, 'Disabled learning must never construct a client'
            constructed.append(limit)
        def json(self, *a, **k):
            raise AssertionError('No counterfactual recovery exists, so no learning request is justified')
    monkeypatch.setattr(runner, 'model_class', lambda *a: Auxiliary)
    if learning_failure:
        monkeypatch.setattr(runner, 'learn_experience', lambda *a, **k: {'status': 'failed'})
        with pytest.raises(RuntimeError, match='Experience learning failed'):
            runner.worker(tmp_path, 27)
        assert not (folder/'handoff.json').exists()
    else:
        runner.worker(tmp_path, 27)
        runner.accept_worker_outcome(folder, config, state, '27')
        assert state['tasks']['27']['status'] == 'completed'
        assert state['current_graph'] == str(graph)
        manifest = runner.read(folder/'knowledge-handoff.json')
        assert manifest[str(memory.resolve())] == runner.file_hash(memory)
        assert all(runner.file_hash(path) == digest for path, digest in manifest.items())
        if budget:
            assert runner.read(folder/'recovery/learning.json')['status'] == 'skipped'
        else:
            assert not (folder/'diagnosis-calls').exists()
            assert not (folder/'recovery/learning.json').exists()
    assert constructed == ([budget] if budget else [])
    assert runner.read(folder/'official.json')['score'] == official_score
    assert memory.read_bytes() == before
