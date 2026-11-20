import json
from pathlib import Path

import pytest

from scripts import resume_hard_after_outage as runner
from scripts import run_hard_comparison as common


def registration():
    return dict(max_new_starts=27, schedule=[dict(arm='terra', task_id=29)] +
                [dict(arm='qwen', task_id=i) for i in range(26)])


def test_pause_or_failed_worker_cannot_be_restarted():
    config = registration()
    state = {'tasks': {'terra/29': {'status': 'completed'}, 'qwen/0': {'status': 'needs_attention'}}}
    with pytest.raises(ValueError, match='Unresolved'):
        runner.pending(config, state)
    state['tasks']['qwen/0']['status'] = 'started'
    assert len(runner.pending(config, state, 'qwen/0')) == 25
    with pytest.raises(ValueError, match='Unresolved'):
        runner.pending(config, state, 'qwen/1')
    state['tasks']['qwen/0']['status'] = 'completed'
    assert runner.pending(config, state)[0] == dict(arm='qwen', task_id=1)
    config['schedule'][-1] = config['schedule'][0]
    with pytest.raises(ValueError, match='27'):
        runner.pending(config, state)


def test_report_uses_new_attempt_even_when_its_score_is_lower(tmp_path):
    old = tmp_path / 'old'; old.mkdir()
    batch = tmp_path / 'new'; batch.mkdir()
    for arm in ('qwen', 'terra'): (batch / arm).mkdir()
    for folder, score in [(old/'603', 1), (old/'624', 1)]:
        folder.mkdir(parents=True)
        runner.write(folder/'official.json', {'score': score})
    target=tmp_path/'report.html';target.write_text('immutable old report')
    (batch/'comparison-report-before.html').write_bytes(target.read_bytes())
    runner.write(batch/'protocol.json',dict(prior_qwen=str(old),task_ids=[624,603],
        inherited={'624':{'status':'completed'}},schedule=[dict(arm='qwen',task_id=603)],report_path=str(target)))
    runner.write(batch/'state.json',dict(status='registered',tasks={}))
    before=runner.report(batch,common)
    assert before['tasks'][1]['official_score'] is None
    assert before['tasks'][0]['official_score']==1
    (batch/'qwen/603').mkdir()
    runner.write(batch/'qwen/603/official.json', {'score':0})
    runner.write(batch/'state.json',dict(status='running',tasks={'qwen/603':{'status':'started'}}))
    after=runner.report(batch,common)
    assert after['tasks'][1]['official_score']==0
    assert target.read_text().startswith('immutable old report')
    target.write_text('user edit')
    with pytest.raises(ValueError,match='prefix'):
        runner.report(batch,common)
    assert target.read_text()=='user edit'


def test_knowledge_guard_covers_memory_and_vectors(tmp_path):
    graph=tmp_path/'graph';graph.mkdir();(graph/'vectors.npy').write_bytes(b'original')
    memory=tmp_path/'experience.json';memory.write_text('{}')
    k=runner.manifest(graph,memory,tmp_path/'manifest.json')
    saved=json.loads(Path(k['manifest']).read_text())
    memory.write_text('{"changed":true}')
    assert runner.digest(memory)!=saved[str(memory)]
