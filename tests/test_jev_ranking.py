import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from web2kg.common.activity import read_events
from web2kg.common.responses import ResponsesClient, ResponsesError
from web2kg.common.storage import EventLog
from web2kg.stage1.jev_ranking import JevRankingClient, ranked_result, ranking_request, ranking_batches, MAX_BATCH_BYTES


def payload():
    return {'K': 2, 'page': {'candidates': [
        {'id': 'b', 'role': 'link', 'name': 'Entry B', 'xpath': '/main/a[1]'},
        {'id': 'a', 'role': 'link', 'name': 'Entry A', 'xpath': '/main/a[2]'},
        {'id': 'c', 'role': 'button', 'name': 'Menu', 'xpath': '/header/button'}]},
        'history': [{'outcome': 'failed'}, {'outcome': 'no_change'}],
        'parent_pending_candidates': [{'id': 'planned'}]}


def response(scores=(.8, .9, .7), eligibility=(.99, .99, .99)):
    return {'model': 'jev-test', 'answers': {f'{prefix}{i}': {'type': 'noul', 'noul': v}
            for prefix, values in [('c', scores), ('e', eligibility)] for i, v in enumerate(values)},
            'usage': {'input_tokens': 42, 'output_tokens': 3}}


def client(tmp_path, **kwargs):
    env = tmp_path / '.env'
    env.write_text('# Jev api_key\napi_key=private-fixture-secret\n', encoding='utf-8')
    return JevRankingClient(env, EventLog(tmp_path / 'calls'), **kwargs)


def test_batch_retains_history_and_distinguishes_model_scores_from_grouping():
    data = payload()
    original = copy.deepcopy(data)
    request = ranking_request(data, 'jev-test')
    assert request['state']['exploration'] == original
    assert len(request['questions']) == 6
    result, scores = ranked_result(response(), data)
    assert [r['id'] for r in result['selected']] == ['a', 'c']
    assert len(scores) == 3 and all(r['group'].startswith('structure:') for r in scores)
    tied, _ = ranked_result(response((.8, .8, .7)), data)
    assert tied == ranked_result(response((.8, .8, .7)), data)[0]
    assert tied['selected'][0]['id'] == 'a'
    assert data == original


def many_candidates(count=35):
    data=payload()
    data['page']['candidates']=[{'id':str(i),'role':'link','name':f'Entry {i}',
        'xpath':f'/section{i}/a'} for i in range(count)]
    return data


def test_bounded_parts_keep_context_cover_all_ids_and_do_not_mutate():
    data=many_candidates(); original=copy.deepcopy(data)
    parts=ranking_batches(data,'jev-test')
    assert [len(p['page']['candidates']) for p in parts]==[16,16,3]
    assert [r['id'] for p in parts for r in p['page']['candidates']]==[str(i) for i in range(35)]
    for part in parts:
        assert part['history']==data['history'] and part['parent_pending_candidates']==data['parent_pending_candidates']
        request=ranking_request(part,'jev-test')
        assert len(request['questions'])<=32 and len(json.dumps(request).encode())<=MAX_BATCH_BYTES
    assert data==original
    data['history']=['x'*MAX_BATCH_BYTES]
    with pytest.raises(ValueError,match='shared context'):
        ranking_batches(data,'jev-test')


def test_batches_global_ranking_and_late_failure_retains_complete_parts(tmp_path):
    ranker=client(tmp_path,max_calls=20)
    data=many_candidates()
    requests_seen=[]
    def post(url,**kw):
        candidates=kw['json']['state']['exploration']['page']['candidates']
        requests_seen.append([c['id'] for c in candidates])
        scores=[int(c['id'])/40 for c in candidates]
        body=response(scores,tuple(.99 for c in candidates))
        return SimpleNamespace(status_code=200,json=lambda:body)
    ranker.session.post=post
    result=ranker.json('',data,{},'candidate_ranking')
    assert result==ranked_result(response(tuple(i/40 for i in range(35)),(.99,)*35),data)[0]
    assert len(result['first_exploration'])==35 and len(requests_seen)==3
    receipt=json.loads((ranker.log.directory/'llm'/ranker.last_request_id/'batch.json').read_text())
    assert receipt['status']=='completed' and len(receipt['request_ids'])==3
    events=read_events(ranker.log.directory/'events.jsonl')
    assert sum(e['event']=='responses_started' for e in events)==3
    assert sum(e['event']=='responses_received' for e in events)==3

    calls=[]
    def fail_later(url,**kw):
        calls.append(kw['json'])
        if len(calls)==2: return SimpleNamespace(status_code=400)
        return post(url,**kw)
    ranker.session.post=fail_later
    with pytest.raises(ResponsesError): ranker.json('',data,{},'candidate_ranking')
    assert len(calls)==2  # No partial ranking and no resubmission of completed part.


def test_first_exploration_filters_before_dedup_and_keeps_lower_ranked_alternative():
    result, scores = ranked_result(response(eligibility=(.8, .79, .99)), payload())
    assert [r['id'] for r in result['selected']] == ['b', 'c']
    assert result['first_exploration']['a'] == {'score': .79, 'allowed': False}
    assert scores[0]['id'] == 'a' and not scores[0]['first_exploration_allowed']
    assert ranked_result(response(eligibility=(0, .5, .799)), payload())[0]['selected'] == []


@pytest.mark.parametrize('value', [float('nan'), 1.1, True, None])
def test_invalid_first_exploration_cannot_fall_back_to_opportunity(value):
    with pytest.raises(ValueError):
        ranked_result(response(eligibility=(value, .99, .99)), payload())
    data = response()
    del data['answers']['e0']
    with pytest.raises(ValueError):
        ranked_result(data, payload())


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1, 1.1, True, '0.8'])
def test_reject_invalid_probabilities(value):
    with pytest.raises(ValueError):
        ranked_result(response((value, .2, .3)), payload())


def test_reject_missing_extra_and_wrong_typed_answers():
    for mutation in ('missing', 'extra', 'wrong_type'):
        data = response()
        if mutation == 'missing':
            del data['answers']['c0']
        elif mutation == 'extra':
            data['answers']['c3'] = {'type': 'noul', 'noul': .5}
        else:
            data['answers']['c0']['type'] = 'choice'
        with pytest.raises(ValueError):
            ranked_result(data, payload())


def test_live_contract_logging_fork_and_secret_exclusion(tmp_path):
    ranker = client(tmp_path)
    ranker.session.post = Mock(return_value=SimpleNamespace(status_code=200, json=lambda: response()))
    result = ranker.json('unused generative prompt', payload(), {}, 'candidate_ranking')
    assert [r['id'] for r in result['selected']] == ['a', 'c']
    call = ranker.session.post.call_args
    assert call.args == ('https://api.typesafe.ai/v1/systemone',)
    assert call.kwargs['allow_redirects'] is False
    assert set(call.kwargs['json']) == {'model', 'state', 'questions'}
    assert ranker.budget_used() == 1
    for path in (tmp_path / 'calls').rglob('*'):
        if path.is_file():
            assert 'private-fixture-secret' not in path.read_text(encoding='utf-8')
    child = ranker.fork(stop_event=threading.Event())
    assert child.session is not ranker.session and child.session.trust_env is False
    assert child.log is ranker.log and child.max_calls is None
    assert not hasattr(child, 'last_request_id')
    child.session.close()


def test_proxy_survives_background_fork_without_system_proxy_changes(tmp_path):
    ranker = client(tmp_path, proxy='http://127.0.0.1:7890')
    child = ranker.fork()
    assert child.session.proxies == {'https': 'http://127.0.0.1:7890'}
    assert child.session.proxies is not ranker.session.proxies
    assert not child.session.trust_env
    assert child.identity['proxy'] == 'http://127.0.0.1:7890'
    child.session.close()


@pytest.mark.parametrize('status', [401, 451])
def test_permanent_rejection_is_not_retried(tmp_path, status):
    ranker = client(tmp_path, retry_connections=True)
    ranker.session.post = Mock(return_value=SimpleNamespace(status_code=status))
    with pytest.raises(ResponsesError) as raised:
        ranker.json('', payload(), {}, 'candidate_ranking')
    assert raised.value.http_status == status and ranker.session.post.call_count == 1
    assert ranker.budget_used() == 1
    assert not any(e['event'] == 'responses_retry' for e in read_events(ranker.log.path))


def test_malformed_success_is_not_retried_and_stop_prevents_dispatch(tmp_path):
    ranker = client(tmp_path, retry_connections=True, stop_file=tmp_path / 'STOP')
    ranker.session.post = Mock(return_value=SimpleNamespace(status_code=200, json=lambda: response((True, 0, 0))))
    with pytest.raises(ResponsesError):
        ranker.json('', payload(), {}, 'candidate_ranking')
    assert ranker.session.post.call_count == 1
    (tmp_path / 'STOP').touch()
    with pytest.raises(ResponsesError, match='stop'):
        ranker.json('', payload(), {}, 'candidate_ranking')
    assert ranker.session.post.call_count == 1


def test_overload_retries_identical_input_with_accounted_usage(tmp_path, monkeypatch):
    ranker = client(tmp_path, retry_connections=True)
    ranker.session.post = Mock(side_effect=[SimpleNamespace(status_code=529),
        SimpleNamespace(status_code=200, json=lambda: response())])
    monkeypatch.setattr('web2kg.common.responses.time.sleep', lambda _: None)
    ranker.json('', payload(), {}, 'candidate_ranking')
    calls = ranker.session.post.call_args_list
    assert len(calls) == 2 and calls[0].kwargs['json'] == calls[1].kwargs['json']
    assert ranker.budget_used() == 1
    events = read_events(ranker.log.path)
    assert sum(e['event'] == 'responses_started' for e in events) == 2
    assert sum(e['event'] == 'responses_failed' for e in events) == 1
    assert 529 not in ResponsesClient.transient_statuses


def test_only_ranking_and_unique_ids_are_allowed(tmp_path):
    ranker = client(tmp_path)
    ranker.session.post = Mock()
    with pytest.raises(ValueError):
        ranker.json('', payload(), {}, 'transition_description')
    data = payload()
    data['page']['candidates'][0]['id'] = 'a'
    with pytest.raises(ValueError):
        ranker.json('', data, {}, 'candidate_ranking')
    ranker.session.post.assert_not_called()
