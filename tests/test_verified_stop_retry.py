import json
from types import SimpleNamespace

import pytest

from web2kg.common.activity import read_events
from web2kg.common.responses import ResponsesError
from web2kg.common.storage import EventLog
from web2kg.evaluation.verified_answer import validate_stop, stop_instruction, STOP_SPEC
from web2kg.evaluation.webnavigator import OriginalResponses


VALID = 'ACTION: stop [' + json.dumps(dict(task_type='MUTATE', status='SUCCESS',
                                         retrieved_data=None, error_details=None)) + ']'


def client(tmp_path, outputs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    env = tmp_path/'fixture.env'
    env.write_text('CRAWL_API_KEY=fixture\nCRAWL_BASE_URL=https://unused.invalid\nCRAWL_MODEL=fixture\n')
    c = OriginalResponses(env, EventLog(tmp_path/'log'), 1, validate_verified_stop=True, retry_connections=True)
    c.configure_output_retries(3)
    sent = []
    def post(*args, **kwargs):
        sent.append(kwargs['json'])
        output = outputs[min(len(sent)-1, len(outputs)-1)]
        if output == '503':
            return SimpleNamespace(ok=False, status_code=503, content=b'{}', json=lambda:{})
        return SimpleNamespace(ok=True, json=lambda:dict(status='completed',model='fixture',
            usage={'input_tokens':10,'output_tokens':5},output=[{'type':'message','role':'assistant',
            'content':[{'type':'output_text','text':output}]}]))
    c.session.post=post
    return c,sent


@pytest.mark.parametrize('text', ['ACTION: stop [done]', 'ACTION: stop []',
    'ACTION: stop [{"task_type":"MUTATE"}]', VALID.replace('null','NaN',1),
    VALID.replace('"MUTATE"','[]'), VALID.replace('"SUCCESS"','"invented"'),
    VALID.replace('"error_details": null','"error_details": null, "error_details": "duplicate"')])
def test_invalid_public_stop_is_rejected(text):
    with pytest.raises(ResponsesError,match='public stop format'): validate_stop(text)


def test_valid_nested_array_and_note_are_unchanged():
    text='ACTION: stop ['+json.dumps(dict(task_type='RETRIEVE',status='SUCCESS',
        retrieved_data=[{'label':'[bracket]', 'numbers':[1,2]}],error_details=None))+']\nNOTE: Completed.'
    assert validate_stop(text) is None
    assert validate_stop('ACTION: type [12] [a stop [text]] [0]') is None
    assert validate_stop('ACTION: click [1]') is None


@pytest.mark.parametrize('answer', [
    '{"task_type":"MUTATE","status":"SUCCESS"}',
    '{"task_type":"MUTATE","status":"SUCCESS","extra":"ignored"}',
    '{"performed_operation":"navigate","status":"success"}',
    '{"task_type":"RETRIEVE","status":"SUCCESS","retrieved_data":{"count":2}}',
    '```json\n{"task_type":"MUTATE","status":"SUCCESS"}\n```',
])
def test_official_compatible_optional_fields_and_json_wrapping(answer):
    assert validate_stop('ACTION: stop ['+answer+']') is None


@pytest.mark.parametrize('legacy', [True, False])
def test_replaces_actual_legacy_stop_spec_without_changing_other_commands(tmp_path, legacy):
    from pathlib import Path
    old=(Path(__file__).resolve().parents[1]/'third_party/webnavigator/webNavigator/prompts/navigation_specifications/stop.txt').read_text(encoding='utf-8')
    if legacy:
        old = 'stop [answer]: Return text, e.g. stop [5h 47min] or N/A.\nSTRICT RULE: Final answers only.'
    prompt='PREAMBLE\n- click [id]: original click\n- '+old+'\n- type [id] [value]: original type\nOUTPUT'
    updated=stop_instruction(prompt)
    assert '5h 47min' not in updated and 'or N/A' not in updated
    assert updated.count('stop [answer]:')==1 and STOP_SPEC in updated
    assert stop_instruction(updated) == updated
    assert updated.startswith('PREAMBLE\n- click [id]: original click\n')
    assert updated.endswith('\n- type [id] [value]: original type\nOUTPUT')
    c,sent=client(tmp_path,[VALID]);c.unified_verified_stop=True
    c.call([{'role':'user','content':'unchanged observation'}],system_prompt=prompt)
    assert sent[0]['instructions']==updated
    assert sent[0]['input'][0]['content'][0]['text']=='unchanged observation'


def test_format_retry_preserves_input_and_budget_without_browser_action(tmp_path):
    c,sent=client(tmp_path,['ACTION: stop [done]','ACTION: click [9]',VALID])
    assert c.call([{'role':'user','content':'same task and observation'}])==VALID
    assert len(sent)==3 and c.budget_used()==1
    assert all(x['input']==sent[0]['input'] for x in sent)
    events=read_events(c.log.path)
    assert sum(e['event']=='responses_output_retry' for e in events)==2
    assert sum(e['event']=='responses_output_parsed' for e in events)==1
    assert len(list((c.log.directory/'llm').glob('*/response.json')))==3
    with pytest.raises(ResponsesError,match='budget'): c.call([{'role':'user','content':'next'}])
    assert len(sent)==3


def test_output_retry_and_transient_retry_account_separately(tmp_path,monkeypatch):
    c,sent=client(tmp_path,['ACTION: stop [done]','503',VALID])
    monkeypatch.setattr('time.sleep',lambda _:None)
    assert c.call([{'role':'user','content':'same'}])==VALID
    assert c.budget_used()==1 and len(sent)==3
    events=read_events(c.log.path)
    assert sum(e['event']=='responses_retry' for e in events)==1
    assert sum(e['event']=='responses_output_retry' for e in events)==1


def test_bounded_exhaustion_and_selector_untouched(tmp_path):
    c,sent=client(tmp_path,['ACTION: stop [done]'])
    with pytest.raises(ResponsesError): c.call([{'role':'user','content':'same'}])
    assert len(sent)==4
    c,sent=client(tmp_path/'selector',['target_page: r'])
    assert c.call([{'role':'user','content':'query'}],system_prompt='target_page')=='target_page: r'
    assert len(sent)==1
