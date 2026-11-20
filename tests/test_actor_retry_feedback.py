import copy
import json
from pathlib import Path
import pytest
from test_verified_stop_retry import client
from scripts.qwen_hard_transport import chat_request
from web2kg.common.activity import read_events
from web2kg.stage2.actor_prompt import PROTOCOL

SAVED = Path('runs/reddit-hard31-225-flash-20261106-v1/66/llm')


@pytest.mark.parametrize('request_id', [
    '0e9163f1e9f44ff5ac2cd5a1f667f7ca','f9b72e7ce6bd44c4b5228c2e6496df53',
    '9830e2ba86d145b197dd37a17a3341af','fd53f2cd41f349acb247e3dc5b9d8a04'])
def test_saved66_rejection_is_returned_exactly_with_observation_unchanged(tmp_path,request_id):
    raw=json.loads((SAVED/request_id/'chat-response.json').read_text(encoding='utf-8'))
    rejected=raw['choices'][0]['message']['content']
    c,sent=client(tmp_path,[rejected,'ACTION: go_back'])
    observations=[{'role':'user','content':'Current page with original element IDs'}]
    before=copy.deepcopy(observations)
    assert c.call(observations,system_prompt=PROTOCOL)=='ACTION: go_back'
    assert observations==before and sent[1]['input'][:-2]==sent[0]['input']
    wire=chat_request(sent[1]);prior=wire['messages'][-2];feedback=wire['messages'][-1]
    assert prior['role']=='assistant' and prior['content']==[{'type':'text','text':rejected}]
    assert feedback['role']=='user'
    text=feedback['content'][0]['text']
    assert 'multiple sections' in text and 'Repeated identical ACTION sections' in text
    assert 'No action from it was executed' in text and 'wait for its new observation' in text
    assert c.budget_used()==1 and len(sent)==2
    events=read_events(c.log.path)
    assert len([e for e in events if e['event']=='responses_output_parsed'])==1
    assert len([e for e in events if e['event']=='responses_output_feedback'])==1


def test_correction_keeps_only_latest_rejection_and_does_not_leak_to_next_call(tmp_path):
    a='ACTION: go_back\nACTION: go_back'
    b='ACTION: click [7]\nACTION: stop [done]'
    c,sent=client(tmp_path,[a,b,'ACTION: click [7]','ACTION: go_back'])
    c.max_calls=2
    obs=[{'role':'user','content':'same'}]
    assert c.call(obs,system_prompt=PROTOCOL)=='ACTION: click [7]'
    assert len(sent[2]['input'])==len(sent[0]['input'])+2
    assert sent[1]['input'][-2]['content'][0]['text']==a
    assert sent[2]['input'][-2]['content'][0]['text']==b
    assert 'Format correction 2/3' in sent[2]['input'][-1]['content'][0]['text']
    assert c.call(obs,system_prompt=PROTOCOL)=='ACTION: go_back'
    assert sent[3]['input']==sent[0]['input'] and c.budget_used()==2
