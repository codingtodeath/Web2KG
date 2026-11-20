import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from test_output_retry import client
from web2kg.common.activity import read_events
from web2kg.common.responses import ResponsesError
from web2kg.stage1 import function_cards as cards


FAILED = Path('runs/reddit-hard22-feedback-flash-20261106-v1/624/graph-cards/llm/a69cbcfa7d6d40a8ab9e1632fd74ad49')


def evidence(tmp_path):
    image = tmp_path / 'page.png'; image.write_bytes(b'fixture image sent only to mocked transport')
    return dict(template_id='T1', members=['node-original'], representatives=[dict(
        node_id='node-original', title='Fixture page', page_description='Observed page',
        screenshot=dict(path=str(image), sha256=hashlib.sha256(image.read_bytes()).hexdigest()))])


def valid():
    return dict(name='页面', summary='仅描述可见功能', importance=.5, common_functions=[],
                variants=[dict(when='页面可见时', functions=[], evidence_nodes=['P1'])], uncertainties=[])


def test_saved624_empty_evidence_is_rejected_then_corrected_with_original_context(tmp_path):
    response=json.loads((FAILED/'response.json').read_text(encoding='utf-8'))
    raw=response['text']
    original=json.loads((FAILED/'request.json').read_text(encoding='utf-8'))
    # Establish that the old wire schema accepted this exact rejected output.
    value=json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
    Draft202012Validator(original['text']['format']['schema']).validate(value)
    assert [i for i,v in enumerate(value['variants']) if not v['evidence_nodes']]==[1,2]
    c,sent=client(tmp_path,[raw,json.dumps(valid(),ensure_ascii=False)])
    c.allow_json_fence=True
    src=evidence(tmp_path); before=copy.deepcopy(src)
    result=cards.request_card(src,c)
    assert len(sent)==2 and c.budget_used()==1 and src==before
    assert result['variants'][0]['evidence_nodes']==['node-original']
    assert sent[1]['input'][:-2]==sent[0]['input']
    assert sent[1]['input'][-2]=={'role':'assistant','content':[{'type':'input_text','text':raw}]}
    assert 'minItems' in sent[1]['input'][-1]['content'][0]['text']
    assert 'evidence_nodes' in sent[1]['input'][-1]['content'][0]['text']
    stored=json.loads((c.log.directory/'llm'/c.last_request_id/'request.json').read_text())
    assert stored['input'][-2:]==sent[1]['input'][-2:] and stored['local_images']
    assert 'data:image' not in json.dumps(stored)
    from scripts.qwen_hard_transport import chat_request
    wire=chat_request(sent[1])
    assert [m['role'] for m in wire['messages'][-2:]]==['assistant','user']
    assert wire['messages'][-2]['content']==[{'type':'text','text':raw}]


def test_semantic_validation_shares_retry_limit_and_preserves_bad_wire_value(tmp_path,monkeypatch):
    # Simulate a schema that cannot express a domain rule; verify the second layer
    # is in the same correction loop, without mutating P1 into a stored node ID.
    schema=copy.deepcopy(cards.CARD_SCHEMA);schema['properties']['name'].pop('pattern')
    monkeypatch.setattr(cards,'CARD_SCHEMA',schema)
    bad={**valid(),'name':' '}; raw=json.dumps(bad)
    c,sent=client(tmp_path,[raw,json.dumps(valid())])
    cards.request_card(evidence(tmp_path),c)
    assert sent[1]['input'][-2]['content'][0]['text']==raw
    assert 'Invalid card text' in sent[1]['input'][-1]['content'][0]['text']
    events=read_events(c.log.path)
    assert sum(e['event']=='responses_semantic_rejected' for e in events)==1
    assert sum(e['event']=='responses_semantic_validated' for e in events)==1
    assert c.budget_used()==1


def test_latest_feedback_only_transport_identical_and_next_card_clean(tmp_path,monkeypatch):
    bad=valid();bad['variants'][0]['evidence_nodes']=[]
    other={**valid(),'name':' '}
    outputs=[json.dumps(bad),'503',json.dumps(other),json.dumps(valid()),json.dumps(valid())]
    c,sent=client(tmp_path,outputs);c.retry_connections=True
    monkeypatch.setattr('time.sleep',lambda _:None)
    src=evidence(tmp_path);cards.request_card(src,c)
    assert len(sent)==4 and sent[1]==sent[2] and c.budget_used()==1
    assert sent[3]['input'][:-2]==sent[0]['input']
    assert sent[3]['input'][-2]['content'][0]['text']==json.dumps(other)
    assert '_output_feedback' not in c.fork().__dict__
    c.max_calls=2;cards.request_card(src,c)
    assert sent[4]['input']==sent[0]['input']


@pytest.mark.parametrize('kind',['empty','unknown','too_many','whitespace','too_long'])
def test_invalid_cards_exhaust_one_shared_budget_without_guessing(tmp_path,kind):
    bad=valid()
    if kind=='empty':bad['variants'][0]['evidence_nodes']=[]
    elif kind=='unknown':bad['variants'][0]['evidence_nodes']=['P2']
    elif kind=='too_many':bad['variants']*=5
    elif kind=='whitespace':bad['name']='  '
    else:bad['common_functions']=['x'*501]
    c,sent=client(tmp_path,[json.dumps(bad)])
    with pytest.raises(ResponsesError):cards.request_card(evidence(tmp_path),c)
    assert len(sent)==4 and c.budget_used()==1 and c._output_feedback==[]
    assert all(r['input'][-2]['content'][0]['text']==json.dumps(bad) for r in sent[1:])


def test_no_variants_is_valid_and_asset_or_programming_errors_do_not_retry(tmp_path):
    value={**valid(),'variants':[]};c,sent=client(tmp_path,[json.dumps(value)])
    src=evidence(tmp_path);assert cards.request_card(src,c)['variants']==[]
    Path(src['representatives'][0]['screenshot']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='screenshot'):cards.request_card(src,c)
    assert len(sent)==1
    c.max_calls=2
    def broken(value):raise TypeError('programming error')
    with pytest.raises(TypeError,match='programming'):
        c.json('test',{},cards.CARD_SCHEMA,'template_card',validate=broken)
    assert len(sent)==2 and c._output_feedback==[]


def test_stop_during_card_feedback_prevents_retry(tmp_path):
    bad=valid();bad['variants'][0]['evidence_nodes']=[]
    c,sent=client(tmp_path,[json.dumps(bad),json.dumps(valid())]);c.stop_file=tmp_path/'STOP'
    emit=c.log.emit
    def stopped(event,**data):
        emit(event,**data)
        if event=='responses_output_feedback':c.stop_file.write_text('pause')
    c.log.emit=stopped
    with pytest.raises(ResponsesError):cards.request_card(evidence(tmp_path),c)
    assert len(sent)==1 and c._output_feedback==[]
