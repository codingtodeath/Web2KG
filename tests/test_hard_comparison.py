import importlib
import json
from types import SimpleNamespace

import pytest
import requests

from scripts.run_hard_comparison import public_tasks, task_stats, report, repair_schedule
from web2kg.common.activity import read_events
from web2kg.common.responses import ResponsesClient, ResponsesError, object_schema
from web2kg.common.storage import EventLog, write_json
from web2kg.evaluation.webnavigator import OriginalResponses, set_ax_limit
from test_function_cards import saved, CardModel


@pytest.mark.parametrize('structured', [False, True])
@pytest.mark.parametrize('failure', ['502', 'transport'])
def test_persistent_connections_keep_identical_input_and_logical_budget(tmp_path, monkeypatch, structured, failure):
    env = tmp_path/'private.env'
    env.write_text('CRAWL_API_KEY=fixture\nCRAWL_BASE_URL=https://example.invalid/v1\nCRAWL_MODEL=fixture')
    cls = ResponsesClient if structured else OriginalResponses
    client = cls(env, EventLog(tmp_path), max_calls=1, retry_connections=True)
    monkeypatch.setattr('web2kg.common.responses.time.sleep', lambda _: None)
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs['json'])
        if len(calls) <= 5:
            if failure == 'transport':
                raise requests.ConnectionError('private detail')
            return SimpleNamespace(ok=False, status_code=502, content=b'{}', json=lambda: {})
        return SimpleNamespace(ok=True, json=lambda: {'status':'completed', 'model':'fixture',
            'usage':{'input_tokens':7,'output_tokens':3}, 'output':[{'type':'message','role':'assistant',
            'phase':'final_answer','status':'completed','content':[{'type':'output_text',
            'text':'{"ok":true}' if structured else 'ACTION: stop [done]'}]}]})
    client.session.post=post
    def invoke():
        return client.json('system', {'task':'fixture'}, object_schema({'ok':{'type':'boolean'}}), 'test') if structured else client.call([{'role':'user','content':'fixture'}])
    assert invoke() == ({'ok':True} if structured else 'ACTION: stop [done]')
    assert len(calls)==6 and all(x==calls[0] for x in calls)
    assert client.budget_used()==1
    with pytest.raises(ResponsesError, match='budget'):
        invoke()
    events=read_events(client.log.path)
    assert sum(e['event']=='responses_retry' for e in events)==5
    row=task_stats(tmp_path, 1, {})
    assert row['physical_requests']==6 and row['unknown_usage']==5
    assert row['models']['fixture']['input_tokens']==7
    assert 'private detail' not in client.log.path.read_text()


@pytest.mark.parametrize('structured',[False,True])
@pytest.mark.parametrize('status',[400,401,403])
def test_permanent_errors_are_not_connection_retries(tmp_path, structured, status):
    env=tmp_path/'private.env'
    env.write_text('CRAWL_API_KEY=fixture\nCRAWL_BASE_URL=https://example.invalid/v1\nCRAWL_MODEL=fixture')
    client=(ResponsesClient if structured else OriginalResponses)(env,EventLog(tmp_path),retry_connections=True)
    calls=[]
    client.session.post=lambda *a,**k:calls.append(1) or SimpleNamespace(ok=False,status_code=status,content=b'{}',json=lambda:{})
    with pytest.raises(ResponsesError) as error:
        client.json('',{},object_schema({}),'test') if structured else client.call([{'role':'user','content':'test'}])
    assert error.value.http_status==status and calls==[1]


def test_stop_interrupts_wait_before_another_request(tmp_path,monkeypatch):
    env=tmp_path/'private.env'
    env.write_text('CRAWL_API_KEY=fixture\nCRAWL_BASE_URL=https://example.invalid/v1\nCRAWL_MODEL=fixture')
    stop=tmp_path/'STOP'
    client=ResponsesClient(env,EventLog(tmp_path),retry_connections=True,stop_file=stop)
    monkeypatch.setattr('web2kg.common.responses.time.sleep',lambda _:stop.touch())
    client.session.post=lambda *a,**k:SimpleNamespace(ok=False,status_code=503)
    with pytest.raises(ResponsesError) as error:
        client.json('',{},object_schema({}),'test')
    assert error.value.kind=='cancelled'
    assert sum(e['event']=='responses_started' for e in read_events(client.log.path))==1


@pytest.mark.parametrize('structured',[False,True])
def test_invalid_200_json_is_not_a_connection_retry(tmp_path,structured):
    env=tmp_path/'private.env'
    env.write_text('CRAWL_API_KEY=fixture\nCRAWL_BASE_URL=https://example.invalid/v1\nCRAWL_MODEL=fixture')
    client=(ResponsesClient if structured else OriginalResponses)(env,EventLog(tmp_path),retry_connections=True)
    def bad_json():
        raise requests.exceptions.JSONDecodeError('invalid','body',0)
    client.session.post=lambda *a,**k:SimpleNamespace(ok=True,json=bad_json)
    with pytest.raises(ResponsesError):
        client.json('',{},object_schema({}),'test') if structured else client.call([{'role':'user','content':'test'}])
    assert sum(e['event']=='responses_started' for e in read_events(client.log.path))==1


def test_1500_rows_use_optimized_visibility_and_restore():
    from web2kg.stage2.agent import load_runtime
    load_runtime()
    opt=importlib.import_module('webNavigator.obs_opt')
    env=importlib.import_module('webNavigator.env')
    agent=importlib.import_module('webNavigator.webNavigator')
    tree=opt.parse_text_to_tree('[1] RootWebArea root\n'+'\n'.join(f'\t[{i}] link item{i}' for i in range(2,1702)))
    tree.children[0].visible=False
    old=opt.translate_node_to_str
    restore=set_ax_limit(1500)
    try:
        for module in (opt,env,agent):
            rows=module.translate_node_to_str(tree).splitlines()
            assert len(rows)==1500 and not any('item2\'' in s for s in rows)
            assert 'item1501' in rows[-1]
        assert module.translate_node_to_str(tree,mode='name_only').count('\n')==1499
    finally:
        restore()
    assert opt.translate_node_to_str is old
    assert len(opt.translate_node_to_str(tree).splitlines())==1000


def test_tasks_are_fixed_filtered_and_never_include_answers(tmp_path):
    root=tmp_path/'assets/dataset'
    write_json(root/'subsets/webarena-verified-hard.json', {'task_ids':[1,2,3]})
    rows=[dict(task_id=i,sites=['reddit'] if i!=3 else ['shopping'],intent='fixture',
               start_urls=['__REDDIT__'],eval={'secret_answer':'never included'}) for i in range(1,5)]
    write_json(root/'webarena-verified.json',rows)
    selected=public_tasks(tmp_path,7)
    assert {r['task_id'] for r in selected}=={1,2}
    assert selected==public_tasks(tmp_path,7)
    assert all(set(r)=={'task_id','sites','intent','start_urls'} for r in selected)
    assert 'secret_answer' not in json.dumps(selected)


def test_paired_report_keeps_missing_and_all_directions(tmp_path):
    write_json(tmp_path/'protocol.json',{'task_ids':[1,2,3,4,5], 'comparison_limits':['fixture']})
    state={'tasks':{}}
    for arm,scores in [('ours',[1,1,0,0,None]),('webnavigator',[1,0,1,0,None])]:
        for i,score in enumerate(scores,1):
            if score is not None:
                state['tasks'][f'{arm}/{i}']={'status':'completed'}
                write_json(tmp_path/arm/str(i)/'official.json',{'score':score})
    write_json(tmp_path/'state.json',state)
    report(tmp_path)
    result=json.loads((tmp_path/'summary.json').read_text())
    assert result['paired']=={'paired_scored':4,'both_success':1,'ours_only':1,'webnavigator_only':1,'both_failure':1}
    assert result['arms']['ours']['unrun']==1
    assert result['arms']['ours']['official_successes']==2


def test_repair_repeats_entire_baseline_and_only_unstarted_ours():
    protocol={'schedule':[{'arm':a,'task_id':t} for t in (1,2) for a in ('ours','webnavigator')]}
    state={'status':'user_stopped','tasks':{'ours/1':{'status':'completed'},'webnavigator/1':{'status':'completed'}}}
    assert repair_schedule(protocol,state)==[{'arm':'webnavigator','task_id':1},
        {'arm':'ours','task_id':2},{'arm':'webnavigator','task_id':2}]
    state['tasks']['ours/1']['status']='started'
    with pytest.raises(ValueError,match='boundary'):
        repair_schedule(protocol,state)


def test_repair_report_reads_frozen_evidence_without_overwriting_it(tmp_path):
    old=tmp_path/'old/ours/1'
    write_json(old/'official.json',{'score':1})
    before=(old/'official.json').read_bytes()
    batch=tmp_path/'new'
    write_json(batch/'protocol.json',{'task_ids':[1], 'comparison_limits':[], 'repair_of':'old'})
    write_json(batch/'state.json',{'tasks':{'ours/1':{'status':'completed','evidence_directory':str(old)}}})
    report(batch)
    summary=json.loads((batch/'summary.json').read_text())
    assert summary['arms']['ours']['official_successes']==1
    assert summary['arms']['webnavigator']['unrun']==1
    assert '../old/ours/trajectories/task-1.html' in (batch/'report.html').read_text('utf-8')
    assert (old/'official.json').read_bytes()==before


def test_two_completed_tasks_inherit_graph_index_cards_and_initial_framework(saved, site, tmp_path):
    from test_execution_assets import indexed_copy
    from test_execution_graph import hashes
    from web2kg.common.browser import Browser
    from web2kg.stage1.site_framework import build_framework, load_framework
    from web2kg.stage2.graph_update import ExecutionGraph
    from web2kg.stage2.template_context import TemplateKnowledge
    from web2kg.stage2.recovery import memory_namespace
    from scripts.run_hard_comparison import cards_path, advance_graph
    base, cards, encoder = indexed_copy(saved,tmp_path)
    import shutil
    shutil.copytree(cards.parent,base.directory/'cards')
    cards=base.directory/'cards/cards.json'
    layer=json.loads((base.directory/'templates/test/templates.json').read_text())
    model=SimpleNamespace(model='fixture-luna', json=lambda *a:{'overview':'Items and catalog',
        'areas':[{'name':'Items','purpose':'Inspect items','templates':[layer['templates'][0]['id']]}],
        'navigation':['Follow observed links'],'limitations':['Partial coverage']})
    build_framework(base,cards,model,base.directory/'site-framework/test',publish=True)
    original=hashes(base.directory)
    initial=base
    namespace=memory_namespace(base)
    for i in range(2):
        log=EventLog(tmp_path/f'task{i}')
        knowledge=TemplateKnowledge(base,cards,log)
        view=load_framework(base)
        with Browser(site+'/catalog',log,identity_mode='interaction',settle_ms=20) as browser:
            update=ExecutionGraph(base,browser,log,knowledge,card_client=CardModel(tmp_path/f'luna{i}'),
                                  max_cards=10,index_client=encoder)
            update.graph.data['experience_namespace']=namespace
            write_json(update.graph.directory/'initial-website-framework.json',view)
            before=browser.observe();update.observe(before)
            action=next(a for a in before['actions'] if a['name']=='Green mug')
            browser.act(action);after=browser.observe();update.record(before,action,after)
            result=update.finish()
        write_json(log.directory/'result.json',{'execution_graph':result})
        assert advance_graph(log.directory)==update.graph.directory
        assert set(base.nodes)<=set(update.graph.nodes)
        base=update.graph;cards=cards_path(base.directory)
        assert load_framework(base)['inherited_initial_view']
        assert memory_namespace(base)==namespace
    assert hashes(initial.directory)==original
