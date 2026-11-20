import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts.run_agentoccam_flash_hard import prepare_upstream
from web2kg.evaluation.agentoccam import unavailable
from web2kg.evaluation.verified_answer import STOP_SPEC

ROOT=Path(__file__).resolve().parents[1]


def test_upstream_changes_are_only_registered_environment_and_stop(tmp_path):
    source=ROOT/'third_party/agentoccam'; target=tmp_path/'upstream'
    changes=prepare_upstream(source,target)
    assert set(changes)=={'browser_env/envs.py','browser_env/processors.py','AgentOccam/prompts/navigation_specifications/stop.txt'}
    manifest=json.loads((source/'provenance.json').read_text('utf-8'))
    originals={**manifest['upstream_files'], **manifest.get('distribution_overrides', {})}
    for name,sha in originals.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha
        if name not in changes:
            assert (target/name).read_bytes()==(source/name).read_bytes()
    assert (target/'AgentOccam/prompts/navigation_specifications/stop.txt').read_text('utf-8').strip()==STOP_SPEC
    with pytest.raises(FileExistsError): prepare_upstream(source,target)


def test_disabled_provider_fails_instead_of_inventing_response():
    with pytest.raises(RuntimeError,match='Unregistered'): unavailable()


def test_detached_copy_preserves_native_pruning_and_properties(tmp_path):
    program=r'''
import ast,json,random,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from web2kg.evaluation.agentoccam import load_upstream,copy_node_fields
class NoModel:
 def call(self,*a,**k): raise AssertionError('No model in copy test')
root=Path(sys.argv[1]);load_upstream(root/'third_party/agentoccam',NoModel(),Path(sys.argv[2]),'http://127.0.0.1:1')
from browser_env.processors import TreeNode
from AgentOccam.obs_opt import prune_tree,translate_node_to_str,construct_new_DOM_with_visible_nodes
source=ast.parse((root/'third_party/agentoccam/browser_env/processors.py').read_text('utf-8'))
cls=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name=='TreeNode')
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='copy')
space={};exec(compile(ast.Module(body=[method],type_ignores=[]),'original-copy','exec'),space)
old=space['copy']
def make(n,seed=0):
 rng=random.Random(seed);nodes=[TreeNode(0,'RootWebArea','Test',0,properties={})]
 for i in range(1,n):
  parent=nodes[rng.randrange(min(len(nodes),10))]
  node=TreeNode(i,rng.choice(['generic','link','button','StaticText','heading']),str(i)+' Test',parent.depth+1,properties={'nested':[{'value':i}]})
  node.visible=rng.random()>.15;parent.add_child(node);nodes.append(node)
 return nodes
def signature(n):
 return [n.node_id,n.role,n.name,n.depth,n.visible,n.properties,[signature(c) for c in n.children]] if n else None
for seed in range(20):
 nodes=make(45,seed);before=signature(nodes[0])
 for node in nodes:
  a=old(node);b=copy_node_fields(node)
  assert vars(a)==vars(b)
  assert not b.children and b.parent is None
  assert b.properties is not node.properties
 TreeNode.copy=old;a=prune_tree('',nodes[0],mode='node')
 TreeNode.copy=copy_node_fields;b=prune_tree('',nodes[0],mode='node')
 assert signature(a)==signature(b)
 assert translate_node_to_str(a)==translate_node_to_str(b)
 assert signature(nodes[0])==before
timings=[]
for n in (100,300,600):
 nodes=make(n);outputs=[];seconds=[]
 for copier in (old,copy_node_fields):
  TreeNode.copy=copier;started=time.perf_counter()
  result=construct_new_DOM_with_visible_nodes(nodes[0]);seconds.append(time.perf_counter()-started);outputs.append(signature(result))
 assert outputs[0]==outputs[1]
 timings.append(dict(nodes=n,old_seconds=seconds[0],new_seconds=seconds[1],identical=True))
print(json.dumps(dict(seeds=20,copies=900,timings=timings)))
'''
    proc=subprocess.run([sys.executable,'-X','utf8','-B','-c',program,str(ROOT),str(tmp_path)],
                        cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',timeout=180)
    assert proc.returncode==0,proc.stderr[-4000:]
    evidence=json.loads(proc.stdout.strip().splitlines()[-1])
    assert evidence['copies']==900
    # Timings are evidence, not a flaky machine-speed assertion.
    (tmp_path/'copy-equivalence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence))


def test_full_page_bounds_do_not_change_observation_or_live_coordinates(tmp_path):
    prepare_upstream(ROOT/'third_party/agentoccam',tmp_path/'upstream')
    program=r'''
import ast,copy,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,str(Path(sys.argv[2])/'upstream'))
import browser_env.processors as m
source=ast.parse((Path(sys.argv[1])/'third_party/agentoccam/browser_env/processors.py').read_text('utf-8'))
cls=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name=='TextObervationProcessor')
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='fetch_page_accessibility_tree')
space=dict(vars(m));exec(compile(ast.Module(body=[method],type_ignores=[]),'original-fetch','exec'),space)
original=space['fetch_page_accessibility_tree']
nodes=[dict(nodeId='1',backendDOMNodeId=1,role={'value':'RootWebArea'},name={'value':'Page'},childIds=['2','3']),
       dict(nodeId='2',backendDOMNodeId=2,parentId='1',role={'value':'button'},name={'value':'Click'},childIds=[]),
       dict(nodeId='3',backendDOMNodeId=3,parentId='1',role={'value':'link'},name={'value':'Below'},childIds=[])]
class Client:
 def send(self,name,params):
  assert name=='Accessibility.getFullAXTree';return {'nodes':copy.deepcopy(nodes)}
processor=m.TextObervationProcessor.__new__(m.TextObervationProcessor)
calls=[]
def bounds(client,node_id):
 calls.append(str(node_id));return {'result':{'value':{'x':10,'y':20 if str(node_id)=='2' else 2000,'width':100,'height':40}}}
processor.get_bounding_client_rect=bounds
client=Client();a=original(processor,{},client,False);assert calls==['2','3'];calls.clear()
b=processor.fetch_page_accessibility_tree({},client,False);assert not calls
ta,ma,na=processor.parse_accessibility_tree(a);tb,mb,nb=processor.parse_accessibility_tree(b)
assert ta==tb
assert {k:v['backend_id'] for k,v in ma.items()}=={k:v['backend_id'] for k,v in mb.items()}
processor.obs_nodes_info=mb;processor.viewport_size={'width':1000,'height':1000}
class Page: client=client
assert processor.get_element_center('2',Page())==(.06,.04)
assert calls==['2']
print('same full-page text/IDs; zero prequeries; actual target center queried from live DOM')
'''
    proc=subprocess.run([sys.executable,'-X','utf8','-B','-c',program,str(ROOT),str(tmp_path)],
                        cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',timeout=45)
    assert proc.returncode==0,proc.stderr[-4000:]


@pytest.mark.parametrize('mode',['finish','limit','budget','planning','invalid'])
def test_native_agent_with_real_browser_without_paid_model(tmp_path,mode):
    batch=tmp_path.resolve(); prepare_upstream(ROOT/'third_party/agentoccam',batch/'upstream')
    cfg=yaml.safe_load((batch/'upstream/AgentOccam/configs/AgentOccam.yml').read_text('utf-8'))
    cfg.update(max_steps=3,verbose=0,logname='offline')
    cfg['agent']['others'].update(max_steps=3,verbose=0,logging=True)
    for name in ('actor','critic','judge'): cfg['agent'][name].update(model='qwen3.7-flash',verbose=0)
    (batch/'config.yml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    (batch/'output').mkdir()
    # Dedicated subprocess preserves isolation from WebNavigator imports and uses
    # native pruning/history/action parsing on actual Chromium DOM observations.
    program=r'''
import json,re,sys,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from web2kg.evaluation.agentoccam import run_occam
from web2kg.common.responses import ResponsesError
class Page(BaseHTTPRequestHandler):
 def log_message(self,*a): pass
 def do_GET(self):
  self.send_response(200); self.send_header('Content-Type','text/html'); self.end_headers()
  self.wfile.write(b'<html><title>Fixture</title><body><h1>Test page</h1><a href="/second">Open item</a><p>Visible detail</p></body></html>')
server=ThreadingHTTPServer(('127.0.0.1',0),Page);threading.Thread(target=server.serve_forever,daemon=True).start()
backend='http://127.0.0.1:'+str(server.server_port)
class Fake:
 retry_seconds=0
 calls=0
 def call(self,messages,system_prompt='',**kw):
  self.calls+=1
  assert 'retrieved_data' in system_prompt and 'branch' in system_prompt and 'prune' in system_prompt
  assert all(p['type']=='text' for m in messages for p in m['content'])
  text=''.join(p['text'] for m in messages for p in m['content'])
  if sys.argv[3]=='budget': raise ResponsesError('Responses call budget exhausted')
  if sys.argv[3]=='planning' and self.calls<=2:
   action='branch [0] [Read the visible item]' if self.calls==1 else 'prune [0] [Return to the root plan]'
  elif sys.argv[3]=='invalid' and self.calls==1: action='click [987654321]'
  elif (self.calls==1 and sys.argv[3]!='invalid') or sys.argv[3]=='limit':
   match=re.search(r'link \[(\d+)\].*Open item',text)
   assert match,text
   action='click ['+match[1]+']'
  else: action='stop [{"task_type":"RETRIEVE","status":"SUCCESS","retrieved_data":["Visible detail"],"error_details":null}]'
  return 'INTERACTION HISTORY SUMMARY:\nRead fixture.\nOBSERVATION DESCRIPTION:\nItem visible.\nREASON FOR ACTION：\nRead the item.\nACTION：\n'+action+'\nOBSERVATION HIGHLIGHT:\n'
fake=Fake();batch=Path(sys.argv[2])
result=run_occam(batch,dict(backend=backend,max_steps=3,verified_login=False),dict(task_id=1,sites=['reddit'],intent='Read the visible item',start_urls=[backend]),client=fake)
server.shutdown();print(json.dumps(dict(result=result,calls=fake.calls)))
'''
    proc=subprocess.run([sys.executable,'-X','utf8','-B','-c',program,str(ROOT),str(batch),mode],
                        cwd=batch,capture_output=True,text=True,encoding='utf-8',timeout=120)
    assert proc.returncode==0,proc.stderr[-4000:]
    result=json.loads((batch/'1/result.json').read_text('utf-8'))
    if mode=='finish':
        assert result['status']=='agent_stopped',result
        assert result['steps']==2 and json.loads(result['answer'])['status']=='SUCCESS'
    elif mode=='limit':
        assert result['status']=='step_limit' and result['steps']==3,result
        assert not result['answer']
    elif mode=='budget':
        assert result['status']=='model_error' and result['steps']==0,result
        assert not result['answer']
    elif mode=='planning':
        assert result['status']=='agent_stopped' and result['steps']==3,result
        trajectory=json.loads((batch/'1/trajectory.json').read_text('utf-8'))
        assert trajectory[0]['action'].startswith('branch')
        assert trajectory[1]['action'].startswith('prune')
    else:
        assert result['status']=='agent_stopped' and result['steps']==1,result
        assert json.loads(proc.stdout.strip().splitlines()[-1])['calls']==2
    assert (batch/'1/network.har').exists()
    assert not list((batch/'upstream').glob('play*'))
    assert not list((batch/'upstream').glob('trash*'))
    assert (batch/'1/step-0.png').exists()
