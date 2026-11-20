"""Independent, checkpointed index-only continuation of a user-paused graph build."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


class CheckpointClient:
    def __init__(self, client, directory):
        self.client=client;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.log=client.log;self.endpoint=client.endpoint;self.done=0

    def identity(self): return self.client.identity()

    def encode(self, value, kind, expected):
        import numpy as np
        if kind!='image': return self.client.encode(value,kind,expected)
        content=sha(value)
        key=hashlib.sha256(json.dumps([content,expected],sort_keys=True).encode()).hexdigest()
        meta=self.directory/(key+'.json');array=self.directory/(key+'.npy')
        if meta.exists():
            receipt=read(meta)
            if receipt['image_sha256']!=content or receipt['encoder']!=expected or sha(array)!=receipt['vector_sha256']:
                raise ValueError('Checkpoint identity/integrity mismatch')
            rows=np.load(array,allow_pickle=False)
            reused=True
        else:
            rows=self.client.encode(value,kind,expected)
            if sha(value)!=content: raise ValueError('Screenshot changed during checkpoint encoding')
            temp=array.with_suffix('.npy.tmp')
            with temp.open('wb') as f:np.save(f,rows,allow_pickle=False)
            temp.replace(array)
            write(meta,{'image_sha256':content,'encoder':expected,'vector_sha256':sha(array)})
            reused=False
        if rows.ndim!=2 or rows.shape[1]!=128 or not np.isfinite(rows).all():
            raise ValueError('Invalid checkpoint vector shape/values')
        self.done+=1
        self.log.emit('index_checkpoint',number=self.done,reused=reused,image_sha256=content)
        print(json.dumps({'indexed_pages':self.done,'reused':reused}),flush=True)
        return rows


def verify(out, project):
    m=read(out/'manifest.json')
    for rel,h in m['source'].items():
        if sha(out/'source'/rel)!=h:raise ValueError('Frozen source changed: '+rel)
    for rel,h in m['preserved'].items():
        if sha(project/rel)!=h:raise ValueError('Preserved input changed: '+rel)
    for rel,h in m['graph_inputs'].items():
        if sha(out/'graph'/rel)!=h:raise ValueError('Copied graph asset changed: '+rel)
    if sha(out/'plan.json')!=m['plan_sha256']:raise ValueError('Plan changed')


def register(out, prior):
    if not out.is_relative_to(ROOT/'runs') or not prior.is_relative_to(ROOT/'runs'):
        raise ValueError('Directories must be in this workspace runs')
    if read(prior/'state.json')['status']!='user_stopped' or not (prior/'STOP').exists():
        raise ValueError('Expected explicitly user-paused input')
    old=read(prior/'source-manifest.json')
    for rel,h in old['source'].items():
        if sha(prior/'source'/rel)!=h:raise ValueError('Prior frozen source changed')
    for rel,h in old['preserved'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Prior protected input changed')
    if sha(prior/'plan.json')!=old['plan_sha256']:raise ValueError('Prior plan changed')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(prior/'source',out/'source')
    target=out/'source/scripts/complete_paused_graph_index.py';target.write_bytes(Path(__file__).read_bytes())
    shutil.copytree(prior/'graph',out/'graph')
    plan={'authorization':'2026-11-04 user explicitly resumes; index-only continuation.',
          'created_at':datetime.now(timezone.utc).isoformat(),'prior':prior.relative_to(ROOT).as_posix(),
          'llm_calls':0,'browser_actions':0,'reset_website':False,'publish_active_graph':False,
          'encoder_expected':read(prior/'encoder-identity.json')['encoder'],
          'encoder_python':'D:/Anaconda/envs/web2kg-jina-gpu/python.exe','port':8796,
          'gpu_memory':'4GiB','memory_note':'Reserve activation headroom on 6GiB GPU; same frozen encoder scripts/model/precision/pipeline, no prior vectors to mix.',
          'checkpoint':'One validated image vector per content hash and complete encoder identity; no partial index publication.'}
    write(out/'plan.json',plan)
    preserved=dict(old['preserved'])
    for path in prior.rglob('*'):
        if path.is_file():preserved[path.relative_to(ROOT).as_posix()]=sha(path)
    write(out/'manifest.json',{'source':{p.relative_to(out/'source').as_posix():sha(p) for p in (out/'source').rglob('*') if p.is_file()},
        'preserved':preserved,'graph_inputs':{p.relative_to(out/'graph').as_posix():sha(p) for p in (out/'graph').rglob('*') if p.is_file() and p.suffix!='.html'},
        'plan_sha256':sha(out/'plan.json')})
    write(out/'state.json',{'status':'setup','phase':'registered'})
    print(json.dumps({'directory':str(out),'registered':True,'llm_calls':0}))


def run(out, project):
    import numpy as np
    from web2kg.common.storage import Graph,EventLog
    from web2kg.common.jina import JinaClient
    from web2kg.stage1.visual_index import build_visual_index,retrieve_visual
    from web2kg.stage1.site_framework import load_framework
    from web2kg.stage2.template_context import TemplateKnowledge
    from web2kg.visualization.viewer import export_viewer
    from web2kg.visualization.template_report import export_template_report
    if read(out/'state.json')['status']!='setup' or (out/'started.json').exists():
        raise ValueError('Never restart a started continuation; preserve its checkpoints')
    verify(out,project)
    graph=Graph.load(out/'graph');log=EventLog(out/'index-calls')
    knowledge=TemplateKnowledge(graph,graph.directory/'function-cards/initial/cards.json',log)
    if (set(knowledge.groups)!=set(knowledge.cards) or load_framework(graph) is None
            or set(knowledge.layer['assignments'])!=set(graph.nodes)):
        raise ValueError('Preserved card/framework assets invalid')
    plan=read(out/'plan.json');port=plan['port']
    with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    client=JinaClient(f'http://127.0.0.1:{port}/v1/embeddings',log,timeout=1800,retry_connections=True,stop_file=out/'STOP')
    start=time.monotonic();state={'status':'running','phase':'encoder_setup','timings':{}}
    write(out/'started.json',{'pid':os.getpid(),'time':datetime.now(timezone.utc).isoformat()})
    write(out/'state.json',state);process=None;stream=None
    try:
        stream=(out/'encoder.log').open('x',encoding='utf-8')
        process=subprocess.Popen([plan['encoder_python'],'-B',str(ROOT/'scripts/serve_jina_gpu.py'),
            '--port',str(port),'--vision-fp32','--gpu-memory',plan['gpu_memory'],'--log',str(out/'encoder-events.jsonl')],
            cwd=ROOT,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'},
            stdout=stream,stderr=stream,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        write(out/'owned-encoder.json',{'pid':process.pid,'parent':os.getpid(),'port':port})
        while True:
            if (out/'STOP').exists():raise RuntimeError('User pause')
            if process.poll() is not None:raise RuntimeError('Owned encoder exited')
            try:identity=client.identity();break
            except OSError:time.sleep(1)
        if identity!=plan['encoder_expected']:raise ValueError('Encoder identity differs from paused run')
        write(out/'encoder-identity.json',{'encoder':identity,'owned_process':process.pid,'offline':True,'gpu_memory':plan['gpu_memory']})
        state['timings']['encoder_setup_seconds']=time.monotonic()-start
        state['phase']='local_index';write(out/'state.json',state)
        checkpoint=CheckpointClient(client,out/'vector-checkpoints');tick=time.monotonic()
        index=build_visual_index(graph,checkpoint)
        state['timings']['index_seconds']=time.monotonic()-tick
        state['phase']='validation';write(out/'state.json',state)
        with np.load(graph.directory/index['vectors'],allow_pickle=False) as vectors:
            valid=set(vectors.files)==set(graph.nodes)==set(index['ids']) and all(
                vectors[k].ndim==2 and vectors[k].shape[1]==128 and np.isfinite(vectors[k]).all() for k in index['ids'])
        if not valid:raise ValueError('Final index coverage/values invalid')
        results=retrieve_visual(graph,'account menu and settings',5,client)
        if not results or not all(np.isfinite(r['score']) for r in results):raise ValueError('Retrieval smoke check failed')
        write(out/'retrieval-check.json',{'query':'account menu and settings','results':results,
            'scope':'Local pipeline smoke check, not task success or retrieval accuracy evaluation.'})
        verify(out,project)
        export_template_report(graph,knowledge.layer,graph.directory/'templates/initial',graph.directory/'function-cards/initial/cards.json')
        export_viewer(graph)
        write(out/'asset-check.json',{'nodes':len(graph.nodes),'edges':len(graph.data['edges']),'cards':len(knowledge.cards),
            'indexed_pages':len(index['ids']),'all_nodes_assigned':set(knowledge.layer['assignments'])==set(graph.nodes),
            'index_matches_nodes':valid,'encoder_identity_matches':True,'prior_inputs_unchanged':True,
            'framework_loaded':True,'llm_calls':0,'browser_actions':0})
        state.update(status='completed',phase='completed')
    except Exception as error:
        state.update(status='user_stopped' if (out/'STOP').exists() else 'failed',error_type=type(error).__name__)
        raise
    finally:
        if process and process.poll() is None:process.terminate();process.wait(timeout=30)
        if stream:stream.close()
        state['wall_seconds']=time.monotonic()-start;write(out/'state.json',state)
        print(json.dumps(state),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['register','run'])
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--prior',type=Path)
    parser.add_argument('--project',type=Path,default=ROOT);args=parser.parse_args()
    if args.mode=='register':register(args.directory.resolve(),args.prior.resolve())
    else:run(args.directory.resolve(),args.project.resolve())
