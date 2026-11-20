"""Unpaid, read-only local-page integration check with a deterministic stop reply."""
import argparse
import ast
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]


def check(batch,url):
    import yaml
    from scripts.run_agentoccam_flash_hard import prepare_upstream
    from web2kg.evaluation import agentoccam as adapter
    from web2kg.common.storage import write_json
    if urlsplit(url).netloc!='127.0.0.1:10599' or urlsplit(url).scheme!='http':
        raise ValueError('Only owned local Reddit check')
    label=subprocess.check_output(['docker','inspect','web2kg-template-eval','--format','{{index .Config.Labels "web2kg.template_eval"}}'],text=True).strip()
    if label!='template-agent-hard10-v1': raise ValueError('Wrong local service owner')
    if batch.exists(): raise ValueError('Use a new check directory')
    batch.mkdir(parents=True);(batch/'output').mkdir()
    prepare_upstream(ROOT/'third_party/agentoccam',batch/'upstream')
    cfg=yaml.safe_load((batch/'upstream/AgentOccam/configs/AgentOccam.yml').read_text('utf-8'))
    cfg.update(max_steps=1,verbose=0,logname='offline')
    cfg['agent']['others'].update(max_steps=1,verbose=0,logging=True)
    for role in ('actor','critic','judge'):cfg['agent'][role].update(model='qwen3.7-flash',verbose=0)
    (batch/'config.yml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    phases=[];original_load=adapter.load_upstream

    def load(*args,**kwargs):
        module=original_load(*args,**kwargs)
        from browser_env.envs import ScriptBrowserEnv
        from browser_env.processors import TextObervationProcessor,TreeNode
        def instrument(cls,name):
            original=getattr(cls,name)
            def timed(*a,**k):
                start=time.perf_counter()
                with (batch/'phases.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(event='start',name=name,time=time.time()))+'\n')
                try:return original(*a,**k)
                finally:
                    phases.append(dict(name=name,seconds=time.perf_counter()-start))
                    write_json(batch/'phases.json',phases)
            setattr(cls,name,timed)
        for name in ('setup','modify_page','_get_obs'):instrument(ScriptBrowserEnv,name)
        instrument(TextObervationProcessor,'fetch_page_accessibility_tree')
        syntax=ast.parse((ROOT/'third_party/agentoccam/browser_env/processors.py').read_text('utf-8'))
        cls=next(x for x in syntax.body if isinstance(x,ast.ClassDef) and x.name=='TreeNode')
        fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='copy')
        space={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original-copy','exec'),space)
        old=space['copy'];samples=[]
        def measured_copy(node):
            if len(samples)<2:
                start=time.perf_counter();a=old(node);old_time=time.perf_counter()-start
                start=time.perf_counter();b=adapter.copy_node_fields(node);new_time=time.perf_counter()-start
                assert vars(a)==vars(b)
                samples.append(dict(original_seconds=old_time,optimized_seconds=new_time,identical=True))
                write_json(batch/'real-node-copy.json',samples)
                return b
            return adapter.copy_node_fields(node)
        TreeNode.copy=measured_copy
        return module
    adapter.load_upstream=load

    class Fake:
        retry_seconds=0
        calls=0
        def call(self,messages,system_prompt='',**kw):
            self.calls+=1
            texts=[p['text'] for m in messages for p in m['content']]
            write_json(batch/'observed-size.json',dict(text_chars=sum(map(len,texts)),paid_model_calls=0,mock_calls=self.calls))
            return ('REASON:\nUnpaid observation check.\nACTION:\nstop '
                '[{"task_type":"RETRIEVE","status":"UNKNOWN_ERROR","retrieved_data":null,'
                '"error_details":"Offline adapter check; no task executed."}]\nOBSERVATION HIGHLIGHT:\n')
    os.chdir(batch)
    with (batch/'stack-samples.log').open('w',encoding='utf-8') as log:
        faulthandler.dump_traceback_later(30,repeat=True,file=log)
        try:
            result=adapter.run_occam(batch,dict(backend='http://127.0.0.1:10599',max_steps=1),
                dict(task_id=0,sites=['reddit'],intent='Offline observation check; do not modify the site.',start_urls=[url]),client=Fake())
        finally:faulthandler.cancel_dump_traceback_later()
    print(json.dumps({k:result.get(k) for k in ('status','steps','seconds','error')}))
    if result['status']!='agent_stopped':raise RuntimeError('Offline browser check failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--url',required=True)
    args=parser.parse_args();check(args.directory.resolve(),args.url)
