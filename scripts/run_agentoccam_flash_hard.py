"""Independent AgentOccam / Qwen3.7-flash Reddit Hard42 baseline."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scripts.run_midterm_qwen import freeze, queue_lock
from scripts.run_reddit_hard10_225 import read, hashes, selected_tasks
from scripts.run_webnavigator_flash_hard import verify, check_hashes, fresh_state, outcome
from web2kg.common.activity import read_events
from web2kg.common.storage import write_json
from web2kg.evaluation.verified import reset_copy, score
from web2kg.evaluation.verified_answer import STOP_SPEC
from web2kg.stage1.function_cards import file_hash

MODEL = 'qwen3.7-flash'
REVISION = 'c078ba629212ea7cee35b2718dc8df72c05e57c7'
REFERENCE = 'runs/webnavigator-flash-hard42-20261110-v4'


def prepare_upstream(source, target):
    """Fresh independent files; old upstream trees and experiment patches untouched."""
    provenance = read(source/'provenance.json')
    snapshot = {**provenance['upstream_files'], **provenance.get('distribution_overrides', {})}
    check_hashes({str(source/k): v for k,v in snapshot.items()})
    shutil.copytree(source,target)
    changes = {}
    name = 'browser_env/envs.py'
    path = target/name; original = path.read_text('utf-8')
    anchor = '''            device_scale_factor=1,
        )
        if self.save_trace_enabled:'''
    replacement = '''            device_scale_factor=1,
            record_har_path=instance_config.get("record_har_path"),
            record_har_content="embed",
            service_workers="block",
        )
        if instance_config.get("allowed_origin"):
            from web2kg.stage2.transport import route_request
            self.context.route("**/*", lambda route: route_request(route, instance_config, self.context))
        if self.save_trace_enabled:'''
    if original.count(anchor) != 1:
        raise ValueError('Upstream browser context changed')
    path.write_text(original.replace(anchor,replacement),encoding='utf-8',newline='\n')
    changes[name] = dict(upstream=provenance['upstream_files'][name], adapted=file_hash(path),
                        reason='Verified local login headers scoped to local origin; HAR; disable service workers for capture')
    name='browser_env/processors.py'
    path=target/name; original=path.read_text('utf-8')
    anchor='''                node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
            else:
                response = self.get_bounding_client_rect('''
    replacement='''                node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
            elif not current_viewport_only:
                # Full-page observations do not filter on cached bounds. Action
                # visibility/centers still query the current DOM when needed.
                node["union_bound"] = None
            else:
                response = self.get_bounding_client_rect('''
    if original.count(anchor)!=1: raise ValueError('Upstream AX bounds implementation changed')
    path.write_text(original.replace(anchor,replacement),encoding='utf-8',newline='\n')
    changes[name]=dict(upstream=provenance['upstream_files'][name],adapted=file_hash(path),
        reason='Skip unused per-node geometry only for full-page AX; live action geometry unchanged')
    name = 'AgentOccam/prompts/navigation_specifications/stop.txt'
    path = target/name; path.write_text(STOP_SPEC+'\n',encoding='utf-8',newline='\n')
    changes[name] = dict(upstream=provenance['upstream_files'][name], adapted=file_hash(path),
                        reason='Same public Verified JSON final-answer contract as WebNavigator')
    return changes


def register(batch, checks, repair_from=None):
    if batch.exists() or not batch.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a fresh directory within runs')
    receipt = read(checks)
    if not receipt.get('passed'): raise ValueError('Offline checks required')
    check_hashes(receipt['source_sha256'])
    repair_guards = {}
    if repair_from:
        previous=read(repair_from/'state.json')
        diagnosis=read(repair_from/'failure-diagnosis.json')
        if previous['status']=='running' or not diagnosis or diagnosis.get('status')!='confirmed':
            raise ValueError('Repair requires stopped, diagnosed source')
        verify(repair_from,full=True)
        repair_guards={path:sha for path,sha in hashes(repair_from).items() if 'auth' not in Path(path).parts}
    prior = ROOT/REFERENCE; old = read(prior/'protocol.json')
    if read(prior/'state.json')['status'] != 'completed':
        raise ValueError('Reference run must have stopped')
    tasks = selected_tasks(ROOT,old['backend'],limit=42)
    if [t['task_id'] for t in tasks] != old['task_ids']:
        raise ValueError('Task schedule differs from reference')
    batch.mkdir(parents=True)
    runtime = freeze(batch)
    # freeze() serves older runners too. Extend only this new manifest.
    shutil.copytree(ROOT/'third_party/agentoccam',runtime/'third_party/agentoccam')
    manifest = read(runtime/'manifest.json')
    manifest.update({p.relative_to(runtime).as_posix():file_hash(p)
                     for p in (runtime/'third_party/agentoccam').rglob('*') if p.is_file()})
    write_json(runtime/'manifest.json',manifest)
    changes = prepare_upstream(runtime/'third_party/agentoccam',batch/'upstream')
    import yaml
    cfg = yaml.safe_load((batch/'upstream/AgentOccam/configs/AgentOccam.yml').read_text('utf-8'))
    cfg.update(max_steps=30, verbose=0, logname='AgentOccam', logging=True)
    cfg['agent']['others'].update(max_steps=30,verbose=0,logging=True)
    for name in ('actor','critic','judge'):
        cfg['agent'][name].update(model=MODEL,verbose=0)
    assert cfg['agent']['actor']['number']==1 and not cfg['agent']['critic']['mode'] and not cfg['agent']['judge']['mode']
    (batch/'config.yml').write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    write_json(batch/'public-inputs.json',tasks)
    p = dict(version='agentoccam-flash-hard42-v2',created_at=datetime.now(timezone.utc).isoformat(),
        authorization='User requested AgentOccam baseline with the same Qwen as the completed WebNavigator comparison.',
        runtime=str(runtime),workspace=str(runtime),upstream_revision=REVISION,
        model=MODEL,provider='bailian',role_models={'actor':MODEL},credential_file=old['credential_file'],
        task_ids=old['task_ids'],seed=old['seed'],backend=old['backend'],logical_origin=old['backend'],
        container=old['container'],owner=old['owner'],image_id=old['image_id'],verified_root=old['verified_root'],
        max_steps=30,max_calls=31,output_tokens_per_call=4096,retry_connections=True,
        observation_line_limit=1000,node_copy='detached-fields-v1',
        stop_file=str(batch/'STOP'),shared_lock=old['shared_lock'],continue_task_errors=True,
        upstream_file_adaptations=changes,reference=str(prior),paid_preflight=0,
        limits=['Native text-only full-page pruning with its own 1000-line serializer; native history/highlight and branch/prune.',
            'Equivalent detached node copy avoids cloning parent/children links discarded by original copy().',
            'Full-page AX skips unused bounds prequeries; viewport filtering and live action coordinates unchanged.',
            '30 decision rounds count planning/note too. Native invalid-output regenerations consume the 31-call budget.',
            'No graph/Jina/selector, no added diagnosis or cards. Not an exact original-paper reproduction.',
            'Same Qwen transport/output budget, task order, site reset and official evaluator as reference.',
            'Whole-line heading compatibility; no new single-action prompt or format-retry algorithm.',
            'Model/task failures recorded and queue continues; shared integrity/setup errors stop.',
            'Exposed development tasks; no blind/generalization or component-causal claim.'])
    write_json(batch/'protocol.json',p)
    write_json(batch/'run_material_passport.json',dict(method='AgentOccam',repository='https://github.com/amazon-science/AgentOccam',
        revision=REVISION,source_manifest_sha256=file_hash(runtime/'manifest.json'),inputs='public-inputs.json',
        model=MODEL,adaptations=changes,limits=p['limits'],prior_scores_inherited=False))
    inputs = hashes(batch/'upstream')
    for path in (batch/'config.yml',batch/'public-inputs.json',checks,Path(p['credential_file']),ROOT/'auth/llm-crawl.env'):
        inputs[str(path.resolve())]=file_hash(path)
    write_json(batch/'runtime-inputs.json',inputs)
    protected={**repair_guards,**inputs,**hashes(ROOT/'third_party/agentoccam')}
    if repair_from:
        lineage=dict(previous=str(repair_from),previous_summary=read(repair_from/'summary.json'),
            reason='Redundant detached-node tree copying and unused full-page geometry prequeries; output equivalence and live action lookup verified offline.',
            restart='Full42 new attempts; zero inherited scores; old completed and interrupted attempts/costs separate.')
        write_json(batch/'repair-lineage.json',lineage)
        protected[str(batch/'repair-lineage.json')]=file_hash(batch/'repair-lineage.json')
    for relative in ('runs/active-graph.json','third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt',
                     REFERENCE+'/protocol.json',REFERENCE+'/summary.json',REFERENCE+'/final-audit.json',
                     'experiments/webarena-verified/assets/dataset/webarena-verified.json',
                     'experiments/webarena-verified/assets/dataset/subsets/webarena-verified-hard.json'):
        path=ROOT/relative; protected[str(path)]=file_hash(path)
    protected.update(hashes(Path(p['verified_root'])/'src'))
    write_json(batch/'protected-inputs.json',protected)
    write_json(batch/'registration-integrity.json',{str(path):file_hash(path) for path in
        [batch/'protocol.json',batch/'run_material_passport.json',batch/'runtime-inputs.json',batch/'protected-inputs.json',runtime/'manifest.json']})
    write_json(batch/'state.json',dict(status='registered',tasks={}))
    verify(batch,full=True); report(batch)
    print(json.dumps(dict(registered=str(batch),tasks=42,sources=len(manifest),protected=len(protected))))


def result_folder(batch, protocol, task):
    return Path(protocol.get('inherited_tasks', {}).get(str(task), batch/str(task)))


def report(batch):
    p=read(batch/'protocol.json'); state=read(batch/'state.json'); rows=[]
    for task in p['task_ids']:
        folder=result_folder(batch,p,task); result=read(folder/'result.json',{}); official=read(folder/'official.json',{})
        inherited=str(task) in p.get('inherited_tasks',{})
        events=read_events(folder/'events.jsonl')
        starts=[e for e in events if e['event']=='responses_started']
        receipts={e['request_id']:e for e in events if e['event']=='responses_received'}
        status=(p['inherited_statuses'][str(task)] if inherited else
                state['tasks'].get(str(task),{}).get('status','not_started'))
        trace=(os.path.relpath(folder.parent/f'trajectories/task-{task}.html',batch).replace('\\','/')
               if inherited else f'trajectories/task-{task}.html')
        rows.append(dict(task_id=task,status=status,inherited=inherited,result_folder=str(folder),
            agent_status=result.get('status'),official_score=official.get('score'),steps=result.get('steps'),
            seconds=result.get('seconds'),error=result.get('error'),physical_requests=len(starts),
            input_tokens=sum(e.get('usage',{}).get('input_tokens',0) for e in receipts.values()),
            output_tokens=sum(e.get('usage',{}).get('output_tokens',0) for e in receipts.values()),
            unknown_usage=sum(not receipts.get(e['request_id'],{}).get('usage') for e in starts),
            retries=sum(e['event']=='responses_retry' for e in events),
            actual_models=sorted({str(e.get('actual_model')) for e in receipts.values()}),
            roles=dict(Counter(e.get('purpose') for e in starts)),
            trace=trace if result else None))
    resolved=sum(r['status'] in {'completed','failed'} for r in rows)
    scored=sum(r['official_score'] is not None for r in rows); success=sum(r['official_score']==1 for r in rows)
    summary=dict(planned=42,resolved=resolved,official_scored=scored,official_successes=success,
        errors=sum(r['status']=='failed' for r in rows),status=state['status'],model=p['model'],tasks=rows,
        success_rate=success/42 if resolved==scored==42 else None,seconds=state.get('seconds'),
        totals={k:sum(r[k] for r in rows) for k in ('physical_requests','input_tokens','output_tokens','unknown_usage','retries')},
        interrupted_attempts=p.get('interrupted_attempts',[]),
        continuation=p.get('continuation'),new_tasks=len(p.get('run_task_ids',p['task_ids'])))
    write_json(batch/'summary.json',summary)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in
        ('task_id','status','official_score','steps','seconds','physical_requests'))+
        (f'<td><a href="{r["trace"]}">轨迹</a></td>' if r['trace'] else '<td>—</td>')+'</tr>' for r in rows)
    page=('<!doctype html><meta charset="utf-8"><title>AgentOccam · Flash · Hard42</title>'
        '<style>body{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 16px}td,th{border:1px solid #ddd;padding:6px}table{border-collapse:collapse}</style>'
        f'<h1>AgentOccam × Qwen3.7-flash</h1><p>已结束 {resolved}/42，已评分 {scored}，官方成功 {success}。状态：{state["status"]}。</p>'
        '<p>同一 Reddit Hard42、30 决策步、31 逻辑调用、4096 输出。保留原版全文结构裁剪、1000 行文本输出、历史压缩、branch/prune；无图谱或 Jina。节点复制去掉随后被丢弃的父子树深拷贝，已做输出等价检查。最终 stop 适配公共 Verified JSON。格式重生成沿用原算法并占调用额度；传输重试单独计数。失败保留并继续下一题，不重跑同一尝试。</p>'
        '<p>与 WebNavigator 的观察、规划和外部知识不同，属于整套系统对照；这些任务已用于开发，不是盲测或组件因果实验。</p>'
        '<table><tr><th>任务</th><th>状态</th><th>官方分</th><th>决策步</th><th>执行秒</th><th>物理请求</th><th>证据</th></tr>'+table+'</table>'
        '<p><a href="summary.json">结果与用量</a> · <a href="protocol.json">登记协议</a></p>')
    (batch/'report.html').write_text(page,encoding='utf-8')
    for row in rows:
        if row['trace'] and not row['inherited'] and not (batch/row['trace']).exists():
            export_trace(batch,row)
    return summary


def export_trace(batch,row):
    """Show saved model text and actual snapshots, never HAR or login config."""
    folder=batch/str(row['task_id']); output=batch/'trajectories'; output.mkdir(exist_ok=True)
    panels=[]; step=0
    for event in read_events(folder/'events.jsonl'):
        if event['event']=='responses_started':
            key=event['request_id']; directory=folder/'llm'/key
            request=read(directory/'request.json',{}); response=read(directory/'response.json',{})
            inputs=[part.get('text','') for msg in request.get('input',[]) for part in msg.get('content',[]) if part.get('type')=='input_text']
            visible='\n'.join(part.get('text','') for msg in response.get('messages',[]) for part in msg.get('content',[]) if part.get('type')=='output_text')
            picture=f'../{row["task_id"]}/step-{step}.png'
            image=f'<img src="{picture}" alt="本轮保存的页面截图；原版 AgentOccam 的模型输入仅文本">' if (folder/f'step-{step}.png').exists() else ''
            panels.append(f'<section><h2>观察 {step} · 模型请求 {key[:8]}</h2>'+image+
                '<details><summary>实际系统提示词</summary><pre>'+html.escape(request.get('instructions',''))+'</pre></details>'+
                '<details><summary>实际发送的观察和历史</summary><pre>'+html.escape('\n'.join(inputs))+'</pre></details>'+
                '<h3>模型回复</h3><pre>'+html.escape(visible or '(没有可解析回复，见请求错误记录)')+'</pre></section>')
        elif event['event']=='agent_action':
            panels.append('<p><strong>执行决策 '+str(event['step'])+'</strong>：<code>'+html.escape(str(event.get('action')) or '(规划动作)')+'</code></p>')
            step=event['step']
    page=('<!doctype html><meta charset="utf-8"><title>AgentOccam 任务 '+str(row['task_id'])+'</title>'
        '<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:30px auto;padding:0 16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f7f9;padding:16px}img{max-width:100%;border:1px solid #ddd}section{border-top:1px solid #bbb;margin-top:32px}</style>'
        '<h1>AgentOccam 任务 '+str(row['task_id'])+'</h1><p>截图为浏览器证据，模型按原版配置仅接收文本观察。下方只展示实际记录的输入与输出；不导出登录信息、HAR 或评分参考答案。</p>'+''.join(panels))
    (output/f'task-{row["task_id"]}.html').write_text(page,encoding='utf-8')


def worker(batch,task_id):
    p=verify(batch); folder=batch/str(task_id); row=read(batch/'state.json')['tasks'][str(task_id)]
    if row['status']!='started' or file_hash(folder/'input.json')!=row['input_sha256']:
        raise ValueError('Worker input/start mismatch')
    from web2kg.evaluation.agentoccam import run_occam
    task=next(t for t in read(batch/'public-inputs.json') if t['task_id']==task_id)
    result=run_occam(batch,p,task)
    official=score(p['verified_root'],batch,task_id,p['logical_origin']); write_json(folder/'official.json',official)
    status=outcome(result,read_events(folder/'events.jsonl'))
    if official.get('score') is None: status='needs_attention'
    write_json(folder/'closeout.json',dict(status=status,official_score=official.get('score')))
    if status!='completed': raise RuntimeError('Task anomaly: '+str(result.get('error') or status))


def live(batch):
    p=verify(batch,full=True)
    if Path(p['runtime'])!=ROOT: raise ValueError('Use frozen entry')
    state=read(batch/'state.json'); fresh_state(state)
    with queue_lock(p['shared_lock']):
        state.update(status='running',queue_pid=os.getpid(),started_at=time.time())
        write_json(batch/'state.json',state)
        try:
            for task_id in p.get('run_task_ids',p['task_ids']):
                if Path(p['stop_file']).exists(): state['status']='user_stopped'; break
                verify(batch)
                state.update(phase='reset',current_task=task_id); write_json(batch/'state.json',state)
                container=reset_copy(p['container'],p['image_id'],p['backend'],10598,p['logical_origin'],p['owner'])
                folder=batch/str(task_id); folder.mkdir(exist_ok=False); (folder/'output').mkdir()
                write_json(folder/'input.json',p)
                row=dict(status='started',started_at=time.time(),container_id=container,input_sha256=file_hash(folder/'input.json'))
                state['tasks'][str(task_id)]=row; state['phase']='task'; write_json(batch/'state.json',state)
                with (folder/'console.log').open('w',encoding='utf-8') as output:
                    process=subprocess.Popen([sys.executable,'-X','utf8','-B',str(Path(__file__)),
                        '--directory',str(batch),'--worker',str(task_id)],cwd=folder,stdout=output,stderr=output,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'})
                    row['worker_pid']=process.pid; write_json(batch/'state.json',state); code=process.wait()
                closeout=read(folder/'closeout.json',{})
                row.update(status=closeout.get('status','needs_attention'),seconds=time.time()-row['started_at'])
                if Path(p['stop_file']).exists():
                    state['status']='user_stopped'; write_json(batch/'state.json',state); break
                if code or row['status']!='completed':
                    row.update(status='failed',worker_exit_code=code,error=read(folder/'result.json',{}).get('error') or 'Worker/evaluator error')
                    if not (folder/'official.json').exists():
                        write_json(folder/'official.json',score(p['verified_root'],batch,task_id,p['logical_origin']))
                write_json(batch/'state.json',state); report(batch)
                print(json.dumps(dict(task=task_id,status=row['status'],score=read(folder/'official.json',{}).get('score'),resolved=len(state['tasks']))),flush=True)
            else: state['status']='completed'
        except Exception as error:
            state.update(status='user_stopped' if Path(p['stop_file']).exists() else 'needs_attention',error=type(error).__name__+': '+str(error))
        finally:
            state['seconds']=time.time()-state['started_at']; write_json(batch/'state.json',state)
            verify(batch,full=True)
            write_json(batch/'final-audit.json',dict(status=state['status'],guards_passed=True,
                sources=len(read(ROOT/'manifest.json')),protected=len(read(batch/'protected-inputs.json'))))
            report(batch)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True); parser.add_argument('--checks',type=Path)
    parser.add_argument('--live',action='store_true'); parser.add_argument('--worker',type=int)
    parser.add_argument('--repair-from',type=Path)
    parser.add_argument('--continue-from',type=Path)
    args=parser.parse_args(); batch=args.directory.resolve()
    if args.worker is not None: worker(batch,args.worker)
    elif args.live: live(batch)
    elif args.continue_from:
        if args.repair_from: parser.error('Continuation and repair are different registrations')
        from scripts.continue_agentoccam_flash_hard import register_pause_continuation
        register_pause_continuation(batch,args.continue_from.resolve(),args.checks.resolve())
    else: register(batch,args.checks.resolve(),args.repair_from.resolve() if args.repair_from else None)
