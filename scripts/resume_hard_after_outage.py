"""Explicitly registered outage replacement and continuation; old attempts stay immutable."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pending(protocol, state, active=None):
    keys = [f'{r["arm"]}/{r["task_id"]}' for r in protocol['schedule']]
    if len(keys) != 27 or len(set(keys)) != 27 or protocol['max_new_starts'] != 27:
        raise ValueError('Expected exactly 27 registered new starts')
    if not set(state['tasks']).issubset(keys):
        raise ValueError('Unregistered task')
    for key, row in state['tasks'].items():
        if row['status'] != 'completed' and not (key == active and row['status'] == 'started'):
            raise ValueError('Unresolved task; never restart: ' + key)
    return [r for r in protocol['schedule'] if f'{r["arm"]}/{r["task_id"]}' not in state['tasks']]


def verify(batch, active=None):
    for path, expected in read(batch / 'registration-integrity.json').items():
        if digest(path) != expected:
            raise ValueError('Registration changed: ' + path)
    protocol, state = read(batch / 'protocol.json'), read(batch / 'state.json')
    for path, expected in protocol['guards'].items():
        if digest(path) != expected:
            raise ValueError('Protected input changed: ' + path)
    for arm, knowledge in state['knowledge'].items():
        if digest(knowledge['manifest']) != knowledge['sha256']:
            raise ValueError('Knowledge manifest changed: ' + arm)
        for path, expected in read(knowledge['manifest']).items():
            if digest(path) != expected:
                raise ValueError('Knowledge changed: ' + path)
    return protocol, state, pending(protocol, state, active)


def load_runtime(protocol, arm):
    runtime = Path(protocol['runtimes'][arm])
    sys.path[:0] = [str(runtime), str(runtime / 'scripts')]
    from scripts import run_verified_template as base
    from scripts import run_hard_comparison as common
    return runtime, base, common


def manifest(graph, experience, target):
    files = [p for p in Path(graph).rglob('*') if p.is_file()]
    files.append(Path(experience))
    write(target, {str(p.resolve()): digest(p) for p in sorted(files)})
    return {'graph': str(graph), 'manifest': str(target), 'sha256': digest(target)}


def worker(batch, arm, task_id):
    protocol, state, _ = verify(batch, f'{arm}/{task_id}')
    runtime, base, common = load_runtime(protocol, arm)
    folder = batch / arm / str(task_id)
    if digest(folder / 'input.json') != state['tasks'][f'{arm}/{task_id}']['input_sha256']:
        raise ValueError('Task input changed')
    config = read(folder / 'input.json')
    base.worker(batch / arm, config, task_id)
    official = common.score(config['verified_root'], batch / arm, task_id, config['logical_origin'])
    write(folder / 'official.json', official)
    if official.get('score') is None:
        raise RuntimeError('Missing official score')
    result = read(folder / 'result.json')
    events = [e for name in ('events.jsonl', 'diagnosis-calls/events.jsonl',
              'graph-annotations/events.jsonl', 'graph-cards/events.jsonl')
              for e in common.read_events(folder / name)]
    recovered = {e.get('request_id') for e in events if e['event'] == 'provider_fallback'}
    if (result.get('status') == 'runtime_error' or
        result.get('provider_failure', {}).get('kind') in {'cancelled', 'http', 'transport'} or
        any(e['event'] == 'responses_failed' and e.get('request_id') not in recovered and
            (e.get('failure_kind') == 'cancelled' or e.get('http_status') in {400, 401, 402, 403, 404}) for e in events)):
        raise RuntimeError('Engineering/permanent/cancelled failure; preserve evidence')
    graph = common.advance_graph(folder)
    client = base.execution_clients(batch / arm, config)[1] if arm == 'qwen' else common.ResponsesClient
    learner = client(runtime / 'auth/llm-crawl.env', common.EventLog(folder / 'diagnosis-calls'),
        config['diagnosis_calls'], 2048, model=config['diagnosis_model'],
        retry_connections=True, stop_file=config['stop_file'])
    memory = common.ExperienceStore(config['experience_store'], common.memory_namespace(common.Graph.load(Path(config['graph']))))
    common.learn_experience(folder, learner, memory, success=official['score'] == 1,
        verifier='WebArena-Verified official score=1')
    handoff = manifest(graph, config['experience_store'], folder / 'knowledge-handoff.json')
    write(folder / 'handoff.json', dict(knowledge=handoff, official_score=official['score']))


def report(batch, common):
    protocol, state = read(batch / 'protocol.json'), read(batch / 'state.json')
    old = Path(protocol['prior_qwen'])
    rows = []
    for task_id in protocol['task_ids']:
        key = f'qwen/{task_id}'
        folder = batch / key if key in state['tasks'] else old / str(task_id)
        row_state = state['tasks'].get(key, protocol['inherited'].get(str(task_id), {}))
        # The interrupted old603 is not the replacement row before its new start.
        if str(task_id) not in protocol['inherited'] and key not in state['tasks']:
            folder = batch / key
        row = common.task_stats(folder, task_id, row_state)
        row['evidence_directory'] = str(folder)
        rows.append(row)
    local = {}
    for arm in ('qwen', 'terra'):
        local[arm] = [common.task_stats(batch / arm / str(r['task_id']), r['task_id'],
            state['tasks'].get(f'{arm}/{r["task_id"]}', {})) for r in protocol['schedule']
            if r['arm'] == arm and (batch / arm / str(r['task_id']) / 'result.json').exists()]
        write(batch / arm / 'summary.json', {'tasks': local[arm]})
        common.export_report([batch / arm], batch / arm / 'trajectories')
    summary = dict(status=state['status'], planned=42, tasks=rows,
        official_scored=sum(r['official_score'] is not None for r in rows),
        official_successes=sum(r['official_score'] == 1 for r in rows),
        unrun=sum(r['status'] == 'not_started' for r in rows), terra29=local['terra'],
        new_completed=sum(r['status'] == 'completed' for r in state['tasks'].values()), new_planned=27)
    write(batch / 'summary.json', summary)
    target = Path(protocol['report_path'])
    link = lambda p: html.escape(Path(os.path.relpath(p, target.parent)).as_posix(), quote=True)
    parts = ['<section id="method-qwen-outage-resume"><h2>本方法 Qwen：中转中断后的补跑与续跑</h2>',
        '<p>原16个完成结果引用原证据；603另建补跑，后续25题保持原顺序。原603官方0及成本保留于上方历史组，不按最高分择优。28、730为百炼输出length截断，非中转故障，未重跑。全组分母42；已有任务知识的开发实验，不支持盲测、泛化或组件因果结论。</p>',
        f'<p>状态 {html.escape(state["status"])}；成功 {summary["official_successes"]}/42；评分 {summary["official_scored"]}/42；本次完成 {summary["new_completed"]}/27（含独立Terra29）。</p>',
        '<table><tr><th>任务</th><th>状态 / 官方分</th><th>已知输入 / 输出token</th><th>物理请求 / 未知usage</th></tr>']
    for row in rows:
        folder = Path(row['evidence_directory'])
        trace = folder.parent / 'trajectories' / f'task-{row["task_id"]}.html'
        label = f'<a href="{link(trace)}">{row["task_id"]}</a>' if trace.exists() else str(row['task_id'])
        inp = sum(v['input_tokens'] for v in row['models'].values())
        out = sum(v['output_tokens'] for v in row['models'].values())
        parts.append(f'<tr><td>{label}</td><td>{html.escape(str(row["status"]))} / {row["official_score"]}</td><td>{inp} / {out}</td><td>{row["physical_requests"]} / {row["unknown_usage"]}</td></tr>')
    parts.append('</table><h3>原Terra方法29：独立补跑</h3><p>原29曾因缺静态资源在0步/0模型调用失败，原官方0保留。补跑沿用原Terra模型、原AX1500实现及当时636入场图，不回灌旧后续题，不改写旧33/42。</p>')
    for row in local['terra']:
        parts.append(f'<p><a href="{link(batch / "terra/trajectories/task-29.html")}">Terra29轨迹</a>：官方分 {row["official_score"]}；状态 {html.escape(row["status"])}。</p>')
    parts.append(f'<p><a href="{link(batch / "summary.json")}">逐题用量汇总</a> · <a href="{link(batch / "impact-audit.json")}">中断与补跑登记</a></p></section>')
    prefix = (batch / 'comparison-report-before.html').read_bytes()
    if not target.read_bytes().startswith(prefix):
        raise ValueError('Old report prefix changed')
    temp = target.with_suffix('.tmp'); temp.write_bytes(prefix + ''.join(parts).encode('utf-8')); temp.replace(target)
    return summary


def queue(batch):
    import msvcrt
    protocol = read(batch / 'protocol.json')
    with Path(protocol['shared_lock']).open('a+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        protocol, state, remaining = verify(batch)
        runtime, _, common = load_runtime(protocol, 'qwen')
        config = read(batch / 'qwen/protocol.json')
        encoder = common.LocalEncoder(common.Graph.load(Path(state['knowledge']['qwen']['graph'])), runtime, config['encoder_python'])
        try:
            encoder.ensure()
            for item in remaining:
                if Path(protocol['stop_file']).exists():
                    state['status'] = 'user_stopped'; break
                protocol, state, _ = verify(batch)
                arm, task_id = item['arm'], item['task_id']; key = f'{arm}/{task_id}'
                config = read(batch / arm / 'protocol.json')
                graph = Path(state['knowledge'][arm]['graph'])
                config.update(graph=str(graph), cards=str(common.cards_path(graph)))
                folder = batch / key; folder.mkdir(exist_ok=False); write(folder / 'input.json', config)
                row = dict(status='started', time=time.time(), input_sha256=digest(folder / 'input.json'), graph_before=str(graph))
                state['tasks'][key] = row; state['status'] = 'running'; write(batch / 'state.json', state)
                print(f'{key} started ({len(state["tasks"])}/27)', flush=True); start = time.monotonic()
                try:
                    row['container_id'] = common.reset_copy(config['container'], config['image_id'], config['backend'],10598,config['logical_origin'],config['owner'])
                    with (folder / 'console.log').open('w',encoding='utf-8') as out:
                        child = subprocess.Popen([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--directory',str(batch),'--worker',arm,str(task_id)],
                            cwd=protocol['workspace'],stdout=out,stderr=out,creationflags=subprocess.CREATE_NO_WINDOW,
                            env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'})
                        row['worker_pid']=child.pid;write(batch / 'state.json',state);code=child.wait()
                    if code: raise RuntimeError(f'Worker exited {code}; never restart automatically')
                    handoff=read(folder / 'handoff.json');state['knowledge'][arm]=handoff['knowledge']
                    row.update(status='completed',seconds=round(time.monotonic()-start,3));write(batch / 'state.json',state)
                    report(batch,common);print(f'{key}: official={handoff["official_score"]}',flush=True)
                except Exception as error:
                    row.update(status='needs_attention',error=f'{type(error).__name__}: {error}')
                    state['status']='needs_attention';write(batch / 'state.json',state);raise
            else:
                verify(batch);state['status']='completed'
            write(batch / 'state.json',state)
        finally:
            encoder.close();report(batch,common)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--worker',nargs=2)
    parser.add_argument('--audit-runtime',choices=['qwen','terra'])
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();batch=args.directory.resolve()
    if args.worker: worker(batch,args.worker[0],int(args.worker[1]))
    elif args.audit_runtime:
        protocol=read(batch / 'protocol.json');_,base,_=load_runtime(protocol,args.audit_runtime)
        base.check_inputs(read(batch / args.audit_runtime / 'protocol.json'));print(args.audit_runtime+' frozen source/input check passed')
    elif args.check:
        _,_,remaining=verify(batch)
        for arm in ('qwen','terra'):
            subprocess.run([sys.executable,'-X','utf8','-B',__file__,'--directory',str(batch),'--audit-runtime',arm],check=True)
        print(f'Guards passed; {len(remaining)} new starts remaining')
    elif args.live: queue(batch)
    else:
        protocol=read(batch / 'protocol.json');_,_,common=load_runtime(protocol,'qwen');report(batch,common)


if __name__ == '__main__':
    main()
