"""Registered Reddit Hard comparison; cumulative new method versus released WebNavigator."""
import argparse
import html
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_verified_template import check_inputs, source_hash, worker as template_worker
from web2kg.common.activity import read_events
from web2kg.common.jina_runtime import LocalEncoder
from web2kg.common.responses import ResponsesClient
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.evaluation.verified import reset_copy, score
from web2kg.evaluation.webnavigator import run_original
from web2kg.stage1.function_cards import file_hash
from web2kg.stage1.site_framework import load_framework
from web2kg.stage2.recovery import ExperienceStore, learn_experience, memory_namespace
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.visualization.trajectory_report import export_report

ARMS = ('ours', 'webnavigator')


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def cards_path(graph):
    return graph / read(graph / 'function-cards/active.json')['cards']


def public_tasks(verified, seed):
    dataset = verified / 'assets/dataset'
    hard = set(read(dataset / 'subsets/webarena-verified-hard.json')['task_ids'])
    # Filter metadata before reading/retaining intents. No evaluator fields reach the worker.
    chosen = [t for t in read(dataset / 'webarena-verified.json') if t['task_id'] in hard and t['sites'] == ['reddit']]
    chosen.sort(key=lambda t: t['task_id'])
    random.Random(seed).shuffle(chosen)
    return [{'task_id': t['task_id'], 'sites': t['sites'], 'intent': t['intent'],
             'start_urls': [u.replace('__REDDIT__', 'http://127.0.0.1:10499') for u in t['start_urls']]}
            for t in chosen]


def copy_initial(source, target):
    """Copy only current knowledge, retaining all old experiments at their original paths."""
    index = read(source / 'index.json')
    selected = ['graph.json', 'index.json', index['vectors'], 'snapshots', 'transitions',
                'templates/active.json', read(source / 'templates/active.json')['directory'],
                'function-cards/active.json', str(cards_path(source).parent.relative_to(source)),
                'site-framework/active.json', str(Path(read(source / 'site-framework/active.json')['file']).parent)]
    for name in selected:
        a, b = source / name, target / name
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(a, b) if a.is_dir() else shutil.copy2(a, b)
    graph = Graph.load(target)
    if graph.data.get('execution_update') or graph.data.get('experience_namespace'):
        raise ValueError('Initial graph contains task execution increments')
    knowledge = TemplateKnowledge(graph, cards_path(target), EventLog(target / 'audit'))
    if not load_framework(graph):
        raise ValueError('Initial global framework missing')
    return dict(states=len(graph.nodes), urls=len({n['url'] for n in graph.nodes.values()}),
                edges=len(graph.data['edges']), templates=len(knowledge.groups), cards=len(knowledge.cards),
                indexed=len(index['ids']), framework='retained', prior_task_experiences=0)


def prepare(batch):
    import yaml
    batch.mkdir(parents=True, exist_ok=False)
    verified = Path('D:/Agent/MyCode/web2kg/experiments/webarena-verified')
    original = ROOT / 'runs/webnavigator-original-hard10-v1'
    source = ROOT / 'runs/webnavigator-reddit-llm-120-v4'
    seed = 20261014
    tasks = public_tasks(verified, seed)
    initial = batch / 'initial-graph'
    inventory = copy_initial(source, initial)
    base = batch / 'webnavigator'
    for name in ('upstream', 'import-stubs', 'released-graph'):
        shutil.copytree(original / name, base / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'events.jsonl'))
    cfg = yaml.safe_load((original / 'config.yml').read_text(encoding='utf-8'))
    cfg['logdir'] = cfg['agent']['others']['logdir'] = str(base / 'upstream-logs')
    cfg['agent']['actor']['domain_paths']['reddit'] = str(base / 'released-graph')
    cfg['env']['max_browser_rows'] = 1500
    (base / 'config.yml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    for arm in ARMS:
        (batch / arm).mkdir(exist_ok=True)
        write_json(batch / arm / 'public-inputs.json', tasks)
    protected = {}
    for folder in (initial, base / 'upstream', base / 'import-stubs', base / 'released-graph'):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix != '.pyc' and p.name != 'events.jsonl':
                protected[str(p)] = file_hash(p)
    for p in [base / 'config.yml', *(batch / a / 'public-inputs.json' for a in ARMS),
              source / 'graph.json', source / 'index.json', cards_path(source),
              source / 'templates' / 'active.json', source / read(source / 'site-framework/active.json')['file']]:
        protected[str(p)] = file_hash(p)
    ids = [t['task_id'] for t in tasks]
    schedule = [{'arm': a, 'task_id': key} for i, key in enumerate(ids)
                for a in (ARMS if i % 2 == 0 else ARMS[::-1])]
    common = dict(workspace=str(ROOT), source_sha256=source_hash(), protected_inputs=protected,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        graph=str(initial), task_ids=ids, seed=seed, model='gpt-5.6-terra', max_steps=30,
        max_calls=31, output_tokens_per_call=2048, top_k=5, ax_line_limit=1500,
        retry_connections=True, stop_file=str(batch / 'STOP'), max_new_task_starts=2 * len(ids),
        encoder_endpoint=read(initial / 'index.json')['endpoint'],
        encoder_python='D:/Anaconda/envs/web2kg-jina-gpu/python.exe',
        container='web2kg-template-eval', owner='template-agent-hard10-v1',
        image_id='sha256:0594908059a03e5f610005689440a95c05e776e734006197c7b1eba12734cb46',
        backend='http://127.0.0.1:10599', graph_origin='http://127.0.0.1:10499',
        verified_root=str(verified), price_cny=None,
        authorization='User requested all Reddit Hard tasks, original WebNavigator control, Terra execution/diagnosis, Luna auxiliary, AX1500, clean initial graph, cumulative growth and persistent connection retries.',
        exposure='Previously exposed benchmark; full system developmental comparison, not blind/generalization or component-causal evidence.',
        retry_policy='Identical logical request on transport/timeout/HTTP408,429,500,502,503,504; backoff 2..60s indefinitely. All physical requests/unknown usage logged. No browser action replay for transport retries. Permanent auth/credit/bad request stops. STOP file cancels.',
        call_budget_unit='Logical calls exclude only transient failed transport attempts. No finite physical-request or wall-time cap; user explicitly authorized continuing connection retries. Successful malformed outputs still consume budget.',
        initial_inventory=inventory, schedule=schedule,
        comparison_limits=['Ours starts with120-state v4 plus current24 templates/cards/framework; original released graph225 nodes/201 encoded screenshots.',
            'Baseline image encoder provenance is unavailable; original released vectors retained as requested; local Jina v4 query encoder.',
            'Ours adds Luna annotations/cards, Terra diagnosis/learning, attached notes and safe replay. Whole system comparison, not matched total compute.',
            'Ours retains task graph/experience in fixed order; original graph stays fixed. All website state resets between tasks/arms.',
            'Experiences use only official score=1, not evaluator reference answers. This is outcome-supervised online development, not an independent test.',
            'Initial website framework remains initial coverage; new templates use refreshed cards, not a regenerated global description.'])
    write_json(batch / 'protocol.json', common)
    ours = dict(common, logical_origin=common['graph_origin'], cards=str(cards_path(initial)), framework='auto',
        update_graph=True, finalize_graph_assets=True, transition_calls=30, card_calls=10,
        diagnosis_calls=5, diagnosis_model='gpt-5.6-terra', max_diagnoses=3, recovery_actions=12,
        max_total_model_calls=76, experience_store=str(batch / 'ours-experiences.json'))
    control = dict(common, logical_origin=common['backend'], max_total_model_calls=31)
    write_json(batch / 'ours/protocol.json', ours)
    write_json(base / 'protocol.json', control)
    write_json(batch / 'state.json', {'tasks': {}, 'current_graph': str(initial),
        'current_graph_sha256': file_hash(initial / 'graph.json'), 'status': 'registered'})
    report(batch)
    print(json.dumps({'registered_tasks': len(ids), 'new_starts': len(schedule), 'initial': inventory}))


def repair_schedule(protocol, state):
    if state.get('status') != 'user_stopped' or any(r['status'] != 'completed' for r in state['tasks'].values()):
        raise ValueError('Stop the old queue at a completed task boundary before registering a repair')
    # Repeat the entire baseline, regardless of score; our completed tasks are never rerun.
    return [item for item in protocol['schedule'] if item['arm'] == 'webnavigator'
            or f"ours/{item['task_id']}" not in state['tasks']]


def prepare_repair(batch, previous):
    import yaml
    previous = previous.resolve()
    old, state = read(previous/'protocol.json'), read(previous/'state.json')
    schedule = repair_schedule(old, state)
    if any(file_hash(p) != h for p, h in old['protected_inputs'].items()):
        raise ValueError('Old registered assets changed')
    snapshot = previous/'source-snapshot'
    manifest = read(snapshot/'manifest.json')
    if any(file_hash(snapshot/p) != h for p,h in manifest.items()):
        raise ValueError('Original source snapshot changed')
    batch.mkdir(parents=True, exist_ok=False)
    frozen = batch/'frozen-ours-runtime'
    shutil.copytree(snapshot, frozen)
    # Credentials stay outside the source snapshot. The unchanged old worker expects ROOT/auth.
    quote = lambda p: "'" + str(p).replace("'", "''") + "'"
    subprocess.run(['powershell.exe','-NoProfile','-Command',
        "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path " + quote(frozen/'auth')
        + ' -Target ' + quote(ROOT/'auth') + ' | Out-Null'], check=True, capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW)
    base = batch/'webnavigator'
    for name in ('upstream','import-stubs','released-graph'):
        shutil.copytree(previous/'webnavigator'/name, base/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    cfg = yaml.safe_load((previous/'webnavigator/config.yml').read_text('utf-8'))
    cfg['logdir'] = cfg['agent']['others']['logdir'] = str(base/'upstream-logs')
    cfg['agent']['actor']['domain_paths']['reddit'] = str(base/'released-graph')
    (base/'config.yml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    for arm in ARMS:
        (batch/arm).mkdir(exist_ok=True)
        shutil.copy2(previous/arm/'public-inputs.json', batch/arm/'public-inputs.json')
    protected = dict(old['protected_inputs'])
    for folder in (base/'upstream',base/'import-stubs',base/'released-graph'):
        protected.update({str(p):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.suffix != '.pyc'})
    protected.update({str(frozen/p):h for p,h in manifest.items()})
    for p in [base/'config.yml',*(batch/a/'public-inputs.json' for a in ARMS)]:
        protected[str(p)] = file_hash(p)
    inherited = {}
    for key,row in state['tasks'].items():
        if key.startswith('ours/'):
            folder = previous/key
            inherited[key] = {**row, 'evidence_directory':str(folder), 'inherited_from':str(previous)}
            for name in ('result.json','official.json','events.jsonl'):
                protected[str(folder/name)] = file_hash(folder/name)
    memory = previous/'ours-experiences.json'
    if memory.exists():
        shutil.copy2(memory,batch/'ours-experiences.json')
        protected[str(memory)] = file_hash(memory)
    protocol = {**old, 'source_sha256':source_hash(), 'protected_inputs':protected,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'schedule':schedule, 'max_new_task_starts':len(schedule), 'stop_file':str(batch/'STOP'),
        'frozen_ours_runtime':str(frozen), 'frozen_ours_source_sha256':old['source_sha256'],
        'repair_of':str(previous), 'inherited_ours_tasks':len(inherited),
        'authorization':'2026-10-14 user requested pausing WebNavigator, diagnosing faults, then rerunning. Repeat all baseline tasks once; continue only unstarted ours tasks with its exact frozen original implementation.',
        'comparison_limits':[*old['comparison_limits'],
            'Baseline rerun repairs transport/action syntax only; all42 original scores retained separately. Ours completed tasks are retained in chronological order and remaining tasks use the original frozen runtime. New baseline runs later in time; not a contemporaneously interleaved blind comparison.']}
    write_json(batch/'protocol.json',protocol)
    control = {**read(previous/'webnavigator/protocol.json'), **{k:protocol[k] for k in
               ('source_sha256','source_commit','protected_inputs','stop_file')}, 'actor_compatibility':True}
    write_json(base/'protocol.json',control)
    ours = {**read(previous/'ours/protocol.json'), 'stop_file':str(batch/'STOP'),
            'experience_store':str(batch/'ours-experiences.json'), 'protected_inputs':protected}
    write_json(batch/'ours/protocol.json',ours)
    write_json(batch/'state.json',{'tasks':inherited,'current_graph':state['current_graph'],
        'current_graph_sha256':state['current_graph_sha256'],'status':'registered'})
    paths = sorted([*(ROOT/'web2kg').rglob('*.py'),*(ROOT/'scripts').glob('*.py'),
        *[p for p in (ROOT/'third_party/webnavigator').rglob('*') if p.is_file() and p.suffix in {'.py','.txt','.json','.js'}],
        ROOT/'examples/webnavigator-execution.yml'])
    captured = {}
    for p in paths:
        relative=p.relative_to(ROOT)
        target=batch/'source-snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);captured[relative.as_posix()]=file_hash(p)
    write_json(batch/'source-snapshot/manifest.json',captured)
    report(batch)
    print(json.dumps({'new_baseline_starts':42,'unstarted_ours':len(schedule)-42,'inherited_ours':len(inherited)}))


def task_stats(folder, task_id, state):
    result, official = read(folder / 'result.json', {}), read(folder / 'official.json', {})
    events = []
    for name in ('events.jsonl', 'graph-annotations/events.jsonl', 'graph-cards/events.jsonl', 'diagnosis-calls/events.jsonl'):
        events += read_events(folder / name)
    starts = [e for e in events if e['event'] == 'responses_started']
    received = [e for e in events if e['event'] == 'responses_received']
    known = {e['request_id'] for e in received if e.get('usage') and 'input_tokens' in e['usage'] and 'output_tokens' in e['usage']}
    channels = {}
    for model in sorted({e.get('model', 'unknown') for e in starts}):
        req = {e['request_id'] for e in starts if e.get('model') == model}
        responses = [e for e in received if e['request_id'] in req]
        channels[model] = dict(requests=len(req), unknown_usage=len(req-known),
            input_tokens=sum(e.get('usage', {}).get('input_tokens', 0) for e in responses),
            output_tokens=sum(e.get('usage', {}).get('output_tokens', 0) for e in responses))
    return dict(task_id=task_id, status=state.get('status', 'not_started'), official_score=official.get('score'),
        agent_status=result.get('status'), steps=result.get('steps'), atomic_actions=result.get('atomic_actions'),
        seconds=result.get('seconds'), total_seconds=result.get('total_seconds', result.get('seconds')),
        queue_seconds=state.get('seconds'), physical_requests=len(starts), responses=len(received),
        unknown_usage=len(starts)-len(known), models=channels,
        actual_models=sorted({str(e.get('actual_model')) for e in received}),
        retry_attempts=sum(e['event'] == 'responses_retry' for e in events),
        connection_retry_seconds=sum(e.get('seconds', 0) for e in events if e['event'] == 'responses_retry_wait_finished'),
        graph_before=state.get('graph_before'), graph_after=result.get('execution_graph'),
        recovery=result.get('recovery'), learning=read(folder / 'recovery/learning.json'),
        navigation=result.get('navigation'), navigation_attempts=sum(e['event'] in {'navigation','original_navigation'} for e in events),
        navigation_reported_success=sum(e.get('success') is True or e.get('returned') is True for e in events
            if e['event'] in {'navigation','original_navigation'}))


def wilson(success, total):
    if not total:
        return None
    z = 1.959963984540054
    p, den = success/total, 1+z*z/total
    center = (p+z*z/(2*total))/den
    half = z*((p*(1-p)/total+z*z/(4*total*total))**.5)/den
    return [center-half, center+half]


def report(batch):
    protocol, state = read(batch / 'protocol.json'), read(batch / 'state.json', {'tasks': {}})
    summaries = {}
    for arm in ARMS:
        rows = []
        for t in protocol['task_ids']:
            entry = state['tasks'].get(f'{arm}/{t}', {})
            folder = Path(entry.get('evidence_directory', batch/arm/str(t)))
            row = task_stats(folder,t,entry)
            row['evidence_directory'] = str(folder)
            rows.append(row)
        successes = sum(r['official_score'] == 1 for r in rows)
        scored = sum(r['official_score'] is not None for r in rows)
        summary = dict(planned=len(rows), official_successes=successes, official_scored=scored,
            unrun=sum(r['status'] == 'not_started' for r in rows),
            started_unscored=sum(r['status'] != 'not_started' and r['official_score'] is None for r in rows),
            descriptive_wilson95_scored=wilson(successes, scored), tasks=rows)
        write_json(batch / arm / 'summary.json', summary)
        summaries[arm] = summary
    pairs = [(a,b) for a,b in zip(summaries['ours']['tasks'], summaries['webnavigator']['tasks'])
             if a['official_score'] is not None and b['official_score'] is not None]
    flips = {'paired_scored': len(pairs), 'both_success': 0, 'ours_only': 0, 'webnavigator_only': 0, 'both_failure': 0}
    for a,b in pairs:
        flips[['both_failure','webnavigator_only','ours_only','both_success'][2*int(a['official_score']==1)+int(b['official_score']==1)]] += 1
    write_json(batch / 'summary.json', {'arms': summaries, 'paired': flips, 'status': state.get('status'),
        'limits': protocol['comparison_limits'], 'cost_cny': None})
    lines = ['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Reddit Hard 完整对比</title>',
        '<style>body{font:16px/1.6 system-ui;margin:30px auto;max-width:1200px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
        '<h1>Reddit Hard · 当前方法 / 原版 WebNavigator</h1>',
        '<p>42题×两组，AX1500；Terra执行及诊断，Luna维护。清洁初图保留模板卡与全局视野；本方法按固定顺序累积任务图及成功修复经验。旧任务已暴露，不是盲测。两组整体配置不同，不能把差异单独归因于根因定位。</p>',
        '<p>连接错误持续重试；已用token是有usage的响应之和，未知请求单列，费用未知。导航到达不等于任务成功。区间仅描述小样本比例，不支持泛化结论。</p>']
    if protocol.get('repair_of'):
        lines.append('<p>WebNavigator 修复后整组重跑；本方法已完成任务直接保留，剩余任务沿用原冻结源码。旧对照结果另存，不择优合并。两组运行时间不完全交错。</p>')
    for arm, summary in summaries.items():
        lines.append(f'<h2>{arm}: 官方 {summary["official_successes"]}/{summary["planned"]}，已评分{summary["official_scored"]}，未运行{summary["unrun"]}，已开始未评分{summary["started_unscored"]}</h2><p><a href="{arm}/trajectories/index.html">展开逐步轨迹 / 模型上下文 / 截图 / AX</a></p><table><tr><th>任务</th><th>状态</th><th>官方分</th><th>步数</th><th>秒（含收尾）</th><th>物理请求 / 重试 / 未知</th><th>已知输入 / 输出token</th><th>诊断/返回</th></tr>')
        for r in summary['tasks']:
            tok = [sum(x[k] for x in r['models'].values()) for k in ('input_tokens','output_tokens')]
            trajectory = Path(r['evidence_directory']).parent/'trajectories'/f'task-{r["task_id"]}.html'
            href = Path(os.path.relpath(trajectory,batch)).as_posix()
            vals = [f'<a href="{html.escape(href)}">{r["task_id"]}</a>',
                html.escape(str(r['status'])),str(r['official_score']),str(r['steps']),str(r['total_seconds']),
                f'{r["physical_requests"]}/{r["retry_attempts"]}/{r["unknown_usage"]}', f'{tok[0]}/{tok[1]}',
                html.escape(json.dumps(r['recovery'], ensure_ascii=False))]
            lines.append('<tr>' + ''.join('<td>'+v+'</td>' for v in vals) + '</tr>')
        lines.append('</table>')
    lines.append('<h2>配对成败互换</h2><pre>'+html.escape(json.dumps(flips,ensure_ascii=False,indent=2))+'</pre>')
    (batch / 'report.html').write_text(''.join(lines), encoding='utf-8')
    return summaries


def run_worker(batch, arm, task_id):
    config = read(batch / arm / str(task_id) / 'input.json')
    check_inputs(config)
    if arm == 'ours':
        template_worker(batch / arm, config, task_id)
    else:
        task = next(t for t in read(batch / arm / 'public-inputs.json') if t['task_id'] == task_id)
        run_original(batch / arm, config, task)


def advance_graph(folder):
    result = read(folder / 'result.json', {})
    update = result.get('execution_graph') or {}
    if update.get('index_status') != 'completed':
        raise RuntimeError('Cumulative graph index not complete; preserve task and inspect before advancing')
    graph = Path(update['directory'])
    TemplateKnowledge(Graph.load(graph), cards_path(graph), EventLog(folder / 'handoff-audit'))
    load_framework(Graph.load(graph))
    return graph


def queue(batch, one_task=False):
    import msvcrt
    protocol = read(batch / 'protocol.json')
    with (batch / 'queue.lock').open('a+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        check_inputs(protocol)
        state = read(batch / 'state.json')
        encoder = LocalEncoder(Graph.load(Path(protocol['graph'])), ROOT, protocol['encoder_python'])
        try:
            encoder.ensure()
            for item in protocol['schedule']:
                arm, task_id = item['arm'], item['task_id']
                key = f'{arm}/{task_id}'
                if key in state['tasks']:
                    if state['tasks'][key]['status'] != 'completed':
                        raise RuntimeError('Started task has unresolved evidence; never restart it automatically')
                    continue
                if Path(protocol['stop_file']).exists():
                    state['status'] = 'user_stopped'; break
                check_inputs(protocol)
                config = read(batch / arm / 'protocol.json')
                graph = Path(state['current_graph'])
                if file_hash(graph / 'graph.json') != state['current_graph_sha256']:
                    raise ValueError('Cumulative graph changed outside the queue')
                if arm == 'ours':
                    config.update(graph=str(graph), cards=str(cards_path(graph)))
                folder = batch / arm / str(task_id)
                folder.mkdir(exist_ok=False)
                write_json(folder / 'input.json', config)
                started = time.monotonic()
                row = {'status': 'started', 'time': time.time(), 'graph_before': str(graph) if arm == 'ours' else str(batch / arm / 'released-graph')}
                state['tasks'][key] = row; state['status'] = 'running'
                write_json(batch / 'state.json', state)
                print(f'{key} started ({len(state["tasks"])}/{len(protocol["schedule"])})', flush=True)
                try:
                    row['container_id'] = reset_copy(config['container'], config['image_id'], config['backend'],
                        10598, config['logical_origin'], config['owner'])
                    with (folder / 'console.log').open('w', encoding='utf-8') as out:
                        worker_script = (Path(protocol['frozen_ours_runtime'])/'scripts/run_hard_comparison.py'
                                         if arm == 'ours' and protocol.get('frozen_ours_runtime') else Path(__file__).resolve())
                        child = subprocess.Popen([sys.executable, str(worker_script), '--directory', str(batch),
                            '--worker', arm, str(task_id)], cwd=ROOT, stdout=out, stderr=out,
                            env={**os.environ, 'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'}, creationflags=subprocess.CREATE_NO_WINDOW)
                        row['worker_pid'] = child.pid
                        write_json(batch / 'state.json', state)
                        code = child.wait()  # Connection waits have no wall-time kill, as requested.
                    if code:
                        raise RuntimeError(f'Worker exited {code}; inspect console, never restart task')
                    official = score(config['verified_root'], batch / arm, task_id, config['logical_origin'])
                    write_json(folder / 'official.json', official)
                    if official.get('score') is None:
                        raise RuntimeError('Official evaluator did not score this task')
                    result = read(folder / 'result.json')
                    events = [e for p in [folder/'events.jsonl',folder/'graph-annotations/events.jsonl',
                              folder/'graph-cards/events.jsonl',folder/'diagnosis-calls/events.jsonl'] for e in read_events(p)]
                    if any(e['event']=='responses_failed' and e.get('http_status') in {400,401,402,403,404}
                           for e in events) or result.get('provider_failure',{}).get('kind') == 'cancelled':
                        raise RuntimeError('Permanent service error or requested stop; scoring retained')
                    if arm == 'ours':
                        next_graph = advance_graph(folder)
                        learner = ResponsesClient(ROOT / 'auth/llm-crawl.env', EventLog(folder/'diagnosis-calls'),
                            config['diagnosis_calls'], 2048, model=config['diagnosis_model'],
                            retry_connections=True, stop_file=config['stop_file'])
                        memory = ExperienceStore(config['experience_store'], memory_namespace(Graph.load(graph)))
                        learn_experience(folder, learner, memory, success=official.get('score') == 1,
                                         verifier='WebArena-Verified official score=1')
                        state.update(current_graph=str(next_graph), current_graph_sha256=file_hash(next_graph/'graph.json'))
                    row.update(status='completed', seconds=round(time.monotonic()-started,3))
                    write_json(batch/'state.json',state)
                    report(batch)
                    export_report([batch / arm], batch / arm / 'trajectories')
                    print(f'{key}: official={official.get("score")}, status={result["status"]}', flush=True)
                    if one_task:
                        state['status']='checkpoint'; break
                except Exception as error:
                    row.update(status='needs_attention', error=type(error).__name__+': '+str(error), seconds=round(time.monotonic()-started,3))
                    state['status']='needs_attention'
                    write_json(batch/'state.json',state)
                    raise
            else:
                state['status']='completed'
                check_inputs(protocol)
                write_json(batch/'integrity_closeout.json',{'protected_inputs_verified':len(protocol['protected_inputs']),
                    'source_verified':True,'initial_graph_unchanged':True,'time':time.time()})
            write_json(batch/'state.json',state)
        finally:
            encoder.close()
            report(batch)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--repair-baseline-from',type=Path)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--one-task',action='store_true',help='Finish one unstarted task, score it, then stop at a boundary')
    parser.add_argument('--worker',nargs=2,metavar=('ARM','TASK'))
    args=parser.parse_args(); batch=args.directory.resolve()
    if args.repair_baseline_from:
        prepare_repair(batch,args.repair_baseline_from)
    elif args.prepare:
        prepare(batch)
    elif args.worker:
        run_worker(batch,args.worker[0],int(args.worker[1]))
    elif args.live:
        queue(batch,one_task=args.one_task)
    else:
        report(batch)


if __name__=='__main__':
    main()
