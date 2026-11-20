"""Independent ten-task development test of the 225-state graph and latest maintenance."""
import argparse
from datetime import datetime, timezone
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
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from scripts import run_verified_template as base
from scripts.qwen_method_transport import clients
from scripts.run_midterm_qwen import freeze, queue_lock
from scripts.run_hard_comparison import cards_path, advance_graph, task_stats
from scripts.run_method_qwen_hard import export_existing, knowledge_manifest, service_error
from web2kg.common.activity import read_events
from web2kg.common.storage import Graph, EventLog, write_json
from web2kg.evaluation.verified import reset_copy, score
from web2kg.stage1.function_cards import file_hash, load_layer, build_cards
from web2kg.stage1.site_framework import build_framework
from web2kg.stage2.recovery import ExperienceStore, memory_namespace, learn_experience


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def hashes(folder):
    return {str(p.resolve()): file_hash(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def selected_tasks(project, logical_origin, seed=20261104, limit=10):
    dataset = project / 'experiments/webarena-verified/assets/dataset'
    hard = set(read(dataset / 'subsets/webarena-verified-hard.json')['task_ids'])
    rows = sorted((t for t in read(dataset / 'webarena-verified.json')
                   if t['task_id'] in hard and t['sites'] == ['reddit']), key=lambda t: t['task_id'])
    random.Random(seed).shuffle(rows)
    return [dict(task_id=t['task_id'], sites=t['sites'], intent=t['intent'],
                 start_urls=[u.replace('__REDDIT__', logical_origin) for u in t['start_urls']]) for t in rows[:limit]]


def register(batch, model, repair_from=None):
    if batch.exists() or not batch.is_relative_to(ROOT / 'runs'):
        raise ValueError('Use a fresh workspace run directory')
    source = ROOT / 'runs/reddit-complete-index-20261104-v1/reddit-jev-flash-expand225-20261104-v2/graph'
    prior = ROOT / 'runs/reddit-jev-flash-expand225-20261104-v2'
    verification = read(ROOT / 'runs/navigation-upgrade-20261104-v1/verification.json')
    repair = read(repair_from / 'failure-diagnosis.json') if repair_from else None
    if repair_from:
        verify(repair_from)
        if not repair or repair.get('status') != 'confirmed' or read(repair_from/'state.json')['status'] == 'running':
            raise ValueError('A stopped, diagnosed version is required')
    for path, expected in verification['source_sha256'].items():
        expected = (repair or {}).get('repaired_source_sha256', {}).get(path, expected)
        if file_hash(ROOT / path) != expected:
            raise ValueError('Tested implementation changed: ' + path)
    for path, expected in (repair or {}).get('repaired_source_sha256', {}).items():
        if file_hash(ROOT/path) != expected:
            raise ValueError('Repaired implementation changed: ' + path)
    graph = Graph.load(source)
    index = read(source / 'index.json')
    if len(graph.nodes) != 225 or set(index['ids']) != set(graph.nodes):
        raise ValueError('Expected complete 225-state source')
    batch.mkdir()
    initial = batch / 'initial-graph'
    shutil.copytree(source, initial)
    logical = graph.nodes[graph.data['root']]['url'].rstrip('/')
    tasks = selected_tasks(ROOT, logical)
    write_json(batch / 'public-inputs.json', tasks)
    runtime = freeze(batch)
    encoder = batch / 'encoder-runtime/scripts'
    encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py', 'jina_gpu.py', 'jina_vision_fp32.py'):
        shutil.copy2(prior / 'source/scripts' / name, encoder / name)
    plan = read(prior / 'plan.json')
    protocol = dict(version='hard10-225-navigation-v1', project=str(ROOT), runtime=str(runtime),
        authorization='2026-11-04 user authorized 10 Reddit Hard tasks on the 225-node version; preserve initial graph.',
        created_at=datetime.now(timezone.utc).isoformat(), model=model, auxiliary_model='qwen3.7-flash',
        diagnosis_model='qwen3.7-flash', execution_provider='qwen', auxiliary_provider='qwen', provider='bailian',
        credential_file=str(ROOT / 'auth/qwen-hard.env'), task_ids=[t['task_id'] for t in tasks], seed=20261104,
        source_graph=str(source), initial_graph=str(initial), graph=str(initial), cards=str(cards_path(initial)),
        verified_root=str(ROOT / 'experiments/webarena-verified'), site='reddit',
        container='web2kg-template-eval', owner='template-agent-hard10-v1', image_id=plan['image_id'],
        backend='http://127.0.0.1:10599', logical_origin=logical,
        encoder_endpoint='http://127.0.0.1:8797/v1/embeddings', encoder_expected=index['encoder'],
        encoder_python='D:/Anaconda/envs/web2kg-jina-gpu/python.exe', encoder_runtime=str(encoder.parent),
        encoder_timeout=1800, gpu_memory='4GiB', max_calls=31, max_steps=30, top_k=5,
        output_tokens_per_call=4096, auxiliary_output_tokens=4096, diagnosis_output_tokens=4096,
        transition_calls=None, card_calls=None, diagnosis_calls=5, max_diagnoses=3, recovery_actions=12,
        update_graph=True, finalize_graph_assets=True, max_graph_nodes=225, retention_half_life=20,
        template_clustering='average', template_tolerance=.05, template_refresh_min=5, template_refresh_fraction=.2,
        framework='auto', framework_refresh_updates=0, ax_line_limit=800, ax_search=True,
        experience_store=str(batch / 'experience.json'), stop_file=str(batch / 'STOP'),
        max_retries=0, retry_connections=True, max_output_retries=3, request_read_timeout=300,
        validate_verified_stop=bool(repair_from),
        new_card_preparation='Regenerate all existing template cards with up to five images and the framework on a separate starting graph; cost separate.',
        stop_policy='No wall-time/physical-call kill. Same-input transient transport retries; bounded schema regeneration. No browser retries. Stop on first engineering/permanent/asset error. Never restart a started task.',
        knowledge_policy='Cumulative task graphs and empty initial experience only inside this batch; initial graph and source immutable.',
        limits='Previously exposed development tasks, no baseline/causal/generalization claim; initial construction costs inherited by reference.')
    if repair_from:
        prepared = repair_from/'starting-graph'
        if file_hash(prepared/'graph.json') != file_hash(initial/'graph.json'):
            raise ValueError('Prepared graph must contain no task updates')
        from web2kg.stage2.template_context import TemplateKnowledge
        from web2kg.stage1.site_framework import load_framework
        TemplateKnowledge(Graph.load(prepared), cards_path(prepared), EventLog(batch/'registration-events'))
        if not load_framework(Graph.load(prepared)):
            raise ValueError('Prepared framework missing')
        protocol.update(prepared_graph=str(prepared), failed_version=str(repair_from),
            new_card_preparation='Reuse the verified eight five-page cards/framework prepared before any v1 task; no new preparation model calls.',
            restart_policy='Full same ten tasks from original 225 states and empty experience; no prior scores or task graph inherited.')
        write_json(batch/'repair-lineage.json',repair)
    write_json(batch / 'protocol.json', protocol)
    write_json(batch / 'run_material_passport.json', dict(mode='run', status='registered',
        source='Local WebArena-Verified single-site Reddit Hard subset, seeded sample before outcomes.',
        method='Latest dev2 implementation, retained 225-node graph, complete local Jina index.',
        model_roles=dict(actor=model, selector=model, description='qwen3.7-flash', cards='qwen3.7-flash',
                        framework='qwen3.7-flash', diagnosis_learning='qwen3.7-flash'),
        initial_graph_preserved=True, historical_construction=str(prior / 'summary.json'),
        paid_preflight_calls=0, official_scorer_only=True, limitations=protocol['limits']))
    protected = {**hashes(source), **hashes(initial), **hashes(encoder.parent)}
    if repair_from:
        protected.update(hashes(repair_from/'starting-graph'))
        for name in ('failure-diagnosis.json','state.json','summary.json','STOP','final-audit.json'):
            protected[str(repair_from/name)]=file_hash(repair_from/name)
    for name in ('protocol.json', 'public-inputs.json', 'run_material_passport.json'):
        protected[str(batch / name)] = file_hash(batch / name)
    for path in (ROOT / 'runs/active-graph.json', ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt'):
        protected[str(path)] = file_hash(path)
    write_json(batch / 'protected-inputs.json', protected)
    write_json(batch / 'state.json', dict(status='registered', tasks={}, current_graph=str(initial)))
    print(json.dumps({'registered': str(batch), 'tasks': protocol['task_ids'], 'protected_files':len(protected)}))


def verify(batch, *, full=True, record=True):
    started = time.monotonic()
    config = read(batch / 'protocol.json')
    scoped = config.get('integrity_policy') == 'boundary-v1'
    manifest = 'protected-inputs.json' if full or not scoped else 'runtime-inputs.json'
    if scoped and not full:
        expected = read(batch / 'protected-inputs.json')[str((batch / manifest).resolve())]
        if file_hash(batch / manifest) != expected:
            raise ValueError('Runtime input manifest changed')
    for rel, expected in read(Path(config['runtime']) / 'manifest.json').items():
        if file_hash(Path(config['runtime']) / rel) != expected:
            raise ValueError('Frozen source changed: ' + rel)
    inputs = read(batch / manifest)
    for path, expected in inputs.items():
        if file_hash(path) != expected:
            raise ValueError('Protected input changed: ' + path)
    state = read(batch / 'state.json')
    if state.get('knowledge_manifest'):
        if file_hash(state['knowledge_manifest']) != state['knowledge_manifest_sha256']:
            raise ValueError('Knowledge handoff manifest changed')
        for path, expected in read(state['knowledge_manifest']).items():
            if file_hash(path) != expected:
                raise ValueError('Knowledge handoff changed: ' + path)
    if scoped and record:
        EventLog(batch / 'timings').emit('integrity_verified', scope='full' if full else 'task',
            protected_files=len(inputs), seconds=time.monotonic()-started)
    return config, state


def model_class(batch, config, model):
    routed = clients(config['credential_file'], batch / 'routing-state.json', provider='bailian', model_override=model)
    class FixedQwen(routed):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.request_timeout = (10, config['request_read_timeout'])
            self.configure_output_retries(config['max_output_retries'])
            self.validate_verified_stop = config.get('validate_verified_stop', False)
            self.unified_verified_stop = config.get('unified_verified_stop', False)
    return FixedQwen


def statistics(folder):
    events = [e for p in folder.rglob('events.jsonl') for e in read_events(p)] if folder.exists() else []
    received = [e for e in events if e['event'] == 'responses_received']
    received_ids = {e['request_id'] for e in received}
    known = {e['request_id'] for e in received if e.get('usage') and
             {'input_tokens', 'output_tokens'} <= set(e['usage'])}
    starts = [e for e in events if e['event'] == 'responses_started']
    return dict(physical_requests=len(starts), unknown_usage=sum(e['request_id'] not in known for e in starts),
        input_tokens=sum(e.get('usage', {}).get('input_tokens', 0) for e in received),
        output_tokens=sum(e.get('usage', {}).get('output_tokens', 0) for e in received),
        actual_models=sorted({e.get('actual_model', 'unknown') for e in received}),
        request_seconds=sum(e.get('seconds', 0) for e in events if e['event'] == 'responses_received' or
                            (e['event'] == 'responses_failed' and e.get('request_id') not in received_ids)),
        output_retries=sum(e.get('output_retry_number', 0) > 0 for e in starts),
        transport_retries=sum(e['event'] == 'responses_retry' for e in events))


def report(batch):
    config, state = read(batch / 'protocol.json'), read(batch / 'state.json')
    title = html.escape(config.get('report_title', 'Reddit Hard10 · 225节点'))
    rows = []
    for task_id in config['task_ids']:
        folder = batch / str(task_id)
        row = task_stats(folder, task_id, state['tasks'].get(str(task_id), {}))
        row['roles'] = {role: statistics(folder / sub) for role, sub in
                       [('description','graph-annotations'),('cards','graph-cards'),('diagnosis_learning','diagnosis-calls')]}
        row['output_retries'] = statistics(folder)['output_retries']
        recovery = read(folder / 'recovery/recovery.json', {})
        row['recovery'] = {k: recovery.get(k) for k in ('decisions', 'attempted', 'return', 'step_refund', 'stopped')}
        row['recovery']['learning'] = read(folder / 'recovery/learning.json')
        row['recovery']['report'] = f'{task_id}/recovery/report.html' if (folder / 'recovery/report.html').exists() else None
        row['inherited'] = False
        row['source_version'] = config['version'] if 'version' in config else None
        row['graph_capacity'] = config.get('max_graph_nodes', 225)
        row['actor_model'] = config['model']
        rows.append(row)
    inherited = []
    for item in config.get('inherited_results', []):
        folder = Path(item['directory'])
        row = task_stats(folder, item['task_id'], item['state'])
        row.update(inherited=True, source_directory=str(folder), output_retries=statistics(folder)['output_retries'])
        source_config = read(folder.parent / 'protocol.json', {})
        row.update(source_version=source_config.get('version'), graph_capacity=source_config.get('max_graph_nodes'),
                   actor_model=source_config.get('model'))
        inherited.append(row)
    rows = inherited + rows
    excluded = []
    for item in config.get('excluded_attempts', []):
        folder = Path(item['directory'])
        row = task_stats(folder, item['task_id'], item['state'])
        row.update(source_directory=str(folder), exclusion_reason=item['reason'])
        excluded.append(row)
    summary = dict(status=state['status'], planned=len(rows), new_planned=len(config['task_ids']), tasks=rows,
        excluded_attempts=excluded,
        official_successes=sum(r['official_score'] == 1 for r in rows),
        official_scored=sum(r['official_score'] is not None for r in rows),
        completed=sum(r['status'] == 'completed' for r in rows),
        budget_exhausted=sum(r['status'] == 'budget_exhausted' for r in rows),
        preparation=statistics(batch / 'preparation-events'),
        limits=config['limits'], initial_graph=config['initial_graph'], current_graph=state['current_graph'])
    write_json(batch / 'summary.json', summary)
    if any((batch / str(t) / 'result.json').exists() for t in config['task_ids']):
        export_existing(batch)
    table = ''
    for r in rows:
        origin = Path(r['source_directory']).parent if r.get('inherited') else batch
        trace = origin / 'trajectories' / f"task-{r['task_id']}.html"
        href = html.escape(Path(os.path.relpath(trace, batch)).as_posix(), quote=True)
        label = f"<a href='{href}'>{r['task_id']}</a>" if trace.exists() else str(r['task_id'])
        if r.get('inherited'):
            label += '（保留原结果）'
        table += '<tr>' + ''.join(f'<td>{v}</td>' for v in [label, r['status'], r['official_score'],
            r['steps'], r['total_seconds'], r['physical_requests'], r['unknown_usage'],
            html.escape(str(r.get('source_version') or '未注明'))]) + '</tr>'
    exclusion_text = ''
    for row in excluded:
        trace = Path(row['source_directory']).parent / 'trajectories' / f"task-{row['task_id']}.html"
        href = html.escape(Path(os.path.relpath(trace, batch)).as_posix(), quote=True)
        exclusion_text += (f'<p>另存历史尝试 <a href="{href}">{row["task_id"]}</a>：'
            f'{html.escape(row["exclusion_reason"])}。原官方分 {row["official_score"]}、'
            f'{row["physical_requests"]} 次物理请求及 {row["unknown_usage"]} 次未知用量保留，'
            '不计入当前选定任务的分母，也不从历史尝试和成本中删除。</p>')
    page = ('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60">'
        f'<title>{title}开发验证</title><style>body{{font:17px/1.6 system-ui;max-width:1200px;margin:36px auto;padding:0 20px}}table{{border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ddd}}</style>'
        f'<h1>{title}</h1><p>状态 {html.escape(state["status"])}；完整交接 {summary["completed"]}/{summary["planned"]}，调用额度耗尽 {summary["budget_exhausted"]}，官方成功 {summary["official_successes"]}，已评分 {summary["official_scored"]}。本批新开始上限{summary["new_planned"]}；保留结果单列，未评分不计为失败或成功。</p>'
        f'<p>Actor/选择器 {html.escape(config["model"])}；辅助角色 Qwen3.7-flash，百炼直连。30步、31逻辑Actor/selector调用；诊断/学习共5次。每题重置专用环境，图谱在副本中累积，收尾容量{config.get("max_graph_nodes", 225)}。新节点保护可能导致暂时超额。准备阶段成本另列；诊断详情见逐题summary及轨迹。</p>'
        '<p>开发样本曾经暴露；若保留旧结果，这是用户授权的多版本开发视图，不选择最高分。没有对照组，不推断泛化、组件因果收益或虚构价格。原始图、旧结果和默认发布图保留。</p>'
        + exclusion_text +
        '<table><tr><th>任务/轨迹</th><th>状态</th><th>官方分</th><th>步数</th><th>总秒数</th><th>物理请求</th><th>未知用量</th><th>源码版本</th></tr>'+table+'</table>'
        '<p><a href="summary.json">逐题/模型/角色用量</a> · <a href="initial-graph/viewer.html">保留初始图</a> · <a href="protocol.json">协议</a> · <a href="run_material_passport.json">实验边界</a></p>')
    (batch / 'report.html').write_text(page, encoding='utf-8')
    return summary


def stop_for_error(result, events):
    update = result.get('execution_graph') or {}
    return (service_error(result, events) or result.get('status') == 'model_error' or result.get('execution_graph_error')
        or update.get('description_stopped') or (update.get('cards') or {}).get('stopped')
        or (update.get('card_refresh') or {}).get('unavailable')
        or (update.get('card_refresh') or {}).get('pending')
        or (update.get('template_global') or {}).get('status') == 'failed'
        or (update.get('framework_refresh') or {}).get('status') == 'refresh_failed'
        or update.get('index_status') != 'completed')


def ordinary_call_exhaustion(config, result, events, actor_events):
    """Opt-in terminal task failure; never pass along its incomplete task graph."""
    if config.get('budget_exhaustion_policy') != 'retain_score_keep_previous_handoff':
        return False
    if result.get('status') != 'model_error' or result.get('error') != 'ResponsesError: Responses call budget exhausted':
        return False
    from web2kg.stage2.action_budget import POLICY, action_budget_used
    if any(e.get('action_budget_policy') == POLICY for e in actor_events):
        used = action_budget_used(actor_events)
    else:
        used = sum(e['event'] == 'responses_started' and not e.get('output_retry_number') for e in actor_events)
        if config.get('retry_connections'):
            used -= sum(e['event'] == 'responses_retry' and e.get('persistent', False) for e in actor_events)
    if used != config['max_calls']:
        return False
    update = dict(result.get('execution_graph') or {})
    cards = update.get('cards') or {}
    if cards.get('stopped') and cards.get('calls') == 0:
        # Agent disables all card calls on model_error, including ordinary limits.
        # Preserve that evidence; only the previous complete graph is reusable.
        update['cards'] = {**cards, 'stopped': False}
        update['card_refresh'] = {**(update.get('card_refresh') or {}), 'pending': [], 'unavailable': []}
    return not stop_for_error({**result, 'status': 'budget_exhausted', 'execution_graph': update}, events)


def accept_worker_outcome(folder, config, state, key):
    outcome = read(folder / 'outcome.json', {})
    if outcome.get('status') == 'budget_exhausted':
        if config.get('budget_exhaustion_policy') != 'retain_score_keep_previous_handoff' or outcome.get('graph_advanced') is not False:
            raise ValueError('Unregistered budget-exhaustion continuation')
        state['tasks'][key]['status'] = 'budget_exhausted'
    else:
        state['tasks'][key]['status'] = 'completed'
        state.update(read(folder / 'handoff.json'))


def worker(batch, task_id):
    config, state = verify(batch, full=False)
    if ROOT != Path(config['runtime']):
        raise ValueError('Use the registered frozen worker')
    key = str(task_id)
    if state['tasks'].get(key, {}).get('status') != 'started':
        raise ValueError('Worker must have exactly one registered start')
    folder = batch / key
    if file_hash(folder / 'input.json') != state['tasks'][key]['input_sha256']:
        raise ValueError('Task input changed')
    config = read(folder / 'input.json')
    actor = model_class(batch, config, config['model'])
    auxiliary = model_class(batch, config, config['auxiliary_model'])
    # This worker just verified its frozen source, current inputs and handoff.
    # Legacy protocols retain the original second check.
    base.check_inputs = ((lambda ignored: None) if config.get('integrity_policy') == 'boundary-v1'
                         else (lambda ignored: verify(batch)))
    base.execution_clients = lambda *args: (actor, auxiliary)
    base.worker(batch, config, task_id)
    score_started = time.monotonic()
    official = score(config['verified_root'], batch, task_id, config['logical_origin'])
    if config.get('integrity_policy') == 'boundary-v1':
        EventLog(batch / 'timings').emit('official_scoring', task=task_id, seconds=time.monotonic()-score_started)
    write_json(folder / 'official.json', official)
    if official.get('score') is None:
        raise RuntimeError('Official score missing')
    result = read(folder / 'result.json')
    events = [e for p in ('events.jsonl','graph-annotations/events.jsonl','graph-cards/events.jsonl','diagnosis-calls/events.jsonl')
              for e in read_events(folder / p)]
    if ordinary_call_exhaustion(config, result, events, read_events(folder / 'events.jsonl')):
        verify(batch, full=False)  # Previous complete graph and experience must still match.
        write_json(folder / 'outcome.json', dict(status='budget_exhausted', graph_advanced=False,
            official_score=official['score'], reason=result['error'],
            knowledge_policy='Retain previous complete handoff; failed task graph preserved but not inherited.'))
        return
    if stop_for_error(result, events):
        raise RuntimeError('Engineering/model/asset error; preserve attempt and stop')
    graph = advance_graph(folder)
    # Diagnosis and its post-score learning share one budget. Zero disables both,
    # including client creation; normal graph/knowledge handoff still runs.
    if config.get('diagnosis_calls', 0):
        learner = auxiliary(ROOT / 'auth/llm-crawl.env', EventLog(folder / 'diagnosis-calls'),
            config['diagnosis_calls'], 4096, model=config['diagnosis_model'], retry_connections=True, stop_file=config['stop_file'])
        memory = ExperienceStore(config['experience_store'], memory_namespace(Graph.load(Path(config['graph']))))
        learned = learn_experience(folder, learner, memory, success=official['score'] == 1,
                                   verifier='WebArena-Verified official score=1')
        if learned['status'] == 'failed':
            raise RuntimeError('Experience learning failed; preserve attempt and stop')
    handoff_started = time.monotonic()
    manifest, sha = knowledge_manifest(graph, config['experience_store'], folder / 'knowledge-handoff.json')
    if config.get('integrity_policy') == 'boundary-v1':
        EventLog(batch / 'timings').emit('knowledge_manifest', task=task_id, seconds=time.monotonic()-handoff_started)
    write_json(folder / 'handoff.json', dict(current_graph=str(graph), knowledge_manifest=manifest, knowledge_manifest_sha256=sha))


def queue(batch):
    config, state = verify(batch)
    if ROOT != Path(config['runtime']):
        raise ValueError('Use the registered frozen queue')
    if state['status'] != 'registered' or state['tasks']:
        raise ValueError('Never restart a previously started experiment')
    from scripts import build_qwen_graph
    # The encoder must use the exact static scripts that generated the input index.
    build_qwen_graph.ROOT = Path(config['encoder_runtime'])
    with queue_lock(batch / 'queue.lock'):
        state.update(status='running', phase='encoder', started_at=time.time())
        write_json(batch / 'state.json', state)
        try:
            encoder_started = time.monotonic()
            with build_qwen_graph.local_encoder(batch, 8797, config['encoder_python'],
                    gpu_memory=config['gpu_memory'], expected=config['encoder_expected']):
                if config.get('integrity_policy') == 'boundary-v1':
                    EventLog(batch / 'timings').emit('encoder_ready', seconds=time.monotonic()-encoder_started)
                state['phase'] = 'five_page_cards_and_framework'; write_json(batch / 'state.json', state)
                copy_started = time.monotonic()
                start = batch / 'starting-graph'; shutil.copytree(config.get('prepared_graph',config['initial_graph']), start)
                if config.get('integrity_policy') == 'boundary-v1':
                    EventLog(batch / 'timings').emit('starting_graph_copy', seconds=time.monotonic()-copy_started)
                graph = Graph.load(start)
                if not config.get('prepared_graph'):
                    card_dir = start / 'function-cards/five-page'
                    model = model_class(batch, config, config['auxiliary_model'])(ROOT / 'auth/llm-crawl.env',
                        EventLog(batch / 'preparation-events'), 9, 4096, model=config['auxiliary_model'],
                        retry_connections=True, stop_file=config['stop_file'])
                    layer, _ = load_layer(graph)
                    cards = build_cards(graph, layer, card_dir, model)
                    if len(cards['cards']) != 8 or any(r['status'] != 'completed' for r in cards['cards'].values()):
                        raise RuntimeError('Five-page card preparation failed')
                    build_framework(graph, card_dir / 'cards.json', model, start / 'site-framework/five-page', publish=True)
                    write_json(start / 'function-cards/active.json', {'cards':'function-cards/five-page/cards.json'})
                memory = (read(config['initial_experience']) if config.get('initial_experience') else
                          ExperienceStore(config['experience_store'], memory_namespace(graph)).load())
                write_json(config['experience_store'], memory)
                manifest, sha = knowledge_manifest(start, config['experience_store'], batch / 'starting-knowledge.json')
                state.update(current_graph=str(start), knowledge_manifest=manifest, knowledge_manifest_sha256=sha, phase='tasks')
                write_json(batch / 'state.json', state); report(batch)
                for task_id in config['task_ids']:
                    if Path(config['stop_file']).exists():
                        state['status']='user_stopped'; break
                    verify(batch, full=False)
                    key = str(task_id); folder = batch / key; folder.mkdir()
                    current = Path(state['current_graph'])
                    task_config = {**config, 'graph':str(current), 'cards':str(cards_path(current))}
                    write_json(folder / 'input.json', task_config)
                    state['tasks'][key] = dict(status='started', start=time.time(), input_sha256=file_hash(folder / 'input.json'),
                        graph_before={'states':len(Graph.load(current).nodes)})
                    write_json(batch / 'state.json', state)
                    reset_started = time.monotonic()
                    reset_copy(config['container'], config['image_id'], config['backend'], 10598,
                               config['logical_origin'], config['owner'])
                    if config.get('integrity_policy') == 'boundary-v1':
                        EventLog(batch / 'timings').emit('environment_reset', task=task_id, seconds=time.monotonic()-reset_started)
                    with (folder / 'console.log').open('x',encoding='utf-8') as stream:
                        process = subprocess.Popen([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),
                            '--directory',str(batch),'--worker',key],cwd=ROOT,stdout=stream,stderr=stream,
                            env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'},creationflags=subprocess.CREATE_NO_WINDOW)
                        state['tasks'][key]['worker_pid']=process.pid; write_json(batch / 'state.json',state)
                        code=process.wait()
                    state['tasks'][key]['seconds']=time.time()-state['tasks'][key]['start']
                    if code:
                        state['tasks'][key]['status']='needs_attention'
                        raise RuntimeError(f'Worker {task_id} exited {code}; no task restart')
                    accept_worker_outcome(folder, config, state, key)
                    write_json(batch / 'state.json', state); report(batch)
                else:
                    state.update(status='completed', phase='audit')
                write_json(batch / 'state.json', state)
                verify(batch)
        except Exception as error:
            state.update(status='user_stopped' if Path(config['stop_file']).exists() else 'needs_attention',
                         error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            state['total_seconds']=time.time()-state['started_at']
            write_json(batch / 'state.json',state); report(batch)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--worker',type=int)
    parser.add_argument('--model',default='qwen3.7-flash')
    parser.add_argument('--repair-from',type=Path)
    args=parser.parse_args();batch=args.directory.resolve()
    if args.register: register(batch,args.model,args.repair_from.resolve() if args.repair_from else None)
    elif args.check: verify(batch);print('Frozen source and inputs verified')
    elif args.worker is not None: worker(batch,args.worker)
    elif args.live: queue(batch)
    else: report(batch)
