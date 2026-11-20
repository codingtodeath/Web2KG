"""Three independent Reddit-Verified arms, ordered non-Hard then Hard."""
import argparse
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
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from scripts import run_reddit_hard10_225 as method
from scripts import run_webnavigator_flash_hard as original
from scripts.run_midterm_qwen import freeze, queue_lock, protect_graph
from scripts.run_method_qwen_hard import export_existing, knowledge_manifest
from web2kg.common.activity import read_events
from web2kg.common.storage import Graph, EventLog, write_json
from web2kg.evaluation.verified import reset_copy
from web2kg.stage1.site_framework import load_framework
from web2kg.stage2.recovery import ExperienceStore, memory_namespace
from web2kg.stage2.template_context import TemplateKnowledge

ARMS = ('ours60', 'ours225', 'webnavigator')
DATASET = ROOT / 'experiments/webarena-verified/assets/dataset'
SOURCES = {
    'ours60': ROOT / 'runs/reddit-complete-index-20261104-v1/reddit-jev-flash-expand60-20261104-v1/graph',
    'ours225': ROOT / 'runs/reddit-hard10-225-flash-20261104-v1/starting-graph',
}
read = method.read


def task_order(dataset=DATASET):
    rows = [t for t in read(dataset / 'webarena-verified.json') if t['sites'] == ['reddit']]
    hard = set(read(dataset / 'subsets/webarena-verified-hard.json')['task_ids'])
    nonhard = set(read(dataset / 'subsets/webarena-verified-non-hard.json')['task_ids'])
    ids = {t['task_id'] for t in rows}
    if len(rows) != 106 or len(ids) != 106 or ids - (hard | nonhard) or hard & nonhard:
        raise ValueError('Require the complete, uniquely classified 106 single-site Reddit tasks')
    rows.sort(key=lambda t: (t['task_id'] in hard, t['task_id']))
    return [dict(task_id=t['task_id'], difficulty='hard' if t['task_id'] in hard else 'non-hard',
                 sites=t['sites'], intent=t['intent'], start_urls=t['start_urls']) for t in rows]


def public_tasks(tasks, origin):
    return [dict(task_id=t['task_id'], sites=t['sites'], intent=t['intent'],
                 start_urls=[u.replace('__REDDIT__', origin) for u in t['start_urls']]) for t in tasks]


def model_output_failure(folder, *, method_assets=False):
    result = read(folder / 'result.json', {})
    official = read(folder / 'official.json', {})
    if result.get('status') != 'model_error' or official.get('score') is None:
        return False
    root_events = read_events(folder / 'events.jsonl')
    failures = [e for e in root_events if e['event'] == 'responses_failed']
    if not failures:
        return False
    last = failures[-1]
    received = {e.get('request_id') for e in root_events if e['event'] == 'responses_received'}
    kind = last.get('failure_kind') or ''
    error = read(folder / 'llm' / last['request_id'] / 'http_error.json', {})
    inspected = (last.get('http_status') == 400 and error.get('http_status') == 400
                 and error.get('code') == 'data_inspection_failed')
    if not inspected and (last.get('http_status') or last['request_id'] not in received):
        return False
    response = read(folder / 'llm' / last['request_id'] / 'chat-response.json', {})
    length = (not kind and result.get('error') == 'ResponsesError: Incomplete response'
              and any(c.get('finish_reason') == 'length' for c in response.get('choices', [])))
    invalid = (kind.startswith('output:') and last.get('purpose') in {'original_actor', 'original_selector'}
               and (result.get('provider_failure') or {}).get('kind') == kind)
    if not (length or invalid or inspected):
        return False
    if method_assets:
        update = dict(result.get('execution_graph') or {})
        cards = update.get('cards') or {}
        if cards.get('stopped') and cards.get('calls') == 0:
            update['cards'] = {**cards, 'stopped': False}
            skipped = {e.get('template') for e in read_events(folder / 'graph-cards/events.jsonl')
                       if e['event'] == 'template_card_refresh_skipped' and e.get('reason') == 'provider_failure'}
            refresh = dict(update.get('card_refresh') or {})
            for key in ('pending', 'unavailable'):
                if key in refresh:
                    refresh[key] = [t for t in refresh[key] if t not in skipped]
            update['card_refresh'] = refresh
        events = [e for name in ('events.jsonl', 'graph-annotations/events.jsonl',
                  'graph-cards/events.jsonl', 'diagnosis-calls/events.jsonl') for e in read_events(folder / name)]
        normalized = {**result, 'status': 'budget_exhausted', 'execution_graph': update}
        if inspected:
            normalized['provider_failure'] = {}
            events = [e for e in events if not (e['event'] == 'responses_failed' and e.get('request_id') == last['request_id'])]
        if method.stop_for_error(normalized, events):
            return False
    return True


def continuation_material(previous, inherited):
    """Bind completed boundary receipts and selected evidence, plus all live inputs."""
    guards = {}
    batches = {Path(r['source_directory']).parent for r in inherited.values()} | {previous}
    for batch in batches:
        for name in ('suite.json', 'protocol.json', 'state.json', 'public-inputs.json', 'summary.json',
                     'run_material_passport.json', 'registration-integrity.json', 'protected-inputs.json',
                     'runtime-inputs.json', 'final-audit.json'):
            path = batch / name
            if path.is_file():
                guards[str(path.resolve())] = method.file_hash(path)
        guards.update(method.hashes(batch / 'trajectories'))
    for row in inherited.values():
        folder = Path(row['source_directory'])
        if row.get('input_sha256') and method.file_hash(folder / 'input.json') != row['input_sha256']:
            raise ValueError('Inherited task input changed: ' + str(folder))
        for name in ('input.json', 'result.json', 'official.json', 'events.jsonl', 'outcome.json',
                     'closeout.json', 'error.txt', 'graph-annotations/events.jsonl', 'graph-cards/events.jsonl',
                     'diagnosis-calls/events.jsonl', 'recovery/recovery.json', 'recovery/learning.json'):
            path = folder / name
            if path.is_file():
                guards[str(path.resolve())] = method.file_hash(path)
    for arm in ARMS[:2]:
        state = read(previous / arm / 'state.json')
        guards.update(protect_graph(Path(state['current_graph'])))
        path = Path(state['knowledge_manifest'])
        guards[str(path.resolve())] = method.file_hash(path)
        guards.update(read(path))
    return guards


def arm_settings(suite, arm, reference, tasks):
    batch = suite / arm
    config = dict(reference)
    for key in ('previous_version', 'comparison_version', 'failed_version', 'initial_experience',
                'inherited_tasks', 'inherited_sources', 'initial_knowledge_manifest'):
        config.pop(key, None)
    config.update(version='reddit-verified-106-' + arm + '-v1',
        created_at=datetime.now(timezone.utc).isoformat(),
        authorization='2026-11-12 user authorized all Reddit-Verified tasks in three arms, easier tasks first.',
        runtime=str(suite / 'frozen-runtime'), workspace=str(suite / 'frozen-runtime'),
        task_ids=[t['task_id'] for t in tasks], model='qwen3.7-flash',
        encoder_runtime=str(suite / 'encoder-runtime'), stop_file=str(suite / 'STOP'),
        shared_lock=str(ROOT / 'runs/reddit-hard-comparison-v2/queue.lock'),
        inherited_results=[], excluded_attempts=[], inherited_tasks={}, inherited_sources=[],
        integrity_policy='boundary-v1', max_steps=30, max_calls=31,
        output_tokens_per_call=4096, paid_preflight=0,
        limits='Exposed developmental setting; cumulative knowledge within each method arm; unequal graph inputs, prompts and whole-system compute. No component-causal or blind generalization claim.')
    if arm != 'webnavigator':
        graph = batch / 'initial-graph'
        config.update(source_graph=str(SOURCES[arm]), initial_graph=str(graph), prepared_graph=str(graph),
            graph=str(graph), cards=str(method.cards_path(graph)),
            experience_store=str(batch / 'experience.json'), max_graph_nodes=int(arm[4:]),
            auxiliary_model='qwen3.7-flash', diagnosis_model='qwen3.7-flash',
            budget_exhaustion_policy='retain_score_keep_previous_handoff',
            actor_prompt='single-action-v2', validate_verified_stop=True, unified_verified_stop=True,
            action_budget_policy='actor-action-v1',
            knowledge_policy='Original construction graph and empty task experience; independent cumulative task handoffs.',
            new_card_preparation='Reuse construction-only complete cards/framework; zero paid preparation.')
    else:
        config.update(source_assets=str(batch), continue_task_errors=False,
                      action_budget_policy='legacy-shared-calls', ax_search=False, ax_line_limit=1500)
    return config


def register(suite, checks, previous=None):
    if suite.exists() or not suite.is_relative_to(ROOT / 'runs'):
        raise ValueError('Require a fresh workspace run directory')
    receipt = read(checks)
    if not receipt.get('passed'):
        raise ValueError('Offline checks must pass')
    original.check_hashes({str(ROOT / p): sha for p, sha in receipt['source_sha256'].items()})
    inherited = {}
    if previous:
        verify(previous)
        if not read(previous / 'final-audit.json', {}).get('guards_passed'):
            raise ValueError('Prior complete boundary audit required')
        old_state = read(previous / 'state.json')
        if old_state['status'] != 'needs_attention':
            raise ValueError('Require the stopped prior suite')
        old_manifest = read(Path(read(previous / 'suite.json')['runtime']) / 'manifest.json')
        for name, sha in old_manifest.items():
            if name != 'scripts/run_reddit_verified_comparison.py' and method.file_hash(ROOT / name) != sha:
                raise ValueError('Continuation changes method source: ' + name)
        for arm in ARMS:
            verify(previous, arm=arm)
        for key, row in old_state['tasks'].items():
            folder = Path(row.get('source_directory', previous / key))
            if row['status'] not in {'completed', 'budget_exhausted'} and not (
                    row['status'] == 'failed' or model_output_failure(folder, method_assets=not key.startswith('webnavigator/'))):
                raise ValueError('Unresolved prior task: ' + key)
            inherited[key] = {**row, 'status': 'failed' if row['status'] == 'needs_attention' else row['status'],
                              'source_directory': str(folder)}
    tasks = task_order()
    suite.mkdir()
    runtime = freeze(suite)
    reference = read(ROOT / 'runs/reddit-hard724-flash-20261111-v2/protocol.json')
    baseline = read(ROOT / 'runs/webnavigator-flash-hard42-20261110-v4/protocol.json')
    encoder = suite / 'encoder-runtime/scripts'
    encoder.mkdir(parents=True)
    for name in ('serve_jina_gpu.py', 'jina_gpu.py', 'jina_vision_fp32.py'):
        shutil.copy2(Path(reference['encoder_runtime']) / 'scripts' / name, encoder / name)
    protected = method.hashes(encoder.parent)
    if previous:
        protected.update(continuation_material(previous, inherited))
    for arm in ARMS:
        batch = suite / arm
        batch.mkdir()
        old = baseline if arm == 'webnavigator' else reference
        if arm == 'webnavigator':
            retained = Path(baseline['source_assets'])
            for name in ('upstream', 'import-stubs', 'released-graph'):
                shutil.copytree(retained / name, batch / name,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            adaptation = original.install_stop_prompt(batch, runtime)
            import yaml
            cfg = yaml.safe_load((retained / 'config.yml').read_text('utf-8'))
            cfg['agent']['actor'].update(model='qwen3.7-flash', model_family='gpt',
                                         domain_paths={'reddit': str(batch / 'released-graph')})
            cfg['agent']['actor']['selector']['model'] = 'qwen3.7-flash'
            cfg['env'].update(ax_search=False, max_browser_rows=1500)
            (batch / 'config.yml').write_text(yaml.safe_dump(cfg, allow_unicode=True), 'utf-8')
            protected.update({k: v for name in ('upstream', 'import-stubs', 'released-graph')
                              for k, v in method.hashes(retained / name).items()})
        else:
            shutil.copytree(SOURCES[arm], batch / 'initial-graph')
            graph = Graph.load(batch / 'initial-graph')
            index = read(graph.directory / 'index.json')
            TemplateKnowledge(graph, method.cards_path(graph.directory), EventLog(batch / 'registration-events'))
            if (len(graph.nodes) != int(arm[4:]) or set(index['ids']) != set(graph.nodes)
                    or index['encoder'] != reference['encoder_expected'] or not load_framework(graph)):
                raise ValueError('Incomplete construction graph or encoder mismatch: ' + arm)
            protected.update(method.hashes(SOURCES[arm]))
        config = arm_settings(suite, arm, old, tasks)
        if previous and arm != 'webnavigator':
            memory = batch / 'initial-experience.json'
            shutil.copy2(previous / arm / 'experience.json', memory)
            config['initial_experience'] = str(memory)
        if arm == 'webnavigator':
            config.update(prompt_adaptation=adaptation, stop_prompt_sha256=adaptation['sha256'])
        write_json(batch / 'protocol.json', config)
        write_json(batch / 'public-inputs.json', public_tasks(tasks, config['logical_origin']))
        write_json(batch / 'run_material_passport.json', dict(mode='run', arm=arm, planned=106,
            initial_states=225 if arm == 'webnavigator' else int(arm[4:]),
            model_roles={'all_llm_roles': 'qwen3.7-flash'}, limits=config['limits'],
            initial_experience=('prior completed handoff' if previous else 'empty') if arm != 'webnavigator' else 'not used',
            paid_preflight=0))
        inputs = method.hashes(batch)
        inputs.update(method.hashes(encoder.parent))
        for name in ('qwen-hard.env', 'llm-crawl.env'):
            path = ROOT / 'auth' / name
            inputs[str(path)] = method.file_hash(path)
        write_json(batch / 'runtime-inputs.json', inputs)
        protected.update(inputs)
        state = dict(status='registered', tasks={})
        if arm != 'webnavigator':
            state['current_graph'] = str(batch / 'initial-graph')
        if previous:
            state['tasks'] = {key.split('/')[1]: row for key, row in inherited.items() if key.startswith(arm + '/')}
            if arm != 'webnavigator':
                state['current_graph'] = read(previous / arm / 'state.json')['current_graph']
        write_json(batch / 'state.json', state)
    for path in (checks, DATASET / 'webarena-verified.json',
                 DATASET / 'subsets/webarena-verified-hard.json',
                 DATASET / 'subsets/webarena-verified-non-hard.json', ROOT / 'runs/active-graph.json',
                 ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt'):
        protected[str(path.resolve())] = method.file_hash(path)
    plan = dict(version='reddit-verified-comparison-v1', model='qwen3.7-flash',
        created_at=datetime.now(timezone.utc).isoformat(), runtime=str(runtime), tasks=tasks,
        schedule=[dict(arm=a, task_id=t['task_id']) for t in tasks for a in ARMS],
        planned=318, checks=str(checks), paid_preflight=0,
        inherited_tasks=inherited, new_starts=318-len(inherited),
        task_order='64 non-Hard then42 Hard, ascending task ID within each tier; each task ours60, ours225, WebNavigator.',
        stop_policy='Official task failure, ordinary budget limits, received/scored output validation failures and scored data_inspection_failed request rejections continue. Failed method tasks retain previous complete handoff. Engineering, auth/credit/service, integrity or independent asset failures stop the suite. No task restarts.',
        integrity_scope='Current frozen source, encoder, inputs and handoff ancestry; selected scores, role ledgers and actual trajectories. Archived raw bulk is retained by reference to completed boundary audits and their protected manifests.',
        budget_difference='Method31 action decisions excludes AX search and selector. Original31 shared logical calls includes selector and original format regeneration. Both30 browser steps and4096 output tokens.',
        limitations='Exposed developmental tasks; fixed order cumulative learning; unequal starting knowledge and whole-system compute. Construction costs inherited, no new paid preflight.')
    write_json(suite / 'suite.json', plan)
    if previous:
        write_json(suite / 'continuation.json', dict(previous=str(previous), inherited=inherited,
            method='All Agent, prompts, model and graph code byte-identical; scheduler classifies isolated model failures and scopes continuation verification.',
            policy='Retain all prior scores and usage exactly once; all scored model-output failures are retained and never rerun.'))
        protected[str(suite / 'continuation.json')] = method.file_hash(suite / 'continuation.json')
    write_json(suite / 'protected-inputs.json', protected)
    write_json(suite / 'registration-integrity.json', {str(p): method.file_hash(p) for p in
        [suite / 'suite.json', suite / 'protected-inputs.json', runtime / 'manifest.json'] +
        [suite / a / 'runtime-inputs.json' for a in ARMS]})
    write_json(suite / 'state.json', dict(status='registered', tasks=inherited))
    verify(suite, full=True)
    report(suite)
    print(json.dumps(dict(registered=str(suite), planned=318, protected=len(protected))), flush=True)


def verify(suite, full=False, arm=None):
    original.check_hashes(read(suite / 'registration-integrity.json'))
    plan = read(suite / 'suite.json')
    runtime = Path(plan['runtime'])
    original.check_hashes({str(runtime / p): sha for p, sha in read(runtime / 'manifest.json').items()})
    if full:
        original.check_hashes(read(suite / 'protected-inputs.json'))
    if arm:
        batch = suite / arm
        original.check_hashes(read(batch / 'runtime-inputs.json'))
        state = read(batch / 'state.json')
        if state.get('knowledge_manifest'):
            original.check_hashes({state['knowledge_manifest']: state['knowledge_manifest_sha256']})
            original.check_hashes(read(Path(state['knowledge_manifest'])))
    return plan


def report(suite):
    plan, state = read(suite / 'suite.json'), read(suite / 'state.json')
    summary = dict(status=state['status'], planned=318, arms={})
    sections = []
    tiers = {t['task_id']: t['difficulty'] for t in plan['tasks']}
    for arm in ARMS:
        batch = suite / arm
        p, s = read(batch / 'protocol.json'), read(batch / 'state.json')
        previous = {r['task_id']: r for r in read(batch / 'summary.json', {}).get('tasks', [])}
        rows = []
        for task in p['task_ids']:
            task_state = s['tasks'].get(str(task), {})
            folder = Path(task_state.get('source_directory', batch / str(task)))
            status = task_state.get('status', 'not_started')
            row = previous.get(task)
            if not row or row['status'] != status or status in {'started', 'needs_attention'}:
                row = method.task_stats(folder, task, s['tasks'].get(str(task), {}))
                row.update(difficulty=tiers[task], accounting=method.statistics(folder),
                    source_directory=str(folder), inherited='source_directory' in task_state,
                    roles={name: method.statistics(folder / sub) for name, sub in
                        [('description', 'graph-annotations'), ('cards', 'graph-cards'), ('diagnosis_learning', 'diagnosis-calls')]})
                row['roles']['actor_selector'] = {key: row['accounting'][key] - sum(v[key] for v in row['roles'].values())
                    for key in ('physical_requests', 'unknown_usage', 'input_tokens', 'output_tokens',
                                'request_seconds', 'output_retries', 'transport_retries')}
            rows.append(row)
        totals = {key: sum(r['accounting'][key] for r in rows) for key in
            ('physical_requests', 'unknown_usage', 'input_tokens', 'output_tokens',
             'request_seconds', 'output_retries', 'transport_retries')}
        totals['actual_models'] = sorted({model for r in rows for model in r['accounting']['actual_models']})
        stats = dict(planned=106, resolved=sum(r['status'] in {'completed', 'budget_exhausted', 'failed'} for r in rows),
            scored=sum(r['official_score'] is not None for r in rows),
            successes=sum(r['official_score'] == 1 for r in rows), totals=totals, tasks=rows)
        stats['tiers'] = {tier: dict(planned=sum(r['difficulty'] == tier for r in rows),
            scored=sum(r['difficulty'] == tier and r['official_score'] is not None for r in rows),
            successes=sum(r['difficulty'] == tier and r['official_score'] == 1 for r in rows)) for tier in ('non-hard', 'hard')}
        write_json(batch / 'summary.json', stats)
        if any((batch / str(t) / 'result.json').exists() for t in p['task_ids']):
            export_existing(batch)
        summary['arms'][arm] = stats
        table = []
        for row in rows:
            trace = Path(row['source_directory']).parent / 'trajectories' / f'task-{row["task_id"]}.html'
            label = (f'<a href="{Path(os.path.relpath(trace, suite)).as_posix()}">{row["task_id"]}</a>'
                     if trace.exists() else str(row['task_id']))
            table.append('<tr>' + ''.join(f'<td>{v}</td>' for v in [label, row['difficulty'],
                html.escape(row['status']), row['official_score'], row.get('steps'), row.get('total_seconds'),
                row.get('physical_requests'), row.get('unknown_usage')]) + '</tr>')
        sections.append(f'<h2>{arm}</h2><p>已结束 {stats["resolved"]}/106，已评分 {stats["scored"]}，成功 {stats["successes"]}。</p>'
            '<table><tr><th>任务与轨迹</th><th>难度层</th><th>状态</th><th>官方分</th><th>步数</th><th>秒</th><th>请求</th><th>未知用量</th></tr>' + ''.join(table) + '</table>')
    write_json(suite / 'summary.json', summary)
    page = ('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60">'
        '<title>Reddit-Verified 三组全量比较</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd}</style>'
        f'<h1>Reddit-Verified 三组全量比较</h1><p>状态 {html.escape(state["status"])}。每组106题，总计318次新尝试，全部角色 Qwen3.7-flash，百炼直连。</p>'
        '<p>先64道非Hard，再42道Hard，各层按编号升序。同题依次执行60节点、225节点和WebNavigator，每次重置专用环境。本方法各组从建图资产和空任务经验独立开始，组内累积知识。</p>'
        '<p>均为30步、每次输出4096token。本方法31次行动决策额度，AX搜索和页面选择另计用量。WebNavigator保留31次共享调用与AX1500。图谱、提示词和总计算投入存在差异。</p>'
        '<p>数据属于开发期间已暴露的任务。当前表保留未运行、缺失和失败，完成全量后计算整体及分层成功率。历史建图成本按来源另列。</p>'
        '<p><a href="suite.json">实验登记</a> · <a href="summary.json">成绩和用量</a> · <a href="state.json">队列状态</a></p>' + ''.join(sections))
    (suite / 'report.html').write_text(page, 'utf-8')


def worker(suite, arm, task):
    verify(suite, arm=arm)
    if Path(read(suite / 'suite.json')['runtime']) != ROOT:
        raise ValueError('Use the frozen worker')
    batch = suite / arm
    if arm == 'webnavigator':
        original.verify = lambda ignored, **kwargs: (verify(suite, arm=arm) and read(batch / 'protocol.json'))
        original.worker(batch, task)
    else:
        method.verify = lambda ignored, **kwargs: (verify(suite, arm=arm) and
            (read(batch / 'protocol.json'), read(batch / 'state.json')))
        method.worker(batch, task)


def live(suite):
    plan = verify(suite, full=True)
    state = read(suite / 'state.json')
    if (Path(plan['runtime']) != ROOT or state['status'] != 'registered'
            or state['tasks'] != plan.get('inherited_tasks', {})):
        raise ValueError('Require frozen entry and a never-started suite')
    from scripts import build_qwen_graph
    config = read(suite / 'ours60/protocol.json')
    build_qwen_graph.ROOT = Path(config['encoder_runtime'])
    with queue_lock(config['shared_lock']):
        state.update(status='running', queue_pid=os.getpid(), started_at=time.time(), phase='encoder')
        write_json(suite / 'state.json', state)
        try:
            with build_qwen_graph.local_encoder(suite, 8797, config['encoder_python'],
                    gpu_memory=config['gpu_memory'], expected=config['encoder_expected']):
                for arm in ARMS[:2]:
                    batch = suite / arm
                    c, s = read(batch / 'protocol.json'), read(batch / 'state.json')
                    memory = ExperienceStore(c['experience_store'], memory_namespace(Graph.load(Path(c['graph']))))
                    write_json(c['experience_store'], read(c['initial_experience']) if c.get('initial_experience') else memory.load())
                    manifest, sha = knowledge_manifest(s['current_graph'], c['experience_store'], batch / 'initial-handoff.json')
                    s.update(status='running', knowledge_manifest=manifest, knowledge_manifest_sha256=sha)
                    write_json(batch / 'state.json', s)
                for item in plan['schedule']:
                    if f"{item['arm']}/{item['task_id']}" in plan.get('inherited_tasks', {}):
                        continue
                    if (suite / 'STOP').exists():
                        state['status'] = 'user_stopped'
                        break
                    arm, task = item['arm'], item['task_id']
                    verify(suite, arm=arm)
                    batch = suite / arm
                    c, s = read(batch / 'protocol.json'), read(batch / 'state.json')
                    if str(task) in s['tasks']:
                        raise ValueError('Never restart a started task')
                    state.update(phase='reset', current=item)
                    write_json(suite / 'state.json', state)
                    reset_copy(c['container'], c['image_id'], c['backend'], 10598, c['logical_origin'], c['owner'])
                    folder = batch / str(task)
                    folder.mkdir()
                    inputs = dict(c)
                    if arm != 'webnavigator':
                        inputs.update(graph=s['current_graph'], cards=str(method.cards_path(Path(s['current_graph']))))
                    write_json(folder / 'input.json', inputs)
                    row = dict(status='started', start=time.time(), input_sha256=method.file_hash(folder / 'input.json'))
                    s['tasks'][str(task)] = row
                    write_json(batch / 'state.json', s)
                    key = f'{arm}/{task}'
                    state['tasks'][key] = row
                    state.update(phase='task')
                    write_json(suite / 'state.json', state)
                    with (folder / 'console.log').open('x', encoding='utf-8') as output:
                        process = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(Path(__file__)),
                            '--directory', str(suite), '--worker', arm, str(task)], cwd=ROOT,
                            stdout=output, stderr=output, creationflags=subprocess.CREATE_NO_WINDOW,
                            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
                        row['worker_pid'] = process.pid
                        write_json(batch / 'state.json', s)
                        write_json(suite / 'state.json', state)
                        code = process.wait()
                    row['seconds'] = time.time() - row['start']
                    if code and model_output_failure(folder, method_assets=arm != 'webnavigator'):
                        verify(suite, arm=arm)
                        row.update(status='failed', worker_exit_code=code, reason='Task model request or output failure. Official score retained.')
                        code = 0
                    elif code:
                        row['status'] = 'needs_attention'
                    elif arm == 'webnavigator':
                        row['status'] = 'completed'
                    else:
                        method.accept_worker_outcome(folder, c, s, str(task))
                    write_json(batch / 'state.json', s)
                    write_json(suite / 'state.json', state)
                    report(suite)
                    print(json.dumps(dict(arm=arm, task=task, status=row['status'],
                        score=read(folder / 'official.json', {}).get('score'), started=len(state['tasks']))), flush=True)
                    if code:
                        raise RuntimeError(f'{key} worker exited {code}; retain attempt and inspect')
                else:
                    state['status'] = 'completed'
        except Exception as error:
            state.update(status='user_stopped' if (suite / 'STOP').exists() else 'needs_attention',
                         error=type(error).__name__ + ': ' + str(error))
        finally:
            state['seconds'] = time.time() - state['started_at']
            write_json(suite / 'state.json', state)
            verify(suite, full=True)
            write_json(suite / 'final-audit.json', dict(status=state['status'], guards_passed=True))
            report(suite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--checks', type=Path)
    parser.add_argument('--continue-from', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', nargs=2, metavar=('ARM', 'TASK'))
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.register:
        register(directory, args.checks.resolve(), args.continue_from.resolve() if args.continue_from else None)
    elif args.live:
        live(directory)
    elif args.worker:
        worker(directory, args.worker[0], int(args.worker[1]))
    elif args.check:
        verify(directory, full=True)
    else:
        report(directory)
