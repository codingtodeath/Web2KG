"""New, versioned all-Qwen Reddit/GitLab Hard comparison with regression stops."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
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
from scripts.run_hard_comparison import cards_path, task_stats, advance_graph
from scripts.run_method_qwen_hard import export_existing, knowledge_manifest, service_error
from web2kg.common.activity import read_events
from web2kg.common.jina import JinaClient
from web2kg.common.jina_runtime import LocalEncoder
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.evaluation.verified import reset_copy, score
from web2kg.stage1.function_cards import file_hash
from web2kg.stage2.recovery import ExperienceStore, learn_experience, memory_namespace

MODEL = 'qwen/qwen3.8-27b:free'
OWNER = 'midterm-qwen-20261027-v1'
ARMS = ('ours', 'webnavigator')
SEED = 20261027


def read(path, default=None):
    return json.loads(Path(path).read_text('utf-8')) if Path(path).exists() else default


def public_tasks(site, project=ROOT):
    dataset = project / 'experiments/webarena-verified/assets/dataset'
    hard = set(read(dataset / 'subsets/webarena-verified-hard.json')['task_ids'])
    # Metadata selects the full single-site subset, not outcomes or task wording.
    rows = sorted((t for t in read(dataset / 'webarena-verified.json')
                   if t['task_id'] in hard and t['sites'] == [site]), key=lambda t: t['task_id'])
    random.Random(SEED).shuffle(rows)
    endpoint = 'http://127.0.0.1:10499' if site == 'reddit' else 'http://127.0.0.1:10799'
    return [dict(task_id=t['task_id'], sites=t['sites'], intent=t['intent'],
                 start_urls=[u.replace('__' + site.upper() + '__', endpoint) for u in t['start_urls']]) for t in rows]


def regression(rows, previous=None):
    """Deterministic engineering alarm; never a significance test or bug verdict."""
    ours = [r for r in rows if r['arm'] == 'ours' and r['official_score'] is not None]
    if previous:
        paired = [(r['official_score'], previous[str(r['task_id'])]) for r in ours
                  if str(r['task_id']) in previous]
        if len(paired) >= 3 and all(a == 0 and b == 1 for a, b in paired[-3:]):
            return 'three_consecutive_historical_successes_now_failed'
        if len(paired) >= 6 and sum(b-a for a, b in paired) >= 3 and sum(b-a for a, b in paired)/len(paired) >= .25:
            return 'historical_prefix_drop_at_least_25pp_and_three_tasks'
    baseline = {r['task_id']: r['official_score'] for r in rows
                if r['arm'] == 'webnavigator' and r['official_score'] is not None}
    pairs = [(r['official_score'], baseline[r['task_id']]) for r in ours if r['task_id'] in baseline]
    if len(pairs) >= 6 and sum(b-a for a, b in pairs) >= 3 and sum(b-a for a, b in pairs)/len(pairs) >= .30:
        return 'paired_baseline_gap_at_least_30pp_and_three_tasks'
    return None


def pending(protocol, state, active=None):
    keys = [f"{r['arm']}/{r['task_id']}" for r in protocol['schedule']]
    inherited = protocol.get('inherited_completed', {})
    if (len(keys) != len(set(keys)) or len(keys) - len(inherited) != protocol['max_new_task_starts']
            or not set(inherited).issubset(keys)):
        raise ValueError('Duplicate or inconsistent registered starts')
    for key in inherited:
        if state['tasks'].get(key, {}).get('status') != 'completed':
            raise ValueError('Inherited completion missing: ' + key)
    for key, row in state['tasks'].items():
        if key not in keys or (row['status'] != 'completed' and key != active):
            raise ValueError('Unresolved or unregistered started task: ' + key)
    return [r for r, key in zip(protocol['schedule'], keys) if key not in state['tasks']]


def verify(batch, active=None):
    protocol, state = read(batch / 'protocol.json'), read(batch / 'state.json')
    for path, expected in read(batch / 'registration-integrity.json').items():
        if file_hash(path) != expected:
            raise ValueError('Registration changed: ' + path)
    base.check_inputs(protocol)
    if state.get('knowledge_manifest'):
        if file_hash(state['knowledge_manifest']) != state['knowledge_manifest_sha256']:
            raise ValueError('Handoff manifest changed')
        for path, expected in read(state['knowledge_manifest']).items():
            if file_hash(path) != expected:
                raise ValueError('Cumulative knowledge changed: ' + path)
    return protocol, state, pending(protocol, state, active)


def source_files(root):
    for name in ('web2kg', 'scripts', 'third_party/webnavigator', 'examples'):
        for path in (root / name).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.js', '.html', '.txt', '.json', '.yml'} and '__pycache__' not in path.parts:
                yield path


def encoder_material(root):
    """Guard immutable encoder code; runtime logs are outputs, not inputs."""
    manifest_path = root / 'manifest.json'
    manifest = read(manifest_path)
    paths = [manifest_path]
    for name, record in manifest.items():
        path = root / 'scripts' / name
        if Path(name).name != name or file_hash(path) != record['sha256']:
            raise ValueError('Encoder source manifest mismatch: ' + name)
        paths.append(path)
    return paths


def freeze(suite):
    target = suite / 'frozen-runtime'
    for path in source_files(ROOT):
        destination = target / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    subprocess.run(['powershell.exe', '-NoProfile', '-Command',
        f"New-Item -ItemType Junction -Path '{target / 'auth'}' -Target '{ROOT / 'auth'}' | Out-Null"],
        check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    manifest = {p.relative_to(target).as_posix(): file_hash(p) for p in source_files(target)}
    write_json(target / 'manifest.json', manifest)
    return target


def protect_graph(graph):
    guards = {}
    current, visited = graph, set()
    while current not in visited:
        visited.add(current)
        # Protect current material and the ancestry metadata used to validate it.
        paths = current.rglob('*') if current == graph else [current / 'graph.json', *current.glob('templates/**/*.json'),
            *current.glob('site-framework/**/*.json'), *current.glob('*website-framework.json')]
        for path in paths:
            if path.is_file() and path.name not in {'events.jsonl'} and path.suffix != '.pyc':
                guards[str(path.resolve())] = file_hash(path)
        source = read(current / 'graph.json').get('execution_update', {}).get('source_graph')
        if not source:
            break
        current = Path(source).resolve()
    return guards


def prepare_site(suite, site, graph, released, cached_guards=None):
    import yaml
    batch = suite / site
    batch.mkdir(exist_ok=False)
    runtime = suite / 'frozen-runtime'
    project = suite.parent.parent
    tasks = public_tasks(site, project)
    for arm in ARMS:
        write_json(batch / arm / 'public-inputs.json', tasks)
    baseline = batch / 'webnavigator'
    old = project / 'runs/reddit-hard-comparison-v2/webnavigator'
    for name in ('upstream', 'import-stubs'):
        shutil.copytree(old / name, baseline / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Assets remain immutable. The upstream loader expects this conventional path.
    subprocess.run(['powershell.exe', '-NoProfile', '-Command',
        f"New-Item -ItemType Junction -Path '{baseline / 'released-graph'}' -Target '{released}' | Out-Null"],
        check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    cfg = yaml.safe_load((old / 'config.yml').read_text('utf-8'))
    actor = cfg['agent']['actor']
    actor.update(model=MODEL, model_family='gpt', domain_paths={site: str(baseline / 'released-graph')})
    actor['selector']['model'] = MODEL
    cfg['env'].update(ax_search=False, max_browser_rows=1500)
    (baseline / 'config.yml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    # Compute against the frozen source, not the editable development checkout.
    previous_root = base.ROOT
    base.ROOT = runtime
    try:
        source_sha = base.source_hash()
    finally:
        base.ROOT = previous_root
    guard = dict(cached_guards) if cached_guards is not None else protect_graph(graph)
    if cached_guards is not None:
        if str((graph / 'graph.json').resolve()) not in guard:
            raise ValueError('Repair source graph was not protected by the previous registration')
        for path, expected in guard.items():
            if file_hash(path) != expected:
                raise ValueError('Original repair input changed: ' + path)
    for directory in (runtime, suite / 'encoder-runtime', baseline / 'upstream', baseline / 'import-stubs', released):
        paths = encoder_material(directory) if directory == suite / 'encoder-runtime' else directory.rglob('*')
        for path in paths:
            if path.is_file() and 'auth' not in path.relative_to(directory).parts and path.suffix != '.pyc' and '__pycache__' not in path.parts:
                guard[str(path.resolve())] = file_hash(path)
    for path in [baseline / 'config.yml', *(batch / a / 'public-inputs.json' for a in ARMS),
                 project / 'auth/qwen-hard.env', project / 'auth/llm-crawl.env']:
        guard[str(path.resolve())] = file_hash(path)
    index = read(graph / 'index.json')
    schedule = [dict(arm=arm, task_id=t['task_id']) for number, t in enumerate(tasks)
                for arm in (ARMS if number % 2 == 0 else ARMS[::-1])]
    endpoint = 'http://127.0.0.1:10599' if site == 'reddit' else 'http://127.0.0.1:10799'
    config = dict(site=site, workspace=str(runtime), frozen_runtime=str(runtime), source_sha256=source_sha,
        protected_inputs=guard, task_ids=[t['task_id'] for t in tasks], schedule=schedule,
        max_new_task_starts=len(schedule), graph=str(graph), cards=str(cards_path(graph)),
        execution_provider='qwen', auxiliary_provider='qwen', credential_file=str(project / 'auth/qwen-hard.env'),
        model=MODEL, auxiliary_model=MODEL, diagnosis_model=MODEL, seed=SEED,
        output_tokens_per_call=4096, auxiliary_output_tokens=4096, diagnosis_output_tokens=4096,
        max_steps=30, max_calls=31, diagnosis_calls=5, max_diagnoses=3, recovery_actions=12,
        transition_calls=None, card_calls=None, max_total_model_calls=None,
        update_graph=True, finalize_graph_assets=True, framework='auto', top_k=5,
        ax_line_limit=800, ax_search=True, encoder_endpoint=index['endpoint'], encoder_timeout=600,
        encoder_python='D:/Anaconda/envs/web2kg-jina-gpu/python.exe', encoder_runtime=str(suite / 'encoder-runtime'), backend=endpoint,
        logical_origin='http://127.0.0.1:10499' if site == 'reddit' else endpoint,
        graph_origin='http://127.0.0.1:10499' if site == 'reddit' else endpoint,
        owner='template-agent-hard10-v1' if site == 'reddit' else OWNER,
        container='web2kg-template-eval' if site == 'reddit' else 'web2kg-midterm-gitlab',
        image_id='am1n3e/webarena-verified-' + site + ':latest',
        verified_root=str(ROOT / 'experiments/webarena-verified'), retry_connections=True,
        stop_file=str(suite / 'STOP'), shared_lock=str(project / 'runs/reddit-hard-comparison-v2/queue.lock'),
        experience_store=str(batch / 'ours/experiences.json'),
        role_models={role: MODEL for role in ('actor', 'selector', 'description', 'backfill', 'cards', 'diagnosis', 'learning')},
        limits=['Exposed-task developmental comparison; Reddit graph includes old task knowledge and old-model artifacts.',
                'All NEW language-model calls use Qwen; Jina remains a local embedding model.',
                'System inputs and compute differ; not component-causal or generalization evidence.',
                'Failures/restarts retained by version; no best-score selection.'],
        regression_policy={'historical': '3 consecutive lost successes, or >=6 paired and >=25pp net drop / >=3 tasks',
                           'baseline': '>=6 paired and >=30pp net gap / >=3 tasks',
                           'engineering': 'first runtime/service/asset failure stops before next task'},
        retry_policy='Identical-input temporary transport/408/429/5xx backoff 2..60 sec; no browser-action retry or wall-time kill')
    config['verified_root'] = str(project / 'experiments/webarena-verified')
    history = read(project / 'runs/reddit-hard-compact-output-v1/current-comparison-summary.json')['groups']['本方法 Qwen']['tasks'] if site == 'reddit' else []
    config['historical_scores'] = {str(t['task_id']): t['official_score'] for t in history}
    original = graph / 'experiences/memory.json'
    memory = batch / 'ours/experiences.json'
    if original.exists():
        shutil.copy2(original, memory)
    else:
        write_json(memory, dict(version=1, namespace=memory_namespace(Graph.load(graph)), pages=[], templates={}))
    shutil.copy2(memory, memory.with_name('initial-experiences.json'))
    guard[str(memory.with_name('initial-experiences.json'))] = file_hash(memory.with_name('initial-experiences.json'))
    write_json(batch / 'protocol.json', config)
    manifest, digest = knowledge_manifest(graph, memory, batch / 'initial-handoff.json')
    write_json(batch / 'state.json', dict(status='registered', tasks={}, current_graph=str(graph),
        knowledge_manifest=manifest, knowledge_manifest_sha256=digest))
    write_json(batch / 'run_material_passport.json', dict(mode='run', type='code experiment', site=site,
        authorization='2026-10-27 user requests both methods, all single-site Hard tasks, all Qwen roles; stop/fix/restart on confirmed bugs.',
        source_sha256=source_sha, tasks=config['task_ids'], scope=config['limits']))
    write_json(batch / 'registration-integrity.json', {str(p): file_hash(p) for p in
        (batch / 'protocol.json', batch / 'run_material_passport.json', batch / 'initial-handoff.json')})
    report(suite)
    return config


def prepare(suite, previous=None, repair=None):
    suite.mkdir(exist_ok=False)
    freeze(suite)
    # Encoder fingerprints include source bytes. Git checkout newline conversion is
    # harmless to inference but changes that fingerprint; retain original exact bytes.
    original_encoder = ROOT / 'runs/reddit-hard-method-qwen-v1/frozen-runtime/scripts'
    encoder_files = {}
    for name in ('jina_gpu.py', 'jina_vision_fp32.py', 'serve_jina_gpu.py'):
        source = original_encoder / name
        destination = suite / 'encoder-runtime/scripts' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        encoder_files[name] = dict(source=str(source.resolve()), sha256=file_hash(source))
    write_json(suite / 'encoder-runtime/manifest.json', encoder_files)
    write_json(suite / 'suite.json', dict(version=1, created=datetime.now(timezone.utc).isoformat(),
        status='preparing', sites=['reddit', 'gitlab'], planned={'reddit': 84, 'gitlab': 114},
        gitlab_status='preparation_pending', all_new_llm_roles='Qwen3.8-27B',
        old_experiments='immutable', restart_rule='Only after a confirmed bug, repair + offline checks; new version from original inputs. Otherwise stop.'))
    source = Path(read(ROOT / 'runs/active-graph.json')['graph'])
    cached = None
    if previous:
        old = read(previous / 'reddit/protocol.json')
        old_state = read(previous / 'reddit/state.json')
        diagnosis = read(repair) if repair else None
        if repair and (not diagnosis or diagnosis.get('status') != 'confirmed_bug'
                       or diagnosis.get('repair') != 'encoder_static_inputs_only'
                       or diagnosis.get('full_restart_from_original_inputs') is not True):
            raise ValueError('Full restart needs a concrete bug diagnosis and original-input authorization')
        if (old_state['tasks'] and not diagnosis) or old['graph'] != str(source):
            raise ValueError('This setup repair only accepts a zero-start version with identical original inputs')
        cached = {path: digest for path, digest in old['protected_inputs'].items()
                  if not Path(path).is_relative_to(previous)}
        if repair:
            cached[str(repair)] = file_hash(repair)
        lineage = read(previous / 'repair-lineage.json', {})
        preflight_receipt = previous / 'preflight/closeout.json'
        if not preflight_receipt.exists():
            preflight_receipt = Path(lineage['inherited_preflight'])
        if read(preflight_receipt).get('status') != 'completed':
            raise ValueError('Missing completed inherited preflight')
        write_json(suite / 'repair-lineage.json', dict(previous=str(previous),
            reason=diagnosis['reason'] if diagnosis else 'Exact encoder source bytes pinned to existing index; LF/CRLF conversion changed only identity hash.',
            diagnosis=str(repair) if repair else None, inherited_preflight=str(preflight_receipt),
            retained_versions=diagnosis.get('retained_versions', []) if diagnosis else [str(previous)],
            old_task_starts=len(old_state['tasks']), full_restart_from_original_inputs=True,
            restart='Whole suite from original inputs, no score selection'))
    prepare_site(suite, 'reddit', source, (ROOT / 'runs/reddit-hard-comparison-v2/webnavigator/released-graph').resolve(), cached)


def preflight(suite):
    """One registered multimodal/schema probe; never counted as a task."""
    from web2kg.common.responses import object_schema
    folder = suite / 'preflight'
    if folder.exists():
        raise ValueError('Preflight already started; inspect its evidence instead of repeating')
    protocol = read(suite / 'reddit/protocol.json')
    graph = Graph.load(Path(protocol['graph']))
    qwen = clients(ROOT / 'auth/qwen-hard.env', suite / 'reddit/routing-state.json')
    client = qwen(ROOT / 'auth/llm-crawl.env', EventLog(folder), 1, 512,
                  model=MODEL, retry_connections=True, stop_file=suite / 'STOP')
    schema = object_schema({'image_received': {'type': 'boolean'}, 'summary': {'type': 'string'}})
    answer = client.json('Check that this recorded screenshot is readable. Summarize its visible layout in one sentence.',
                         {'purpose': 'multimodal structured transport preflight, not task execution'}, schema,
                         'midterm_preflight', images=[graph.directory / 'snapshots' / graph.data['root'] / 'page.png'])
    if answer['image_received'] is not True:
        raise RuntimeError('Image preflight did not confirm receipt')
    write_json(folder / 'closeout.json', dict(status='completed', response=answer, actual_provider=client.provider,
        actual_model=client.model, scope='Separate preflight cost, no task starts'))


def report(suite):
    summary = {'sites': {}, 'scope': 'Exposed-task developmental comparison; all new LLM roles Qwen3.8-27B'}
    content = ['<!doctype html><meta charset="utf-8"><title>Midterm all-Qwen comparison</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:35px auto}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ccc}</style>',
        '<h1>中期补充实验 · 全 Qwen · Reddit / GitLab Hard</h1>',
        '<p>新调用的执行、选择、描述、卡片、诊断和学习均为 Qwen3.8-27B。Reddit继承当前累积图及历史多模型资产。',
        '不同图谱、提示和额外计算的整体系统开发比较；非盲测、泛化或组件因果证据。未评分与未运行不计为失败。历史结果不覆盖。</p>']
    lineage = read(suite / 'continuation.json')
    if lineage:
        summary['continuation'] = lineage
        summary['interrupted_attempts'] = []
        for attempt in lineage['interrupted_attempts']:
            row = task_stats(Path(attempt['evidence_directory']), attempt['task_id'], {'status': 'user_interrupted'})
            row.update(arm=attempt['arm'], site=attempt['site'], evidence_directory=attempt['evidence_directory'],
                       official_score_excluded_from_current_view=True)
            summary['interrupted_attempts'].append(row)
        prior = Path(os.path.relpath(Path(lineage['previous']) / 'report.html', suite)).as_posix()
        content.append(f'<p>用户暂停后的同方法续跑，已完成结果引用原证据；中断尝试不计入当前成功率，其请求、用量和耗时单列保留。<a href="{html.escape(prior)}">暂停前报告及中断证据</a></p>')
    for site in ('reddit', 'gitlab'):
        batch = suite / site
        if not (batch / 'protocol.json').exists():
            content.append(f'<h2>{site}</h2><p>准备中，尚无正式任务开始。</p>')
            continue
        protocol, state = read(batch / 'protocol.json'), read(batch / 'state.json')
        groups, all_rows = {}, []
        for arm in ARMS:
            rows = []
            for task_id in protocol['task_ids']:
                key = f'{arm}/{task_id}'
                evidence = Path(protocol.get('inherited_completed', {}).get(key, batch / key))
                row = task_stats(evidence, task_id, state['tasks'].get(key, {}))
                row['arm'] = arm
                row['evidence_directory'] = str(evidence)
                rows.append(row)
            all_rows.extend(rows)
            n = sum(r['official_score'] is not None for r in rows)
            wins = sum(r['official_score'] == 1 for r in rows)
            groups[arm] = dict(planned=len(rows), official_scored=n, successes=wins,
                success_rate=wins/n if n else None, unrun=sum(r['status'] == 'not_started' for r in rows), tasks=rows)
            write_json(batch / arm / 'summary.json', groups[arm])
            if any((batch / arm / str(t) / 'result.json').exists() for t in protocol['task_ids']):
                export_existing(batch / arm)
            rate = f'{100*wins/n:.1f}%' if n else '—'
            content.append(f'<h2>{site} / {arm}: {rate}（已评分 {n}/{len(rows)}）</h2>')
            content.append('<table><tr><th>Task</th><th>Status</th><th>Official</th><th>Steps</th><th>Seconds</th><th>Requests / unknown usage</th><th>Known input / output tokens</th></tr>')
            for row in rows:
                trajectory = Path(row['evidence_directory']).parent / 'trajectories' / f'task-{row["task_id"]}.html'
                trace = Path(os.path.relpath(trajectory, suite)).as_posix()
                label = f'<a href="{trace}">{row["task_id"]}</a>' if (suite / trace).exists() else row['task_id']
                tokens = [sum(v[k] for v in row['models'].values()) for k in ('input_tokens', 'output_tokens')]
                cells = [label, row['status'], row['official_score'], row['steps'], row['total_seconds'],
                    f'{row["physical_requests"]} / {row["unknown_usage"]}', f'{tokens[0]} / {tokens[1]}']
                content.append('<tr>' + ''.join(f'<td>{v}</td>' for v in cells) + '</tr>')
            content.append('</table>')
        summary['sites'][site] = dict(status=state['status'], groups=groups,
            regression=regression(all_rows, protocol.get('historical_scores')))
        content.append('<p>Queue: ' + html.escape(state['status']) + '</p>')
        write_json(batch / 'summary.json', summary['sites'][site])
    write_json(suite / 'summary.json', summary)
    (suite / 'report.html').write_text(''.join(content), encoding='utf-8')
    return summary


def login_gitlab(batch, folder):
    from playwright.sync_api import sync_playwright
    path = folder / 'private/gitlab-session.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto('http://127.0.0.1:10799/users/sign_in', wait_until='domcontentloaded', timeout=60000)
        page.get_by_label('Username or email').fill('byteblaze')
        page.get_by_label('Password', exact=True).fill(os.environ['VERIFIED_GITLAB_PASSWORD'])
        page.get_by_role('button', name='Sign in', exact=True).click()
        page.wait_for_url(lambda url: '/users/sign_in' not in url, timeout=60000)
        context.storage_state(path=str(path))
        browser.close()
    return str(path)


def reset_gitlab(config):
    name = 'web2kg-midterm-gitlab'
    current = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
    if current.returncode == 0:
        info = json.loads(current.stdout)[0]
        if (info['Config'].get('Labels') or {}).get('web2kg.midterm') != OWNER:
            raise ValueError('GitLab container owner mismatch')
        subprocess.run(['docker', 'rm', '-f', name], check=True, capture_output=True)
    subprocess.run(['docker', 'run', '--pull=never', '-d', '--name', name, '--label', 'web2kg.midterm=' + OWNER,
        '-p', '127.0.0.1:10799:8023', '-p', '127.0.0.1:10798:8877', '--shm-size=256m',
        config['image_id']], check=True, capture_output=True)
    # Startup health is bounded; task requests themselves have no wall-time kill.
    import requests
    session = requests.Session(); session.trust_env = False
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            if session.get('http://127.0.0.1:10799/users/sign_in', timeout=3).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        raise RuntimeError('Owned GitLab initialization timeout')
    result = subprocess.run(['docker', 'exec', name, 'env-ctrl', 'init', '--base-url', 'http://127.0.0.1:10799'],
                            capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError('Owned GitLab base URL initialization failed')


def build_gitlab(suite):
    """Task-independent bounded crawl; construction costs are outside task scores."""
    from web2kg.common.browser import Browser
    from web2kg.stage1.crawl import CrawlBudget, crawl
    from web2kg.stage1.llm_exploration import LLMExplorer
    from web2kg.stage1.templates import build_templates
    from web2kg.stage1.function_cards import build_cards
    from web2kg.stage1.site_framework import build_framework
    from web2kg.stage1.visual_index import build_visual_index
    directory = suite / 'gitlab-build'
    if directory.exists():
        if read(directory / 'closeout.json', {}).get('status') == 'completed':
            return directory / 'graph'
        raise ValueError('Unresolved GitLab construction; inspect, never silently repeat model calls')
    directory.mkdir()
    write_json(directory / 'protocol.json', dict(status='registered', seed=SEED, model=MODEL,
        start_url='http://127.0.0.1:10799/', max_states=120, max_depth=5, actions_per_page=12,
        max_attempts=240, max_calls=360, output_tokens=4096, task_inputs_used=False,
        scope='New bounded site construction; no task intents or evaluator references. Costs separate from execution.'))
    reset_gitlab({'image_id': 'am1n3e/webarena-verified-gitlab:latest'})
    storage = login_gitlab(directory, directory)
    qwen = clients(ROOT / 'auth/qwen-hard.env', directory / 'routing-state.json')
    planner = qwen(ROOT / 'auth/llm-crawl.env', EventLog(directory / 'construction-calls'), 360, 4096,
                   model=MODEL, retry_connections=True, stop_file=suite / 'STOP')
    log = EventLog(directory / 'graph')
    explorer = LLMExplorer(planner, log.directory)
    with Browser('http://127.0.0.1:10799/', log, storage_state=storage, identity_mode='interaction',
                 viewport={'width': 1280, 'height': 720}) as browser:
        graph = crawl(browser, log.directory, log, CrawlBudget(max_states=120, max_depth=5,
                      actions_per_page=12, max_attempts=240), explorer=explorer)
    if len(graph.nodes) < 2:
        raise RuntimeError('GitLab crawl has no useful transitions; inspect before formal starts')
    graph.initialize_url_entries(); graph.save()
    layer = build_templates(graph)
    layer_path = graph.directory / 'templates/initial/templates.json'
    write_json(layer_path, layer)
    write_json(graph.directory / 'templates/active.json', {'directory': 'templates/initial'})
    card_dir = graph.directory / 'function-cards/initial'
    card_client = qwen(ROOT / 'auth/llm-crawl.env', EventLog(directory / 'card-calls'), None, 4096,
                       model=MODEL, retry_connections=True, stop_file=suite / 'STOP')
    build_cards(graph, layer_path, card_dir, card_client)
    write_json(graph.directory / 'function-cards/active.json', {'cards': 'function-cards/initial/cards.json'})
    framework_client = qwen(ROOT / 'auth/llm-crawl.env', EventLog(directory / 'framework-calls'), 1, 4096,
                            model=MODEL, retry_connections=True, stop_file=suite / 'STOP')
    build_framework(graph, card_dir / 'cards.json', framework_client, graph.directory / 'site-framework/initial', publish=True)
    # Start the same offline encoder already attested by the existing Reddit index.
    reddit = Graph.load(Path(read(suite / 'reddit/protocol.json')['graph']))
    encoder = LocalEncoder(reddit, Path(read(suite / 'reddit/protocol.json')['encoder_runtime']), 'D:/Anaconda/envs/web2kg-jina-gpu/python.exe')
    try:
        encoder.ensure()
        index = build_visual_index(graph, JinaClient(encoder.index['endpoint'], log, timeout=600))
    finally:
        encoder.close()
    write_json(directory / 'closeout.json', dict(status='completed', nodes=len(graph.nodes),
        edges=len(graph.data['edges']), indexed=len(index['ids']), templates=len(layer['templates']),
        construction_cost_logs=['construction-calls', 'card-calls', 'framework-calls']))
    return graph.directory


def suite_queue(suite):
    status = read(suite / 'suite.json')
    try:
        queue(suite, 'reddit')
        if read(suite / 'reddit/state.json')['status'] != 'completed':
            status['status'] = 'stopped_after_reddit'
            return
        if (suite / 'STOP').exists():
            status['status'] = 'user_stopped'
            return
        released = suite.parent / 'midterm-qwen-assets-20261027/gitlab-released'
        if not read(released / 'download-complete.json', {}).get('verified'):
            status.update(status='needs_attention', gitlab_status='published_graph_download_incomplete')
            return
        if not (suite / 'gitlab/protocol.json').exists():
            status.update(status='gitlab_preparation', gitlab_status='building')
            write_json(suite / 'suite.json', status)
            graph = build_gitlab(suite)
            prepare_site(suite, 'gitlab', graph, released)
        queue(suite, 'gitlab')
        status.update(status=read(suite / 'gitlab/state.json')['status'], gitlab_status=read(suite / 'gitlab/state.json')['status'])
    except Exception as error:
        status.update(status='needs_attention', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(suite / 'suite.json', status)
        report(suite)


def worker(batch, arm, task_id):
    key = f'{arm}/{task_id}'
    protocol, state, _ = verify(batch, key)
    folder = batch / key
    if file_hash(folder / 'input.json') != state['tasks'][key]['input_sha256']:
        raise ValueError('Task input changed')
    config = read(folder / 'input.json')
    if arm == 'ours':
        base.worker(batch / arm, config, task_id)
    else:
        from web2kg.evaluation import webnavigator
        qwen = clients(config['credential_file'], batch / arm / 'routing-state.json')
        webnavigator.OriginalResponses = qwen
        task = next(t for t in read(batch / arm / 'public-inputs.json') if t['task_id'] == task_id)
        webnavigator.run_original(batch / arm, config, task)
    result = read(folder / 'result.json')
    official = score(config['verified_root'], batch / arm, task_id, config['logical_origin'], config['site'])
    write_json(folder / 'official.json', official)
    if official.get('score') is None:
        raise RuntimeError('Official score missing')
    events = [event for relative in ('events.jsonl', 'graph-annotations/events.jsonl',
        'graph-cards/events.jsonl', 'diagnosis-calls/events.jsonl') for event in read_events(folder / relative)]
    if service_error(result, events):
        raise RuntimeError('Engineering or permanent service failure; preserve evidence and stop')
    handoff = dict(official_score=official['score'])
    if arm == 'ours':
        graph = advance_graph(folder)
        update = result.get('execution_graph') or {}
        if update.get('description_stopped') or (update.get('cards') or {}).get('stopped') or (update.get('card_refresh') or {}).get('unavailable'):
            raise RuntimeError('Graph description/card finalization failed; inspect before continuing')
        qwen = clients(config['credential_file'], batch / 'routing-state.json')
        learner = qwen(ROOT / 'auth/llm-crawl.env', EventLog(folder / 'diagnosis-calls'), config['diagnosis_calls'],
                       config['diagnosis_output_tokens'], model=MODEL, retry_connections=True, stop_file=config['stop_file'])
        memory = ExperienceStore(config['experience_store'], memory_namespace(Graph.load(Path(config['graph']))))
        learn_experience(folder, learner, memory, success=official['score'] == 1,
                         verifier='WebArena-Verified official score=1')
        manifest, digest = knowledge_manifest(graph, config['experience_store'], folder / 'knowledge-handoff.json')
        handoff.update(current_graph=str(graph), knowledge_manifest=manifest, knowledge_manifest_sha256=digest)
    write_json(folder / 'handoff.json', handoff)


@contextmanager
def queue_lock(path):
    import msvcrt
    with Path(path).open('a+b') as stream:
        stream.write(b'0'); stream.flush(); stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def queue(suite, site):
    batch = suite / site
    config = read(batch / 'protocol.json')
    with queue_lock(config['shared_lock']):
        config, state, remaining = verify(batch)
        if state['status'] in {'completed', 'regression_stop', 'needs_attention'}:
            return
        encoder = LocalEncoder(Graph.load(Path(state['current_graph'])), Path(config['encoder_runtime']), config['encoder_python'])
        try:
            encoder.ensure()
            for item in remaining:
                if Path(config['stop_file']).exists():
                    state['status'] = 'user_stopped'; break
                config, state, _ = verify(batch)
                arm, task_id = item['arm'], item['task_id']
                key = f'{arm}/{task_id}'
                folder = batch / key
                folder.mkdir(exist_ok=False)
                inputs = {**config, 'graph': state['current_graph'] if arm == 'ours' else config['graph']}
                if arm == 'ours':
                    inputs['cards'] = str(cards_path(Path(inputs['graph'])))
                else:
                    inputs.update(logical_origin=config['backend'], ax_search=False, ax_line_limit=1500, actor_compatibility=True)
                # Reset/login happens before claiming an official task start.
                if site == 'reddit':
                    reset_copy(config['container'], config['image_id'], config['backend'], 10598, inputs['logical_origin'], config['owner'])
                else:
                    reset_gitlab(config)
                    inputs['storage_state'] = login_gitlab(batch, folder)
                write_json(folder / 'input.json', inputs)
                row = dict(status='started', time=time.time(), graph_before=inputs['graph'], input_sha256=file_hash(folder / 'input.json'))
                state['tasks'][key] = row; state['status'] = 'running'
                write_json(batch / 'state.json', state)
                print(f'{site}/{key} started ({len(state["tasks"])}/{len(config["schedule"])})', flush=True)
                started = time.monotonic()
                try:
                    with (folder / 'console.log').open('w', encoding='utf-8') as out:
                        child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(Path(config['frozen_runtime']) / 'scripts/run_midterm_qwen.py'),
                            '--directory', str(suite), '--site', site, '--worker', arm, str(task_id)],
                            cwd=config['frozen_runtime'], stdout=out, stderr=out,
                            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'}, creationflags=subprocess.CREATE_NO_WINDOW)
                        row['worker_pid'] = child.pid; write_json(batch / 'state.json', state)
                        code = child.wait()
                    if code:
                        raise RuntimeError(f'Worker exited {code}; preserve evidence, no automatic task restart')
                    handoff = read(folder / 'handoff.json')
                    state.update({k: handoff[k] for k in ('current_graph', 'knowledge_manifest', 'knowledge_manifest_sha256') if k in handoff})
                    row.update(status='completed', seconds=round(time.monotonic()-started, 3), official_score=handoff['official_score'])
                    write_json(batch / 'state.json', state)
                    summary = report(suite)
                    print(f'{site}/{key}: official={handoff["official_score"]}', flush=True)
                    alarm = summary['sites'][site]['regression']
                    if alarm:
                        state.update(status='regression_stop', regression_reason=alarm)
                        write_json(batch / 'regression-stop.json', dict(reason=alarm, task=key,
                            instruction='Diagnose concrete traces; restart whole new version only after confirmed bug repair and offline tests. Otherwise stop.'))
                        break
                except Exception as error:
                    row.update(status='needs_attention', error=f'{type(error).__name__}: {error}', seconds=round(time.monotonic()-started, 3))
                    state['status'] = 'needs_attention'
                    write_json(batch / 'state.json', state)
                    raise
            else:
                verify(batch)
                state['status'] = 'completed'
                write_json(batch / 'integrity-closeout.json', dict(status='verified', tasks=len(state['tasks']), time=time.time()))
            write_json(batch / 'state.json', state)
        finally:
            encoder.close()
            report(suite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--repair-diagnosis', type=Path)
    parser.add_argument('--site', choices=['reddit', 'gitlab'], default='reddit')
    parser.add_argument('--worker', nargs=2, metavar=('ARM', 'TASK'))
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--suite-live', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(); suite = args.directory.resolve()
    if args.prepare:
        prepare(suite, args.previous.resolve() if args.previous else None,
                args.repair_diagnosis.resolve() if args.repair_diagnosis else None)
    elif args.preflight:
        preflight(suite)
    elif args.worker:
        worker(suite / args.site, args.worker[0], int(args.worker[1]))
    elif args.suite_live:
        suite_queue(suite)
    elif args.live:
        queue(suite, args.site)
    elif args.check:
        _, state, remaining = verify(suite / args.site)
        print(json.dumps(dict(status=state['status'], unstarted=len(remaining))))
    else:
        report(suite)


if __name__ == '__main__':
    main()
