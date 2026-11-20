"""One fresh, registered Qwen crawl followed by clustered cards/framework and local Jina."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_method_transport import clients
from web2kg.common.activity import read_events
from web2kg.common.browser import Browser
from web2kg.common.jina import JinaClient
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.stage1.crawl import CrawlBudget, crawl
from web2kg.stage1.llm_exploration import LLMExplorer
from web2kg.stage1.pipeline import PipelinedExplorer
from web2kg.stage1.jev_ranking import JevRankingClient
from web2kg.stage1.jev_equivalence import JevEquivalenceClient
from web2kg.stage1.template_assets import initialize_assets, rebuild_assets
from web2kg.stage1.visual_index import build_visual_index
from web2kg.stage1.site_framework import load_framework
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.visualization.template_report import export_template_report
from web2kg.visualization.viewer import export_viewer


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_inputs(directory, project):
    receipt = read(directory / 'source-manifest.json')
    for relative, expected in receipt['source'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen source mismatch: ' + relative)
    for relative, expected in receipt['preserved'].items():
        if sha(project / relative) != expected:
            raise ValueError('Preserved input changed: ' + relative)
    if sha(directory / 'plan.json') != receipt['plan_sha256']:
        raise ValueError('Registered plan changed')


@contextmanager
def local_encoder(directory, port, python, *, gpu_memory=None, expected=None):
    """Fresh vectors may use the recorded local identity; never replace a shared service."""
    client = JinaClient(f'http://127.0.0.1:{port}/v1/embeddings', EventLog(directory / 'index-calls'),
                        timeout=1800, retry_connections=True, stop_file=directory / 'STOP')
    process = stream = None
    try:
        try:
            identity = client.identity()
        except OSError:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', port))
            stream = (directory / 'encoder.log').open('x', encoding='utf-8')
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
            process = subprocess.Popen([str(python), '-B', str(ROOT / 'scripts/serve_jina_gpu.py'),
                '--port', str(port), '--vision-fp32', '--log', str(directory / 'encoder-events.jsonl'),
                *(['--gpu-memory', gpu_memory] if gpu_memory else [])],
                cwd=ROOT, env=env, stdout=stream, stderr=stream,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            while True:
                if (directory / 'STOP').exists():
                    raise RuntimeError('User stop file present')
                if process.poll() is not None:
                    raise RuntimeError('Local encoder exited; inspect encoder.log')
                try:
                    identity = client.identity()
                    break
                except OSError:
                    time.sleep(1)
        if expected is not None and identity != expected:
            raise ValueError('Encoder identity differs from expansion seed')
        write_json(directory / 'encoder-identity.json', {'endpoint': client.endpoint, 'encoder': identity,
                    'owned_process': process.pid if process else None, 'offline': True})
        # An actual local query checks the model before any paid construction calls.
        rows = client.encode('website navigation', 'text', identity)
        write_json(directory / 'encoder-check.json', {'shape': list(rows.shape), 'encoder': identity})
        yield client
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        if stream:
            stream.close()


def summarize(directory):
    roles = defaultdict(lambda: dict(physical=0, received=0, parsed=0, retries=0,
        output_retries=0, transport_retries=0,
        input_tokens=0, output_tokens=0, reasoning_tokens=0, request_seconds=0., retry_wait_seconds=0.,
        unknown_usage=0, awaiting_response=0))
    latencies = defaultdict(list)
    providers, models = Counter(), Counter()
    for path in sorted(directory.glob('*-calls/events.jsonl')):
        events = read_events(path)
        starts = {e['request_id']: e for e in events if e['event'] == 'responses_started'}
        received = {e['request_id']: e for e in events if e['event'] == 'responses_received'}
        failed = {e['request_id'] for e in events if e['event'] == 'responses_failed'}
        for key, event in starts.items():
            row = roles[event['purpose']]
            row['physical'] += 1
            if event.get('output_retry_number', 0):
                row['retries'] += 1
                row['output_retries'] += 1
            receipt = received.get(key)
            if receipt is None or not receipt.get('usage'):
                row['unknown_usage'] += 1
            if receipt is None and key not in failed:
                row['awaiting_response'] += 1
        for event in events:
            kind = event['event']
            source = starts.get(event.get('request_id'), {})
            role = event.get('purpose') or source.get('purpose')
            if kind == 'provider_request':
                providers[event['provider']] += 1
            if not role:
                continue
            row = roles[role]
            if kind == 'responses_received':
                row['received'] += 1
                models[event['actual_model']] += 1
                row['input_tokens'] += event.get('usage', {}).get('input_tokens', 0)
                row['output_tokens'] += event.get('usage', {}).get('output_tokens', 0)
                row['reasoning_tokens'] += event.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens', 0)
                row['request_seconds'] += event['seconds']
                latencies[role].append(event['seconds'])
            elif kind == 'responses_failed' and event['request_id'] not in received:
                row['request_seconds'] += event['seconds']
            elif kind == 'responses_output_parsed':
                row['parsed'] += 1
            elif kind == 'responses_retry':
                row['retries'] += 1
                row['transport_retries'] += 1
                row['retry_wait_seconds'] += event['delay_seconds']
    now = datetime.now(timezone.utc)
    state = read(directory / 'state.json')
    plan = read(directory / 'plan.json')
    asset_name = 'expanded' if plan.get('seed_graph') else 'initial'
    if state['status'] == 'running':
        state['observed_wall_seconds'] = (now - datetime.fromisoformat(state['started_at'])).total_seconds()
    for role, row in roles.items():
        row['logical_started'] = row['physical'] - row['retries']
        row['median_received_seconds'] = statistics.median(latencies[role]) if latencies[role] else None
    result = dict(state=state, roles=dict(roles), observed_at=now.isoformat(),
                  providers=dict(providers), actual_models=dict(models))
    path = directory / 'graph/graph.json'
    if path.exists():
        graph = Graph.load(path.parent)
        events = read_events(path.parent / 'events.jsonl')
        finish = next((e for e in reversed(events) if e['event'] == 'crawl_finished'), {})
        result['graph'] = dict(nodes=len(graph.nodes), edges=len(graph.data['edges']),
            unique_urls=len({n['url'] for n in graph.nodes.values()}), reachable=len(graph.reachable()),
            attempts=graph.data.get('attempts'), stop_reason=finish.get('reason'),
            outcomes=dict(Counter(r['outcome'] for r in graph.data.get('transition_records', []))))
        if graph.data.get('expansion'):
            inherited = graph.data['expansion']
            result['expansion'] = {**inherited,
                'annotation_recoveries': dict(Counter(r.get('annotation', {}).get('status') for r in graph.data.get('annotation_recoveries', []))),
                'new_nodes': len(graph.nodes) - inherited['inherited_nodes'],
                'new_attempts': graph.data.get('attempts', 0) - inherited['inherited_attempts'],
                'new_outcomes': dict(Counter(r['outcome'] for r in graph.data.get('transition_records', [])[inherited['inherited_records']:]))}
        if graph.data.get('llm_explorer', {}).get('pipeline'):
            pipeline_events = read_events(directory / 'construction-calls/events.jsonl')
            timeline = sorted((datetime.fromisoformat(e['time']).timestamp(),
                1 if e['event'] == 'pipeline_started' else -1)
                for e in pipeline_events if e['event'] in {'pipeline_started', 'pipeline_finished'})
            active = peak = 0
            overlap = 0.
            previous = None
            for timestamp, delta in timeline:
                if previous is not None and active > 1:
                    overlap += timestamp - previous
                active += delta
                peak = max(peak, active)
                previous = timestamp
            records = graph.data.get('transition_records', [])
            result['pipeline'] = {**graph.data.get('pipeline_summary', {}),
                'peak_active_jobs': peak, 'overlap_seconds': overlap,
                'unused_rankings': sum(e['event'] == 'ranking_unused' for e in pipeline_events),
                'annotations': dict(Counter(r['annotation']['status'] for r in records)),
                'skipped_reasons': dict(Counter(r['annotation'].get('reason') for r in records
                    if r['annotation']['status'] == 'not_requested')),
                'provisional_replay_steps': sum(e['event'] == 'replay_step' and e.get('provisional', False) for e in events)}
        update = path.parent / f'templates/{asset_name}/update.json'
        if update.exists():
            result['clustering'] = read(update)
            layer = read(path.parent / f'templates/{asset_name}/templates.json')
            result['template_sizes'] = sorted((len(g['members']) for g in layer['templates']), reverse=True)
        index = path.parent / 'index.json'
        if index.exists():
            result['indexed'] = len(read(index)['ids'])
    write_json(directory / 'summary.json', result)
    shown = {k: v for k, v in result.items() if k != 'clustering'}
    if 'clustering' in result:
        shown['clustering'] = {k: result['clustering'][k] for k in
            ('method', 'templates_before', 'templates_after', 'timings', 'selection')}
        shown['clustering']['selection'] = {k: v for k,v in shown['clustering']['selection'].items()
                                            if k not in {'predicted', 'candidates'}}
    plan = read(directory / 'plan.json')
    construction_only = plan.get('construction_only', False)
    title = 'Reddit · Qwen 并行建图' if construction_only else 'Reddit · Qwen 新图与平均链接模板'
    if plan.get('ranking_provider') == 'jev':
        title = 'Reddit · Jev 排名与 Qwen 描述' + ('' if construction_only else '、模板与框架')
    body = '<h1>' + title + '</h1><p>独立建图实验；没有运行评测任务，不据此推断任务成功率。</p>'
    body += '<p>当前阶段：' + html.escape(state['phase']) + '；状态：' + html.escape(state['status']) + '</p>'
    links = {'graph/viewer.html': '图谱', 'graph/templates/initial/report.html': '模板',
             'graph/function-cards/initial/cards.html': '功能卡', 'graph/site-framework/initial/report.html': '框架',
             'plan.json': '预先计划', 'summary.json': '原始汇总', 'test-receipt.json': '离线检查'}
    links = {path.replace('/initial/', f'/{asset_name}/'): label for path, label in links.items()}
    if plan.get('seed_graph'):
        body += f"<p>原{plan['inherited_nodes']}状态图保持不变；本副本目标{plan['max_states']}状态。模型调用表只统计本次新增调用，旧图成本不重复计入。相同状态数不代表相同覆盖。输出格式重试与传输重试分别保存在汇总中；不重试浏览器动作。</p>"
    body += '<p>' + ' · '.join(f'<a href="{path}">{label}</a>' for path,label in links.items()
                               if (directory / path).exists()) + '</p>'
    if 'graph' in result:
        g = result['graph']
        body += f"<p>{g['nodes']} 个状态 · {g['edges']} 条边 · {g['unique_urls']} 个 URL · {g['reachable']} 个可回放可达状态</p>"
    body += '<h2>模型调用</h2><table><tr><th>角色</th><th>物理请求</th><th>成功解析</th><th>重试</th><th>未知用量</th><th>输入 token</th><th>输出 token</th><th>其中推理 token</th><th>响应中位秒</th></tr>'
    for role, row in roles.items():
        values = [role] + [row[k] for k in ('physical','parsed','retries','unknown_usage','input_tokens','output_tokens','reasoning_tokens','median_received_seconds')]
        body += '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>'
    body += '</table><p>推理 token 是服务商报告的输出子集，不再累加收费总量；缺失用量不按零成本处理，运行中也包含尚未返回的请求（awaiting_response 单列）。耗时含失败请求；没有按未知价格估算金额。</p>'
    body += '<h2>阶段、分组与审计</h2>'
    body += '<pre>' + html.escape(json.dumps(shown, ensure_ascii=False, indent=2)) + '</pre>'
    body += ('<p>本次只运行建图，没有生成卡片、框架或向量索引；已知页面跳过和暂定路径回放若未在本轮出现，应以离线检查标明，不能称为本轮实测。</p>' if construction_only else
        '<p>δ=0.05来自既有开发样本调参。本次建图的覆盖还受起点、探索预算及模型决策影响；没有独立功能标签时，轮廓分不能视作分类准确率。普通卡7/75%更新和跨任务全局触发另由离线集成测试核验，不冒充本次首次建图的实测收益。</p>')
    (directory / 'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 20px}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#1759af}</style>' + body, encoding='utf-8')
    return result


def run(directory, project, storage, encoder_python, port):
    plan = read(directory / 'plan.json')
    ranking_provider = plan.get('ranking_provider', 'qwen')
    if ranking_provider not in {'qwen', 'jev'}:
        raise ValueError('Unknown ranking provider')
    if ranking_provider == 'jev' and not plan.get('parallel_exploration', True):
        raise ValueError('Separate Jev ranking requires parallel exploration')
    if read(directory / 'state.json')['status'] != 'setup' or (directory / 'started.json').exists():
        raise ValueError('This runner never resumes a started experiment')
    verify_inputs(directory, project)
    started = time.monotonic()
    state = dict(status='running', phase='local_setup', timings={}, started_at=datetime.now(timezone.utc).isoformat())
    write_json(directory / 'started.json', {'pid': os.getpid(), 'time': state['started_at']})

    def phase(name):
        state['phase'] = name
        state['wall_seconds'] = time.monotonic() - started
        write_json(directory / 'state.json', state)
        print(name, flush=True)

    try:
        phase('local_setup')
        with (nullcontext(None) if plan.get('construction_only') else local_encoder(directory, port, encoder_python,
                gpu_memory=plan.get('gpu_memory'), expected=plan.get('encoder_expected'))) as encoder:
            qwen = clients(project / 'auth/qwen-hard.env', directory / 'routing-state.json', provider=plan['provider'],
                           model_override=plan['model'])

            def model(folder, limit):
                client = qwen(project / 'auth/llm-crawl.env', EventLog(directory / folder), limit,
                    plan['output_tokens'], model=plan['model'], retry_connections=True, stop_file=directory / 'STOP')
                client.configure_output_retries(plan.get('max_output_retries', 0))
                return client

            planner = model('construction-calls', plan['max_construction_calls'])
            planner.request_timeout = (10, plan.get('request_read_timeout', 120))
            ranking = (JevRankingClient(project / '.env', EventLog(directory / 'ranking-calls'),
                plan['max_construction_calls'], model=plan.get('ranking_model'),
                retry_connections=True, stop_file=directory / 'STOP', proxy=plan.get('ranking_proxy'))
                if ranking_provider == 'jev' else planner)
            log = EventLog(directory / 'graph')
            judge = (JevEquivalenceClient(project / '.env', EventLog(directory / 'equivalence-calls'),
                plan.get('max_equivalence_calls', 20), model=plan.get('ranking_model'),
                retry_connections=True, stop_file=directory / 'STOP', proxy=plan.get('ranking_proxy'))
                if plan.get('state_equivalence') else None)
            with Browser(plan['start_url'], log, storage_state=str(storage), identity_mode='interaction',
                         viewport={'width': 1280, 'height': 720}) as browser:
                account = os.getenv('VERIFIED_REDDIT_USER', 'MarvelsGrantMan136')
                authenticated = browser.page.get_by_role('button', name=account, exact=True).count() > 0
                write_json(directory / 'authenticated-ui.json', {'passed': authenticated, 'title': browser.page.title()})
                if not authenticated:
                    raise RuntimeError('Fresh browser storage failed actual account-menu login check')
                phase('construction')
                tick = time.monotonic()
                explorer = (PipelinedExplorer(planner, log.directory,
                    description_workers=plan.get('description_workers', 2), max_pending=plan.get('max_pending', 4),
                    ranking_client=ranking)
                    if plan.get('parallel_exploration', True) else LLMExplorer(planner, log.directory))
                graph = crawl(browser, log.directory, log, CrawlBudget(max_states=plan['max_states'],
                    max_depth=plan['max_depth'], actions_per_page=plan['actions_per_page'],
                    max_attempts=plan['max_attempts']), explorer=explorer, state_judge=judge,
                    seed=project / plan['seed_graph'] if plan.get('seed_graph') else None,
                    recover_annotations=plan.get('recover_annotations', False))
                state['timings']['construction_seconds'] = time.monotonic() - tick
            if len(graph.nodes) < 2:
                raise RuntimeError('Construction has no useful transitions')
            graph.initialize_url_entries()
            graph.save()
            if plan.get('construction_only'):
                export_viewer(graph)
                verify_inputs(directory, project)
                state['status'] = 'completed'
                phase('completed')
                return
            phase('initial_clustering_cards_framework')
            tick = time.monotonic()
            asset_name = 'expanded' if plan.get('seed_graph') else 'initial'
            staged = (rebuild_assets if plan.get('seed_graph') else initialize_assets)(graph, model('card-calls', plan['max_card_calls']),
                framework_client=model('framework-calls', plan['max_framework_calls']),
                method=plan['clustering'], tolerance=plan['tolerance'], name=asset_name)
            state['timings'].update(staged['audit']['timings'])
            state['timings']['initial_assets_seconds'] = time.monotonic() - tick
            phase('local_index')
            tick = time.monotonic()
            if plan.get('seed_graph'):
                from complete_paused_graph_index import CheckpointClient
                encoder = CheckpointClient(encoder, directory / 'vector-checkpoints')
            index = build_visual_index(graph, encoder)
            state['timings']['index_seconds'] = time.monotonic() - tick
            phase('asset_validation')
            knowledge = TemplateKnowledge(graph, graph.directory / f'function-cards/{asset_name}/cards.json',
                                          EventLog(directory / 'validation'))
            framework = load_framework(graph)
            checks = dict(all_nodes_assigned=set(staged['layer']['assignments']) == set(graph.nodes),
                all_cards_completed=all(r['status'] == 'completed' for r in staged['cards']['cards'].values()),
                all_templates_have_cards=set(knowledge.groups) == set(knowledge.cards),
                framework_loaded=framework is not None, index_matches_nodes=set(index['ids']) == set(graph.nodes))
            write_json(directory / 'asset-check.json', checks)
            if not all(checks.values()):
                raise ValueError('Final asset validation failed')
            export_template_report(graph, staged['layer'], graph.directory / f'templates/{asset_name}',
                                   graph.directory / f'function-cards/{asset_name}/cards.json')
            export_viewer(graph)
            verify_inputs(directory, project)
        state['status'] = 'completed'
        phase('completed')
    except Exception as error:
        state.update(status='user_stopped' if (directory / 'STOP').exists() else 'failed', error_type=type(error).__name__)
        # No credentials or remote error bodies in the public receipt.
        phase(state['phase'])
        raise
    finally:
        state['wall_seconds'] = time.monotonic() - started
        write_json(directory / 'state.json', state)
        if (directory / 'graph/graph.json').exists():
            export_viewer(Graph.load(directory / 'graph'))
        summarize(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--storage', type=Path)
    parser.add_argument('--encoder-python', type=Path, default=Path('D:/Anaconda/envs/web2kg-jina-gpu/python.exe'))
    parser.add_argument('--encoder-port', type=int, default=8796)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summarize(args.directory.resolve())
    else:
        if args.storage is None:
            parser.error('--storage is required for a fresh live run')
        run(args.directory.resolve(), args.project.resolve(), args.storage.resolve(), args.encoder_python, args.encoder_port)
