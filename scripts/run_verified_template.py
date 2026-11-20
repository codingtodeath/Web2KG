"""Resume only unstarted, preregistered tasks in an isolated Verified benchmark copy."""
import argparse
import hashlib
import html
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.common.jina import JinaClient
from web2kg.common.jina_runtime import LocalEncoder
from web2kg.common.responses import ResponsesClient
from web2kg.common.safety import origin
from web2kg.stage1.function_cards import file_hash
from web2kg.stage2.agent import run_task
from web2kg.evaluation.webnavigator import OriginalResponses
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.evaluation.verified import reset_copy, score
from web2kg.stage1.site_framework import load_framework
from web2kg.stage2.recovery import ExperienceStore, memory_namespace, learn_experience


def source_hash():
    paths = sorted(list((ROOT / 'web2kg').rglob('*.py')) + list((ROOT / 'scripts').glob('*.py')) +
                   [p for p in (ROOT / 'third_party/webnavigator').rglob('*') if p.is_file() and p.suffix in {'.py', '.txt', '.json', '.js'}] +
                   [ROOT / 'examples/webnavigator-execution.yml'])
    return hashlib.sha256(json.dumps([(p.relative_to(ROOT).as_posix(), file_hash(p)) for p in paths]).encode()).hexdigest()


def check_inputs(config):
    if source_hash() != config['source_sha256']:
        raise ValueError('Source changed during registered batch')
    if any(file_hash(path) != expected for path, expected in config['protected_inputs'].items()):
        raise ValueError('Registered input changed')


def provider_gate(result):
    """Stop the batch on the first service failure; never consume more task starts."""
    return (result.get('provider_failure', {}).get('kind') in {'http', 'transport'} or
            bool(re.search(r'\bHTTP [45]\d\d\b', result.get('error', ''))))


def execution_clients(batch, config):
    if config.get('execution_provider') == 'qwen':
        from qwen_method_transport import clients
        qwen = clients(config['credential_file'], batch / 'routing-state.json')
        return qwen, qwen
    return OriginalResponses, ResponsesClient


def worker(batch, config, task_id):
    check_inputs(config)
    public = json.loads((batch / 'public-inputs.json').read_text(encoding='utf-8'))
    task = next(t for t in public if t['task_id'] == task_id)
    if set(task) != {'task_id', 'sites', 'intent', 'start_urls'}:
        raise ValueError('Non-public task fields rejected')
    graph = Graph.load(Path(config['graph']))
    log = EventLog(batch / str(task_id))
    actor_class, diagnosis_class = execution_clients(batch, config)
    client = actor_class(ROOT / 'auth/llm-crawl.env', log, config['max_calls'], config.get('output_tokens_per_call', 2048),
        model=config['model'], max_retries=config.get('max_retries', 0),
        retry_connections=config.get('retry_connections', False), stop_file=config.get('stop_file'))
    connection = dict(retry_connections=config.get('retry_connections', False), stop_file=config.get('stop_file'))
    knowledge = TemplateKnowledge(graph, config['cards'], log)
    description_limit = config.get('transition_calls', None if config.get('update_graph') else 0)
    if (description_limit is not None and description_limit < 0) or (description_limit != 0 and not config.get('update_graph', False)):
        raise ValueError('Register graph updates and a nonnegative annotation budget together')
    auxiliary_class = diagnosis_class if config.get('auxiliary_provider') == 'qwen' else ResponsesClient
    auxiliary_tokens = config.get('auxiliary_output_tokens', 2048)
    auxiliary_model = config.get('auxiliary_model', 'gpt-6-luna')
    descriptions = (auxiliary_class(ROOT / 'auth/llm-crawl.env', EventLog(log.directory / 'graph-annotations'),
                    description_limit, auxiliary_tokens, model=auxiliary_model, **connection) if description_limit != 0 else None)
    card_limit = config.get('card_calls', None if config.get('update_graph') else 0)
    if (card_limit is not None and card_limit < 0) or (card_limit != 0 and not config.get('update_graph', False)):
        raise ValueError('Register graph updates and a nonnegative card budget together')
    card_client = (auxiliary_class(ROOT / 'auth/llm-crawl.env', EventLog(log.directory / 'graph-cards'),
                                  card_limit, auxiliary_tokens, model=auxiliary_model, **connection) if card_limit != 0 else None)
    diagnostic_limit = config.get('diagnosis_calls', 0)
    total_cap = config.get('max_total_model_calls')
    limits = (config['max_calls'], description_limit, card_limit, diagnostic_limit)
    if total_cap is not None and (None in limits or total_cap < sum(limits)):
        raise ValueError('A finite total cap conflicts with the registered per-role limits')
    diagnosis = (diagnosis_class(ROOT / 'auth/llm-crawl.env', EventLog(log.directory / 'diagnosis-calls'),
        diagnostic_limit, config.get('diagnosis_output_tokens', 2048), model=config['diagnosis_model'], **connection) if diagnostic_limit else None)
    framework = load_framework(graph, None if config.get('framework') == 'auto' else config['framework']) if config.get('framework') else None
    memory = ExperienceStore(config['experience_store'], memory_namespace(graph)) if config.get('experience_store') else None
    embedding = JinaClient(config['encoder_endpoint'], log, timeout=config.get('encoder_timeout', 180), **connection)
    site = config.get('site', 'reddit')
    headers = ({'X-Postmill-Auto-Login': os.getenv('VERIFIED_REDDIT_USER', 'MarvelsGrantMan136') + ':' +
               os.environ['VERIFIED_REDDIT_PASSWORD']} if site == 'reddit' else {})
    result = run_task(graph, task['intent'], log, client, embedding,
        start_url=task['start_urls'][0], backend=config['backend'], headers=headers,
        site=site, storage_state=config.get('storage_state'),
        allow_writes=True, max_steps=config['max_steps'], top_k=config['top_k'], knowledge=knowledge,
        update_graph=config.get('update_graph', False), description_client=descriptions, max_descriptions=description_limit,
        card_client=card_client, max_cards=card_limit,
        finalize_graph_assets=config.get('finalize_graph_assets', False), framework=framework,
        framework_refresh_updates=config.get('framework_refresh_updates', 10),
        template_clustering=config.get('template_clustering', 'average'), template_tolerance=config.get('template_tolerance', .05),
        template_refresh_min=config.get('template_refresh_min', 5), template_refresh_fraction=config.get('template_refresh_fraction', .2),
        max_graph_nodes=config.get('max_graph_nodes', 500), retention_half_life=config.get('retention_half_life', 20),
        diagnosis_client=diagnosis, experience_store=memory, max_diagnoses=config.get('max_diagnoses', 3),
        max_replay_actions=config.get('recovery_actions', 12), recovery_seed=config.get('seed', 0),
        ax_line_limit=config.get('ax_line_limit', 1000), ax_search=config.get('ax_search', True),
        direction_confidence=config.get('direction_confidence'))
    # Preserve the exact model final answer; the official evaluator validates its schema.
    (log.directory / 'agent_response.json').write_text(result.get('answer', ''), encoding='utf-8')


def summarize(batch, config, state):
    rows = []
    for task_id in config['task_ids']:
        folder = batch / str(task_id)
        row = {'task_id': task_id, 'status': state.get(str(task_id), {}).get('status', 'not_started'), 'official_score': None}
        if (folder / 'result.json').exists():
            result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
            row.update(agent_status=result['status'], navigation=result.get('navigation'),
                       steps=result.get('steps'), seconds=result.get('seconds'), final_url=result.get('url'))
        if (folder / 'official.json').exists():
            official = json.loads((folder / 'official.json').read_text(encoding='utf-8'))
            row.update(official_score=official.get('score'), official_status=official.get('status'))
        events = [json.loads(s) for s in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines()] if (folder / 'events.jsonl').exists() else []
        annotation_log = folder / 'graph-annotations/events.jsonl'
        annotations = [json.loads(s) for s in annotation_log.read_text(encoding='utf-8').splitlines()] if annotation_log.exists() else []
        row['annotation_calls'] = sum(e['event'] == 'responses_started' for e in annotations)
        card_log = folder / 'graph-cards/events.jsonl'
        card_events = [json.loads(s) for s in card_log.read_text(encoding='utf-8').splitlines()] if card_log.exists() else []
        row['card_calls'] = sum(e['event'] == 'responses_started' for e in card_events)
        diagnosis_log = folder / 'diagnosis-calls/events.jsonl'
        diagnostic_events = [json.loads(s) for s in diagnosis_log.read_text(encoding='utf-8').splitlines()] if diagnosis_log.exists() else []
        row['diagnosis_and_learning_calls'] = sum(e['event'] == 'responses_started' for e in diagnostic_events)
        row['recovery'] = result.get('recovery') if (folder / 'result.json').exists() else None
        events += annotations + card_events + diagnostic_events
        received = [e for e in events if e['event'] == 'responses_received']
        row.update(calls=sum(e['event'] == 'responses_started' for e in events),
            responses=len(received), input_tokens=sum(e.get('usage', {}).get('input_tokens', 0) for e in received),
            output_tokens=sum(e.get('usage', {}).get('output_tokens', 0) for e in received),
            actual_models=sorted({e['actual_model'] for e in received}),
            unknown_template_observations=sum(e['event'] == 'template_match' and e['template_id'] is None for e in events))
        rows.append(row)
    summary = {'planned': len(rows), 'official_successes': sum(r['official_score'] == 1 for r in rows),
               'official_scored': sum(r['official_score'] is not None for r in rows), 'tasks': rows,
               'scope': 'Previously exposed benchmark; no blind/generalization or causal template benefit claim'}
    write_json(batch / 'summary.json', summary)
    table = ''.join('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in (
        r['task_id'], r['status'], r.get('agent_status', ''), r['official_score'],
        (r.get('navigation') or {}).get('success', ''), r.get('steps', ''), r['calls'],
        r['annotation_calls'], r['card_calls'], r['input_tokens'], r['output_tokens'])) + '</tr>' for r in rows)
    page = ('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        '<title>Reddit Hard 10 · 模板Agent</title><style>body{font:16px/1.6 system-ui;max-width:1200px;'
        'margin:30px auto}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style>'
        '<h1>Reddit Hard · 模板Agent</h1><p>复用登记图谱与功能卡；'
        f'top-k={config.get("top_k", 5)}，每题{config.get("max_steps", 30)}步/'
        f'{config.get("max_calls", 31)}次Agent调用，另限{config.get("transition_calls", "不限")}次Luna描述、制卡上限{config.get("card_calls", "不限")}。'
        '调用和token合计包含描述与制卡，次数另列。旧项目访问过这些题，不是新盲测；没有对照组，不能归因图谱收益。</p>'
        f'<p>官方成功 {summary["official_successes"]}/{len(rows)}；已评分 {summary["official_scored"]}/{len(rows)}。'
        '未评分不算失败或成功。金额未知，仅报告调用/token。</p><table><tr>' +
        ''.join('<th>' + c + '</th>' for c in ['任务','批次状态','Agent状态','官方分','路径成功','Agent步数','全部调用','其中描述调用','其中制卡调用','输入token','输出token']) +
        '</tr>' + table + '</table><p>每题原始结果、截图、完整轨迹和官方评分均保留。评分轨迹仅去除内部转发重复项，不改浏览器请求/响应；完整传输HAR另存。</p>')
    (batch / 'report.html').write_text(page, encoding='utf-8')
    public_report = Path(config['graph']) / 'reports' / (batch.name + '.html')
    public_report.parent.mkdir(exist_ok=True)
    public_report.write_text(page, encoding='utf-8')
    return summary


def queue(batch, config, *, worker_script=None, summarizer=summarize):
    import msvcrt
    with (batch / 'queue.lock').open('a+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        check_inputs(config)
        graph = Graph.load(Path(config['graph']))
        encoder = LocalEncoder(graph, ROOT, config['encoder_python'])
        state_path = batch / 'state.json'
        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
        try:
            encoder.ensure()
            for task_id in config['task_ids']:
                key = str(task_id)
                if key in state:
                    if state[key]['status'] in {'infrastructure_error', 'provider_error', 'started'}:
                        raise RuntimeError('Prior unresolved interruption/gate; inspect before continuation')
                    continue
                check_inputs(config)
                folder = batch / key; folder.mkdir(exist_ok=False)
                state[key] = {'status': 'started', 'time': time.time()}
                write_json(state_path, state)
                print(f'Task {task_id} started', flush=True)
                try:
                    container_id = reset_copy(config['container'], config['image_id'], config['backend'],
                        10598, config['logical_origin'], config['owner'])
                    state[key]['container_id'] = container_id
                    write_json(state_path, state)
                    with (folder / 'console.log').open('w', encoding='utf-8') as out:
                        process = subprocess.Popen([sys.executable, str(worker_script or Path(__file__).resolve()), '--directory',
                            str(batch), '--worker', key], cwd=ROOT, stdout=out, stderr=out,
                            env={**os.environ, 'PYTHONUTF8': '1'}, creationflags=subprocess.CREATE_NO_WINDOW)
                        try:
                            code = process.wait(timeout=config['task_timeout_seconds'])
                        except subprocess.TimeoutExpired:
                            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                            raise RuntimeError('Owned task worker exceeded wall-time cap')
                    if code:
                        raise RuntimeError(f'Worker exited {code}; see task console')
                    official = score(config['verified_root'], batch, task_id, config['logical_origin'])
                    write_json(folder / 'official.json', official)
                    if official.get('score') is None:
                        raise RuntimeError('Official scorer did not produce a score; stopping before further paid tasks')
                    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
                    if config.get('diagnosis_calls') and config.get('experience_store'):
                        _, diagnosis_class = execution_clients(batch, config)
                        diagnostic_client = diagnosis_class(ROOT / 'auth/llm-crawl.env', EventLog(folder / 'diagnosis-calls'),
                            config['diagnosis_calls'], 2048, model=config['diagnosis_model'],
                            retry_connections=config.get('retry_connections', False), stop_file=config.get('stop_file'))
                        memory = ExperienceStore(config['experience_store'], memory_namespace(graph))
                        learn_experience(folder, diagnostic_client, memory, success=official.get('score') == 1,
                                         verifier='WebArena-Verified official score=1')
                    gated = provider_gate(result)
                    state[key]['status'] = 'provider_error' if gated else 'completed'
                    write_json(state_path, state)
                    summary = summarizer(batch, config, state)
                    print(f'Task {task_id}: official={official.get("score")}; {result["status"]}; total calls={sum(r["calls"] for r in summary["tasks"])}', flush=True)
                    if gated:
                        break
                except Exception as error:
                    state[key].update(status='infrastructure_error', error=type(error).__name__ + ': ' + str(error))
                    write_json(state_path, state)
                    summarizer(batch, config, state)
                    raise
        finally:
            encoder.close()
            summarizer(batch, config, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    batch = args.directory.resolve()
    config = json.loads((batch / 'protocol.json').read_text(encoding='utf-8'))
    worker(batch, config, args.worker) if args.worker is not None else queue(batch, config)


if __name__ == '__main__':
    main()
