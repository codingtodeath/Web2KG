"""Compare Jev on saved Qwen candidate payloads; never execute browser actions."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web2kg.common.activity import read_events
from web2kg.common.responses import ResponsesError
from web2kg.common.storage import EventLog, write_json
from web2kg.stage1.jev_ranking import JevRankingClient


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(baseline, out, env_file, model, live, proxy=None):
    if not live:
        raise ValueError('Pass --live for the explicitly bounded Jev experiment')
    out.mkdir(parents=True, exist_ok=False)
    event_file = baseline / 'construction-calls/events.jsonl'
    events = read_events(event_file)
    starts = [e for e in events if e['event'] == 'responses_started' and e['purpose'] == 'candidate_ranking']
    if len(starts) != 5:
        raise ValueError('This comparison expects the five saved Qwen rankings')
    receipts = {e['request_id']: e for e in events if e['event'] == 'responses_received'}
    plans = {}
    inputs = [event_file]
    for path in (baseline / 'graph/rankings').glob('*.json'):
        plan = read(path)['plan']
        plans[plan['request_id']] = plan
        inputs.append(path)
    for event in starts:
        inputs.append(baseline / 'construction-calls/llm' / event['request_id'] / 'request.json')
    inputs += [ROOT / 'runs/active-graph.json', ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt']
    source_paths = [Path(__file__), ROOT / 'web2kg/stage1/jev_ranking.py',
        ROOT / 'web2kg/stage1/selection.py', ROOT / 'web2kg/common/responses.py', ROOT / 'web2kg/common/storage.py']
    before = {str(p.resolve()): sha(p) for p in inputs + source_paths}
    for path in source_paths:
        target = out / 'source' / path.resolve().relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write_json(out / 'plan.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'model': model, 'proxy': proxy, 'baseline': str(baseline), 'max_new_logical_requests': 5,
        'policy': 'One pass over all five saved candidate payloads; stop on first failure, no retries.',
        'scope': 'Candidate ranking only, no browser, Qwen, Jina, cards, tasks or graph publication.',
        'comparison_limit': 'Identical candidate/context data, different typed scoring and structural grouping; Qwen is not ground truth.',
        'hashes': before})
    ranker = JevRankingClient(env_file, EventLog(out / 'ranking-calls'), max_calls=5,
        model=model, retry_connections=False, stop_file=out / 'STOP', proxy=proxy)
    result = {'status': 'running', 'cases': [], 'model_requested': model}
    started = time.monotonic()
    try:
        for event in starts:
            key = event['request_id']
            request = read(baseline / 'construction-calls/llm' / key / 'request.json')
            payload = yaml.safe_load(request['input'][0]['content'][0]['text'])
            tick = time.monotonic()
            selected = ranker.json('', payload, request['text']['format']['schema'], 'candidate_ranking')['selected']
            previous = plans[key]['selected']
            actual = read(out / 'ranking-calls/llm' / ranker.last_request_id / 'response.json')
            row = {'baseline_request_id': key, 'request_id': ranker.last_request_id,
                'candidates': len(payload['page']['candidates']), 'qwen_seconds': receipts[key]['seconds'],
                'jev_seconds': time.monotonic() - tick, 'jev_selected': selected, 'qwen_selected': previous,
                'selected_id_overlap': len({r['id'] for r in selected} & {r['id'] for r in previous}),
                'actual_model': actual['model'], 'usage': actual.get('usage', {})}
            result['cases'].append(row)
            write_json(out / 'summary.json', result)
        result['status'] = 'completed'
    except ResponsesError as error:
        result.update(status='blocked', http_status=error.http_status, failure_kind=error.kind,
            error=str(error), request_id=getattr(ranker, 'last_attempt_id', None))
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__)
    finally:
        ranker.session.close()
        result['wall_seconds'] = time.monotonic() - started
        result['integrity'] = {path: sha(Path(path)) == value for path, value in before.items()}
        if not all(result['integrity'].values()):
            result['status'] = 'integrity_failed'
        calls = read_events(ranker.log.path)
        result['physical_requests'] = sum(e['event'] == 'responses_started' for e in calls)
        result['received'] = sum(e['event'] == 'responses_received' for e in calls)
        result['unknown_usage'] = result['physical_requests'] - sum(
            e['event'] == 'responses_received' and bool(e.get('usage')) for e in calls)
        result['input_tokens'] = sum(e.get('usage', {}).get('input_tokens', 0) for e in calls if e['event'] == 'responses_received')
        result['output_tokens'] = sum(e.get('usage', {}).get('output_tokens', 0) for e in calls if e['event'] == 'responses_received')
        if result['cases']:
            result['median_jev_seconds'] = statistics.median(r['jev_seconds'] for r in result['cases'])
        write_json(out / 'summary.json', result)
        rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in
            (i + 1, row['candidates'], round(row['qwen_seconds'], 3), round(row['jev_seconds'], 3), row['selected_id_overlap'])) + '</tr>'
            for i, row in enumerate(result['cases']))
        (out / 'report.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Jev 候选排名检查</title>
<style>body{font:17px/1.7 system-ui;max-width:1000px;margin:50px auto;padding:0 24px;color:#223}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}code{background:#eee}</style>
<h1>Jev 候选排名检查</h1>''' + f'<p>状态：{html.escape(result["status"])}；模型：{html.escape(model)}</p>'
            + '<p>从已完成的 Qwen 建图复用全部五份候选输入，每份最多调用一次 Jev。描述与浏览器均未运行。模型分数按降序选择，每个现有结构组先保留一个代表；不生成虚构功能组名。</p>'
            + f'<p>完成 {len(result["cases"])}/5；物理请求 {result["physical_requests"]}；未知用量 {result["unknown_usage"]}。'
            + (f'接口返回 HTTP {result["http_status"]}，停止后续请求；没有有效排名输出，不能报告提速或质量结论。' if result.get('http_status') else '') + '</p>'
            + '<table><tr><th>输入</th><th>候选数</th><th>原 Qwen 秒</th><th>Jev 秒</th><th>相同候选数</th></tr>' + rows + '</table>'
            + '<p>选择重合度不等于正确率，原 Qwen 也不是标准答案。五份上下文均来自 Reddit，不能据此推断泛化。HTTP 拒绝没有返回 token 用量，费用未知，不计为零成本。</p>'
            + '<p><a href="summary.json">汇总与完整性</a> · <a href="plan.json">登记及来源</a></p></html>', encoding='utf-8')
    print(json.dumps({k: result.get(k) for k in ['status', 'http_status', 'physical_requests', 'received', 'unknown_usage', 'wall_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--env', type=Path, default=ROOT / '.env')
    parser.add_argument('--model', default='jev-1.14.0')
    parser.add_argument('--proxy', default=None)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    run(args.baseline.resolve(), args.out.resolve(), args.env.resolve(), args.model, args.live, args.proxy)
