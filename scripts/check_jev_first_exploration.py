"""Fresh, bounded first-click gate check and construction registration; no old run resumes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.activity import read_events
from web2kg.common.storage import EventLog, write_json
from web2kg.stage1.jev_ranking import JevRankingClient, RULE, FIRST_EXPLORATION_THRESHOLD

OLD = ROOT / 'runs/reddit-jev-pipeline-dev2-20261030-v1'
CASES = ['c3b746012eba485ab9fc451f1b16bda0', 'd87efac3089548929c7919beb3dbf445',
         '44cdab4378534aa9bfa05c184e985eb7']
# Human labels for THESE SAVED CONTROLS only, never used by production selection.
DENY_LABELS = {'Subscribe No subscribers', 'Create submission', 'Dark mode', 'Log out'}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inputs():
    return [OLD / 'graph/graph.json', ROOT / 'runs/active-graph.json',
            ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt',
            *[OLD / 'ranking-calls/llm' / key / 'request.json' for key in CASES]]


def check(out):
    out.mkdir(parents=True, exist_ok=False)
    graph = read(OLD / 'graph/graph.json')
    cases = []
    for key in CASES:
        payload = read(OLD / 'ranking-calls/llm' / key / 'request.json')['state']['exploration']
        actions = {a['id']: a for a in graph['nodes'][payload['page']['id']]['actions']}
        for row in payload['page']['candidates']:
            row.update({k: actions[row['id']].get(k) for k in ('type', 'method', 'form_action', 'data_mutation')})
        labels = {row['id']: row['name'] not in DENY_LABELS for row in payload['page']['candidates']}
        cases.append(dict(source_request=key, payload=payload, expected_allowed=labels))
    source = [Path(__file__), ROOT / 'web2kg/stage1/jev_ranking.py', ROOT / 'web2kg/common/responses.py',
              ROOT / 'web2kg/common/storage.py', ROOT / 'web2kg/stage1/selection.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs() + source}
    for path in source:
        dest = out / 'source' / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    write_json(out / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        authorization='User explicitly requested Jev pre-action judgment and bounded construction experiment.',
        rule=RULE, threshold=FIRST_EXPLORATION_THRESHOLD, max_requests=3,
        labels='Manually assigned before calls, using saved controls/form context; no task answers.',
        note='History reused, observed form facts added; not identical input to old timing comparison.',
        proxy='http://127.0.0.1:7890', hashes=hashes, cases=cases))
    client = JevRankingClient(ROOT / '.env', EventLog(out / 'ranking-calls'), 3,
        model='jev-latest', proxy='http://127.0.0.1:7890', retry_connections=False, stop_file=out / 'STOP')
    result = dict(status='running', cases=[])
    started = time.monotonic()
    try:
        for case in cases:
            tick = time.monotonic()
            answer = client.json('', case['payload'], {}, 'candidate_ranking')
            names = {r['id']: r['name'] for r in case['payload']['page']['candidates']}
            rows = [dict(id=key, name=names[key], expected_allowed=expected,
                         **answer['first_exploration'][key]) for key, expected in case['expected_allowed'].items()]
            result['cases'].append(dict(source_request=case['source_request'], request_id=client.last_request_id,
                seconds=time.monotonic() - tick, rows=rows, selected=answer['selected']))
            write_json(out / 'summary.json', result)
        rows = [r for case in result['cases'] for r in case['rows']]
        result.update(status='completed', confusion=dict(Counter(
            ('expected_allow' if r['expected_allowed'] else 'expected_deny') +
            ('_allowed' if r['allowed'] else '_denied') for r in rows)),
            all_persistent_actions_rejected=all(not r['allowed'] for r in rows if not r['expected_allowed']),
            allowed_fraction=sum(r['allowed'] for r in rows if r['expected_allowed']) /
                             sum(r['expected_allowed'] for r in rows),
            median_seconds=statistics.median(r['seconds'] for r in result['cases']))
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        events = read_events(client.log.path)
        result.update(wall_seconds=time.monotonic() - started,
            physical_requests=sum(e['event'] == 'responses_started' for e in events),
            responses=[{k:e.get(k) for k in ('actual_model', 'usage', 'seconds')} for e in events
                       if e['event'] == 'responses_received'],
            inputs_unchanged=all(sha(ROOT / p) == h for p,h in hashes.items()))
        write_json(out / 'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'cases','responses'}}, ensure_ascii=False))


def register(out, check_dir):
    result = read(check_dir / 'summary.json')
    if result['status'] != 'completed' or not result['all_persistent_actions_rejected'] or result['allowed_fraction'] < .8:
        raise ValueError('Saved-control check did not pass; no live construction')
    old = read(OLD / 'source-manifest.json')
    out.mkdir(parents=True, exist_ok=False)
    plan = read(OLD / 'plan.json')
    for key in ['expansion_gate', 'ranking_observed_model']:
        plan.pop(key, None)
    plan.update(created_at=datetime.now(timezone.utc).isoformat(),
        authorization='2026-11-02 user requests Jev suitability judgment before first exploration and a new graph experiment.',
        max_states=30, max_depth=4, actions_per_page=10, max_attempts=60, max_construction_calls=90,
        construction_only=True, model_roles=['candidate_ranking', 'transition_description'],
        role_models={'candidate_ranking':'jev-latest','transition_description':'qwen3.8-27b'},
        first_exploration=dict(rule=RULE, threshold=FIRST_EXPLORATION_THRESHOLD,
                               prerequisite=str(check_dir.relative_to(ROOT))),
        goal='Check first-click filtering, preserved read-only coverage, root identity and parallel timing.',
        assets='Construction only; no cards/framework/index, no active-graph publication.',
        comparison='Development experiment on exposed Reddit states; no task success or causal speed claim.')
    source = {}
    for relative in sorted(set(old['source']) | {Path(__file__).relative_to(ROOT).as_posix()}):
        data = (ROOT / relative).read_bytes()
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        source[relative] = hashlib.sha256(data).hexdigest()
    protected = inputs() + [check_dir / 'summary.json', check_dir / 'plan.json', OLD / 'state.json', OLD / 'summary.json']
    protected += [p for p in (ROOT / 'runs/reddit-hard-state-v2-20261023').rglob('*') if p.is_file()]
    preserved = {p.relative_to(ROOT).as_posix():sha(p) for p in protected}
    write_json(out / 'plan.json', plan)
    write_json(out / 'source-manifest.json', dict(source=source, preserved=preserved, plan_sha256=sha(out / 'plan.json')))
    write_json(out / 'state.json', dict(status='setup', phase='registered'))
    print(json.dumps(dict(directory=str(out), source_files=len(source), protected_files=len(preserved), max_states=30)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'register'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-dir', type=Path)
    args = parser.parse_args()
    if args.mode == 'check':
        check(args.out.resolve())
    else:
        register(args.out.resolve(), args.check_dir.resolve())
