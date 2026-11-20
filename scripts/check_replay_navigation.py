"""Real Reddit route checks for every previously failed teleport, without model calls."""
import argparse
import hashlib
import html
import json
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.common.replay_policy import INPUT_KINDS, executable
from web2kg.stage2.navigation import RouteNavigator, state_matches
from web2kg.stage2.transport import LocalSiteProxy

IMAGE = 'sha256:0594908059a03e5f610005689440a95c05e776e734006197c7b1eba12734cb46'
NAME = 'web2kg-execution-check'
OWNER = 'recorded-replay-20261018'
LOGICAL = 'http://127.0.0.1:10499'
BACKEND = 'http://127.0.0.1:10699'


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True, encoding='utf-8').strip()


def create_container():
    names = docker('ps', '-a', '--format', '{{.Names}}').splitlines()
    if NAME in names:
        raise ValueError('Existing check container is preserved; choose/inspect ownership before another run')
    for port in (10699, 10698):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    cid = docker('run', '-d', '--pull=never', '--name', NAME,
        '--label', 'web2kg.execution_check=' + OWNER,
        '-p', '127.0.0.1:10699:80', '-p', '127.0.0.1:10698:8877', IMAGE)
    with requests.Session() as session:
        session.trust_env = False
        for _ in range(120):
            try:
                if session.get(BACKEND, timeout=2).status_code < 500:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('Owned check backend did not start')
    docker('exec', NAME, 'env-ctrl', 'init', '--base-url', LOGICAL)
    return cid


class CheckBrowser(Browser):
    """Same observations/actions; retain compact arrival evidence instead of duplicate DOMs."""
    def __init__(self, *args, proxy, evidence, **kwargs):
        super().__init__(*args, **kwargs)
        self.proxy, self.evidence = proxy, evidence
        self.observations = []
        self.acts = []

    def __enter__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True,
            proxy={'server': self.proxy.url, 'bypass': '<-loopback>'},
            args=['--proxy-bypass-list=<-loopback>'])
        try:
            self.reset()
        except BaseException:
            self.__exit__()
            raise
        return self

    def observe(self, directory=None):
        obs = super().observe()
        compact = {k: v for k, v in obs.items() if k not in {'ax', 'dom', 'text'}}
        key = hashlib.sha256(json.dumps(compact, sort_keys=True).encode()).hexdigest()[:24]
        folder = self.evidence / key
        if not (folder / 'observation.json').exists():
            write_json(folder / 'observation.json', compact, compact=True)
            self.screenshot(folder / 'page.png')
        self.observations.append({'id': obs['id'], 'url': obs['url'], 'evidence': key})
        return obs

    def act(self, action, strategy='role'):
        self.acts.append({k: action.get(k) for k in ('kind', 'role', 'name', 'xpath')})
        return super().act(action, strategy)


class CheckNavigator(RouteNavigator):
    def save(self):
        # Route computation is unchanged. The base graph is already frozen by SHA;
        # only per-check entry registration needs retaining, not 29 full graph copies.
        write_json(self.graph.directory / 'entry-state.json', {
            'base_graph': str(self.base.directory), 'base_graph_sha256': sha(self.base.directory / 'graph.json'),
            'url_entries': self.graph.data.get('url_entries'),
            'task_start': self.graph.data.get('task_start'), 'navigation_home': self.graph.data.get('navigation_home'),
            'added_nodes': {k: v for k, v in self.graph.nodes.items() if k not in self.base.nodes}})


def graph_audit(graph):
    index = json.loads((graph.directory / 'index.json').read_text(encoding='utf-8'))
    reachable = graph.reachable()
    assert set(index['ids']) == reachable, 'Index must exactly match replay reachability'
    inputs = [e for e in graph.data['edges'] if e['action']['kind'] in INPUT_KINDS]
    assert all(e.get('replayable') is True and graph.edge_replayable(e) for e in inputs)
    urls = [e for e in graph.data['edges'] if e['action']['kind'] == 'url_jump']
    assert all(graph.edge_replayable(e) for e in urls)
    ordinary = [e for e in graph.data['edges'] if e not in inputs and e not in urls]
    for edge in ordinary:
        decision = edge.get('replay_review', {}).get('replayable')
        assert isinstance(decision, bool), 'Missing existing Luna decision'
        assert bool(edge['replayable']) == (decision and executable(edge['action'])[0])
    return {'nodes': len(graph.nodes), 'edges': len(graph.data['edges']), 'indexed': len(index['ids']),
        'reachable': len(reachable), 'input_boundaries': len(inputs), 'url_edges': len(urls),
        'ordinary_luna_decisions': len(ordinary), 'status': 'passed', 'graph_changes_needed': False}


def case_plan(graph, audit):
    cases = []
    for row in audit['rows']:
        for nav in row['navigation']:
            if nav['success']:
                continue
            category = row['blocked_edges'][0]['policy'] if row.get('blocked_edges') else 'disconnected'
            cases.append({'id': f"old-{row['task']}-{len(cases)+1:02d}", 'task': row['task'],
                'category': category, 'source': graph.data['root'], 'target': nav['target'],
                'old_result': nav, 'old_events': row['events']})
    for edge in graph.data['edges']:
        if edge['action']['kind'] == 'url_jump':
            cases.append({'id': 'entry-' + edge['id'], 'category': 'entry_url_roundtrip',
                'source': edge['source'], 'target': edge['target']})
    for case in cases:
        path = graph.path(case['source'], case['target'])
        case['registered_path'] = [e['id'] for e in path] if path is not None else None
        case['expected_route_outcome'] = ('excluded' if path is None else 'requires_input'
            if any(e['action']['kind'] in INPUT_KINDS for e in path) else 'arrived')
    return cases


def report(out, results, protocol):
    counts = Counter(r['outcome'] for r in results)
    write_json(out / 'summary.json', {'scope': protocol['scope'], 'planned': len(protocol['cases']),
        'completed': len(results), 'counts': dict(counts), 'model_calls': 0, 'results': results})
    rows = []
    for r in results:
        evidence = r.get('last_observation', {}).get('evidence')
        picture = f'<a href="evidence/{evidence}/page.png">实际页面</a>' if evidence else ''
        rows.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in
            (r['id'], r['category'], r['expected_route_outcome'], r['outcome'],
             r.get('result', {}).get('steps'), r.get('result', {}).get('reason', ''),
             r.get('last_observation', {}).get('url', ''))) + f'<td>{picture} <a href="{r["id"]}/result.json">证据</a></td></tr>')
    (out / 'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Replay checks</title>'
        '<style>body{font:15px system-ui;margin:28px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}td{max-width:420px;overflow-wrap:anywhere}</style>'
        '<h1>Reddit 实际回放检查</h1><p>检查全部 23 个历史失败目标及 6 条入口 URL 边。每次从登记入口的新浏览器上下文开始；'
        '不是原任务中途状态复现，也不是 42 题重测。没有模型调用。输入前暂停、状态失配交接与完整到达分开统计。</p>'
        + '<p>' + html.escape(str(dict(counts))) + '</p><table><tr><th>案例</th><th>原类别</th><th>图预期</th><th>实际</th>'
        '<th>步数</th><th>原因</th><th>实际 URL</th><th>证据</th></tr>' + ''.join(rows) + '</table></html>', encoding='utf-8')


def run(args):
    out, graph = args.out.resolve(), Graph.load(args.graph.resolve())
    if out.exists():
        raise ValueError('Use a new directory; started checks cannot silently rerun')
    audited = graph_audit(graph)
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    cases = case_plan(graph, audit)
    protected = {str(p.resolve()): sha(p) for p in [graph.directory / 'graph.json', graph.directory / 'index.json',
        graph.directory / 'index-whole.json', graph.directory / 'replay-vectors.npz', args.audit,
        ROOT / 'runs/active-graph.json', ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt']}
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'web2kg/stage2/navigation.py',
        ROOT/'web2kg/common/browser.py', ROOT/'web2kg/common/storage.py', ROOT/'web2kg/common/replay_policy.py', ROOT/'web2kg/stage2/transport.py']}
    protocol = {'scope': 'All historical failed teleport targets from fresh registered entry states; route development verification, not task scores.',
        'graph': str(graph.directory), 'graph_audit': audited, 'cases': cases, 'source_hashes': sources,
        'protected_hashes': protected, 'models': [], 'max_case_starts': len(cases), 'retries': 0,
        'container': {'name': NAME, 'image': IMAGE, 'owner_label': 'web2kg.execution_check=' + OWNER},
        'isolation': 'Fresh browser context per case; no input executed, existing ordinary Luna decisions retained; source graph immutable.',
        'evidence': 'Observed controls/headings/URL and PNG retained. No duplicate DOM, raw HAR or API payloads.'}
    write_json(out / 'protocol.json', protocol)
    if not args.live:
        return
    cid = create_container()
    write_json(out / 'container.json', {'id': cid, **protocol['container']})
    headers = {'X-Postmill-Auto-Login': os.getenv('VERIFIED_REDDIT_USER', 'MarvelsGrantMan136') + ':' + os.environ['VERIFIED_REDDIT_PASSWORD']}
    proxy = LocalSiteProxy({'backend': BACKEND, 'allowed_origin': LOGICAL, 'extra_http_headers': headers})
    results = []
    try:
        for case in cases:
            assert all(sha(ROOT/p) == h for p, h in sources.items()), 'Check source changed'
            log = EventLog(out / case['id'])
            write_json(log.directory / 'started.json', {'time': time.time(), 'case': case})
            result = {**case}
            started = time.perf_counter()
            try:
                with CheckBrowser(graph.nodes[case['source']]['url'], log, proxy=proxy,
                        evidence=out/'evidence', viewport=graph.data.get('viewport'),
                        identity_mode=graph.data.get('identity_mode', 'interaction')) as browser:
                    nav = CheckNavigator(browser, graph, log)
                    nav.connect()
                    result['identified_source'] = nav.identify(browser.observe())
                    result['result'] = r = nav.navigate(case['target'])
                    after = browser.observe()
                    result['target_state_matches'] = state_matches(graph.nodes[case['target']], after)
                    result['outcome'] = ('arrived' if r.get('success') and result['target_state_matches'] else
                        'requires_input' if r.get('status') == 'requires_input' else
                        'local_handoff' if r.get('status') == 'requires_local_action' else
                        'excluded' if r.get('reason') in {
                            'target_not_replay_reachable', 'no_recorded_route_from_current_state'} else 'failed')
                    if r.get('status') in {'requires_input', 'requires_local_action'}:
                        acts_before = len(browser.acts)
                        again = nav.navigate(case['target'])
                        result['repeat_guard'] = {'result': again, 'additional_browser_actions': len(browser.acts)-acts_before}
                        assert again['reason'] == 'teleport_requires_local_action' and len(browser.acts) == acts_before
                    result['browser_actions'] = browser.acts
                    assert not any(a['kind'] in INPUT_KINDS for a in browser.acts), 'Historical input executed'
                    result['observations'] = browser.observations
                    result['last_observation'] = browser.observations[-1]
            except Exception as error:
                result.update(outcome='engineering_error', error=str(error))
            result['seconds'] = time.perf_counter()-started
            write_json(log.directory/'result.json', result)
            results.append(result)
            report(out, results, protocol)
            print(json.dumps({k: result.get(k) for k in ('id','outcome','result','seconds')}, ensure_ascii=False), flush=True)
            if result['outcome'] == 'engineering_error':
                break
        unchanged = all(sha(p) == h for p,h in protected.items())
        write_json(out/'closeout.json', {'completed': len(results), 'planned': len(cases), 'protected_unchanged': unchanged,
            'graph_audit': audited, 'models_called': 0, 'model_tokens': 0, 'browser_action_retries': 0,
            'counts': dict(Counter(r['outcome'] for r in results)), 'status': 'completed' if len(results)==len(cases) and unchanged else 'needs_attention'})
    finally:
        proxy.close()
        current_owner = docker('inspect', '--format', '{{index .Config.Labels "web2kg.execution_check"}}', NAME)
        current_id = docker('inspect', '--format', '{{.Id}}', NAME)
        if current_id == cid and current_owner == OWNER:
            docker('stop', NAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    run(parser.parse_args())
