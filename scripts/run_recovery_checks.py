"""Bounded paired development checks; seeded failures are explicitly not natural errors."""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.common.responses import ResponsesClient
from web2kg.common.jina import JinaClient
from web2kg.common.jina_runtime import LocalEncoder
from web2kg.stage1.templates import build_templates
from web2kg.stage1.function_cards import file_hash
from web2kg.stage1.indexing import build_index
from web2kg.stage1.site_framework import build_framework, load_framework
from web2kg.stage2.agent import run_task
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.stage2.recovery import ExperienceStore, memory_namespace, learn_experience
from web2kg.evaluation.webnavigator import OriginalResponses
from web2kg.evaluation.verified import reset_copy
from run_verified_template import source_hash, provider_gate


ITEMS = ['Amber instrument', 'Cobalt instrument', 'Jade instrument']


class CheckSite(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        if self.path == '/':
            body = '<h1>Instrument directory</h1><ul>' + ''.join(f'<li><a href="/item/{i}">{name}</a></li>' for i, name in enumerate(ITEMS)) + '</ul>'
        elif re.fullmatch(r'/item/[0-2]', self.path):
            name = ITEMS[int(self.path[-1])]
            body = f'<h1>{name}</h1><p>This page describes {name}.</p><a href="{self.path}">Refresh view</a>'
        else:
            self.send_error(404); return
        page = '<!doctype html><html><head><title>Instrument archive</title></head><body><nav><a href="/">Directory</a></nav><main>' + body + '</main></body></html>'
        self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8'); self.end_headers()
        self.wfile.write(page.encode())


def fixture_graph(out, url):
    graph = Graph(out / 'graph')
    with Browser(url, EventLog(graph.directory), identity_mode='interaction', settle_ms=20) as browser:
        first = browser.observe(graph.directory)
        graph.data.update(root=first['id'], start_url=url, identity_mode='interaction')
        graph.add_node(first, [])
        for name in ITEMS:
            browser.page.goto(url); browser.settle()
            action = next(a for a in browser.observe()['actions'] if a['name'] == name)
            browser.act(action); after = browser.observe(graph.directory)
            graph.add_node(after, [action]); graph.add_edge(first['id'], after['id'], action)
    graph.save()
    layer = build_templates(graph)
    lp = graph.directory / 'templates/check/templates.json'; write_json(lp, layer)
    write_json(graph.directory / 'templates/active.json', {'directory': 'templates/check'})
    cards = {'source': {'graph_sha256': file_hash(graph.directory / 'graph.json'),
        'layer_sha256': file_hash(lp), 'layer': 'templates/check/templates.json'}, 'cards': {}}
    for group in layer['templates']:
        cards['cards'][group['id']] = {'status': 'completed', 'card': {
            'name': 'Instrument directory' if first['id'] in group['members'] else 'Instrument detail',
            'summary': 'Use the directory to select a named instrument; detail pages describe that exact instrument.',
            'common_functions': ['Inspect visible names'], 'variants': [], 'uncertainties': ['Local synthetic fixture only.']}}
    cp = graph.directory / 'function-cards/check/cards.json'; write_json(cp, cards)
    write_json(graph.directory / 'function-cards/active.json', {'cards': 'function-cards/check/cards.json'})
    build_index(graph)
    return graph, cp


class SeededActor:
    """Both arms receive exactly the same three injected browser actions, then Terra takes over."""
    def __init__(self, client, wrong, log):
        self.client, self.wrong, self.log, self.count = client, wrong, log, 0
        self.model = client.model
    def call(self, messages, **kwargs):
        if 'target_page' not in kwargs.get('system_prompt', '') and self.count < 3:
            name = self.wrong if self.count == 0 else 'Refresh view'
            matches = re.findall(r"link \[(\d+)\] '" + re.escape(name) + "'", json.dumps(messages))
            if not matches:
                raise ValueError('Registered injected control not visible')
            self.count += 1
            self.log.emit('injected_action', action=self.count, control=name, scope='controlled fault, not a model error')
            return f'REASON: Controlled experiment injected action {self.count}.\nACTION: click [{matches[-1]}]'
        return self.client.call(messages, **kwargs)


def run(directory, mode):
    if directory.exists():
        raise ValueError('Choose a fresh experiment directory; no automatic retries')
    directory.mkdir(parents=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    protocol = {'version': 1, 'mode': mode, 'source_commit': commit, 'source_sha256': source_hash(),
        'authorization': '2026-10-13 user requested global framework, root-cause/counterfactual development and multiple experiments',
        'conditions': ['control', 'recovery'], 'actor_model': 'gpt-5.6-terra', 'diagnostic_model': 'gpt-6.1-sol',
        'max_steps': 10 if mode == 'fixture' else 16, 'actor_physical_calls': 14 if mode == 'fixture' else 20,
        'diagnostic_and_learning_calls': 5, 'max_diagnoses': 3, 'recovery_actions': 12,
        'case_ids': list(range(3)) if mode == 'fixture' else ['all_forums', 'wiki_changes'],
        'max_task_starts': 6 if mode == 'fixture' else 4, 'framework_calls': 1 if mode == 'fixture' else 0,
        'transition_calls': 0 if mode == 'fixture' else 4, 'card_calls': 0, 'finalize_graph_assets': False,
        'memory_policy': 'No learned memory fed into either paired arm. Aggregate only after independently successful recoveries.',
        'stopping_rule': 'No retry of started tasks. Stop group on provider failure or runtime infrastructure failure. Per-task model/step caps; no expand-until-positive.',
        'scope': 'Development pipeline check. Fixture faults injected identically before live Terra; Reddit tasks locally authored, not official Hard scores.',
        'execution_budget_seconds': 900, 'deadline_policy': 'Checked between Actor turns/calls; an in-flight call keeps its existing HTTP timeout.'}
    protocol['max_physical_requests'] = protocol['max_task_starts'] * (protocol['actor_physical_calls'] + protocol['transition_calls']) + len(protocol['case_ids']) * 5 + protocol['framework_calls']
    protocol['user_type_prompt_sha256'] = file_hash(ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt')
    write_json(directory / 'protocol.json', protocol)
    server = encoder = None
    try:
        if mode == 'fixture':
            server = ThreadingHTTPServer(('127.0.0.1', 0), CheckSite)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            url = f'http://127.0.0.1:{server.server_port}'
            graph, cards = fixture_graph(directory, url)
            luna = ResponsesClient(ROOT / 'auth/llm-crawl.env', EventLog(directory / 'framework-calls'), 1, 2048, model='gpt-6-luna')
            framework = build_framework(graph, cards, luna, graph.directory / 'site-framework/check', publish=True)
            cases = [(i, f'Open the detail page for {name}, then stop with the instrument name.', url + f'/item/{i}', name) for i, name in enumerate(ITEMS)]
            backend = headers = embedding = None
        else:
            graph = Graph.load(ROOT / 'runs/webnavigator-reddit-llm-120-v4')
            cards = graph.directory / json.loads((graph.directory / 'function-cards/active.json').read_text(encoding='utf-8'))['cards']
            framework = load_framework(graph)
            if framework is None:
                raise ValueError('Generate the registered Reddit framework first')
            url = 'http://127.0.0.1:10499'
            cases = [('all_forums', 'Open the alphabetical directory showing all forums, then stop with its main heading.', url + '/forums/all', 'All forums'),
                     ('wiki_changes', 'Open the wiki recent changes page, then stop with its main heading. Do not edit the wiki.', url + '/wiki/_recent', 'Recent changes')]
            backend = 'http://127.0.0.1:10599'
            headers = {'X-Postmill-Auto-Login': os.getenv('VERIFIED_REDDIT_USER', 'MarvelsGrantMan136') + ':' + os.environ['VERIFIED_REDDIT_PASSWORD']}
            encoder = LocalEncoder(graph, ROOT, 'D:/Anaconda/envs/web2kg-jina-gpu/python.exe')
            encoder.ensure()
            embedding = JinaClient(encoder.index['endpoint'])
        protocol.update(tasks=[{'id': k, 'intent': task, 'checker': {'url': target, 'heading': heading}} for k, task, target, heading in cases],
            graph=str(graph.directory), graph_sha256=file_hash(graph.directory / 'graph.json'), cards_sha256=file_hash(cards),
            framework_source=framework['source'])
        write_json(directory / 'protocol.json', protocol)
        memory = ExperienceStore(directory / 'learned-memory.json', memory_namespace(graph))
        rows = []
        for key, task, target, heading in cases:
            for condition in protocol['conditions']:
                if source_hash() != protocol['source_sha256']:
                    raise ValueError('Source changed during experiment')
                folder = directory / f'{key}-{condition}'
                log = EventLog(folder)
                write_json(folder / 'started.json', {'time': time.time(), 'condition': condition})
                if mode == 'reddit':
                    reset_copy('web2kg-template-eval', 'sha256:0594908059a03e5f610005689440a95c05e776e734006197c7b1eba12734cb46',
                               backend, 10598, url, 'template-agent-hard10-v1')
                actor = OriginalResponses(ROOT / 'auth/llm-crawl.env', log, protocol['actor_physical_calls'], 2048,
                                          model=protocol['actor_model'], verified_protocol=False)
                if mode == 'fixture':
                    actor = SeededActor(actor, ITEMS[(key + 1) % 3], log)
                diagnostic = (ResponsesClient(ROOT / 'auth/llm-crawl.env', EventLog(folder / 'diagnosis-calls'), 5, 2048,
                    model='gpt-6.1-sol') if condition == 'recovery' else None)
                descriptions = (ResponsesClient(ROOT / 'auth/llm-crawl.env', EventLog(folder / 'graph-annotations'), 4, 2048,
                    model='gpt-6-luna') if mode == 'reddit' else None)
                def check(browser):
                    return browser.page.url.rstrip('/') == target.rstrip('/') and heading in browser.page.locator('h1').all_text_contents()
                result = run_task(graph, task, log, actor, embedding, start_url=url, home_url=url,
                    knowledge=TemplateKnowledge(graph, cards, log), backend=backend, headers=headers,
                    max_steps=protocol['max_steps'], update_graph=True, finalize_graph_assets=False,
                    description_client=descriptions, max_descriptions=protocol['transition_calls'],
                    framework=framework, diagnosis_client=diagnostic, recovery_seed=20261013, final_check=check,
                    time_budget_seconds=protocol['execution_budget_seconds'])
                result['page_check'] = result.get('local_success')
                result['local_success'] = (None if result['page_check'] is None else
                    result['page_check'] is True and result['status'] == 'agent_stopped'
                    and heading.casefold() in result.get('answer', '').casefold())
                if diagnostic:
                    result['learning'] = learn_experience(folder, diagnostic, memory, success=result.get('local_success') is True,
                                                          verifier='Independent exact live URL, main heading, stop and answer-label check')
                write_json(folder / 'result.json', result)
                rows.append({'case': key, 'condition': condition, **{k: result.get(k) for k in ('status', 'local_success', 'steps', 'seconds', 'recovery', 'learning')}})
                write_json(directory / 'summary.json', {'protocol': protocol, 'runs': rows})
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
                auxiliary_errors = [json.loads(line) for p in folder.rglob('events.jsonl')
                                    for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
                provider_failed = any(e['event'] == 'responses_failed' and
                    re.search(r'HTTP [45]\d\d|ConnectionError|Timeout', e.get('error', '')) for e in auxiliary_errors)
                if provider_failed or provider_gate(result) or result['status'] in {'runtime_error', 'model_error'}:
                    return
    finally:
        if encoder:
            encoder.close()
        if server:
            server.shutdown(); server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--mode', choices=['fixture', 'reddit'], required=True)
    args = parser.parse_args()
    run(args.directory.resolve(), args.mode)


if __name__ == '__main__':
    main()
