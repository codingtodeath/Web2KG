"""Controlled-latency scheduling experiment; no model, browser, or historical graph changes."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.stage1.pipeline import PipelinedExplorer


class DelayedModel:
    identity = {'model': 'controlled-fixture', 'max_calls': 20}
    max_calls = 20

    def __init__(self, directory, delay):
        self.log, self.delay = EventLog(directory), delay

    def budget_used(self):
        return 0

    def stopped(self):
        return False

    def fork(self, stop_event=None):
        return copy.copy(self)

    def json(self, system, payload, schema, purpose, images=()):
        time.sleep(self.delay)
        self.last_request_id = uuid4().hex
        if purpose == 'candidate_ranking':
            return {'selected': [{'id': a['id'], 'group': a['name'], 'reason': 'Controlled fixture'}
                for a in payload['page']['candidates'][:payload['K']]]}
        return {'replayable': True, 'effect': 'Opened destination', 'uncertainties': [],
            'page_descriptions': {side: 'Page ' + payload[side + '_page']['id']
                                 for side in payload['page_descriptions_requested']}}


def trial(directory, serial, delay):
    directory.mkdir(parents=True)
    graph = Graph(directory)
    graph.data.update(root='root', start_url='http://localhost/', budget={'max_depth': 2, 'actions_per_page': 1})
    action = {'id': 'open', 'kind': 'click', 'role': 'button', 'name': 'Open', 'nth': 0}
    for key in ['root'] + [str(i) for i in range(6)]:
        graph.add_node(dict(id=key, url='http://localhost/' + key, title=key, headings=[key],
                            text='Visible page', actions=[action]), [] if key == 'root' else [action])
    images = [directory / name for name in ('before.png', 'after.png')]
    for path in images:
        Image.new('RGB', (40, 40), 'white').save(path)
    explorer = PipelinedExplorer(DelayedModel(directory, delay), directory)
    started = time.perf_counter()
    try:
        explorer.select(graph, graph.nodes['root'], 1)
        for i in range(6):
            explorer.before_attempt(graph)
            target = str(i)
            edge = graph.add_edge('root', target, action)
            explorer.record(graph, graph.nodes['root'], action, graph.nodes[target], i+1,
                'transition', 'capture_result', True, edge, new_state=True, images=images)
            if i == 0:
                if serial:
                    while explorer.pending:
                        explorer.poll(graph)
                        time.sleep(.001)
                explorer.prefetch(graph, [target], [{'name': 'Another parent action'}])
                if serial:
                    explorer.wait(graph, explorer.rank[1])
            if serial:
                while explorer.pending:
                    explorer.poll(graph)
                    time.sleep(.001)
        explorer.select(graph, graph.nodes['0'], 1)
        explorer.close(graph)
    finally:
        explorer.close(graph, failed=True)
    return {'seconds': time.perf_counter()-started, 'reserved_jobs': explorer.reserved,
        'pages': {k: n['page_description']['text'] for k,n in graph.nodes.items()},
        'edges': [(e['source'], e['target'], e['replayable']) for e in graph.data['edges']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--delay', type=float, default=.15)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for repeat in range(3):
        serial = trial(args.out / str(repeat) / 'serial', True, args.delay)
        parallel = trial(args.out / str(repeat) / 'parallel', False, args.delay)
        assert serial['pages'] == parallel['pages'] and serial['edges'] == parallel['edges']
        assert serial['reserved_jobs'] == parallel['reserved_jobs'] == 8
        rows.append({'repeat': repeat, 'serial_seconds': serial['seconds'], 'parallel_seconds': parallel['seconds'],
                     'same_final_graph_and_page_descriptions': True, 'jobs_per_arm': 8})
    result = {'scope': 'Scheduling only; controlled request latency, no real model/network/browser cost.',
              'delay_seconds': args.delay, 'trials': rows}
    write_json(args.out / 'summary.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
