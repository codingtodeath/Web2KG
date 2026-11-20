"""Offline entry/type policy upgrade, preserving source evidence and saved vectors."""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from update_replay_graph import prune_index
from web2kg.common.jina import sha256
from web2kg.common.page_text import fingerprint
from web2kg.common.replay_policy import executable, mark_input_boundary
from web2kg.common.storage import Graph, digest, write_json
from web2kg.stage1.function_cards import load_layer, inherit_cards, card_source, export_cards
from web2kg.stage1.site_framework import load_framework
from web2kg.stage2.graph_update import immutable_copy
from web2kg.visualization.viewer import export_viewer


def upgrade(base, vectors, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Choose a new destination; never overwrite an existing graph')
    original_hash = sha256(base.directory / 'graph.json')
    vector_hash = sha256(vectors.directory / 'graph.json')
    print('Verifying saved template and vector sources', flush=True)
    _, layer = load_layer(base)
    index = json.loads((vectors.directory / 'index.json').read_text(encoding='utf-8'))
    if fingerprint(base) != fingerprint(vectors) or index['fingerprint'] != fingerprint(base):
        raise ValueError('Vector source does not describe the same node contents')
    old_index = json.loads((base.directory / 'index.json').read_text(encoding='utf-8'))
    if index['encoder'] != old_index['encoder']:
        raise ValueError('Encoder identities differ')
    for node, expected in index['images_sha256'].items():
        if (sha256(base.directory / 'snapshots' / node / 'page.png') != expected
                or sha256(vectors.directory / 'snapshots' / node / 'page.png') != expected):
            raise ValueError('Snapshot content differs from saved vector source')
    graph = Graph(destination)
    graph.data = copy.deepcopy(base.data)
    graph.data['execution_update'] = {'source_graph': str(base.directory), 'source_sha256': original_hash,
                                     'maintenance': 'entry-url-and-actor-input-boundaries-v1'}
    before_edges = len(graph.data['edges'])
    for edge in graph.data['edges']:
        decision = edge.get('replay_review') or edge.get('description', {}).get('annotation', {})
        if type(decision.get('replayable')) is bool:
            edge.update(replayable=decision['replayable'] and executable(edge['action'])[0],
                        replay_policy='luna-v1', replay_reason=decision.get('reason', decision.get('replay_reason', '')))
        else:
            edge.update(replayable=False, replay_policy='luna-v1', replay_reason='missing_luna_decision')
        mark_input_boundary(edge)
        if edge.get('description'):
            edge['description'].update({k: edge[k] for k in ('replayable', 'replay_policy', 'replay_reason')})
    graph.invalidate_reachability()
    graph.initialize_url_entries()
    provenance = entry_provenance(base)
    for node, proof in provenance.items():
        graph.register_url_entry(node, proof)
    graph.save()
    shutil.copytree(base.directory / 'snapshots', graph.directory / 'snapshots', copy_function=immutable_copy)
    print('Reusing saved vectors for retained nodes', flush=True)
    vector_result = prune_index(vectors, graph)
    layer = copy.deepcopy(layer)
    layer.update(graph_digest=digest(graph.data), graph_sha256=sha256(graph.directory / 'graph.json'))
    layer_path = graph.directory / 'templates/navigation/templates.json'
    write_json(layer_path, layer)
    write_json(graph.directory / 'templates/active.json', {'directory': 'templates/navigation'})
    old_cards = json.loads((base.directory / 'function-cards/active.json').read_text(encoding='utf-8'))['cards']
    card_dir = graph.directory / 'function-cards/inherited'
    rows = inherit_cards(graph, layer, base.directory / old_cards, card_dir)
    identity = type('Identity', (), {'identity': {'model': 'gpt-6-luna'}, 'model': 'gpt-6-luna'})()
    card_data = {'source': card_source(graph, layer_path, identity), 'cards': rows}
    write_json(card_dir / 'cards.json', card_data)
    export_cards(card_data, card_dir)
    write_json(graph.directory / 'function-cards/active.json', {'cards': 'function-cards/inherited/cards.json'})
    for relative in ('initial-website-framework.json', 'experiences/memory.json'):
        source = base.directory / relative
        if source.exists():
            target = graph.directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    export_viewer(graph)
    print('Checking final graph, templates and framework ancestry', flush=True)
    load_layer(graph)
    load_framework(graph)
    if sha256(base.directory / 'graph.json') != original_hash or sha256(vectors.directory / 'graph.json') != vector_hash:
        raise ValueError('Original graph changed')
    report = {'status': 'completed', 'source': str(base.directory), 'source_sha256': original_hash,
        'vector_source': str(vectors.directory), 'vector_source_sha256': vector_hash,
        'nodes': len(graph.nodes), 'edges': len(graph.data['edges']), 'url_edges': len(graph.data['edges']) - before_edges,
        'entries': graph.data['url_entries'], 'reachable': len(graph.reachable()),
        'input_boundaries': sum(e['action'].get('kind') in {'type', 'fill'} for e in graph.data['edges']),
        'index': vector_result, 'cards_inherited': len(rows), 'new_model_calls': 0,
        'runtime_validation': 'Entry URL edges are registered routes; every execution must verify actual arrival.'}
    write_json(graph.directory / 'maintenance.json', report)
    write_json(graph.directory / 'entry-provenance.json', provenance)
    return graph, report


def entry_provenance(graph):
    """Use recorded task-start fields in the hash-checked ancestry, never arbitrary islands."""
    incoming = {e['target'] for e in graph.data['edges'] if e['action'].get('kind') != 'url_jump'}
    candidates = {k for k, n in graph.nodes.items() if not n.get('path') and k not in incoming}
    candidates.add(graph.data['root'])
    found, visited, ancestor = {}, set(), graph
    while candidates - found.keys():
        path = ancestor.directory.resolve()
        if path in visited:
            raise ValueError('Cyclic entry ancestry')
        visited.add(path)
        for key in (ancestor.data['root'], ancestor.data.get('task_start')):
            if key in candidates and key not in found:
                found[key] = {'graph': str(path), 'sha256': sha256(path / 'graph.json'),
                              'field': 'root' if key == ancestor.data['root'] else 'task_start'}
        update = ancestor.data.get('execution_update', {})
        if not update.get('source_graph'):
            break
        ancestor = Graph.load(update['source_graph'])
        if sha256(ancestor.directory / 'graph.json') != update['source_sha256']:
            raise ValueError('Entry ancestry source changed')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--vectors-from', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    graph, report = upgrade(Graph.load(args.graph.resolve()), Graph.load(args.vectors_from.resolve()), args.out)
    if args.publish:
        write_json(ROOT / 'runs/active-graph.json', {'graph': str(graph.directory), 'source': report['source']})
    print(json.dumps({k: v for k, v in report.items() if k not in {'entries', 'index'}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
