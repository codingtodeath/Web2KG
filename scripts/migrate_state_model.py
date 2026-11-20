"""Offline, provenance-preserving migration to the shared interaction state model."""
import argparse
import copy
import json
import shutil
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from web2kg.common.interaction import VERSION, CONTROL_FIELDS, regions_from_ax, state_id, state_matches
from web2kg.common.jina import sha256
from web2kg.common.page_text import fingerprint
from web2kg.common.storage import Graph, digest, write_json
from web2kg.stage1.function_cards import load_layer, inherit_cards, card_source, export_cards, validate_card
from web2kg.stage1.site_framework import load_framework
from web2kg.stage2.graph_update import immutable_copy
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.visualization.viewer import export_viewer


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def remap(value, identities):
    """Rewrite exact ID references, never substrings in text, URLs or provenance paths."""
    if isinstance(value, str):
        return identities.get(value, value)
    if isinstance(value, list):
        return [remap(v, identities) for v in value]
    if isinstance(value, dict):
        keys = [identities.get(k, k) for k in value]
        if len(set(keys)) != len(keys):
            raise ValueError('State equivalence requires explicit asset conflict resolution')
        return {identities.get(k, k): remap(v, identities) for k, v in value.items()}
    return value


def navigation_depths(graph):
    adjacency = {}
    for edge in graph.data['edges']:
        if graph.edge_replayable(edge):
            adjacency.setdefault(edge['source'], []).append(edge['target'])
    depths, queue = {graph.data['root']: 0}, deque([graph.data['root']])
    while queue:
        source = queue.popleft()
        for target in adjacency.get(source, []):
            if target not in depths:
                depths[target] = depths[source] + 1
                queue.append(target)
    return depths


def migrate(base, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Use a new destination; historical evidence is immutable')
    if base.data.get('identity_mode') != 'interaction':
        raise ValueError('Only interaction graphs can be migrated by this command')
    print('Verifying source assets and reconstructing saved regions', flush=True)
    layer_path, layer = load_layer(base)
    framework = load_framework(base)
    cards_path = base.directory / read(base.directory / 'function-cards/active.json')['cards']
    index = read(base.directory / 'index.json')
    if index['mode'] != 'jina_screenshot' or index.get('tiling'):
        raise ValueError('This migration supports the current whole-page screenshot index only')
    if index['fingerprint'] != fingerprint(base):
        raise ValueError('Source index fingerprint mismatch')
    protected = {}

    def protect(path):
        path = Path(path).resolve()
        if not path.is_relative_to(base.directory.resolve()):
            raise ValueError('Asset escaped source graph')
        relative = path.relative_to(base.directory.resolve()).as_posix()
        if relative not in protected:
            protected[relative] = sha256(path)
        return protected[relative]

    for path in (base.directory / 'graph.json', base.directory / 'index.json',
                 base.directory / index['vectors'], layer_path, cards_path,
                 base.directory / 'templates/active.json', base.directory / 'function-cards/active.json'):
        protect(path)
    if protect(base.directory / index['vectors']) != index['vectors_sha256']:
        raise ValueError('Source vector digest mismatch')
    prepared, identities = {}, {}
    for key, saved in base.nodes.items():
        folder = base.directory / 'snapshots' / key
        for path in folder.iterdir():
            if path.is_file():
                protect(path)
        node = copy.deepcopy(saved)
        node['regions'] = regions_from_ax(read(folder / 'ax.json'))
        if saved.get('identity_version') == 'interaction-v1':
            legacy = digest(['interaction-v1', saved['url'], saved.get('headings', []), node['regions'],
                             [{k: a[k] for k in CONTROL_FIELDS if k in a} for a in saved['actions']]])
            if legacy != key:
                raise ValueError(f'Saved AX does not reproduce the original state: {key}')
        identities[key] = state_id(node)
        node.update(id=identities[key], identity_version=VERSION)
        prepared[key] = node
    if len(set(identities.values())) != len(identities):
        # Do not silently pick a screenshot, template, permission or experience on collisions.
        raise ValueError('Equivalent states found; review their dependent assets before merging')
    edge_ids = {e['id']: digest([identities[e['source']], identities[e['target']], e['action']])
                for e in base.data['edges']}
    refs = {**identities, **edge_ids}
    graph = Graph(destination)
    graph.data = remap(base.data, refs)
    graph.data['nodes'] = {identities[k]: remap(n, refs) for k, n in prepared.items()}
    graph.data['execution_update'] = {'source_graph': str(base.directory.resolve()),
        'source_sha256': protect(base.directory / 'graph.json'), 'maintenance': VERSION}
    graph.data['state_model'] = VERSION
    write_json(destination / 'state-id-map.json', {'version': VERSION, 'nodes': identities, 'edges': edge_ids})
    graph.data['state_migration'] = {'mapping': 'state-id-map.json',
        'sha256': sha256(destination / 'state-id-map.json'), 'source': str(base.directory.resolve())}
    depths = navigation_depths(graph)
    for node in graph.nodes.values():
        node['replay_depth'] = depths.get(node['id'])
        # path/depth remain the actual historical discovery route, not a fabricated shorter trace.
    graph.save()
    print('Writing independent observation metadata; reusing immutable screenshots and AX', flush=True)
    for old, new in identities.items():
        source, target = base.directory / 'snapshots' / old, destination / 'snapshots' / new
        target.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file() and path.name not in {'observation.json', 'annotations.json'}:
                immutable_copy(path, target / path.name)
        if (source / 'annotations.json').exists():
            write_json(target / 'annotations.json', remap(read(source / 'annotations.json'), refs))
        observed = remap(read(source / 'observation.json'), refs)
        observed.update(id=new, regions=prepared[old]['regions'], identity_version=VERSION)
        if not state_matches(graph.nodes[new], observed) or state_id(observed) != new:
            raise ValueError('Migrated graph and saved observation disagree')
        observed['migration_source'] = {'file': str(source / 'observation.json'),
                                       'sha256': protect(source / 'observation.json')}
        write_json(target / 'observation.json', observed, compact=True)
    new_layer = remap(layer, refs)
    new_layer['snapshot_sha256'] = {
        '/'.join([identities[relative.split('/')[0]], *relative.split('/')[1:]]): expected
        for relative, expected in layer['snapshot_sha256'].items()}
    new_layer.update(graph_digest=digest(graph.data), graph_sha256=sha256(destination / 'graph.json'))
    new_layer_path = destination / 'templates/state-model/templates.json'
    write_json(new_layer_path, new_layer)
    write_json(destination / 'templates/active.json', {'directory': 'templates/state-model'})

    # First validate historical cards against their actual original evidence.
    validated_dir = destination / 'function-cards/source-evidence'
    original_rows = inherit_cards(base, layer, cards_path, validated_dir)
    card_dir = destination / 'function-cards/inherited'
    rows = {}
    for template, row in original_rows.items():
        original_input = cards_path.parent / 'inputs' / f'{template}.json'
        protect(original_input)
        evidence = remap(read(original_input), refs)
        result = copy.deepcopy(row)
        result['card'] = remap(result['card'], refs)
        result.update(input_hash=digest(evidence), identity_migration={'version': VERSION,
            'source_input': str(original_input), 'source_sha256': protect(original_input),
            'original_input_hash': row['input_hash'], 'new_model_calls': 0})
        validate_card(result['card'], evidence)
        write_json(card_dir / 'inputs' / f'{template}.json', evidence)
        rows[template] = result
    identity = type('Identity', (), {'model': read(cards_path)['source']['model']})()
    cards = {'source': card_source(graph, new_layer_path, identity), 'cards': rows}
    write_json(card_dir / 'cards.json', cards)
    export_cards(cards, card_dir)
    write_json(destination / 'function-cards/active.json', {'cards': 'function-cards/inherited/cards.json'})
    for relative in ('initial-website-framework.json', 'experiences/memory.json'):
        path = base.directory / relative
        if path.exists():
            protect(path)
            value = read(path)
            if relative.startswith('experiences/'):
                value = remap(value, identities)
            write_json(destination / relative, value)
    if framework is not None:
        write_json(destination / 'initial-website-framework.json', framework)
        pointer = base.directory / 'site-framework/active.json'
        if pointer.exists():
            protect(pointer)
            protect(base.directory / read(pointer)['file'])

    print('Rekeying and checking saved vectors without encoding', flush=True)
    reachable = graph.reachable()
    kept = [k for k in index['ids'] if identities[k] in graph.nodes]
    for key in index['ids']:
        if protect(base.directory / 'snapshots' / key / 'page.png') != index['images_sha256'][key]:
            raise ValueError('Indexed screenshot changed')
    with np.load(base.directory / index['vectors'], allow_pickle=False) as archive:
        if set(archive.files) != set(index['ids']):
            raise ValueError('Unexpected source vector keys')
        np.savez(destination / 'state-vectors.npz', **{identities[k]: archive[k] for k in kept})
    new_index = copy.deepcopy(index)
    positions = {key: i for i, key in enumerate(index['ids'])}
    new_index.update(ids=[identities[k] for k in kept], texts=[index['texts'][positions[k]] for k in kept],
        node_scope='all_graph_nodes' if len(kept) == len(graph.nodes) else 'retained_source_pages',
        fingerprint=fingerprint(graph), vectors='state-vectors.npz',
        vectors_sha256=sha256(destination / 'state-vectors.npz'),
        images_sha256={identities[k]: index['images_sha256'][k] for k in kept},
        identity_migration={'source_index_sha256': protect(base.directory / 'index.json'),
                            'mapping': 'state-id-map.json', 'encoded': 0})
    # Old incremental/pruning records describe the historical source, not current node IDs.
    for key in ('incremental', 'replay_pruning'):
        new_index.pop(key, None)
    write_json(destination / 'index.json', new_index)
    write_json(destination / 'index-whole.json', new_index)
    with np.load(base.directory / index['vectors'], allow_pickle=False) as source, np.load(
            destination / 'state-vectors.npz', allow_pickle=False) as migrated:
        assert all(np.array_equal(source[k], migrated[identities[k]]) for k in kept)
    assert {identities[k] for k in base.reachable()} == set(reachable)
    for original, migrated in zip(base.data['edges'], graph.data['edges']):
        assert base.edge_replayable(original) == graph.edge_replayable(migrated)
    assert all(state_id(n) == k for k, n in graph.nodes.items())
    load_layer(graph)
    load_framework(graph)
    TemplateKnowledge(graph, card_dir / 'cards.json', None)
    export_viewer(graph)
    print('Verifying protected originals after migration', flush=True)
    assert all(sha256(base.directory / relative) == expected for relative, expected in protected.items())
    report = {'status': 'completed', 'version': VERSION, 'source': str(base.directory.resolve()),
        'source_sha256': protect(base.directory / 'graph.json'), 'nodes': len(graph.nodes),
        'edges': len(graph.data['edges']), 'merged': 0, 'reachable': len(reachable), 'indexed': len(kept),
        'cards': len(rows), 'entries': len(graph.data.get('url_entries', {})),
        'max_replay_depth': max(depths.values()), 'max_discovery_depth': max(n['depth'] for n in graph.nodes.values()),
        'new_model_calls': 0, 'reencoded': 0, 'protected_files': len(protected),
        'checks': ['state_id_equals_replay_signature', 'snapshot_identity', 'template_and_card_sources',
                   'framework_ancestry', 'same_reachability_and_permissions', 'exact_vector_arrays', 'source_hashes']}
    write_json(destination / 'source-manifest.json', protected)
    write_json(destination / 'maintenance.json', report)
    return graph, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    graph, report = migrate(Graph.load(args.graph.resolve()), args.out)
    if args.publish:
        write_json(ROOT / 'runs/active-graph.json', {'graph': str(graph.directory), 'source': report['source'],
                                                 'state_model': VERSION})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
