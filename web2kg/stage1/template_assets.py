"""Shared initial and task-boundary template/card/framework generation."""
import copy
import shutil
import time

from ..common.storage import digest, write_json
from .function_cards import build_cards, card_source, file_hash, export_cards
from .site_framework import build_framework
from .template_global import regroup, reset_maintenance
from .template_reconciliation import apply_graph_templates
from .templates import build_templates


def update_summary(graph, layer):
    groups = layer['templates']
    layer['summary'].update(nodes=len(graph.nodes), templates=len(groups),
        low_evidence_nodes=sum(not p['evidence']['sufficient'] for p in layer['profiles'].values()),
        singleton_templates=sum(len(g['members']) == 1 for g in groups),
        cross_url_templates=sum(len({graph.nodes[k]['url'] for k in g['members']}) > 1 for g in groups),
        local_variants=sum(a['reason'] == 'observed_local_state_variant' for a in layer['assignments'].values()),
        observed_edges=len(graph.data['edges']), template_transitions=len(layer['transitions']))


def stage_assets(graph, layer, card_client, *, framework_client=None, name='global'):
    """Generate everything without changing active pointers or graph annotations."""
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):
        raise ValueError('Use a simple new asset version name')
    folder = graph.directory / 'templates' / name
    folder.mkdir(parents=True, exist_ok=False)
    candidate = copy.deepcopy(layer)
    update_summary(graph, candidate)
    candidate['graph_digest'] = digest(graph.data)
    candidate['graph_sha256'] = file_hash(graph.directory / 'graph.json')
    shutil.copy2(graph.directory / 'graph.json', folder / 'input-graph.json')
    write_json(folder / 'staged-templates.json', candidate, compact=True)
    target = graph.directory / 'function-cards' / name
    started = time.monotonic()
    bundle = build_cards(graph, folder / 'staged-templates.json', target, card_client)
    card_seconds = time.monotonic() - started
    started = time.monotonic()
    framework = build_framework(graph, target / 'cards.json', framework_client or card_client,
                                graph.directory / 'site-framework' / name)
    shutil.copy2(target / 'cards.json', target / 'generated-cards.json')
    return dict(name=name, layer=candidate, cards=bundle, framework=framework,
                timings={'cards_seconds': card_seconds, 'framework_seconds': time.monotonic() - started})


def publish_assets(graph, staged, card_client):
    """Publish after successful model generation; disk errors still propagate."""
    name, layer, bundle, framework = (staged[k] for k in ('name', 'layer', 'cards', 'framework'))
    folder = graph.directory / 'templates' / name
    target = graph.directory / 'function-cards' / name
    apply_graph_templates(graph, layer)
    graph.save()
    layer['graph_digest'] = digest(graph.data)
    layer['graph_sha256'] = file_hash(graph.directory / 'graph.json')
    write_json(folder / 'templates.json', layer, compact=True)
    generation_source = {**copy.deepcopy(bundle['source']), 'graph_file': f'templates/{name}/input-graph.json'}
    for row in bundle['cards'].values():
        row['generation_source'] = copy.deepcopy(generation_source)
    bundle['source'] = card_source(graph, folder / 'templates.json', card_client)
    write_json(target / 'cards.json', bundle)
    export_cards(bundle, target)
    framework['generation_source'] = {**copy.deepcopy(framework['source']),
        'cards': f'function-cards/{name}/generated-cards.json', 'graph_file': f'templates/{name}/input-graph.json'}
    # Keep the actual framework model identity if it differs from the card client.
    framework['source'] = {**bundle['source'], 'model': framework['source']['model'],
        'cards': f'function-cards/{name}/cards.json', 'cards_sha256': file_hash(target / 'cards.json')}
    write_json(graph.directory / f'site-framework/{name}/framework.json', framework)
    write_json(graph.directory / 'templates/active.json', {'directory': f'templates/{name}'})
    write_json(graph.directory / 'function-cards/active.json', {'cards': f'function-cards/{name}/cards.json'})
    write_json(graph.directory / 'site-framework/active.json', {'file': f'site-framework/{name}/framework.json'})
    return staged


def initialize_assets(graph, card_client, *, framework_client=None, method='average', tolerance=.05, name='initial'):
    """The first post-crawl rebuild always clusters, regardless of change-count thresholds."""
    if (graph.directory / 'templates/active.json').exists():
        raise ValueError('Initial build requires a graph without published templates')
    return rebuild_assets(graph, card_client, framework_client=framework_client,
                          method=method, tolerance=tolerance, name=name)


def rebuild_assets(graph, card_client, *, framework_client=None, method='average', tolerance=.05, name='expanded'):
    """Recluster a completed copy; retain old pointers until all generated assets succeed."""
    if not graph.nodes:
        raise ValueError('Cannot summarize an empty graph')
    started = time.monotonic()
    discovery = build_templates(graph)
    discovery_seconds = time.monotonic() - started
    started = time.monotonic()
    layer, audit = regroup(graph, discovery, method=method, tolerance=tolerance)
    audit['timings'] = {'discovery_seconds': discovery_seconds, 'clustering_seconds': time.monotonic() - started}
    reset_maintenance(layer)
    staged = stage_assets(graph, layer, card_client, framework_client=framework_client, name=name)
    audit['timings'].update(staged['timings'])
    write_json(graph.directory / f'templates/{name}/discovery-templates.json', discovery, compact=True)
    publish_assets(graph, staged, card_client)
    audit.update(status='published', initial=name == 'initial')
    write_json(graph.directory / f'templates/{name}/update.json', audit)
    return {**staged, 'audit': audit}
