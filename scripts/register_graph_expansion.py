"""Register an independent expansion of a completed graph, retaining all seed evidence."""
import argparse
from datetime import datetime, timezone
import shutil
from pathlib import Path

from check_state_equivalence import ROOT, read, sha
from web2kg.common.storage import write_json


def register(out, prior, target, recover_annotations=False):
    if not out.is_relative_to(ROOT / 'runs') or not prior.is_relative_to(ROOT / 'runs'):
        raise ValueError('Run directories must belong to this workspace')
    if read(prior / 'state.json')['status'] != 'completed' and not (
            recover_annotations and read(prior / 'state.json')['status'] == 'failed'):
        raise ValueError('Expansion requires a completed seed')
    graph = read(prior / 'graph/graph.json')
    if target <= len(graph['nodes']):
        raise ValueError('Expansion target must exceed the seed size')
    old = read(prior / ('manifest.json' if (prior / 'manifest.json').exists() else 'source-manifest.json'))
    for rel, expected in old['source'].items():
        if sha(prior / 'source' / rel) != expected:
            raise ValueError('Seed source changed: ' + rel)
    for rel, expected in old['preserved'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('Seed protected input changed: ' + rel)
    if sha(prior / 'plan.json') != old['plan_sha256']:
        raise ValueError('Seed plan changed')
    plan = read(ROOT / 'runs/reddit-jev-flash-rebuild-20261103-v2/plan.json')
    for key in ('repair', 'repair_from'):
        plan.pop(key, None)
    plan.update(created_at=datetime.now(timezone.utc).isoformat(),
        authorization='2026-11-04 user explicitly requests bounded error retry, continuing to WebNavigator size, and a separate 58-to-60 graph.',
        seed_graph=(prior / 'graph').relative_to(ROOT).as_posix(),
        inherited_nodes=len(graph['nodes']), inherited_edges=len(graph['edges']),
        target_reference={'site': 'reddit', 'states': 225, 'edges': 224,
            'file': 'runs/hard-comparison-retained-20261029/webnavigator-terra/assets/reddit-hard-comparison-v2/webnavigator/released-graph/web_graph.pkl'},
        max_states=target, max_depth=10, actions_per_page=32, max_attempts=1500 if target > 60 else 300,
        max_construction_calls=1200 if target > 60 else 300, max_equivalence_calls=120, max_card_calls=target,
        max_output_retries=3, recover_annotations=recover_annotations,
        stop_policy='JSON/schema output errors regenerate at most 3 times from the same evidence, all physical usage retained. No browser-action retries. Permanent service errors or exhausted output retries stop; started runners never restart.',
        goal=f'Expand a copy to {target} total states; actual edges are not forced to match the baseline.',
        expansion_policy='Retain all seed states/edges/descriptions/rules; re-rank unattempted candidates only. Never repeat inherited attempts. New attempt budget is additional, IDs remain cumulative.',
        gpu_memory='4GiB', encoder_expected=read(prior / 'plan.json')['encoder_expected'],
        cost_policy='New requests only in new role logs; inherited graph costs remain referenced, not duplicated.',
        seed_costs=list(dict.fromkeys(['runs/reddit-jev-flash-rebuild-20261103-v2/summary.json',
            *read(prior / 'plan.json').get('seed_costs', []), prior.relative_to(ROOT).as_posix() + '/summary.json'])),
        assets='Full average-link regroup and all cards/framework on expanded copy, then identity-checked local index checkpoints. Published graph and seed untouched.',
        limitations=['Matching node count is not matching website coverage or state granularity.',
                    'Frontier exhaustion or engineering failure may prevent the target; never fabricate states.',
                    'No evaluation-task inputs, no task-success or generalization claims.'])
    out.mkdir(parents=True, exist_ok=False)
    paths = set(old['source']) | {'scripts/register_graph_expansion.py'}
    sources = {}
    encoder_files = {'scripts/serve_jina_gpu.py', 'scripts/jina_gpu.py', 'scripts/jina_vision_fp32.py'}
    for rel in sorted(paths):
        source = prior / 'source' / rel if rel in encoder_files else ROOT / rel
        dest = out / 'source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        sources[rel] = sha(dest)
    shutil.copytree(prior / 'vector-checkpoints', out / 'vector-checkpoints')
    preserved = dict(old['preserved'])
    for path in prior.rglob('*'):
        if path.is_file():
            preserved[path.relative_to(ROOT).as_posix()] = sha(path)
    reference = ROOT / plan['target_reference']['file']
    preserved[reference.relative_to(ROOT).as_posix()] = sha(reference)
    write_json(out / 'plan.json', plan)
    write_json(out / 'source-manifest.json', dict(source=sources, preserved=preserved,
        plan_sha256=sha(out / 'plan.json')))
    write_json(out / 'state.json', {'status': 'setup', 'phase': 'registered'})
    print({'registered': str(out), 'seed_states': len(graph['nodes']), 'target': target,
           'source_files': len(sources), 'protected_files': len(preserved)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--target', type=int, default=225)
    parser.add_argument('--recover-annotations', action='store_true')
    args = parser.parse_args()
    register(args.out.resolve(), args.prior.resolve(), args.target, args.recover_annotations)
