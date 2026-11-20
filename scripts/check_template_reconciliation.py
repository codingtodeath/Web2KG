"""Offline timing and preview of per-task template consolidation; never publishes or calls models."""
import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web2kg.common.storage import Graph, write_json
from web2kg.stage1.function_cards import file_hash, load_layer
from web2kg.stage1.templates import adopt_matching_policy, refresh_representatives
from web2kg.stage1.template_reconciliation import reconcile


def run(directory, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; preserve earlier previews')
    start = time.perf_counter()
    graph = Graph.load(directory)
    path, layer = load_layer(graph)
    loaded = time.perf_counter() - start
    protected = {p: file_hash(p) for p in (graph.directory / 'graph.json', path,
                 graph.directory / 'templates/active.json')}
    for p in (Path('runs/active-graph.json'), graph.directory / 'function-cards/active.json'):
        if p.exists():
            protected[p] = file_hash(p)
    prepared = copy.deepcopy(layer)
    adopt_matching_policy(prepared)
    start = time.perf_counter()
    promotions = refresh_representatives(prepared)
    representative_seconds = time.perf_counter() - start
    cold, warm = [], []
    for _ in range(3):
        result, audit = reconcile(prepared)
        _, cached = reconcile(result)
        cold.append(audit['seconds'])
        warm.append(cached['seconds'])
    assert all(file_hash(p) == value for p, value in protected.items())
    report = {'source': str(graph.directory.resolve()), 'source_files_unchanged': True,
              'source_sha256': {str(p.resolve()): v for p, v in protected.items()},
              'load_and_validation_seconds': loaded, 'representative_seconds': representative_seconds,
              'representative_changes': promotions, 'cold_seconds': cold, 'warm_seconds': warm,
              'cold_median_seconds': statistics.median(cold), 'warm_median_seconds': statistics.median(warm),
              'audit': audit, 'cached_audit': cached, 'model_calls': 0, 'published': False,
              'scope': 'Local structure comparison and in-memory regrouping only; excludes card generation and asset export.'}
    write_json(output / 'audit.json', report)
    write_json(output / 'preview-layer.json', result, compact=True)
    print(json.dumps({k: report[k] for k in ('cold_median_seconds', 'warm_median_seconds',
          'representative_seconds', 'load_and_validation_seconds', 'model_calls', 'published')}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.graph, args.out)
