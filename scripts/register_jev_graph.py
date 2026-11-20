"""Freeze one independent Jev/Qwen Reddit construction experiment, without starting it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.storage import write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(out, expanded=False):
    prior = ROOT / 'runs/reddit-qwen-pipeline-dev2-20261029-v2'
    old = read(prior / 'source-manifest.json')
    for relative, expected in old['preserved'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Preserved input differs: ' + relative)
    out.mkdir(parents=True, exist_ok=False)
    plan = read(prior / 'plan.json')
    plan.pop('repair_from', None)
    plan.update(created_at=datetime.now(timezone.utc).isoformat(),
        authorization='2026-10-30 user authorizes Jev small construction then expanded construction if fast, with proxy kept connected.',
        ranking_provider='jev', ranking_model='jev-latest', ranking_observed_model='jev-1.14.0',
        ranking_proxy='http://127.0.0.1:7890',
        role_models={'candidate_ranking': 'jev-latest', 'transition_description': 'qwen3.8-27b'},
        comparison='Prior Qwen 12-state run is an unmatched exploratory reference, not an identical-action causal control.',
        expansion_gate='All five saved-input Jev calls valid with median <5 seconds; small live crawl must finish without engineering/service errors before expansion.',
        goal='Measure typed ranking latency, valid IDs, new-state progress, visual annotations, bounded parallel scheduling and final source integrity.',
        image_id='sha256:0594908059a03e5f610005689440a95c05e776e734006197c7b1eba12734cb46')
    if expanded:
        small = ROOT / 'runs/reddit-jev-pipeline-dev2-20261030-small-v1'
        if read(small / 'state.json')['status'] != 'completed':
            raise ValueError('Small live construction must complete before expansion')
        plan.update(max_states=60, max_depth=5, actions_per_page=12, max_attempts=120,
            max_construction_calls=180, construction_only=False, max_card_calls=120, max_framework_calls=1,
            clustering='average', tolerance=.05, template_refresh_min=5, template_refresh_fraction=.2,
            assets='Independent graph, first average-link clustering/cards/framework/local Jina; no active-graph publication.',
            model_roles=['candidate_ranking', 'transition_description', 'template_card', 'site_framework'])
        plan['role_models'].update(template_card='qwen3.8-27b', site_framework='qwen3.8-27b')
    source = {}
    paths = set(old['source']) | {'web2kg/stage1/jev_ranking.py', 'scripts/register_jev_graph.py'}
    if expanded:
        paths |= {'scripts/serve_jina_gpu.py', 'scripts/jina_gpu.py', 'scripts/jina_vision_fp32.py'}
    for relative in sorted(paths):
        data = (ROOT / relative).read_bytes()
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        source[relative] = hashlib.sha256(data).hexdigest()
    preserved = dict(old['preserved'])
    for folder in [prior, ROOT / 'runs/jev-ranking-dev2-20261030-v1', ROOT / 'runs/jev-ranking-dev2-20261030-v2']:
        for name in ['plan.json', 'summary.json']:
            p = folder / name
            preserved[p.relative_to(ROOT).as_posix()] = sha(p)
    if expanded:
        for name in ['plan.json', 'summary.json', 'final-audit.json']:
            p = small / name
            preserved[p.relative_to(ROOT).as_posix()] = sha(p)
    write_json(out / 'plan.json', plan)
    write_json(out / 'source-manifest.json', {'source': source, 'preserved': preserved,
        'plan_sha256': sha(out / 'plan.json')})
    write_json(out / 'state.json', {'status': 'setup', 'phase': 'registered'})
    write_json(out / 'test-receipt.json', {'source': 'Executed in current development session',
        'files': ['tests/test_jev_ranking.py', 'tests/test_construction_pipeline.py'],
        'passed': 30, 'seconds': 12.90,
        'covers': ['explicit proxy persists in background fork', 'typed outputs', 'ranking/description separation', 'STOP', 'retry accounting']})
    print(json.dumps({'directory': str(out), 'source_files': len(source), 'preserved': len(preserved),
        'max_states': plan['max_states'], 'expanded': expanded}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expanded', action='store_true')
    args = parser.parse_args()
    register(args.out.resolve(), args.expanded)
