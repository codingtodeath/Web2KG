"""Register a fresh bounded graph after the user's state-abstraction clarification."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from check_state_equivalence import ROOT, freeze, read, sha
from web2kg.common.storage import write_json
from web2kg.stage1.jev_equivalence import THRESHOLD, RULE


def register(out, repair_from=None):
    if not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh output must be inside workspace runs')
    out.mkdir(parents=True, exist_ok=False)
    plan = read(ROOT / 'runs/reddit-jev-first-exploration-20261102-v1/plan.json')
    plan.update(created_at=datetime.now(timezone.utc).isoformat(),
        authorization='2026-11-03 user explicitly requests graph rebuild after accepting hidden-parameter state abstraction.',
        model='qwen3.7-flash', provider='bailian', ranking_model='jev-1.14.0',
        state_equivalence=True, max_equivalence_calls=30,
        equivalence_threshold=THRESHOLD, equivalence_protocol=RULE,
        max_states=60, max_depth=5, actions_per_page=12, max_attempts=120,
        max_construction_calls=180, construction_only=False,
        max_card_calls=120, max_framework_calls=1, clustering='average', tolerance=.05,
        role_models={'candidate_ranking':'jev-1.14.0','state_equivalence':'jev-1.14.0',
                     'transition_description':'qwen3.7-flash','template_card':'qwen3.7-flash',
                     'site_framework':'qwen3.7-flash'},
        model_roles=['candidate_ranking','state_equivalence','transition_description','template_card','site_framework'],
        goal='Fresh bounded graph with current-control clicks, scoped query equivalence and full independent assets.',
        comparison='Development rebuild, no task-score or isolated causal speed claim.',
        assets='New graph/templates/cards/framework/local index only; never switch the published active graph.',
        state_abstraction='Hidden query variation can share a visible UI state. Live controls supply current parameters; matching does not grant exploration/replay permission.',
        prior_counterexample='Old negative labels/results unchanged. User accepts opaque hidden-transaction example as a UI abstraction boundary, not necessarily a state error. This is a post-result target clarification, not independent accuracy validation.',
        limitations=['Bounded coverage; local equivalence semantics remain model judgments.',
                     'Prior Flash malformed-JSON failure is retained; malformed output stops, never transport-retried.',
                     'No preflight model call; all new model usage belongs to this fresh run.'])
    prior = ROOT / 'runs/state-equivalence-negatives-20261103-v1'
    extra=[prior/'plan.json',prior/'cases.json',prior/'summary.json',prior/'conclusion.json',
           ROOT/'runs/reddit-jev-flash-equivalence-20261103-v1/summary.json']
    if repair_from:
        if read(repair_from/'state.json')['status']!='failed':
            raise ValueError('Repair lineage requires a stopped failed run')
        plan.update(repair_from=str(repair_from.relative_to(ROOT)),
            repair='Bound Jev question batches; aggregate all candidate scores before global selection. Fresh crawl, no inherited nodes/results.',
            ranking_batch_limits={'questions':32,'wire_bytes':60000})
        extra += [repair_from/name for name in ['plan.json','summary.json','state.json','final-audit.json',
                  'failure-diagnosis.json','diagnostic-call.json']]
    freeze(out, plan, extra=extra)
    manifest = read(out/'source-manifest.json')
    for relative in ('scripts/register_equivalence_rebuild.py','scripts/serve_jina_gpu.py',
                     'scripts/jina_gpu.py','scripts/jina_vision_fp32.py'):
        target=out/'source'/relative; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes());manifest['source'][relative]=sha(target)
    write_json(out/'source-manifest.json',manifest)
    write_json(out/'state.json',{'status':'setup','phase':'registered'})
    write_json(out/'test-receipt.json',{'file':'tests/test_state_equivalence.py','passed':25,'seconds':6.65,
        'new_browser_cases':['live link uses new query despite saved action','live POST form uses new query despite saved action'],
        'model_calls':0,'prior_receipt':'43 related offline checks in previous threshold experiment; overlapping, not added.'})
    if repair_from:
        write_json(out/'test-receipt.json',{'files':['tests/test_jev_ranking.py','tests/test_state_equivalence.py','tests/test_construction_pipeline.py'],
            'passed':63,'seconds':26.79,'model_calls':0,
            'covers':['current DOM parameters','threshold boundaries','batch completeness and global ordering',
                      'late failure retains completed parts','parallel pipeline and STOP'],
            'real_failed_payload_offline':read(repair_from/'offline-batch-repair-check.json')})
    print({'directory':str(out),'source_files':len(manifest['source']),'protected_files':len(manifest['preserved'])})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repair-from',type=Path)
    args=parser.parse_args();register(args.out.resolve(),args.repair_from.resolve() if args.repair_from else None)
