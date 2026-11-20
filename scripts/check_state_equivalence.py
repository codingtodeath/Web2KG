"""Fresh small-difference check and independently frozen Qwen Flash construction registration."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.browser import Browser
from web2kg.common.interaction import state_id, state_matches, VERSION
from web2kg.common.state_equivalence import StateResolver, RULE, small_difference
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.common.replay import verify
from web2kg.stage1.jev_equivalence import JevEquivalenceClient
from web2kg.common.activity import read_events


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def freeze(out, plan, extra=()):
    prior = ROOT / 'runs/reddit-jev-first-exploration-20261102-v1'
    paths = set(read(prior / 'source-manifest.json')['source']) | {
        'web2kg/common/state_equivalence.py', 'web2kg/stage1/jev_equivalence.py',
        'scripts/check_state_equivalence.py'}
    sources = {}
    for relative in sorted(paths):
        source = ROOT / relative
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sources[relative] = sha(target)
    protected = [ROOT / 'runs/active-graph.json',
        ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt',
        prior / 'summary.json', prior / 'state.json', *extra]
    protected += [p for p in (ROOT / 'runs/reddit-hard-state-v2-20261023').rglob('*') if p.is_file()]
    write_json(out / 'plan.json', plan)
    write_json(out / 'source-manifest.json', {'source': sources,
        'preserved': {p.relative_to(ROOT).as_posix():sha(p) for p in protected},
        'plan_sha256': sha(out / 'plan.json')})


def check(out, storage):
    out.mkdir(parents=True, exist_ok=False)
    freeze(out, {'authorization':'2026-11-03 user requests Jev scoped state-equivalence experiment.',
        'rule': RULE, 'model':'jev-1.14.0', 'proxy':'http://127.0.0.1:7890', 'max_calls':6,
        'positive':'Actual account-menu request parameter rotation across fresh browser contexts.',
        'negative':['object query parameter', 'ambiguous token in capability link',
                    'menu expansion', 'subscription label', 'changed page URL'],
        'predeclared_gate':'Actual menu accepted and reused; no negative accepted; no extra calls on reuse.',
        'website_reset':False, 'qwen_calls':0})
    judge = JevEquivalenceClient(ROOT / '.env', EventLog(out / 'equivalence-calls'), 6,
        model='jev-1.14.0', proxy='http://127.0.0.1:7890', stop_file=out / 'STOP')
    graph = Graph(out / 'graph'); log = EventLog(graph.directory)
    started = time.monotonic(); results=[]
    try:
        with Browser('http://localhost:10599/', log, storage_state=str(storage), identity_mode='interaction',
                     viewport={'width':1280,'height':720}) as browser:
            root = browser.observe(graph.directory)
            action = next(a for a in root['actions'] if a['role']=='button' and a['name']=='MarvelsGrantMan136')
            graph.data.update(root=root['id'], start_url=root['url'])
            graph.add_node(root, [])
            browser.act(action)
            saved = browser.observe(graph.directory)
            graph.add_node(saved, [action]); graph.save()
            resolver = StateResolver(graph, judge, log=log)
            browser.state_resolver = resolver
            for visit in range(1, 4):
                browser.reset(); browser.act(action)
                observed = browser.observe()
                strict = state_matches(saved, observed)
                tick = time.monotonic()
                try:
                    resolved = verify(browser, saved['id'], saved)
                except ValueError:
                    resolved = observed
                results.append({'case':'actual_menu_visit_' + str(visit), 'strict_equal':strict,
                    'passed':resolved['id']==saved['id'], 'raw_state_id':state_id(resolved),
                    'calls_so_far':judge.budget_used(), 'seconds':time.monotonic()-tick})
                if visit == 1:
                    learned = deepcopy(graph.data['state_equivalence'])
                    # Subsequent fresh visits test persistence, with no judge available.
                    browser.state_resolver = StateResolver(Graph.load(graph.directory), log=log)
            results[0]['one_learned_rule'] = len(learned['rules']) == 1
            results[0]['passed'] = results[0]['passed'] and len(learned['rules']) == 1 and judge.budget_used() == 1
            # Reuse must not ignore any field outside the learned query slot.
            for name in ('menu_expansion','subscription_label','changed_page_url'):
                changed = deepcopy(saved)
                if name=='menu_expansion': changed['actions'][0]['state']={'expanded':True}
                elif name=='subscription_label': changed['actions'][0]['name']='Unsubscribe'
                else: changed['url'] += 'different'
                changed['id'] = state_id(changed)
                result = resolver.resolve(changed, learn=True)
                results.append({'case':name,'passed':result['id']!=saved['id'],
                                'model_required':small_difference(saved,changed) is not None})
        for name, control, url in [('object_query','Open post','/post?post_id='),
                                    ('ambiguous_token','Open shared item','/share?token=')]:
            g = Graph(out / name)
            a = dict(identity_version=VERSION,url='http://fixture.test/',text='Choose item',headings=['Items'],
                     regions=[],actions=[dict(role='link',name=control,href=url+'first-value')])
            a['id']=state_id(a);g.nodes[a['id']]=a
            b=deepcopy(a);b['actions'][0]['href']=url+'second-value';b['id']=state_id(b)
            r=StateResolver(g,judge,log=log)
            result=r.resolve(b,learn=True)
            results.append({'case':name,'passed':result['id']!=a['id'],
                            'judgment':g.data['state_equivalence']['decisions'][-1]['judgment']})
        status='completed'
    except Exception as error:
        status='failed'
        write_json(out/'failure.json', {'error_type':type(error).__name__})
        raise
    finally:
        events=read_events(out/'equivalence-calls/events.jsonl')
        receipts=[{k:e.get(k) for k in ('actual_model','usage','seconds')}
                  for e in events if e['event']=='responses_received']
        write_json(out/'summary.json',{'status':locals().get('status','failed'),'cases':results,
            'passed':len(results)==8 and all(r['passed'] for r in results),
            'physical_calls':sum(e['event']=='responses_started' for e in events),
            'responses':receipts,'wall_seconds':time.monotonic()-started})
        from build_qwen_graph import verify_inputs
        verify_inputs(out,ROOT)
    print(json.dumps({'directory':str(out),'status':status,'passed':all(r['passed'] for r in results)}))


def register(out, check_dir):
    precheck = read(check_dir/'summary.json')
    negatives = [r for r in precheck['cases'] if not r['case'].startswith('actual_menu_')]
    if precheck['status'] != 'completed' or len(negatives) != 5 or not all(r['passed'] for r in negatives):
        raise ValueError('State equivalence negative controls did not pass')
    out.mkdir(parents=True, exist_ok=False)
    plan=read(ROOT/'runs/reddit-jev-first-exploration-20261102-v1/plan.json')
    plan.update(created_at=datetime.now(timezone.utc).isoformat(),
        authorization='2026-11-03 user requests learned query-slot state equivalence and Qwen3.7-flash experiment.',
        model='qwen3.7-flash', provider='bailian',ranking_model='jev-1.14.0',state_equivalence=True,
        max_equivalence_calls=20,max_states=30,max_attempts=60,max_construction_calls=90,
        role_models={'candidate_ranking':'jev-1.14.0','state_equivalence':'jev-1.14.0',
                     'transition_description':'qwen3.7-flash'},
        model_roles=['candidate_ranking','state_equivalence','transition_description'],
        goal='Validate graph-local equivalence reuse, no duplicate token states and actual Flash visual descriptions.',
        comparison='Joint development change: browser repairs, equivalence and model differ; no isolated causal timing claim.',
        assets='Independent construction-only graph; no published graph/index/card changes.',
        equivalence_precheck=str(check_dir.relative_to(ROOT)),
        equivalence_precheck_passed=precheck['passed'],
        limitation='The real menu positive did not meet the unchanged .90 threshold. This small Flash run tests conservative fallback and integration only; it is not promotion to a full graph.')
    freeze(out,plan,extra=[check_dir/'plan.json',check_dir/'summary.json'])
    write_json(out/'state.json',{'status':'setup','phase':'registered'})
    print(json.dumps({'directory':str(out),'model':plan['model'],'max_states':plan['max_states']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['check','register'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--storage',type=Path)
    parser.add_argument('--check-dir',type=Path)
    args=parser.parse_args()
    if args.mode=='check': check(args.out.resolve(),args.storage.resolve())
    else: register(args.out.resolve(),args.check_dir.resolve())
