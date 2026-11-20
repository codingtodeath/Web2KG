"""No-model checks of the six recorded Reddit click failures; never reset or submit data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog, write_json
from web2kg.common.interaction import control_signature


def run(out, storage):
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *[ROOT / ('web2kg/' + p) for p in [
        'common/browser.py','common/replay.py','common/interaction.py','stage1/crawl.py','stage1/llm_exploration.py']]]
    for path in sources:
        target = out / 'source' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    protected = [ROOT / 'runs/active-graph.json', ROOT / 'runs/reddit-hard-state-v2-20261023/graph.json',
                 ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt']
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources + protected}
    write_json(out / 'plan.json', dict(source_and_inputs=hashes, models=0, site_reset=False,
        controls=['Comments','Formatting help +','Image','URL','Jump to sidebar','Jump to main content'],
        policy='Fresh observations, one attempt each, no posting/subscription/logout, no runtime retries.'))
    result = dict(cases=[], model_calls=0, website_reset=False)
    def browser(path, name):
        return Browser('http://localhost:10599/' + path, EventLog(out / name), storage_state=str(storage),
                       identity_mode='interaction', viewport={'width':1280,'height':720})
    with browser('', 'comments') as b:
        a = next(a for a in b.observe()['actions'] if a['name'] == 'Comments')
        started = time.monotonic()
        try:
            b.act(a)
            result['cases'].append(dict(control='Comments',passed=b.page.url.endswith('/comments'),
                                       seconds=time.monotonic()-started))
        except Exception as error:
            result['cases'].append(dict(control='Comments',passed=False,error_type=type(error).__name__,
                                       seconds=time.monotonic()-started))
    with browser('submit', 'form') as b:
        for name in ['Formatting help +','Image','URL']:
            a = next(a for a in b.observe()['actions'] if a['name'] == name)
            # The label text can change from '+' to '-' after expansion.
            target = b.page.get_by_role(a['role'],name=name,exact=True).element_handle()
            checked = target.is_checked()
            started = time.monotonic()
            try:
                b.act(a)
                result['cases'].append(dict(control=name,passed=target.is_checked() != checked,
                                           seconds=time.monotonic()-started))
            except Exception as error:
                result['cases'].append(dict(control=name,passed=False,error_type=type(error).__name__,
                                           seconds=time.monotonic()-started))
            finally:
                target.dispose()
    with browser('wiki', 'skip-links') as b:
        for name in ['Jump to sidebar','Jump to main content']:
            a = next(a for a in b.observe()['actions'] if a['name'] == name)
            started = time.monotonic()
            try:
                b.act(a)
                result['cases'].append(dict(control=name,passed=False,reason='unexpected_pointer_dispatch'))
            except ValueError as error:
                result['cases'].append(dict(control=name,passed=bool(a.get('pointer_unavailable')),
                    filtered_before_ranking=bool(a.get('pointer_unavailable')), reason=type(error).__name__,
                    seconds=time.monotonic()-started))
    with browser('', 'menu') as b:
        observations=[]
        for i in range(2):
            if i: b.reset()
            a=next(a for a in b.observe()['actions'] if a['role']=='button' and a['name']=='MarvelsGrantMan136')
            b.act(a);observations.append(b.observe())
        left,right=observations
        old={a['id']:control_signature(a) for a in left['actions']}
        new={a['id']:control_signature(a) for a in right['actions']}
        diff=[dict(control=old[k]['name'],fields=[f for f in old[k].keys()|new[k].keys()
                    if old[k].get(f)!=new[k].get(f)]) for k in old.keys()&new.keys() if old[k]!=new[k]]
        result['menu']=dict(same_identity=left['id']==right['id'],changed_control_fields=diff,
            ordinary_token_policy_unchanged=True,
            scope='Only field names exported. Repeated identical source restoration is stopped; underlying identity mismatch remains.')
    result['all_click_checks_passed']=all(r['passed'] for r in result['cases'])
    result['source_and_inputs_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    write_json(out / 'summary.json', result)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--storage',type=Path,required=True)
    args=parser.parse_args()
    run(args.out.resolve(),args.storage.resolve())
