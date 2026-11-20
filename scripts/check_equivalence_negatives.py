"""Prelabel query-only counterexamples, then test scoped Jev acceptance once per case."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_state_equivalence import freeze, read, sha
from build_qwen_graph import verify_inputs
from web2kg.common.activity import read_events
from web2kg.common.interaction import state_id, VERSION
from web2kg.common.state_equivalence import StateResolver, small_difference, judgment_payload
from web2kg.common.storage import Graph, EventLog, write_json
from web2kg.stage1.jev_equivalence import JevEquivalenceClient, THRESHOLD


def fixtures():
    # Labels/latent semantics are saved separately and never included in model payloads.
    specs = [
        ('record_id','Records','Recent records. Open the selected record.', 'Open record',
         '/records/view?id=record01', '/records/view?id=record02', False,
         'different','Different record identifiers despite identical generic link text.'),
        ('page_cursor','Activity','Activity stream. Load the next page of results.', 'Next',
         '/activity?cursor=cursor01', '/activity?cursor=cursor02', False,
         'different','Different cursor selects a different result window.'),
        ('sort','Discussions','Browse discussions. Change the ordering.', 'Sort results',
         '/discussions?sort=newest', '/discussions?sort=oldest', False,
         'different','Ordering controls which results appear next.'),
        ('filter','Issues','Issue tracker. View the selected issue list.', 'View issues',
         '/issues?status=open', '/issues?status=done', False,
         'different','Different issue-status filter.'),
        ('tenant','Workspace','Workspace dashboard. Switch workspace.', 'Open workspace',
         '/workspace?tenant=tenant01', '/workspace?tenant=tenant02', False,
         'different','Workspace scope changes; visible snapshot alone cannot equate tenants.'),
        ('account','Account menu','Account menu. Profile Settings Sign out', 'Open profile',
         '/profile?account=person01', '/profile?account=person02', False,
         'different','Profile link targets another account.'),
        ('revision','Document','Document history. Open a saved revision.', 'Open revision',
         '/document?revision=version1', '/document?revision=version2', False,
         'different','Different historical document revisions.'),
        ('invite','Invitations','Workspace invitations. Accept the invitation to join a workspace.', 'Accept invitation',
         '/invite/accept?token=invite01', '/invite/accept?token=invite02', True,
         'different','Invitation capabilities refer to different invitations.'),
        ('password_reset','Account recovery','Choose a new password for your account. New password Confirm password', 'Save password',
         '/password/reset?token=recover1', '/password/reset?token=recover2', True,
         'different','Recovery capabilities identify different recovery authorizations.'),
        ('file_capability','Files','Shared files. Open a file using its access link.', 'Open shared file',
         '/files/access?key=access01', '/files/access?key=access02', False,
         'different','Access capability selects a different shared file or permission scope.'),
        ('delete_target','Record actions','Manage a record. Deleting a record cannot be undone.', 'Delete record',
         '/record/delete?target=record01', '/record/delete?target=record02', True,
         'different','Different deletion targets; no action is actually executed.'),
        ('logout_session','Account menu','Account menu. Manage active sessions. Sign out this session.', 'Sign out',
         '/logout?session=session1', '/logout?session=session2', True,
         'different','A named session identifies a different logout target, not a request nonce.'),
        ('opaque_csrf_name','Approval','Approval request. Review details and confirm.', 'Confirm',
         '/approval/confirm?csrf_nonce=approve1', '/approval/confirm?csrf_nonce=approve2', True,
         'insufficient_evidence','Fixture backend uses this misleadingly named value to select an approval transaction. Name alone cannot establish equivalence.'),
        ('opaque_logout','Account menu','Account menu. Profile Settings Sign out', 'Sign out',
         '/logout?token=opaque01', '/logout?token=opaque02', True,
         'insufficient_evidence','Adversarial fixture token selects the session to revoke. Visible context does not distinguish it from validation rotation.'),
        ('validation_nonce','Preferences','Preferences form. A one-time request-validation nonce is refreshed whenever this form is rendered. Theme Notifications Save preferences', 'Save preferences',
         '/preferences/save?nonce=nonce001', '/preferences/save?nonce=nonce002', True,
         'same','Fixture nonce only validates requests; it never identifies a user, resource, or business transaction.'),
        ('logout_validation','Account menu','Profile Settings Sign out. The sign-out form refreshes its request-validation token on each display; the signed-in account and current session are unchanged.', 'Sign out',
         '/logout?token=nonce001', '/logout?token=nonce002', True,
         'same','Explicit synthetic validation-token rotation; no browser action performed.'),
    ]
    cases=[]
    for name, title, text, label, before_url, after_url, form, expected, rationale in specs:
        action = {'role':'button' if form else 'link','name':label}
        field = 'form_action' if form else 'href'
        action[field]=before_url
        if form: action.update(method='POST',type='submit')
        a={'identity_version':VERSION,'url':'https://fixture.test/view',
           'text':text,'headings':[title],'regions':[],
           'actions':[{'role':'link','name':'Home','href':'/'},
                      {'role':'link','name':'Help','href':'/help'}, action]}
        a['id']=state_id(a)
        b=deepcopy(a); b['actions'][-1][field]=after_url; b['id']=state_id(b)
        slots=small_difference(a,b)
        assert slots is not None and a['id'] != b['id'], name
        cases.append({'id':name,'expected':expected,'rationale':rationale,
                      'before':a,'after':b,'payload':judgment_payload(a,b,slots)})
    return cases


def render(out, summary):
    rows=''.join('<tr>'+''.join('<td>'+escape(str(row.get(k,'')))+'</td>' for k in
        ('case','expected','score','decision','merged','rule_reused','other_change_rejected','seconds'))+'</tr>'
        for row in summary['cases'])
    html='''<!doctype html><meta charset="utf-8"><title>Jev 0.80 局部状态等价反例实验</title>
<style>body{max-width:1200px;margin:35px auto;font:16px/1.6 system-ui;padding:0 20px}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccc;padding:7px}h1{font-size:26px}</style>
<h1>Jev 0.80 局部状态等价反例实验</h1>
<p>冻结标签后，每例仅调用一次 Jev；使用完整已存上下文＋精准差异，不改变提示词与前置筛选。14个合成负例（12个明确业务差异、2个含义无法由观察确定的值）及2个合成正例。全部能通过小范围差异筛选。标签和隐藏业务语义不发送给模型。</p>
<p>拒绝不确定案例属于保守决策要求。两个含义不明反例的真实差异来自fixture定义，模型无法从当前可见输入充分确定；它们用于检查错误自信，不是声称真实网站必然如此。正例文字明确说明验证值用途，也不能据此证明真实网页可靠。</p>
<p>本实验不访问网站、不执行提交/删除/退出操作、不调用Qwen，不修改发布图。结果为开发检查，非独立泛化率；Jev分数不是校准概率。</p>'''
    html += '<pre>'+escape(json.dumps({k:v for k,v in summary.items() if k!='cases'},ensure_ascii=False,indent=2))+'</pre>'
    html += '<table><tr>'+''.join('<th>'+h+'</th>' for h in ['case','expected','score','decision','merged','本地复用','其他变化仍拒绝','seconds'])+'</tr>'+rows+'</table>'
    html += '<p><a href="cases.json">预登记样例与标签</a> · <a href="plan.json">实验计划</a> · <a href="summary.json">结果</a> · <a href="final-audit.json">完整性检查</a></p>'
    (out/'report.html').write_text(html,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    if not out.is_relative_to(ROOT/'runs'): raise ValueError('Output must be inside workspace runs')
    cases=fixtures();out.mkdir(parents=True,exist_ok=False)
    write_json(out/'cases.json',cases)
    plan={'registered_at':datetime.now(timezone.utc).isoformat(),'threshold':THRESHOLD,
          'model':'jev-1.14.0','proxy':'http://127.0.0.1:7890','max_calls':len(cases),
          'cases_sha256':sha(out/'cases.json'),'labels_predeclared':True,
          'negative_count':14,'positive_count':2,'repetitions':1,'retries':False,
          'primary':'Count negative cases creating a scoped equivalence rule at threshold 0.80.',
          'gate':'No negative accepts; report all errors, never tune prompt or labels after results.',
          'limitations':'Synthetic counterexamples; 2 opaque cases deliberately not identifiable from visible input.',
          'website_actions':0,'qwen_calls':0}
    freeze(out,plan,extra=(ROOT/'runs/state-equivalence-check-20261103-v3/summary.json',out/'cases.json'))
    manifest=read(out/'source-manifest.json')
    for rel in ('scripts/check_equivalence_negatives.py',):
        target=out/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/rel).read_bytes());manifest['source'][rel]=sha(target)
    write_json(out/'source-manifest.json',manifest)
    judge=JevEquivalenceClient(ROOT/'.env',EventLog(out/'equivalence-calls'),len(cases),
                              model=plan['model'],proxy=plan['proxy'],stop_file=out/'STOP')
    results=[];status='failed';started=time.monotonic()
    try:
        for c in cases:
            g=Graph(out/'graphs'/c['id']);g.nodes[c['before']['id']]=deepcopy(c['before'])
            resolver=StateResolver(g,judge)
            tick=time.monotonic();resolved=resolver.resolve(deepcopy(c['after']),learn=True)
            elapsed=time.monotonic()-tick
            decision=g.data['state_equivalence']['decisions'][-1]['judgment']
            merged=resolved['id']==c['before']['id']
            reuse=None;other_rejected=None
            if merged:
                # A fresh third value exercises the actual persisted masking rule with no judge.
                future=deepcopy(c['after']);slot=small_difference(c['before'],c['after'])[0]
                future['actions'][slot['control']][slot['field']]+='third';future['id']=state_id(future)
                local=StateResolver(Graph.load(g.directory))
                reuse=local.resolve(future)['id']==c['before']['id']
                changed=deepcopy(future);changed['text']+=' Changed business information.';changed['id']=state_id(changed)
                other_rejected=local.resolve(changed)['id']!=c['before']['id']
            row={'case':c['id'],'expected':c['expected'],**decision,'merged':merged,
                 'request_id':judge.last_request_id,'rule_reused':reuse,'other_change_rejected':other_rejected,
                 'seconds':elapsed}
            results.append(row);write_json(out/'progress.json',results)
            print(json.dumps({k:row[k] for k in ('case','score','decision','merged')}),flush=True)
        status='completed'
    except Exception as error:
        write_json(out/'failure.json',{'error_type':type(error).__name__})
        raise
    finally:
        events=read_events(out/'equivalence-calls/events.jsonl')
        received=[e for e in events if e['event']=='responses_received']
        negatives=[r for r in results if r['expected']!='same']
        summary={'status':status,'threshold':THRESHOLD,'cases':results,
                 'negatives_completed':len(negatives),'false_accepts':[r['case'] for r in negatives if r['merged']],
                 'positive_accepts':sum(r['merged'] for r in results if r['expected']=='same'),
                 'physical_calls':sum(e['event']=='responses_started' for e in events),
                 'responses':len(received),'actual_models':sorted(set(e['actual_model'] for e in received)),
                 'usage':{k:sum((e.get('usage') or {}).get(k,0) for e in received) for k in ('input_tokens','output_tokens')},
                 'wall_seconds':time.monotonic()-started}
        write_json(out/'summary.json',summary)
        verify_inputs(out,ROOT)
        assert sha(out/'cases.json')==plan['cases_sha256']
        assert all(sha(out/'source'/rel)==h for rel,h in manifest['source'].items())
        write_json(out/'final-audit.json',{'source_files':len(manifest['source']),
            'protected_files':len(manifest['preserved']),'sources_inputs_labels_unchanged':True,
            'all_cases_mechanically_eligible':True,'threshold':THRESHOLD})
        render(out,summary)


if __name__=='__main__': main()
