"""Read frozen calibration evidence; no model/browser calls or threshold publication."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_reddit_hard10_225 import statistics as usage_stats
from web2kg.stage1.function_cards import file_hash
from web2kg.stage2.confidence import parse_confidence
from web2kg.common.storage import write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def events(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def fires(row, immediate, low, lifecycle=False):
    if lifecycle and (not row['diagnosis_available'] or row['cooldown']):
        return False
    values = row['window']
    return bool(row['score'] is not None and row['score'] <= immediate or
        len(values) == 3 and all(v is not None for v in values) and
        sum(values)/3 < low and sum(v < low for v in values) >= 2)


def metrics(rows, immediate, low, lifecycle=False):
    counted = [r for r in rows if r['label'] != 'ambiguous' and r['score'] is not None]
    needed = [r for r in counted if r['label'] == 'review_needed']
    normal = [r for r in counted if r['label'] == 'reasonable']
    tp = sum(fires(r, immediate, low, lifecycle) for r in needed)
    fp = sum(fires(r, immediate, low, lifecycle) for r in normal)
    return dict(immediate=immediate, window_mean=low, review_hits=tp, review_steps=len(needed),
        normal_alerts=fp, normal_steps=len(normal), recall=tp/len(needed) if needed else None,
        false_alarm_rate=fp/len(normal) if normal else None,
        precision=tp/(tp+fp) if tp+fp else None,
        first_hit={str(task):next((r['step'] for r in needed if r['task_id']==task and fires(r,immediate,low,lifecycle)),None)
                   for task in sorted({r['task_id'] for r in needed})})


def analyze(batch):
    protocol, state, labels = read(batch/'protocol.json'), read(batch/'state.json'), read(batch/'step-labels.json')
    if state['status'] != 'completed':
        raise ValueError('Only completed calibration is accepted; do not conceal an unresolved run')
    # Deduplicate the shared protected inputs; retain all task handoffs in the audit.
    guards = {}
    def merge(items):
        for path, sha in items.items():
            if path in guards and guards[path] != sha:
                raise ValueError('Conflicting protected identity: '+path)
            guards[path] = sha
    runtime = Path(protocol['runtime'])
    merge({str(runtime/rel):sha for rel,sha in read(runtime/'manifest.json').items()})
    cases, rows, role_events = [], [], defaultdict(list)
    for case in protocol['cases']:
        key = str(case['task_id']); child = batch/f'case-{key}'; folder=child/key
        config, child_state = read(child/'protocol.json'), read(child/'state.json')
        merge(read(child/'protected-inputs.json'))
        handoff = Path(child_state['knowledge_manifest'])
        merge({str(handoff):child_state['knowledge_manifest_sha256']}); merge(read(handoff))
        trajectory = read(folder/'trajectory.json')
        judgments = labels['cases'][key]['labels']
        assert len(trajectory) == len(judgments)
        proposals, committed = {}, []
        diagnoses, returned = 0, False
        for e in events(folder/'events.jsonl'):
            if e['event'] == 'recovery_diagnosis': diagnoses += 1
            elif e['event'] == 'counterfactual_return': returned = True
            elif e['event'] == 'direction_confidence':
                proposals[e['proposal']] = {**e, 'diagnosis_available': diagnoses < config['max_diagnoses'] and not returned}
            elif e['event'] == 'direction_confidence_committed': committed.append(e)
        assert len(committed) == len(trajectory)
        for i, (t, label, commit) in enumerate(zip(trajectory, judgments, committed), 1):
            signal = proposals[commit['proposal']]
            assert commit['score'] == signal['score'] == parse_confidence(t.get('direction_confidence'))
            rows.append(dict(task_id=case['task_id'], family=case['family'], step=i, label=label,
                score=signal['score'], window=signal['window'], cooldown=signal['cooldown'],
                diagnosis_available=signal['diagnosis_available'], action=t['action'],
                observation=t.get('observation_description'), reason=t.get('reason')))
        for log in folder.rglob('events.jsonl'):
            for e in events(log):
                if e['event'] in ('responses_started','responses_received','responses_failed','responses_retry','provider_request'):
                    role_events[e.get('purpose') or 'transport'].append(e)
        recovery = read(folder/'recovery/recovery.json')
        cases.append({**case, 'official_score':read(folder/'official.json')['score'],
            'steps':len(trajectory), 'scores':[parse_confidence(t.get('direction_confidence')) for t in trajectory],
            'seconds':child_state['total_seconds'], 'usage':usage_stats(folder),
            'diagnoses':recovery['decisions'], 'return':recovery['return'],
            'trace':f'case-{key}/trajectories/task-{key}.html'})
    mismatches = [path for path,sha in guards.items() if not Path(path).is_file() or file_hash(path)!=sha]
    if mismatches: raise ValueError('Integrity mismatch: '+str(mismatches[:3]))
    grid = [metrics(rows, a, w) for a in protocol['analysis_plan']['immediate_grid']
            for w in protocol['analysis_plan']['window_mean_grid']]
    sensitivity = [metrics(rows,a,.6) for a in [.3,.8,.9,.95,1.]]
    strict = [r for r in rows if r['task_id'] in (406,722,68)]
    # Train on whole families only, never randomly split neighboring steps.
    folds = []
    for family in sorted({r['family'] for r in rows}):
        train, test = [r for r in rows if r['family']!=family], [r for r in rows if r['family']==family]
        feasible = [metrics(train,a,w) for a in protocol['analysis_plan']['immediate_grid']
                    for w in protocol['analysis_plan']['window_mean_grid']]
        feasible = [m for m in feasible if (m['false_alarm_rate'] or 0)<=.1]
        best = max((m['recall'] or 0 for m in feasible), default=0)
        winners = [m for m in feasible if (m['recall'] or 0)==best]
        folds.append(dict(held_out_family=family, training_review_hits_max=best,
            equally_ranked_candidates=len(winners),
            held_out_default=metrics(test,.3,.6),
            note='No unique threshold selected; the false-alarm constraint is a descriptive post-run sensitivity rule, not a preregistered optimizer.'))
    totals={k:sum(c['usage'][k] for c in cases) for k in
        ('physical_requests','unknown_usage','input_tokens','output_tokens','request_seconds','output_retries','transport_retries')}
    roles={}
    for purpose, es in role_events.items():
        if purpose=='transport':continue
        started=[e for e in es if e['event']=='responses_started']
        received=[e for e in es if e['event']=='responses_received']
        known={e['request_id'] for e in received if {'input_tokens','output_tokens'}<=set(e.get('usage',{}))}
        roles[purpose]=dict(physical_requests=len(started),unknown_usage=sum(e['request_id'] not in known for e in started),
            input_tokens=sum(e.get('usage',{}).get('input_tokens',0) for e in received),
            output_tokens=sum(e.get('usage',{}).get('output_tokens',0) for e in received),
            reasoning_tokens_in_output=sum(e.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0) for e in received),
            request_seconds=sum(e.get('seconds',0) for e in received))
    report=dict(status='completed', created_at=datetime.now(timezone.utc).isoformat(),
        analysis_script_sha256=file_hash(Path(__file__)),
        cases=cases, steps=rows, labels=labels, totals=totals, roles=roles,
        actual_models=sorted({m for c in cases for m in c['usage']['actual_models']}),
        providers=sorted({e['provider'] for e in role_events['transport'] if e['event']=='provider_request'}),
        wall_seconds=state['seconds'], successes=sum(c['official_score']==1 for c in cases),
        score_distribution=dict(Counter(str(r['score']) for r in rows)),
        label_counts=dict(Counter(r['label'] for r in rows)),
        label_medians={label:statistics.median(r['score'] for r in rows if r['label']==label)
                       for label in ('reasonable','review_needed','ambiguous')},
        preregistered_grid=grid, exploratory_high_thresholds=sensitivity,
        lifecycle_default=metrics(rows,.3,.6,True), strict_default=metrics(strict,.3,.6),
        family_holdout=folds,
        watchdog_observed=sum('ten_actions_without_diagnosis' in d['triggers'] for c in cases for d in c['diagnoses']),
        recommendation='Do not raise thresholds to force detections. Keep0.30/0.60 as conservative experimental values only; no uniquely calibrated cutoff or evidence of reliable self-detection. Use the10-action watchdog within existing diagnosis caps; it cannot cover early stops.',
        limits=protocol['limits']+' Single-reviewer labels are review-worthiness judgments, not per-step evaluator truth. Weak forum-fit labels and ambiguous labels are separated. No new active-threshold rerun; offline candidate firing is not an actual repaired outcome. Adjacent steps are correlated. Exact thresholds within the observed score gap cannot be identified.')
    write_json(batch/'analysis.json',report)
    write_json(batch/'final-audit.json',dict(passed=True,protected_files=len(guards),
        frozen_files=len(read(runtime/'manifest.json')),labels_sha256=file_hash(batch/'step-labels.json'),
        tasks=8,starts=8,official_scores=8,actor_actions=len(rows),
        valid_scores=sum(r['score'] is not None for r in rows),unknown_usage=totals['unknown_usage'],
        actual_models=report['actual_models'],providers=report['providers'],
        source_input_and_handoff_verified=True))
    render(batch,report)
    print(json.dumps({k:report[k] for k in ['successes','score_distribution','label_counts','label_medians','totals','lifecycle_default','strict_default','watchdog_observed']},ensure_ascii=False))


def render(batch,r):
    h=html.escape
    cases=''.join(f'<tr><td><a href="{c["trace"]}">{c["task_id"]}</a></td><td>{c["graph_capacity"]}</td>'
        f'<td>{c["official_score"]}</td><td>{c["steps"]}</td><td>{h(", ".join(map(str,c["scores"])))}</td>'
        f'<td>{len(c["diagnoses"])}</td></tr>' for c in r['cases'])
    thresholds=''.join(f'<tr><td>≤{m["immediate"]}</td><td>{m["review_hits"]}/{m["review_steps"]}</td>'
        f'<td>{m["normal_alerts"]}/{m["normal_steps"]}</td></tr>' for m in r['exploratory_high_thresholds'])
    details=''
    for c in r['cases']:
        rows=[s for s in r['steps'] if s['task_id']==c['task_id']]
        detail=''.join(f'<tr><td>{s["step"]}</td><td>{s["score"]}</td><td>{h(s["label"])}</td>'
            f'<td>{h(s["action"])}</td><td>{h(s["observation"] or "")}</td></tr>' for s in rows)
        details+=f'<details><summary>任务{c["task_id"]} · 逐步证据与分数</summary><p>{h(r["labels"]["cases"][str(c["task_id"])]["reason"])}</p><table><tr><th>步</th><th>置信度</th><th>标注</th><th>动作</th><th>动作前观察摘要</th></tr>{detail}</table></details>'
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>路线置信度：8个失败案例重跑与阈值评估</title>
<style>body{font:16px/1.65 system-ui;max-width:1180px;margin:36px auto;padding:0 24px;color:#182433;background:#fafbfd}h1{font-size:30px}table{border-collapse:collapse;width:100%;margin:18px 0}th,td{padding:10px;border-bottom:1px solid #dce2e9;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#eef2f6}details{margin:16px 0;padding:12px;background:white;border:1px solid #dce2e9}summary{cursor:pointer;font-weight:600}.lead{padding:18px;background:#edf4ff;border-left:4px solid #3366cc}a{color:#2453a6}pre{white-space:pre-wrap}</style>
<h1>路线置信度：8个历史失败案例重跑</h1>
<p class="lead">57次实际行动全部给出有效分数。错误路线往往仍报0.9–1.0；低分条件能在离线试算中标出406的过早停止，但提高阈值会大量打断正常步骤。本批不足以选出可靠、唯一的自动触发阈值。</p>
<p>全角色Qwen3.7-flash／百炼。4个非Hard、4个Hard，初始60／225节点图独立副本，每题空任务经验。8题完整结束，2成功、6失败。不是全部Reddit成功率，也不是置信度干预收益：本批只记录分数，原诊断与十步检查仍生效。</p>
<h2>逐题结果</h2><table><tr><th>任务／实际轨迹</th><th>初始节点</th><th>官方分</th><th>行动数</th><th>置信度序列</th><th>实际诊断</th></tr>'''+cases+'''</table>
<h2>阈值为什么不能直接调高</h2><p>先隐藏分数阅读轨迹，再标注33步合理推进、18步值得复查、6步存在歧义。值得复查不等于已经证明偏离，更不能直接禁用控件。论坛适配性的标注较弱；去掉它们后的严格子集另见analysis.json。合理推进的分数中位数是0.9，值得复查的中位数反而是1.0。</p>
<table><tr><th>当前置信度触发线<br>（最近三步条件保持0.60）</th><th>抓到的复查步骤</th><th>误报的正常步骤</th></tr>'''+thresholds+'''</table>
<p>表中0.80以上是明确标注的事后敏感性试算。它衡量保存轨迹上的候选触发，暂不施加次数上限／冷却；不是实际运行中的额外诊断次数或修复率。0.80新增命中的68第15步，实际已经没有诊断机会。预登记30组条件（单步0.2–0.7、三步平均0.4–0.8）产生同样的结果：只识别406最后一步。按任务家族留一也不能区分这些候选。不能因此宣称0.30是拟合出的最优值。去掉较弱的论坛适配标注后，严格子集有6个复查步骤，原条件仍只抓到1个；不是由论坛样本单独造成的结论。</p>
<h2>十步检查与现有诊断</h2><p>601、635、406、722、638、715、66均在十步以内结束。68的第6、8步触发原重复到达诊断，结论on_track；第10步第三次诊断判定重复访问并返回到已在的列表（实际回放0动作），随后不能再诊断。十步条件每次都从最新诊断重新计数，因此本批没有单独触发十步检查；第十次行动边界已用真实浏览器离线用例验证。68最后仍失败，不能声称返回已经修好任务。</p>
<h2>当前建议</h2><p>不提高阈值来强行增加命中。0.30／0.60保留为保守试验值，置信度触发暂不默认启用；诊断启用时保留每十次行动未诊断检查，沿用三次诊断／一次返回总额。需要覆盖3–7步就停止的失败时，更值得下一步验证停止前的证据核验，而不是单纯抬高分数阈值；本批没有实现或测试该额外规则。</p>
<h2>证据、成本与限制</h2><p>原结果、源图与发布指针不变。部分旧Hard交接图已被先前授权清理，本次使用保留初始图；当前精简上下文也不同于旧尝试。只有一位标注者、案例已暴露、样本按失败选择，同行步骤相关，不推断泛化或组件因果收益。</p>
<p>实际请求和用量（包含描述、卡片、框架及诊断；没有额外付费预检，无虚构价格）：134次请求、未知用量0、传输重试0。4次格式重生成都属于卡片；Actor另有1次AX搜索，未计入57次行动。reasoning_tokens_in_output是输出token中的子集，不重复相加。</p><pre>'''+h(json.dumps({'totals':r['totals'],'roles':r['roles'],'wall_seconds':r['wall_seconds']},ensure_ascii=False,indent=2))+'''</pre>
<p><a href="protocol.json">预登记协议</a> · <a href="step-labels.json">先行标注</a> · <a href="analysis.json">逐步数据、阈值网格和角色用量</a> · <a href="final-audit.json">完整性核验</a></p>'''+details
    (batch/'report.html').write_text(page,encoding='utf-8')
    for c in r['cases']:
        if not (batch/c['trace']).is_file(): raise ValueError('Missing actual trajectory link')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    analyze(p.parse_args().directory.resolve())
