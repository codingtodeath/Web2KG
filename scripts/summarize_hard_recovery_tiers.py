"""Audit Books diagnosis validation using public traces and scalar official scores only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scripts import run_reddit_hard10_225 as run


def audit(batch):
    state = run.read(batch/'state.json'); protocol = run.read(batch/'protocol.json')
    if state['status'] != 'completed':
        raise ValueError('Wait for all tasks and final guards; do not summarize an unresolved run as complete')
    manifest = run.read(Path(protocol['runtime'])/'manifest.json')
    for rel, expected in manifest.items():
        if run.file_hash(Path(protocol['runtime'])/rel) != expected:
            raise ValueError('Frozen source changed: '+rel)
    checks = run.read(protocol['checks'])
    for rel, expected in checks['source_sha256'].items():
        if run.file_hash(ROOT/rel) != expected: raise ValueError('Tested source changed: '+rel)
    rows = []
    for task in protocol['tasks']:
        child = batch/f'case-{task}'; folder = child/str(task)
        config = run.read(child/'protocol.json'); child_state = run.read(child/'state.json')
        result = run.read(folder/'result.json'); score = run.read(folder/'official.json').get('score')
        assert child_state['status'] == 'completed' and score in (0, 1)
        full = [e for e in run.read_events(child/'timings/events.jsonl')
                if e['event'] == 'integrity_verified' and e['scope'] == 'full']
        assert len(full) >= 3
        recovery = run.read(folder/'recovery/recovery.json')
        decisions = [r['decision'] for r in recovery['decisions']]
        assert recovery['decision_policy'] == 'three-tier-v1'
        for d in decisions:
            assert d['verdict'] in ('on_track', 'uncertain', 'deviated')
            assert d['deviated'] == (d['verdict'] == 'deviated')
            assert (d['root_step'] > 0) == d['deviated']
        definite = any(d['deviated'] for d in decisions)
        assert bool(recovery['attempted']) == definite
        assert bool(recovery['ban']) == definite
        if not definite:
            assert recovery['return'] is None and recovery['step_refund'] is None
        all_events = [e for p in folder.rglob('events.jsonl') for e in run.read_events(p)]
        main = run.read_events(folder/'events.jsonl')
        actor_calls = [e for e in main if e['event'] == 'responses_started' and e.get('purpose') == 'original_actor']
        advisory_requests = []
        for event in actor_calls:
            request = run.read(folder/'llm'/event['request_id']/'request.json')
            if 'diagnosis_advice' in json.dumps(request, ensure_ascii=False):
                advisory_requests.append(event['request_id'])
        graph = run.Graph.load(folder/'execution-graph'); index = run.read(graph.directory/'index.json')
        assert len(graph.nodes) == 225 and set(index['ids']) == set(graph.nodes)
        old = Path(config['excluded_attempts'][0]['directory'])
        stats = run.statistics(folder)
        stats['logical_calls'] = stats['physical_requests'] - stats['output_retries'] - sum(
            e['event'] == 'responses_retry' and e.get('persistent', False) for e in all_events)
        roles = {name:run.statistics(folder/sub) for name,sub in
                 [('description','graph-annotations'), ('cards_framework','graph-cards'), ('diagnosis_learning','diagnosis-calls')]}
        fields = ('physical_requests','unknown_usage','input_tokens','output_tokens','request_seconds','output_retries','transport_retries')
        roles['actor_selector'] = {k:stats[k]-sum(r[k] for r in roles.values()) for k in fields}
        old_recovery = run.read(old/'recovery/recovery.json')
        rows.append(dict(task_id=task, official_score=score, previous_official_score=run.read(old/'official.json').get('score'),
            steps=result['steps'], task_seconds=result['seconds'], total_seconds=result['total_seconds'],
            maintenance_seconds=result['total_seconds']-result['seconds'], error=result.get('error'),
            original_input=config['source_graph'], previous_attempt=str(old), previous_usage=run.statistics(old),
            previous_decisions=old_recovery['decisions'], previous_ban=old_recovery['ban'],
            usage=stats, roles=roles, decisions=recovery['decisions'], return_result=recovery['return'],
            ban=recovery['ban'], step_refund=recovery['step_refund'], advisory_actor_requests=advisory_requests,
            blocked_controls=sum(e['event']=='counterfactual_control_blocked' for e in main),
            graph=dict(nodes=len(graph.nodes),edges=len(graph.data['edges']),indexed=len(index['ids'])),
            final_guard=full[-1], trajectory=f'case-{task}/trajectories/task-{task}.html',
            diagnosis_report=f'case-{task}/{task}/recovery/report.html'))
    totals = {k:sum(r['usage'][k] for r in rows) for k in
              ('physical_requests','logical_calls','unknown_usage','input_tokens','output_tokens','output_retries','transport_retries')}
    counts = Counter(d['decision']['verdict'] for r in rows for d in r['decisions'])
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), status='completed', tasks=rows,
        successes=sum(r['official_score'] for r in rows), planned=3, usage=totals,
        queue_seconds=state['seconds'], offline_checks=checks['checks'], frozen_files=len(manifest),
        verdict_counts=dict(counts), limits=protocol['limits'])
    run.write_json(batch/'final-audit.json', result)
    esc = html.escape
    table = ''.join(f'<tr><td><a href="{r["trajectory"]}">{r["task_id"]}</a></td><td>{r["previous_official_score"]:g}</td>'
        f'<td>{r["official_score"]:g}</td><td>{r["steps"]}</td><td>{r["task_seconds"]:.1f}</td>'
        f'<td>{r["maintenance_seconds"]:.1f}</td><td>{len(r["decisions"])}</td><td>{"有" if r["ban"] else "无"}</td>'
        f'<td>{r["usage"]["physical_requests"]}</td></tr>' for r in rows)
    detail = ''.join(f'<h3>任务 {r["task_id"]}</h3><p><a href="{r["diagnosis_report"]}">诊断及返回记录</a> · '
        f'实际携带建议的 Actor 请求：{len(r["advisory_actor_requests"])} 次；控件拦截：{r["blocked_controls"]} 次。</p>'
        '<pre>'+esc(json.dumps(dict(decisions=r['decisions'],return_result=r['return_result'],
                                   ban=r['ban'],refund=r['step_refund']),ensure_ascii=False,indent=2))+'</pre>' for r in rows)
    role_table = ''.join(f'<tr><td>{r["task_id"]}</td><td>{role}</td><td>{u["physical_requests"]}</td>'
        f'<td>{u["input_tokens"]:,}</td><td>{u["output_tokens"]:,}</td><td>{u["unknown_usage"]}</td></tr>'
        for r in rows for role,u in r['roles'].items())
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Books 三题三级诊断验证</title><style>body{{max-width:1120px;margin:38px auto;padding:0 24px;font:16px/1.75 system-ui;color:#203040;background:#f6f8fa}}h1{{font-size:30px}}h2{{margin-top:30px}}table{{width:100%;border-collapse:collapse;background:white;font-size:15px}}td,th{{padding:10px;border-bottom:1px solid #dce3e9;text-align:left}}a{{color:#1564a6}}.lead{{padding:18px;background:#eaf3fb;border-left:4px solid #21699f}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}}.scroll{{overflow-x:auto}}</style>
<h1>Books 三题三级诊断验证</h1><p>所有 LLM 角色：Qwen3.7-flash · 每题各自原始225节点图 · 不继承前一道重跑的知识</p>
<p class="lead">官方通过 {result['successes']:g}/3。实际诊断结论：{esc(json.dumps(dict(counts),ensure_ascii=False))}。程序分流正确与任务成功分开判断。</p>
<h2>修改范围</h2><p>路径正确时继续；怀疑偏离但证据不足时仅建议核验；明确偏离才尝试返回并禁用原控件。前两级不消耗返回机会、不返还步数，最新结论与建议进入Actor上下文。诊断、返回和模型调用预算保持不变。未核验默认排序不能直接证明排序错误，反复检查不同详情可能是正常进展；提示词同时说明这些边界。</p>
<p>三级结论仍由模型判断，不能保证它不会把证据不足误判为明确偏离。离线{checks['checks']}项相关检查通过；包括真实本地浏览器在建议后再次点击原控件、建议进入Actor输入、结论升级、唯一返回机会、后台描述等待及首页返回。</p>
<h2>逐题结果</h2><div class="scroll"><table><tr><th>任务/轨迹</th><th>原分</th><th>新分</th><th>步数</th><th>动作秒</th><th>维护秒</th><th>诊断数</th><th>禁用</th><th>请求</th></tr>{table}</table></div>
<h2>真实诊断与处理</h2>{detail}
<h2>用量与时间</h2><p>新增{totals['physical_requests']}次物理请求、{totals['logical_calls']}次逻辑调用，已知输入{totals['input_tokens']:,}、输出{totals['output_tokens']:,} tokens；未知用量{totals['unknown_usage']}次，输出重生成{totals['output_retries']}次、传输重试{totals['transport_retries']}次。旧三次失败用量在审计中单列，不计入新增总量。未估算价格。</p>
<div class="scroll"><table><tr><th>任务</th><th>角色</th><th>请求</th><th>输入token</th><th>输出token</th><th>未知</th></tr>{role_table}</table></div>
<p>队列墙钟{state['seconds']/60:.2f}分钟，包含准备、重置、编码器启动、维护与核验；登记另计。每题结束保护核验通过，最终图各225节点和225页索引。原图、旧失败与41题32/41汇总保留。</p>
<h2>解释边界</h2><p>三个题目已用于开发，且都是同一论坛的信息提取任务。每题只有一次新尝试，模型路径可能不同。本次验证不构成泛化、统计显著性或组件因果证据；没有读取评分器标准答案。除三级诊断及其上下文以外，保留当前导航参数纠错、卡片验证和执行流程。</p>
<p><a href="final-audit.json">完整审计</a> · <a href="protocol.json">登记协议</a> · <a href="../hard-recovery-tiers-checks-20261107.json">离线检查</a></p></html>'''
    (batch/'analysis.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(successes=result['successes'],verdicts=dict(counts),usage=totals,report=str(batch/'analysis.html')),ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    audit(p.parse_args().directory.resolve())
