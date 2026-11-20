"""Offline, credential-free audit of a completed first-exploration experiment."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def run(out, check, project):
    summary, graph = read(out / 'summary.json'), read(out / 'graph/graph.json')
    assert summary['state']['status'] == 'completed'
    target, root = read(check / 'summary.json'), read(out / 'root-check.json')
    manifest = read(out / 'source-manifest.json')
    source_ok = all(hashlib.sha256((out / 'source' / p).read_bytes()).hexdigest() == h
                    for p,h in manifest['source'].items())
    assert source_ok and root['same_state']
    decisions, unused = [], 0
    for folder in sorted((out / 'ranking-calls/llm').iterdir()):
        request = read(folder / 'request.json')['state']['exploration']
        ranking = read(folder / 'ranking.json')
        node = request['page']['id']
        consumed = node in graph.get('rankings', {})
        unused += not consumed
        names = {a['id']:a['name'] for a in request['page']['candidates']}
        for row in ranking['scores']:
            decisions.append(dict(node=node, id=row['id'], name=names[row['id']], consumed=consumed,
                opportunity=row['score'], score=row['first_exploration_score'], allowed=row['first_exploration_allowed']))
    records = graph['transition_records']
    attempted = [r for r in records if r['action_attempted']]
    all_gated = all(graph['rankings'][r['source']]['first_exploration'][r['action']['id']]['allowed'] for r in attempted)
    subscriptions = [r for r in decisions if r['name'].startswith('Subscribe') and r['consumed']]
    assert all_gated and subscriptions and all(not r['allowed'] for r in subscriptions)
    assert not any(r['action']['name'].startswith('Subscribe') for r in attempted)
    graph_events = [json.loads(line) for line in (out / 'graph/events.jsonl').read_text().splitlines()]
    # Export field names only: form_action can contain private session tokens.
    restore = []
    for event in graph_events:
        if event['event'] != 'exploration_failed' or event.get('phase') != 'restore_source':
            continue
        before = {a['id']:a for a in graph['nodes'][event['source']]['actions']}
        after = {a['id']:a for a in event['failure_page']['actions']}
        changed = [dict(control=before[k]['name'], fields=[f for f in ('role','name','href','type','method',
                   'form_action','state','disabled','region','occurrence','match_count')
                   if before[k].get(f) != after[k].get(f)]) for k in before.keys() & after.keys()]
        restore.append(dict(attempt=event['attempt'], source=event['source'],
                            differences=[r for r in changed if r['fields']]))
    audit = dict(completed=True, frozen_sources_verified=source_ok, frozen_source_files=len(manifest['source']),
        protected_files=len(manifest['preserved']),
        protected_verification='Frozen runner verify_inputs passed at entry and after completed crawl, before status=completed.',
        all_attempted_actions_passed_gate=all_gated, first_actions=len(attempted),
        candidate_judgments=len(decisions), rejected=sum(not r['allowed'] for r in decisions),
        consumed_judgments=sum(r['consumed'] for r in decisions),
        consumed_rejected=sum(r['consumed'] and not r['allowed'] for r in decisions),
        subscriptions_rejected=len(subscriptions), subscription_actions=0, unused_ranking_requests=unused,
        home_state_check=root, restore_failures=restore,
        note='Graph reachability is structural, not proof all stored paths replay successfully. No cards/framework/index or task evaluation.')
    (out / 'first-exploration-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    (out / 'gate-decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2),encoding='utf8')
    combined = dict(input_tokens=sum(r['input_tokens'] for r in summary['roles'].values())+
                   sum(r['usage'].get('input_tokens',0) for r in target['responses']),
                   output_tokens=sum(r['output_tokens'] for r in summary['roles'].values())+
                   sum(r['usage'].get('output_tokens',0) for r in target['responses']),
                   physical_requests=sum(r['physical'] for r in summary['roles'].values())+target['physical_requests'])
    (out / 'combined-usage.json').write_text(json.dumps(combined,indent=2),encoding='utf8')
    roles = summary['roles']
    rows = [('图谱状态 / 边 / URL', f"{summary['graph']['nodes']} / {summary['graph']['edges']} / {summary['graph']['unique_urls']}"),
            ('总耗时 / 建图阶段', f"{summary['state']['wall_seconds']:.3f} / {summary['state']['timings']['construction_seconds']:.3f} 秒"),
            ('尝试', f"{summary['graph']['attempts']}：29转换、7无变化、6点击失败、10未执行"),
            ('Jev 判断＋排名', f"{roles['candidate_ranking']['physical']} 请求；中位 {roles['candidate_ranking']['median_received_seconds']:.3f} 秒"),
            ('Qwen 视觉描述', f"{roles['transition_description']['physical']} 请求；中位 {roles['transition_description']['median_received_seconds']:.3f} 秒"),
            ('并行作业峰值 / 重叠时长', f"{summary['pipeline']['peak_active_jobs']} / {summary['pipeline']['overlap_seconds']:.3f} 秒"),
            ('候选判断 / 拒绝', f"{audit['candidate_judgments']} / {audit['rejected']}（含未消费预取）"),
            ('实际消费的候选判断 / 拒绝', f"{audit['consumed_judgments']} / {audit['consumed_rejected']}"),
            ('订阅按钮：拒绝 / 实际点击', f"{audit['subscriptions_rejected']} / 0"),
            ('API 失败 / 重试 / 未知用量', '0 / 0 / 0'),
            ('页面描述 / 尚未完成描述', f"{summary['pipeline']['page_descriptions']} / {summary['pipeline']['pending_annotations']}"),
            ('核验', f"{len(manifest['source'])} 源码文件、{len(manifest['preserved'])} 保护文件；首页签名相同")]
    table = ''.join('<tr><th>'+html.escape(k)+'</th><td>'+html.escape(v)+'</td></tr>' for k,v in rows)
    false_rejects = [r['name'] or '无名称输入框' for case in target['cases'] for r in case['rows']
                    if r['expected_allowed'] and not r['allowed']]
    report = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jev 首次探索判断：Reddit 小规模建图</title><style>body{{font:17px/1.7 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#233}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #ddd;padding:9px}}th{{width:40%}}code{{background:#eee}}a{{color:#167}}.note{{background:#f3f6f7;padding:18px}}</style>
<h1>Jev 首次探索判断：Reddit 小规模建图</h1><p>2026-11-02 · 独立新建图 · 实际模型 Jev 1.14.0 + Qwen3.8-27B</p>
<p class="note">本轮达到30状态上限并正常收尾。订阅按钮在首次点击前被过滤，结束后以原登录状态重新打开首页，状态签名与起点相同。图未替换当前发布版本。</p>
<h2>改动</h2><p>每个候选有两个独立 Noul 问题：探索价值、首次点击是否适合不提交持久修改的建图。放行分≥0.8才参加原结构去重与排名；不确定时暂不探索。全部问题合并在一次原有排名请求中，保留子页预取和Qwen描述并行。模型读取真实角色、表单方法/目标、控件类型及页面上下文，不用订阅关键词作为程序黑名单。首次探索许可不等于后续可回放判断。</p>
<h2>先行检查：保存页面</h2><p>调用前人工标注3份真实页面的94个候选。28个预期拒绝项全部拒绝（25订阅、投稿提交、主题设置、退出）；66个预期允许项放行64个，误拒绝2个：{html.escape('、'.join(false_rejects))}。打开投稿页和创建论坛表单的链接放行。3请求中位 {target['median_seconds']:.3f} 秒。保留原候选并补入已保存的表单事实，与旧时延实验的输入并非完全一致。</p>
<h2>实际建图</h2><table>{table}</table>
<h2>剩余问题</h2><p>10次未执行都发生在同一个账号菜单状态恢复：可见控件的差异仅为退出表单的form_action字段。保留严格身份校验和原失败，未忽略普通token参数。另6次浏览器点击失败涉及点击超时、遮挡或视口问题。图中30状态结构可达，不代表全部路径都已在现场重放验证。</p>
<p>本轮没有在描述返回前实际重放暂定边；不据此判断暂定回放策略是否可靠。还有1份未消费预取，计入用量。0.8为预先固定的开发门槛，尚未跨网站校准；模型仍可能漏放或误拒绝。样本已经暴露，未做任务评分，也未生成卡片、框架或Jina索引。不能将不同探索路径的墙钟时间当作严格因果加速结果。</p>
<h2>用量</h2><p>建图36请求，加先行检查3请求，本轮共 {combined['physical_requests']} 请求；输入 {combined['input_tokens']:,}，输出 {combined['output_tokens']:,} token。Qwen推理token包含在输出内，不重复相加。没有重试或未知用量；未估计价格。历史失败和旧实验不计入本轮。Jev沿用本地代理，Qwen直连；未切节点或关闭代理。</p>
<h2>证据</h2><p><a href="graph/viewer.html">新图谱</a> · <a href="summary.json">运行汇总</a> · <a href="first-exploration-audit.json">检查与恢复问题</a> · <a href="gate-decisions.json">候选判断</a> · <a href="root-check.json">结束首页核验</a> · <a href="test-receipt.json">37项离线检查</a> · <a href="../{check.name}/summary.json">先行检查结果</a></p>
<p>接口合同参考：<a href="https://docs.typesafe.ai/api/system-one">TypeSafe 官方 Noul API</a>。本报告依据本地实际记录。</p></html>'''
    (out / 'analysis.html').write_text(report,encoding='utf8')
    print(json.dumps({k:v for k,v in audit.items() if k not in {'home_state_check','restore_failures'}},ensure_ascii=False))
    print(json.dumps(combined))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--check',type=Path,required=True)
    args=parser.parse_args()
    run(args.directory.resolve(),args.check.resolve(),Path(__file__).resolve().parents[1])
