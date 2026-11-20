"""Audit the three independent domain-feedback retries without reading evaluator answers."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scripts import run_reddit_hard10_225 as run
from web2kg.stage2.action_notes import parse_sections


def audit(batch):
    state = run.read(batch/'state.json'); protocol = run.read(batch/'protocol.json')
    if state['status'] != 'completed': raise ValueError('Wait for all three cases and final guards')
    manifest = run.read(Path(protocol['runtime'])/'manifest.json')
    for rel, expected in manifest.items():
        if run.file_hash(Path(protocol['runtime'])/rel) != expected:
            raise ValueError('Frozen source changed: '+rel)
    checks = run.read(protocol['checks'])
    for rel, expected in checks['source_sha256'].items():
        if run.file_hash(ROOT/rel) != expected: raise ValueError('Tested source changed: '+rel)
    setup = run.read(batch/'setup-retry-audit.json')
    for path, expected in setup['sha256'].items():
        if run.file_hash(path) != expected: raise ValueError('Setup failure evidence changed: '+path)
    old_runtime = Path(setup['failed_registration'])/'frozen-runtime'
    for rel, expected in run.read(old_runtime/'manifest.json').items():
        if run.file_hash(old_runtime/rel) != expected: raise ValueError('Setup frozen source changed')
    rows = []
    for task in protocol['tasks']:
        child = batch/f'case-{task}'; folder = child/str(task)
        config = run.read(child/'protocol.json'); child_state = run.read(child/'state.json')
        result = run.read(folder/'result.json'); score = run.read(folder/'official.json').get('score')
        assert child_state['status'] == 'completed' and score in (0, 1)
        full = [e for e in run.read_events(child/'timings/events.jsonl')
                if e['event'] == 'integrity_verified' and e['scope'] == 'full']
        assert len(full) >= 3  # registration, start, end
        all_events = [e for p in folder.rglob('events.jsonl') for e in run.read_events(p)]
        main = run.read_events(folder/'events.jsonl')
        calls = [e for e in main if e['event'] == 'responses_started' and e.get('purpose') == 'original_actor']
        instructions = [run.read(folder/'llm'/e['request_id']/'request.json')['instructions'] for e in calls]
        assert all("Available navigation site IDs: ['reddit']" in p for p in instructions)
        domains = []
        for path in (folder/'llm').glob('*/parsed.txt'):
            action = parse_sections(path.read_text(encoding='utf-8'), ['action']).get('action', '')
            if action.startswith('optimal_navigate '):
                domains.append(re.search(r'optimal_navigate ?\[([\s\S]+?)\]\s*\[([\s\S]+?)\]', action).group(2).strip())
        assert all(domain == 'reddit' for domain in domains)
        corrections = Counter(e.get('failure_kind') for e in all_events if e['event'] == 'responses_output_retry')
        graph = run.Graph.load(folder/'execution-graph'); index = run.read(graph.directory/'index.json')
        assert len(graph.nodes) == 225 and set(index['ids']) == set(graph.nodes)
        old = Path(config['excluded_attempts'][0]['directory'])
        stats = run.statistics(folder)
        stats['logical_calls'] = stats['physical_requests'] - stats['output_retries'] - sum(
            e['event'] == 'responses_retry' and e.get('persistent', False) for e in all_events)
        roles = {name:run.statistics(folder/sub) for name,sub in
                 [('description','graph-annotations'), ('cards_framework','graph-cards'), ('diagnosis_learning','diagnosis-calls')]}
        aux_fields = ('physical_requests','unknown_usage','input_tokens','output_tokens','request_seconds','output_retries','transport_retries')
        roles['actor_selector'] = {k:stats[k]-sum(r[k] for r in roles.values()) for k in aux_fields}
        rows.append(dict(task_id=task, official_score=score, previous_official_score=run.read(old/'official.json').get('score'),
            steps=result['steps'], task_seconds=result['seconds'], total_seconds=result['total_seconds'],
            maintenance_seconds=round(result['total_seconds']-result['seconds'],3),
            original_input=config['source_graph'], previous_attempt=str(old), previous_usage=run.statistics(old),
            usage=stats, roles=roles, output_corrections=dict(corrections), accepted_navigation_domains=domains,
            allowed_site_ids_in_all_actor_requests=True, final_guard=full[-1],
            graph=dict(nodes=len(graph.nodes),edges=len(graph.data['edges']),indexed=len(index['ids'])),
            trajectory=f'case-{task}/trajectories/task-{task}.html'))
    totals = {k:sum(r['usage'][k] for r in rows) for k in
              ('physical_requests','logical_calls','unknown_usage','input_tokens','output_tokens','output_retries','transport_retries')}
    evidence617 = dict(
        task='Re-post the image of Firework from this forum to news forum using the image URL and title "from /f/pics"',
        selected_post_title='One of the funnier marketing images for my fireworks company.',
        selected_image='http://localhost:10599/submission_images/36addec4cfcbaabf030e2332d846a48e799859a825980194777e0a5e53a48a62.png',
        source_observation='case-617/617/step-1.txt', final_observation='case-617/617/step-16.txt',
        interpretation='Posted to news with the requested title, after failing to verify the intended image and choosing a related marketing image. Content selection is the supported failure hypothesis; no evaluator expected answer was read.',
        scope='Recorded only; no changes to search, recovery, or content selection in this repair.')
    source_obs = (batch/evidence617['source_observation']).read_text(encoding='utf-8')
    final_obs = (batch/evidence617['final_observation']).read_text(encoding='utf-8')
    assert evidence617['selected_post_title'] in source_obs and evidence617['selected_image'] in source_obs
    assert evidence617['selected_image'] in final_obs and "heading 'from /f/pics'" in final_obs
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), status='completed', tasks=rows,
        successes=sum(r['official_score'] for r in rows), planned=3, usage=totals,
        queue_seconds=state['seconds'], offline_checks=checks['checks'], frozen_files=len(manifest),
        setup_failure=dict(directory=setup['failed_registration'],task_workers=0,llm_requests=0,
                           seconds=run.read(Path(setup['failed_registration'])/'state.json')['seconds']),
        failure617=evidence617,
        validation='All actual navigation calls use reddit; no live invalid-domain response occurred. The rejection/feedback branch was exercised offline using all9 saved rejected responses, not claimed as a live self-correction.',
        limits=protocol['limits'])
    run.write_json(batch/'final-audit.json', result)
    esc = html.escape
    table = ''.join(f'<tr><td><a href="{r["trajectory"]}">{r["task_id"]}</a></td><td>{r["previous_official_score"]:g}</td>'
        f'<td>{r["official_score"]:g}</td><td>{r["steps"]}</td><td>{r["task_seconds"]:.1f}</td>'
        f'<td>{r["maintenance_seconds"]:.1f}</td><td>{len(r["accepted_navigation_domains"])}</td>'
        f'<td>{r["usage"]["physical_requests"]}</td></tr>' for r in rows)
    role_table = ''.join(f'<tr><td>{r["task_id"]}</td><td>{role}</td><td>{u["physical_requests"]}</td>'
        f'<td>{u["input_tokens"]:,}</td><td>{u["output_tokens"]:,}</td><td>{u["unknown_usage"]}</td></tr>'
        for r in rows for role,u in r['roles'].items())
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reddit 三题导航参数纠错验证</title><style>body{{max-width:1120px;margin:38px auto;padding:0 24px;font:16px/1.75 system-ui;color:#203040;background:#f6f8fa}}h1{{font-size:30px}}h2{{font-size:22px;margin-top:32px}}table{{width:100%;border-collapse:collapse;background:white;font-size:15px}}td,th{{padding:10px;border-bottom:1px solid #dce3e9;text-align:left}}a{{color:#1564a6}}.lead{{padding:18px;background:#eaf3fb;border-left:4px solid #21699f}}code{{background:#e9edf2;padding:2px 5px}}.scroll{{overflow-x:auto}}</style>
<h1>Reddit 三题导航参数纠错验证</h1>
<p>全部 LLM 角色：Qwen3.7-flash · 百炼直连 · 各自原始225节点输入 · 2026-11-07</p>
<p class="lead"><b>任务通过2/3：615、618通过，617未通过。</b> 三题都没有再因非法站点参数退出。617实际进行了4次图导航，站点参数均为 <code>reddit</code>。任务失败与参数校验是否正确分开判断。</p>
<h2>本次修复</h2><p>提示词列出当前 Actor 已注册的站点名。非法参数在导航前被拒绝，最新失败响应、具体错误与允许名称一起反馈给模型；原观察、截图和控件编号不变，不执行失败动作。不自动把URL改成站点名。最多三次非法站点响应，并受原有三次输出重生成共同上限约束。</p>
<p>119项相关离线检查通过，包括9条旧错误响应、同观察反馈、重试上限、用量统计、下一轮清除反馈和原版对照不受影响。线上未再出现非法domain，因此不能声称线上亲自纠正了三个非法参数；纠错分支的验证来自旧响应的离线回放。诊断、检索、制卡和其他问题未在本次修改。</p>
<h2>逐题结果</h2><div class="scroll"><table><tr><th>任务/轨迹</th><th>原分</th><th>新分</th><th>步骤</th><th>动作阶段秒</th><th>维护秒</th><th>图导航次数</th><th>物理请求</th></tr>{table}</table></div>
<p>每题分别复制原输入：615来自旧68交接图，617来自旧400交接图，618来自旧645交接图。三题不相互交接新知识；原经验文件哈希与对应历史交接一致。三份最终图均为225节点、225页索引，原图与发布指针未变。</p>
<h2>617为什么仍未通过</h2><p>模型多次查找Firework，没有确认目标图片，随后采用了另一篇标题为 <b>{esc(evidence617['selected_post_title'])}</b> 的帖子图片。最终确实在news发出了标题为“from /f/pics”的帖子，但官方评分为0。实际转发图可在<a href="{evidence617['source_observation']}">搜索页证据</a>和<a href="{evidence617['final_observation']}">最终页面</a>对应核对。内容选择未核实是当前最直接的失败解释；未读取评分器标准答案。这类问题按用户要求留待后续修复。</p>
<h2>用量与时间</h2><p>本次97次物理请求、91次逻辑调用，已知输入1,078,782 tokens、输出223,956 tokens；617有1次临时传输失败的用量未知。5次输出重生成中，2次是Actor格式错误、3次是卡片JSON/Schema错误，均在原上限内恢复；非法导航参数重生成0次。以上包含动作、选择、描述、卡片/框架和诊断，不按未知单价估算费用。</p>
<div class="scroll"><table><tr><th>任务</th><th>角色</th><th>物理请求</th><th>已知输入token</th><th>已知输出token</th><th>未知用量请求</th></tr>{role_table}</table></div>
<p>队列墙钟{state['seconds']/60:.2f}分钟，包含独立任务准备、编码器启动、站点重置、维护及完整性核验。表中的动作和维护耗时不含所有队列开销；并行请求时长不能相加当墙钟。旧三次失败的用量在审计JSON中单列，没有计入新请求总量。</p>
<h2>保留记录与核验</h2><p>首轮v1在Docker Desktop关闭时停于站点准备：0个任务worker、0次LLM请求。启动已安装的Docker Desktop后，在新v2目录重新登记；没有删除旧状态或直接恢复旧队列。该失败的时间和证据见<a href="setup-retry-audit.json">启动故障记录</a>。</p>
<p>三个子实验均完成版本开始/结束的全量保护核验，每题约2.98万文件；242个冻结源码文件与测试源码核对通过。旧41题报告仍为32/41，本次是三个已暴露失败的独立开发验证，不自动替换旧汇总，也不作为泛化或组件因果证据。</p>
<p><a href="final-audit.json">完整审计与旧失败用量</a> · <a href="protocol.json">登记协议</a> · <a href="../hard-domain-repair-checks-20261107.json">离线检查记录</a></p></html>'''
    (batch/'analysis.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(successes=result['successes'],usage=totals,report=str(batch/'analysis.html')),ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    audit(p.parse_args().directory.resolve())
