"""Offline report for the Jev comparison and its independent Reddit construction trials."""
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.storage import write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    runs = ROOT / 'runs'
    full = runs / 'reddit-jev-pipeline-dev2-20261030-v1'
    small = runs / 'reddit-jev-pipeline-dev2-20261030-small-v1'
    saved = runs / 'jev-ranking-dev2-20261030-v2'
    reference = runs / 'reddit-qwen-pipeline-dev2-20261029-v2'
    large, little, matched, old = [read(p / 'summary.json') for p in [full, small, saved, reference]]
    diagnosis = read(full / 'failure-diagnosis.json')
    totals = {'jev': {'physical': matched['physical_requests'], 'input_tokens': matched['input_tokens'],
                      'output_tokens': matched['output_tokens'], 'unknown_usage': matched['unknown_usage']},
              'qwen': {'physical': 0, 'input_tokens': 0, 'output_tokens': 0, 'unknown_usage': 0}}
    for result in [little, large]:
        for role, row in result['roles'].items():
            group = totals['jev' if role == 'candidate_ranking' else 'qwen']
            for key in group:
                group[key] += row[key]
    comparison = {'saved_input_rankings': {'cases': 5, 'qwen_median_seconds': old['roles']['candidate_ranking']['median_received_seconds'],
        'jev_median_seconds': matched['median_jev_seconds']},
        'small_completed': little, 'expanded_partial': large, 'new_usage_totals': totals,
        'diagnosis': diagnosis, 'routing': 'Jev explicitly uses existing local proxy 127.0.0.1:7890; Qwen remains direct. No node switch or proxy disconnect.',
        'limitations': ['Different exploration actions; same cap does not make a paired causal control.',
            'No benchmark task evaluation or functional coverage labels.',
            'Expanded graph stopped at 19 states after subscription changed the root; not a completed 60-state run.',
            'Prior direct HTTP451 attempt remains separate with unknown usage.']}
    write_json(full / 'comparison.json', comparison)
    rows = ''
    for label, data in [('原 Qwen 排名', old), ('Jev 小规模', little), ('Jev 扩展（中止）', large)]:
        g, roles = data['graph'], data['roles']
        vals = [label, g['nodes'], g['unique_urls'], g['attempts'],
                f"{data['state']['wall_seconds']:.1f}", f"{roles['candidate_ranking']['median_received_seconds']:.3f}",
                f"{roles['transition_description']['median_received_seconds']:.3f}"]
        rows += '<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in vals) + '</tr>'
    usage_rows = ''.join('<tr><td>' + name + '</td>' + ''.join(f'<td>{data[key]:,}</td>' for key in
        ['physical', 'input_tokens', 'output_tokens', 'unknown_usage']) + '</tr>' for name, data in totals.items())
    output = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jev 排名与 Reddit 建图实验</title><style>
body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#f3f5f7;color:#1d2939;margin:0}
main{max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:34px;line-height:1.25}h2{font-size:23px;margin-top:32px}
.panel{background:white;border:1px solid #dce2e8;border-radius:12px;padding:22px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.number{font-size:33px;font-weight:700;color:#176a52}.caption{color:#526270}.warn{border-left:5px solid #b7791f;background:#fffbef}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:11px 8px;border-bottom:1px solid #e5e9ed;text-align:left}th{color:#475467}a{color:#175cd3}code{background:#edf1f5;padding:2px 4px}.scroll{overflow-x:auto}@media(max-width:700px){.grid{grid-template-columns:1fr}h1{font-size:27px}}
</style><main><div class="caption">2026-10-30 · dev2 · 独立开发实验</div>
<h1>Jev 排名已明显提速，扩展建图暴露了入口变化问题</h1>
<p>排名使用 Jev 1.14.0，转换描述使用百炼 Qwen3.8-27B。保持两个描述线程、一个预取排名线程；Jev 请求走现有本地代理，代理未关闭、节点未切换。</p>
<div class="grid"><div class="panel"><div class="caption">相同五份输入 · 排名中位耗时</div><div class="number">0.312 秒</div><div>原 Qwen：77.454 秒</div></div>
<div class="panel"><div class="caption">12 状态小规模 · 总耗时</div><div class="number">115 秒</div><div>上一轮全 Qwen：409 秒</div></div>
<div class="panel"><div class="caption">扩展实验 · 停止时状态数</div><div class="number">19 个</div><div>上限 60，未完成，不计作成功全量建图</div></div></div>
<div class="panel warn"><strong>扩展阶段为何停止</strong><p>第26次点击 Subscribe 后，订阅按钮变为 Unsubscribe；之后新浏览器上下文看到的首页持续不同于记录的入口。入口校验正确拒绝恢复，但旧控制流程继续尝试其他候选，累计69次未执行尝试。发现后已停止实验。所有30次模型请求均成功，代理与 Jev 接口没有失败。</p>
<p>开发版现已修复为：首次确认入口失配就保留失败、等待在途描述并停止。真实 Chromium 离线回归确认不会继续消耗剩余候选。此修复不改写历史状态、不绕过校验，也没有自动撤销订阅或发起付费重跑。持久状态变化后的继续探索策略尚未实现。</p></div>
<h2>耗时与产出</h2><div class="panel scroll"><table><tr><th>运行</th><th>状态</th><th>URL</th><th>尝试记录</th><th>总秒</th><th>排名中位秒</th><th>描述中位秒</th></tr>''' + rows + '''</table>
<p>小规模总耗时减少约72%，但探索动作不同：这轮8个URL，原轮10个URL。这是运行耗时观察，不能据此声称覆盖质量、任务成功率或普遍加速比例。扩展总耗时含本地编码器准备约66秒，以及入口失配后的无效恢复；69条记录不是实际执行了69次候选动作。</p></div>
<h2>排名效果与适配差别</h2><div class="panel"><p>五份保存输入全部成功解析，分数具有区分度；热门候选包含 Wiki、Forums、Submit、Create forum、All pages 等。没有独立功能标签，选择重合度不能当作正确率。Jev 对每个候选返回 Noul 分数，程序按分数及结构组选择代表；原 Qwen 会生成自由文本功能组，两者不是等价策略。</p>
<p>候选仍可能包含改变服务端状态的操作。Qwen 事后把订阅边标为不可回放，只能约束未来回放，不能防止首次点击改变环境。图谱大小并非本次停滞原因。</p></div>
<h2>图谱与证据</h2><div class="panel"><p><a href="../reddit-jev-pipeline-dev2-20261030-small-v1/graph/viewer.html">完成的小规模图谱</a> · <a href="graph/viewer.html">扩展实验的部分图谱</a> · <a href="../jev-ranking-dev2-20261030-v2/report.html">五份相同输入检查</a></p>
<p><a href="failure-diagnosis.json">入口失配诊断</a> · <a href="final-audit.json">源码/输入核验</a> · <a href="comparison.json">完整比较数据</a> · <a href="plan.json">扩展登记</a></p>
<p>小规模80个源码文件、19个保护输入；扩展83个源码文件、22个保护输入均核对通过。活动图与用户提示未改。扩展没有进入聚类、制卡、框架或索引阶段；19状态图保留的是历史观察，现场入口已变化，不能称为已通过当前回放验证。冻结运行器将任意 STOP 标为 user_stopped，operator-stop.json 明确本次是工程停止，不是用户暂停。</p></div>
<h2>本次新增模型用量</h2><div class="panel scroll"><table><tr><th>模型</th><th>物理请求</th><th>输入 token</th><th>输出 token</th><th>未知用量</th></tr>''' + usage_rows + '''</table>
<p>包括五份保存输入检查、小规模及中止的扩展实验，各计一次；没有重试。旧 Qwen 对照和上一次直连451失败分开保留，451未返回用量，不计为免费。未按未知价格估算金额。两批必要离线回归各30项通过，存在重叠，不相加称为60个独立测试。</p></div></main></html>'''
    (full / 'analysis.html').write_text(output, encoding='utf-8')
    print(json.dumps({'report': str(full / 'analysis.html'), 'usage': totals}, ensure_ascii=False))


if __name__ == '__main__':
    main()
