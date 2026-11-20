"""Offline report of the scoped-state checks and partial Qwen Flash crawl."""
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
    out=ROOT/'runs/reddit-jev-flash-equivalence-20261103-v1'
    checks=[ROOT/'runs'/f'state-equivalence-check-20261103-v{i}' for i in (1,2)]
    summary=read(out/'summary.json')
    graph=read(out/'graph/graph.json')
    prechecks=[read(p/'summary.json') for p in checks]
    decisions=[read(p/'graph/graph.json')['state_equivalence']['decisions'][0]['judgment'] for p in checks]
    totals={k:sum(r[k] for r in summary['roles'].values()) for k in
            ('physical','input_tokens','output_tokens','unknown_usage','retries')}
    for result in prechecks:
        totals['physical']+=result['physical_calls']
        for response in result['responses']:
            totals['input_tokens']+=response['usage']['input_tokens']
            totals['output_tokens']+=response['usage']['output_tokens']
    diagnosis={'status':'not_validated', 'jev_positive_scores':[d['score'] for d in decisions],
        'accept_threshold':.9,'threshold_changed':False,'learned_rules_in_actual_runs':0,
        'first_check':'Minimal difference context returned uncertain; checker stopped at the first mismatch.',
        'second_check':'Unchanged controls and recorded path added. Still uncertain; all five negative controls rejected.',
        'flash_failure':{'request_id':'183f786eca8946c4afda13e51049853d',
            'reason':'invalid_json','detail':'Unescaped ASCII double quotes around index inside the effect string, even after stripping the supported outer JSON fence.',
            'received':True,'transport_failure':False,'retried':False},
        'crawl_equivalence_calls':0,'scope':'Crawl stopped before exercising the account-menu judgment.',
        'whole_graph_completed':False,'new_usage_including_failed_versions':totals,
        'development_followup':'After all workers exited, added browser graph-switch resolver clearing and a regression test; frozen experiment sources unchanged.'}
    write_json(out/'diagnosis.json',diagnosis)
    tests={'batches':[{'passed':47,'seconds':5.59,'scope':'state equivalence, Jev adapter and Qwen transport (latest)'},
                     {'passed':42,'seconds':39.87,'scope':'state equivalence, pipeline and interaction identity'},
                     {'passed':30,'seconds':110.84,'scope':'state equivalence and Actor route navigation'},
                     {'passed':4,'seconds':47.59,'scope':'real Chromium, serial/parallel dynamic-token source with and without judge'}],
           'counts_overlap':True,'paid_models_in_tests':False,
           'warning':'Synthetic judges validate integration, not the real Jev classifier accuracy.'}
    write_json(out/'test-receipt.json',tests)
    row=summary['roles']['transition_description']
    body=f'''<h1>局部状态等价与 Qwen3.7-flash：未通过实测验证</h1>
<p class="lead">代码机制已实现，真实菜单仍未被 Jev 确认；Flash 小图因一次非法 JSON 停在 15 状态。
没有降低门槛、补造规则、重试失败动作或恢复旧实验。</p>
<h2>规则怎样生效</h2><p>先比较原 interaction-v2 身份。只允许最多两个控件 URL 查询参数的值变化；
页面 URL、标题层级、区域、可见文本及其他控件字段必须相同，变化控件数还受 10% 上限约束（至少容许一个）。
Jev 收到脱敏差异、未变控件上下文和原到达路径，等价分≥0.90才学习。规则只属于当前图和原状态，精确定位到控件/字段/查询位置；
再次出现新值时只做本地匹配，不调用模型。来源、目标和后续 Actor 导航共用规则。其他差异、证据不足、多个候选冲突不合并。</p>
<p>原节点 ID 和快照不改；实际观察保留 raw_state_id。普通 token 没有加入全站忽略名单。新实验通过计划显式切换 Flash，环境文件与历史模型配置未改。</p>
<h2>真实 Jev 检查</h2><table><tr><th>检查</th><th>结果</th></tr>
<tr><td>第一版：差异控件与参数长度</td><td>菜单等价分 0.43，uncertain，保留失败记录</td></tr>
<tr><td>第二版：增加未变控件和到达路径</td><td>菜单等价分 0.67，仍低于 0.90；重开浏览器后的两次访问亦未合并</td></tr>
<tr><td>对象查询参数 / 含义不明的 token</td><td>0.03 / 0.06，均不学习</td></tr>
<tr><td>菜单展开 / 订阅标签 / 页面 URL 改变</td><td>三项均由小差异筛选拒绝，不发模型请求</td></tr></table>
<p>第二版是看过第一版后的开发调整，不是独立盲测。三个菜单访问不是三个独立模型样本。
两版共四次 Jev 请求，无接口重试；当前证据不能证明这个判定策略解决了动态 token 问题。</p>
<h2>Flash 建图结果</h2><table><tr><th>项目</th><th>实测</th></tr>
<tr><td>计划 / 实际状态</td><td>上限30 / 实际15；14边、11 URL、18次尝试</td></tr>
<tr><td>总墙钟时间</td><td>{summary['state']['wall_seconds']:.3f} 秒，部分实验</td></tr>
<tr><td>实际响应模型</td><td>qwen3.7-flash ×13；jev-1.14.0 ×4</td></tr>
<tr><td>描述解析</td><td>12/13通过；一次 effect 文本中的 ASCII 双引号未转义</td></tr>
<tr><td>描述响应中位耗时</td><td>{row['median_received_seconds']:.3f}秒</td></tr>
<tr><td>Qwen输出 / 其中推理token</td><td>{row['output_tokens']:,} / {row['reasoning_tokens']:,}（推理已包含在输出）</td></tr>
<tr><td>状态等价请求 / 实际学习规则</td><td>0 / 0，停止前未运行到该判断</td></tr>
<tr><td>终止时标注</td><td>12完成，1非法输出失败，1未完成作业随队列取消；0待处理</td></tr>
<tr><td>接口错误 / 输出错误 / 重试 / 未知用量</td><td>0 / 1 / 0 / 0</td></tr></table>
<p>异常发生在成功收到响应后的 JSON 解析，不是网络超时。没有“修补”模型输出以冒充有效响应。
本轮未完成全图，未做聚类、制卡、框架或索引，未替换活动图。Flash 保留服务端默认推理设置；本轮不能证明它比旧模型更快。
另一个独立问题是 Jev 的保守判定：这不是 Flash 非法 JSON 导致的。</p>
<h2>验证、成本与边界</h2><p>最新核心回归47项通过；另有并行/身份42项、导航30项、真实Chromium同步/并行4项通过，各批存在重叠，不能相加。
本地固定判断器验证了一次学习、多次新值复用、持久化重载、来源/目标一致使用及错误反例；它不能替代真实 Jev 的准确性验证。</p>
<p>两版检查与部分建图合计{totals['physical']}个物理请求，{totals['input_tokens']:,}输入、{totals['output_tokens']:,}输出token，0未知用量，0重试。
所有失败均计入；没有估算价格。冻结源码、保护输入和活动图核验结果见完整性文件，停止后另做无模型首页检查。</p>
<p>结论：局部规则复用在工程上可运行，但当前 Jev 提示与0.90门槛未通过这个真实菜单正例。
下一步需要单独验证判定输入/校准与 Flash 的结构化输出设置；不能据本轮宣称状态识别或建图速度改善。</p>
<nav><a href="../state-equivalence-check-20261103-v1/summary.json">第一版检查</a>
<a href="../state-equivalence-check-20261103-v2/summary.json">第二版检查</a>
<a href="graph/viewer.html">部分图与轨迹</a><a href="summary.json">实测用量</a>
<a href="diagnosis.json">失败诊断</a><a href="final-integrity.json">完整性审计</a>
<a href="root-check.json">首页检查</a><a href="test-receipt.json">离线检查</a></nav>'''
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>局部状态等价与 Flash 验证</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{max-width:980px;margin:48px auto;padding:0 24px;font:16px/1.8 system-ui;color:#253244;background:#f6f8fb}
h1{font-size:29px;line-height:1.4}h2{margin-top:36px;font-size:21px}.lead{border-left:4px solid #bb6430;background:#fff3e9;padding:16px}
table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px 14px;border:1px solid #d8e0e9;text-align:left}th{background:#e9eff7}
nav{display:flex;flex-wrap:wrap;gap:16px;margin:30px 0}a{color:#175eb1}</style><body>'''+body+'</body></html>'
    (out/'analysis.html').write_text(document,encoding='utf-8')
    print(json.dumps({'report':str(out/'analysis.html'),'totals':totals}))


if __name__=='__main__':main()
