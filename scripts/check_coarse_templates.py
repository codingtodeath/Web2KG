"""Reuse frozen local structures; calibrate one silhouette tolerance across sites."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/template-clustering-dev2-20261029-v2/runtime'))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'runs/template-silhouette-dev2-20261029-v1/inputs.json')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    def save(name, data):
        (args.out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    source_hash = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    grid = [i / 100 for i in range(26)]
    sites = ['reddit', 'books', 'quotes', 'webscraper']
    save('plan.json', dict(inputs=str(args.inputs.resolve()), sha256=source_hash,
        grid=grid, sites=sites, primary='average / weighted Jaccard / unique structures',
        selection='Equal-site macro pair F1; within 0.01 of best choose smallest delta. '
                  'Each held-site fold selects delta using the other three sites only.',
        controls='Complete linkage and synthetic corpora reported separately, not used for primary tuning.',
        limitation='All corpora already exposed in development. This is leave-site-out development validation, not a new blind test.'))
    from web2kg.stage1.silhouette_selection import prepare, candidates, score_options, select
    from check_template_clustering import metrics
    inputs = json.loads(args.inputs.read_text(encoding='utf-8'))
    outputs, baseline = {}, {}
    for site, rows in inputs.items():
        profiles, truth = [r['profile'] for r in rows], [r['label'] for r in rows]
        raw = prepare(profiles)
        old = select(raw, candidates(raw, 'average'))
        baseline[site] = {**old, **metrics(truth, old['predicted'])}
        arrays = prepare(profiles, deduplicate=True)
        outputs[site] = {}
        for method in ('average', 'complete'):
            start = time.perf_counter()
            scored = score_options(arrays, candidates(arrays, method))
            seconds = time.perf_counter() - start + arrays['seconds']
            records = []
            for delta in grid:
                result = select(arrays, scored, tolerance=delta)
                result.update(metrics(truth, result['predicted']), delta=delta,
                              selection_seconds=seconds, n=len(rows))
                records.append(result)
            outputs[site][method] = records
    def choose(training):
        means = [statistics.mean(outputs[s]['average'][i]['pair_f1'] for s in training) for i in range(len(grid))]
        eligible = [i for i, value in enumerate(means) if value >= max(means) - .01 - 1e-12]
        return eligible[0], eligible, means
    folds = []
    for held in sites:
        training = [s for s in sites if s != held]
        i, interval, means = choose(training)
        folds.append(dict(held=held, training=training, delta=grid[i],
                          eligible_deltas=[grid[j] for j in interval], train_macro=means[i],
                          test=outputs[held]['average'][i]))
    chosen, eligible, means = choose(sites)
    common = sorted(set.intersection(*(set(f['eligible_deltas']) for f in folds)))
    result = dict(grid=grid, sites=sites, baseline=baseline, results=outputs, folds=folds,
                  chosen_delta=grid[chosen], near_best_deltas=[grid[i] for i in eligible],
                  common_fold_deltas=common, macro_pair_f1=means,
                  held_site_macro=statistics.mean(f['test']['pair_f1'] for f in folds))
    save('results.json', result)
    save('verification.json', dict(input_unchanged=source_hash == hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [Path(__file__), ROOT / 'web2kg/stage1/silhouette_selection.py', ROOT / 'web2kg/stage1/template_clustering.py']}))
    render(args.out, result, inputs)
    print(json.dumps({k: result[k] for k in ('chosen_delta', 'near_best_deltas', 'common_fold_deltas', 'held_site_macro')}))


def render(out, result, inputs):
    grid, sites = result['grid'], result['sites']
    outputs, baseline, folds = result['results'], result['baseline'], result['folds']
    chosen = grid.index(result['chosen_delta'])
    common, means = result['common_fold_deltas'], result['macro_pair_f1']
    e = html.escape
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>模板粗粒度聚类调参</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1150px;margin:35px auto;padding:20px;background:#f6f8fa}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #cbd5df;padding:8px}code{background:#e7eef4}</style>',
        '<h1>减少模板过细切分：跨站点共同容差</h1>',
        '<p>先按计分结构去重，平均链接建树；选择 S≥Smax−δ 的最少模板切分，再映射回全部页面。孤立空结构不参与。保留完全链接选项。</p>',
        f'<p>全站点等权调参默认 δ={grid[chosen]:.2f}；宏平均F1距最佳不超过0.01的网格：{e(str(result["near_best_deltas"]))}。各留出折共同范围：{e(str(common))}。</p>',
        '<p>这不是共同的绝对轮廓分范围；每站相对于自己的最高分取容差。δ使用粗标签调参，运行时不读取标签。每个留出折的δ只由另外三个站点选定；所有数据已经用于开发，不是盲测泛化证明。</p>',
        '<h2>全部页面评价与制卡数量代理</h2><table><tr><th>数据</th><th>页数→独特结构</th><th>旧最大轮廓 K / F1</th><th>仅去重 K / F1</th><th>共同δ K / F1 / 精确率 / 召回率</th><th>最终 S / Smax</th><th>选参毫秒</th></tr>']
    for site in inputs:
        old, zero, new = baseline[site], outputs[site]['average'][0], outputs[site]['average'][chosen]
        body.append(f'<tr><td>{e(site)}</td><td>{new["n"]}→{new["selection_points"]}</td><td>{old["clusters"]} / {old["pair_f1"]:.3f}</td><td>{zero["clusters"]} / {zero["pair_f1"]:.3f}</td><td>{new["clusters"]} / {new["pair_f1"]:.3f} / {new["precision"]:.3f} / {new["recall"]:.3f}</td><td>{new["silhouette"]:.3f} / {new["maximum_silhouette"]:.3f}</td><td>{new["selection_seconds"]*1000:.2f}</td></tr>')
    body += ['</table><h2>逐站留出</h2><table><tr><th>留出站</th><th>另外三站选δ</th><th>留出F1</th><th>模板数</th></tr>']
    for f in folds:
        body.append(f'<tr><td>{f["held"]}</td><td>{f["delta"]:.2f}</td><td>{f["test"]["pair_f1"]:.3f}</td><td>{f["test"]["clusters"]}</td></tr>')
    reddit_fold = next(f for f in folds if f['held'] == 'reddit')
    body += [f'</table><p>留出Reddit时，其余三个简单站点选δ={reddit_fold["delta"]:.2f}，该折F1={reddit_fold["test"]["pair_f1"]:.3f}。全量调参共同δ的宏平均F1={means[chosen]:.3f}，严格逐站留出的宏平均为{result["held_site_macro"]:.3f}，两者不能混作泛化结果。</p><h2>参数曲线</h2><table><tr><th>δ</th>' + ''.join(f'<th>{s}: K / F1</th>' for s in sites) + '<th>宏平均F1</th></tr>']
    for i, delta in enumerate(grid):
        body.append(f'<tr><td>{delta:.2f}</td>' + ''.join(f'<td>{outputs[s]["average"][i]["clusters"]} / {outputs[s]["average"][i]["pair_f1"]:.3f}</td>' for s in sites) + f'<td>{means[i]:.3f}</td></tr>')
    body.append('</table>')
    preview = out / 'current-graph-preview.json'
    if preview.exists():
        data = json.loads(preview.read_text(encoding='utf-8'))
        body.append(f'<p><a href="current-graph-preview.json">当前图只读预览</a>：{data["nodes"]}状态，补齐{len(data["missing_profiles_restored"])}个遗漏结构状态后{data["templates_before"]}组，候选{data["templates_after"]}组；未发布或真实制卡。保留局部变体后的实际重整含代表更新约{data["seconds"]:.2f}秒，不是64页聚类计时。</p>')
    body += ['<p>完整数据同时含完全链接结果和逐页归属。新增三个公开演示网站仅两种粗布局；Reddit及合成控制已有开发暴露。耗时仅编码、矩阵、树和评分，不含采集、制卡或任务执行。模板数是成本代理，没有真实模型费用、卡片质量或Agent成功率结论。轮廓系数不能比较全站只有一类的方案；除完全相同结构或只有一个有效样本外，此选择器不会主动选择一类。</p>',
             '<p><a href="plan.json">运行前计划</a> · <a href="results.json">全部结果/归属</a> · <a href="verification.json">来源校验</a></p></html>']
    (out / 'report.html').write_text(''.join(body), encoding='utf-8')


if __name__ == '__main__':
    main()
