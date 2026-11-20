"""Multi-site silhouette selection, annotation-only evaluation, and stability checks."""
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
# Existing isolated compatibility dependency; never change the shared environment.
RUNTIME = ROOT / 'runs/template-clustering-dev2-20261029-v2/runtime'
sys.path.insert(0, str(RUNTIME))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(rows, seeds):
    from sklearn.metrics import adjusted_rand_score
    from web2kg.stage1.silhouette_selection import prepare, candidates, select
    from check_template_clustering import metrics
    arrays = prepare([r['profile'] for r in rows])
    truth = [r['label'] for r in rows]
    outputs = []
    for method in ('average', 'complete', 'kmeans'):
        for seed in (seeds if method == 'kmeans' else [0]):
            cache = None
            for geometry in ('jaccard', 'euclidean'):
                start = time.perf_counter()
                if cache is None or method != 'kmeans':
                    options = candidates(arrays, method, geometry, seed)
                    fit_seconds = time.perf_counter()-start
                    if method == 'kmeans':
                        cache = (options, fit_seconds)
                else:
                    options, fit_seconds = cache
                start = time.perf_counter()
                result = select(arrays, options, geometry)
                selection_seconds = time.perf_counter()-start
                # Annotations are first consulted after parameter selection.
                result.update(metrics(truth, result['predicted']))
                result.update(method=method, geometry=geometry, seed=seed,
                              preparation_seconds=arrays['seconds'], fit_seconds=fit_seconds,
                              scoring_seconds=selection_seconds,
                              total_seconds=arrays['seconds']+fit_seconds+selection_seconds)
                outputs.append(result)
    return outputs


def render(out, result, corpora):
    e = html.escape
    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>轮廓系数自动选参</title>',
            '<style>body{max-width:1280px;margin:35px auto;padding:0 22px;font:16px/1.65 system-ui;background:#f5f7fa;color:#18314a}table{border-collapse:collapse;background:white;width:100%;margin:20px 0}th,td{padding:8px;border:1px solid #d4dde8;text-align:left}.group{background:white;padding:14px;margin:16px 0}.page{display:inline-block;vertical-align:top;width:225px;margin:6px}.page img{width:220px;height:180px;object-fit:cover;object-position:top}select{font:inherit;padding:7px}small{color:#526274}</style>',
            '<h1>用轮廓系数自动选择模板聚类参数</h1>',
            '<p>dev2 · 离线实验 · 四个真实站点，另列合成控制。参数选择不接收标签、URL、标题或旧模板编号。</p>',
            '<p>主比较：全部方法按现有加权Jaccard距离计算轮廓系数。层次方法遍历所有可实现的树切分；K-means遍历2到可行最大K，每K含20次初始化。K-means仍在归一化欧氏向量上拟合。敏感性比较：三种方法统一在同一欧氏空间聚类并计算轮廓系数。</p>',
            '<p>真实新增页面是公开服务端HTML快照，用本地Chromium提取DOM/AX；保留CSS，关闭脚本、iframe和远程图片/字体，不代表动态网页全部状态。标签提前保存，助手阅图复核。两个商品站及语录站仅有2类粗布局，明显比Reddit简单。跨站点是无标签重新适配，并非固定分类器迁移。</p>',
            '<h2>选参之后才用标签评价</h2><table><tr><th>站点 / n / 标签类数</th><th>距离</th><th>方法</th><th>轮廓</th><th>页面对F1</th><th>ARI</th><th>组数</th><th>选参秒</th></tr>']
    for row in result['summary']:
        body.append(f"<tr><td>{e(row['corpus'])} / {row['n']} / {row['label_count']}</td><td>{row['geometry']}</td><td>{row['method']}</td><td>{row['silhouette']:.3f}</td><td>{row['pair_f1']:.3f}</td><td>{row['ari']:.3f}</td><td>{row['clusters']:.1f}</td><td>{row['total_seconds']:.3f}</td></tr>")
    body += ['</table><p>K-means为5个预定随机种子的均值，不挑最好种子。不同距离的轮廓系数不能直接排大小。选参耗时包含编码、距离矩阵、全部候选拟合与评分，不含采集/标注/制卡；层次方法为一次，K-means为5个种子的均值，不是正式运行时基准。</p>',
             '<h2>稳定性与重复结构</h2><table><tr><th>站点</th><th>距离 / 方法</th><th>80%子样本与全量分组ARI均值</th><th>子样本F1均值</th><th>每种结构仅保留1页的F1</th><th>去重后页数</th></tr>']
    for row in result['robustness_summary']:
        body.append(f"<tr><td>{row['corpus']}</td><td>{row['geometry']} / {row['method']}</td><td>{row['overlap_ari']:.3f}</td><td>{row['pair_f1']:.3f}</td><td>{row['unique_f1']:.3f}</td><td>{row['unique_n']}</td></tr>")
    body += ['</table><p>子样本固定5次随机抽取80%页面，所有方法使用同一批抽样；每次重新无标签选参。去重改变任务分布；相同结构有不同标签时，只保留稳定顺序首项并报告冲突，不能把此敏感性分数当作独立基准。</p>',
             '<h2>全部真实样本分组</h2><select id="site">' + ''.join(f'<option>{e(s)}</option>' for s in result['real_sites']) + '</select><select id="geometry"><option>jaccard</option><option>euclidean</option></select><select id="method"><option>average</option><option>complete</option><option>kmeans</option></select><div id="groups"></div>',
             '<h2>解释限制</h2><p>轮廓系数奖励内部紧凑和组间分离；并不知道模板功能、希望容忍的差异或制卡成本。高分不是泛化证明。Reddit和合成控制已用于前一轮开发；新增站点规模小且为抓取演示网站，不能外推到GitLab等复杂应用。没有Agent成功率、卡片质量或显著性检验。</p>',
             '<p><a href="plan.json">预先计划</a> · <a href="results.json">全部数据</a> · <a href="inputs.json">标签与输入结构</a> · <a href="public-labels-before-extraction.json">提取前标签</a> · <a href="experiment-result.md">实验记录</a> · <a href="https://scikit-learn.org/1.2/modules/generated/sklearn.metrics.silhouette_score.html">轮廓系数定义</a></p>']
    display = {}
    for site in result['real_sites']:
        display[site] = dict(rows=[dict(id=r['id'], label=r['label'], image=os.path.relpath(r['screenshot'], out).replace('\\','/')) for r in corpora[site]], predictions={})
        for row in result['full'][site]:
            if row['seed'] == 0:
                display[site]['predictions'][row['geometry']+'/'+row['method']] = row['predicted']
    payload = json.dumps(display, ensure_ascii=False).replace('<','\\u003c')
    body.append('<script>const data='+payload+";function render(){const d=data[document.getElementById('site').value];const key=document.getElementById('geometry').value+'/'+document.getElementById('method').value;const box=document.getElementById('groups');box.replaceChildren();const groups={};d.predictions[key].forEach((g,i)=>(groups[g]??=[]).push(d.rows[i]));Object.entries(groups).forEach(([id,rows])=>{const div=document.createElement('div');div.className='group';const h=document.createElement('h3');h.textContent='组 '+id+' / '+rows.length+' 页';div.append(h);rows.forEach(r=>{const card=document.createElement('div');card.className='page';const a=document.createElement('a');a.href=r.image;a.target='_blank';const img=document.createElement('img');img.src=r.image;img.loading='lazy';a.append(img);const p=document.createElement('div');p.textContent=r.id+' / '+r.label;card.append(a,p);div.append(card)});box.append(div)})}['site','geometry','method'].forEach(id=>document.getElementById(id).onchange=render);render();</script></html>")
    (out/'report.html').write_text('\n'.join(body), encoding='utf-8')


def run(out):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    from threadpoolctl import threadpool_info, threadpool_limits
    if (out/'results.json').exists():
        raise FileExistsError('Completed results already exist')
    prior = ROOT/'runs/template-clustering-dev2-20261029-v2'
    graph = ROOT/'runs/reddit-hard-state-v2-20261023/templates/state-model/templates.json'
    profiles = read(graph)['profiles']
    corpora = {'reddit': [dict(id=r['node'], label=r['label'], profile=profiles[r['node']], screenshot=r['screenshot']) for r in read(prior/'labels.json')['samples']]}
    for row in read(out/'corpora.json')['rows']:
        corpora.setdefault(row['site'], []).append(row)
    real_sites = list(corpora)
    for split in ['evaluation', 'imbalanced']:
        corpora['html_'+split] = [r for r in read(prior/'synthetic.json')['rows'] if r['split']==split]
    save(out/'inputs.json', corpora)
    protected = [graph, prior/'labels.json', prior/'synthetic.json', ROOT/'runs/active-graph.json',
                 ROOT/'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt']
    guard = {str(p): sha(p) for p in protected}
    source = [Path(__file__), ROOT/'web2kg/stage1/silhouette_selection.py', ROOT/'web2kg/stage1/template_clustering.py', ROOT/'web2kg/stage1/page_structure.py']
    save(out/'source-hashes.json', {str(p): sha(p) for p in source})
    full, robustness, unique_results, unique_info = {}, {}, {}, {}
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        # Warm up K-means first so any delayed OpenMP runtime is recorded.
        evaluate(corpora['reddit'][:8], [0])
        pools_before = threadpool_info()
        for site, rows in corpora.items():
            full[site] = evaluate(rows, [0,1,2,3,4])
            print('full completed', site, len(rows), flush=True)
            save(out/'full-progress.json', full)
            robustness[site] = []
            for seed in range(5):
                indices = sorted(np.random.RandomState(710+seed).choice(len(rows), int(.8*len(rows)), replace=False).tolist())
                partial = evaluate([rows[i] for i in indices], [0])
                for r in partial:
                    base = next(b for b in full[site] if b['method']==r['method'] and b['geometry']==r['geometry'] and b['seed']==0)
                    r['overlap_ari'] = float(adjusted_rand_score([base['predicted'][i] for i in indices], r['predicted']))
                    r['sample_seed'] = seed
                robustness[site].extend(partial)
            uniq = {}
            for row in rows:
                key = json.dumps(row['profile']['region_features'], sort_keys=True)
                uniq.setdefault(key, []).append(row)
            unique_info[site] = dict(n=len(uniq), conflicting_profiles=sum(len({r['label'] for r in group})>1 for group in uniq.values()))
            unique_results[site] = evaluate([group[0] for group in uniq.values()], [0])
            print('robustness completed', site, 'unique', len(uniq), flush=True)
        pools_after = threadpool_info()
    summaries, robustness_summary = [], []
    for site, rows in corpora.items():
        for geometry in ('jaccard','euclidean'):
            for method in ('average','complete','kmeans'):
                matched = [r for r in full[site] if r['geometry']==geometry and r['method']==method]
                summary = dict(corpus=site, n=len(rows), label_count=len({r['label'] for r in rows}), geometry=geometry, method=method)
                for key in ['silhouette','pair_f1','ari','clusters','false_merges','false_splits','total_seconds']:
                    values = [r[key] for r in matched if r[key] is not None]
                    summary[key] = statistics.mean(values) if values else float('nan')
                summary['f1_range'] = [min(r['pair_f1'] for r in matched), max(r['pair_f1'] for r in matched)]
                summaries.append(summary)
                partial = [r for r in robustness[site] if r['geometry']==geometry and r['method']==method]
                uniq = next(r for r in unique_results[site] if r['geometry']==geometry and r['method']==method)
                robustness_summary.append(dict(corpus=site, geometry=geometry, method=method,
                    overlap_ari=statistics.mean(r['overlap_ari'] for r in partial),
                    pair_f1=statistics.mean(r['pair_f1'] for r in partial), unique_f1=uniq['pair_f1'], unique_n=unique_info[site]['n']))
    errors = [path for path,value in guard.items() if sha(path)!=value]
    assert not errors, errors
    result = dict(status='completed', real_sites=real_sites, full=full, robustness=robustness,
                  unique_results=unique_results, unique_info=unique_info, summary=summaries,
                  robustness_summary=robustness_summary, elapsed_seconds=time.perf_counter()-start,
                  pools_before=pools_before, pools_after=pools_after, protected_inputs=guard, guard_errors=errors)
    save(out/'results.json', result)
    render(out, result, corpora)
    print('completed in', round(result['elapsed_seconds'],2), 'seconds', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    run(args.directory)
