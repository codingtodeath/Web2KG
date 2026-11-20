"""Local, pre-labelled comparison of average/complete linkage and K-means.

Run --prepare once, then --run. Neither operation publishes a graph or calls an LLM.
"""
import argparse
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


FAMILIES = ['card_list', 'discussion', 'article', 'table', 'editor', 'search_filter', 'preferences', 'confirmation']


def fixture_html(family, variant, seed):
    """Labels are the intended reusable layout, assigned before feature extraction."""
    rng = random.Random(seed * 1000 + variant)
    count = rng.choice([1, 2, 5, 12])
    words = f'Example {seed}-{variant}'
    link = f'<a href="/item/{seed}/{variant}">{words}</a>'
    p = f'<p>{words} paragraph with changing text.</p>'
    button = '<button>Open</button>'
    image = '<img alt="illustration" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7">'
    items = ''.join(f'<li><h2>{link}</h2>{p}<a href="/author">Author</a>{button}</li>' for _ in range(count))
    bodies = {
        'card_list': f'<ul>{items}</ul>',
        'discussion': f'<article><h2>{words}</h2>{p}<form method="post"><textarea aria-label="Reply"></textarea><button>Post</button></form></article>'
                      + ''.join(f'<blockquote>{p}{link}<button>Reply</button></blockquote>' for _ in range(count)),
        'article': f'<article><header><h2>{words}</h2></header>{p * count}<figure>{image}<figcaption>{words}</figcaption></figure><footer>{link}</footer></article>',
        'table': '<table><thead><tr><th>Name</th><th>Amount</th><th>Action</th></tr></thead><tbody>'
                 + ''.join(f'<tr><td>{link}</td><td>{i}</td><td>{button}</td></tr>' for i in range(count)) + '</tbody></table>',
        'editor': f'<form method="post"><label>Title<input value="{words}"></label><label>Body<textarea>{words}</textarea></label><label>Category<select><option>A</option><option>B</option></select></label><button>Save</button></form>',
        'search_filter': f'<form method="get"><label>Query<input type="search" value="{words}"></label><label><input type="checkbox">Active</label><button>Search</button></form><ol>{items}</ol>',
        'preferences': '<form method="post"><fieldset><legend>Display</legend><label><input type="checkbox">Enabled</label><label><input type="radio" name="mode">A</label><label><input type="radio" name="mode">B</label><input type="range" aria-label="Size"></fieldset><button>Save</button></form>',
        'confirmation': f'{p}<form method="post"><button>Confirm</button>{link}</form>',
    }
    body = bodies[family]
    # These changes are deliberately independent of the layout label.
    if rng.random() < .45:
        body += '<div><button>Help</button><a href="/help">Guide</a></div>'
    if rng.random() < .35:
        body = '<p role="status">Status message</p>' + body
    if rng.random() < .35:
        body = f'<div><section>{body}</section></div>'
    sidebar = '<aside><h2>Related</h2>' + link + (p if rng.random() < .5 else button) + '</aside>' if rng.random() < .6 else ''
    nav = '<nav><a href="/">Home</a><a href="/account">Account</a>' + (button if rng.random() < .5 else '') + '</nav>'
    return f'<!doctype html><html><head><title>{words}</title></head><body>{nav}<main><h1>{words}</h1>{body}</main>{sidebar}<footer>{link}</footer></body></html>'


def prepare(out):
    from playwright.sync_api import sync_playwright
    from web2kg.stage1.page_structure import structure_profile
    if (out / 'synthetic.json').exists():
        raise FileExistsError('Synthetic inputs already prepared')
    rows = []
    for split, counts, seed in [('development', [6] * 8, 71), ('evaluation', [8] * 8, 83),
                                ('imbalanced', [24, 16, 12, 8, 6, 4, 2, 1], 97)]:
        for family, count in zip(FAMILIES, counts):
            for i in range(count):
                key = f'{split}-{family}-{i:02}'
                rows.append(dict(id=key, label=family, split=split, html=fixture_html(family, i, seed)))
    save(out / 'synthetic-labels-before-extraction.json', rows)
    start = time.perf_counter()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1000, 'height': 800})
        client = page.context.new_cdp_session(page)
        for i, row in enumerate(rows):
            page.set_content(row['html'])
            ax = client.send('Accessibility.getFullAXTree')['nodes']
            dom = client.send('DOM.getDocument', {'depth': -1})['root']
            row['profile'] = structure_profile(ax, dom)
            if i % 40 == 0:
                print(f'prepared {i + 1}/{len(rows)}', flush=True)
        browser.close()
    save(out / 'synthetic.json', {'seconds': time.perf_counter() - start, 'rows': rows})
    print('preparation completed', flush=True)


def metrics(truth, predicted):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    a, b = np.asarray(truth), np.asarray(predicted)
    i, j = np.triu_indices(len(a), 1)
    same_a, same_b = a[i] == a[j], b[i] == b[j]
    tp = int((same_a & same_b).sum())
    fp = int((~same_a & same_b).sum())
    fn = int((same_a & ~same_b).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return dict(pair_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
                precision=precision, recall=recall, ari=float(adjusted_rand_score(a, b)),
                true_same_pairs=int(same_a.sum()), false_merges=fp, false_splits=fn,
                clusters=len(set(predicted)), labels=len(set(truth)))


def prepare_arrays(rows):
    from web2kg.stage1.template_clustering import encode_profiles, similarity_matrix, kmeans_matrix
    encoded = encode_profiles([r['profile'] for r in rows])
    return encoded, similarity_matrix(encoded), kmeans_matrix(encoded)


def predict(method, arrays, parameter, seed=0):
    from web2kg.stage1.template_clustering import hierarchical_labels, kmeans_labels
    encoded, similarity, x = arrays
    return (kmeans_labels(x, encoded.isolated, int(parameter), seed) if method == 'kmeans'
            else hierarchical_labels(similarity, encoded.isolated, method, parameter))


def evaluate_corpus(name, subsets, plan, out):
    selections, searches, results = {}, {}, []
    arrays = {split: prepare_arrays(rows) for split, rows in subsets.items()}
    development = subsets['development']
    for method in ['average', 'complete', 'kmeans']:
        candidates = plan['k_candidates'] if method == 'kmeans' else plan['thresholds']
        table = []
        for parameter in candidates:
            seeds = [0, 1, 2] if method == 'kmeans' else [0]
            scores = [metrics([r['label'] for r in development], predict(method, arrays['development'], parameter, seed))
                      for seed in seeds]
            table.append(dict(parameter=parameter, f1=statistics.mean(s['pair_f1'] for s in scores),
                              precision=statistics.mean(s['precision'] for s in scores), runs=scores))
        best = min(table, key=lambda r: (-r['f1'], -r['precision'], r['parameter']))
        selections[method] = best['parameter']
        searches[method] = table
    # Save selections BEFORE consulting evaluation labels.
    save(out / f'{name}-development-selection.json', {'selected': selections, 'search': searches})
    for split, rows in subsets.items():
        if split == 'development':
            continue
        for method, parameter in selections.items():
            seeds = plan['evaluation_seeds'] if method == 'kmeans' else [0]
            runs = []
            for seed in seeds:
                predicted = predict(method, arrays[split], parameter, seed).tolist()
                runs.append(dict(seed=seed, predicted=predicted, **metrics([r['label'] for r in rows], predicted)))
            results.append(dict(corpus=name, split=split, method=method, parameter=parameter,
                                n=len(rows), runs=runs,
                                means={key: statistics.mean(run[key] for run in runs) for key in
                                       ['pair_f1', 'precision', 'recall', 'ari', 'false_merges', 'false_splits', 'clusters']},
                                f1_range=[min(run['pair_f1'] for run in runs), max(run['pair_f1'] for run in runs)]))
    return selections, results


def timing(rows, selected, repetitions=5):
    from web2kg.stage1.template_clustering import (encode_profiles, similarity_matrix, kmeans_matrix,
                                                  hierarchical_labels, kmeans_labels)
    methods = ['average', 'complete', 'kmeans']
    results = {m: [] for m in methods}
    for repeat in range(-1, repetitions):
        for method in methods[repeat % 3:] + methods[:repeat % 3]:
            t0 = time.perf_counter()
            encoded = encode_profiles([r['profile'] for r in rows])
            t1 = time.perf_counter()
            data = kmeans_matrix(encoded) if method == 'kmeans' else similarity_matrix(encoded)
            t2 = time.perf_counter()
            labels = (kmeans_labels(data, encoded.isolated, int(selected[method]), 0) if method == 'kmeans'
                      else hierarchical_labels(data, encoded.isolated, method, selected[method]))
            t3 = time.perf_counter()
            if repeat >= 0:
                results[method].append(dict(encoding_ms=(t1-t0)*1000, preparation_ms=(t2-t1)*1000,
                                           clustering_ms=(t3-t2)*1000, total_ms=(t3-t0)*1000,
                                           clusters=len(set(labels)), dimensions=encoded.dimensions))
    return {m: {'runs': runs, 'median': {key: statistics.median(r[key] for r in runs) for key in runs[0]}}
            for m, runs in results.items()}


def render(out, result, corpora, real_rows):
    esc = html.escape
    summary_link = 'final-summary.json' if result.get('timing_source') else 'results.json'
    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>模板聚类比较</title>',
            '<style>body{max-width:1250px;margin:35px auto;padding:0 20px;font:16px/1.65 system-ui;color:#172838;background:#f5f7fa}table{border-collapse:collapse;background:white;width:100%;margin:18px 0}th,td{padding:9px;border:1px solid #d4dce4;text-align:left}h1,h2{line-height:1.3}small{color:#526274}.group{background:white;padding:15px;margin:15px 0;border-radius:10px}.page{display:inline-block;width:230px;vertical-align:top;margin:8px}.page img{width:230px;height:180px;object-fit:cover;object-position:top;border:1px solid #ddd}code{background:#e5ecf2;padding:2px 5px}button,select{font:inherit;padding:8px}</style>',
            '<h1>平均链接、完全链接与 K-means</h1><p>本地开发实验 · 2026-10-29 · dev2 · 无模型调用、无图谱发布</p>',
            '<p>64 个 Reddit 页面由助手先阅图标注（7 种宽松布局类型），32 页选参、32 页评价。另有 185 个预标注 HTML 页面：48 开发、64 评价、73 不平衡压力样本。所有方法只读取无文本结构特征；标签和 URL 不进入聚类。</p>',
            '<p>主要指标为页面对 F1：兼顾误合并和误拆分；ARI 为校正随机一致性的聚类分数，1 最好。K-means 表中为 5 个随机种子的均值，错误对数因此可能为小数。</p>',
            '<h2>预先确定标签后的评价结果</h2><table><tr><th>数据</th><th>方法 / 参数</th><th>页面对 F1</th><th>ARI</th><th>误合并 / 误拆分</th><th>模板数</th></tr>']
    for row in result['quality']:
        s = row['means']
        body.append(f"<tr><td>{row['corpus']} / {row['split']} ({row['n']})</td><td>{row['method']} / {row['parameter']}</td><td>{s['pair_f1']:.3f}</td><td>{s['ari']:.3f}</td><td>{s['false_merges']:.1f} / {s['false_splits']:.1f}</td><td>{s['clusters']:.1f}</td></tr>")
    body += ['</table><h2>耗时（5 轮中位数，毫秒）</h2><p>完整路径包含编码、方法专用准备和聚类；不包含文件读取、浏览器采集、制卡或框架生成。单 CPU 线程；K-means 每轮含 20 次初始化。500/2000 页为从现有资料重采样的规模测试，不是新增网站数据。</p><table><tr><th>规模</th><th>方法</th><th>编码</th><th>距离矩阵 / K-means 准备</th><th>聚类</th><th>总计</th></tr>']
    for name, methods in result['timing'].items():
        for method, data in methods.items():
            s = data['median']
            body.append(f"<tr><td>{name}</td><td>{method}</td>" + ''.join(f'<td>{s[k]:.2f}</td>' for k in ['encoding_ms', 'preparation_ms', 'clustering_ms', 'total_ms']) + '</tr>')
    body += ['</table><h2>真实样本分组浏览</h2><p>按固定 seed=0 展示评价集；点击截图查看原始完整截图。可直接核查误合并和误拆分。</p><select id="method"><option>average</option><option>complete</option><option>kmeans</option></select><div id="groups"></div>',
             '<h2>解释边界</h2><ul><li>真实标签是助手一次阅图判断，不是多人人工共识；样本来自同一已接触网站，关联状态可能跨开发/评价集。</li><li>容忍内容数量、空列表、菜单/回复展开和相近功能；字母索引与行式目录分开。边界样本已在结果产生前标记。</li><li>层次方法使用原加权 Jaccard；K-means 使用同一特征集合的分块归一化欧氏空间。因此比较的是完整分类流程，并非单独隔离聚类器的因果效果。</li><li>三种方法本次均不叠加旧的局部输入变体继承；单独比较结构聚类。缺结构页面保持单例。</li><li>类别数/K、阈值只在开发集选择；评价集不反向调整。没有测 Agent 成功率或制卡质量，模板数只是潜在调用数。</li></ul>',
             f'<p><a href="plan.json">预先保存的计划</a> · <a href="labels.json">截图标签</a> · <a href="{summary_link}">全部指标和计时</a> · <a href="assignments.json">所有分组</a> · <a href="experiment-result.md">实验记录</a></p>',
             '<p>算法来源：<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html">SciPy linkage</a>；<a href="https://scikit-learn.org/stable/modules/clustering.html#k-means">sklearn K-means</a>；<a href="https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation">聚类评价</a>。</p>']
    display_rows = corpora['reddit']['evaluation']
    predictions = {r['method']: r['runs'][0]['predicted'] for r in result['quality'] if r['corpus'] == 'reddit' and r['split'] == 'evaluation'}
    lookup = {r['node']: r for r in real_rows}
    display = [dict(id=r['id'], label=r['label'], index=lookup[r['id']]['index'],
                    image=os.path.relpath(lookup[r['id']]['screenshot'], out).replace('\\', '/')) for r in display_rows]
    payload = json.dumps({'rows': display, 'predictions': predictions}, ensure_ascii=False).replace('<', '\\u003c')
    body.append('<script>const data=' + payload + ";const box=document.getElementById('groups');function render(){box.replaceChildren();const groups={};data.predictions[document.getElementById('method').value].forEach((g,i)=>(groups[g]??=[]).push(data.rows[i]));for(const [id,rows] of Object.entries(groups)){const div=document.createElement('div');div.className='group';const h=document.createElement('h3');h.textContent='组 '+id+' · '+rows.length+' 页';div.append(h);for(const r of rows){const card=document.createElement('div');card.className='page';const a=document.createElement('a');a.href=r.image;a.target='_blank';const img=document.createElement('img');img.src=r.image;img.loading='lazy';a.append(img);const p=document.createElement('div');p.textContent='#'+r.index+' · 标注 '+r.label;card.append(a,p);div.append(card)}box.append(div)}}document.getElementById('method').onchange=render;render();</script></html>")
    (out / 'report.html').write_text('\n'.join(body), encoding='utf-8')


def run(out):
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits
    from web2kg.stage1.page_structure import page_similarity
    from web2kg.stage1.template_clustering import encode_profiles, similarity_matrix
    if (out / 'results.json').exists():
        raise FileExistsError('Never overwrite a completed experiment')
    start = time.perf_counter()
    plan = read(out / 'plan.json')
    assert sha(out / 'labels.json') == plan['labels_sha256']
    for path, expected in read(out / 'source-hashes.json').items():
        assert sha(path) == expected, path
    layer = read(ROOT / 'runs/reddit-hard-state-v2-20261023/templates/state-model/templates.json')
    real = read(out / 'labels.json')['samples']
    synthetic = read(out / 'synthetic.json')
    corpora = {'reddit': {}, 'html': {}}
    for split in ['development', 'evaluation']:
        corpora['reddit'][split] = [dict(id=r['node'], label=r['label'], profile=layer['profiles'][r['node']], boundary_case=r['boundary_case'])
                                   for r in real if r['split'] == split]
    for split in ['development', 'evaluation', 'imbalanced']:
        corpora['html'][split] = [r for r in synthetic['rows'] if r['split'] == split]
    quality, selected = [], {}
    full = [dict(id=key, profile=p) for key, p in sorted(layer['profiles'].items())]
    with threadpool_limits(limits=1):
        runtime = threadpool_info()
        # Exact all-pairs parity check on all 407 source profiles, before any quality claim.
        matrix = similarity_matrix(encode_profiles([r['profile'] for r in full]))
        scalar_times, matrix_times = [], []
        for _ in range(3):
            t = time.perf_counter()
            reference = np.array([[page_similarity(a['profile'], b['profile']) for b in full] for a in full])
            scalar_times.append(time.perf_counter() - t)
            t = time.perf_counter()
            actual = similarity_matrix(encode_profiles([r['profile'] for r in full]))
            matrix_times.append(time.perf_counter() - t)
        error = float(np.abs(matrix - reference).max())
        assert error < 1e-12, error
        print(f'matrix parity passed; max_error={error}', flush=True)
        for name, subsets in corpora.items():
            selected[name], rows = evaluate_corpus(name, subsets, plan, out)
            quality.extend(rows)
            print(f'{name} evaluation completed', flush=True)
        timings = {}
        timings['reddit-407'] = timing(full, selected['reddit'])
        rng = random.Random(20261029)
        for size in [500, 2000]:
            sampled = [rng.choice(full) for _ in range(size)]
            timings[f'resampled-{size}'] = timing(sampled, selected['reddit'])
            print(f'timing {size} completed', flush=True)
        full_arrays = prepare_arrays(full)
        full_predictions = {m: predict(m, full_arrays, p).tolist() for m, p in selected['reddit'].items()}
    assignments = {name: {split: [dict(id=r['id'], label=r['label']) for r in rows] for split, rows in subsets.items()}
                   for name, subsets in corpora.items()}
    assignments['evaluations'] = quality
    assignments['full_graph'] = {'ids': [r['id'] for r in full], 'predictions': full_predictions}
    save(out / 'assignments.json', assignments)
    for path, expected in read(out / 'source-hashes.json').items():
        assert sha(path) == expected, path
    result = dict(status='completed', elapsed_seconds=time.perf_counter()-start, selected=selected,
                  quality=quality, timing=timings, threadpools=runtime, python=sys.version, platform=platform.platform(),
                  versions={k: importlib.metadata.version(k) for k in ['numpy', 'scipy', 'scikit-learn', 'threadpoolctl']},
                  matrix_parity={'profiles': len(full), 'max_error': error, 'scalar_seconds': scalar_times, 'matrix_seconds': matrix_times},
                  input_hashes={p.name: sha(p) for p in [out/'plan.json', out/'labels.json', out/'synthetic.json']},
                  source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'web2kg/stage1/template_clustering.py', ROOT/'web2kg/stage1/page_structure.py']},
                  source_guards_unchanged=True, full_graph_clusters={m: len(set(x)) for m, x in full_predictions.items()},
                  synthetic_collection_seconds=synthetic['seconds'])
    save(out / 'results.json', result)
    render(out, result, corpora, real)
    print(json.dumps({'status': 'completed', 'seconds': result['elapsed_seconds'], 'selected': selected,
                      'report': str(out/'report.html')}, ensure_ascii=False), flush=True)


def retime(out):
    """A separate receipt fixes the initially unbounded, late-loaded Intel OpenMP pool."""
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits
    from web2kg.stage1.page_structure import page_similarity
    from web2kg.stage1.template_clustering import encode_profiles, similarity_matrix
    if (out / 'timing-controlled.json').exists():
        raise FileExistsError('Controlled timing receipt already exists')
    original = read(out / 'results.json')
    layer = read(ROOT / 'runs/reddit-hard-state-v2-20261023/templates/state-model/templates.json')
    full = [dict(id=key, profile=p) for key, p in sorted(layer['profiles'].items())]
    selected = original['selected']['reddit']
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        assert all(p['num_threads'] == 1 for p in pools), pools
        tables = {'reddit-407': timing(full, selected)}
        rng = random.Random(20261029)
        for size in [500, 2000]:
            tables[f'resampled-{size}'] = timing([rng.choice(full) for _ in range(size)], selected)
            print(f'controlled timing {size} completed', flush=True)
        scalar_times, matrix_times = [], []
        for _ in range(3):
            t = time.perf_counter()
            reference = np.array([[page_similarity(a['profile'], b['profile']) for b in full] for a in full])
            scalar_times.append(time.perf_counter() - t)
            t = time.perf_counter()
            actual = similarity_matrix(encode_profiles([r['profile'] for r in full]))
            matrix_times.append(time.perf_counter() - t)
        pools_after = threadpool_info()
        assert all(p['num_threads'] == 1 for p in pools_after), pools_after
    receipt = dict(reason='Initial timing receipt contained a late-loaded Intel OpenMP pool at 16 threads; all pools now confirmed at 1 before/after. Quality results unchanged; only timing repeated.',
                   timing=tables, threadpools=pools, threadpools_after=pools_after,
                   matrix_parity=dict(profiles=len(full), max_error=float(np.abs(actual-reference).max()),
                                      scalar_seconds=scalar_times, matrix_seconds=matrix_times),
                   runner_sha256=sha(__file__), original_results_sha256=sha(out/'results.json'))
    assert receipt['matrix_parity']['max_error'] < 1e-12
    save(out / 'timing-controlled.json', receipt)
    summary = {**original, 'timing': tables, 'threadpools': pools, 'matrix_parity': receipt['matrix_parity'],
               'timing_source': 'timing-controlled.json', 'quality_source': 'results.json'}
    save(out / 'final-summary.json', summary)
    real = read(out / 'labels.json')['samples']
    corpora = {'reddit': {'evaluation': [dict(id=r['node'], label=r['label']) for r in real if r['split']=='evaluation']}}
    render(out, summary, corpora, real)
    print('controlled timing completed; final-summary.json and report.html updated', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--timing', action='store_true')
    args = parser.parse_args()
    out = args.directory.resolve()
    sys.path.insert(0, str(out / 'runtime'))
    # Set BEFORE numeric libraries load; threadpoolctl alone misses a late-loaded runtime.
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    if args.prepare:
        prepare(out)
    elif args.timing:
        retime(out)
    else:
        run(out)


if __name__ == '__main__':
    main()
