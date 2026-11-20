"""Offline leaf-pruning validation: controlled cases, repeated growth, scaling, optional graph preview."""
import argparse
import copy
import hashlib
import html
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.storage import Graph, write_json
from web2kg.common.backbone import shortest_backbone
from web2kg.stage2.retention import RULE, plan_pruning, record_retrieval


def make_graph(directory, members, importance, idle=None, edges=None):
    graph = Graph(directory)
    graph.data.update(root='root', retention={'retrieval_round': 100})
    graph.data['nodes'] = {k: {'id': k, 'last_retrieved_round': 100 - (idle or {}).get(k, 0)}
                           for nodes in members.values() for k in nodes}
    graph.data['edges'] = [{'id': str(i), 'source': a, 'target': b} for i, (a, b) in enumerate(
        edges if edges is not None else [('root', k) for k in graph.nodes if k != 'root'])]
    layer = {'assignments': {k: {'template': t} for t, nodes in members.items() for k in nodes}}
    return graph, layer, {t: {'importance': value} for t, value in importance.items()}


def scan_plan(graph, layer, cards, limit, protected=()):
    """Independent full re-ranking baseline; intentionally recomputes candidates each step."""
    remaining = set(graph.nodes)
    protected = set(protected) | {graph.data['root']}
    removed, evaluations = [], 0
    now = graph.data['retention']['retrieval_round']
    tree = shortest_backbone(graph)['parents']
    while len(remaining) > limit:
        counts = Counter(layer['assignments'][k]['template'] for k in remaining)
        nonleaves = {parent['node'] for child, parent in tree.items()
                     if parent and child in remaining and parent['node'] in remaining}
        candidates = remaining - nonleaves - protected
        if not candidates:
            break
        def score(key):
            template = layer['assignments'][key]['template']
            idle = max(0, now - graph.nodes[key]['last_retrieved_round'])
            return math.log2((cards[template]['importance'] + 1 / counts[template]) / 2) - idle / 20, key
        evaluations += len(candidates)
        key = min(candidates, key=score)
        removed.append(key)
        remaining.remove(key)
    return removed, evaluations


def fingerprint(*objects):
    return hashlib.sha256(json.dumps(objects, sort_keys=True).encode()).hexdigest()


def controlled(directory):
    cases = [
        ('function', {'R': ['root'], 'A': ['low'], 'B': ['high']}, {'R': 1, 'A': .1, 'B': .9}, {}, None, 2, [], ['low']),
        ('structure', {'R': ['root'], 'A': ['a1', 'a2', 'a3'], 'B': ['b1']}, {'R': 1, 'A': .5, 'B': .5}, {}, None, 4, [], ['a1']),
        ('idle', {'R': ['root'], 'A': ['old', 'recent']}, {'R': 1, 'A': .5}, {'old': 40}, None, 2, [], ['old']),
        ('batch_rerank', {'R': ['root'], 'A': ['a1', 'a2', 'a3'], 'B': ['b1', 'b2']}, {'R': 1, 'A': .2, 'B': .1}, {}, None, 4, [], ['a1', 'b1']),
        ('exposed_parent', {'R': ['root'], 'A': ['a', 'parent'], 'B': ['b']}, {'R': 1, 'A': 0, 'B': 1}, {}, [('root', 'parent'), ('parent', 'a'), ('root', 'b')], 2, [], ['a', 'parent']),
        ('new_protection', {'R': ['root'], 'A': ['new', 'old']}, {'R': 1, 'A': .1}, {'new': 80}, None, 2, ['new'], ['old']),
        ('cycle_backbone', {'R': ['root'], 'A': ['a', 'b', 'leaf']}, {'R': 1, 'A': .5}, {}, [('root', 'a'), ('a', 'b'), ('b', 'a'), ('root', 'leaf')], 1, [], ['b', 'a', 'leaf']),
    ]
    results = []
    for name, members, importance, idle, edges, limit, protected, expected in cases:
        graph, layer, cards = make_graph(directory, members, importance, idle, edges)
        before = fingerprint(graph.data, layer, cards)
        plan = plan_pruning(graph, layer, cards, max_nodes=limit, protected=protected)
        actual = [r['node'] for r in plan['removed']]
        assert actual == expected, (name, actual, expected)
        assert before == fingerprint(graph.data, layer, cards)
        results.append({'name': name, 'passed': True, 'expected': expected, 'plan': plan})
    return results


def synthetic(directory, size, seed):
    rng = random.Random(seed)
    members = {'R': ['root'], **{f'T{i:02}': [] for i in range(16)}}
    idle = {}
    for i in range(size - 1):
        key = f'n{i:05}'
        members[f'T{rng.randrange(16):02}'].append(key)
        idle[key] = rng.randrange(101)
    importance = {t: rng.choice([.1, .3, .5, .7, .9]) for t in members}
    return make_graph(directory, members, importance, idle)


def scaling(directory):
    results = []
    for size, limit in ((128, 96), (500, 400), (2000, 1500)):
        args = synthetic(directory, size, 20261029)
        heap_times, scan_times = [], []
        for _ in range(3):
            start = time.perf_counter()
            plan = plan_pruning(*args, max_nodes=limit)
            heap_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            expected, evaluations = scan_plan(*args, limit)
            scan_times.append(time.perf_counter() - start)
            assert expected == [r['node'] for r in plan['removed']]
        results.append({'nodes': size, 'limit': limit, 'removed': len(expected), 'same_order': True,
            'heap_median_ms': statistics.median(heap_times) * 1000,
            'scan_median_ms': statistics.median(scan_times) * 1000,
            'heap_score_evaluations': plan['score_evaluations'], 'scan_score_evaluations': evaluations,
            'heap_samples_seconds': heap_times, 'scan_samples_seconds': scan_times})
    return results


def repeated_growth(directory):
    rng = random.Random(20261029)
    graph, layer, cards = synthetic(directory, 80, 7)
    rows = []
    for task in range(1, 31):
        added = []
        for i in range(rng.randint(2, 8)):
            key = f'new-{task:02}-{i}'
            graph.nodes[key] = {'id': key, 'last_retrieved_round': graph.data['retention']['retrieval_round']}
            layer['assignments'][key] = {'template': f'T{rng.randrange(16):02}'}
            graph.data['edges'].append({'id': key, 'source': 'root', 'target': key})
            added.append(key)
        for _ in range(8):
            record_retrieval(graph, rng.sample(sorted(graph.nodes), 5))
        plan = plan_pruning(graph, layer, cards, max_nodes=80, protected=added)
        expected, _ = scan_plan(graph, layer, cards, 80, added)
        removed = {r['node'] for r in plan['removed']}
        assert expected == [r['node'] for r in plan['removed']]
        assert not removed.intersection(added) and plan['after'] == 80
        for key in removed:
            graph.nodes.pop(key)
            layer['assignments'].pop(key)
        graph.data['edges'] = [e for e in graph.data['edges'] if e['target'] not in removed]
        rows.append({'task': task, 'added': added, 'plan': plan})
    return {'tasks': 30, 'retrieval_rounds': 240, 'limit': 80, 'same_order_all': True,
            'all_boundaries_within_limit': True, 'new_nodes_preserved_all': True,
            'added_total': sum(len(r['added']) for r in rows), 'rows': rows}


def real_preview(directory, limit):
    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    hashes = {}
    graph = Graph(directory)
    graph.data = read(directory / 'graph.json')
    pointer = read(directory / 'templates/active.json')
    layer = read(directory / pointer['directory'] / 'templates.json')
    pointer = read(directory / 'function-cards/active.json')
    data = read(directory / pointer['cards'])
    cards = {k: r['card'] for k, r in data['cards'].items() if r['status'] == 'completed'}
    active = ROOT / 'runs/active-graph.json'
    if active.exists():
        read(active)
    start = time.perf_counter()
    plan = plan_pruning(graph, layer, cards, max_nodes=limit)
    seconds = time.perf_counter() - start
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == value for p, value in hashes.items())
    return {'source': str(directory), 'input_hashes': hashes, 'source_unchanged': True,
            'missing_card_importance': sum('importance' not in r for r in cards.values()),
            'missing_idle_history': sum('last_retrieved_round' not in n for n in graph.nodes.values()),
            'seconds': seconds, 'plan': plan,
            'limitation': 'Read-only capacity stress preview. Missing importance uses 0.5; unknown idle starts now. No task outcomes used.'}


def report(data):
    rows = ''.join(f'<tr><td>{html.escape(r["name"])}</td><td>{r["plan"]["before"]} → {r["plan"]["after"]}</td>'
                   f'<td>{html.escape(" → ".join(r["expected"]))}</td><td>{html.escape(r["plan"]["status"])}</td></tr>'
                   for r in data['controlled'])
    timings = ''.join(f'<tr><td>{r["nodes"]} → {r["limit"]}</td><td>{r["heap_median_ms"]:.3f}</td>'
                      f'<td>{r["scan_median_ms"]:.3f}</td><td>{r["heap_score_evaluations"]} / {r["scan_score_evaluations"]}</td></tr>'
                      for r in data['scaling'])
    real = data.get('real_preview')
    extra = '' if real is None else (f'<h2>当前图只读预览</h2><p>{real["plan"]["before"]} → {real["plan"]["after"]}，'
        f'目标 {real["plan"]["limit"]}，剩余超额 {real["plan"]["remaining_excess"]}。'
        f'未修改源文件；{real["missing_card_importance"]} 张卡缺重要性，使用中性值。没有历史闲置记录的节点从本轮开始计数。</p>')
    return f'''<!doctype html><meta charset="utf-8"><title>叶子剪枝离线验证</title>
<style>body{{max-width:960px;margin:40px auto;font:16px/1.7 system-ui;color:#17202a}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:8px 14px;border-bottom:1px solid #ddd;text-align:left}}code{{background:#eee;padding:3px}}</style>
<h1>叶子剪枝离线验证 · dev2</h1><p>算法 {RULE}；种子 20261029。没有浏览器任务、外部模型调用或正式评分。</p>
<p>保留分 = (功能重要性 + 1 / 当前模板成员数) / 2 × 2<sup>−闲置检索轮数 / 20</sup>。每步选择最低分叶子，随后更新模板人数与新叶子；同分按节点ID。</p>
<h2>受控案例</h2><table><tr><th>案例</th><th>节点数</th><th>实际删除顺序（符合预期）</th><th>状态</th></tr>{rows}</table>
<p>batch_rerank 的初始前两名为 a1、a2，但删掉 a1 后 A 模板变稀缺，下一名变为 b1。静态排序后直接截断会得到不同结果。</p>
<h2>30轮连续增图模拟</h2><p>容量80；每题新增2–8页，共新增{data['growth']['added_total']}页；240轮模拟检索。30次收尾都回到80节点，本题新页均保留，顺序均与全扫描参考实现一致。该模拟采用星形图，不代表真实网站的环路分布。</p>
<h2>规模与耗时</h2><table><tr><th>裁剪规模</th><th>动态队列 / ms</th><th>全扫描 / ms</th><th>评分次数（队列 / 扫描）</th></tr>{timings}</table>
<p>每种规模运行3次取中位数，动态队列包含完整初始排序与逐步审计；只计规划，不含文件清理、索引、制卡。所有删除顺序完全一致。</p>{extra}
<p>这些检查验证算法行为与本地开销，不证明任务成功率提升，也不证明权重或半衰期最优。参数未用旧评测答案调优。</p><p><a href="audit.json">完整参数、分数、逐步顺序与耗时记录</a></p>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--graph', type=Path)
    parser.add_argument('--preview-limit', type=int, default=350)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a new experiment directory')
    args.out.mkdir(parents=True)
    data = {'rule': RULE, 'seed': 20261029, 'model_calls': 0, 'controlled': controlled(args.out),
            'scaling': scaling(args.out), 'growth': repeated_growth(args.out)}
    if args.graph:
        data['real_preview'] = real_preview(args.graph.resolve(), args.preview_limit)
    write_json(args.out / 'audit.json', data)
    (args.out / 'report.html').write_text(report(data), encoding='utf-8')
    print(json.dumps({'controlled_passed': len(data['controlled']), 'growth_tasks': 30,
        'growth_added': data['growth']['added_total'], 'scaling': data['scaling'],
        'real_preview': {k: data.get('real_preview', {}).get('plan', {}).get(k)
                         for k in ('before', 'after', 'remaining_excess', 'status')},
        'report': str((args.out / 'report.html').resolve())}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
