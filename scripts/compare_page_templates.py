"""Offline development comparison: synthetic calibration first, saved graph second."""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from playwright.sync_api import sync_playwright
from web2kg.common.storage import Graph, write_json
from web2kg.stage1.templates import page_similarity, structure_profile
from scripts.build_page_templates import run


def calibration_cases():
    def listing(n, tag='ul', item='li', word='Item'):
        return '<main><h1>Catalog</h1><' + tag + '>' + ''.join(
            f'<{item}><h2><a href="/{i}">{word} {i}</a></h2><p>Text</p>'
            f'<button>Open</button></{item}>' for i in range(n)) + f'</{tag}></main>'
    simple = '<main><h1>Detail</h1><article><p>Text</p><button>Open</button></article></main>'
    form = '<main><form><label>Field<input></label><button>Save</button></form></main>'
    empty = '<main><h1>Empty</h1><p>No items</p></main>'
    table = '<main><h1>Table</h1><table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td><input></td></tr></table></main>'
    return [
        ('repeat_count', True, listing(1), listing(8)),
        ('entity_text', True, listing(3), listing(3, word='Product')),
        ('transparent_wrappers', True, listing(3), '<div><div>' + listing(3) + '</div></div>'),
        ('list_div_equivalence', True, listing(3), listing(5, 'div', 'div')),
        ('input_values', True, form, form.replace('<input>', '<input value="changed">')),
        ('sidebar_change', True, listing(3) + '<aside><a href="/a">Help</a></aside>',
         listing(3) + '<aside><form><input><button>Search</button></form></aside>'),
        ('aux_region_change', True, simple + '<nav><a href="/a">Index</a></nav>',
         simple + '<aside><button>Help</button></aside>'),
        ('list_vs_form', False, listing(3), form),
        ('list_vs_detail', False, listing(3), simple),
        ('detail_vs_form', False, simple, form),
        ('list_vs_flat_controls', False, listing(3), '<main><h1>Catalog</h1><a href="/a">A</a><p>Text</p><button>Open</button></main>'),
        ('form_field_type', False, form, form.replace('<input>', '<input type="checkbox">')),
        ('table_vs_form', False, table, form),
        ('empty_vs_list', False, empty, listing(3)),
        ('parent_child_layout', False,
         '<main><section><h2>A</h2><input><a href="/a">A</a></section><section><h2>B</h2><input><button>Open</button></section></main>',
         '<main><section><h2>A</h2><input><a href="/a">A</a><button>Open</button></section><section><h2>B</h2><input></section></main>'),
        # Semantic distinctions deliberately outside the name-free structure criterion.
        ('ambiguous_empty_purpose', None, empty, empty.replace('Empty', 'Messages')),
        ('ambiguous_form_purpose', None, form, form.replace('Field', 'Quantity').replace('Save', 'Delete')),
    ]


def calibrate(out):
    cases = calibration_cases()
    grid = [.70, .75, .80, .85, .90, .95]
    write_json(out / 'calibration_protocol.json', {
        'scope': 'Synthetic development only; not independent or cross-site validation',
        'criterion': 'Same main structural family, ignoring names, values, counts and auxiliary changes',
        'threshold_grid': grid, 'selection': 'Fewest errors; tie: highest threshold',
        'cases': [{'id': key, 'same_structure': expected, 'left_html': left, 'right_html': right}
                  for key, expected, left, right in cases]})
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for key, expected, left, right in cases:
            profiles = []
            for content in (left, right):
                page.set_content(content)
                cdp = page.context.new_cdp_session(page)
                profiles.append(structure_profile(cdp.send('Accessibility.getFullAXTree')['nodes'],
                                                  cdp.send('DOM.getDocument', {'depth': -1})['root']))
                cdp.detach()
            rows.append({'id': key, 'expected': expected, 'score': page_similarity(*profiles)})
        browser.close()
    sweep = []
    for threshold in grid:
        scored = [r for r in rows if r['expected'] is not None]
        fp = sum(not r['expected'] and r['score'] >= threshold for r in scored)
        fn = sum(r['expected'] and r['score'] < threshold for r in scored)
        sweep.append({'threshold': threshold, 'false_merges': fp, 'false_splits': fn,
                      'errors': fp + fn, 'pairs': len(scored)})
    chosen = min(sweep, key=lambda r: (r['errors'], -r['threshold']))['threshold']
    # Persist the choice before inspecting the real graph's candidate results.
    write_json(out / 'calibration.json', {'cases': rows, 'sweep': sweep, 'selected_threshold': chosen})
    return chosen, rows, sweep


def pairs(layer):
    return {tuple(sorted(pair)) for g in layer['templates']
            for pair in itertools.combinations(g['members'], 2)}


def compare(directory, out):
    if out.exists():
        raise ValueError('Use a new comparison directory; preserve prior analyses')
    # Hash existing source assets, templates, cards and viewer, never dump their content.
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in directory.rglob('*') if p.is_file()}
    threshold, cases, calibration = calibrate(out)
    graph = Graph.load(directory)
    active = json.loads((directory / 'templates/active.json').read_text(encoding='utf-8'))
    old = json.loads((directory / active['directory'] / 'templates.json').read_text(encoding='utf-8'))
    old_pairs = pairs(old)
    results, layers = [], {}
    for value in [.75, .80, .85, .90, .95]:
        layer = run(directory, threshold=value)
        layers[value] = layer
        current = pairs(layer)
        results.append({'threshold': value, **layer['summary'],
                        'newly_joined_pairs': len(current - old_pairs),
                        'newly_split_pairs': len(old_pairs - current)})
    chosen = layers.get(threshold) or run(directory, threshold=threshold)
    changes = []
    new_pairs = pairs(chosen)
    for kind, changed in [('joined', new_pairs - old_pairs), ('split', old_pairs - new_pairs)]:
        for a, b in sorted(changed):
            changes.append({'change': kind, 'left': a, 'right': b,
                'left_url': graph.nodes[a]['url'], 'right_url': graph.nodes[b]['url'],
                'page_similarity': page_similarity(chosen['profiles'][a], chosen['profiles'][b])})
    audit = {'protected_files': len(protected), 'unchanged': all(
        p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in protected.items())}
    if not audit['unchanged']:
        raise RuntimeError('An existing source asset changed during comparison')
    report = {'scope': 'Offline development comparison, no API or task runs',
              'selected_threshold': threshold, 'old_layer': active['directory'],
              'old_summary': old['summary'], 'sweep': results, 'changed_pairs': changes, 'audit': audit}
    write_json(out / 'comparison.json', report)
    lines = ['# 页面门槛离线比较', '', '父/入边/页面权重 0.5/0.5/1；只有页面分数决定是否可合并，加权总分只排序。',
             '', f'合成开发样例先选择阈值 **{threshold}**，之后才运行 Reddit 图。不是跨站验证。', '',
             '|页面阈值|模板数|单节点模板|新合并页面对|新拆分页面对|', '|---|---:|---:|---:|---:|']
    for row in results:
        lines.append(f"|{row['threshold']}|{row['templates']}|{row['singleton_templates']}|{row['newly_joined_pairs']}|{row['newly_split_pairs']}|")
    lines += ['', '## 合成对照', '', '|样例|预期同结构|页面分数|', '|---|---|---:|']
    lines += [f"|{r['id']}|{r['expected']}|{r['score']:.4f}|" for r in cases]
    lines += ['', '同名不同功能、无内容页面等纯结构无法辨别的情况不计校准准确率，单独保留。',
              '页面对变化与模板数量都不代表准确率；需核查成员及截图。',
              f"旧资产核对：{audit['protected_files']} 文件不变。旧活动层及卡片不切换，不生成付费卡片。", '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'selected_threshold': threshold, 'sweep': results, 'audit': audit}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    compare(args.directory.resolve(), args.out.resolve())
