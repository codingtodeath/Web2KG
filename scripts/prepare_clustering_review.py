"""Prepare a fixed screenshot sample for annotation BEFORE running any clustering."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/template-clustering-dev2-20261029-v1'
GRAPH = ROOT / 'runs/reddit-hard-state-v2-20261023'
# Coverage selection by visible page type, never by an algorithm's predicted cluster.
SELECTION = {
    'feed': [1, 3, 10, 17, 18, 19, 28, 57, 59, 61, 62, 66],
    'detail': [8, 9, 24, 26, 35, 39, 44, 47, 70, 75, 76, 80, 86, 90, 101, 115],
    'search': [71, 72, 73, 74, 81, 87, 102, 108, 114, 117],
    'submission_form': [68, 69, 88, 89, 93, 94, 95, 100],
    'directory': [0, 6, 13, 31, 32, 33, 64, 65],
    'profile': [14, 23, 37, 49, 54, 99],
    'comment_list': [7, 16, 22, 40],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    graph = json.loads((GRAPH / 'graph.json').read_text(encoding='utf-8'))
    layer = json.loads((GRAPH / 'templates/state-model/templates.json').read_text(encoding='utf-8'))
    unique = {}
    for key, profile in layer['profiles'].items():
        fingerprint = hashlib.sha256(json.dumps(profile['region_features'], sort_keys=True).encode()).hexdigest()
        unique.setdefault(fingerprint, []).append(key)
    representatives = [keys[0] for keys in unique.values()]
    rows = []
    for coverage, indices in SELECTION.items():
        for index in indices:
            key = representatives[index]
            node = graph['nodes'][key]
            screenshot = GRAPH / 'snapshots' / key / 'page.png'
            rows.append(dict(index=index, node=key, coverage=coverage, title=node['title'], url=node['url'],
                             screenshot=str(screenshot), screenshot_sha256=sha(screenshot)))
    for offset in range(0, len(rows), 8):
        canvas = Image.new('RGB', (2000, 1500), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(rows[offset:offset + 8]):
            x, y = (j % 4) * 500, (j // 4) * 750
            with Image.open(row['screenshot']) as original:
                shot = original.convert('RGB')
                shot.thumbnail((495, 700))
                canvas.paste(shot, (x, y + 45))
            draw.text((x + 5, y + 5), f"{row['index']} | {row['coverage']} | {row['node'][:8]}", fill='black')
            draw.text((x + 5, y + 23), row['title'][:65], fill='black')
        canvas.save(OUT / f'review-{offset // 8 + 1}.png')
    (OUT / 'review-candidates.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    guards = [GRAPH / 'graph.json', GRAPH / 'templates/active.json', GRAPH / 'templates/state-model/templates.json',
              ROOT / 'runs/active-graph.json', ROOT / 'third_party/webnavigator/webNavigator/prompts/navigation_specifications/type.txt']
    (OUT / 'source-hashes.json').write_text(json.dumps({str(p): sha(p) for p in guards}, indent=2), encoding='utf-8')
    print(json.dumps({'samples': len(rows), 'sheets': (len(rows) + 7) // 8, 'directory': str(OUT)}))


if __name__ == '__main__':
    main()
