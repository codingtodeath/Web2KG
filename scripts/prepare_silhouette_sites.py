"""Acquire small public static-site corpora; freeze labels before extracting profiles."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main(out):
    import requests
    from bs4 import BeautifulSoup
    from playwright.sync_api import sync_playwright
    from PIL import Image, ImageDraw
    from web2kg.stage1.page_structure import structure_profile
    if (out / 'corpora.json').exists():
        raise FileExistsError('Prepared corpus already exists')
    cache = out / 'http-cache'
    cache.mkdir(exist_ok=True)

    def fetch(url):
        key = hashlib.sha256(url.encode()).hexdigest()
        path = cache / (key + '.body')
        if path.exists():
            return path.read_bytes()
        session = requests.Session()
        session.trust_env = False
        response = session.get(url, timeout=25)
        response.raise_for_status()
        path.write_bytes(response.content)
        save(cache / (key + '.json'), dict(url=url, status=response.status_code,
                                         bytes=len(response.content), sha256=hashlib.sha256(response.content).hexdigest()))
        return response.content

    def links(url, selector):
        soup = BeautifulSoup(fetch(url), 'html.parser')
        return list(dict.fromkeys(urljoin(url, a['href']) for a in soup.select(selector)))

    books = 'https://books.toscrape.com/'
    quotes = 'https://quotes.toscrape.com/'
    shop = 'https://webscraper.io/test-sites/e-commerce/'
    groups = {}
    categories = links(books, '.side_categories a[href]')[2:26]
    products = []
    for i in range(1, 3):
        products += links(books + f'catalogue/page-{i}.html', 'h3 a[href]')
    groups['books'] = [('listing', u) for u in categories] + [('detail', u) for u in products[:24]]
    authors, tags = [], []
    for i in range(1, 5):
        u = quotes + f'page/{i}/'
        authors += links(u, '.quote a[href^="/author/"]')
        tags += links(u, '.quote .tags a[href]')
    authors, tags = list(dict.fromkeys(authors)), list(dict.fromkeys(tags))
    groups['quotes'] = ([('listing', quotes + f'page/{i}/') for i in range(1, 11)]
                        + [('listing', u) for u in tags[:8]] + [('author', u) for u in authors[:18]])
    listing = ([shop + f'static/computers/laptops?page={i}' for i in range(1, 11)]
               + [shop + f'static/computers/tablets?page={i}' for i in range(1, 4)]
               + [shop + 'static/phones/touch', shop + 'allinone', shop + 'allinone/computers/laptops',
                  shop + 'allinone/computers/tablets', shop + 'allinone/phones/touch'])
    detail = []
    for u in listing[:4]:
        detail += links(u, '.product-wrapper a.title[href]')
    groups['webscraper'] = [('listing', u) for u in listing] + [('detail', u) for u in list(dict.fromkeys(detail))[:24]]
    rows = [dict(site=site, id=f'{site}-{i:03}', label=label, url=url)
            for site, items in groups.items() for i, (label, url) in enumerate(items)]
    save(out / 'public-labels-before-extraction.json', dict(
        rule='Coarse reusable layout: category/home/paginated/filter product or quote lists share listing; product detail and author biography separate. URL used only to obtain and prelabel pages, never as clustering feature.',
        rows=rows))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, [r['url'] for r in rows]))
    print('downloaded', len(rows), 'documents', flush=True)
    for name in ['html', 'screenshots', 'review']:
        (out / name).mkdir(exist_ok=True)
    errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1200, 'height': 1000})
        page.route('**/*', lambda route: route.abort())
        client = page.context.new_cdp_session(page)
        for i, row in enumerate(rows):
            soup = BeautifulSoup(fetch(row['url']), 'html.parser')
            for script in soup.select('script, iframe'):
                script.decompose()
            for link in soup.select('link[rel="stylesheet"]'):
                href = urljoin(row['url'], link.get('href', ''))
                # Keep acquisition bounded to the same public host.
                if urlparse(href).hostname == urlparse(row['url']).hostname:
                    try:
                        css = fetch(href).decode('utf-8', errors='replace')
                        style = soup.new_tag('style')
                        style.string = css
                        link.replace_with(style)
                    except requests.RequestException as exc:
                        errors.append(dict(url=href, error=type(exc).__name__, kind='stylesheet'))
            text = str(soup)
            (out / 'html' / (row['id'] + '.html')).write_text(text, encoding='utf-8')
            page.set_content(text, wait_until='domcontentloaded')
            ax = client.send('Accessibility.getFullAXTree')['nodes']
            dom = client.send('DOM.getDocument', {'depth': -1})['root']
            row['profile'] = structure_profile(ax, dom)
            row['screenshot'] = str((out / 'screenshots' / (row['id'] + '.png')).resolve())
            page.screenshot(path=row['screenshot'], full_page=False)
            row['html_sha256'] = hashlib.sha256(text.encode()).hexdigest()
            if i % 20 == 0:
                print('extracted', i + 1, '/', len(rows), flush=True)
        browser.close()
    # All images have visible IDs; unique profiles are used only to cover visual review.
    for site in groups:
        chosen, seen = [], set()
        for row in (r for r in rows if r['site'] == site):
            fingerprint = json.dumps(row['profile']['region_features'], sort_keys=True)
            if fingerprint not in seen:
                chosen.append(row)
                seen.add(fingerprint)
        for start in range(0, len(chosen), 8):
            canvas = Image.new('RGB', (2000, 1100), 'white')
            draw = ImageDraw.Draw(canvas)
            for j, row in enumerate(chosen[start:start+8]):
                x, y = j % 4 * 500, j // 4 * 550
                with Image.open(row['screenshot']) as im:
                    im.thumbnail((495, 505))
                    canvas.paste(im, (x, y+40))
                draw.text((x+5, y+5), row['id']+' | '+row['label'], fill='black')
            canvas.save(out / 'review' / f'{site}-{start//8}.png')
    save(out / 'corpora.json', dict(rows=rows, acquisition='Public server-rendered HTML with same-host CSS; scripts/iframes and remote images/fonts disabled. DOM/AX extracted by local Chromium. No login, model calls or service mutation.', errors=errors))
    print('prepared', {site: len([r for r in rows if r['site']==site]) for site in groups}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    main(parser.parse_args().directory)
