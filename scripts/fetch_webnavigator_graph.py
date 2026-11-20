"""Fetch published graph data, with immutable revision and per-file digest receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests


def fetch(site, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    host, repo = 'https://hf-mirror.com', 'Jimzhang324/webNavigator'
    metadata = session.get(f'{host}/api/datasets/{repo}', timeout=(10, 30))
    metadata.raise_for_status()
    revision = metadata.json()['sha']
    prefix = {'gitlab': 'webarena/webNodes/run_id-20251026-190417-gitlab'}[site]
    url = f'{host}/api/datasets/{repo}/tree/{revision}/{prefix}?recursive=true&limit=1000'
    rows = []
    while url:
        response = session.get(url, timeout=(10, 30))
        response.raise_for_status()
        rows.extend(r for r in response.json() if r['type'] == 'file')
        url = response.links.get('next', {}).get('url')
    receipt = directory / 'download-manifest.json'
    expected = {'repository': repo, 'revision': revision, 'prefix': prefix, 'files': rows}
    if receipt.exists() and json.loads(receipt.read_text('utf-8')) != expected:
        raise ValueError('Existing download receipt belongs to another revision')
    receipt.write_text(json.dumps(expected, indent=2), encoding='utf-8')

    def one(row):
        relative = Path(row['path']).relative_to(prefix)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        oid = row.get('lfs', {}).get('oid', '').removeprefix('sha256:')
        def valid():
            if not target.exists() or target.stat().st_size != row['size']:
                return False
            with target.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256' if oid else 'sha1').hexdigest()
            if oid:
                return actual == oid
            data = target.read_bytes()
            return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == row['oid']
        if valid():
            return
        failure = None
        for attempt in range(6):
            try:
                with requests.Session() as http:
                    http.trust_env = False
                    with http.get(f'{host}/datasets/{repo}/resolve/{revision}/{row["path"]}',
                                  stream=True, timeout=(10, 90)) as response:
                        response.raise_for_status()
                        with target.with_suffix(target.suffix + '.partial').open('wb') as output:
                            for chunk in response.iter_content(1024 * 1024):
                                output.write(chunk)
                target.with_suffix(target.suffix + '.partial').replace(target)
                if not valid():
                    raise ValueError('Published graph file hash mismatch: ' + str(relative))
                return
            except (requests.RequestException, ValueError) as error:
                failure = error
                time.sleep(min(30, 2 ** (attempt + 1)))
        raise failure

    with ThreadPoolExecutor(max_workers=2) as pool:
        for number, _ in enumerate(pool.map(one, rows), 1):
            if number % 50 == 0 or number == len(rows):
                print(f'{number}/{len(rows)} graph files verified', flush=True)
    complete = {'revision': revision, 'files': len(rows), 'bytes': sum(r['size'] for r in rows), 'verified': True}
    (directory / 'download-complete.json').write_text(json.dumps(complete, indent=2), encoding='utf-8')
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', choices=['gitlab'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    fetch(args.site, args.out)
