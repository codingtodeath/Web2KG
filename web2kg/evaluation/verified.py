"""Isolated local Reddit transport and offline WebArena-Verified scoring."""
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from ..common.browser import Browser
from ..common.safety import origin
from ..common.storage import write_json


class VerifiedBrowser(Browser):
    """Keep graph URLs visible, but forward every site request to an isolated copy."""
    def __init__(self, *args, backend, headers=None, **kwargs):
        self.backend, self.forward_headers = backend.rstrip('/'), headers or {}
        if urlsplit(backend).hostname not in {'127.0.0.1', 'localhost'}:
            raise ValueError('Evaluation backend must be local')
        super().__init__(*args, **kwargs)

    def context_options(self):
        return {**super().context_options(), 'record_har_path': str(self.log.directory / 'network-transport.har'),
                'record_har_mode': 'full', 'record_har_content': 'embed'}

    def _route(self, route):
        request = route.request
        if origin(request.url) != origin(self.start_url):
            route.continue_()
            return
        parts = urlsplit(request.url)
        target = self.backend + (parts.path or '/') + ('?' + parts.query if parts.query else '')
        headers = {k: v for k, v in request.all_headers().items() if k.lower() not in {'host', 'content-length'}}
        # Credentials go only to the local backend, never to Actor input or event output.
        response = route.fetch(url=target, headers={**headers, **self.forward_headers}, max_redirects=0,
                               timeout=20000)
        route.fulfill(response=response)

    def __exit__(self, *args):
        try:
            if self.context:
                self.context.close()
        finally:
            super().__exit__(*args)
        raw = json.loads((self.log.directory / 'network-transport.har').read_text(encoding='utf-8'))
        entries = raw['log']['entries']
        # route.fetch also records internal relay requests. Keep the unchanged browser-facing
        # entries for scoring, and preserve the complete raw transport trace separately.
        raw['log']['entries'] = [e for e in entries if origin(e['request']['url']) == origin(self.start_url)]
        write_json(self.log.directory / 'network.har', raw)
        self.log.emit('har_exported', browser_entries=len(raw['log']['entries']), transport_entries=len(entries))


FINAL_INSTRUCTION = '''This is a local benchmark. Perform only the changes requested in the task.
When action=stop, answer must be a JSON-encoded object with these public protocol fields:
task_type: RETRIEVE, MUTATE or NAVIGATE; status: SUCCESS, ACTION_NOT_ALLOWED_ERROR,
NOT_FOUND_ERROR, PERMISSION_DENIED_ERROR, DATA_VALIDATION_ERROR or UNKNOWN_ERROR;
retrieved_data: a JSON array for retrieval tasks, otherwise null; error_details: null on SUCCESS,
otherwise a short explanation. Preserve the task's requested data keys and types.
Only claim SUCCESS if current observations support it. Do not infer global absence from one empty search.
For other actions use an empty answer. No evaluator reference answers are supplied.'''


def reset_copy(name, image_id, backend, control_port, logical_origin, owner):
    if name != 'web2kg-template-eval' or origin(backend) != origin('http://127.0.0.1:10599') or control_port != 10598:
        raise ValueError('Only the dedicated template evaluation copy may be reset')
    current = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
    if current.returncode == 0:
        data = json.loads(current.stdout)[0]
        if (data['Config'].get('Labels') or {}).get('web2kg.template_eval') != owner:
            raise ValueError('Container belongs to another run; refusing reset')
        subprocess.run(['docker', 'rm', '-f', name], check=True, capture_output=True, timeout=60)
    subprocess.run(['docker', 'run', '--pull=never', '-d', '--name', name,
        '--label', 'web2kg.template_eval=' + owner, '-p', '127.0.0.1:10599:80',
        '-p', '127.0.0.1:10598:8877', image_id], check=True, capture_output=True, timeout=60)
    session = requests.Session(); session.trust_env = False
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            if session.get(backend, timeout=2).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        raise RuntimeError('Isolated Reddit startup timeout')
    subprocess.run(['docker', 'exec', name, 'env-ctrl', 'init', '--base-url', logical_origin],
                   check=True, capture_output=True, timeout=90)
    return subprocess.check_output(['docker', 'inspect', name, '--format', '{{.Id}}'], text=True).strip()


def score(verified_root, batch, task_id, logical_origin, site='reddit'):
    if site not in {'reddit', 'gitlab'}:
        raise ValueError('Unsupported registered single-site evaluator')
    config = batch / 'evaluator-config.json'
    write_json(config, {'environments': {'__' + site.upper() + '__': {'urls': [logical_origin], 'active_url_idx': 0}}})
    result = subprocess.run([str(Path(verified_root) / '.venv/Scripts/webarena-verified.exe'),
        'eval-tasks', '--task-ids', str(task_id), '--output-dir', str(batch), '--config', str(config)],
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120,
        env={**os.environ, 'PYTHONUTF8': '1'})
    folder = batch / str(task_id)
    (folder / 'official-eval.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    path = folder / 'eval_result.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'score': None, 'status': 'evaluator_error', 'returncode': result.returncode}
