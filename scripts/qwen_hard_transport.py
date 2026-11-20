"""Chat transport for the separately registered Qwen baseline; pinned Agent unchanged."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests
from dotenv import dotenv_values


def daily_exhausted(status, data):
    error = data.get('error', {})
    message = str(error.get('message', '')).lower() if isinstance(error, dict) else ''
    return status == 429 and ('free-models-per-day' in message or
                             ('daily' in message and ('limit' in message or 'quota' in message)))


def free_unavailable(status, data):
    error = data.get('error', {})
    message = str(error.get('message', '')).lower() if isinstance(error, dict) else ''
    return status == 404 and ('unavailable for free' in message or 'no endpoints found' in message)


def chat_request(request):
    messages = [{'role': 'system', 'content': request['instructions']}]
    for message in request['input']:
        content = []
        for part in message['content']:
            if part['type'] == 'input_text':
                content.append({'type': 'text', 'text': part['text']})
            elif part['type'] == 'input_image':
                content.append({'type': 'image_url', 'image_url': {'url': part['image_url']}})
            else:
                raise ValueError('Unsupported input part')
        messages.append({'role': message['role'], 'content': content})
    return {'model': request['model'], 'messages': messages,
            'max_tokens': request['max_output_tokens'], 'stream': False}


class ChatSession:
    def __init__(self, client):
        self.client = client
        self.http = requests.Session()
        self.http.trust_env = False

    def post(self, url, *, json, headers, **kwargs):
        client = self.client
        request = chat_request(json)
        folder = client.log.directory / 'llm' / client.last_attempt_id
        saved = copy.deepcopy(request)
        for message in saved['messages']:
            if isinstance(message['content'], list):
                for part in message['content']:
                    if part['type'] == 'image_url':
                        part['image_url'] = {'sha256_of_data_url': hashlib.sha256(
                            part['image_url']['url'].encode()).hexdigest()}
        client.write_json(folder / 'chat-request.json', saved)
        client.log.emit('provider_request', request_id=client.last_attempt_id,
                        provider=client.provider, model=client.model,
                        endpoint=client.base + '/chat/completions',
                        request_sha256=hashlib.sha256(__import__('json').dumps(request, sort_keys=True).encode()).hexdigest())
        response = self.http.post(client.base + '/chat/completions', json=request, headers=headers, **kwargs)
        data = response.json()
        client.last_quota = None
        if client.provider == 'openrouter':
            if daily_exhausted(response.status_code, data):
                client.last_quota = 'daily_quota'
            elif free_unavailable(response.status_code, data):
                client.last_quota = 'free_unavailable'
        if not response.ok:
            return response
        client.write_json(folder / 'chat-response.json', data)
        choices = data.get('choices', [])
        choice = choices[0] if len(choices) == 1 else {}
        content = choice.get('message', {}).get('content')
        usage = data.get('usage') or {}
        normalized = {**usage}
        if 'prompt_tokens' in usage:
            normalized['input_tokens'] = usage['prompt_tokens']
        if 'completion_tokens' in usage:
            normalized['output_tokens'] = usage['completion_tokens']
        converted = {'id': data.get('id'), 'model': data.get('model'), 'usage': normalized,
                     'status': 'completed' if choice.get('finish_reason') == 'stop' else 'incomplete',
                     'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                                 'content': [{'type': 'output_text', 'text': content or ''}]}]}
        response._content = __import__('json').dumps(converted).encode()
        return response


def install(env_path, routing_path):
    from web2kg.evaluation import webnavigator
    from web2kg.common.responses import ResponsesError
    from web2kg.common.storage import write_json
    original = webnavigator.OriginalResponses
    credentials = dotenv_values(env_path)
    for prefix in ('OPENROUTER', 'BAILIAN'):
        if not all(credentials.get(prefix + '_' + key) for key in ('API_KEY', 'BASE_URL', 'MODEL')):
            raise ValueError('Missing provider configuration: ' + prefix)

    class QwenResponses(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.write_json = write_json
            self.session = ChatSession(self)
            self.last_quota = None

        def choose(self):
            path = Path(routing_path)
            state = json.loads(path.read_text('utf-8')) if path.exists() else {}
            today = datetime.now(timezone.utc).date().isoformat()
            if state.get('reason') == 'daily_quota' and state.get('utc_day') != today:
                state = {}
            if state.get('reason') == 'free_unavailable' and time.time() - state.get('checked_at', 0) >= 1800:
                try:
                    response = self.session.http.get(credentials['OPENROUTER_BASE_URL'].rstrip('/') + '/models', timeout=(10, 30))
                    if response.ok:
                        available = any(x['id'] == credentials['OPENROUTER_MODEL'] for x in response.json()['data'])
                        state = {} if available else {**state, 'checked_at': time.time()}
                        write_json(path, state)
                        self.log.emit('free_catalog_checked', available=available)
                except (requests.RequestException, ValueError, KeyError):
                    self.log.emit('free_catalog_check_failed')
            self.provider = 'bailian' if state.get('reason') else 'openrouter'
            prefix = self.provider.upper()
            self.key, self.base, self.model = [credentials[prefix + '_' + k] for k in ('API_KEY', 'BASE_URL', 'MODEL')]
            self.base = self.base.rstrip('/')

        def _call_once(self, *args, **kwargs):
            self.choose()
            self.last_quota = None
            try:
                return super()._call_once(*args, **kwargs)
            except ResponsesError:
                if not self.last_quota:
                    raise
                state = {'reason': self.last_quota, 'utc_day': datetime.now(timezone.utc).date().isoformat(),
                         'checked_at': time.time()}
                write_json(routing_path, state)
                self.log.emit('provider_fallback', reason=self.last_quota, source='openrouter', target='bailian',
                              request_id=self.last_attempt_id)
                # A failed route consumes a physical request, but no browser action or logical call.
                self.log.emit('responses_retry', persistent=True, retry_number=1, delay_seconds=0,
                              http_status=429 if self.last_quota == 'daily_quota' else 404,
                              failure_kind='provider_fallback', request_id=self.last_attempt_id,
                              budget_unit='logical_request')
                return self._call_once(*args, **kwargs)

    webnavigator.OriginalResponses = QwenResponses
    return QwenResponses
