"""Reuse registered Qwen routing for Actor and schema-validated diagnosis calls."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import time

from web2kg.common.prompt_text import to_yaml
from web2kg.common.responses import ResponsesClient, ResponsesError
from web2kg.common.storage import write_json


def clients(env_path, routing_path, *, provider=None, model_override=None):
    if provider not in {None, 'bailian', 'openrouter'}:
        raise ValueError('Unknown Qwen provider')
    if model_override is not None and (not provider or not isinstance(model_override, str) or not model_override.strip()):
        raise ValueError('An explicit model override requires a fixed provider and nonempty model')
    # Load an isolated module: the historical baseline transport remains unchanged.
    spec = importlib.util.spec_from_file_location('method_qwen_transport', Path(__file__).with_name('qwen_hard_transport.py'))
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    convert = transport.chat_request

    def chat_request(request):
        wire = convert(request)
        form = request.get('text', {}).get('format', {})
        if form.get('type') == 'json_schema':
            wire['messages'][0]['content'] += '\nRequired output JSON schema (described in YAML):\n' + to_yaml(form['schema'])
        return wire

    transport.chat_request = chat_request
    from web2kg.evaluation import webnavigator
    original = webnavigator.OriginalResponses
    try:
        routed = transport.install(env_path, routing_path)
    finally:
        webnavigator.OriginalResponses = original

    class QwenMethod(routed):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.allow_json_fence = True
            self.identity['json_compatibility'] = 'single-whole-json-fence-v1'
            if provider:
                self.choose()
                self.identity.update(endpoint=self.base + '/chat/completions', model=self.model,
                                     provider=self.provider, protocol='qwen-chat-schema-v1')

        def choose(self):
            if provider is None:
                return super().choose()
            from dotenv import dotenv_values
            values = dotenv_values(env_path)
            self.provider = provider
            self.key, self.base, self.model = [values[provider.upper() + '_' + k]
                                             for k in ('API_KEY', 'BASE_URL', 'MODEL')]
            self.base = self.base.rstrip('/')
            if model_override is not None:
                self.model = model_override

        def _json_once(self, *args, **kwargs):
            self.choose()
            self.last_quota = None
            try:
                return ResponsesClient._json_once(self, *args, **kwargs)
            except ResponsesError:
                if provider or not self.last_quota:
                    raise
                write_json(routing_path, {'reason': self.last_quota,
                    'utc_day': datetime.now(timezone.utc).date().isoformat(), 'checked_at': time.time()})
                self.log.emit('provider_fallback', reason=self.last_quota, source='openrouter', target='bailian',
                              request_id=self.last_attempt_id)
                self.log.emit('responses_retry', persistent=True, retry_number=1, delay_seconds=0,
                    http_status=429 if self.last_quota == 'daily_quota' else 404, failure_kind='provider_fallback',
                    request_id=self.last_attempt_id, budget_unit='logical_request')
                return self._json_once(*args, **kwargs)

    return QwenMethod
