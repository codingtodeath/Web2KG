"""Transport and benchmark adapters; the upstream Agent and navigator stay unchanged."""
import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import re
from uuid import uuid4

import requests

from ..common.activity import read_events
from ..common.jina import JinaClient
from ..common.response_output import output_messages
from ..common.responses import ResponsesClient, ResponsesError
from ..common.storage import EventLog, write_json
from .verified import FINAL_INSTRUCTION


class EmptyFinalResponse(ValueError):
    pass


def final_text(messages, on_fallback=None):
    final = any(m.get('phase') == 'final_answer' for m in messages)
    values = []
    for message in messages:
        if final and message.get('phase') == 'commentary':
            continue
        if message.get('status') not in (None, 'completed'):
            raise ValueError('Incomplete visible message')
        parts = message['content']
        if any(p.get('type') != 'output_text' for p in parts):
            raise ValueError('Refusal or unsupported visible content')
        value = ''.join(p['text'] for p in parts).strip()
        if value and value not in values:
            values.append(value)
    if not values and final:
        commentary = [m for m in messages if m.get('phase') == 'commentary']
        value = final_text(commentary)
        # Only a complete, unambiguous action response can replace an empty final.
        if len(re.findall(r'^ACTION:[ \t]*', value, re.MULTILINE)) != 1 or not re.search(
                r'^ACTION:[ \t]*(?:\r?\n[ \t]*)?\S', value, re.MULTILINE):
            raise EmptyFinalResponse('Commentary has no unique action section')
        if on_fallback:
            on_fallback()
        return value
    if not values:
        raise EmptyFinalResponse('Missing final message')
    if len(values) != 1:
        raise ValueError('Missing or conflicting final messages')
    return values[0]


class OriginalResponses(ResponsesClient):
    def __init__(self, *args, verified_protocol=True, max_retries=0, actor_compatibility=False,
                 validate_verified_stop=False, unified_verified_stop=False, **kwargs):
        if max_retries not in (0, 1, 2):
            raise ValueError('max_retries must be 0, 1 or 2')
        super().__init__(*args, **kwargs)
        self.verified_protocol = verified_protocol
        self.max_retries = max_retries
        self.actor_compatibility = actor_compatibility
        self.validate_verified_stop = validate_verified_stop
        self.unified_verified_stop = unified_verified_stop

    def call(self, messages, model_id=None, system_prompt='', **kwargs):
        from ..stage2.actor_prompt import PROTOCOL, remove_legacy_warnings
        from ..stage2.action_budget import POLICY
        if kwargs.get('action_budget_policy') == POLICY:
            # Group regeneration, transport retries and provider fallback while
            # keeping every physical request and its usage individually logged.
            kwargs['action_budget_id'] = uuid4().hex
        single_action_protocol = PROTOCOL in system_prompt and 'target_page' not in system_prompt
        if single_action_protocol:
            system_prompt = remove_legacy_warnings(system_prompt)
        prompt = system_prompt
        attempt_messages = messages
        self._stop_retry_required = False
        invalid_domains = 0
        for attempt in range(self.max_output_retries + 1):
            if self.stopped():
                raise ResponsesError('User stop file present', kind='cancelled')
            try:
                self._output_retry_number = attempt
                self._output_retry_first_physical = bool(attempt)
                return self._call_transport(attempt_messages, model_id, prompt, **kwargs)
            except ResponsesError as error:
                if error.kind == 'output:actor_domain_invalid':
                    invalid_domains += 1
                retryable = error.kind == 'output:verified_stop_invalid' or (
                    single_action_protocol and error.kind in {'output:actor_action_invalid', 'output:actor_domain_invalid'})
                # Preserve the upstream three-invalid-domain limit, within the
                # existing shared output-regeneration budget (no nested retries).
                if not retryable or attempt >= self.max_output_retries or invalid_domains >= 3:
                    raise
                self.log.emit('responses_output_retry', request_id=getattr(self, 'last_attempt_id', None),
                    purpose='original_actor', retry_number=attempt + 1, maximum=self.max_output_retries,
                    failure_kind=error.kind.removeprefix('output:'), budget_unit='logical_request', browser_action_repeated=False)
                self._stop_retry_required = self._stop_retry_required or error.kind == 'output:verified_stop_invalid'
                if error.kind in {'output:actor_action_invalid', 'output:actor_domain_invalid'}:
                    prompt = system_prompt + ('\nYour previous response was rejected: ' + str(error) + '. '
                        'No action from that response was executed. Using the SAME observation and element IDs, '
                        'regenerate with exactly ONE ACTION section containing exactly ONE command. '
                        'Do not combine click/type/submit with stop, and do not include future actions. '
                        'Wait for a new observation after execution before judging success.')
                    if self._stop_retry_required:
                        prompt += ' The previous stop answer still requires one corrected stop command with the public JSON answer format.'
                else:
                    prompt = system_prompt + ('\nYour previous stop answer was rejected: ' + str(error) + '. '
                        'Using the SAME observations, return exactly one ACTION: stop with an actual JSON object inside brackets. '
                        'Follow the complete examples in the stop specification; choose the fields from your own observations. '
                        'Do not output the words "JSON object", a bare status, null, or another browser action. '
                        'Do not claim success without evidence.')
                if single_action_protocol:
                    # Keep the original observation untouched. Replace, rather than
                    # accumulate, the failed-response pair on each regeneration.
                    rejected = getattr(self, '_rejected_actor_output', None)
                    feedback = (f'Format correction {attempt + 1}/{self.max_output_retries}. '
                        f'Validation error: {error}. '
                        'The preceding rejected assistant response is NOT executed action history. '
                        'No action from it was executed and the original observation and element IDs are unchanged. '
                        'Correct that response using exactly ONE ACTION section with ONE complete command. '
                        'Repeated identical ACTION sections are also invalid. Do not copy the rejected response twice. '
                        'If several operations are needed, output only the next one, then wait for its new observation. '
                        'Follow the original response fields and public stop format; do not add a second answer or think tags.')
                    if self._stop_retry_required:
                        feedback += ' Return one corrected stop command; do not substitute another browser action.'
                    attempt_messages = list(messages)
                    if rejected:
                        attempt_messages.append({'role': 'assistant', 'content': rejected})
                    attempt_messages.append({'role': 'user', 'content': feedback})
                    self.log.emit('responses_output_feedback', request_id=getattr(self,'last_attempt_id',None),
                        retry_number=attempt+1, feedback_version='previous-response-v1',
                        rejected_response_included=bool(rejected),
                        rejected_sha256=hashlib.sha256(rejected.encode()).hexdigest() if rejected else None)
            finally:
                self._output_retry_number = 0
                self._output_retry_first_physical = False

    def _call_transport(self, messages, model_id=None, system_prompt='', **kwargs):
        if self.retry_connections:
            return self.connection_call(lambda: self._call_once(messages, model_id, system_prompt, **kwargs))
        for attempt in range(self.max_retries + 1):
            try:
                return self._call_once(messages, model_id, system_prompt, **kwargs)
            except ResponsesError as error:
                transient = (error.kind in {'transport', 'empty_response'} or
                             error.http_status in {408, 429, 500, 502, 503, 504})
                calls = sum(e['event'] == 'responses_started' for e in read_events(self.log.path))
                scoped = kwargs.get('action_budget_id') is not None
                if scoped:
                    from ..stage2.action_budget import action_budget_used
                    calls = action_budget_used(read_events(self.log.path))
                if not transient or attempt >= self.max_retries or (not scoped and calls >= self.max_calls):
                    raise
                delay = (2, 5)[attempt]
                self.log.emit('responses_retry', retry_number=attempt + 1, delay_seconds=delay,
                              http_status=error.http_status, failure_kind=error.kind,
                              remaining_calls=self.max_calls - calls)
                time.sleep(delay)

    def _call_once(self, messages, model_id=None, system_prompt='', **kwargs):
        self._rejected_actor_output = None
        from ..stage2.action_budget import POLICY, action_budget_used
        scoped = kwargs.get('action_budget_policy') == POLICY
        events = read_events(self.log.path) if scoped else []
        calls = action_budget_used(events) if scoped else self.budget_used()
        output_retry = getattr(self, '_output_retry_number', 0)
        purpose = 'page_selector' if 'target_page' in system_prompt else 'original_actor'
        budget_id = kwargs.get('action_budget_id')
        reserved = scoped and any(e.get('action_budget_id') == budget_id
                                 for e in events if e['event'] == 'responses_started')
        if (self.max_calls is not None and calls >= self.max_calls and not output_retry
                and not reserved and not (scoped and purpose == 'page_selector')):
            raise ResponsesError('Responses call budget exhausted')
        inputs, saved = [], []
        for message in messages:
            content = message['content']
            content = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
            converted, evidence = [], []
            for part in content:
                if part['type'] == 'text':
                    value = {'type': 'input_text', 'text': part['text']}
                    converted.append(value); evidence.append(value)
                elif part['type'] == 'image_url':
                    url = part['image_url']['url']
                    if not url.startswith('data:image/'):
                        raise ValueError('Only local screenshot data accepted')
                    raw = base64.b64decode(url.split(',', 1)[1], validate=True)
                    if raw.startswith(b'\x89PNG\r\n\x1a\n'):
                        url = 'data:image/png;base64,' + url.split(',', 1)[1]
                    converted.append({'type': 'input_image', 'image_url': url})
                    sha = hashlib.sha256(raw).hexdigest()
                    image = self.log.directory / 'model-images' / (sha + ('.png' if raw.startswith(b'\x89PNG') else '.jpg'))
                    image.parent.mkdir(exist_ok=True)
                    if not image.exists():
                        image.write_bytes(raw)
                    evidence.append({'type': 'image_sha256', 'sha256': sha,
                                     'file': image.relative_to(self.log.directory).as_posix()})
                else:
                    raise ValueError('Unsupported upstream message part')
            inputs.append({'role': message['role'], 'content': converted})
            saved.append({'role': message['role'], 'content': evidence})
        instructions = system_prompt
        if purpose == 'original_actor' and self.verified_protocol:
            if self.unified_verified_stop:
                from .verified_answer import stop_instruction
                instructions = stop_instruction(instructions)
            else:
                instructions += '\n' + FINAL_INSTRUCTION.replace('When action=stop, answer must be', 'When using stop [answer], answer must be').replace('For other actions use an empty answer.', '')
        request = dict(model=self.model, store=False, max_output_tokens=self.max_output_tokens,
                       instructions=instructions, input=inputs, text={'format': {'type': 'text'}})
        key = uuid4().hex; folder = self.log.directory / 'llm' / key
        self.last_attempt_id = key
        write_json(folder / 'request.json', {**request, 'input': saved})
        self.log.emit('responses_started', request_id=key, purpose=purpose, model=self.model,
                      call=calls + 1, output_reservation=self.max_output_tokens,
                      output_retry_number=output_retry if getattr(self, '_output_retry_first_physical', False) else 0,
                      **({'action_budget_policy': POLICY, 'action_budget_id': budget_id} if scoped else {}))
        self._output_retry_first_physical = False
        started = time.monotonic()
        try:
            response = self.session.post(self.base + '/responses', json=request,
                headers={'Authorization': 'Bearer ' + self.key}, timeout=(10, 120), allow_redirects=False)
            if not response.ok:
                detail = {'http_status': response.status_code, 'body_bytes': len(response.content),
                          'body_sha256': hashlib.sha256(response.content).hexdigest()}
                try:
                    error = response.json().get('error', {})
                    for field in ('code', 'type'):
                        value = error.get(field) if isinstance(error, dict) else None
                        if isinstance(value, str) and re.fullmatch(r'[a-zA-Z0-9_.-]{1,80}', value):
                            detail[field] = value
                except (ValueError, AttributeError):
                    pass
                write_json(folder / 'http_error.json', detail)
                raise ResponsesError(f'Responses HTTP {response.status_code}',
                                     http_status=response.status_code, kind='http')
            data = response.json(); visible = output_messages(data)
            if purpose == 'original_actor':
                # Only visible answer text, never provider reasoning_content.
                self._rejected_actor_output = '\n'.join(
                    p.get('text','') for m in visible for p in m.get('content',[])
                    if p.get('type') == 'output_text')
            write_json(folder / 'response.json', {'messages': visible, 'usage': data.get('usage'),
                'model': data.get('model'), 'status': data.get('status'), 'response_id': data.get('id')})
            self.log.emit('responses_received', request_id=key, purpose=purpose,
                actual_model=data.get('model'), usage=data.get('usage') or {}, seconds=time.monotonic()-started)
            if data.get('status') != 'completed':
                raise ValueError('Incomplete response')
            def select_text(messages):
                return final_text(messages, on_fallback=lambda: self.log.emit(
                    'responses_commentary_fallback', request_id=key, purpose=purpose,
                    reason='empty_final_with_unique_action'))
            from ..stage2.actor_prompt import PROTOCOL
            try:
                if self.actor_compatibility and purpose == 'original_actor':
                    from .actor_output import actor_output
                    value, changes = actor_output(visible, select_text)
                    if changes:
                        self.log.emit('actor_output_adapted', request_id=key, changes=changes,
                                      adapter='original-action-syntax-v1')
                else:
                    value = select_text(visible)
            except ValueError as error:
                if purpose == 'original_actor' and PROTOCOL in system_prompt:
                    raise ResponsesError(str(error), kind='output:actor_action_invalid') from None
                raise
            if purpose == 'original_actor' and PROTOCOL in system_prompt:
                from .actor_output import validate_single_action
                validate_single_action(value)
                if scoped:
                    from ..stage2.action_notes import parse_sections
                    action = parse_sections(value, ['action'], require_single_action=True)['action']
                    if action.startswith('search_ax ') and not kwargs.get('ax_search_available', False):
                        raise ResponsesError('AX search is exhausted or unavailable; choose one browser action or stop.',
                                             kind='output:actor_action_invalid')
                if kwargs.get('navigation_domains') is not None:
                    from .actor_output import validate_navigation_domain
                    validate_navigation_domain(value, kwargs['navigation_domains'])
            if purpose == 'original_actor' and self.verified_protocol and self.validate_verified_stop:
                from .verified_answer import validate_stop
                validate_stop(value, required=getattr(self, '_stop_retry_required', False))
            (folder / 'parsed.txt').write_text(value, encoding='utf-8')
            self.log.emit('responses_output_parsed', request_id=key, purpose=purpose, parser='visible-text-v1')
            if scoped and purpose == 'original_actor' and PROTOCOL in system_prompt and action.startswith('search_ax '):
                self.log.emit('action_budget_search', action_budget_policy=POLICY,
                              action_budget_id=budget_id, request_id=key, browser_step=0)
            return value
        except Exception as error:
            reason = str(error) if isinstance(error, (ResponsesError, ValueError)) else type(error).__name__
            status = getattr(error, 'http_status', None)
            kind = ('transport' if isinstance(error, requests.RequestException) and
                    not isinstance(error, requests.exceptions.JSONDecodeError) else getattr(error, 'kind', None))
            if isinstance(error, EmptyFinalResponse):
                kind = 'empty_response'
            self.log.emit('responses_failed', request_id=key, purpose=purpose, error=reason,
                          http_status=status, failure_kind=kind, retry=False, seconds=time.monotonic()-started)
            raise ResponsesError(reason, http_status=status, kind=kind) from None


def load_upstream(batch, backend):
    os.environ['REDDIT'] = backend
    sys.path[:0] = [str(batch / 'import-stubs'), str(batch / 'upstream')]
    return importlib.import_module('webNavigator.webNavigator')


def parse_heading_elements(self, text, key_list):
    """Original field extraction with whole-line headings and one reason alias."""
    elements = {}
    for key in key_list:
        heading = re.escape(key.upper())
        if key.lower() == 'reason':
            heading = r'(?:REASON|REASON FOR ACTION)'
        match = re.search(
            rf'^[ \t]*{heading}[ \t]*[:：][ \t]*(.*?)'
            rf'(?=^[ \t]*[A-Z][A-Z \t\d]*[:：]|\Z)', text, re.MULTILINE | re.DOTALL)
        elements[key] = match.group(1).strip() if match else ''
    return elements


def set_ax_limit(rows):
    """Apply the same post-pruning line limit to both pinned execution backends."""
    from functools import partial
    if not 1 <= rows <= 10000:
        raise ValueError('Invalid AX line limit')
    opt = importlib.import_module('webNavigator.obs_opt')
    modules = [opt, importlib.import_module('webNavigator.env'), importlib.import_module('webNavigator.webNavigator')]
    previous = [(module, module.translate_node_to_str) for module in modules]
    def translate(node, mode='concise', **kwargs):
        lines = []
        opt.parse_node_descendants(node, partial(opt.action_return_visible_node, intent_bias=node.depth,
                                               mode=mode, **kwargs), tree_buffer=lines)
        return '\n'.join(lines[:rows])
    for module in modules:
        module.translate_node_to_str = translate
    def restore():
        for module, original in previous:
            module.translate_node_to_str = original
    return restore


def install_adapters(module, client, batch, endpoint):
    graph_class = importlib.import_module('webNavigator.navigation.web_graph').WebGraph
    navigate = graph_class.navigate_to_node
    def recorded_navigation(self, page, node):
        before = page.url; started = time.monotonic(); returned = None; error = None
        try:
            returned = navigate(self, page, node)
            return returned
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            client.log.emit('original_navigation', before=before, after=page.url,
                target_url=node.web_metadata_instance.url, target=node.get_unique_id(),
                direct=node.can_arrival_directly, returned=returned, error=error,
                seconds=time.monotonic()-started)
    graph_class.navigate_to_node = recorded_navigation
    retriever_module = importlib.import_module('webNavigator.retrieval.retriever')
    class LocalRetriever(retriever_module.JinaRetriever):
        def __init__(self, args):
            folder = batch / 'released-graph'
            args.persistence_file = str(folder / 'image_embedding.pkl')
            super().__init__(args)
            self.local = JinaClient(endpoint, client.log, timeout=600 if client.actor_compatibility else 180,
                                    retry_connections=client.retry_connections if client.actor_compatibility else False,
                                    stop_file=client.stop_file)
            self.identity = self.local.identity()
            client.log.emit('released_embeddings_loaded', pages=len(self.names),
                image_encoder_attestation='not_present_in_upstream_asset',
                query_encoder=self.identity, selection='User explicitly requested original released vectors')

        def encode_text(self, query, task='retrieval.query', return_multivector=False):
            value = self.local.encode(query, 'text', self.identity)
            return value.tolist(), len(value)

        def rerank(self, query):
            result = super().rerank(query)
            client.log.emit('original_retrieval', query=query, candidates=result)
            return result

    domain = importlib.import_module('webNavigator.domain_navigator')
    domain.JinaRetriever = LocalRetriever
    # The loader's file-existence check is replaced, not its graph/navigation code.
    domain.DomainNavigator._load_retriever = lambda self: setattr(self, 'retriever', LocalRetriever(self.args))
    module.CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP['gpt'] = client.call
    module.CALL_MODEL_MAP['gpt'] = lambda prompt, **kw: client.call([{'role': 'user', 'content': prompt}], **kw)


def run_original(batch, config, task):
    import yaml
    from PIL import Image
    from types import SimpleNamespace
    def namespace(value):
        return SimpleNamespace(**{k: namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value

    folder = batch / str(task['task_id']); log = EventLog(folder)
    client = OriginalResponses(Path(config['workspace']) / 'auth/llm-crawl.env', log,
                               config['max_calls'], config['output_tokens_per_call'], config['model'],
                               actor_compatibility=config.get('actor_compatibility', False),
                               unified_verified_stop=config.get('unified_verified_stop', False),
                               retry_connections=config.get('retry_connections', False), stop_file=config.get('stop_file'))
    site = config.get('site', 'reddit')
    os.environ[site.upper()] = config['backend']
    module = load_upstream(batch, config['backend'])
    original_parser = module.Agent.parse_elements
    if config.get('section_parser') == 'anchored-heading-v1':
        module.Agent.parse_elements = parse_heading_elements
        log.emit('original_parser_compatibility', version='anchored-heading-v1',
                 change='Whole-line headings; REASON FOR ACTION alias; ASCII/fullwidth colon')
    restore_ax = set_ax_limit(config.get('ax_line_limit', 1000))
    install_adapters(module, client, batch, config['encoder_endpoint'])
    from webNavigator.env import WebArenaEnvironmentWrapper
    from webNavigator.prompts import webnavigator_prompt
    from browser_env import ActionTypes
    settings = yaml.safe_load((batch / 'config.yml').read_text(encoding='utf-8'))
    settings['env']['ax_search'] = config.get('ax_search', True)
    settings['agent']['others']['logname'] = str(task['task_id'])
    settings['agent']['others']['logdir'] = str(folder / 'upstream-log')
    cfg = namespace(settings)

    class RecordedEnvironment(WebArenaEnvironmentWrapper):
        def update_webarena_metrics(self, action_cmd=None):
            if action_cmd:
                self.trajectory.append(action_cmd)
                if action_cmd['action_type'] == ActionTypes.STOP:
                    self.is_done = True
            if not self.is_done:
                self.trajectory.append({'observation': self.obs, 'info': self.info})

        def done(self):
            return self.is_done or self.steps >= self.max_steps

        def snapshot(self):
            Image.fromarray(self.obs['image']).save(folder / f'step-{self.steps}.png')
            (folder / f'step-{self.steps}.txt').write_text(self.obs['text'][0], encoding='utf-8')

        def step(self, action, action_elements=None):
            before = self.webarena_env.page.url
            value = super().step(action, action_elements)
            self.snapshot()
            log.emit('agent_action', step=self.steps, action=action, before=before, url=self.webarena_env.page.url,
                     implementation='original_webnavigator',
                     after=self.webarena_env.page.url, selected=(action_elements or {}).get('target_page'))
            return value

    task_config = {'task_id': task['task_id'], 'sites': task['sites'], 'intent': task['intent'],
        'start_url': task['start_urls'][0].replace(config['graph_origin'], config['backend']),
        'storage_state': config.get('storage_state'), 'record_har_path': str(folder / 'network.har'),
        'extra_http_headers': ({'X-Postmill-Auto-Login': os.getenv('VERIFIED_REDDIT_USER', 'MarvelsGrantMan136') + ':' + os.environ['VERIFIED_REDDIT_PASSWORD']} if site == 'reddit' else {}),
        'allowed_origin': config['backend']}
    write_json(folder / 'browser-task.json', task_config)
    env = agent = None; started = time.monotonic()
    result = {'status': 'not_started', 'navigation': {}, 'steps': 0, 'answer': ''}
    try:
        env = RecordedEnvironment(config_file=str(folder / 'browser-task.json'), max_browser_rows=config.get('ax_line_limit', 500),
            max_steps=config['max_steps'], slow_mo=0, observation_type='accessibility_tree',
            current_viewport_only=False, viewport_size={'width': 1280, 'height': 720}, headless=True, global_config=cfg)
        env.snapshot()
        agent = module.webNavigator(config=cfg.agent,
            prompt_dict={k: v for k, v in vars(webnavigator_prompt).items() if isinstance(v, dict)})
        agent.act(objective=env.get_objective(), env=env)
        result['status'] = 'agent_stopped' if env.is_done else 'step_limit'
    except Exception as error:
        result.update(status='model_error' if isinstance(error, ResponsesError) else 'runtime_error',
                      error=type(error).__name__ + ': ' + str(error)[:300])
        import traceback
        (folder / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        module.Agent.parse_elements = original_parser
        restore_ax()
        if env:
            actions = [a for a in env.trajectory if 'action_type' in a]
            write_json(folder / 'actions.json', json.loads(json.dumps(actions, default=str)))
            stops = [a for a in actions if a['action_type'] == ActionTypes.STOP]
            result.update(steps=env.steps, url=env.webarena_env.page.url,
                          atomic_actions=len(actions),
                          answer=stops[-1].get('answer', '') if stops else '')
            if agent:
                write_json(folder / 'trajectory.json', json.loads(json.dumps(agent.get_trajectory(), default=str)))
            env.webarena_env.context.close()
            env.close()
        result['seconds'] = round(time.monotonic()-started, 3)
        result['connection_retry_seconds'] = round(client.retry_seconds, 3)
        write_json(folder / 'result.json', result)
        (folder / 'agent_response.json').write_text(result['answer'], encoding='utf-8')
        log.emit('original_task_finished', **{k: result[k] for k in ('status', 'steps', 'seconds')})
    return result
