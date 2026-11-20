"""Pinned AgentOccam with provider, evidence and Verified environment adapters.

No graph, retrieval, our Actor prompt, diagnosis or recovery is installed here.
The upstream Agent/observation algorithms remain in the separately frozen source.
"""
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import ModuleType, SimpleNamespace

from ..common.responses import ResponsesError
from ..common.storage import EventLog, write_json
from .webnavigator import parse_heading_elements


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value


def unavailable(*args, **kwargs):
    raise RuntimeError('Unregistered upstream provider or legacy evaluator was invoked')


def copy_node_fields(self):
    """Equivalent detached copy without traversing links discarded by upstream.

    Upstream TreeNode fields are scalar values and JSON-like properties plus the
    parent/children links. Deep-copy the retained fields, not the entire graph.
    """
    from copy import deepcopy
    node = type(self).__new__(type(self))
    node.__dict__.update(deepcopy({k:v for k,v in vars(self).items() if k not in {'parent','children'}}))
    node.parent = None
    node.children = []
    return node


def load_upstream(upstream, client, folder, backend):
    """Run in a dedicated worker: import names collide with other upstream agents."""
    if 'AgentOccam.AgentOccam' in sys.modules or 'browser_env' in sys.modules:
        raise RuntimeError('AgentOccam requires an isolated worker process')
    os.environ['REDDIT'] = backend
    sys.path.insert(0, str(upstream))
    # Unused SDK imports must not download assets or instantiate credentials.
    # Every callable fails closed, rather than impersonating a model response.
    for family in ('claude', 'mistral', 'cohere', 'llama', 'titan', 'gemini'):
        stub = ModuleType('AgentOccam.llms.' + family)
        for name in ('call_' + family, 'call_' + family + '_with_messages', 'arrange_message_for_' + family):
            setattr(stub, name, unavailable)
        sys.modules[stub.__name__] = stub
    package = ModuleType('evaluation_harness'); package.__path__ = []
    evaluator = ModuleType('evaluation_harness.evaluators'); evaluator.evaluator_router = unavailable
    sys.modules[package.__name__] = package
    sys.modules[evaluator.__name__] = evaluator
    module = importlib.import_module('AgentOccam.AgentOccam')
    from browser_env.processors import TreeNode
    TreeNode.copy = copy_node_fields
    module.MODEL_FAMILIES.append('qwen')
    module.CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP['qwen'] = client.call
    module.CALL_MODEL_MAP['qwen'] = lambda prompt, **kw: client.call([{'role': 'user', 'content': prompt}], **kw)
    module.ARRANGE_MESSAGE_FOR_MODEL_MAP['qwen'] = module.arrange_message_for_gpt
    module.Agent.parse_elements = parse_heading_elements
    original_init = module.Actor.__init__

    def actor_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.output_play_path = str(folder / 'upstream-play.txt')
        self.output_trash_path = str(folder / 'upstream-trash.txt')

    module.Actor.__init__ = actor_init
    return module


def run_occam(batch, config, task, client=None):
    import yaml
    from PIL import Image
    folder = batch / str(task['task_id']); folder.mkdir(exist_ok=True)
    log = EventLog(folder)
    if client is None:
        from scripts.qwen_method_transport import clients
        cls = clients(config['credential_file'], batch/'routing-state.json',
                      provider='bailian', model_override=config['model'])
        client = cls(Path(config['workspace'])/'auth/llm-crawl.env', log,
                     config['max_calls'], config['output_tokens_per_call'], config['model'],
                     verified_protocol=False, actor_compatibility=False,
                     retry_connections=True, stop_file=config['stop_file'])
    module = load_upstream(batch/'upstream', client, folder, config['backend'])
    from AgentOccam.env import WebArenaEnvironmentWrapper
    from AgentOccam.prompts import AgentOccam_prompt
    from browser_env import ActionTypes
    settings = yaml.safe_load((batch/'config.yml').read_text('utf-8'))
    settings['agent']['others'].update(logname=str(task['task_id']), verbose=0, logging=True)
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
            Image.fromarray(self.obs['image']).save(folder/f'step-{self.steps}.png')
            (folder/f'step-{self.steps}.txt').write_text(self.obs['text'][0], encoding='utf-8')

        def step(self, action):
            before = self.webarena_env.page.url
            # Native step counts note/branch/prune decisions too, even if no DOM action.
            status = super().step(action)
            self.snapshot()
            log.emit('agent_action', step=self.steps, action=action, before=before,
                     after=self.webarena_env.page.url, url=self.webarena_env.page.url,
                     implementation='original_agentoccam')
            return status

    task_config = dict(task_id=task['task_id'], sites=task['sites'], intent=task['intent'],
        start_url=task['start_urls'][0], storage_state=None,
        record_har_path=str(folder/'network.har'), allowed_origin=config['backend'],
        extra_http_headers=({'X-Postmill-Auto-Login':
            os.getenv('VERIFIED_REDDIT_USER', 'MarvelsGrantMan136')+':'+os.environ['VERIFIED_REDDIT_PASSWORD']}
            if config.get('verified_login', True) else {}))
    write_json(folder/'browser-task.json', task_config)
    random.seed(42)
    env = agent = None; started = time.monotonic()
    result = dict(status='not_started', steps=0, answer='')
    try:
        env = RecordedEnvironment(config_file=str(folder/'browser-task.json'),
            max_browser_rows=cfg.env.max_browser_rows, max_steps=config['max_steps'],
            slow_mo=0, observation_type='accessibility_tree', current_viewport_only=False,
            viewport_size={'width':1280, 'height':720}, headless=True, global_config=cfg)
        env.snapshot()
        agent = module.AgentOccam(config=cfg.agent,
            prompt_dict={k:v for k,v in vars(AgentOccam_prompt).items() if isinstance(v,dict)})
        agent.act(objective=env.get_objective(), env=env)
        result['status'] = 'agent_stopped' if env.is_done else 'step_limit'
    except Exception as error:
        import traceback
        result.update(status='model_error' if isinstance(error, ResponsesError) else 'runtime_error',
                      error=type(error).__name__+': '+str(error)[:300])
        (folder/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        if env:
            actions = [a for a in env.trajectory if 'action_type' in a]
            write_json(folder/'actions.json', json.loads(json.dumps(actions, default=str)))
            stops = [a for a in actions if a['action_type'] == ActionTypes.STOP]
            result.update(steps=env.steps, url=env.webarena_env.page.url, atomic_actions=len(actions),
                          answer=stops[-1].get('answer','') if stops else '')
            if agent:
                write_json(folder/'trajectory.json', json.loads(json.dumps(agent.get_trajectory(),default=str)))
            try:
                env.webarena_env.context.close()
            finally:
                env.close()
        result['seconds'] = round(time.monotonic()-started,3)
        result['connection_retry_seconds'] = round(getattr(client,'retry_seconds',0),3)
        write_json(folder/'result.json',result)
        (folder/'agent_response.json').write_text(result['answer'],encoding='utf-8')
        log.emit('original_task_finished', **{k:result[k] for k in ('status','steps','seconds')})
    return result
