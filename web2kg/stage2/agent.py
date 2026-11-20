"""The pinned WebNavigator Agent, with our graph and optional template context."""
import base64
import importlib
import json
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from ..common.browser import Browser
from ..common.responses import ResponsesError
from ..common.storage import write_json
from .action_notes import AttachedNotes
from .action_budget import POLICY as ACTION_BUDGET_POLICY
from .actor_prompt import OUTPUT_FIELDS, CONTEXT_POLICY
from .actor_input import POLICY as INPUT_POLICY, navigation_context
from .confidence import DirectionConfidence
from .navigation import RouteNavigator
from .observation import NUMERIC_TEXT_POLICY, install_numeric_text_guard
from .prompt_context import context_messages, site_system_prompt
from .retrieval import retrieve

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "third_party/webnavigator"
COMMANDS = ["click", "type", "stop", "go_back", "optimal_navigate"]


def load_runtime():
    sys.path[:0] = [str(RUNTIME / "import-stubs"), str(RUNTIME)]
    module = importlib.import_module("webNavigator.webNavigator")
    if not Path(module.__file__).resolve().is_relative_to(RUNTIME.resolve()):
        raise RuntimeError("Another WebNavigator version is already loaded; use a fresh worker")
    return module


def configuration(log, model="gpt-5.6-terra", top_k=5, max_steps=30, use_graph=True, site="reddit", headed=False):
    import yaml
    data = yaml.safe_load((ROOT / "examples/webnavigator-execution.yml").read_text(encoding="utf-8"))
    data.update(logdir=str(log.directory), max_steps=max_steps)
    data["agent"]["others"].update(logdir=str(log.directory / "upstream-log"), logname="task", max_steps=max_steps)
    actor = data["agent"]["actor"]
    actor["model"] = actor["selector"]["model"] = model
    actor['model_family'] = 'gpt'  # This adapter injects the client into the GPT message interface.
    actor["retriever"]["topk"] = top_k
    actor["domain_paths"] = {site: "current-graph"} if use_graph else {}
    actor["navigation_command"] = COMMANDS if use_graph else COMMANDS[:-1]
    actor["note_mode"] = "attached_field_v1"
    actor['output'] = list(OUTPUT_FIELDS)
    actor['context_policy'] = CONTEXT_POLICY
    actor['input_policy'] = INPUT_POLICY
    actor['input'] = ['task memory' if k == 'previous plans' else k for k in actor['input']]
    actor['history_mode_analysis'] = False
    actor['call_budget_policy'] = ACTION_BUDGET_POLICY
    data["env"]["headless"] = not headed
    return data


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value


class LiveBrowser(Browser):
    """Observe/replay in the upstream browser; never start a second execution browser."""
    def __init__(self, env, graph, log, start_url, allow_writes):
        super().__init__(start_url, log, allow_writes=allow_writes, identity_mode=graph.data.get("identity_mode", "semantic"))
        self.viewport = {"width": 1280, "height": 720}
        self.env = env
        self.document_response = None
        self.last_page, self.current_action = {}, None
        self.sync()

    def sync(self):
        self.page = self.env.webarena_env.page
        self.context = self.page.context
        self.cdp = self.page.client

    def observe(self, directory=None):
        self.sync()
        return super().observe(directory)


class CurrentGraph:
    def __init__(self, graph, navigator, log, knowledge=None):
        self.navigator, self.log, self.knowledge = navigator, log, knowledge
        self.nodes = {key: SimpleNamespace(key=key, screen_path=key + ".png") for key in graph.nodes}
        self.last_result = {"success": False, "reason": "not_requested", "steps": 0}

    def navigate_to_node(self, page, node):
        try:
            self.last_result = self.navigator.navigate(node.key,
                self.knowledge.replay_action if self.knowledge else None)
        except Exception as error:
            self.last_result = {"success": False, "reason": "navigation_error", "error": str(error), "steps": 0}
            self.log.emit("navigation", target=node.key, **self.last_result)
        return self.last_result["success"]


def run_task(graph, task, log, client, embedding, *, start_url=None, home_url=None, knowledge=None,
             max_steps=30, top_k=5, storage_state=None, headed=False, allow_writes=False,
             use_graph=True, site="reddit", backend=None, headers=None, final_check=None,
             update_graph=False, description_client=None, max_descriptions=None, card_client=None, max_cards=None,
             finalize_graph_assets=True, framework='auto', diagnosis_client=None, experience_store=None,
             max_diagnoses=3, max_replay_actions=12, recovery_seed=0, time_budget_seconds=None, ax_line_limit=1000,
             ax_search=True, framework_refresh_updates=10, max_graph_nodes=500, retention_half_life=20,
             template_clustering='average', template_tolerance=.05,
             template_refresh_min=5, template_refresh_fraction=.2, direction_confidence=None):
    """Use the upstream Agent with graph/context/transport and attached-note adaptations."""
    if max_steps < 1 or not 1 <= top_k <= 20 or any(v is not None and v < 0 for v in (max_descriptions, max_cards)):
        raise ValueError("Invalid execution budget")
    if diagnosis_client and not (update_graph and use_graph):
        raise ValueError('Diagnosis requires the observed execution graph and safe graph navigation')
    confidence = DirectionConfidence(direction_confidence, log)
    if confidence.config['mode'] == 'active' and not diagnosis_client:
        raise ValueError('Active direction confidence requires diagnosis; use shadow to record scores only')
    if framework == 'auto':
        from ..stage1.site_framework import load_framework
        framework = load_framework(graph)
    module = load_runtime()
    from ..evaluation.webnavigator import set_ax_limit
    restore_ax = set_ax_limit(ax_line_limit)
    from webNavigator.env import WebArenaEnvironmentWrapper
    from webNavigator.prompts import webnavigator_prompt
    from browser_env import ActionTypes
    from PIL import Image

    start_url = start_url or graph.data["start_url"]
    cfg_data = configuration(log, getattr(client, "model", "gpt-5.6-terra"), top_k, max_steps, use_graph, site, headed)
    cfg_data['env']['ax_search'] = ax_search
    cfg_data['env']['numeric_text_policy'] = NUMERIC_TEXT_POLICY
    if confidence.enabled:
        cfg_data['agent']['actor']['output'].append('direction confidence')
        cfg_data['direction_confidence'] = confidence.config
    if diagnosis_client:
        cfg_data['diagnosis_watchdog_actions'] = confidence.config['watchdog']
    if use_graph:
        cfg_data['agent']['actor']['domain_paths'] = {site: str(graph.directory.resolve())}
    write_json(log.directory / "execution-config.json", cfg_data)
    cfg = namespace(cfg_data)
    live = navigator = env = agent = adapter = updater = recovery = None
    retrieval_candidates = None

    def check_recovery(observed, *, reasons=(), pending=None):
        count = len(recovery.decisions)
        returned = recovery.check(observed, extra_triggers=reasons, pending_action=pending)
        if len(recovery.decisions) > count:
            confidence.mark_diagnosis()
        return returned

    class Environment(WebArenaEnvironmentWrapper):
        refunded_steps = 0

        def update_webarena_metrics(self, action_cmd=None):
            if action_cmd is not None:
                self.trajectory.append(action_cmd)
                if action_cmd["action_type"] == ActionTypes.STOP:
                    self.is_done = True
            if not self.is_done:
                self.trajectory.append({"observation": self.obs, "info": self.info})

        def done(self):
            return self.is_done or self.steps >= self.max_steps or (time_budget_seconds is not None and time.monotonic() - started >= time_budget_seconds)

        def status(self):
            return {**super().status(), 'budget_steps': self.steps - self.refunded_steps,
                    'refunded_steps': self.refunded_steps, 'step_budget': max_steps}

        def snapshot(self):
            Image.fromarray(self.obs["image"]).save(log.directory / f"step-{self.steps}.png")
            (log.directory / f"step-{self.steps}.txt").write_text(self.obs["text"][0], encoding="utf-8")

        def refresh_after_recovery(self):
            import shutil
            refunded = recovery.refund_steps(self.steps)
            self.refunded_steps += refunded
            self.max_steps += refunded
            for suffix in ('png', 'txt'):
                shutil.copy2(log.directory / f'step-{self.steps}.{suffix}',
                             log.directory / f'step-{self.steps}-before-recovery.{suffix}')
            self.obs = self.webarena_env._get_obs()
            self.info = {'observation_metadata': self.webarena_env.observation_handler.get_observation_metadata()}
            self.snapshot()

        def step(self, action, action_elements=None):
            self.current_action_error = None
            trajectory_start = len(self.trajectory)
            requested_target = (action_elements or {}).get('target_page')
            before = live.observe() if knowledge or updater else None
            match = re.match(r"(click|type) \[([^\]]+)\]", action)
            observed = next((a for a in before["actions"] if match and str(a.get("ax_node_id")) == match[2]), None) if before else None
            context = (recovery.actor_step(before['id'], (action_elements or {}).get('observation description'))
                       if recovery else nullcontext())
            with context:
                result = super().step(action, action_elements)
            # Upstream assumes any selected page was reached. Correct only the feedback
            # when our verified replay fails; never tell its history a false success.
            if action.startswith("optimal_navigate") and (action_elements or {}).get("target_page") and not adapter.last_result["success"]:
                for row in reversed(self.trajectory):
                    if row.get('action_type') == ActionTypes.AUTONAVIGATE:
                        row['requested_target_page'] = row['target_page']
                        row['target_page'] = None
                        row['navigation_result'] = adapter.last_result
                        break
                action_elements["target_page"] = None
            execution = {'status': 'execution_error' if self.current_action_error else
                         ('executed' if len(self.trajectory) > trajectory_start else 'not_recorded'),
                         'actual_url': self.webarena_env.page.url}
            if self.current_action_error:
                execution['error'] = self.current_action_error
            if action.startswith('optimal_navigate'):
                if not self.current_action_error:
                    execution['status'] = 'navigation_attempted'
                fields = action_elements or {}
                nav_result = adapter.last_result if requested_target else {'success': False, 'reason': 'no_target_selected'}
                adapter.last_result = nav_result
                execution['navigation'] = navigation_context(nav_result,
                    query=fields.get('navigate_query', ''), domain=fields.get('navigate_domain', ''),
                    target=requested_target, actual_url=self.webarena_env.page.url)
            if knowledge and not action.startswith("optimal_navigate"):
                knowledge.after_action({**observed, "kind": match[1]} if observed else None)
            self.snapshot()
            log.emit("agent_action", step=self.steps, action=action, url=self.webarena_env.page.url,
                     implementation="upstream_webnavigator", selected=(action_elements or {}).get("target_page"),
                     note=(action_elements or {}).get('note', ''))
            if recovery or confidence.enabled:
                confidence.commit(action_elements or {})
            can_recover = not self.is_done and (time_budget_seconds is None or time.monotonic() - started < time_budget_seconds)
            periodic = confidence.periodic_reasons() if recovery else []
            if recovery and can_recover and check_recovery(live.observe(), reasons=periodic):
                self.refresh_after_recovery()
                execution['recovery'] = {'returned': True, 'actual_url': self.webarena_env.page.url,
                                         'note': 'Return does not undo previous writes.'}
            if agent:
                agent.actor.record_outcome(execution)
            if action_elements is not None:
                action_elements['execution result'] = execution
            return self.status() if isinstance(result, dict) else result

    class Retriever:
        return_multivector = True
        def rerank(self, query):
            nonlocal retrieval_candidates
            if navigator.awaiting_local_action():
                raise ValueError('Teleport requires a local action before another retrieval')
            current, allowed = navigator.retrieval_scope(live.observe())
            rows = retrieve(graph, query, embedding, top_k, allowed_ids=allowed)
            retrieval_candidates = rows
            if not rows:
                adapter.last_result = {'success': False, 'steps': 0,
                    'reason': 'no_reachable_indexed_candidates', 'source': current}
            if updater:
                updater.retrieved(r['id'] for r in rows)
            log.emit("retrieval", query=query, source=current, candidates=rows)
            return [{**r, "image_name": r["id"] + ".png"} for r in rows]

    class Domain:
        def __init__(self, *args):
            self.retriever, self.webgraph = Retriever(), adapter
            self.screenshots = [{"image_name": key + ".png", "image_data": base64.b64encode(
                (graph.directory / "snapshots" / key / "page.png").read_bytes()).decode("ascii")}
                for key in graph.nodes if (graph.directory / "snapshots" / key / "page.png").is_file()]

    def call(messages, system_prompt="", **kwargs):
        if updater:
            updater.poll_descriptions()
        if 'target_page' in system_prompt and retrieval_candidates == []:
            log.emit('page_selector_skipped', reason='no_reachable_indexed_candidates')
            return json.dumps({'target_page': 'None', 'reason': 'No indexed pages reachable from the current state.'})
        if time_budget_seconds is not None and time.monotonic() - started >= time_budget_seconds:
            raise TimeoutError('Task execution time budget reached before model call')
        context = extra = None
        if knowledge and "target_page" not in system_prompt:
            observed = live.observe()
            if updater:
                updater.prepare_actor(observed)
            context = knowledge.observe(observed, task)
        if framework or recovery or experience_store:
            observed = live.observe()
            if recovery:
                template = knowledge.previous_match.get('template_id') if knowledge and knowledge.previous_match else None
                extra = recovery.context(observed, template)
            else:
                extra = {'website_framework': framework['framework'] if framework else None}
                if experience_store:
                    from .recovery import template_key
                    template = knowledge.previous_match.get('template_id') if knowledge and knowledge.previous_match else None
                    extra['experience'] = experience_store.context(observed['id'], template_key(knowledge.layer, template) if knowledge else None)
            log.emit('global_recovery_context', node=observed['id'], context=extra,
                     target='selector' if 'target_page' in system_prompt else 'actor')
        if navigator and 'target_page' not in system_prompt:
            feedback = navigator.feedback()
            if feedback:
                extra = {**(extra or {}), 'navigation_feedback': feedback}
        if agent and 'target_page' not in system_prompt:
            kwargs['navigation_domains'] = tuple(sorted(k for k, v in agent.actor.domain_navigators.items() if v is not None))
        kwargs['action_budget_policy'] = ACTION_BUDGET_POLICY
        search = getattr(env, 'ax_search', None)
        kwargs['ax_search_available'] = bool(search is not None and search.available)
        return client.call(context_messages(messages, context, extra, compact_actor='target_page' not in system_prompt),
                           system_prompt=site_system_prompt(system_prompt, site), **kwargs)

    class LoggedActor(AttachedNotes, module.Actor):
        def predict_action(self, criticism_elements):
            while True:
                candidates = super().predict_action(criticism_elements)
                if not confidence.enabled:
                    return candidates
                proposal = candidates[0]
                signal = confidence.propose(proposal)
                if confidence.config['mode'] != 'active' or not signal['would_trigger']:
                    return candidates
                count = len(recovery.decisions)
                returned = check_recovery(live.observe(), reasons=signal['would_trigger'], pending=proposal)
                if len(recovery.decisions) == count or recovery.decisions[-1]['decision']['verdict'] == 'on_track':
                    return candidates
                # No history/finalize/browser action exists for the discarded proposal.
                log.emit('confidence_proposal_discarded', proposal=signal['proposal'], returned=returned)
                if returned:
                    env.refresh_after_recovery()
                observation, url = env.observation(), env.get_url()
                agent.update_online_state(url=url, observation=observation)
                self.update_online_state(url=url, observation=observation)

        def update_history(self, **interaction_dict):
            if ('optimal_navigate' in interaction_dict.get('action', '') and adapter
                    and adapter.last_result.get('status') in {'requires_input', 'requires_local_action'}):
                if adapter.last_result['status'] == 'requires_input':
                    interaction_dict['action'] = ('[TELEPORT PAUSED BEFORE INPUT] Target was not reached. '
                        'Use the current observation for task-specific local input; do not repeat teleport on this state.')
                else:
                    interaction_dict['action'] = ('[TELEPORT INTERRUPTED: STATE MISMATCH] Target arrival is not confirmed. '
                        'Remaining route actions were not executed. Continue the task from the current observation with local actions; '
                        'do not repeat teleport on this unchanged state.')
            return super().update_history(**interaction_dict)

        def are_valid_actions(self, actions):
            if navigator and 'optimal_navigate' in actions and navigator.awaiting_local_action():
                log.emit('teleport_loop_blocked', reason='unchanged_handoff_state')
                raise ValueError('Teleport paused: a local action is required; repeating teleport is invalid')
            return super().are_valid_actions(actions)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.output_play_path = str(log.directory / "play.txt")
            self.output_trash_path = str(log.directory / "trash.txt")
            self.note_log = log

    old_actor = module.Actor
    module.Actor = LoggedActor
    old_domain = module.DomainNavigator
    old_plain = module.CALL_MODEL_MAP["gpt"]
    module.CALL_MODEL_MAP["gpt"] = lambda prompt, **kw: call([{"role": "user", "content": prompt}], **kw)
    old_call = module.CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP["gpt"]
    module.DomainNavigator = Domain
    module.CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP["gpt"] = call
    task_config = dict(task_id="task", sites=[site], intent=task, start_url=start_url, storage_state=str(storage_state) if storage_state else None,
        record_har_path=str(log.directory / "network-transport.har"), allowed_origin=start_url,
        allow_writes=allow_writes, backend=backend, extra_http_headers=headers)
    write_json(log.directory / "browser-task.json", task_config)
    started = time.monotonic()
    result = {"status": "not_started", "answer": "", "steps": 0, "url": start_url, "benchmark_success": None}
    restore_numeric_text = install_numeric_text_guard()
    try:
        env = Environment(config_file=str(log.directory / "browser-task.json"), max_browser_rows=ax_line_limit,
            max_steps=max_steps, slow_mo=0, observation_type="accessibility_tree", current_viewport_only=False,
            viewport_size={"width": 1280, "height": 720}, headless=not headed, global_config=cfg)
        live = LiveBrowser(env, graph, log, start_url, allow_writes)
        if update_graph and use_graph:
            from .graph_update import ExecutionGraph
            updater = ExecutionGraph(graph, live, log, knowledge, description_client, max_descriptions,
                                     card_client, max_cards, index_client=embedding, finalize_assets=finalize_graph_assets,
                                     framework_refresh_updates=framework_refresh_updates,
                                     max_graph_nodes=max_graph_nodes, retention_half_life=retention_half_life,
                                     template_clustering=template_clustering, template_tolerance=template_tolerance,
                                     template_refresh_min=template_refresh_min, template_refresh_fraction=template_refresh_fraction)
            initial = live.observe()
            updater.observe(initial)
            if initial['id'] in updater.graph.nodes:
                updater.graph.data['task_start'] = initial['id']
            updater.wrap_step(env.webarena_env)
            from .recovery import memory_namespace
            updater.graph.data['experience_namespace'] = memory_namespace(graph)
            if framework:
                write_json(updater.graph.directory / 'initial-website-framework.json', framework)
        if use_graph:
            navigator = RouteNavigator(live, graph, log, home_url=home_url)
            if updater:
                navigator.recorder = updater
                updater.navigator = navigator
            adapter = CurrentGraph(graph, navigator, log, knowledge)
            connected = navigator.connect()
            log.emit("graph_connection_ready", connected=connected, actions=navigator.used)
            env.obs = env.webarena_env._get_obs()
        if diagnosis_client:
            from .recovery import Recovery, memory_namespace
            updater.graph.data['experience_namespace'] = memory_namespace(graph)
            recovery = Recovery(task, updater.graph, navigator, log, diagnosis_client, layer=updater.layer,
                framework=framework, memory=experience_store, max_diagnoses=max_diagnoses,
                max_replay_actions=max_replay_actions, seed=recovery_seed,
                historical_records=graph.data.get('transition_records', []),
                page_summarizer=updater.summarize_recovery_page,
                prepare_evidence=updater.poll_descriptions)
            def record_transition(before, action, after, **kwargs):
                recovery.record(before, action, after, execution_step=env.steps, **kwargs)
            updater.on_transition = record_transition
            updater.action_guard = navigator.action_guard = recovery.guard
            navigator.on_url_jump = record_transition
            recovery.start(live.observe())
            check_recovery(live.observe())
            env.obs = env.webarena_env._get_obs()
        env.snapshot()
        # Upstream catches browser exceptions. Retain their message before that catch
        # so the prompt ledger cannot silently present an unsuccessful attempt as executed.
        execute = env.webarena_env.step
        def capture_step_result(command):
            try:
                return execute(command)
            except Exception as error:
                env.current_action_error = type(error).__name__ + ': ' + str(error)
                raise
        env.webarena_env.step = capture_step_result
        agent = module.webNavigator(config=cfg.agent,
            prompt_dict={k: v for k, v in vars(webnavigator_prompt).items() if isinstance(v, dict)})
        outcome = agent.act(objective=env.get_objective(), env=env)
        result["status"] = "agent_stopped" if env.is_done else "step_limit"
        if outcome.get('error'):
            result.update(status='agent_error', error=outcome['error'])
        if final_check:
            result["local_success"] = final_check(live)
    except Exception as error:
        result.update(status="model_error" if isinstance(error, ResponsesError) else "runtime_error",
                      error=type(error).__name__ + ": " + str(error)[:300])
        if isinstance(error, ResponsesError) and error.kind:
            result['provider_failure'] = {'kind': error.kind, 'http_status': error.http_status}
        import traceback
        (log.directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        restore_numeric_text()
        restore_ax()
        module.CALL_MODEL_MAP["gpt"] = old_plain
        module.Actor = old_actor
        module.DomainNavigator = old_domain
        module.CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP["gpt"] = old_call
        if env:
            if live:
                try:
                    final = live.observe(log.directory)
                    write_json(log.directory / 'final_observation.json', {k: final[k] for k in ('id', 'url', 'title', 'headings')})
                except Exception as error:
                    log.emit('final_observation_failed', error=str(error))
            if updater:
                try:
                    if result['status'] == 'model_error' and updater.live_cards:
                        updater.live_cards.stopped = True
                    updater.observe(live.observe())
                except Exception as error:
                    log.emit('execution_final_observation_failed', error=type(error).__name__ + ': ' + str(error))
            actions = [a for a in env.trajectory if "action_type" in a]
            stops = [a for a in actions if a["action_type"] == ActionTypes.STOP]
            result.update(steps=env.steps, atomic_actions=len(actions), url=env.webarena_env.page.url,
                          budget_steps=env.steps - env.refunded_steps, refunded_steps=env.refunded_steps,
                          step_budget=max_steps,
                          answer=stops[-1].get("answer", "") if stops else "")
            write_json(log.directory / "actions.json", json.loads(json.dumps(actions, default=str)))
            if agent:
                write_json(log.directory / "trajectory.json", json.loads(json.dumps(agent.get_trajectory(), default=str)))
            env.webarena_env.context.close()
            env.close()
        result.update(navigation=adapter.last_result if adapter else {}, seconds=round(time.monotonic()-started, 3), task=task)
        from ..common.activity import read_events
        from .action_budget import action_budget_used
        budget_events = read_events(log.path)
        result['action_budget'] = {'policy': ACTION_BUDGET_POLICY,
            'used': action_budget_used(budget_events), 'limit': getattr(client, 'max_calls', None),
            'ax_search_requests': sum(e['event'] == 'action_budget_search' for e in budget_events),
            'selector_requests': len({e['action_budget_id'] for e in budget_events
                if e['event'] == 'responses_started' and e.get('action_budget_policy') == ACTION_BUDGET_POLICY
                and e.get('purpose') == 'page_selector'})}
        if recovery:
            recovery.save()
            result['recovery'] = {'diagnoses': len(recovery.decisions), 'attempted': recovery.attempted,
                                  'step_refund': recovery.step_refund,
                                  'return': recovery.return_result, 'stopped': recovery.stopped}
        write_json(log.directory / "result.json", result)
        (log.directory / "agent_response.json").write_text(result["answer"], encoding="utf-8")
        har = log.directory / "network-transport.har"
        if har.exists():
            from ..common.safety import origin
            data = json.loads(har.read_text(encoding="utf-8"))
            data["log"]["entries"] = [e for e in data["log"]["entries"] if origin(e["request"]["url"]) == origin(start_url)]
            write_json(log.directory / "network.har", data)
        # Browser execution and its evidence are complete before local indexing/card maintenance.
        if updater:
            try:
                result['execution_graph'] = updater.finish(allow_card_calls=result['status'] != 'model_error')
            except Exception as error:
                log.emit('execution_graph_export_failed', error=type(error).__name__ + ': ' + str(error))
                result['execution_graph_error'] = type(error).__name__ + ': ' + str(error)
        result['total_seconds'] = round(time.monotonic() - started, 3)
        result['connection_retry_seconds'] = round(sum(getattr(c, 'retry_seconds', 0) for c in
            (client, description_client, card_client, diagnosis_client) if c is not None), 3)
        if recovery and experience_store and final_check:
            from .recovery import learn_experience
            result['experience_learning'] = learn_experience(log.directory, diagnosis_client, experience_store,
                success=result.get('local_success') is True, verifier='local_final_check')
        write_json(log.directory / "result.json", result)
        log.emit("task_finished", **result)
    return result
