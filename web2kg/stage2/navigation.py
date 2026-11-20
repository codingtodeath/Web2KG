"""Recorded routes from the live state or logical root URL, with input handoffs."""
import copy
from ..common.interaction import VERSION as STATE_VERSION, state_matches, control_signature
from ..common.browser import canonical_url
from ..common.storage import Graph
from ..common.replay_policy import INPUT_KINDS
from ..stage1.page_admission import admitted


class RouteNavigator:
    def __init__(self, browser, graph, log, home_url=None, max_bridge_actions=12, max_bridge_depth=2, strategy="role"):
        self.browser, self.base, self.log = browser, graph, log
        self.graph = Graph(log.directory / "navigation")
        self.graph.data = copy.deepcopy(graph.data)
        from ..common.state_equivalence import StateResolver
        self.resolver = StateResolver(self.graph, log=log) if graph.data.get('state_equivalence') else None
        if self.resolver is not None:
            browser.state_resolver = self.resolver
        elif getattr(browser, 'state_resolver', None) is not None:
            browser.state_resolver = None
        self.graph.initialize_url_entries()
        self.home_url = canonical_url(home_url or browser.start_url)
        self.explicit_home = home_url is not None
        if strategy not in {'role', 'xpath'}:
            raise ValueError('Unknown replay locator strategy')
        self.strategy = strategy
        self.used = 0
        self.on_action = self.recorder = self.action_guard = self.on_url_jump = None
        self.local_pause = None
        self.root_url = canonical_url(graph.nodes[graph.data['root']]['url'])

    def root_sources(self):
        entry = self.graph.data.get('root_entry', {})
        return {k for k in [self.base.data['root'], *entry.get('states', [])]
                if k in self.graph.nodes and canonical_url(self.graph.nodes[k]['url']) == self.root_url}

    def remember_root(self, observed):
        """Replace the live entry binding, preserving historical state IDs/evidence."""
        if canonical_url(observed['url']) != self.root_url or not admitted(observed):
            return False
        key = observed['id']
        entry = self.graph.data.get('root_entry', {})
        if entry.get('current_state') == key:
            return True
        if key not in self.graph.nodes:
            self.browser.save_observation(observed, self.graph.directory)
            self.graph.add_node(observed, [])
        entry = {'policy': 'url-entry-current-state-v1', 'url': self.root_url,
                 'reference_state': self.base.data['root'], 'current_state': key,
                 'states': sorted(self.root_sources() | {key})}
        self.graph.data['root_entry'] = entry
        if self.recorder:
            self.recorder.observe(observed)
            self.recorder.graph.data['root_entry'] = copy.deepcopy(entry)
            if not getattr(self.recorder, '_batching_record', False):
                self.recorder.save()
        self.log.emit('navigation_root_bound', url=self.root_url, state=key)
        if not self.recorder or not getattr(self.recorder, '_batching_record', False):
            self.save()
        return True

    def retrieval_scope(self, observed):
        if self.recorder:
            self.recorder.poll_descriptions()
        self.remember_root(observed)
        current = self.identify(observed)
        allowed = set(self.graph.reachable(current)) if current is not None else set()
        for root in self.root_sources():
            allowed.update(self.graph.reachable(root))
        return current, allowed

    @staticmethod
    def root_action(action, observed):
        """Bind historical intent to one real current control, never a stale nth()."""
        semantic = ('role', 'name', 'href', 'type', 'method', 'form_action', 'data_mutation')
        signature = control_signature(action)
        candidates = [a for a in observed['actions'] if not a.get('disabled') and all(
            control_signature(a).get(k, '') == signature.get(k, '') for k in semantic)]
        if len(candidates) > 1:
            region = action.get('region', {})
            candidates = [a for a in candidates if all(a.get('region', {}).get(k) == region.get(k)
                          for k in ('role', 'name'))]
        if len(candidates) != 1:
            raise ValueError('Root control missing or ambiguous in current page')
        return {**candidates[0], **{k: action[k] for k in ('kind', 'value') if k in action}}

    def save(self):
        self.graph.data['connection'] = {'home_url': self.home_url, 'actions': self.used,
            'policy': 'recorded-entry-v1', 'base_graph': str(self.base.directory),
            'base_root': self.base.data['root']}
        self.graph.save()

    def identify(self, observation):
        if self.resolver is not None:
            resolved = self.resolver.resolve(observation)
            if self.resolver.matches(resolved['id'], observation):
                return resolved['id']
        key = observation['id']
        if key in self.graph.nodes and state_matches(self.graph.nodes[key], observation):
            return key
        matches = [key for key, node in self.graph.nodes.items() if state_matches(node, observation)]
        if len(matches) == 1:
            self.log.emit('navigation_state_match', observed=observation['id'], saved=matches[0],
                          rule=STATE_VERSION)
            return matches[0]
        return None

    def matches(self, key, observation):
        return (self.resolver.matches(key, observation) if self.resolver is not None
                else state_matches(self.graph.nodes[key], observation))

    def capture(self):
        observed = self.browser.observe(self.graph.directory)
        if not admitted(observed):
            return observed, None
        key = self.identify(observed)
        if key is None:
            self.graph.add_node(observed, [])
            key = observed['id']
        return observed, key

    def connect(self):
        """Register the actual task initial state; do not explore or reset the page."""
        observed, initial = self.capture()
        if initial is None:
            self.log.emit('navigation_initial_state_excluded', reason=observed['page_content']['kind'])
            return False
        self.graph.data['task_start'] = initial
        self.graph.register_url_entry(initial, 'observed_task_initial_state')
        homes = [k for k, entry in self.graph.data['url_entries'].items() if entry['url'] == self.home_url]
        if not self.explicit_home or self.home_url == observed['url']:
            home = initial
            self.home_url = observed['url']
        elif len(homes) == 1:
            home = homes[0]
        else:
            raise ValueError('Home URL must identify a recorded initial entry state')
        self.graph.data['navigation_home'] = home
        self.remember_root(observed)
        if self.recorder:
            self.recorder.observe(observed)
            self.recorder.graph.initialize_url_entries()
            self.recorder.graph.register_url_entry(initial, 'observed_task_initial_state')
            self.recorder.save()
        self.save()
        return self.graph.path(initial, self.base.data['root']) is not None

    def awaiting_local_action(self, observed=None):
        if self.local_pause is None:
            return False
        observed = observed or self.browser.observe()
        previous = self.local_pause['observation']
        same = state_matches(previous, observed)
        if not same and self.resolver is not None:
            left, right = self.resolver.resolve(previous), self.resolver.resolve(observed)
            same = left['id'] == right['id'] and self.resolver.matches(left['id'], observed)
        if not same:
            self.local_pause = None
            return False
        return True

    def awaiting_input(self, observed=None):
        # Compatibility for callers of the original input-only handoff guard.
        return self.awaiting_local_action(observed)

    def feedback(self):
        if not self.awaiting_local_action():
            return None
        return {k: v for k, v in self.local_pause.items() if k != 'observation'}

    def state_handoff(self, observed, edge, target, steps, phase):
        result = {'success': False, 'status': 'requires_local_action',
            'reason': phase + '_mismatch', 'steps': steps, 'edge': edge['id'],
            'requested_target': target, 'expected_state': edge['source' if phase == 'source' else 'target'],
            'observed_state': observed['id'], 'current_url': observed['url'],
            'instruction': 'Teleport interrupted because the observed state differs from the recorded route. '
                'The browser remains on the current page; remaining route actions were NOT executed and target arrival is NOT confirmed. '
                'Inspect the fresh observation and continue the task with local actions, or stop if the task is already complete. '
                'Do not repeat optimal_navigate on this unchanged state.'}
        self.local_pause = {**result, 'observation': copy.deepcopy(observed)}
        self.log.emit('replay_state_handoff', **result)
        return result

    def home(self):
        target = self.graph.data.get('navigation_home')
        result = self.return_to_observed(target)
        if not result['success']:
            raise ValueError('Cannot return to initial entry: ' + result['reason'])
        return self.capture()

    def replay(self, path, *, flexible_root=False):
        for index, edge in enumerate(path):
            observed = self.browser.observe(self.graph.directory)
            at_root = (flexible_root and edge['source'] in self.root_sources()
                       and canonical_url(observed['url']) == self.root_url)
            if not at_root and not self.matches(edge['source'], observed):
                return self.state_handoff(observed, edge, path[-1]['target'], index, 'source')
            action = edge['action']
            if not self.graph.edge_replayable(edge):
                return {'success': False, 'steps': index, 'reason': 'transition_not_replayable'}
            if action.get('kind') in INPUT_KINDS:
                result = {'success': False, 'status': 'requires_input', 'reason': 'actor_input_required',
                    'steps': index, 'source': edge['source'], 'edge': edge['id'],
                    'input_control': {k: action[k] for k in ('role', 'name', 'xpath') if k in action},
                    'instruction': 'Teleport paused BEFORE input. Inspect the current observation and perform a local action with task-specific input. Do not repeat optimal_navigate on this unchanged state; the historical input and destination were NOT replayed.'}
                self.local_pause = {**result, 'observation': copy.deepcopy(observed)}
                self.log.emit('replay_input_required', **result)
                return result
            if self.action_guard and self.action_guard(observed, action):
                return {'success': False, 'steps': index, 'reason': 'counterfactual_control_forbidden'}
            attempted = recorded = False
            try:
                if at_root:
                    self.remember_root(observed)
                    if action.get('kind') != 'url_jump' and not self.matches(edge['source'], observed):
                        action = self.root_action(action, observed)
                if self.recorder:
                    self.recorder.observe(observed)
                    self.recorder.capture_before(observed)
                attempted = True
                if action.get('kind') == 'url_jump':
                    self.browser.page.goto(action['url'], wait_until='domcontentloaded', timeout=20000)
                    self.browser.settle()
                else:
                    self.browser.act(action, self.strategy)
                if self.on_action:
                    self.on_action(observed, action)
                after = self.browser.observe(self.graph.directory)
                if action.get('kind') == 'url_jump':
                    if self.recorder:
                        self.recorder.observe(after)
                        self.recorder.events.emit('entry_url_jump', edge=edge['id'], source=edge['source'],
                                                  observed=after['id'])
                    if self.on_url_jump:
                        self.on_url_jump(observed, {**action, 'role': 'browser'}, after, phase='replay')
                elif self.recorder:
                    self.recorder.record(observed, action, after, phase='replay')
                    recorded = True
                root_arrival = flexible_root and edge['target'] in self.root_sources() and self.remember_root(after)
                if not root_arrival and not self.matches(edge['target'], after):
                    self.log.emit('replay_step', edge=edge['id'], step=index, success=False,
                                  error='Recorded destination state changed', status='requires_local_action')
                    return self.state_handoff(after, edge, path[-1]['target'], index + 1, 'destination')
                self.log.emit('replay_step', edge=edge['id'], step=index, action=action, success=True)
            except Exception as error:
                if at_root and not attempted:
                    result = self.state_handoff(observed, edge, path[-1]['target'], index, 'source')
                    result.update(reason='root_control_unavailable', error=str(error))
                    self.local_pause.update(result)
                    return result
                if self.recorder and attempted and not recorded and action.get('kind') != 'url_jump':
                    self.recorder.record(observed, action, None, phase='replay', error=type(error).__name__)
                self.log.emit('replay_step', edge=edge['id'], step=index, success=False, error=str(error))
                return {'success': False, 'steps': index + int(attempted), 'reason': 'replay_failed', 'error': str(error)}
        return {'success': True, 'steps': len(path)}

    def root_path(self, target):
        paths = [p for root in sorted(self.root_sources()) if (p := self.graph.path(root, target)) is not None]
        return min(paths, key=len) if paths else None

    def goto_root(self, before):
        """Visit the logical entry in the same session, preserving the actual jump evidence."""
        action = {'kind': 'url_jump', 'url': self.root_url, 'role': 'browser'}
        self.browser.page.goto(self.root_url, wait_until='domcontentloaded', timeout=20000)
        self.browser.settle()
        if self.on_action:
            self.on_action(before, action)
        after = self.browser.observe(self.graph.directory)
        if not self.remember_root(after):
            raise ValueError('Entry goto did not reach its URL or returned a blank page')
        if self.on_url_jump:
            self.on_url_jump(before, action, after, phase='root_entry')
        self.log.emit('navigation_root_goto', url=self.root_url, observed_state=after['id'])
        return after

    def navigate(self, target, on_action=None):
        self.on_action = on_action
        if target not in self.base.nodes:
            raise ValueError('Target is not an indexed graph node')
        if self.awaiting_local_action():
            result = {'success': False, 'steps': 0, 'reason': 'teleport_requires_local_action',
                      'status': self.local_pause['status']}
        else:
            observed = self.browser.observe()
            self.remember_root(observed)
            current = self.identify(observed)
            if self.recorder:
                self.recorder.wait_for_route([current, *sorted(self.root_sources())], target)
            path = self.graph.path(current, target)
            entry_steps = 0
            if path is None:
                path = self.root_path(target)
                if path is not None and canonical_url(observed['url']) != self.root_url:
                    try:
                        entry_steps = 1
                        observed = self.goto_root(observed)
                    except Exception as error:
                        result = {'success': False, 'steps': entry_steps, 'reason': 'root_entry_failed',
                                  'error': str(error)}
                        self.log.emit('navigation', target=target, route='root_entry', **result)
                        self.save()
                        return result
                if path is not None:
                    # The goto may restore a known root version with its own
                    # usable route (for example, a menu that must first open).
                    live_path = self.graph.path(self.identify(observed), target)
                    if live_path is not None:
                        path = live_path
            result = self.replay(path, flexible_root=True) if path is not None else {
                'success': False, 'steps': 0, 'reason': 'no_recorded_route_from_current_or_root'}
            result['steps'] += entry_steps
            if self.local_pause is not None and result.get('status') in {'requires_input', 'requires_local_action'}:
                self.local_pause['steps'] = result['steps']
        self.log.emit('navigation', target=target, route='recorded', **result)
        self.save()
        return result

    def return_to_observed(self, target, max_actions=12):
        """Prefer the current route, then the logical root; budget includes the entry goto."""
        if target not in self.graph.nodes:
            return {'success': False, 'steps': 0, 'reason': 'root_not_in_observed_graph'}
        observed = self.browser.observe()
        self.remember_root(observed)
        current = self.identify(observed)
        if self.recorder:
            self.recorder.wait_for_route([current], target, max_actions=max_actions)
        path = self.graph.path(current, target)
        if path is not None and len(path) <= max_actions:
            # A failed browser replay is not a reason to retry through another route.
            return {**self.replay(path), 'route': 'recorded'}

        entry_steps = int(canonical_url(observed['url']) != self.root_url)
        remaining = max_actions - entry_steps
        if self.recorder and remaining >= 0:
            self.recorder.wait_for_route(sorted(self.root_sources()), target, max_actions=remaining)
        path = self.root_path(target)
        if path is None or len(path) > remaining:
            return {'success': False, 'steps': 0, 'reason': 'no_recorded_route_within_cap'}
        if entry_steps:
            try:
                observed = self.goto_root(observed)
            except Exception as error:
                self.save()
                return {'success': False, 'steps': 1, 'route': 'root_entry',
                        'reason': 'root_entry_failed', 'error': str(error)}
        current = self.identify(observed)
        if self.recorder:
            self.recorder.wait_for_route([current], target, max_actions=remaining)
        # A known live root may need extra actions (e.g. opening a closed menu).
        # Prefer its actual route and check the budget again before replaying.
        live_path = self.graph.path(current, target)
        if live_path is not None:
            path = live_path
        if len(path) > remaining:
            result = {'success': False, 'steps': 0, 'reason': 'no_recorded_route_within_cap'}
        else:
            result = self.replay(path, flexible_root=True)
            # A flexible homepage binding must not masquerade as an exact recovery
            # to an old observed state and incorrectly earn a step refund.
            if result['success']:
                after = self.browser.observe()
                if not self.matches(target, after):
                    result.update(success=False, reason='return_target_mismatch',
                                  expected_state=target, observed_state=after['id'])
        result['steps'] += entry_steps
        if self.local_pause is not None and result.get('status') in {'requires_input', 'requires_local_action'}:
            self.local_pause['steps'] = result['steps']
        self.save()
        return {**result, 'route': 'root_entry'}


def navigate(browser, graph, target, log, strategy='role', on_action=None):
    return RouteNavigator(browser, graph, log, strategy=strategy).navigate(target, on_action)
