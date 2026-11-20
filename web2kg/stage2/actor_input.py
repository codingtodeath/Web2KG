"""Conservative prompt views; never shorten the stored execution evidence."""
import copy
import re

from ..common.prompt_text import to_yaml

POLICY = 'task-memory-history-v1'


def navigation_context(result, *, query='', domain='', target=None, actual_url=None):
    """A selected target is not proof of arrival. Only replay supplies success."""
    result = result or {}
    status = result.get('status') or ('arrived' if result.get('success') is True else 'not_confirmed')
    row = {'query': query, 'site': domain, 'target': target or result.get('requested_target'),
           'status': status, 'arrival_confirmed': result.get('success') is True}
    if actual_url or result.get('current_url'):
        row['actual_url'] = actual_url or result['current_url']
    for key in ('reason', 'error'):
        if result.get(key):
            row[key] = result[key]
    control = result.get('input_control')
    if control:
        row['input_control'] = {k: v for k, v in control.items() if k in ('role', 'name')}
    if status == 'requires_input':
        row['next'] = 'Paused BEFORE input; use task-specific local input. Do not repeat navigation on this unchanged page.'
    elif status == 'requires_local_action':
        row['next'] = 'Remaining route actions were NOT executed; inspect this page and act locally. Do not repeat navigation on this unchanged page.'
    elif row['arrival_confirmed']:
        row['next'] = 'Inspect the actual page and continue locally; arrival does not establish task completion.'
    else:
        row['next'] = 'Target arrival is NOT confirmed. Inspect the actual page; use local controls/search.'
    return {k: v for k, v in row.items() if v is not None and v != ''}


def compact_old_rows(rows):
    """Exact-only deduplication: no inference about which facts will matter later.

    Keep every action/argument/result. Repeated pre-action fields use an explicit
    earlier-step reference; unique reasoning is retained as intent, not fact.
    Numbers, negation, spelling and whitespace are never normalized away.
    """
    seen, result = {}, []
    for original in rows:
        row = copy.deepcopy(original)
        for field in ('url', 'before', 'highlight', 'intent', 'navigation_intent'):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                continue
            # Only remove an intent when its complete text is already represented.
            if field == 'intent' and value in (original.get('before'), original.get('highlight')):
                del row[field]
                continue
            key = field, value
            if key in seen:
                reference = {field + '_same_as_step': seen[key]}
                # A reference to a two-word fact can cost more than the fact itself.
                if len(to_yaml(reference)) < len(to_yaml({field: value})):
                    del row[field]
                    row.update(reference)
            else:
                seen[key] = row['step']
        result.append(row)
    return result


class ActorInput:
    """Our adapter only; the released Actor keeps its original input/history."""
    def get_task_memory(self):
        # Project legacy notes as well as notes appended by the upstream error path.
        # Keep original timing and contents; the plan tree remains an internal API.
        notes, visited = [], set()
        nodes = [getattr(self, 'plan_tree_root', None), getattr(self, 'active_node', None)]
        while nodes:
            node = nodes.pop(0)
            if node is None or id(node) in visited:
                continue
            visited.add(id(node))
            notes.extend(getattr(node, 'note', []))
            nodes.extend(getattr(node, 'children', []))
        return to_yaml({'notes': notes}) if notes else ''

    def record_outcome(self, outcome):
        if not hasattr(self, 'execution_outcomes'):
            self.execution_outcomes = {}
        self.execution_outcomes[self.get_step()] = copy.deepcopy(outcome)

    def update_history(self, **interaction_dict):
        # The upstream frees the old DOM immediately after this call. Capture only
        # the action's labels first; keep its full arguments in the original log.
        if not hasattr(self, 'action_target_labels'):
            self.action_target_labels = {}
        index = self.get_step()
        observation = interaction_dict.get('observation')
        node = observation.get('node') if isinstance(observation, dict) else None
        self.action_target_labels[index] = self._target_labels(node, interaction_dict.get('action', ''))
        return super().update_history(**interaction_dict)

    def _history_value(self, key, index, default=''):
        values = self.previous_interactions.get(key, [])
        return values[index] if index < len(values) else default

    def _history_highlight(self, index):
        node = self._history_value('observation highlight', index, None)
        if not node:
            return ''
        if isinstance(node, str):
            return node
        from webNavigator.obs_opt import translate_node_to_str
        return translate_node_to_str(node, mode='name_only',
            retained_ids=self._history_value('retained element ids', index, []))

    def _target_labels(self, node, action):
        ids = self.parse_actions_to_element_ids(action)
        if not ids or node is None:
            return []
        labels = []
        for identifier in ids:
            target = node.search_node_by_id(identifier)
            if target is not None and str(target.node_id) == str(identifier):
                labels.append({'role': target.role, 'name': target.name})
        return labels

    def get_interaction_history(self, interaction_history_config=False, mode='highlight'):
        config = interaction_history_config or self.config.interaction_history
        steps = self.active_node.steps_taken
        if not steps:
            return []
        recent = len(steps) if config.step_num == 'all' else max(0, int(config.step_num))
        boundary = max(0, len(steps) - recent)
        rows = []
        for index in steps:
            action = self._history_value('action', index)
            row = {'step': index, 'url': self._history_value('url', index),
                   'before': self._history_value('observation summary', index)}
            highlight = self._history_highlight(index)
            if highlight and highlight != row['before']:
                row['highlight'] = highlight
            if not row['before'] and not highlight:
                observation = self._history_value('observation', index)
                text = observation.get('text', '') if isinstance(observation, dict) else observation
                if isinstance(text, str) and len(text.split()) < 200:
                    row['before'] = text
            reason = self._history_value('reason', index)
            if reason:
                row['intent'] = reason
            thought = self._history_value('navigate_think', index)
            if thought and thought not in (reason, row.get('before'), highlight,
                                            self._history_value('navigate_query', index)):
                # A navigation rationale can contain data not repeated in the query.
                # Remove only exact redundancy, not potentially useful unique content.
                row['navigation_intent'] = thought
            row['action'] = action
            outcome = getattr(self, 'execution_outcomes', {}).get(index)
            if outcome:
                if outcome.get('navigation'):
                    # Suppress the verbose legacy expansion and the duplicated thinking.
                    row['action'] = 'optimal_navigate'
                row['result'] = copy.deepcopy(outcome)
            elif re.match(r'^optimal_navigate\b', action):
                row['action'] = 'optimal_navigate'
                row['result'] = {'navigation': navigation_context({},
                    query=self._history_value('navigate_query', index),
                    domain=self._history_value('navigate_domain', index),
                    target=self._history_value('target_page', index, None))}
            labels = getattr(self, 'action_target_labels', {}).get(index, [])
            if labels:
                row['targets'] = labels
            rows.append({k: v for k, v in row.items() if v != ''})
        text = ('Historical AX IDs are not current controls. before/highlight and intent are pre-action '
                'Actor evidence, not proof the action succeeded. executed means the browser command returned, '
                'not verified business success. Errors may follow partial execution: verify before retrying writes. '
                '*_same_as_step references identical text in that earlier row.\n')
        if boundary:
            text += 'OLDER ACTION LEDGER:\n' + to_yaml(compact_old_rows(rows[:boundary]))
        text += 'RECENT STEPS:\n' + to_yaml(rows[boundary:])
        messages = [('text', text)]
        if 'image' in config.type:
            # Preserve optional recent image history; current production config is text-only.
            for index in steps[boundary:]:
                observation = self._history_value('observation', index, {})
                image = observation.get('image') if isinstance(observation, dict) else None
                if image is not None:
                    messages.extend([('text', f'Historical step {index} image:\n'), ('image', image)])
        return self.prune_message_list(messages)

    def get_online_input(self, criticism_elements):
        prefix, suffix = self.prompt_template['input_template'].split('{input}')
        values = {'step': self.get_step(), 'objective': self.objective,
                  'task memory': self.get_task_memory(),
                  'interaction history': self.get_interaction_history(),
                  'current observation': self.get_observation_text()}
        messages, seen = [('text', prefix)], set()
        for configured in self.config.input:
            field = 'task memory' if configured == 'previous plans' else configured
            if field in seen or field == 'current visual observation':
                continue
            seen.add(field)
            value = values.get(field)
            if field.startswith('critic: ') and criticism_elements:
                value = criticism_elements.get(field[len('critic: '):])
                field = 'FROM USER: ' + field[len('critic: '):]
            if field == 'step':
                messages.append(('text', f'CURRENT STEP:\n{value}\n'))
            elif value:
                messages.append(('text', field.upper() + ':\n'))
                messages.extend(value if isinstance(value, list) else [('text', str(value) + '\n')])
        if 'image' in self.config.current_observation.type:
            messages.extend([('text', 'CURRENT VISUAL OBSERVATION:\n'),
                             ('image', self.get_observation_image())])
        messages.append(('text', suffix))
        return self.prune_message_list(messages)
