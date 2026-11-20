"""Optional memory beside an action, never a browser command."""
import re
from .actor_input import ActorInput

NOTE_LIMIT = 240
SECTION_HEADER = re.compile(r'^[ \t]*([A-Z][A-Z\d_ \t]*):[ \t]*(.*)')


def parse_sections(text, keys, *, require_single_action=False):
    """Read complete heading lines; headings inside bracketed action arguments are text."""
    fields = {key: '' for key in keys}
    current, lines, seen, depth = None, [], set(), 0

    def finish():
        value = ''.join(lines).strip()
        if require_single_action and current == 'action' and current in seen:
            raise ValueError('Expected exactly one ACTION section; multiple sections were returned')
        if current == 'action' and current in seen and value != fields[current]:
            raise ValueError('Ambiguous Actor output: conflicting ACTION sections')
        if current in fields and current not in seen:
            fields[current] = value
            seen.add(current)

    for line in text.splitlines(keepends=True):
        heading = SECTION_HEADER.match(line) if not (current == 'action' and depth > 0) else None
        if heading:
            finish()
            current = ' '.join(heading[1].lower().split())
            if current == 'reason for action' and 'reason' in fields:
                current = 'reason'
            lines, depth = [], 0
            line = line[heading.start(2):]
        if current is not None:
            lines.append(line)
            if current == 'action':
                # The existing command language uses nested square-bracket arguments.
                depth += line.count('[') - line.count(']')
    finish()
    return fields


from .actor_prompt import NOTE_SPEC, HIGHLIGHT_LIMIT


def bounded_highlights(raw, observation):
    """Keep visible AX IDs in model priority order; never treat prose numbers as IDs."""
    visible = set(re.findall(r'^[ \t]*[A-Za-z][\w-]*[ \t]+\[(\d+)\](?=[ \t]|$)',
                             observation or '', re.M))
    selected, seen = [], set()
    counts = dict(invalid=0, duplicates=0, over_limit=0)
    for part in raw.split(','):
        token = part.strip()
        if not token:
            continue
        if token not in visible:
            counts['invalid'] += 1
        elif token in seen:
            counts['duplicates'] += 1
        else:
            seen.add(token)
            if len(selected) < HIGHLIGHT_LIMIT:
                selected.append(token)
            else:
                counts['over_limit'] += 1
    return ', '.join(selected), counts


class AttachedNotes(ActorInput):
    """Mixin for our execution adapter; the pinned upstream baseline is unchanged."""
    def get_output_specifications(self):
        from .actor_prompt import output_specifications
        return output_specifications(self.config.output)

    def get_actor_instruction(self, examples=None):
        from .actor_prompt import instruction
        return instruction(self)

    def parse_elements(self, text, key_list):
        return parse_sections(text, list(dict.fromkeys([*key_list, 'note'])))

    def are_valid_actions(self, actions):
        # Recognize obsolete note commands so they cannot hide behind a valid click.
        commands = self.config.planning_command + self.config.navigation_command + ['goto', 'note']
        parts = self.parse_str_to_action_list(actions, commands)
        from ..evaluation.actor_output import single_action
        return len(parts) == 1 and single_action(actions) and not re.match(r'^note\b', parts[0]) and self.is_valid_action(parts[0])

    def take_note(self, action, note_as_action=True):
        # Literal "note [...]" in typed content must never become memory.
        return action

    def get_observation_highlight(self, action_elements):
        raw = action_elements.get('observation highlight', '')
        selected, counts = bounded_highlights(raw, self.get_observation_text())
        action_elements['observation highlight'] = selected
        if any(counts.values()):
            self.note_log.emit('actor_highlight_filtered', step=self.get_step() + 1,
                               kept=selected, limit=HIGHLIGHT_LIMIT, **counts)
        if not selected:
            # Empty highlights use the observation summary in history, not the full DOM.
            action_elements['observation highlight idxs'] = ''
            action_elements['observation highlight'] = None
            action_elements['retained element ids'] = self.parse_actions_to_element_ids(action_elements['action'])
            return
        return super().get_observation_highlight(action_elements)

    def finalize_action(self, action_elements):
        result = super().finalize_action(action_elements)
        text = ' '.join(result.get('note', '').split())
        note = text[:NOTE_LIMIT]
        result['note'] = note
        if note:
            self.active_node.note.append(f'STEP {self.get_step()} (before action): {note}')
            self.note_log.emit('agent_note', step=self.get_step() + 1, note=note,
                               timing='before_action', truncated=len(text) > NOTE_LIMIT)
        return result
