"""Replay semantics belong to Luna; text input is an Actor handoff boundary."""
from urllib.parse import urlsplit

INPUT_KINDS = {'type', 'fill'}


def executable(action, start_url=None, target_url=None):
    kind = action.get('kind')
    if kind == 'go_back':
        return True, 'browser_history_with_arrival_check'
    if kind == 'url_jump':
        parts = urlsplit(action.get('url', ''))
        return (True, 'recorded_url') if parts.scheme in {'http', 'https'} and parts.netloc else (False, 'invalid_url')
    if kind not in {'click', 'type', 'fill', 'press', 'select'}:
        return False, 'unsupported_replay_action'
    if not action.get('xpath') and not (action.get('role') and 'name' in action):
        return False, 'missing_replay_locator'
    if kind in {'press', 'select'} and (not isinstance(action.get('value'), str) or not action['value']):
        return False, 'missing_replay_value'
    if action.get('disabled'):
        return False, 'unavailable_control'
    return True, 'actor_input_boundary' if kind in INPUT_KINDS else 'executable'


def awaiting_execution_description(edge):
    annotation = edge.get('annotation', edge.get('description', {}).get('annotation', {}))
    return (edge.get('replay_policy') == 'execution-description-v2'
            and annotation.get('status') != 'completed')


def replay_allowed(edge, start_url=None):
    if awaiting_execution_description(edge):
        return False
    if edge['action'].get('kind') in INPUT_KINDS:
        return True  # Reachability includes the destination; playback never executes this input.
    return edge.get('replayable', True) is True and executable(edge['action'])[0]


def mark_input_boundary(edge):
    if awaiting_execution_description(edge):
        edge['replayable'] = False
        return
    if edge['action'].get('kind') in INPUT_KINDS:
        edge.update(replayable=True, replay_policy='actor-input-boundary-v1',
                    replay_reason='Pause before input; Actor supplies task-specific content.')
