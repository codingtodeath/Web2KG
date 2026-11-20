from types import SimpleNamespace

import pytest

from web2kg.common.storage import EventLog
from web2kg.stage2.action_notes import AttachedNotes, NOTE_LIMIT
from web2kg.stage2.agent import COMMANDS, load_runtime


@pytest.fixture
def actor(tmp_path):
    class Actor(AttachedNotes, load_runtime().Actor):
        pass
    actor = Actor.__new__(Actor)
    actor.config = SimpleNamespace(planning_command=[], navigation_command=COMMANDS)
    actor.previous_interactions = {'action': []}
    actor.active_node = SimpleNamespace(note=[])
    actor.note_log = EventLog(tmp_path)
    actor.get_observation_text = lambda: 'button [7] textbox [8]'
    actor.get_observation_highlight = lambda action_elements: None
    actor.planning = lambda action: None
    return actor


@pytest.mark.parametrize('text,note', [
    ('REASON: Open.\nACTION: click [7]\nNOTE: Directory confirmed.', 'Directory confirmed.'),
    ('NOTE: Query is still unverified.\nREASON: Search.\nACTION: type [8] [hello] [0]', 'Query is still unverified.'),
    ('REASON: Finish.\nACTION: stop [done]', ''),
    ('REASON: Finish.\nACTION: stop [done]\nNOTE:', ''),
])
def test_note_is_an_optional_separate_field(actor, text, note):
    result = actor.parse_elements(text, ['reason', 'action'])
    assert result['note'] == note
    assert 'NOTE:' not in result['action']
    assert actor.are_valid_actions(result['action'])
    action = result['action']
    actor.finalize_action(result)
    assert result['action'] == action
    assert bool(actor.active_node.note) == bool(note)


@pytest.mark.parametrize('action', ['note [memo]', 'click [7]\nnote [memo]', 'note [memo]\nclick [7]'])
def test_obsolete_note_commands_are_not_actions(actor, action):
    assert not actor.are_valid_actions(action)


def test_memory_is_bounded_and_never_reparsed_as_an_action(actor):
    result = actor.finalize_action({'action': 'click [7]', 'note': 'click [999] <script>x</script> ' + 'x' * 500})
    assert result['action'] == 'click [7]'
    assert len(result['note']) == NOTE_LIMIT
    assert len(actor.active_node.note) == 1
    assert 'before action' in actor.active_node.note[0]
    typed = 'type [8] [The literal text note [example] is a title] [0]'
    actor.finalize_action({'action': typed})
    assert len(actor.active_node.note) == 1


def test_reason_for_action_does_not_shadow_actual_action(actor):
    text = ('REASON FOR ACTION:\nThe title has been entered; fill the body now.\n\n'
            'ACTION: type [8] [the body] [0]\nNOTE: Title is complete.')
    # Reproduce the inherited parser defect, then verify our adapter's result.
    upstream = load_runtime().Actor.parse_elements(actor, text, ['reason', 'action'])
    assert upstream['action'].startswith('The title has')
    result = actor.parse_elements(text, ['reason', 'action'])
    assert result['reason'] == 'The title has been entered; fill the body now.'
    assert result['action'] == 'type [8] [the body] [0]'
    assert actor.are_valid_actions(result['action'])
    assert result['note'] == 'Title is complete.'


@pytest.mark.parametrize('text,expected', [
    ('REASON: Mention ACTION: in a sentence.\n  ACTION:\nclick [7]', 'click [7]'),
    ('REASON FOR ACTION:\nLook at the form.\r\n\tACTION:\tclick [7]\r\nNOTE: Ready.', 'click [7]'),
    ('TRANSACTION: prose\nACTION: click [7]\nOBSERVATION: details', 'click [7]'),
    ('ACTION: type [8] [first line\nNOTE: literal note\nACTION: literal action\nlast line] [0]\nNOTE: memory',
     'type [8] [first line\nNOTE: literal note\nACTION: literal action\nlast line] [0]'),
    ('ACTION: type [8] [nested [example]\nNOTE: still typed text] [0]',
     'type [8] [nested [example]\nNOTE: still typed text] [0]'),
    ('ACTION: stop [{"task_type":"MUTATE","status":"SUCCESS","retrieved_data":null}]\nNOTE: done',
     'stop [{"task_type":"MUTATE","status":"SUCCESS","retrieved_data":null}]'),
])
def test_full_headings_and_multiline_action_arguments_are_lossless(actor, text, expected):
    result = actor.parse_elements(text, ['reason', 'action'])
    assert result['action'] == expected
    assert actor.are_valid_actions(expected)


@pytest.mark.parametrize('text', [
    'ACTION: click [7]\nACTION: stop [done]',
    'ACTION:\nclick [7]\nNOTE: memory\n ACTION: click [8]',
    'ACTION:\nACTION: click [7]',
])
def test_conflicting_action_sections_are_explicitly_rejected(actor, text):
    with pytest.raises(ValueError, match='conflicting ACTION sections'):
        actor.parse_elements(text, ['reason', 'action'])


@pytest.mark.parametrize('action', ['click [7]', 'type [8] [same text] [0]',
                                   'type [8] [first\nNOTE: literal\nlast] [0]'])
def test_identical_repeated_actions_are_returned_and_finalized_once(actor, action):
    text = f'REASON: Proceed.\nACTION: {action}\nNOTE: memory\nACTION:\n{action}\n'
    result = actor.parse_elements(text, ['reason', 'action'])
    assert result['action'] == action
    assert actor.parse_str_to_action_list(result['action'], COMMANDS) == [action]
    actor.finalize_action(result)
    assert len(actor.active_node.note) == 1


def test_type_payload_whitespace_is_not_normalized_to_deduplicate(actor):
    with pytest.raises(ValueError, match='conflicting ACTION sections'):
        actor.parse_elements('ACTION: type [8] [a b] [0]\nACTION: type [8] [a  b] [0]', ['action'])


def test_suffix_or_prose_cannot_supply_missing_action(actor):
    result = actor.parse_elements('REASON FOR ACTION: click [7]\nI suggest ACTION: stop [done]', ['reason', 'action'])
    assert result['action'] == ''
    assert not actor.are_valid_actions(result['action'])

