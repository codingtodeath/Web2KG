import importlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from web2kg.common.activity import read_events
from web2kg.common.storage import EventLog, Graph
from web2kg.stage2.action_notes import AttachedNotes, bounded_highlights
from web2kg.stage2.actor_prompt import CONTEXT_POLICY, OUTPUT_FIELDS, instruction
from web2kg.stage2.agent import configuration, load_runtime, namespace, run_task
from test_verified_stop_retry import client, VALID


def test_configuration_and_prompt_use_four_fields(tmp_path):
    cfg = configuration(EventLog(tmp_path))['agent']['actor']
    example = yaml.safe_load(Path('examples/webnavigator-execution.yml').read_text(encoding='utf-8'))['agent']['actor']
    for key in ('output', 'context_policy', 'input_policy', 'input', 'history_mode_analysis'):
        assert example[key] == cfg[key]
    assert cfg['output'] == ['observation description', 'reason', 'action', 'observation highlight']
    assert cfg['context_policy'] == CONTEXT_POLICY
    assert not cfg['history_mode_analysis']
    prompt = instruction(SimpleNamespace(config=namespace(cfg)))
    for field in OUTPUT_FIELDS:
        assert prompt.count('\n' + field.upper() + ':') == 1
    for field in ('repeat final object', 'extract constraints', 'interaction history summary', 'mode analysis'):
        assert '\n' + field.upper() + ':' not in prompt
    assert 'at most 5' in prompt and 'Usually 1-3' in prompt and 'NOTE:' in prompt


def test_highlights_only_accept_displayed_ids_in_priority_order():
    observation = "button [8] 'Has text [999]'\n  link [7] 'Catalog'\nStaticText [9] 'Price'"
    kept, counts = bounded_highlights('9, 999, 8, 8, 7, [7], 007', observation)
    assert kept == '9, 8, 7'
    assert counts == {'invalid': 3, 'duplicates': 1, 'over_limit': 0}


def test_hundreds_of_highlights_are_bounded_without_regeneration():
    observation = '\n'.join(f"button [{i}] 'Control'" for i in range(600))
    kept, counts = bounded_highlights(','.join(str(i) for i in range(599, -1, -1)), observation)
    assert kept == '599, 598, 597, 596, 595'
    assert counts['over_limit'] == 595


def test_search_revealed_ids_become_eligible_only_when_displayed():
    load_runtime()
    Search = importlib.import_module('webNavigator.ax_search').AXSearch
    search = Search('\n'.join(f"button [{i}] 'Target{i}'" for i in range(1800)))
    assert bounded_highlights('1700', search.observation_text())[0] == ''
    search.search('search_ax [Target1700]')
    assert bounded_highlights('1700', search.observation_text())[0] == '1700'


@pytest.mark.parametrize('raw', ['', '999'])
def test_empty_highlights_fall_back_without_expanding_dom(tmp_path, raw):
    class Actor(AttachedNotes, load_runtime().Actor):
        pass
    actor = Actor.__new__(Actor)
    actor.note_log = EventLog(tmp_path)
    actor.get_step = lambda: 0
    actor.get_observation_text = lambda: "button [7] 'Next'"
    actor.get_observation_node = lambda: pytest.fail('Empty highlight must not expand a DOM')
    actor.parse_actions_to_element_ids = lambda action: [7]
    fields = {'action': 'click [7]', 'observation highlight': raw, 'observation description': 'Ready.'}
    actor.get_observation_highlight(fields)
    assert fields['observation highlight'] is None
    assert fields['observation highlight idxs'] == ''
    assert fields['retained element ids'] == [7]
    assert fields['observation description'] == 'Ready.'


@pytest.mark.parametrize('compact', [False, True])
def test_history_omits_mode_only_when_opted_in(compact):
    actor = load_runtime().Actor.__new__(load_runtime().Actor)
    # Legacy configs lack this option and keep their original history format.
    actor.config = SimpleNamespace(**({'history_mode_analysis': False} if compact else {}))
    actor.active_node = SimpleNamespace(id=0, steps_taken=[0])
    actor.previous_interactions = {
        'observation summary': ['Verified fact'], 'plan': [''], 'reason': ['Open next page'],
        'action': ['click [7]'], 'mode analysis': ['LOW-LEVEL ACTION'], 'url': ['http://fixture/']}
    actor.get_observation_node_str = lambda *args: ''
    actor.get_observation_node = lambda *args: None
    actor.get_observation_text = lambda *args: 'Long page ' * 250
    actor.get_observation_image = lambda *args: None
    actor.prune_message_list = lambda message_list: message_list
    result = actor.get_interaction_history(SimpleNamespace(step_num=5, verbose=True, type=['text']))[0][1]
    assert ('MODE_ANALYSIS:' in result) is not compact
    assert 'Verified fact' in result and 'Open next page' in result and 'click [7]' in result


def test_real_actor_keeps_bounded_evidence_and_next_turn_history(site, tmp_path):
    c, unused = client(tmp_path, [])
    c.max_calls = 2
    sent = []

    def post(*args, **kwargs):
        request = kwargs['json']
        sent.append(request)
        if len(sent) == 1:
            control = re.search(r"link \[(\d+)\] 'Catalog'", json.dumps(request['input']))[1]
            value = ('OBSERVATION DESCRIPTION: Catalog link is visible.\n'
                     'REASON: Open the catalog to inspect products.\n'
                     f'ACTION: click [{control}]\n'
                     f'OBSERVATION HIGHLIGHT: {control}, {control}, 99999999\n'
                     'NOTE: Catalog is the relevant entry.')
        else:
            value = VALID
        return SimpleNamespace(ok=True, json=lambda: dict(status='completed', model='fixture',
            usage={'input_tokens': 10, 'output_tokens': 5}, output=[{'type': 'message', 'role': 'assistant',
            'content': [{'type': 'output_text', 'text': value}]}]))

    c.session.post = post
    result = run_task(Graph(tmp_path/'graph'), 'Open catalog', c.log, c, None, use_graph=False,
        start_url=site, max_steps=2, final_check=lambda b: b.page.url == site + '/catalog')
    assert result['local_success'] and result['steps'] == 2, result
    assert len(sent) == 2 and c.budget_used() == 2
    history = json.dumps(sent[1]['input'])
    assert 'MODE_ANALYSIS:' not in history and 'MODE ANALYSIS:' not in history
    assert 'Open the catalog to inspect products.' in history and 'Catalog' in history
    events = read_events(c.log.path)
    filtered = [e for e in events if e['event'] == 'actor_highlight_filtered']
    assert len(filtered) == 1 and filtered[0]['duplicates'] == 1 and filtered[0]['invalid'] == 1
    assert not any(e['event'] == 'responses_output_retry' for e in events)


def test_legacy_extra_sections_do_not_revive_removed_fields_or_change_action():
    class Actor(AttachedNotes, load_runtime().Actor):
        pass
    actor = Actor.__new__(Actor)
    value = ('REPEAT FINAL OBJECT: Old goal\nEXTRACT CONSTRAINTS: Old constraints\n'
             'INTERACTION HISTORY SUMMARY: Old summary\nMODE ANALYSIS: EXPLORATION\n'
             'OBSERVATION DESCRIPTION: Search form visible.\nREASON FOR ACTION: Find the item.\n'
             'ACTION: type [7] [literal\nMODE ANALYSIS: title content\nACTION: still typed text] [0]\n'
             'OBSERVATION HIGHLIGHT: 7\nNOTE: Observed fact.')
    fields = actor.parse_elements(value, list(OUTPUT_FIELDS))
    assert set(fields) == {*OUTPUT_FIELDS, 'note'}
    assert fields['reason'] == 'Find the item.'
    assert fields['action'] == 'type [7] [literal\nMODE ANALYSIS: title content\nACTION: still typed text] [0]'
    assert fields['observation highlight'] == '7'


def test_highlight_tree_and_action_ids_keep_their_distinct_types(tmp_path):
    class Actor(AttachedNotes, load_runtime().Actor):
        pass
    Node = importlib.import_module('browser_env.processors').TreeNode
    actor = Actor.__new__(Actor)
    actor.config = namespace(configuration(EventLog(tmp_path / 'config'))['agent']['actor'])
    actor.note_log = EventLog(tmp_path)
    actor.get_step = lambda: 0
    root = Node(1, 'RootWebArea', 'Page', 0)
    root.add_child(Node(7, 'button', 'Continue', 1))
    root.add_child(Node(8, 'link', 'Evidence', 1))
    actor.get_observation_node = lambda: root
    actor.get_observation_text = lambda: "button [7] 'Continue'\nlink [8] 'Evidence'"
    fields = {'action': 'click [7]', 'observation highlight': '8, 8, 999',
              'observation description': 'Page contains evidence.'}
    actor.get_observation_highlight(fields)
    assert fields['observation highlight idxs'] == '8'
    assert fields['retained element ids'] == [7]
    assert fields['observation highlight'].search_node_by_id(8).name == 'Evidence'
    assert all(node.visible for node in [root, *root.children])
    # The trajectory serializes IDs as text/list and safely skips the DOM object.
    runner = load_runtime().webNavigator.__new__(load_runtime().webNavigator)
    runner.objective = 'Inspect evidence'
    runner.online_url = 'http://fixture/'
    runner.online_observation = {'text': actor.get_observation_text()}
    runner.trajectory = []
    runner.log_step({'done': False}, **fields)
    row = json.loads(json.dumps(runner.trajectory[0]))
    assert row['observation_highlight_idxs'] == '8'
    assert row['retained_element_ids'] == [7]
    assert row['observation_description'] == 'Page contains evidence.'
    assert row['action'] == 'click [7]'


@pytest.mark.parametrize('multimodal', [False, True])
def test_seven_step_history_keeps_summaries_highlights_and_blank_compatibility_fields(multimodal):
    module = load_runtime()
    Node = importlib.import_module('browser_env.processors').TreeNode
    actor = module.Actor.__new__(module.Actor)
    actor.config = SimpleNamespace(history_mode_analysis=False)
    actor.active_node = SimpleNamespace(id=0, steps_taken=list(range(7)))
    actor.previous_interactions = {
        'observation summary': [f'Summary {i}' for i in range(7)],
        'observation highlight': [Node(i + 1, 'button', f'Raw evidence {i}', 0) for i in range(7)],
        'retained element ids': [[i + 1] for i in range(7)],
        'plan': [''] * 7, 'reason': [f'Reason {i}' for i in range(7)],
        'action': [f'click [{i + 1}]' for i in range(7)],
        'mode analysis': [''] * 7, 'url': ['http://fixture/'] * 7}
    assert actor.equal_history_length()
    actor.get_observation_node_str = lambda idx=None: 'current' if idx is None else f'old {idx}'
    actor.get_observation_image = lambda idx=None: f'image-{idx}'
    history = actor.get_interaction_history(SimpleNamespace(
        step_num=5, verbose=True, type=['text', 'image'] if multimodal else ['text']))
    text = '\n'.join(value for kind, value in history if kind == 'text')
    assert 'MODE_ANALYSIS:' not in text
    for i in range(2):
        assert f'Summary {i}' in text and f'Raw evidence {i}' not in text
    for i in range(2, 7):
        assert f'Raw evidence {i}' in text
    for i in range(7):
        assert f'Reason {i}' in text and f'click [{i + 1}]' in text
