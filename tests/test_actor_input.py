import copy
import importlib
import json
import re
from types import SimpleNamespace

import pytest
import yaml

from web2kg.common.storage import EventLog, Graph
from web2kg.stage2.action_notes import AttachedNotes
from web2kg.stage2.actor_input import compact_old_rows, navigation_context
from web2kg.stage2.agent import configuration, load_runtime, namespace, run_task
from web2kg.stage2.prompt_context import context_messages, global_context, template_context
from test_prompt_context import template


def actor_fixture(tmp_path):
    class Actor(AttachedNotes, load_runtime().Actor):
        pass
    actor = Actor.__new__(Actor)
    actor.config = namespace(configuration(EventLog(tmp_path))['agent']['actor'])
    actor.active_node = SimpleNamespace(id=0, steps_taken=[], note=[], children=[])
    actor.plan_tree_root = actor.active_node
    actor.previous_interactions = {k: [] for k in ('action', 'url', 'observation summary',
        'observation highlight', 'reason', 'observation', 'retained element ids')}
    actor.online_interaction = {'observation': {'text': 'CURRENT UNIQUE AX', 'image': 'current-image'}}
    actor.prompt_template = {'input_template': '<input>\n{input}\n</input>'}
    actor.objective = 'UNIQUE OBJECTIVE 001'
    return actor


def prompt_text(actor):
    return '\n'.join(v for k, v in actor.get_online_input(None) if k == 'text')


def test_task_memory_migrates_notes_without_plan_wrapper_or_duplicate_objective(tmp_path):
    actor = actor_fixture(tmp_path)
    actor.active_node.text = actor.objective
    actor.active_node.note = ['STEP 0 (before action): Item 001 costs 12.50.']
    actor.active_node.children = [SimpleNamespace(note=['Earlier unsuccessful query: red mug'], children=[])]
    for inputs in (actor.config.input, ['objective', 'previous plans', 'task memory', 'current observation']):
        actor.config.input = inputs
        text = prompt_text(actor)
        assert text.count(actor.objective) == 1
        assert text.count('TASK MEMORY:') == 1
        assert 'Item 001 costs 12.50.' in text and 'unsuccessful query: red mug' in text
        assert 'PREVIOUS PLANS' not in text and 'Active Plan' not in text
    actor.active_node.note.clear()
    actor.active_node.children.clear()
    assert 'TASK MEMORY:' not in prompt_text(actor)


def test_old_history_preserves_unique_facts_full_actions_errors_and_order():
    body = 'First line\nACTION: literal text\nAmount 001.20  [draft]'
    rows = []
    for i in range(14):
        rows.append({'step': i, 'url': 'http://site/compose',
            'before': 'Draft title entered; submission is not yet verified.',
            'highlight': 'Owner alice_001; stock 2, price 12.50.',
            'intent': 'Use the form to continue.', 'action': f'type [8] [{body}] [0]',
            'result': {'status': 'executed', 'actual_url': 'http://site/compose'}})
    rows[1]['result'] = {'status': 'browser_error', 'error': 'Timeout waiting for Submit',
                         'actual_url': 'http://site/compose'}
    rows[2]['before'] = 'Post 135152 now appears in My posts.'
    rows[3]['before'] = 'No results for Title 001; do not confuse with Title 01.'
    rows[4]['highlight'] = 'Only highlighter saw price 12.05 and author bob_002.'
    rows[5]['intent'] = 'Compare with the earlier 9.75 value; name only in reason: Charlie.'
    original = copy.deepcopy(rows)
    compact = compact_old_rows(rows)
    assert rows == original and len(compact) == len(rows)
    # Resolve the compact ledger and require exact information preservation.
    resolved = {}
    for row in compact:
        full = copy.deepcopy(row)
        for key in ('url', 'before', 'highlight', 'intent'):
            ref = full.pop(key + '_same_as_step', None)
            if ref is not None:
                full[key] = resolved[ref][key]
        resolved[row['step']] = full
    assert list(resolved.values()) == rows
    assert len(json.dumps(compact)) < len(json.dumps(rows))
    assert compact[2]['before'] == rows[2]['before']
    assert compact[5]['intent'] == rows[5]['intent']
    assert all(r['action'] == rows[i]['action'] for i, r in enumerate(compact))


def test_exact_only_dedup_never_drops_numbers_negation_or_whitespace():
    facts = ['Votes: 2', 'Votes: 20', 'Posted', 'Not posted', 'a  b', 'a b', 'Votes: 2']
    rows = [{'step': i, 'before': text, 'action': 'click [1]'} for i, text in enumerate(facts)]
    result = compact_old_rows(rows)
    assert [r['before'] for r in result[:6]] == facts[:6]
    assert result[6]['before'] == 'Votes: 2'  # Short facts stay inline rather than growing a reference.
    long_fact = 'Title 001 remains a draft; submission was not confirmed.'
    repeated = compact_old_rows([{'step': i, 'before': long_fact, 'action': 'click [1]'} for i in range(2)])
    assert repeated[1]['before_same_as_step'] == 0
    identical = compact_old_rows([{'step': 0, 'before': 'Fact', 'intent': 'Fact', 'action': 'click [1]'}])
    assert 'intent' not in identical[0]


def test_recent_steps_keep_summary_highlight_reason_and_old_data_still_visible(tmp_path):
    actor = actor_fixture(tmp_path)
    for i in range(9):
        actor.active_node.steps_taken.append(i)
        actor.update_history(action=f'click [{i + 1}]', url='http://site/',
            **{'observation summary': f'Fact {i}: {i}.25', 'observation highlight': f'Raw selected evidence {i}',
               'reason': f'Reason {i}', 'retained element ids': [i + 1],
               'observation': {'text': 'AX', 'image': f'image-{i}', 'node': None}})
    original = copy.deepcopy(actor.previous_interactions)
    text = prompt_text(actor)
    assert 'OLDER ACTION LEDGER:' in text and 'RECENT STEPS:' in text
    old, recent = text.split('RECENT STEPS:', 1)
    for i in range(4):
        assert f'Fact {i}: {i}.25' in old and f'Raw selected evidence {i}' in old
    for i in range(4, 9):
        assert f'Fact {i}: {i}.25' in recent and f'Raw selected evidence {i}' in recent
        assert f'Reason {i}' in recent and f'click [{i + 1}]' in recent
    assert actor.previous_interactions == original and actor.equal_history_length()
    actor.config.interaction_history.type = ['text', 'image']
    actor.config.interaction_history.step_num = 'all'
    images = [v for k, v in actor.get_interaction_history() if k == 'image']
    assert images == [f'image-{i}' for i in range(9)]  # Index zero must not select current image.


def test_target_labels_survive_upstream_dom_release(tmp_path):
    actor = actor_fixture(tmp_path)
    Node = importlib.import_module('browser_env.processors').TreeNode
    root = Node(1, 'RootWebArea', 'Page', 0)
    root.add_child(Node(7, 'button', 'Submit post', 1))
    actor.active_node.steps_taken = [0]
    actor.update_history(action='click [7]', url='http://site/compose',
        **{'observation summary': 'Title set.', 'observation highlight': '', 'reason': 'Submit once.',
           'retained element ids': [7], 'observation': {'text': 'button [7] Submit post', 'node': root}})
    actor.del_observation_node()
    assert actor.previous_interactions['observation'][0]['node'] is None
    text = prompt_text(actor)
    assert 'Submit post' in text and 'click [7]' in text


def test_navigation_keeps_data_present_only_in_its_rationale(tmp_path):
    actor = actor_fixture(tmp_path)
    actor.active_node.steps_taken = [0, 1]
    actor.previous_interactions.update({
        'action': ['optimal_navigate [why] [reddit] [Directory]'] * 2,
        'navigate_think': ['Owner Bob_001 had 27 posts; compare later.', 'Directory'],
        'navigate_query': ['Directory'] * 2, 'navigate_domain': ['reddit'] * 2,
        'url': ['http://site/'] * 2, 'reason': ['Find entry.'] * 2,
        'observation summary': ['Home.'] * 2})
    actor.execution_outcomes = {0: {'navigation': navigation_context({'success': True}, query='Directory')},
                                1: {'navigation': navigation_context({'success': False}, query='Directory')}}
    text = prompt_text(actor)
    assert 'Owner Bob_001 had 27 posts; compare later.' in text
    assert text.count('navigation_intent:') == 1


@pytest.mark.parametrize('result,status,confirmed', [
    ({'success': True}, 'arrived', True),
    ({'success': False, 'reason': 'source_mismatch'}, 'not_confirmed', False),
    ({'success': False, 'status': 'requires_input'}, 'requires_input', False),
    ({'success': False, 'status': 'requires_local_action'}, 'requires_local_action', False),
    ({}, 'not_confirmed', False)])
def test_navigation_keeps_target_actual_arrival_and_constraints(result, status, confirmed):
    row = navigation_context(result, query='Find posts', domain='reddit', target='saved-state',
                             actual_url='http://site/actual')
    assert row['status'] == status and row['arrival_confirmed'] is confirmed
    assert row['target'] == 'saved-state' and row['actual_url'] == 'http://site/actual'
    assert row['query'] == 'Find posts' and row['site'] == 'reddit' and row['next']


def test_framework_card_projection_keeps_caveats_all_entrances_and_original_assets():
    raw = template('T1')
    raw['function_card']['importance'] = .9
    raw['nearby'].append({**raw['nearby'][0], 'control_label': 'Another mug', 'current_control_ids': ['43']})
    framework = {'overview': 'Browse and manage shop products.', 'areas': [
        {'name': 'Catalog', 'purpose': 'Repeated page-level detail.', 'templates': ['T1']}],
        'navigation': ['Search requires text'], 'limitations': ['Historical; writes are not verified.']}
    evidence = {'website_framework': framework, 'navigation_feedback': {'success': False,
        'status': 'requires_input', 'reason': 'actor_input_required', 'input_control': {'name': 'Search', 'role': 'textbox', 'xpath': '//input'}}}
    original = copy.deepcopy((raw, evidence))
    projected = template_context(raw, compact_actor=True)
    assert 'importance' not in projected['function_card']
    assert projected['function_card']['uncertainties'] == raw['function_card']['uncertainties']
    assert len(projected['nearby']) == 1 and len(projected['nearby'][0]['entrances']) == 2
    assert projected['nearby'][0]['target_summary'] == 'Product details'
    messages = context_messages([], raw, evidence, compact_actor=True)
    text = json.dumps(messages)
    assert 'Repeated page-level detail.' not in text and 'Historical; writes are not verified.' in text
    assert 'Search requires text' in text and 'BEFORE input' in text and 'xpath' not in text
    assert global_context(evidence)['website_framework'] == framework  # Selector stays unchanged.
    assert (raw, evidence) == original


def test_live_actor_memory_and_browser_error_survive_without_extra_requests(site, tmp_path, monkeypatch):
    load_runtime()
    env_module = importlib.import_module('browser_env.envs')
    from browser_env import ActionTypes
    execute = env_module.execute_action
    def fail_first_click(action, *args, **kwargs):
        if action['action_type'] == ActionTypes.CLICK:
            raise ValueError('fixture: Submit rejected; no mutation committed')
        return execute(action, *args, **kwargs)
    monkeypatch.setattr(env_module, 'execute_action', fail_first_click)
    sent = []
    class Client:
        def call(self, messages, **kwargs):
            sent.append(messages)
            text = json.dumps(messages)
            if len(sent) == 1:
                control = re.search(r"link \[(\d+)\] 'Catalog'", text)[1]
                return (f'OBSERVATION DESCRIPTION: Catalog is visible.\nREASON: Open catalog.\nACTION: click [{control}]\n'
                        'NOTE: Remember item 001; price 12.50.')
            assert 'TASK MEMORY:' in text and 'Remember item 001; price 12.50.' in text
            assert 'PREVIOUS PLANS:' not in text and 'execution_error' in text
            assert 'fixture: Submit rejected; no mutation committed' in text
            assert 'Catalog' in text
            return 'OBSERVATION DESCRIPTION: The click failed.\nREASON: Report the failure.\nACTION: stop [Could not click]'
    result = run_task(Graph(tmp_path/'graph'), 'Inspect catalog', EventLog(tmp_path/'run'), Client(), None,
        use_graph=False, start_url=site, max_steps=2)
    assert result['status'] == 'agent_stopped' and len(sent) == result['steps'] == 2, result
    # Saved model input remains available through the normal upstream trace, with no paid calls.
    assert (tmp_path/'run/step-1.txt').exists()
