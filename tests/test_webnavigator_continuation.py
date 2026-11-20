import pytest

from scripts.continue_hard_webnavigator import pending_tasks


def test_only_unstarted_baseline_in_original_order():
    protocol = {'schedule': [{'arm': a, 'task_id': t} for a, t in
                            [('webnavigator', 29), ('ours', 29), ('ours', 400),
                             ('webnavigator', 400), ('webnavigator', 401)]]}
    state = {'tasks': {'ours/29': {'status': 'needs_attention'},
                       'webnavigator/29': {'status': 'completed'}}}
    assert pending_tasks(protocol, state, [400, 401]) == [400, 401]
    assert state['tasks']['ours/29']['status'] == 'needs_attention'
    state['tasks']['webnavigator/400'] = {'status': 'started'}
    with pytest.raises(ValueError, match='unresolved'):
        pending_tasks(protocol, state, [400, 401])


def test_no_unregistered_or_duplicate_starts():
    protocol = {'schedule': [{'arm': 'webnavigator', 'task_id': 400},
                             {'arm': 'webnavigator', 'task_id': 401}]}
    with pytest.raises(ValueError, match='Unregistered'):
        pending_tasks(protocol, {'tasks': {}}, [400])
    protocol['schedule'].append(protocol['schedule'][0])
    with pytest.raises(ValueError, match='Invalid'):
        pending_tasks(protocol, {'tasks': {}}, [400, 401])
