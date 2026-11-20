"""Action decisions and auxiliary requests have separate accounting units."""

POLICY = 'actor-action-v1'


def action_budget_used(events):
    reserved = {e['action_budget_id'] for e in events
                if e['event'] == 'responses_started' and e.get('action_budget_policy') == POLICY
                and e.get('purpose') == 'original_actor'}
    searches = {e['action_budget_id'] for e in events
                if e['event'] == 'action_budget_search' and e.get('action_budget_policy') == POLICY}
    return len(reserved - searches)
