import json


def read_events(path):
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A running writer may not have completed the last line.
    return events


def candidate_activity(graph, events):
    """Selection is intent; only explicit execution evidence counts as attempted."""
    activity = {}

    def entry(source, action):
        return activity.setdefault(source, {}).setdefault(action, {
            "selected": False, "attempted": False, "results": []})

    for event in events:
        source = event.get("source_page", {}).get("id") or event.get("source") or event.get("node")
        if not source:
            continue
        if event["event"] == "page_candidates":
            for action in event["candidates"]:
                entry(source, action["id"])["selected"] = action["status"] == "selected"
        action = event.get("action")
        action_id = action.get("id") if isinstance(action, dict) else action
        if not action_id:
            continue
        row = entry(source, action_id)
        kind = event["event"]
        if kind == "candidate_selection":
            row["selected"] = event["reason"] == "selected"
        elif kind == "exploration_action_started":
            row.update(selected=True, attempted=True)
        elif kind == "exploration_pruned":
            row["results"].append({"outcome": "pruned", "phase": event["phase"]})
        elif kind == "execution_transition":
            row.update(attempted=True, selected=True)
            row['results'].append({'outcome': 'success' if event['outcome'] == 'transition' else event['outcome'],
                                   'target': event.get('target'), 'phase': event.get('phase'), 'error': event.get('error')})
        elif kind in {"transition_discovered", "exploration_no_change", "exploration_failed", "agent_action", "agent_error"}:
            attempted = kind in {"transition_discovered", "exploration_no_change", "agent_action"} or event.get("action_attempted") is True
            row["attempted"] |= attempted
            row["selected"] |= attempted
            outcome = ("success" if kind in {"transition_discovered", "agent_action"} else
                       "no_change" if kind == "exploration_no_change" else
                       "failed" if attempted else "not_attempted" if event.get("action_attempted") is False else "unknown")
            row["results"].append({"outcome": outcome, "target": event.get("target"),
                                   "phase": event.get("phase"), "error": event.get("error_summary") or event.get("error", "").split("\n")[0]})
    # Old graphs may have edges but no corresponding event file.
    for edge in graph.data["edges"]:
        row = entry(edge["source"], edge["action"]["id"])
        row.update(selected=True, attempted=True)
        if not any(r["outcome"] == "success" and r["target"] == edge["target"] for r in row["results"]):
            row["results"].append({"outcome": "success", "target": edge["target"]})
    return activity


def annotation_states(activity):
    return {key: "attempted" if row["attempted"] else "selected" if row["selected"] else "candidate"
            for key, row in activity.items()}
