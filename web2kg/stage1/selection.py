import re

from ..common.safety import assess
from ..common.diagnostics import page_ref


def group_key(action):
    region = action.get("region", {})
    if region.get("role") in {"navigation", "banner"}:
        return f"navigation:{action['role']}:{action['name']}:{action.get('href', '')}"
    if action["role"] != "link":
        return f"control:{action['role']}:{action['name']}"
    shape = re.sub(r"\[\d+\]", "[]", action["xpath"])
    return f"repeat:{shape}"


def select_actions(observation, start_url, log, limit=12, per_group=1, model=None, eligible_paths=None):
    safe, decisions = [], []
    for action in observation["actions"]:
        allowed, reason = assess(action, start_url)
        if allowed and eligible_paths is not None and action["xpath"] not in eligible_paths:
            allowed, reason = False, "inherited_dom_path"
        decisions.append({"id": action["id"], "name": action["name"], "allowed": allowed, "reason": reason})
        if allowed:
            safe.append(action)
    hints = {}
    # One bounded batch per page. The model can only remove candidates or merge groups.
    representatives = {}
    for action in safe:
        representatives.setdefault(group_key(action), []).append(action)
    groups = list(representatives.values())[:32]
    batch = [group[0] for group in groups]
    if model and batch:
        try:
            result = model.json(
                'Review navigation candidates in a resettable test site. Group equivalent repeated items and flag risks. '
                'Candidates already represent structural groups. Return only additional risk flags or group merges as '
                '{"decisions":[{"id":"...","group":"short group name","risky":false,"reason":"short reason"}]}. '
                'Omit unchanged groups. Preserve distinct functional entrances. Never recommend modifying business data.',
                {"url": observation["url"], "title": observation["title"],
                 "candidates": [{"id": group[0]["id"], "role": group[0]["role"],
                                 "name": group[0]["name"][:160], "href": group[0]["href"],
                                 "count": len(group), "examples": [a["name"][:80] for a in group[:3]]}
                                for group in groups]},
            )
            valid_ids = {a["id"] for a in batch}
            for hint in result.get("decisions", []):
                if (isinstance(hint, dict) and hint.get("id") in valid_ids and
                        isinstance(hint.get("risky"), bool) and isinstance(hint.get("group"), str)):
                    hints[hint["id"]] = hint
            for group in groups:
                if group[0]["id"] in hints:
                    hint = hints[group[0]["id"]]
                    for member in group[1:]:
                        hints[member["id"]] = {**hint, "id": member["id"]}
            log.emit("llm_review", node=observation["id"], reviewed=len(batch), decisions=list(hints.values()))
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            log.emit("llm_review_failed", node=observation["id"], error=str(error), fallback="rules")
    counts, chosen = {}, []
    outcomes = {d["id"]: {"reason": d["reason"], "status": "filtered"} for d in decisions if not d["allowed"]}
    for action in safe:
        hint = hints.get(action["id"], {})
        rule_key = group_key(action)
        key = "llm:" + hint["group"] if hint.get("group") else rule_key
        if hint.get("risky"):
            reason = "llm_risk"
        elif counts.get(key, 0) >= per_group or counts.get(rule_key, 0) >= per_group:
            reason = "group_budget"
        elif len(chosen) >= limit:
            reason = "page_budget"
        else:
            reason = "selected"
            for group in {key, rule_key}:
                counts[group] = counts.get(group, 0) + 1
            chosen.append(action)
        log.emit("candidate_selection", node=observation["id"], action=action["id"], group=key, reason=reason)
        outcomes[action["id"]] = {"reason": reason, "status": "selected" if reason == "selected" else "skipped", "group": key}
    log.emit("risk_filter", node=observation["id"], decisions=decisions,
             total=len(decisions), safe=len(safe), selected=len(chosen))
    log.emit("page_candidates", node=observation["id"], source_page=page_ref(observation),
             candidates=[{**action, "number": action.get("number", number), **outcomes[action["id"]]}
                         for number, action in enumerate(observation["actions"], 1)],
             total=len(decisions), selected=len(chosen))
    return chosen
