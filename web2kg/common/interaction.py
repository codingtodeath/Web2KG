"""Navigation state and observable control changes, independent of LLM labels."""
import json
import re
from collections import Counter
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .storage import digest


VERSION = "interaction-v2"
LANDMARKS = {"navigation", "main", "complementary", "banner", "contentinfo", "region", "dialog", "search"}
STATE_PROPERTIES = {"checked", "selected", "expanded", "disabled", "pressed", "required", "level"}
CSRF_PARAMETERS = frozenset({"_token", "csrf", "csrf_token", "csrftoken", "xsrf_token", "authenticity_token"})
CONTROL_FIELDS = ("role", "name", "href", "type", "method", "form_action", "state",
                  "disabled", "data_mutation", "value_digest", "region", "occurrence", "match_count")


def control_signature(action):
    value = {key: action[key] for key in CONTROL_FIELDS if key in action}
    if str(value.get("method", "")).lower() == "post" and "form_action" in value:
        parts = urlsplit(value["form_action"] or "")
        # This is the existing replay rule, not a classifier for arbitrary tokens.
        query = [(k, "<session-token>" if k.lower() in CSRF_PARAMETERS else v)
                 for k, v in parse_qsl(parts.query, keep_blank_values=True)]
        value["form_action"] = urlunsplit(parts._replace(query=urlencode(query)))
    return value


def regions_from_ax(ax):
    return [[n.get("role", {}).get("value", ""),
             re.sub(r"\s+", " ", str(n.get("name", {}).get("value", "") or "")).strip(),
             {p["name"]: p["value"].get("value") for p in n.get("properties", [])
              if p["name"] in STATE_PROPERTIES}]
            for n in ax if not n.get("ignored")
            and n.get("role", {}).get("value") in LANDMARKS | {"alertdialog"}]


def state_signature(page, regions=None):
    # Keep order/multiplicity: role locators depend on occurrence among equal names.
    if regions is None:
        regions = page["regions"] if "regions" in page else regions_from_ax(page.get("ax", []))
    signature = [VERSION, page["url"], page.get("headings", []), regions,
                 [control_signature(a) for a in page["actions"]]]
    visual = page.get('page_content', {}).get('visual_sha256')
    return signature + [visual] if visual else signature


def state_id(page, regions=None):
    return digest(state_signature(page, regions))


def state_matches(saved, observed):
    # Never trust a stored ID alone; compare the same representation used to build it.
    return state_signature(saved) == state_signature(observed)


def changes(before, after):
    if after is None:
        return {"available": False, "reason": "after_observation_missing"}
    old = {a["id"]: a for a in before["actions"]}
    new = {a["id"]: a for a in after["actions"]}
    changed = []
    for key in old.keys() & new.keys():
        fields = [field for field in CONTROL_FIELDS if old[key].get(field) != new[key].get(field)]
        if fields:
            changed.append({"before_ref": "before:" + key, "after_ref": "after:" + key,
                            "fields": fields, "before": control_signature(old[key]),
                            "after": control_signature(new[key])})
    def bag(page):
        rows = ["control:" + json.dumps(control_signature(a), sort_keys=True, ensure_ascii=False)
                for a in page["actions"]]
        rows += ["text:" + line.strip() for line in page.get("text", "").splitlines() if line.strip()]
        rows += ["heading:" + text for text in page.get("headings", [])]
        return Counter(rows)
    left, right = bag(before), bag(after)
    total = sum(left.values()) + sum(right.values())
    similarity = 2 * sum((left & right).values()) / total if total else 1.0
    return {"available": True, "similarity": round(similarity, 6),
            "similarity_method": "multiset_dice_controls_text_lines_headings",
            "url_changed": before["url"] != after["url"],
            "headings_changed": before.get("headings", []) != after.get("headings", []),
            "content_changed": before.get("semantic_id", before["id"]) != after.get("semantic_id", after["id"]),
            "added": [{"ref": "after:" + k, "control": new[k]} for k in new if k not in old],
            "removed": [{"ref": "before:" + k, "control": old[k]} for k in old if k not in new],
            "changed": sorted(changed, key=lambda row: row["before_ref"])}
