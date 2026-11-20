"""Skip previously visited destinations, without claiming an unexecuted edge."""
from urllib.parse import urlsplit

from ..common.browser import canonical_url
from ..common.diagnostics import page_ref
from ..common.safety import assess


def known_targets(graph, node, action):
    if (action.get("kind") != "click" or action.get("role") != "link"
            or action.get("navigation_kind") != "plain_link"
            or action.get('method')  # A form action is not a plain hyperlink, even with a stale label.
            or any(k in action.get("state", {}) for k in ("expanded", "selected", "pressed", "checked"))
            or not assess(action, graph.data["start_url"])[0]):
        return []
    href = canonical_url(action.get("href", ""))
    # Preserve same-page interactions, anchors and hash-based routes.
    if urlsplit(href).fragment or href == canonical_url(node["url"]).split("#", 1)[0]:
        return []
    reachable = graph.reachable()
    return [key for key, target in graph.nodes.items()
            if key in reachable and canonical_url(target["url"]) == href]


def record_pruned(graph, node, action, targets, log, phase):
    records = graph.data.setdefault("pruned_links", [])
    if any(r["source"] == node["id"] and r["action"]["id"] == action["id"] for r in records):
        return
    row = {"source": node["id"], "source_page": page_ref(node), "action": action,
           "target_url": canonical_url(action["href"]), "known_targets": targets,
           "reason": "known_navigation_url", "phase": phase, "action_attempted": False,
           "verified_edge": False}
    records.append(row)
    log.emit("exploration_pruned", **row)
