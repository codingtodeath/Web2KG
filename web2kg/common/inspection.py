"""Read-only graph audit metadata for both inspection interfaces."""
from collections import Counter, defaultdict

from .activity import read_events


def discovery_path(graph, target):
    """Resolve the saved path, without substituting a shorter route."""
    actions, current, path = graph.nodes[target]["path"], target, []
    for index in range(len(actions) - 1, -1, -1):
        matches = [edge for edge in graph.data["edges"]
                   if edge["target"] == current and edge["action"] == actions[index]
                   and graph.nodes[edge["source"]]["path"] == actions[:index]]
        if len(matches) != 1:
            raise ValueError("保存的发现路径缺失或不唯一，无法核验到达过程")
        edge = matches[0]
        path.insert(0, edge)
        current = edge["source"]
    if current != graph.data["root"]:
        raise ValueError("保存的路径没有从图谱起点开始")
    return path


def graph_inspection(graph, events=None):
    events = read_events(graph.directory / "events.jsonl") if events is None else events
    urls = defaultdict(list)
    for node in graph.nodes.values():
        urls[node["url"]].append(node["id"])
    failures = Counter(e.get("source") or e.get("source_page", {}).get("id")
                       for e in events if e.get("event") == "exploration_failed"
                       and e.get("phase") == "restore_source")
    ranked = {e.get("source") or e.get("node") or e.get("source_page", {}).get("id")
              for e in events if e.get("event") in {"llm_ranking", "page_candidates"}}
    details, paths = {}, {}
    for node in graph.nodes.values():
        key = node["id"]
        try:
            path = discovery_path(graph, key)
            paths[key] = [edge["id"] for edge in path]
        except ValueError:
            path = None
        siblings = urls[node["url"]]
        details[key] = {"siblings": siblings, "variant": siblings.index(key) + 1,
                        "variants": len(siblings), "ranked": key in ranked,
                        "source_restore_failures": failures[key],
                        "path_restore_failures": sum(failures[e["source"]] for e in path) if path is not None else None,
                        "saved_path_valid": path is not None,
                        "entry_action": ({k: node["path"][-1].get(k) for k in ("kind", "role", "name")}
                                         if node["path"] else None)}
    reachable, frontier = {graph.data["root"]}, [graph.data["root"]]
    for source in frontier:
        for edge in graph.data["edges"]:
            if edge["source"] == source and edge["target"] not in reachable:
                reachable.add(edge["target"])
                frontier.append(edge["target"])
    return {"nodes": details, "paths": paths,
            "summary": {"urls": len(urls), "multi_state_urls": sum(len(v) > 1 for v in urls.values()),
                        "reachable": len(reachable & graph.nodes.keys()),
                        "saved_paths": len(paths), "ranked": len(ranked & graph.nodes.keys()),
                        "source_restore_failures": sum(failures.values()),
                        "source_restore_nodes": len(failures),
                        "pruned": sum(e.get("event") == "exploration_pruned" for e in events)}}
