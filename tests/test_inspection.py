from web2kg.common.inspection import graph_inspection
from web2kg.common.storage import Graph


def test_saved_discovery_route_and_source_failures_are_separate_from_reachability(tmp_path):
    graph = Graph(tmp_path)
    actions = [dict(id=str(i), kind="click", role="button", name=str(i)) for i in range(3)]
    for key, path in (("root", []), ("menu", actions[:1]), ("form", actions[:2])):
        graph.nodes[key] = dict(id=key, url="http://example.test/", path=path)
    graph.data["root"] = "root"
    for source, target, action in (("root", "menu", actions[0]), ("menu", "form", actions[1]),
                                   ("root", "form", actions[2])):
        graph.data["edges"].append(dict(id=action["id"], source=source, target=target, action=action))
    events = [dict(event="exploration_failed", source="menu", phase="restore_source"),
              dict(event="exploration_failed", source="root", phase="action"),
              dict(event="llm_ranking", source="root"), dict(event="exploration_pruned")]
    original = repr(graph.data)
    audit = graph_inspection(graph, events)
    assert audit["paths"]["form"] == ["0", "1"]  # Not the one-step shortcut.
    assert audit["nodes"]["menu"]["source_restore_failures"] == 1
    assert audit["nodes"]["form"]["path_restore_failures"] == 1
    assert audit["nodes"]["menu"]["variant"] == 2
    assert audit["summary"]["reachable"] == audit["summary"]["saved_paths"] == 3
    assert audit["summary"]["urls"] == audit["summary"]["source_restore_nodes"] == 1
    assert audit["summary"]["pruned"] == audit["summary"]["ranked"] == 1
    assert repr(graph.data) == original


def test_legacy_missing_path_does_not_become_a_replay_claim(tmp_path):
    graph = Graph(tmp_path)
    graph.nodes.update(root=dict(id="root", url="/", path=[]), child=dict(id="child", url="/", path=[]))
    graph.data.update(root="root", edges=[dict(id="e", source="root", target="child", action={})])
    audit = graph_inspection(graph, [])
    assert audit["summary"]["reachable"] == 2
    assert audit["summary"]["saved_paths"] == 1
    assert not audit["nodes"]["child"]["saved_path_valid"]
