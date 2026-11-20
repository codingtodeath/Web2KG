from types import SimpleNamespace

import pytest

from web2kg.common.config import model_config
from web2kg.stage1.indexing import build_index
from web2kg.stage2.retrieval import choose, retrieve
from web2kg.common.safety import assess
from web2kg.stage1.selection import select_actions
from web2kg.common.storage import EventLog, Graph
from web2kg.visualization.viewer import export_viewer
from web2kg.common.activity import candidate_activity


def test_candidate_activity_separates_selection_execution_and_unknown_failures(tmp_path):
    graph = Graph(tmp_path)
    graph.data["edges"] = [{"source": "root", "target": "target", "action": {"id": "old_success"}}]
    events = [{"event": "page_candidates", "node": "root", "candidates": [
        {"id": key, "status": "selected"} for key in ("pending", "restore", "failed", "unchanged", "legacy")]}]
    events += [
        {"event": "exploration_failed", "source": "root", "action": {"id": "restore"}, "action_attempted": False},
        {"event": "exploration_failed", "source": "root", "action": {"id": "failed"}, "action_attempted": True},
        {"event": "exploration_failed", "source": "root", "action": {"id": "legacy"}},
        {"event": "exploration_no_change", "source": "root", "action": {"id": "unchanged"}},
    ]
    rows = candidate_activity(graph, events)["root"]
    assert rows["pending"]["selected"] and not rows["pending"]["attempted"]
    assert not rows["restore"]["attempted"] and rows["restore"]["results"][0]["outcome"] == "not_attempted"
    assert rows["failed"]["attempted"] and rows["failed"]["results"][0]["outcome"] == "failed"
    assert rows["unchanged"]["attempted"] and rows["unchanged"]["results"][0]["outcome"] == "no_change"
    assert not rows["legacy"]["attempted"] and rows["legacy"]["results"][0]["outcome"] == "unknown"
    assert rows["old_success"]["attempted"] and rows["old_success"]["results"][0]["target"] == "target"


def action(name="Catalog", href="http://localhost/catalog", **kwargs):
    return {"id": name, "kind": "click", "name": name, "href": href, "role": "link", "xpath": "/html/body/ul/li[1]/a[1]", **kwargs}


@pytest.mark.parametrize("candidate", [
    action("Delete account"), action(href="http://localhost/delete?id=1"),
    action(href="http://elsewhere/catalog"), action(href="javascript:alert(1)"),
    action(method="post"), action(data_mutation=True), action(disabled=True),
    action("删除商品"), action("Continue", href="", role="button"),
])
def test_only_mechanical_candidate_validation(candidate):
    assert assess(candidate, "http://localhost")[0] is not bool(candidate.get("disabled"))


def test_model_receives_controls_without_keyword_exclusion(tmp_path):
    class Model:
        def json(self, system, payload):
            assert any(a["id"] == "Delete" for a in payload["candidates"])
            return {"decisions": [{"id": "Delete", "group": "safe", "risky": False}]}
    observation = {"id": "root", "url": "http://localhost", "title": "Catalog", "actions": [action(), action("Delete", role="button")]}
    selected = select_actions(observation, "http://localhost", EventLog(tmp_path), model=Model())
    assert [a["name"] for a in selected] == ["Catalog", "Delete"]


def test_repeated_items_share_budget(tmp_path):
    actions = [action(str(i), xpath=f"/html/body/ul/li[{i}]/a[1]", href=f"http://localhost/item/{i}") for i in range(1, 10)]
    obs = {"id": "root", "url": "http://localhost", "title": "Catalog", "actions": actions}
    assert len(select_actions(obs, "http://localhost", EventLog(tmp_path))) == 1


def test_llm_cannot_split_rule_groups_or_miss_risky_peers(tmp_path):
    actions = [action(str(i), xpath=f"/html/body/ul/li[{i}]/a[1]") for i in range(1, 4)]
    obs = {"id": "root", "url": "http://localhost", "title": "Catalog", "actions": actions}
    class Model:
        def json(self, system, payload):
            assert len(payload["candidates"]) == 1
            assert payload["candidates"][0]["count"] == 3
            return {"decisions": [{"id": "1", "group": "unsafe", "risky": True}]}
    assert not select_actions(obs, "http://localhost", EventLog(tmp_path), model=Model())


def test_exact_path_check_does_not_accept_subforum():
    from web2kg.evaluation.tasks import check_task
    browser = SimpleNamespace(page=SimpleNamespace(
        url="http://localhost/f/example/comments",
        locator=lambda _: SimpleNamespace(inner_text=lambda: "Recent comments")))
    assert not check_task(browser, {"path_equals": "/comments", "text_contains": "Recent comments"})


def small_graph(tmp_path):
    graph = Graph(tmp_path)
    for key, text in [("catalog", "Mugs and ceramic products"), ("help", "Shipping and returns policy")]:
        graph.nodes[key] = {"id": key, "title": key, "url": f"http://localhost/{key}",
                            "headings": [text], "text": text, "actions": [], "depth": 0, "path": []}
    graph.data.update(root="catalog", start_url="http://localhost")
    graph.add_edge("catalog", "help", action())
    return graph


def test_directed_paths_and_same_node(tmp_path):
    graph = small_graph(tmp_path)
    assert len(graph.path("catalog", "help")) == 1
    assert graph.path("help", "catalog") is None
    assert graph.path("missing", "missing") is None
    assert graph.path("help", "help") == []


def test_lexical_retrieval_and_stale_index(tmp_path):
    graph = small_graph(tmp_path)
    build_index(graph)
    assert retrieve(graph, "returns policy")[0]["id"] == "help"
    graph.nodes["help"]["title"] = "Changed"
    with pytest.raises(ValueError, match="rebuild"):
        retrieve(graph, "returns")


def test_embedding_model_mismatch(tmp_path):
    import numpy as np
    class Model:
        config = SimpleNamespace(provider="local", model="a")
        def embed(self, texts):
            return np.ones((len(texts), 2))
    graph = small_graph(tmp_path)
    model = Model()
    build_index(graph, model)
    model.config = SimpleNamespace(provider="local", model="b")
    with pytest.raises(ValueError, match="match"):
        retrieve(graph, "help", model)


@pytest.mark.parametrize("choice", ["invented", [], 3])
def test_selector_rejects_invalid_ids(tmp_path, choice):
    model = SimpleNamespace(json=lambda *args: {"node_id": choice})
    with pytest.raises((ValueError, TypeError)):
        choose("help", [{"id": "help"}], model, EventLog(tmp_path))


def test_selector_can_reject_all(tmp_path):
    model = SimpleNamespace(json=lambda *args: {"node_id": None, "reason": "No relevant page"})
    assert choose("unrelated", [{"id": "help"}], model, EventLog(tmp_path)) is None


def test_secrets_excluded_from_repr(tmp_path, monkeypatch):
    for name in ["WEB2KG_CHAT_PROVIDER", "WEB2KG_CHAT_MODEL", "CHATANYWHERE_API_KEY", "CHATANYWHERE_BASE_URL"]:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("CHATANYWHERE_API_KEY=private-test-key\nCHATANYWHERE_BASE_URL=https://example.org\n")
    config = model_config(env_file=env)
    assert "private-test-key" not in repr(config)
    assert config.base_url.endswith("/v1")


def test_viewer_escapes_page_content(tmp_path):
    graph = small_graph(tmp_path)
    graph.nodes["help"]["title"] = "</script><script>alert('xss')</script>"
    html = export_viewer(graph).read_text(encoding="utf-8")
    assert "</script><script>alert" not in html
    assert "\\u003c/script>" in html
