import json
import copy
from types import SimpleNamespace

import pytest

from web2kg.common.browser import Browser
from web2kg.common.responses import ResponsesClient, object_schema
from web2kg.common.storage import EventLog, Graph
from web2kg.stage1.crawl import CrawlBudget, crawl
from web2kg.stage1.llm_exploration import LLMExplorer, page_context
from web2kg.visualization.viewer import export_viewer


class FakeClient:
    identity = {"model": "test", "protocol": "fixture"}
    last_request_id = "fixture-request"

    def __init__(self, log, fail=False):
        self.log, self.fail, self.calls = log, fail, []

    def fork(self, **kwargs):
        return copy.copy(self)

    def json(self, system, payload, schema, purpose, **kwargs):
        self.calls.append((purpose, payload))
        if purpose == "candidate_ranking":
            assert "enum" not in schema["properties"]["selected"]["items"]["properties"]["reason"]
            assert schema["properties"]["selected"]["items"]["properties"]["id"]["enum"] == [a["id"] for a in payload["page"]["candidates"]]
            return {"selected": [{"id": a["id"], "group": f"function-{i}", "reason": "Test priority"}
                                 for i, a in enumerate(payload["page"]["candidates"][::-1][:payload["K"]])]}
        if self.fail:
            self.fail = False
            raise RuntimeError("Simulated provider failure")
        if 'page_descriptions' in schema['properties']:
            return {'replayable': True, 'effect': 'Observed transition', 'uncertainties': [],
                    'control_evidence': ['B1'], 'page_descriptions': {
                        side: side.title() for side in payload['page_descriptions_requested']}}
        assert "enum" not in schema["properties"]["before_page_description"]
        return {"before_page_description": "Before", "after_page_description": "After",
                "replayable": True, "replay_reason": "Fixture: observed read-only navigation",
                "operation_description": "Click", "result_description": payload["runtime_result"]["outcome"],
                "control_evidence": [{"ref": payload["provided_evidence"][0]["ref"], "description": "Actual control"}],
                "uncertainties": ["No business completion claim"]}


def observation():
    def action(key):
        return {"id": key, "kind": "click", "role": "link", "name": key,
                "xpath": f"/html[1]/body[1]/a[{len(key)}]", "href": "http://localhost/" + key}
    return {"id": "root", "url": "http://localhost/", "title": "Home", "headings": [], "text": "",
            "actions": [action("A"), action("BB"), action("Delete")]}


def test_priority_persisted_rules_not_overridden_and_cached(tmp_path):
    graph = Graph(tmp_path); node = observation()
    graph.data["start_url"] = node["url"]; graph.add_node(node, [])
    model = FakeClient(EventLog(tmp_path)); explorer = LLMExplorer(model, tmp_path)
    result = explorer.select(graph, node, 2)
    assert [a["id"] for a in result] == ["Delete", "BB"]
    assert len(model.calls) == 1
    assert "Delete" in {a["id"] for a in model.calls[0][1]["page"]["candidates"]}
    explorer.select(graph, node, 2)
    assert len(model.calls) == 1
    events = [json.loads(x) for x in model.log.path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["candidates"][-1]["status"] == "selected"
    assert events[-1]["candidates"][1]["priority"] == 2


@pytest.mark.parametrize("rows", [
    [{"id": "invented", "group": "x", "reason": "x"}],
    [{"id": "A", "group": "x", "reason": "x"}] * 2,
])
def test_invalid_rank_never_executes_or_silently_falls_back(tmp_path, rows):
    graph = Graph(tmp_path); node = observation(); graph.data["start_url"] = node["url"]
    model = FakeClient(EventLog(tmp_path)); model.json = lambda *args: {"selected": rows}
    with pytest.raises(ValueError):
        LLMExplorer(model, tmp_path).select(graph, node, 2)
    assert not graph.data["rankings"]


def test_context_switch_preserves_hierarchy_and_local_text(tmp_path):
    node = observation()
    node["dom"] = {"backendNodeId": 1, "children": [{"nodeName": "HTML", "nodeType": 1,
        "backendNodeId": 2, "children": [{"nodeName": "BODY", "nodeType": 1, "backendNodeId": 3,
        "children": [{"nodeName": "A", "nodeType": 1, "backendNodeId": 4}]}]}]}
    node["ax"] = [{"nodeId": "1", "role": {"value": "main"}, "name": {"value": "Products"}},
                  {"nodeId": "2", "parentId": "1", "backendDOMNodeId": 4,
                   "role": {"value": "link"}, "name": {"value": "A"}},
                  {"nodeId": "3", "parentId": "1", "role": {"value": "StaticText"},
                   "name": {"value": "Nearby product explanation"}}]
    small = page_context(node, tmp_path, node["actions"][:1])
    assert small["mode"] == "small_ax" and small["ax_tree"][1]["ax_parent"] == "1"
    large = page_context(node, tmp_path, node["actions"][:1], small_limit=1)
    assert large["mode"] == "large_regions" and "ax_tree" not in large
    assert large["regions"][0]["name"] == "Products"
    assert "Nearby product explanation" in large["candidates"][0]["nearby_text"]


def test_live_browser_transition_evidence_and_resume_without_reexecution(site, tmp_path):
    log = EventLog(tmp_path); budget = CrawlBudget(6, 2, 4, 1, 8)
    model = FakeClient(log, fail=True); explorer = LLMExplorer(model, tmp_path)
    with Browser(site, log, identity_mode="webnavigator", settle_ms=10) as browser:
        with pytest.raises(RuntimeError, match="annotation failed"):
            crawl(browser, tmp_path, log, budget, explorer=explorer)
    saved = Graph.load(tmp_path)
    assert saved.data["transition_records"][0]["annotation"]["status"] == "failed"
    first_source = saved.data["transition_records"][0]["source"]
    first_action = saved.data["transition_records"][0]["action"]["id"]
    first_rank = saved.data["rankings"][first_source]
    with Browser(site, log, identity_mode="webnavigator", settle_ms=10) as browser:
        graph = crawl(browser, tmp_path, log, budget, resume=True, explorer=explorer)
    events = [json.loads(x) for x in log.path.read_text(encoding="utf-8").splitlines()]
    started = [e for e in events if e["event"] == "crawl_attempt_started"]
    assert sum((e["source"], e["action"]["id"]) == (first_source, first_action) for e in started) == 1
    assert graph.data["rankings"][first_source] == first_rank
    assert len(graph.data["transition_records"]) == graph.data["attempts"]
    assert any(r["annotation"]["status"] == "completed" for r in graph.data["transition_records"])
    assert all(e.get("description") for e in graph.data["edges"])
    html = export_viewer(graph).read_text(encoding="utf-8")
    assert "LLM 探索排序" in html and "结构化 Transition JSON" in html
    assert len(graph.nodes) <= 6


def test_responses_wire_format_logging_and_unknown_request_cap(tmp_path):
    env = tmp_path / "private.env"
    env.write_text("CRAWL_API_KEY=secret-example\nCRAWL_BASE_URL=https://example.org/v1\nCRAWL_MODEL=gpt-6-luna\n")
    log = EventLog(tmp_path / "run")
    client = ResponsesClient(env, log, max_calls=1)
    def post(url, **kw):
        assert url.endswith("/responses") and kw["json"]["text"]["format"]["strict"]
        assert kw["json"]["input"][0]["content"][0]["type"] == "input_text"
        assert kw["allow_redirects"] is False
        return SimpleNamespace(ok=True, json=lambda: {"status": "completed", "model": "gpt-6-luna",
            "usage": {"input_tokens": 10, "output_tokens": 2}, "output": [{"type": "message",
            "content": [{"type": "output_text", "text": '{"ok":true}'}]}]})
    client.session.post = post
    assert client.json("JSON", {}, object_schema({"ok": {"type": "boolean"}}), "preflight") == {"ok": True}
    with pytest.raises(RuntimeError, match="budget"):
        client.json("JSON", {}, {}, "preflight")
    assert "secret-example" not in log.path.read_text()
    assert all("secret-example" not in p.read_text() for p in (log.directory / "llm").rglob("*.json"))
    resumed = ResponsesClient(env, log, max_calls=1)
    with pytest.raises(RuntimeError, match="budget"):
        resumed.json("JSON", {}, {}, "preflight")


def test_invalid_description_reference_preserves_executed_transition(tmp_path):
    graph = Graph(tmp_path); node = observation(); graph.add_node(node, [])
    graph.data["start_url"] = node["url"]
    model = FakeClient(EventLog(tmp_path)); explorer = LLMExplorer(model, tmp_path)
    selected = explorer.select(graph, node, 1)
    original = model.json
    def invalid(*args):
        result = original(*args)
        result["control_evidence"][0]["ref"] = "after:invented"
        return result
    model.json = invalid
    with pytest.raises(RuntimeError):
        explorer.describe(graph, node, selected[0], node, 1, "no_change", "capture_result", True)
    record = json.loads((tmp_path / "transitions/00001.json").read_text())
    assert record["outcome"] == "no_change" and record["action_attempted"]
    assert record["annotation"]["status"] == "failed"


def test_repeated_function_group_keeps_first_priority_and_logs_others(tmp_path):
    graph = Graph(tmp_path); node = observation(); graph.data["start_url"] = node["url"]
    model = FakeClient(EventLog(tmp_path))
    rows = [{"id": "BB", "group": "detail", "reason": "first"}, {"id": "A", "group": "DETAIL", "reason": "second"}]
    model.json = lambda *args: {"selected": rows}
    selected = LLMExplorer(model, tmp_path).select(graph, node, 2)
    assert [a["id"] for a in selected] == ["BB"]
    assert graph.data["rankings"][node["id"]]["repeated_group_candidates"] == rows[1:]
    event = json.loads(model.log.path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["candidates"][0]["reason"] == "llm_group_representative"


def test_viewer_displays_priority_and_transition_in_real_browser(tmp_path):
    from playwright.sync_api import sync_playwright
    graph = Graph(tmp_path); node = observation()
    graph.data.update(start_url=node["url"], root=node["id"], review_mode="llm_ranked")
    graph.add_node(node, [])
    target = dict(node, id="target", title="Target", url="http://localhost/target")
    graph.add_node(target, [node["actions"][0]])
    model = FakeClient(EventLog(tmp_path)); explorer = LLMExplorer(model, tmp_path)
    selected = explorer.select(graph, node, 2)
    edge = graph.add_edge(node["id"], target["id"], selected[0])
    explorer.describe(graph, node, selected[0], target, 1, "transition", "capture_result", True, edge)
    file = export_viewer(graph)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(file.resolve().as_uri())
        assert "LLM 探索排序" in page.locator("#detail").inner_text()
        assert "Test priority" in page.locator("#detail").inner_text()
        assert "LLM 功能排序" in page.locator("#note").inner_text()
        assert page.locator(".selected-controls .candidate-row").first.get_attribute("data-action") == selected[0]["id"]
        assert "优先级 1" in page.locator(".selected-controls .candidate-row").first.inner_text()
        page.locator(".connection-row").first.click()
        assert "前页面：Before" in page.locator("#edgeDetail").inner_text()
        page.locator("#logTab").click()
        annotation = page.locator(".log-row").filter(has_text="Transition 结构化描述").first
        annotation.locator("summary").first.click()
        assert "前页面：Before" in annotation.locator(".event-detail").inner_text()
        assert not errors
        browser.close()


def test_http_failure_counts_against_cap_without_body_or_retry(tmp_path):
    env = tmp_path / "private.env"
    env.write_text("CRAWL_API_KEY=hidden-token\nCRAWL_BASE_URL=https://example.org/v1\nCRAWL_MODEL=gpt-6-luna\n")
    log = EventLog(tmp_path / "run")
    client = ResponsesClient(env, log, max_calls=1)
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(ok=False, status_code=403, text="hidden-token private error body")
    client.session.post = post
    with pytest.raises(RuntimeError, match="HTTP 403"):
        client.json("Return JSON", {}, {}, "test")
    with pytest.raises(RuntimeError, match="budget"):
        ResponsesClient(env, log, max_calls=1).json("Return JSON", {}, {}, "test")
    assert calls == [1]
    assert "hidden-token" not in log.path.read_text()
    events = [json.loads(x) for x in log.path.read_text().splitlines()]
    assert [x["event"] for x in events] == ["responses_started", "responses_failed"]


def test_candidate_cap_round_robins_groups_and_history_counts_actual_attempts(tmp_path):
    graph = Graph(tmp_path); node = observation()
    prototype = node["actions"][0]
    node["actions"] = [dict(prototype, id=str(i), name=f"Item {i}", xpath=f"/ul/li[{i}]/a[1]") for i in range(20)]
    node["actions"].append(dict(prototype, id="menu", name="Forums", xpath="/nav/a[1]"))
    graph.data["start_url"] = node["url"]
    graph.data["function_history"] = [{"group": "attempted", "action_attempted": True},
                                        {"group": "only_selected", "action_attempted": False}]
    model = FakeClient(EventLog(tmp_path))
    selected = LLMExplorer(model, tmp_path, candidate_limit=2).select(graph, node, 2)
    assert {a["id"] for a in selected} == {"0", "menu"}
    payload = model.calls[0][1]
    assert payload["attempted_function_counts"] == {"attempted": 1}
    assert payload["omitted_safe_candidates"] == 19
