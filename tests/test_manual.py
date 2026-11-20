import json
import queue
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog, Graph
from web2kg.evaluation.manual import ManualSession, discovery_path
from web2kg.stage1.indexing import build_index
from web2kg.visualization.manual import create_server, execute_command


@pytest.fixture(scope="module")
def manual_graph(site, tmp_path_factory):
    directory = tmp_path_factory.mktemp("manual-graph")
    graph = Graph(directory)
    with Browser(site, EventLog(directory), settle_ms=20) as browser:
        root = browser.observe(directory)
        graph.data.update(root=root["id"], start_url=site)
        graph.add_node(root, [])
        help_action = next(a for a in root["actions"] if a["name"] == "Help")
        browser.act(help_action)
        help_page = browser.observe(directory)
        graph.add_node(help_page, [help_action])
        graph.add_edge(root["id"], help_page["id"], help_action)
        returns = next(a for a in help_page["actions"] if a["name"] == "Returns")
        browser.act(returns)
        target = browser.observe(directory)
        graph.add_node(target, [help_action, returns])
        graph.add_edge(help_page["id"], target["id"], returns)
    graph.save()
    build_index(graph)
    return graph


def test_manual_replays_recorded_path_from_verified_root_every_time(site, manual_graph, tmp_path, monkeypatch):
    graph = manual_graph
    target = next(n for n in graph.nodes.values() if "Returns" in n["headings"])
    session = ManualSession(graph, tmp_path)
    query_id = session.query({"text": "Returns Return within 30 days", "top_k": 3})["query_id"]
    saved_index = json.loads((graph.directory / "index.json").read_text(encoding="utf-8"))
    assert {n["id"]: n["index_text"] for n in session.nodes()} == dict(zip(saved_index["ids"], saved_index["texts"]))
    with Browser(site, session.log, settle_ms=20) as browser:
        session.browser = browser
        reset, act = browser.reset, browser.act
        resets, actions = [], []

        def tracked_reset():
            resets.append(True)
            reset()

        def tracked_act(action, strategy="role"):
            actions.append(action["name"])
            act(action, strategy)

        monkeypatch.setattr(browser, "reset", tracked_reset)
        monkeypatch.setattr(browser, "act", tracked_act)
        for _ in range(2):
            result = session.visit({"node_id": target["id"], "source": "topk", "query_id": query_id})
            assert result["success"] and result["steps"] == 2
            assert browser.page.url == site + "/help"
            assert browser.page.locator("#tab-title").inner_text() == "Returns"
        assert len(resets) == 2 and actions == ["Help", "Returns", "Help", "Returns"]
    session.judge({"query_id": query_id, "verdict": "hit", "expected_node_id": target["id"], "note": "正确到达同 URL 的退货标签"})
    resumed = ManualSession(graph, tmp_path)
    assert len(resumed.data["attempts"]) == 2 and resumed.data["judgments"][0]["verdict"] == "hit"
    for row in resumed.data["attempts"]:
        events = [json.loads(line) for line in (tmp_path / "attempts" / row["id"] / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        kinds = [e["event"] for e in events]
        assert kinds.index("manual_root_verified") < kinds.index("replay_step")


def test_manual_failed_root_never_clicks_and_records_failure(site, manual_graph, tmp_path, monkeypatch):
    session = ManualSession(manual_graph, tmp_path)
    with Browser(site, session.log, settle_ms=20) as browser:
        session.browser = browser
        reset = browser.reset

        def wrong_root():
            reset()
            browser.page.locator("h1").evaluate("e=>e.textContent='Changed root'")

        monkeypatch.setattr(browser, "reset", wrong_root)
        monkeypatch.setattr(browser, "act", lambda *a: pytest.fail("must not click after root mismatch"))
        target = next(n for n in manual_graph.nodes.values() if n["depth"] == 2)
        result = session.visit({"node_id": target["id"], "source": "all"})
        assert not result["success"] and result["steps"] == 0
        assert session.data["attempts"][-1]["phase"] == "root_restore"
        assert list((tmp_path / "attempts").glob("*/failures/*/page.png"))


def test_manual_never_substitutes_shortest_path(manual_graph):
    graph = Graph(manual_graph.directory)
    graph.data = json.loads(json.dumps(manual_graph.data))
    target = next(n for n in graph.nodes.values() if n["depth"] == 2)
    fake = {"id": "shortcut", "source": graph.data["root"], "target": target["id"],
            "action": {"id": "fake", "name": "Shortcut"}}
    graph.data["edges"].insert(0, fake)
    assert len(graph.path(graph.data["root"], target["id"])) == 1
    assert len(discovery_path(graph, target["id"])) == 2


def test_manual_stops_after_intermediate_state_mismatch(site, manual_graph, tmp_path, monkeypatch):
    graph = Graph(manual_graph.directory)
    graph.data = json.loads(json.dumps(manual_graph.data))
    target = next(n for n in graph.nodes.values() if n["depth"] == 2)
    graph.data["edges"][0]["check"]["state_id"] = "wrong-state"
    session = ManualSession(graph, tmp_path)
    with Browser(site, session.log, settle_ms=20) as browser:
        session.browser = browser
        act, actions = browser.act, []

        def track(action, strategy="role"):
            actions.append(action["name"])
            act(action, strategy)

        monkeypatch.setattr(browser, "act", track)
        result = session.visit({"node_id": target["id"], "source": "all"})
        assert not result["success"] and result["failed_step"] == 0
        assert actions == ["Help"]
        assert browser.page.locator("#tab-title").inner_text() == "Shipping"
        assert result["screenshot"]


def test_manual_browsing_without_index_and_resume_interrupted(manual_graph, tmp_path):
    directory = tmp_path / "graph"
    directory.mkdir()
    (directory / "graph.json").write_bytes((manual_graph.directory / "graph.json").read_bytes())
    graph = Graph.load(directory)
    session = ManualSession(graph, tmp_path / "review")
    assert len(session.nodes()) == len(graph.nodes)
    assert all(n["index_text"] is None for n in session.nodes())
    with pytest.raises(ValueError, match="索引缺失"):
        session.query({"text": "Returns", "top_k": 2})
    session.data["attempts"].append({"id": "interrupted-test", "status": "running"})
    session.save()
    resumed = ManualSession(graph, tmp_path / "review")
    assert resumed.data["attempts"][0]["phase"] == "interrupted"


def test_manual_query_failure_and_human_coverage_verdicts(manual_graph, tmp_path):
    session = ManualSession(manual_graph, tmp_path)
    query_id = session.query({"text": "Test shop", "top_k": 1})["query_id"]
    candidates = session.data["queries"][-1]["candidates"]
    omitted = next(key for key in manual_graph.nodes if key != candidates[0]["id"])
    row = session.judge({"query_id": query_id, "verdict": "retrieval_miss", "expected_node_id": omitted})
    assert row["expected_node_id"] == omitted
    with pytest.raises(ValueError):
        session.judge({"query_id": query_id, "verdict": "hit", "expected_node_id": omitted})
    session.judge({"query_id": query_id, "verdict": "coverage_miss"})
    session.index_files["index.json"] = "changed"
    with pytest.raises(ValueError, match="索引已被修改"):
        session.query({"text": "Shipping", "top_k": 2})
    assert not session.data["queries"][-1]["success"]
    assert json.loads(session.result_path.read_text(encoding="utf-8"))["queries"][-1]["error"]


def test_manual_http_ui_commands_are_queued(manual_graph, tmp_path):
    session = ManualSession(manual_graph, tmp_path)
    commands = queue.Queue()
    server = create_server(session, commands, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url + "/api/state") as response:
            token = json.load(response)["token"]
        body = json.dumps({"op": "query", "text": "Returns", "top_k": 2}).encode()
        request = Request(url + "/api/command", body, {"Content-Type": "application/json", "X-Review-Token": token})
        assert urlopen(request).status == 202
        assert not session.data["queries"]  # HTTP threads only enqueue commands.
        with pytest.raises(HTTPError) as busy:
            urlopen(request)
        assert busy.value.code == 409
        execute_command(session, commands.get_nowait())
        assert len(session.data["queries"]) == 1
        request.add_header("Origin", "http://example.org")
        with pytest.raises(HTTPError) as denied:
            urlopen(request)
        assert denied.value.code == 403
        with pytest.raises(HTTPError) as missing:
            urlopen(url + "/.env")
        assert missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
