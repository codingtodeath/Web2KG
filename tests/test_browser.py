import copy
import json
import struct
from types import SimpleNamespace

import pytest

from fixture_site import FixtureHandler
from web2kg.common.browser import Browser
from web2kg.stage1.crawl import CrawlBudget, crawl
from web2kg.common.diagnostics import record_failure
from web2kg.evaluation.replay import replay_experiment
from web2kg.stage1.indexing import build_index
from web2kg.common.replay import execute_path, restore
from web2kg.stage2.navigation import navigate
from web2kg.common.storage import EventLog
from web2kg.visualization.viewer import export_viewer


@pytest.fixture(scope="module")
def fixture_graph(site, tmp_path_factory):
    directory = tmp_path_factory.mktemp("graph")
    log = EventLog(directory)
    with Browser(site, log, settle_ms=30) as browser:
        graph = crawl(browser, directory, log, CrawlBudget(max_states=12, max_depth=3, actions_per_page=8))
    return graph


def test_bfs_discovers_tabs_and_groups_products(fixture_graph):
    graph = fixture_graph
    assert 5 <= len(graph.nodes) <= 12
    assert sum("/item/" in node["url"] for node in graph.nodes.values()) == 1
    help_nodes = [n for n in graph.nodes.values() if n["url"].endswith("/help")]
    assert len(help_nodes) == 2
    assert {tuple(n["headings"]) for n in help_nodes} == {("Help center", "Shipping"), ("Help center", "Returns")}
    for node in graph.nodes.values():
        assert (graph.directory / "snapshots" / node["id"] / "page.png").exists()
        assert len(node["path"]) == node["depth"]


def test_replay_same_url_state_and_restore(site, fixture_graph, tmp_path):
    log = EventLog(tmp_path)
    target = next(n["id"] for n in fixture_graph.nodes.values() if "Returns" in n["headings"])
    with Browser(site, log, settle_ms=30) as browser:
        assert navigate(browser, fixture_graph, target, log)["success"]
        assert browser.page.locator("#tab-text").inner_text() == "Return within 30 days."
        restore(browser, fixture_graph, fixture_graph.data["root"], log)
        assert browser.observe()["id"] == fixture_graph.data["root"]


def test_failed_postcondition_stops_path(site, fixture_graph, tmp_path):
    log = EventLog(tmp_path)
    target = next(n["id"] for n in fixture_graph.nodes.values() if "Returns" in n["headings"])
    path = copy.deepcopy(fixture_graph.path(fixture_graph.data["root"], target))
    path[0]["check"]["state_id"] = "nonexistent-state"
    with Browser(site, log, settle_ms=30) as browser:
        result = execute_path(browser, fixture_graph, path, log)
        assert not result["success"] and result["failed_step"] == 0
        assert browser.page.locator("#tab-title").inner_text() == "Shipping"
    events = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    failure = next(e for e in events if e["event"] == "replay_step" and not e["success"])
    assert failure["phase"] == "target_check" and failure["action_attempted"]
    assert failure["source_page"]["url"].rstrip("/") == site
    assert failure["action"]["name"] == "Help"
    assert failure["failure_page"]["url"].endswith("/help")
    assert (tmp_path / failure["failure_page"]["snapshot_dir"] / "page-annotated.png").exists()


def test_browser_does_not_block_writes_by_keyword_or_method(site, tmp_path):
    with Browser(site, EventLog(tmp_path), settle_ms=30) as browser:
        assert browser.page.evaluate("fetch('/change',{method:'POST'}).then(r=>r.ok).catch(()=>false)")
        assert browser.page.evaluate("fetch('/delete').then(r=>r.ok).catch(()=>false)")


def test_hash_changes_when_link_destination_changes(site, tmp_path):
    with Browser(site, EventLog(tmp_path), settle_ms=30) as browser:
        before = browser.observe()["id"]
        assert browser.observe()["id"] == before
        browser.page.get_by_role("link", name="Catalog", exact=True).evaluate("e => e.href='/help'")
        assert browser.observe()["id"] != before


def test_repeated_unnamed_landmarks_are_located_by_occurrence(site, tmp_path):
    with Browser(site, EventLog(tmp_path), settle_ms=30) as browser:
        browser.page.set_content('<nav><a href="/help">Open</a></nav><nav><a href="/catalog">Open</a></nav>')
        observation = browser.observe()
        action = next(a for a in observation["actions"] if a["href"].endswith("/catalog"))
        assert action["region"]["occurrence"] == 1
        assert action["region"]["match_count"] == 2
        browser.act(action)
        assert browser.page.url == site + "/catalog"


def test_replay_experiment_both_locators(site, fixture_graph, tmp_path):
    log = EventLog(tmp_path)
    with Browser(site, log, settle_ms=30) as browser:
        result = replay_experiment(browser, fixture_graph, log, count=2, repeats=2)
    assert len(result["trials"]) == 8
    assert all(t["success"] for t in result["trials"])


def test_search_fill_then_enter(site, tmp_path):
    with Browser(site + "/catalog", EventLog(tmp_path), settle_ms=30) as browser:
        field = next(a for a in browser.observe()["actions"] if a["role"] == "searchbox")
        browser.act({**field, "kind": "fill", "value": "blue mug"})
        browser.act({**field, "kind": "press", "value": "Enter"})
        assert browser.page.locator("h1").inner_text() == "Search results"


def test_viewer_renders_and_switches_tabs(site, fixture_graph, tmp_path):
    path = export_viewer(fixture_graph)
    with Browser(site, EventLog(tmp_path), settle_ms=30) as browser:
        # A separate unrestricted page only opens a locally generated artifact.
        page = browser.browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(path.resolve().as_uri())
        assert page.locator(".node").count() == len(fixture_graph.nodes)
        assert page.locator(".edge-label").count() == len(fixture_graph.nodes) - 1
        page.locator(".edge-label").first.click()
        assert page.locator("#edgeDetail").is_visible()
        assert "点击" in page.locator("#edgeDetail").inner_text()
        page.locator("#edgeMode").select_option("all")
        assert page.locator(".edge").count() == len(fixture_graph.data["edges"])
        assert page.locator(".selected-controls [data-status='attempted']").count() > 0
        assert page.locator(".activity-box.attempted").count() > 0
        assert page.locator("#detail .snapshot img").get_attribute("src").endswith("page-annotated.png")
        page.locator(".candidate-row").first.click()
        assert page.get_by_role("dialog").is_visible()
        assert "#1" in page.locator("#previewTitle").inner_text()
        page.get_by_role("button", name="关闭预览").click()
        page.get_by_role("button", name="事件日志", exact=True).click()
        assert page.locator("#logPanel").is_visible()
        page.get_by_role("button", name="只看失败").click()
        assert not errors
        page.close()


def test_annotation_keeps_original_state_and_cleans_up_on_error(site, tmp_path, monkeypatch):
    from playwright.sync_api import Error as BrowserError
    with Browser(site, EventLog(tmp_path), settle_ms=30) as browser:
        observation = browser.observe(tmp_path)
        folder = tmp_path / "snapshots" / observation["id"]
        original = (folder / "page.png").read_bytes()
        metadata = browser.save_annotation(observation, folder)
        assert (folder / "page.png").read_bytes() == original
        assert browser.observe()["id"] == observation["id"]
        assert len(metadata["items"]) == len(observation["actions"])
        assert [a["number"] for a in metadata["items"]] == list(range(1, len(metadata["items"]) + 1))
        assert (folder / "page-annotated.png").read_bytes() != original
        first, second = observation["actions"][:2]
        colored = browser.save_annotation(observation, folder, {first["id"]: "selected", second["id"]: "attempted"})
        assert colored["items"][0]["color"] == "#ea580c"
        assert colored["items"][1]["color"] == "#c026d3"
        assert (folder / "page.png").read_bytes() == original
        def fail(**kwargs):
            raise BrowserError("screenshot unavailable")
        monkeypatch.setattr(browser.page, "screenshot", fail)
        with pytest.raises(BrowserError):
            browser.save_annotation(observation, folder)
        assert browser.page.locator("[data-web2kg-overlay]").count() == 0
        assert browser.observe()["id"] == observation["id"]


def test_full_document_screenshots_reach_below_viewport_and_feed_vision(site, tmp_path):
    def dimensions(path):
        return struct.unpack(">II", path.read_bytes()[16:24])

    log = EventLog(tmp_path)
    with Browser(site, log, settle_ms=30, viewport={"width": 640, "height": 480}) as browser:
        browser.page.set_content('''<style>body{margin:0;width:1500px;height:2600px}
            button{position:absolute;top:2500px;left:1200px}</style>
            <h1>Long document</h1><button>Bottom action</button>''')
        observation = browser.observe(tmp_path)
        folder = tmp_path / "snapshots" / observation["id"]
        original = (folder / "page.png").read_bytes()
        assert dimensions(folder / "page.png") == (1500, 2600)
        assert dimensions(folder / "page-annotated.png") == (1500, 2600)
        marks = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))
        button = next(action for action in observation["actions"] if action["name"] == "Bottom action")
        bottom = next(item for item in marks["items"] if item["id"] == button["id"])
        assert bottom["status"] == "marked" and bottom["rect"]["y"] >= 2500
        record_failure(browser, log, "test_failure", observation, observation["actions"][0],
                       ValueError("Failed at bottom button"), "action", True)
        failure = json.loads(log.path.read_text(encoding="utf-8").splitlines()[-1])
        actual = tmp_path / failure["failure_page"]["snapshot_dir"]
        assert dimensions(actual / "page.png") == (1500, 2600)
        assert dimensions(actual / "page-annotated.png") == (1500, 2600)
        assert (folder / "page.png").read_bytes() == original


def test_restore_failure_does_not_claim_candidate_was_clicked(site, tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("source cannot be restored")
    monkeypatch.setattr("web2kg.stage1.crawl.restore", fail)
    log = EventLog(tmp_path)
    with Browser(site, log, settle_ms=30) as browser:
        crawl(browser, tmp_path, log, CrawlBudget(max_attempts=1))
    events = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    failure = next(e for e in events if e["event"] == "exploration_failed")
    assert failure["phase"] == "restore_source" and not failure["action_attempted"]
    assert failure["source_page"]["title"] == "Test shop"
    assert failure["action"]["name"] and failure["action"]["number"]
    assert failure["failure_page"]["url"].rstrip("/") == site


def test_agent_failure_and_external_log_assets(site, fixture_graph, tmp_path):
    log = EventLog(tmp_path)
    with Browser(site, log, settle_ms=30) as browser:
        before = browser.observe()
        record_failure(browser, log, "agent_error", before, {"kind": "click", "name": "Unknown control", "id": "unknown"}, ValueError("Unknown control"), "decision_validation", False)
    rows = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    failure = next(e for e in rows if e["event"] == "agent_error")
    assert failure["source_page"]["title"] == "Test shop"
    assert failure["phase"] == "decision_validation" and not failure["action_attempted"]
    path = export_viewer(fixture_graph, log.path)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(path.as_uri())
        data = page.evaluate("JSON.parse(document.getElementById('data').textContent)")
        failure = next(e for e in data["events"] if e["event"] == "agent_error")
        asset = failure["failure_page"]["view"]["annotated"]
        assert asset.startswith("viewer-assets/")
        assert (fixture_graph.directory / asset).is_file()
        page.get_by_role("button", name="事件日志", exact=True).click()
        page.get_by_role("button", name="只看失败", exact=True).click()
        summary = page.locator(".log-row.failure > summary").first
        assert "Test shop" in summary.inner_text() and "未执行" in summary.inner_text()
        summary.click()
        page.get_by_role("button", name="查看失败现场截图").click()
        assert page.get_by_role("dialog").is_visible()
        assert page.locator("#preview img").evaluate("img=>img.complete&&img.naturalWidth>0")
        browser.close()
