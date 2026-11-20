import json

import pytest

from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog, Graph
from web2kg.stage1.crawl import CrawlBudget, crawl
from web2kg.stage1.llm_exploration import LLMExplorer
from web2kg.stage1.url_pruning import known_targets
from test_llm_exploration import FakeClient, observation


def known_graph(tmp_path):
    graph = Graph(tmp_path); node = observation(); graph.data["start_url"] = node["url"]
    node["actions"] = [dict(node["actions"][0], navigation_kind="plain_link")]
    graph.add_node(node, [])
    target = dict(node, id="target", url=node["actions"][0]["href"], http_status=200)
    graph.add_node(target, [node["actions"][0]])
    graph.data['root'] = node['id']
    graph.add_edge(node['id'], target['id'], node['actions'][0])
    return graph, node, node["actions"][0]


def test_known_target_filtered_before_model_without_edge_or_attempt(tmp_path):
    graph, node, action = known_graph(tmp_path)
    model = FakeClient(EventLog(tmp_path)); explorer = LLMExplorer(model, tmp_path)
    assert explorer.select(graph, node, 12) == []
    assert model.calls == [] and len(graph.data['edges']) == 1  # No new edge from pruning.
    record = graph.data["pruned_links"][0]
    assert not record["action_attempted"] and not record["verified_edge"]
    assert record["known_targets"] == ["target"]
    explorer.select(graph, node, 12)
    assert len(graph.data["pruned_links"]) == 1


def test_known_but_unreachable_target_is_not_pruned(tmp_path):
    graph, node, action = known_graph(tmp_path)
    graph.data['edges'][0]['replayable'] = False
    assert known_targets(graph, node, action) == []


@pytest.mark.parametrize("change", [
    {"role": "button"}, {"role": "tab"}, {"kind": "fill"},
    {"navigation_kind": "interactive_or_unknown"}, {"navigation_kind": None},
    {"href": "http://localhost/"}, {"href": "http://localhost/A#reply"},
    {"href": "http://localhost/A?page=2"}, {"method": "post"},
    {"state": {"expanded": False}},
])
def test_local_unknown_query_fragment_and_unsafe_actions_are_not_pruned(tmp_path, change):
    graph, node, action = known_graph(tmp_path)
    assert known_targets(graph, node, dict(action, **change)) == []


@pytest.mark.parametrize("status", [None, 200, 204, 301, 302, 403, 404, 500])
def test_observed_url_is_pruned_independently_of_http_status(tmp_path, status):
    graph, node, action = known_graph(tmp_path)
    graph.nodes["target"]["http_status"] = status
    assert known_targets(graph, node, action) == ["target"]


def test_candidate_href_alone_does_not_mark_destination_visited(tmp_path):
    graph, node, action = known_graph(tmp_path)
    del graph.nodes["target"]
    assert known_targets(graph, node, action) == []


def test_real_browser_classifies_plain_and_interactive_links(site, tmp_path):
    with Browser(site, EventLog(tmp_path), identity_mode="interaction", settle_ms=10) as browser:
        browser.page.set_content('''<a href="/catalog">Plain</a><a href="/catalog" onclick="return false">Script</a>
            <a href="/catalog" data-toggle="menu">Data</a><a href="/catalog" aria-expanded="false">Expand</a>
            <a href="/catalog" target="_blank">Window</a><form><a href="/catalog">Form</a></form>''')
        page = browser.observe()
        assert page["http_status"] == 200
        assert {a["name"] for a in page["actions"] if a.get("navigation_kind") == "plain_link"} == {"Plain"}
        browser.page.evaluate("history.pushState(null, '', '/unobserved-route')")
        assert browser.observe()["http_status"] is None


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("target_status", [200, 404])
def test_two_stage_pruning_keeps_frontier_reachable_and_resume_skips_pruned(site, tmp_path, monkeypatch, interrupted, target_status):
    from fixture_site import FixtureHandler
    from scripts.summarize_llm_crawl import summarize
    original = FixtureHandler.do_GET
    pages = {
        "/prune-root": '''<h1>Root</h1><a href="/prune-target">First</a><a href="/prune-target">Second</a>
            <button aria-expanded="false" onclick="this.setAttribute('aria-expanded',this.getAttribute('aria-expanded')!=='true');
            document.querySelector('nav').hidden=!document.querySelector('nav').hidden">Menu</button>
            <nav hidden><a href="/prune-settings">Settings</a></nav>''',
        "/prune-target": '<h1>Target</h1><a href="/prune-child">Child</a><a href="/prune-root">Root</a>',
        "/prune-child": '<h1>Child</h1><a href="/prune-root">Root</a>',
        "/prune-settings": '<h1>Settings</h1><a href="/prune-root">Root</a>',
    }
    def serve(handler):
        if handler.path not in pages:
            return original(handler)
        handler.send_response(target_status if handler.path == "/prune-target" else 200)
        handler.send_header("Content-Type", "text/html"); handler.end_headers()
        handler.wfile.write(pages[handler.path].encode())
    monkeypatch.setattr(FixtureHandler, "do_GET", serve)
    log = EventLog(tmp_path); model = FakeClient(log)
    if interrupted:
        original_json = model.json
        descriptions = 0
        def fail_after_pruning(system, payload, schema, purpose):
            nonlocal descriptions
            if purpose == "transition_description":
                descriptions += 1
                model.fail = descriptions == 3
            return original_json(system, payload, schema, purpose)
        model.json = fail_after_pruning
    model.identity = {**model.identity, "max_calls": 1000}
    explorer = LLMExplorer(model, tmp_path); budget = CrawlBudget(8, 3, 5, 1, 20)
    if interrupted:
        with Browser(site + "/prune-root", log, identity_mode="interaction", settle_ms=10) as browser:
            with pytest.raises(RuntimeError, match="annotation failed"):
                crawl(browser, tmp_path, log, budget, explorer=explorer)
        assert Graph.load(tmp_path).data["pruned_links"]
    with Browser(site + "/prune-root", log, identity_mode="interaction", settle_ms=10) as browser:
        graph = crawl(browser, tmp_path, log, budget, explorer=explorer, resume=interrupted)
    assert any(n["url"].endswith("/prune-child") for n in graph.nodes.values())
    assert any(n["url"].endswith("/prune-target") and n["http_status"] == target_status for n in graph.nodes.values())
    assert any(n["url"].endswith("/prune-settings") for n in graph.nodes.values())
    if interrupted:
        blocked = {e['target'] for e in graph.data['edges'] if e.get('replayable') is False}
        assert blocked  # Failed Luna annotation remains evidence, never a replay permission.
        assert set(graph.nodes) - graph.reachable() <= blocked
    else:
        assert graph.reachable() == set(graph.nodes)
    pruned = graph.data["pruned_links"]
    assert {r["phase"] for r in pruned} == {"selection", "before_attempt"}
    assert any(r["source"] == graph.data["root"] and r["action"]["name"] == "First" for r in pruned)
    events = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    attempts = {(e["source"], e["action"]["id"]) for e in events if e["event"] == "crawl_attempt_started"}
    assert all((r["source"], r["action"]["id"]) not in attempts for r in pruned)
    assert len(pruned) == len({(r["source"], r["action"]["id"]) for r in pruned})
    assert all(not r["verified_edge"] for r in pruned)
    summary = summarize(tmp_path)
    assert summary["checks"]["executed_in_selected_order"]
    assert summary["checks"]["no_pruned_action_attempted"]
    assert summary["checks"]["no_duplicate_state_action"]
    assert summary["pruned_links"] == len(pruned)
    if not interrupted:
        from playwright.sync_api import sync_playwright
        from web2kg.visualization.viewer import export_viewer
        with sync_playwright() as p:
            browser = p.chromium.launch(); page = browser.new_page(); errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(export_viewer(graph).resolve().as_uri())
            assert "目标 URL 已访问" in page.locator("#detail").inner_text()
            page.locator("#logTab").click()
            row = page.locator(".log-row").filter(has_text="已知目标剪枝（未执行）").first
            row.locator("summary").first.click()
            assert "未生成已验证边" in row.inner_text()
            assert not errors
            browser.close()
