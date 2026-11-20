import copy
import json

import pytest

from web2kg.common.browser import Browser
from web2kg.common.interaction import changes, state_id
from web2kg.common.replay import verify
from web2kg.common.storage import EventLog, Graph
from web2kg.stage1.llm_exploration import LLMExplorer
from web2kg.stage1.crawl import CrawlBudget, crawl
from web2kg.common.replay import restore
from test_llm_exploration import FakeClient, observation


@pytest.mark.parametrize("mode", ["interaction", "webnavigator"])
def test_css_menu_state_and_replay_source_check(site, tmp_path, mode):
    with Browser(site, EventLog(tmp_path), identity_mode=mode, settle_ms=10) as browser:
        browser.page.set_content('''<h1>Portal</h1><button aria-expanded="false"
            onclick="this.setAttribute('aria-expanded',this.getAttribute('aria-expanded')!=='true');
            document.querySelector('nav').hidden=!document.querySelector('nav').hidden">Account</button>
            <nav hidden><a href="/settings">Settings</a></nav><p id="clock">Time 12:00</p>''')
        before = browser.observe()
        browser.page.get_by_role("button").click()
        after = browser.observe()
        assert (after["id"] != before["id"]) == (mode == "interaction")
        assert any(r["control"]["name"] == "Settings" for r in changes(before, after)["added"])
        if mode == "interaction":
            with pytest.raises(ValueError, match="State mismatch"):
                verify(browser, before["id"])
        browser.page.get_by_role("button").click()
        assert browser.observe()["id"] == before["id"]
        browser.page.locator("#clock").evaluate("e => e.textContent='Time 12:01'")
        clock = browser.observe()
        assert clock["id"] == before["id"]
        assert clock["semantic_id"] != before["semantic_id"]


def test_form_values_selection_disabled_dialog_and_entities(site, tmp_path):
    with Browser(site, EventLog(tmp_path), identity_mode="interaction", settle_ms=10) as browser:
        browser.page.set_content('''<h1>Search</h1><input aria-label="Query"><input type="checkbox" aria-label="Only mine">
            <button>Apply</button><a href="/item/1">Item</a><dialog><h2>Confirm</h2></dialog>''')
        root = browser.observe()
        browser.page.get_by_role("textbox").fill("example")
        filled = browser.observe()
        assert filled["id"] != root["id"]
        assert "example" not in json.dumps(filled["actions"])
        browser.page.get_by_role("checkbox").check()
        checked = browser.observe()
        assert checked["id"] != filled["id"]
        browser.page.get_by_role("button").evaluate("e=>e.disabled=true")
        disabled = browser.observe()
        assert disabled["id"] != checked["id"]
        browser.page.get_by_role("link").evaluate("e=>e.href='/item/2'")
        other = browser.observe()
        assert other["id"] != disabled["id"]
        browser.page.locator("dialog").evaluate("e=>e.showModal()")
        assert browser.observe()["id"] != other["id"]


def test_identity_keeps_duplicate_count_order_url_and_headings():
    original = observation(); original["actions"] = original["actions"][:2]
    key = state_id(original, [])
    for change in (lambda n: n["actions"].reverse(), lambda n: n["actions"].append(n["actions"][0]),
                   lambda n: n.update(url="http://localhost/other"), lambda n: n.update(headings=["Other"])):
        changed = copy.deepcopy(original); change(changed)
        assert state_id(changed, []) != key
    layout = copy.deepcopy(original)
    layout["actions"][0]["xpath"] = "/html/body/div/a"
    layout["text"] = "Unrelated clock"
    assert state_id(layout, []) == key


def test_high_similarity_reply_sends_actual_changes_and_ax_without_second_call(tmp_path):
    graph = Graph(tmp_path); before = observation(); graph.data["start_url"] = before["url"]
    before["text"] = "\n".join(f"Comment {i}" for i in range(30)); graph.add_node(before, [])
    model = FakeClient(EventLog(tmp_path)); explorer = LLMExplorer(model, tmp_path)
    chosen = explorer.select(graph, before, 1)[0]
    after = copy.deepcopy(before)
    after["actions"].append({"id": "replybox", "role": "textbox", "name": "Comment", "state": {"required": True}})
    after["ax"] = [{"nodeId": "1", "role": {"value": "textbox"}, "name": {"value": "Comment"},
                    "properties": [{"name": "required", "value": {"value": True}}]}]
    explorer.describe(graph, before, chosen, after, 1, "transition", "capture_result", True, before=before)
    payload = model.calls[-1][1]
    assert len(model.calls) == 2  # Ranking plus one description, no repair API call.
    assert payload["observed_changes"]["similarity"] >= .8
    assert payload["observed_changes"]["added"][0]["control"]["name"] == "Comment"
    assert payload["after_accessibility"]["ax_tree"][0]["state"]["required"]
    assert payload["executed_action"]["control"] == chosen
    assert (tmp_path / "transitions/00001-before.json").exists()
    assert graph.data["transition_records"][0]["before_observation"] == "live"


def test_low_similarity_and_missing_after_do_not_invent_context(tmp_path):
    graph = Graph(tmp_path); before = observation(); graph.data["start_url"] = before["url"]
    model = FakeClient(EventLog(tmp_path)); explorer = LLMExplorer(model, tmp_path)
    chosen = explorer.select(graph, before, 1)[0]
    after = {"id": "empty", "url": before["url"] + "empty", "title": "Empty", "actions": [], "text": "Empty"}
    explorer.describe(graph, before, chosen, after, 1, "transition", "capture_result", True, before=before)
    assert "after_accessibility" not in model.calls[-1][1]
    assert len(model.calls[-1][1]["observed_changes"]["removed"]) == 3
    explorer.describe(graph, before, chosen, None, 2, "not_attempted", "restore_source", False)
    payload = model.calls[-1][1]
    assert not payload["observed_changes"]["available"]
    assert payload["before_observation"] == "saved_node"
    assert "after_accessibility" not in payload


def test_exact_link_history_keeps_failures_distinct_and_does_not_ban_buttons(tmp_path):
    graph = Graph(tmp_path); node = observation(); graph.data["start_url"] = node["url"]
    action = node["actions"][0]
    graph.data["transition_records"] = [
        {"source": "other", "target": "known", "target_page": {"url": action["href"]},
         "action": action, "outcome": "transition"},
        {"source": "failed", "action": action, "outcome": "failed"}]
    model = FakeClient(EventLog(tmp_path)); explorer = LLMExplorer(model, tmp_path)
    explorer.select(graph, node, 2)
    evidence = model.calls[0][1]["known_link_history"][action["id"]]
    assert evidence["observed_targets"] == ["known"]
    assert evidence["outcomes"] == {"transition": 1, "failed": 1}
    assert action["id"] in {a["id"] for a in model.calls[0][1]["page"]["candidates"]}


def test_bfs_reaches_link_inside_css_menu_and_replays_two_steps(site, tmp_path, monkeypatch):
    from fixture_site import FixtureHandler
    original = FixtureHandler.do_GET
    def page(handler):
        if handler.path != "/menu-fixture":
            return original(handler)
        handler.send_response(200); handler.send_header("Content-Type", "text/html"); handler.end_headers()
        handler.wfile.write(b'''<title>Menu</title><h1>Menu</h1><button aria-expanded="false"
            onclick="this.setAttribute('aria-expanded',this.getAttribute('aria-expanded')!=='true');
            document.querySelector('nav').hidden=!document.querySelector('nav').hidden">Account</button>
            <nav hidden><a href="/settings">Settings</a></nav>''')
    monkeypatch.setattr(FixtureHandler, "do_GET", page)
    log = EventLog(tmp_path); model = FakeClient(log)
    with Browser(site + "/menu-fixture", log, identity_mode="interaction", settle_ms=10) as browser:
        graph = crawl(browser, tmp_path, log, CrawlBudget(4, 2, 2, 1, 5), explorer=LLMExplorer(model, tmp_path))
        target = next(n for n in graph.nodes.values() if n["url"].endswith("/settings"))
        assert target["depth"] == 2
        assert len([n for n in graph.nodes.values() if n["url"].endswith("/menu-fixture")]) == 2
        restore(browser, graph, target["id"], log)
        assert browser.observe()["id"] == target["id"]
    assert all(r["before_observation"] == "live" for r in graph.data["transition_records"])


def test_large_ax_and_changes_have_explicit_payload_bounds(tmp_path):
    graph = Graph(tmp_path); before = observation(); graph.data["start_url"] = before["url"]
    before["text"] = "\n".join(f"Row {i}" for i in range(1000))
    model = FakeClient(EventLog(tmp_path)); explorer = LLMExplorer(model, tmp_path)
    chosen = explorer.select(graph, before, 1)[0]
    after = copy.deepcopy(before)
    after["ax"] = [{"nodeId": "1", "role": {"value": "main"}, "name": {"value": "Results"}},
                   {"nodeId": "2", "parentId": "1", "role": {"value": "StaticText"}, "name": {"value": "x" * 13000}}]
    after["actions"] += [{"id": f"new-{i}", "role": "button", "name": f"New {i}"} for i in range(40)]
    explorer.describe(graph, before, chosen, after, 1, "transition", "capture_result", True, before=before)
    payload = model.calls[-1][1]
    assert payload["after_accessibility"]["mode"] == "large_regions"
    assert "ax_tree" not in payload["after_accessibility"]
    assert payload["observed_changes"]["added_omitted"] == 8
    assert len(payload["observed_changes"]["added"]) == 32
    assert len(graph.data["transition_records"][0]["observed_changes"]["added"]) == 40
