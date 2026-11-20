import json

import numpy as np
import pytest

from web2kg.common.browser import Browser
from web2kg.common.jina import maxsim, normalize
from web2kg.common.storage import EventLog, Graph
from web2kg.evaluation.manual import ManualSession
from web2kg.stage1.crawl import CrawlBudget, crawl
from web2kg.stage1.visual_index import build_visual_index
from web2kg.stage2.retrieval import retrieve


def test_maxsim_uses_all_query_tokens_and_normalizes():
    q, d = np.eye(128, dtype=np.float32)[:2], np.eye(128, dtype=np.float32)[:1]
    assert maxsim(q * 3, d * 7) == pytest.approx(0.5)
    assert maxsim(q, q) == pytest.approx(1)
    with pytest.raises(ValueError):
        normalize(np.zeros((2, 128)))
    with pytest.raises(ValueError):
        normalize(np.full((2, 128), np.nan))


class FakeJina:
    endpoint = "http://127.0.0.1:8793/v1/embeddings"
    log = None

    def __init__(self):
        self.calls = []
        self.version = "fixture-encoder"

    def identity(self):
        return {"fingerprint": self.version}

    def encode(self, value, kind, expected):
        self.calls.append((value, kind))
        i = 1 if kind == "text" or value.parent.name == "b" else 0
        return np.eye(128, dtype=np.float32)[i:i+1]


def test_screenshot_index_query_and_tampering(tmp_path):
    graph, client = Graph(tmp_path), FakeJina()
    for key in ("a", "b"):
        graph.add_node({"id": key, "title": key, "url": "http://test/"+key,
                        "headings": [], "actions": [], "text": "display only"}, [])
        folder = tmp_path / "snapshots" / key
        folder.mkdir(parents=True)
        (folder / "page.png").write_bytes(key.encode())
        (folder / "page-annotated.png").write_bytes(b"not for embedding")
    graph.save()
    graph.data.update(root='a', start_url='http://test/a')
    graph.add_edge('a', 'b', {'id': 'b', 'kind': 'click', 'role': 'link', 'name': 'b', 'href': 'http://test/b'})
    index = build_visual_index(graph, client)
    assert all(v.name == "page.png" and kind == "image" for v, kind in client.calls)
    assert index["text_role"] == "display_only_not_embedded"
    assert [r["id"] for r in retrieve(graph, "page", client, 1)] == ["b"]
    assert len(retrieve(graph, "page", client, 10)) == 2
    saved_manifest = (tmp_path / "index.json").read_bytes()
    class Interrupted(FakeJina):
        def encode(self, value, kind, expected):
            raise ConnectionError("encoder stopped")
    with pytest.raises(ConnectionError):
        build_visual_index(graph, Interrupted())
    assert (tmp_path / "index.json").read_bytes() == saved_manifest
    assert retrieve(graph, "page", client, 1)[0]["id"] == "b"
    client.version = "different"
    with pytest.raises(ValueError, match="encoder differs"):
        retrieve(graph, "page", client)
    client.version = "fixture-encoder"
    (tmp_path / "snapshots/b/page.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="screenshot changed"):
        retrieve(graph, "page", client)


def test_structural_identity_has_documented_content_blindness(site, tmp_path):
    with Browser(site, EventLog(tmp_path), identity_mode="webnavigator") as browser:
        before = browser.observe()
        browser.page.locator("h1").evaluate("e=>e.textContent='Changed title'")
        changed = browser.observe()
        assert changed["id"] == before["id"]
        assert changed["semantic_id"] != before["semantic_id"]
        browser.page.locator("main").evaluate("e=>e.insertAdjacentHTML('beforeend','<section>new</section>')")
        assert browser.observe()["id"] != before["id"]


def test_adaptive_crawl_and_manual_replay(site, tmp_path):
    directory = tmp_path / "graph"
    with Browser(site, EventLog(directory), identity_mode="webnavigator", settle_ms=20) as browser:
        graph = crawl(browser, directory, browser.log, CrawlBudget(10, 2, 12, 1, 18))
    assert graph.data["exploration"] == "adaptive_bfs"
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "structural_difference" and e["parent"] for e in events)
    assert any(d["reason"] == "inherited_dom_path" for e in events if e["event"] == "risk_filter" for d in e["decisions"])
    target = next(n for n in graph.nodes.values() if n["url"] == site + "/catalog")
    session = ManualSession(graph, tmp_path / "manual")
    with Browser(site, session.log, identity_mode="webnavigator", settle_ms=20) as browser:
        session.browser = browser
        result = session.visit({"node_id": target["id"], "source": "all"})
        assert result["success"] and result["steps"] == 1
        assert browser.page.url == site + "/catalog"
