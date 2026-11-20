from contextlib import contextmanager

from playwright.sync_api import sync_playwright

from web2kg.common.storage import Graph
from web2kg.visualization.viewer import export_viewer


@contextmanager
def graph_page(tmp_path):
    graph = Graph(tmp_path)
    for i in range(14):
        key = f"node-{i}"
        graph.nodes[key] = dict(id=key, title=f"Page {i}", url=f"http://example.test/{i}",
                                headings=[], actions=[], path=[], depth=0)
    graph.data.update(root="node-0", start_url="http://example.test/0")
    pairs = [(0, i) for i in range(1, 12)] + [(1, 12), (12, 13),
             (1, 0), (1, 0), (2, 1), (1, 2), (2, 2), (13, 0), (0, 13)]
    for i, (source, target) in enumerate(pairs):
        graph.add_edge(f"node-{source}", f"node-{target}",
                       dict(id=f"action-{i}", kind="click", role="link", name=f"Action {i}"))
    path = export_viewer(graph)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(path.resolve().as_uri())
        yield page, graph
        assert not errors
        browser.close()


def assert_fits(page):
    page.wait_for_function("""() => {
        const g = document.querySelector('#graph'), s = g.querySelector('svg').getBoundingClientRect();
        return s.width <= g.clientWidth && s.height <= g.clientHeight &&
            g.scrollWidth <= g.clientWidth && g.scrollHeight <= g.clientHeight;
    }""")


def test_graph_zoom_fit_pan_and_selection(tmp_path):
    with graph_page(tmp_path) as (page, graph):
        assert_fits(page)
        page.locator("#edgeMode").select_option("all")
        assert_fits(page)
        page.locator("#zoomReset").click()
        assert page.locator("#zoomLevel").inner_text() == "100%"
        page.locator("#zoomIn").click()
        assert page.locator("#zoomLevel").inner_text() == "125%"
        page.locator("#zoomOut").click()
        assert page.locator("#zoomLevel").inner_text() == "100%"
        page.locator("#graph").evaluate("g => {g.scrollLeft = 220; g.scrollTop = 150;}")
        rect = page.locator("#graph").bounding_box()
        # Drag the empty left margin; graph items keep their own click handlers.
        page.locator("#graph").evaluate("g => {g.scrollLeft = 0; g.scrollTop = 150;}")
        page.mouse.move(rect["x"] + 8, rect["y"] + 80)
        page.mouse.down()
        page.mouse.move(rect["x"] + 8, rect["y"] + 40, steps=4)
        page.mouse.up()
        assert page.locator("#graph").evaluate("g => g.scrollTop") == 190
        page.locator(".connection-row").last.click()
        assert page.locator("#zoomLevel").inner_text() == "100%"
        assert page.locator("#edgeDetail").is_visible()
        assert page.locator(".edge-group.chosen").get_attribute("data-edge") == graph.data["edges"][-1]["id"]
        page.locator("#zoomFit").click()
        assert_fits(page)
        page.set_viewport_size({"width": 390, "height": 844})
        assert_fits(page)
        page.locator("#logTab").click()
        page.locator("#graphTab").click()
        assert_fits(page)


def test_node_scope_filters_layout_and_labels_legacy_paths(tmp_path):
    with graph_page(tmp_path) as (page, graph):
        assert "保存路径不完整" in page.locator("#edgeMode").inner_text()
        page.locator('[data-node="node-1"]').click()
        page.locator("#nodeScope").select_option("children")
        assert page.locator(".node").count() == 2
        assert_fits(page)
        page.locator("#nodeScope").select_option("all")
        assert page.locator(".node").count() == 14
        assert "保存路径缺失或不唯一" in page.locator("#detail").inner_text()


def test_graph_routes_keep_endpoints_and_avoid_cards(tmp_path):
    with graph_page(tmp_path) as (page, graph):
        page.locator("#edgeMode").select_option("all")
        assert page.locator(".edge-group").count() == len(graph.data["edges"])
        assert page.locator(".connection-row").count() == len(graph.data["edges"])
        problems = page.evaluate("""() => {
            const cards = [...document.querySelectorAll('.node')].map(n => {
                const t = n.transform.baseVal.getItem(0).matrix, b = n.querySelector('rect').getBBox();
                return {id:n.dataset.node, x:t.e, y:t.f, w:b.width, h:b.height};
            });
            const problems = [], routes = new Set();
            for (const group of document.querySelectorAll('.edge-group')) {
                const path = group.querySelector('.edge'), d = path.getAttribute('d');
                if (routes.has(d)) problems.push('identical route');
                routes.add(d);
                const length = path.getTotalLength(), start = path.getPointAtLength(0), end = path.getPointAtLength(length);
                const source = cards.find(n => n.id === group.dataset.source), target = cards.find(n => n.id === group.dataset.target);
                if (Math.abs(start.x - source.x - source.w) > .01 || start.y <= source.y || start.y >= source.y + source.h)
                    problems.push('wrong source port');
                if (Math.abs(end.x - target.x) > .01 || end.y <= target.y || end.y >= target.y + target.h)
                    problems.push('wrong target port');
                for (let t = 1; t < length; t += 3) {
                    const p = path.getPointAtLength(t);
                    if (cards.some(n => p.x > n.x + .1 && p.x < n.x+n.w-.1 && p.y > n.y+.1 && p.y < n.y+n.h-.1)) {
                        problems.push('line crosses a card'); break;
                    }
                }
                if (!/E\\d+ · N\\d+ → N\\d+/.test(group.querySelector('.edge-label').textContent))
                    problems.push('missing persistent endpoints');
            }
            return problems;
        }""")
        assert problems == []


def test_plain_wheel_and_drag_from_node_preserve_clicks(tmp_path):
    with graph_page(tmp_path) as (page, graph):
        page.locator("#zoomReset").click()
        wrap = page.locator("#graph")
        rect = wrap.bounding_box()
        page.mouse.move(rect["x"] + 100, rect["y"] + 100)
        outer_scroll = page.evaluate("window.scrollY")
        page.mouse.wheel(0, -120)
        page.wait_for_function("parseInt(document.querySelector('#zoomLevel').textContent) > 100")
        assert page.evaluate("window.scrollY") == outer_scroll
        page.mouse.wheel(0, 120)
        page.wait_for_function("document.querySelector('#zoomLevel').textContent === '100%'")

        node = page.locator('.node[data-node="node-1"]')
        node.scroll_into_view_if_needed()
        box = node.bounding_box()
        start = wrap.evaluate("g => [g.scrollLeft, g.scrollTop]")
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x - 40, y - 40, steps=4)
        page.mouse.up()
        assert wrap.evaluate("g => [g.scrollLeft, g.scrollTop]") == [start[0] + 40, start[1] + 40]
        assert page.locator('.node.active').get_attribute('data-node') == 'node-0'
        assert "dragging" not in wrap.get_attribute("class")
        node.click()
        assert page.locator('.node.active').get_attribute('data-node') == 'node-1'
        page.locator('.edge-label tspan').first.click()
        assert page.locator('#edgeDetail').is_visible()
