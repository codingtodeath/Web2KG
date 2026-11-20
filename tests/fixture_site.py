"""Small deterministic site: same-URL tabs, repeated links, search and forbidden mutations."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


class FixtureHandler(BaseHTTPRequestHandler):
    writes = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        type(self).writes += 1
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"mutation")

    def do_GET(self):
        path = urlsplit(self.path).path
        common = '<nav aria-label="Primary"><a href="/">Home</a><a href="/catalog">Catalog</a><a href="/help">Help</a></nav>'
        pages = {
            "/": '<h1>Test shop</h1><p>Explore our catalog and help pages.</p><a href="/delete">Delete account</a>'
                 '<a href="http://example.org">External</a><form action="/change" method="post"><button>Continue</button></form>',
            "/catalog": '<h1>Catalog</h1><ul><li><a href="/item/1">Blue mug</a></li>'
                        '<li><a href="/item/2">Red mug</a></li><li><a href="/item/3">Green mug</a></li></ul>'
                        '<form role="search" action="/search" method="get"><label>Search products<input type="search" name="q"></label>'
                        '<button>Search</button></form>',
            "/help": '<h1>Help center</h1><div role="tablist"><button role="tab" aria-selected="true" '
                     'onclick="showTab(this,\'Shipping\',\'Delivery takes three days.\')">Shipping</button>'
                     '<button role="tab" aria-selected="false" onclick="showTab(this,\'Returns\',\'Return within 30 days.\')">Returns</button></div>'
                     '<section role="tabpanel"><h2 id="tab-title">Shipping</h2><p id="tab-text">Delivery takes three days.</p></section>'
                     '<script>function showTab(b,t,s){document.querySelectorAll("[role=tab]").forEach(x=>x.setAttribute("aria-selected",String(x===b)));'
                     'document.getElementById("tab-title").textContent=t;document.getElementById("tab-text").textContent=s}</script>',
            "/item/1": '<h1>Blue mug</h1><p>A blue ceramic mug.</p><button onclick="fetch(\'/change\',{method:\'POST\'})">Buy now</button>',
            "/item/2": '<h1>Red mug</h1><p>A red ceramic mug.</p>',
            "/item/3": '<h1>Green mug</h1><p>A green ceramic mug.</p>',
        }
        if path == "/delete":
            type(self).writes += 1
        if path == "/search":
            query = parse_qs(urlsplit(self.path).query).get("q", [""])[0]
            import html
            body = '<h1>Search results</h1><p>' + html.escape(query) + '</p><a href="/item/1">Blue mug</a>'
        else:
            body = pages.get(path, '<h1>Unknown page</h1>')
        html = '<!doctype html><html><head><title>' + {"/": "Test shop", "/help": "Help center", "/catalog": "Catalog"}.get(path, "Product") + '</title></head><body>' + common + '<main>' + body + '</main></body></html>'
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())


if __name__ == "__main__":
    print("Fixture: http://127.0.0.1:8766", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8766), FixtureHandler).serve_forever()
