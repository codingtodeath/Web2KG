import re
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from playwright.sync_api import sync_playwright

from .diagnostics import page_ref
from .storage import digest, write_json
from .interaction import VERSION as INTERACTION_VERSION, state_id as interaction_state_id
from .interaction import LANDMARKS, STATE_PROPERTIES


INTERACTIVE = {"link", "button", "tab", "textbox", "searchbox", "combobox", "checkbox",
               "radio", "menuitem", "switch", "spinbutton", "slider"}

# Negative-position accessibility links cannot be reached by scrolling. Ordinary
# below-the-fold controls are deliberately NOT classified as unavailable.
POINTER_UNAVAILABLE = """el => {
    if (!el || !el.matches('a[href], [role="link"]')) return false;
    const css = getComputedStyle(el), box = el.getBoundingClientRect();
    if (css.direction === 'rtl' || !['absolute','fixed'].includes(css.position)) return false;
    return box.width > 0 && box.right + (css.position === 'fixed' ? 0 : Math.max(0, scrollX)) <= 0;
}"""

CLICK_LABEL = """el => {
    if (!el.matches('input[type="checkbox"], input[type="radio"]') || el.disabled) return null;
    for (const label of el.labels || []) {
        const css = getComputedStyle(label), box = label.getBoundingClientRect();
        if (!box.width || !box.height || css.visibility !== 'visible' || css.display === 'none') continue;
        label.scrollIntoView({block:'nearest', inline:'nearest'});
        const r = label.getBoundingClientRect();
        const hit = document.elementFromPoint(r.x + r.width/2, r.y + r.height/2);
        const interactive = hit && hit.closest('a,button,input,select,textarea');
        if (label.control === el && hit && label.contains(hit) && (!interactive || interactive === el))
            return label;
    }
    return null;
}"""


def canonical_url(url):
    parts = urlsplit(url)
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", query, parts.fragment))


def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


class Browser:
    def __init__(self, start_url, log, headed=False, storage_state=None, allow_writes=False, settle_ms=250, viewport=None, window_bounds=None, identity_mode="semantic"):
        self.start_url, self.log = start_url, log
        self.headed, self.storage_state = headed, storage_state
        self.allow_writes, self.settle_ms = allow_writes, settle_ms
        self.context = None
        if identity_mode not in {"semantic", "webnavigator", "interaction"}:
            raise ValueError("Unknown page identity mode")
        self.identity_mode = identity_mode
        self.window_bounds = window_bounds
        self.viewport = viewport or {"width": 1280, "height": 900}
        if min(self.viewport.values()) < 1:
            raise ValueError("Viewport dimensions must be positive")

    def __enter__(self):
        self.playwright = sync_playwright().start()
        try:
            self.browser = self.playwright.chromium.launch(headless=not self.headed)
            self.reset()
        except Exception:
            try:
                if hasattr(self, "browser"):
                    self.browser.close()
            finally:
                self.playwright.stop()
            raise
        return self

    def __exit__(self, *args):
        try:
            self.browser.close()
        finally:
            self.playwright.stop()

    def _route(self, route):
        route.continue_()

    def context_options(self):
        return {"viewport": self.viewport, "storage_state": self.storage_state, "service_workers": "block"}

    def reset(self):
        self.last_page = {"id": None, "url": self.start_url, "title": "初始页面加载"}
        self.current_action = None
        if self.context:
            self.context.close()
        self.context = self.browser.new_context(**self.context_options())
        self.context.set_default_timeout(5000)
        self.context.route("**/*", self._route)
        self.context.on("page", lambda page: page.on("dialog", lambda dialog: dialog.dismiss()))
        self.page = self.context.new_page()
        self.document_response = None
        def document_response(response):
            if response.request.resource_type == "document" and response.frame == self.page.main_frame:
                self.document_response = (canonical_url(response.url).split("#", 1)[0], response.status)
        self.page.on("response", document_response)
        self.cdp = self.context.new_cdp_session(self.page)
        self.page.goto(self.start_url, wait_until="domcontentloaded", timeout=20000)
        if self.headed and self.window_bounds:
            window = self.cdp.send("Browser.getWindowForTarget")
            self.cdp.send("Browser.setWindowBounds", {"windowId": window["windowId"], "bounds": self.window_bounds})
        self.settle()
        self.log.emit("browser_reset", url=self.page.url)

    def settle(self):
        self.page.wait_for_load_state("domcontentloaded", timeout=15000)
        self.page.wait_for_timeout(self.settle_ms)

    def observe(self, directory=None):
        started = time.perf_counter()
        ax = self.cdp.send("Accessibility.getFullAXTree")["nodes"]
        dom = self.cdp.send("DOM.getDocument", {"depth": -1})["root"]
        dom_map = {}

        def visit(node, path="", form=None, mutation=False):
            attrs = dict(zip(node.get("attributes", [])[::2], node.get("attributes", [])[1::2]))
            if node["nodeName"] == "FORM":
                form = {"method": attrs.get("method", "get"), "form_action": attrs.get("action", "")}
            mutation = mutation or bool(attrs.get("data-post")) or attrs.get("data-method", "get").lower() != "get"
            dom_map[node["backendNodeId"]] = {"xpath": path, "attrs": attrs, "form": form or {},
                                               "data_mutation": mutation, "tag": node["nodeName"]}
            counts = Counter()
            for child in node.get("children", []):
                tag = child["nodeName"].lower()
                if child["nodeType"] != 1:
                    continue
                counts[tag] += 1
                visit(child, f"{path}/{tag}[{counts[tag]}]", form, mutation)

        visit(dom)
        by_id = {n["nodeId"]: n for n in ax}
        # Index peers once per fresh observation, preserving AX order and identities.
        landmark_groups = defaultdict(list)
        for node in ax:
            role = node.get('role', {}).get('value', '')
            if not node.get('ignored') and role in LANDMARKS:
                landmark_groups[(role, clean(node.get('name', {}).get('value', '')))].append(node)
        landmark_regions = {}
        for (role, name), peers in landmark_groups.items():
            for index, node in enumerate(peers):
                landmark_regions[node['nodeId']] = {'role': role, 'name': name,
                    'occurrence': index, 'match_count': len(peers)}
        actions, signature, headings, regions = [], [], [], []
        for node in ax:
            if node.get("ignored"):
                continue
            role = node.get("role", {}).get("value", "")
            name = clean(node.get("name", {}).get("value", ""))
            state = {p["name"]: p["value"].get("value") for p in node.get("properties", [])
                     if p["name"] in STATE_PROPERTIES}
            info = dom_map.get(node.get("backendDOMNodeId"), {})
            attrs = info.get("attrs", {})
            value = "[redacted]" if attrs.get("type") == "password" else clean(node.get("value", {}).get("value", ""))
            # Exclude focus and backend IDs; retain text and control state, including same-URL tabs.
            if role in INTERACTIVE | LANDMARKS | {"heading", "StaticText", "img"}:
                semantics = {}
                if role in INTERACTIVE:
                    semantics = {"href": urljoin(self.page.url, attrs["href"]) if attrs.get("href") else "",
                                 "type": attrs.get("type", ""), **info.get("form", {})}
                signature.append([role, name, value, state, semantics])
            if role == "heading":
                headings.append(name)
            if role in LANDMARKS | {"alertdialog"}:
                regions.append([role, name, state])
            if role not in INTERACTIVE or not info.get("xpath"):
                continue
            region = {}
            parent = by_id.get(node.get("parentId"))
            while parent:
                parent_role = parent.get("role", {}).get("value")
                if parent_role in LANDMARKS:
                    region = dict(landmark_regions[parent['nodeId']])
                    break
                parent = by_id.get(parent.get("parentId"))
            action = {"kind": "click", "role": role, "name": name, "xpath": info["xpath"], "ax_node_id": node["nodeId"],
                      "href": urljoin(self.page.url, attrs["href"]) if attrs.get("href") else "",
                      "type": attrs.get("type", ""), "region": region, "state": state,
                      "disabled": bool(state.get("disabled")), "data_mutation": info["data_mutation"],
                      **info["form"]}
            action["id"] = digest([action["role"], action["name"], action["xpath"]])
            if role == "link":
                plain = (info.get("tag") == "A" and bool(attrs.get("href"))
                         and not any(k.startswith(("on", "data-")) for k in attrs)
                         and not any(k in attrs for k in ("aria-expanded", "aria-controls", "aria-haspopup",
                                                         "aria-selected", "aria-pressed", "aria-checked", "download"))
                         and attrs.get("target", "").lower() in {"", "_self"}
                         and attrs.get("role", "link") == "link" and not info.get("form"))
                action["navigation_kind"] = "plain_link" if plain else "interactive_or_unknown"
            if self.identity_mode == "interaction" and node.get("value") is not None:
                action["value_digest"] = digest(value)
            actions.append(action)
        unavailable = self.page.evaluate("""paths => {
            const unavailable = """ + POINTER_UNAVAILABLE + """;
            return paths.map(path => unavailable(document.evaluate(path, document, null,
                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue));
        }""", [a['xpath'] for a in actions])
        for action, hidden in zip(actions, unavailable):
            if hidden:
                action['pointer_unavailable'] = 'negative_position_link'
        control_groups = defaultdict(list)
        for number, action in enumerate(actions, 1):
            action["number"] = number
            control_groups[(action['role'], action['name'], tuple(action['region'].items()))].append(action)
        for matches in control_groups.values():
            for index, action in enumerate(matches):
                action['occurrence'], action['match_count'] = index, len(matches)
        url = canonical_url(self.page.url)
        result = {"id": digest([url, signature]), "url": url, "title": self.page.title(),
                  "headings": headings, "text": "\n".join(n[1] for n in signature if n[0] == "StaticText")[:16000],
                  "actions": actions, "regions": regions, "ax": ax, "dom": dom}
        result["http_status"] = (self.document_response[1] if self.document_response
                                 and self.document_response[0] == url.split("#", 1)[0] else None)
        if self.identity_mode == "webnavigator":
            paths = sorted({info["xpath"] for info in dom_map.values() if info["xpath"]})
            hashes = sorted(hashlib.md5(path.encode()).hexdigest() for path in paths)
            result.update(semantic_id=result["id"], structural_paths=paths,
                          id=hashlib.md5(json.dumps([hashes, url], separators=(",", ":")).encode()).hexdigest())
        elif self.identity_mode == "interaction":
            result.update(semantic_id=result["id"], identity_version=INTERACTION_VERSION,
                          regions=regions, id=interaction_state_id(result, regions))
        # Only a live probe can confirm blank/loading-only content. Extraction failure alone cannot.
        from io import BytesIO
        from PIL import Image
        content = self.page.evaluate(files('web2kg.common').joinpath('page_content.js').read_text(encoding='utf-8'))
        pixels = None
        if content in {'inspect_pixels', 'visual_candidate'}:
            pixels = self.page.screenshot(full_page=True, animations='disabled', timeout=10000)
            with Image.open(BytesIO(pixels)) as picture:
                extrema = picture.convert('RGB').getextrema()
            content = 'blank' if content == 'inspect_pixels' and all(hi == lo for lo, hi in extrema) else 'visual_content'
        result['page_content'] = {'kind': content}
        # Structurally invisible visual states need pixel evidence to distinguish their identities.
        from ..stage1.page_admission import observation_profile
        if content not in {'blank', 'loading_only'} and observation_profile(result).get('isolated'):
            if pixels is None:
                pixels = self.page.screenshot(full_page=True, animations='disabled', timeout=10000)
            result['page_content']['visual_sha256'] = hashlib.sha256(pixels).hexdigest()
            result['id'] = (interaction_state_id(result, regions) if self.identity_mode == 'interaction'
                            else digest([result['id'], result['page_content']['visual_sha256']]))
        resolver = getattr(self, 'state_resolver', None)
        if resolver is not None and self.identity_mode == 'interaction':
            result = resolver.resolve(result)  # Cached rules only; observing never submits a model call.
        self._observed_pixels = (result['id'], pixels)
        self.last_page = page_ref(result)
        self.current_action = None
        self.log.emit('runtime_timing', operation='browser_observe', seconds=time.perf_counter()-started,
                      ax_nodes=len(ax), controls=len(actions))
        if directory:
            self.save_observation(result, directory)
        return result

    def save_observation(self, observation, directory):
        started = time.perf_counter()
        folder = Path(directory) / "snapshots" / observation["id"]
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "ax.json", observation["ax"], compact=True)
        write_json(folder / "dom.json", observation["dom"], compact=True)
        write_json(folder / "observation.json", {k: v for k, v in observation.items() if k not in {"ax", "dom"}}, compact=True)
        saved_id, pixels = getattr(self, '_observed_pixels', (None, None))
        self.screenshot(folder / "page.png", image=pixels if saved_id == observation['id'] else None)
        self.save_annotation(observation, folder)
        self.log.emit("page_observed", node=observation["id"], source_page=page_ref(observation),
                      candidates=observation["actions"], snapshot_dir=folder.relative_to(Path(directory)).as_posix(),
                      screenshot_mode="full_page", viewport=self.viewport)
        self.log.emit('runtime_timing', operation='snapshot_save', node=observation['id'],
                      seconds=time.perf_counter()-started)

    def screenshot(self, path, *, image=None):
        # Replacing a name breaks inherited hardlinks; writing the old inode would alter history.
        path = Path(path)
        if image is None:
            image = self.page.screenshot(full_page=True, animations="disabled", timeout=10000)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(image)
        temporary.replace(path)
        return image

    def save_annotation(self, observation, folder, states=None):
        """Capture a temporary DOM overlay; never write to the original page.png."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        script = files("web2kg.common").joinpath("annotate.js").read_text(encoding="utf-8")
        actions = [{**action, "annotation_state": (states or {}).get(action["id"], "candidate")}
                   for action in observation["actions"]]
        overlay = self.page.evaluate_handle(script, actions)
        try:
            metadata = overlay.evaluate("r => ({items:r.items, width:r.width, height:r.height})")
            self.screenshot(folder / "page-annotated.png")
        finally:
            overlay.evaluate("r => r.host.remove()")
            overlay.dispose()
        metadata.update(state_id=observation["id"], captured_at=datetime.now(timezone.utc).isoformat())
        write_json(folder / "annotations.json", metadata)
        return metadata

    def act(self, action, strategy="role"):
        self.current_action = action
        if action['kind'] == 'go_back':
            self.page.go_back(wait_until='domcontentloaded', timeout=20000)
            self.settle()
            return
        if strategy == "xpath":
            locator = self.page.locator("xpath=" + action["xpath"])
            if locator.count() != 1:
                raise ValueError("XPath locator is missing or ambiguous")
        else:
            # Use the same nearest-landmark partition as observation. A DOM
            # descendant query also includes nested landmarks and changes nth().
            current = self.observe()
            region = action.get('region', {})
            peers = [a for a in current['actions'] if a['role'] == action['role']
                     and a['name'] == action['name'] and a.get('region', {}) == region]
            if len(peers) != action.get("match_count", 1):
                raise ValueError("Role locator count changed")
            occurrence = action.get('occurrence', 0)
            if not 0 <= occurrence < len(peers):
                raise ValueError('Role locator occurrence changed')
            locator = self.page.locator('xpath=' + peers[occurrence]['xpath'])
            if locator.count() != 1:
                raise ValueError('Observed role target is missing or ambiguous')
        kind = action["kind"]
        if kind == "click":
            if locator.evaluate(POINTER_UNAVAILABLE):
                raise ValueError('Pointer target is a negative-position link; keyboard activation is required')
            label_handle = locator.evaluate_handle(CLICK_LABEL)
            try:
                label = label_handle.as_element()
                # A genuine associated label activates the original input once.
                # Never force a click through an unrelated overlay or retry after dispatch.
                # Keep Playwright's navigation barrier: no_wait_after can observe
                # the old document before the new navigation has even committed.
                navigation = action.get('navigation_kind') == 'plain_link'
                (label or locator).click(**({'timeout': 15000} if navigation else {}))
                self.log.emit('browser_click_dispatched', action_id=action.get('id'),
                              target='associated_label' if label else 'control')
            finally:
                label_handle.dispose()
        elif kind == "fill":
            locator.fill(action["value"])
        elif kind == "press":
            locator.press(action["value"])
        elif kind == "select":
            locator.select_option(label=action["value"])
        else:
            raise ValueError(f"Unsupported action: {kind}")
        if len(self.context.pages) > 1:
            for popup in self.context.pages:
                if popup != self.page:
                    popup.close()
            raise ValueError("New-tab actions are not supported by the prototype")
        self.settle()
