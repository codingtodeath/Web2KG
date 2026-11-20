import json
import copy
import shutil
from importlib.resources import files

from ..common.storage import digest
from ..common.inspection import graph_inspection
from ..common.activity import candidate_activity, read_events


def export_viewer(graph, log_path=None):
    path = log_path or graph.directory / "events.jsonl"
    events = read_events(path)
    activity_events = events if path.resolve() == (graph.directory / "events.jsonl").resolve() else (
        read_events(graph.directory / "events.jsonl") + events)
    activity = candidate_activity(graph, activity_events)
    pages = copy.deepcopy(graph.nodes)

    def decorate(page, directory, relative):
        root = directory.resolve()
        folder = (root / relative).resolve()
        if not folder.is_relative_to(root):
            return
        view = {"original": None, "annotated": None, "annotations": None}
        for key, filename in (("original", "page.png"), ("annotated", "page-annotated.png")):
            source = folder / filename
            if not source.is_file():
                continue
            if root == graph.directory.resolve():
                view[key] = source.relative_to(root).as_posix()
            else:
                destination = graph.directory / "viewer-assets" / digest(str(root)) / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                view[key] = destination.relative_to(graph.directory).as_posix()
        metadata = folder / "annotations.json"
        if metadata.is_file():
            view["annotations"] = json.loads(metadata.read_text(encoding="utf-8"))
        page["view"] = view
        for number, action in enumerate(page.get("actions", []), 1):
            action.setdefault("number", number)

    for node in pages.values():
        decorate(node, graph.directory, f"snapshots/{node['id']}")
    if path.parent.resolve() != graph.directory.resolve():
        for observation_path in (path.parent / "snapshots").glob("*/observation.json"):
            observed = json.loads(observation_path.read_text(encoding="utf-8"))
            decorate(observed, path.parent, observation_path.parent.relative_to(path.parent))
            pages[observed["id"]] = observed
    edges = {edge["id"]: edge for edge in graph.data["edges"]}
    for event in events:
        edge = edges.get(event.get("edge"), {})
        source = event.get("source") or event.get("node") or edge.get("source")
        source = event.get("source_page", {}).get("id") or source
        node = pages.get(source)
        if node:
            event.setdefault("source_page", {k: node[k] for k in ("id", "title", "url")})
        action = event.get("action") or edge.get("action")
        if node and isinstance(action, str):
            action = next((a for a in node["actions"] if a["id"] == action), {"id": action})
        if isinstance(action, dict):
            action = dict(action)
            if node:
                match = next((a for a in node["actions"] if a["id"] == action.get("id")), {})
                if match.get("number"):
                    action.setdefault("number", match["number"])
            event["action"] = action
        if event.get("failure_page"):
            decorate(event["failure_page"], path.parent, event["failure_page"]["snapshot_dir"])
    template_report = None
    active_path = graph.directory / "templates" / "active.json"
    relative = (json.loads(active_path.read_text(encoding="utf-8"))["directory"] if active_path.is_file()
                else "templates/context-structure-v1")
    template_dir = (graph.directory / relative).resolve()
    if (template_dir.is_relative_to((graph.directory / "templates").resolve())
            and (template_dir / "templates.json").is_file() and (template_dir / "report.html").is_file()):
        layer = json.loads((template_dir / "templates.json").read_text(encoding="utf-8"))
        if layer.get("graph_digest") == digest(graph.data):
            template_report = (template_dir / "report.html").relative_to(graph.directory.resolve()).as_posix()
    framework_report = None
    pointer = graph.directory / 'site-framework/active.json'
    if pointer.exists():
        from ..stage1.site_framework import load_framework
        try:
            load_framework(graph)
            report = (graph.directory / json.loads(pointer.read_text(encoding='utf-8'))['file']).with_name('report.html')
            if report.exists():
                framework_report = report.relative_to(graph.directory).as_posix()
        except ValueError:
            pass  # Stale summaries are not advertised as current website knowledge.
    payload = json.dumps({"graph": graph.data, "pages": pages, "events": events, "activity": activity,
                          "inspection": graph_inspection(graph), "template_report": template_report,
                          "framework_report": framework_report},
                         ensure_ascii=False).replace("<", "\\u003c")
    html = files("web2kg.visualization").joinpath("viewer.html").read_text(encoding="utf-8").replace("__PAYLOAD__", payload)
    destination = graph.directory / "viewer.html"
    destination.write_text(html, encoding="utf-8")
    return destination
