"""Export recorded decisions and observations without exposing request files or HAR."""
import hashlib
import html
import json
import os
import re
import shutil
from importlib.resources import files
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_asset(path, output, sources):
    if not path.is_file():
        return None
    checksum = digest(path)
    target = output / "images" / (checksum + ".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(path, target)
    sources[str(path.resolve())] = checksum
    return "images/" + target.name


def visible_output(response):
    return [{"phase": m.get("phase"), "status": m.get("status"),
             "text": "".join(p.get("text", "") for p in m.get("content", [])
                             if p.get("type") == "output_text")}
            for m in (response or {}).get("messages", [])]


def request_view(folder, event):
    # Select only visible observation/context fields; never embed the request itself.
    key = event["request_id"]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", key):
        raise ValueError("Invalid request ID")
    directory = folder / "llm" / key
    request = read(directory / "request.json", {})
    texts = [p.get("text", "") for m in request.get("input", [])
             for p in m.get("content", []) if p.get("type") == "input_text"]
    current = next((t for t in texts if t.startswith("CURRENT STEP:")), "")
    match = re.match(r"CURRENT STEP:\s*(\d+)", current)
    ax = None
    if "CURRENT OBSERVATION:\n" in current:
        ax = current.split("CURRENT OBSERVATION:\n", 1)[1].split("CURRENT VISUAL OBSERVATION:", 1)[0].rstrip()
    error = read(directory / "http_error.json", {})
    return {"id": key, "purpose": event.get("purpose"), "call": event.get("call"),
            "actor_step": int(match[1]) if match else None, "ax": ax,
            "context": [t for t in texts if t.startswith(("Historical template context", "Website framework and conditional experience"))],
            "output": visible_output(read(directory / "response.json", {})),
            "http_status": error.get("http_status"), "error_code": error.get("code")}


def same_pixels(screenshot, snapshot):
    """A full snapshot may extend below the upstream screenshot's height cap."""
    from PIL import Image
    with Image.open(screenshot) as a, Image.open(snapshot) as b:
        if a.width != b.width or a.height > b.height:
            return False
        return a.convert("RGB").tobytes() == b.crop((0, 0, a.width, a.height)).convert("RGB").tobytes()


def collect_task(folder, row, output, sources):
    trajectory = read(folder / "trajectory.json", [])
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    actions = [e for e in events if e["event"] == "agent_action"]
    if len(actions) != len(trajectory) or any(a["action"] != t["action"] or a["step"] != t["num_actions"]
                                            for a, t in zip(actions, trajectory)):
        raise ValueError(f"Cannot align recorded actions and trajectory: {folder}")
    snapshots = []
    for path in sorted(folder.glob("**/page-annotated.png")):
        obs = read(path.parent / "observation.json", {})
        snapshots.append({"path": path.parent, "url": obs.get("url"), "title": obs.get("title"),
                          "source": path.parent.relative_to(folder).as_posix(),
                          "image": image_asset(path, output, sources)})
    steps, pending, calls = [], [], []
    for event in events:
        pending.append(event)
        if event["event"] == "responses_started":
            calls.append(request_view(folder, event))
        if event["event"] != "agent_action":
            continue
        t = trajectory[len(steps)]
        number = event["step"]
        if any(c["actor_step"] != number - 1 for c in calls if c["purpose"] == "original_actor"):
            raise ValueError(f"Actor step does not match executed step: {folder}, {number}")
        before = folder / f"step-{number - 1}.png"
        recovery_before = folder / f"step-{number}-before-recovery.png"
        after = recovery_before if recovery_before.exists() else folder / f"step-{number}.png"
        ax_path = folder / f"step-{number - 1}.txt"
        som = next((s for s in snapshots if s["url"] == t["url"] and before.is_file()
                    and (s["path"] / "page.png").is_file()
                    and same_pixels(before, s["path"] / "page.png")), None)
        notes = {k: t[k] for k in ("plan", "extract_constraints", "interaction_history_summary",
                 "observation_description", "mode_analysis", "reason", "note", "navigate_think",
                 "navigate_query", "target_page", "page_selection_response", "topk_webpages") if t.get(k)}
        trace = [{k: e[k] for k in ("event", "step", "edge", "source", "target", "success", "reason",
                                   "steps", "retry_number", "http_status", "failure_kind", "request_id",
                                   "root_step", "decision", "triggers", "action") if k in e}
                 for e in pending if e["event"] in ("navigation", "bridge_transition", "replay_step",
                 "navigation_home", "responses_retry", "responses_commentary_fallback", "responses_failed",
                 "recovery_diagnosis", "counterfactual_return", "counterfactual_control_blocked")]
        steps.append({"number": number, "action": t["action"], "before_url": t["url"],
                      "after_url": event["url"], "notes": notes, "calls": calls, "events": trace,
                      "before": image_asset(before, output, sources), "after": image_asset(after, output, sources),
                      "after_file": after.name,
                      "raw_ax": ax_path.read_text(encoding="utf-8") if ax_path.is_file() else None,
                      "trajectory_ax": t.get("observation"),
                      "som": {k: som[k] for k in ("image", "source")} if som else None})
        pending, calls = [], []
    result = read(folder / "result.json", {})
    return {"id": row["task_id"], "score": row["official_score"], "source": folder.parent.name,
            "folder": f"runs/{folder.parent.name}/{folder.name}", "objective": trajectory[0].get("objective", "") if trajectory else "",
            "steps": steps, "answer": result.get("answer"), "unassigned_calls": calls,
            "snapshots": [{k: s[k] for k in ("source", "title", "url", "image")} for s in snapshots]}


def export_report(batches, output):
    """Batches are oldest to newest; select the last started run, never best score."""
    latest, history = {}, {}
    for batch in batches:
        for row in read(batch / "summary.json")["tasks"]:
            history.setdefault(row["task_id"], []).append({"source": batch.name, "score": row["official_score"], "status": row["status"]})
            if row["status"] != "not_started":
                latest[row["task_id"]] = (batch / str(row["task_id"]), row)
    output.mkdir(parents=True, exist_ok=True)
    sources, tasks = {}, []
    for folder, row in latest.values():
        task = collect_task(folder, row, output, sources)
        task["history"] = history[row["task_id"]]
        tasks.append(task)
    menu = [{k: t[k] for k in ("id", "score", "source")} for t in tasks]
    template = files("web2kg.visualization").joinpath("trajectories.html").read_text(encoding="utf-8")
    results = Path(__file__).resolve().parents[2] / 'docs/results.html'
    href = (Path(os.path.relpath(results, output.resolve())).as_posix()
            if results.anchor == output.resolve().anchor else results.as_uri())
    template = template.replace('__RESULTS_HREF__', html.escape(href, quote=True))
    for task in tasks:
        payload = json.dumps({"menu": menu, "task": task}, ensure_ascii=False).replace("<", "\\u003c")
        (output / f"task-{task['id']}.html").write_text(template.replace("__PAYLOAD__", payload), encoding="utf-8")
    payload = json.dumps({"menu": menu, "task": None}, ensure_ascii=False).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__PAYLOAD__", payload), encoding="utf-8")
    manifest = {"selection": "last_started_run_in_batch_order", "batches": [str(b.resolve()) for b in batches],
                "tasks": [{"task_id": t["id"], "source": t["source"], "steps": len(t["steps"]),
                           "matched_som_steps": sum(s["som"] is not None for s in t["steps"]),
                           "snapshot_count": len(t["snapshots"])} for t in tasks],
                "images": sources}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
