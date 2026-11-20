import json

import pytest
from PIL import Image

from web2kg.visualization.trajectory_report import collect_task, export_report


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def batch(tmp_path, name, score=1):
    root = tmp_path / name
    row = {"task_id": 1, "official_score": score, "status": "completed"}
    save(root / "summary.json", {"tasks": [row]})
    folder = root / "1"
    save(folder / "trajectory.json", [{"num_actions": 1, "action": "click [7]", "url": "http://site/",
         "objective": "</script><script>bad()</script>", "reason": "Recorded reason", "observation": "optimized tree",
         "note": "Persistent fact <script>bad()</script>"}])
    save(folder / "result.json", {"answer": "done"})
    save(folder / "llm/a/request.json", {"instructions": "PRIVATE_SYSTEM", "input": [
        {"content": [{"type": "input_text", "text": "CURRENT STEP:\n0\nCURRENT OBSERVATION:\nbutton [7]\nCURRENT VISUAL OBSERVATION:\n"}]}]})
    save(folder / "llm/a/response.json", {"messages": [{"phase": "final_answer", "status": "completed",
        "content": [{"type": "output_text", "text": "ACTION: click [7]"}]}], "private_field": "NOT_EXPORTED"})
    events = [{"event": "responses_started", "request_id": "a", "purpose": "original_actor", "call": 1},
              {"event": "agent_action", "step": 1, "action": "click [7]", "url": "http://site/next"}]
    (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    (folder / "step-0.txt").write_text("full tree\nbutton [7]\nextra [8]", encoding="utf-8")
    for n in range(2):
        Image.new("RGB", (20, 20), "white").save(folder / f"step-{n}.png")
    return root, row


def test_latest_started_is_not_best_score_and_raw_inputs_not_exposed(tmp_path):
    a, _ = batch(tmp_path, "first", 1)
    b, _ = batch(tmp_path, "latest", 0)
    output = tmp_path / "report"
    manifest = export_report([a, b], output)
    assert manifest["tasks"][0]["source"] == "latest"
    page = (output / "task-1.html").read_text(encoding="utf-8")
    payload = json.loads(page.split('<script id="payload" type="application/json">')[1].split('</script>')[0])
    t = payload["task"]
    assert t["score"] == 0 and len(t["history"]) == 2
    assert t["steps"][0]["calls"][0]["ax"] == "button [7]"
    assert "extra [8]" in t["steps"][0]["raw_ax"]
    assert "PRIVATE_SYSTEM" not in page and "NOT_EXPORTED" not in page
    assert "</script><script>bad()" not in page
    assert t["steps"][0]["som"] is None
    assert t['steps'][0]['notes']['note'] == 'Persistent fact <script>bad()</script>'


def test_refuse_misaligned_action(tmp_path):
    root, row = batch(tmp_path, "run")
    file = root / "1/trajectory.json"
    d = json.loads(file.read_text()); d[0]["action"] = "click [9]"; save(file, d)
    with pytest.raises(ValueError, match="Cannot align"):
        collect_task(root / "1", row, tmp_path / "out", {})


def test_action_after_image_does_not_silently_show_recovery_destination(tmp_path):
    root, row = batch(tmp_path, 'run')
    folder = root / '1'
    Image.new('RGB', (20, 20), 'red').save(folder / 'step-1-before-recovery.png')
    task = collect_task(folder, row, tmp_path / 'out', {})
    assert task['steps'][0]['after_file'] == 'step-1-before-recovery.png'
    with Image.open(tmp_path / 'out' / task['steps'][0]['after']) as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)


def test_unstarted_later_batch_does_not_replace_existing_trajectory(tmp_path):
    first, _ = batch(tmp_path, "first")
    later = tmp_path / "unstarted"
    save(later / "summary.json", {"tasks": [{"task_id": 1, "status": "not_started", "official_score": None}]})
    result = export_report([first, later], tmp_path / "out")
    assert result["tasks"][0]["source"] == "first"


def test_refuse_request_from_different_step(tmp_path):
    root, row = batch(tmp_path, "run")
    file = root / "1/llm/a/request.json"
    file.write_text(file.read_text().replace('STEP:\\n0', 'STEP:\\n5'), encoding="utf-8")
    with pytest.raises(ValueError, match="Actor step"):
        collect_task(root / "1", row, tmp_path / "out", {})


@pytest.mark.parametrize("matching", [True, False])
def test_som_requires_pixels_not_only_url(tmp_path, matching):
    root, row = batch(tmp_path, "run")
    folder = root / "1/snapshots/state"
    save(folder / "observation.json", {"url": "http://site/", "title": "Page"})
    Image.new("RGB", (20, 30), "white" if matching else "black").save(folder / "page.png")
    Image.new("RGB", (20, 30), "red").save(folder / "page-annotated.png")
    task = collect_task(root / "1", row, tmp_path / "out", {})
    assert (task["steps"][0]["som"] is not None) == matching
    assert len(task["snapshots"]) == 1
