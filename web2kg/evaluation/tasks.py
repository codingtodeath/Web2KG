import json
from collections import Counter
from urllib.parse import urlsplit

from ..stage2.agent import run_task
from ..common.storage import EventLog, write_json


def check_task(browser, expected):
    if not expected or not set(expected) <= {"url_contains", "path_equals", "text_contains"}:
        raise ValueError("Local tasks require URL/path/text checks")
    text = browser.page.locator("body").inner_text()
    checks = {"url_contains": lambda value: value in browser.page.url,
              "path_equals": lambda value: urlsplit(browser.page.url).path == value,
              "text_contains": lambda value: value in text}
    return all(checks[key](value) for key, value in expected.items())


def evaluate(graph, tasks, directory, chat_factory, embedding_factory, max_steps=30, storage_state=None):
    if max_steps < 1 or not tasks:
        raise ValueError("Evaluation requires tasks and a positive step budget")
    rows = []
    for i, task in enumerate(tasks):
        if not isinstance(task.get("task"), str) or not task["task"].strip():
            raise ValueError("Each evaluation task needs nonempty task text")
        expected = task.get("expected")
        if (not isinstance(expected, dict) or not expected or
                not set(expected) <= {"url_contains", "path_equals", "text_contains"} or
                any(not isinstance(v, str) or not v for v in expected.values())):
            raise ValueError("Each evaluation task needs nonempty objective URL/text checks")
        # Alternate ordering to reduce a systematic warm-cache advantage.
        modes = ("baseline", "graph") if i % 2 == 0 else ("graph", "baseline")
        for mode in modes:
            log = EventLog(directory / f"task-{i:02d}-{mode}")
            chat, embedding = chat_factory(log), embedding_factory(log)
            result = run_task(graph, task['task'], log, chat, embedding,
                start_url=task.get('url', graph.data['start_url']), storage_state=storage_state,
                max_steps=max_steps, use_graph=mode == 'graph',
                final_check=lambda browser: check_task(browser, expected))
            success = result.get('local_success', False)
            events = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
            usage = Counter()
            for event in events:
                if event["event"] in {"model_call", "responses_received"}:
                    usage.update({k: v for k, v in event.get("usage", {}).items() if isinstance(v, (int, float))})
            rows.append({"task": task["task"], "mode": mode, "local_success": success,
                         "url": result["url"], "expected": expected, "run": str(log.directory),
                         "agent_steps": result["steps"], "navigation_steps": result["navigation"].get("steps", 0),
                         "seconds": result["seconds"], "usage": dict(usage), "status": result["status"]})
            write_json(directory / "evaluation.json", {"tasks": rows, "kind": "local_navigation_checks"})
    summary = {}
    for mode in ("baseline", "graph"):
        subset = [row for row in rows if row["mode"] == mode]
        summary[mode] = {"tasks": len(subset), "successes": sum(row["local_success"] for row in subset),
                         "mean_seconds": sum(row["seconds"] for row in subset) / max(len(subset), 1),
                         "mean_agent_steps": sum(row["agent_steps"] for row in subset) / max(len(subset), 1)}
    result = {"kind": "local_navigation_checks", "summary": summary, "tasks": rows}
    write_json(directory / "evaluation.json", result)
    return result
