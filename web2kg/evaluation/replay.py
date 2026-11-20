from playwright.sync_api import Error as BrowserError

from ..common.replay import execute_path, verify
from ..common.storage import write_json


def replay_experiment(browser, graph, log, count=10, repeats=3, max_depth=3):
    if min(count, repeats, max_depth) < 1:
        raise ValueError("Experiment parameters must be positive")
    paths = [(node, graph.path(graph.data["root"], node)) for node in graph.nodes if node != graph.data["root"]]
    paths = [(node, path) for node, path in paths if path and len(path) <= max_depth][:count]
    trials = []
    for target, path in paths:
        for strategy in ("xpath", "role"):
            for repeat in range(repeats):
                try:
                    browser.reset()
                    verify(browser, graph.data["root"])
                    result = execute_path(browser, graph, path, log, strategy)
                except (BrowserError, ValueError) as error:
                    result = {"success": False, "failed_step": -1, "steps": 0, "error": str(error)}
                trial = {"target": target, "strategy": strategy, "repeat": repeat, "length": len(path), **result}
                trials.append(trial)
                log.emit("replay_trial", **trial)
    summary = {strategy: {"trials": sum(t["strategy"] == strategy for t in trials),
                          "successes": sum(t["strategy"] == strategy and t["success"] for t in trials)}
               for strategy in ("xpath", "role")}
    result = {"paths": len(paths), "requested_paths": count, "repeats": repeats, "summary": summary, "trials": trials}
    write_json(log.directory / "replay_experiment.json", result)
    return result
