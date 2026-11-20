from uuid import uuid4

from playwright.sync_api import Error as BrowserError


def page_ref(observation):
    return {**{key: observation.get(key) for key in ("id", "url", "title")},
            **{key: observation[key] for key in ('raw_state_id', 'equivalence_rule') if key in observation}}


def record_failure(browser, log, event, observation, action, error, phase, action_attempted=False, **data):
    """Keep source/intent distinct from the page actually visible after a failure."""
    details = {"source_page": page_ref(observation), "action": action, "phase": phase,
               "action_attempted": action_attempted, "error": str(error),
               "error_type": type(error).__name__, "error_summary": str(error).splitlines()[0]}
    try:
        observed = browser.observe()
        folder = log.directory / "failures" / uuid4().hex[:12]
        folder.mkdir(parents=True)
        browser.screenshot(folder / "page.png")
        states = ({action["id"]: "attempted"} if action_attempted and action and action.get("id")
                  and observed["id"] == observation["id"] else {})
        browser.save_annotation(observed, folder, states)
        details["failure_page"] = {**page_ref(observed), "actions": observed["actions"],
                                   "snapshot_dir": folder.relative_to(log.directory).as_posix()}
    except (BrowserError, OSError, ValueError) as capture_error:
        details["capture_error"] = str(capture_error)
    log.emit(event, **data, **details)
