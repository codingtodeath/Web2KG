from playwright.sync_api import Error as BrowserError

from ..common.diagnostics import page_ref
from ..common.replay import restore, verify
from ..common.activity import annotation_states, candidate_activity, read_events


def annotate_graph(browser, graph, log):
    """Backfill developer overlays by verified replay, preserving historical screenshots."""
    completed, failed = 0, 0
    activity = candidate_activity(graph, read_events(graph.directory / "events.jsonl"))
    for node in graph.nodes.values():
        phase = "restore_source"
        try:
            restore(browser, graph, node["id"], log)
            observed = verify(browser, node["id"])
            phase = "annotation"
            if [a["id"] for a in observed["actions"]] != [a["id"] for a in node["actions"]]:
                raise ValueError("Candidate layout changed; cannot align historical element numbers")
            metadata = browser.save_annotation(observed, graph.directory / "snapshots" / node["id"],
                                               annotation_states(activity.get(node["id"], {})))
            log.emit("annotation_saved", node=node["id"], source_page=page_ref(node),
                     marked=sum(item["status"] == "marked" for item in metadata["items"]), total=len(metadata["items"]))
            completed += 1
        except (BrowserError, OSError, ValueError) as error:
            log.emit("annotation_failed", node=node["id"], source_page=page_ref(node), phase=phase, error=str(error))
            failed += 1
    return {"annotated": completed, "failed": failed}
