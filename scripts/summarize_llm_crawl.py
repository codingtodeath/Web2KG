"""Offline accounting and consistency checks; never calls a model or browser."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.activity import read_events
from web2kg.common.storage import Graph, write_json


def summarize(directory):
    graph = Graph.load(directory)
    events = read_events(graph.directory / "events.jsonl")
    separate_calls = graph.directory.parent / 'construction-calls/events.jsonl'
    if separate_calls.exists():
        events += read_events(separate_calls)
        events.sort(key=lambda e: e['time'])
    parallel = bool(graph.data.get('llm_explorer', {}).get('pipeline'))
    records = graph.data.get("transition_records", [])
    received = [e for e in events if e["event"] == "responses_received"]
    starts = [e for e in events if e["event"] == "responses_started"]
    attempts = [e for e in events if e["event"] == "crawl_attempt_started"]
    seen = Counter((e["source"], e["action"]["id"]) for e in attempts)
    actual = defaultdict(list)
    for event in attempts:
        actual[event["source"]].append(event["action"]["id"])
    plans = graph.data.get("rankings", {})
    pruned = graph.data.get("pruned_links", [])
    pruned_pairs = {(r["source"], r["action"]["id"]) for r in pruned}
    checks = {
        "within_node_cap": len(graph.nodes) <= graph.data["budget"]["max_states"],
        "within_attempt_cap": len(attempts) <= graph.data["budget"]["max_attempts"],
        "within_request_cap": (graph.data.get('pipeline_summary', {}).get('reserved_jobs', 0) if parallel
            else len(starts) - sum(e['event'] == 'responses_retry' and e.get('persistent', False) for e in events))
            <= (graph.data['llm_explorer']['max_calls'] or float('inf')),
        "no_duplicate_state_action": max(seen.values(), default=0) <= 1,
        "every_attempt_has_record": {e["attempt"] for e in attempts} == {r["attempt"] for r in records},
        "no_pruned_action_attempted": not pruned_pairs.intersection(seen),
        "executed_in_selected_order": all(ids == [r["id"] for r in plans[node]["selected"]
                                                   if (node, r["id"]) not in pruned_pairs][:len(ids)]
                                         for node, ids in actual.items()),
        "all_edges_have_descriptions": all(e.get('action', {}).get('kind') == 'url_jump' or
            e.get('description', {}).get('annotation', {}).get('status') == 'completed' or
            (parallel and e.get('replay_policy') == 'observed-known-state-v1') for e in graph.data['edges']),
        "all_node_screenshots_exist": all((graph.directory / "snapshots" / key / "page.png").is_file()
                                          for key in graph.nodes),
    }
    finished = next((e for e in reversed(events) if e["event"] == "crawl_finished"), None)
    statuses = Counter(r["annotation"]["status"] for r in records)
    source_urls = {n["url"] for n in graph.nodes.values()}
    result = {
        "scope": "engineering crawl, not task success/generalization evaluation",
        "directory": str(graph.directory.resolve()), "complete": finished is not None,
        "stop_reason": finished["reason"] if finished else None,
        "states": len(graph.nodes), "unique_urls": len(source_urls), "edges": len(graph.data["edges"]),
        "attempts": len(attempts), "outcomes": dict(Counter(r["outcome"] for r in records)),
        "pruned_links": len(pruned), "pruning_phases": dict(Counter(r["phase"] for r in pruned)),
        "ranked_pages": len(plans), "context_modes": dict(Counter(p["context_mode"] for p in plans.values())),
        "provided_candidates": sum(p["provided_candidates"] for p in plans.values()),
        "context_omitted_candidates": sum(p["omitted_safe_candidates"] for p in plans.values()),
        "repeated_group_candidates": sum(len(p.get("repeated_group_candidates", [])) for p in plans.values()),
        "annotation_status": dict(statuses), "depths": dict(Counter(n["depth"] for n in graph.nodes.values())),
        "requests_started": len(starts), "responses_received": len(received),
        "actual_models": dict(Counter(e["actual_model"] for e in received)),
        "input_tokens": sum(e["usage"].get("input_tokens", 0) for e in received),
        "output_tokens": sum(e["usage"].get("output_tokens", 0) for e in received),
        "request_seconds": round(sum(e["seconds"] for e in received), 3),
        "usage_unknown_requests": len({e["request_id"] for e in starts} - {e["request_id"] for e in received}),
        "checks": checks,
    }
    if finished:
        first = next(e for e in events if e["event"] == "crawl_started")
        result["wall_seconds"] = round((datetime.fromisoformat(finished["time"]) - datetime.fromisoformat(first["time"])).total_seconds(), 3)
    write_json(graph.directory / "crawl_summary.json", result)
    return result


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), ensure_ascii=False, indent=2))
