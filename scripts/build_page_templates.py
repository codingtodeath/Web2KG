"""Build a separate template layer from saved snapshots; no browser or model calls."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.storage import Graph, digest, write_json
from web2kg.stage1.templates import (build_templates, DEFAULT_THRESHOLD, DEFAULT_WEIGHTS,
                                    PAGE_RULE, LEGACY_RULE)
from web2kg.visualization.template_report import export_template_report
from web2kg.visualization.viewer import export_viewer


def run(directory, threshold=DEFAULT_THRESHOLD, weights=DEFAULT_WEIGHTS, match_rule=PAGE_RULE, publish=False,
        clustering='average', tolerance=.05):
    graph = Graph.load(directory)
    graph_path = graph.directory / "graph.json"
    before = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    result = build_templates(graph, threshold, weights, match_rule)
    audit = None
    if clustering != 'legacy':
        from web2kg.stage1.template_global import regroup, reset_maintenance
        from web2kg.stage1.template_assets import update_summary
        result, audit = regroup(graph, result, method=clustering, tolerance=tolerance)
        reset_maintenance(result)
        update_summary(graph, result)
    suffix = "-" + digest([match_rule, threshold, list(weights), clustering, tolerance])[:8]
    out = graph.directory / "templates" / (result["version"] + suffix)
    out.mkdir(parents=True, exist_ok=True)
    result["graph_sha256"] = before
    write_json(out / "templates.json", result)
    if audit:
        write_json(out / 'clustering.json', audit)
    (out / "decisions.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n"
                                               for d in result["decisions"]), encoding="utf-8")
    export_template_report(graph, result, out)
    if hashlib.sha256(graph_path.read_bytes()).hexdigest() != before:
        raise RuntimeError("Source graph changed during offline analysis")
    if publish:
        write_json(graph.directory / "templates" / "active.json", {"directory": out.relative_to(graph.directory).as_posix()})
        export_viewer(graph)
    print(json.dumps({"output": str(out), **result["summary"], "published": publish,
                      "source_graph_unchanged": True}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Page structure threshold (legacy mode: weighted total threshold)")
    parser.add_argument('--match-rule', choices=[PAGE_RULE, LEGACY_RULE], default=PAGE_RULE)
    parser.add_argument('--publish', action='store_true', help='Update the viewer active layer; old cards remain bound to their original layer')
    parser.add_argument('--clustering', choices=['average', 'complete', 'legacy'], default='average')
    parser.add_argument('--tolerance', type=float, default=.05)
    parser.add_argument("--weights", type=float, nargs=3, default=DEFAULT_WEIGHTS,
                        metavar=("PARENT", "INCOMING", "PAGE"), help="Parent, incoming control, page structure weights")
    args = parser.parse_args()
    run(args.directory, args.threshold, args.weights, args.match_rule, args.publish, args.clustering, args.tolerance)
