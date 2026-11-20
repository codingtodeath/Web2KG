"""Local smoke queries and real recorded-action replay, not benchmark task evaluation."""
import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.browser import Browser
from web2kg.common.storage import Graph, write_json
from web2kg.evaluation.manual import ManualSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--storage-state", default=str(ROOT / "auth/dev-reddit.json"))
    parser.add_argument("--queries", type=Path, default=ROOT / "examples/webnavigator_review_queries.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    graph = Graph.load(args.graph)
    session = ManualSession(graph, args.out)
    cases = json.loads(args.queries.read_text(encoding="utf-8"))
    results = []
    with Browser(graph.data["start_url"], session.log, storage_state=args.storage_state,
                 viewport=graph.data.get("viewport"), identity_mode=graph.data.get("identity_mode", "semantic")) as browser:
        session.browser = browser
        for case in cases:
            result = {**case}
            try:
                query_id = session.query({"text": case["query"], "top_k": case["top_k"]})["query_id"]
                row = session.query_row(query_id)
                expected = {key for key, node in graph.nodes.items()
                            if re.search(case["expected_path"], urlsplit(node["url"]).path)}
                hits = [c for c in row["candidates"] if c["id"] in expected]
                result.update(query_id=query_id, covered=len(expected), hit_at_k=bool(hits),
                              matching_ranks=[c["rank"] for c in hits], seconds=row["seconds"],
                              candidates=[{k: c[k] for k in ("id", "rank", "title", "url", "score")} for c in row["candidates"]])
                # Always exercise top-1, including irrelevant candidates; do not hide ranking failures.
                selected = row["candidates"][0]
                result["top1_replay"] = session.visit({"node_id": selected["id"], "source": "topk", "query_id": query_id})
                session.judge({"query_id": query_id,
                    "verdict": "hit" if hits else "retrieval_miss" if expected else "coverage_miss",
                    "expected_node_id": hits[0]["id"] if hits else sorted(expected)[0] if expected else None,
                    "note": "Automated URL-pattern smoke check; not a human semantic judgment or benchmark task score."})
            except Exception as error:
                result["error"] = str(error)
            results.append(result)
            write_json(args.out / "smoke_results.json", {"scope": "synthetic retrieval/replay smoke only", "results": results})
            print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
