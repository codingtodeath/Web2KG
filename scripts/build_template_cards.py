"""Summarize each saved template once using Luna; source graph/index remain unchanged."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.responses import ResponsesClient
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.stage1.function_cards import build_cards, load_layer
from web2kg.visualization.template_report import export_template_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=Path)
    parser.add_argument("--env", default="auth/llm-crawl.env")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--max-calls", type=int, default=30)
    parser.add_argument('--seed', type=Path, help='Carry validated cards from this graph or its unchanged execution source')
    parser.add_argument('--only', nargs='+', help='Only generate these missing cards; use a NEW output to retry a failed card')
    parser.add_argument('--publish', action='store_true', help='Update template viewer and active card pointer after completion')
    args = parser.parse_args()
    graph, log = Graph.load(args.graph.resolve()), EventLog(args.out.resolve())
    if args.publish and not args.out.resolve().is_relative_to(graph.directory):
        parser.error('Published cards must be inside the graph directory')
    client = ResponsesClient(args.env, log, args.max_calls, 2048, model=args.model)
    data = build_cards(graph, args.layer, args.out, client, seed=args.seed, only=args.only)
    if args.publish:
        cards = (args.out / 'cards.json').resolve()
        path, layer = load_layer(graph, args.layer)
        export_template_report(graph, layer, path.parent, cards)
        write_json(graph.directory / 'function-cards/active.json', {'cards': cards.relative_to(graph.directory).as_posix()})
    print(f"Cards complete: {sum(r['status'] == 'completed' for r in data['cards'].values())}; {args.out / 'cards.html'}")


if __name__ == "__main__":
    main()
