"""Summarize all template cards and compact transitions into the website layer."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.storage import EventLog, Graph
from web2kg.common.responses import ResponsesClient
from web2kg.stage1.site_framework import build_framework


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cards', type=Path)
    parser.add_argument('--env', default='auth/llm-crawl.env')
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    graph = Graph.load(args.graph.resolve())
    cards = args.cards or graph.directory / json.loads((graph.directory / 'function-cards/active.json').read_text(encoding='utf-8'))['cards']
    # Logs are separate so the target directory's no-overwrite check remains effective.
    client = ResponsesClient(args.env, EventLog(args.out.parent / (args.out.name + '-calls')), 1, 4096, model=args.model)
    result = build_framework(graph, cards, client, args.out, args.publish)
    print(json.dumps({'status': result['status'], 'file': str(args.out / 'framework.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
