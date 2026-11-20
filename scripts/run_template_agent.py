"""Jina top-k -> Terra selection -> recorded path -> live template-aware Agent."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.jina import JinaClient
from web2kg.common.jina_runtime import LocalEncoder
from web2kg.common.responses import ResponsesClient
from web2kg.evaluation.webnavigator import OriginalResponses
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.stage2.agent import run_task
from web2kg.stage2.template_context import TemplateKnowledge
from web2kg.stage1.site_framework import load_framework
from web2kg.stage2.recovery import ExperienceStore, memory_namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("task")
    parser.add_argument("--cards", type=Path, help='Defaults to the graph active card package')
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env", default="auth/llm-crawl.env")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--storage-state", default="auth/dev-reddit.json")
    parser.add_argument("--encoder-python", default="D:/Anaconda/envs/web2kg-jina-gpu/python.exe")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-calls", type=int, default=31,
                        help="Actor action decisions; AX searches and page selection are accounted separately")
    parser.add_argument('--max-output-retries', type=int, choices=range(4), default=3,
                        help='Regenerate invalid Actor output on the same observation, without a browser step')
    parser.add_argument('--no-ax-search', action='store_true', help='Use the legacy fixed AX prefix')
    parser.add_argument("--url", help="Actual task start; defaults to graph start")
    parser.add_argument("--home-url", help="Same-session fallback home; defaults to task start")
    parser.add_argument("--allow-writes", action="store_true")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--without-cards", action="store_true", help="Same execution pipeline, omit template context")
    parser.add_argument("--no-graph-update", action="store_true", help="Keep the graph fixed during execution")
    parser.add_argument('--no-finalize-assets', action='store_true', help='Skip post-task Jina extension and card refresh')
    parser.add_argument('--max-graph-nodes', type=int, default=500, help='Post-task leaf capacity; 0 disables eviction')
    parser.add_argument('--retention-half-life', type=float, default=20, help='Idle score half-life in retrieval rounds')
    parser.add_argument("--transition-calls", type=int, default=None, help="Legacy optional cap; default unlimited, 0 disables descriptions")
    parser.add_argument('--card-calls', type=int, default=None, help='Legacy optional cap; default unlimited, 0 disables generation')
    parser.add_argument('--framework', type=Path, help='Defaults to the graph active website framework')
    parser.add_argument('--framework-refresh-updates', type=int, default=10,
                        help='Refresh website summary after this many template changes; 0 disables generation')
    parser.add_argument('--template-clustering', choices=['average', 'complete', 'legacy', 'off'], default='average')
    parser.add_argument('--template-tolerance', type=float, default=.05, help='Allowed silhouette loss from the best cut')
    parser.add_argument('--template-refresh-min', type=int, default=5, help='Distinct changed templates since global rebuild; 0 disables')
    parser.add_argument('--template-refresh-fraction', type=float, default=.2, help='Also require this fraction of baseline templates')
    parser.add_argument('--no-diagnosis', action='store_true')
    parser.add_argument('--diagnosis-model', default='gpt-6.1-sol')
    parser.add_argument('--max-diagnoses', type=int, default=3)
    parser.add_argument('--diagnosis-calls', type=int, default=5, help='Shared diagnosis/page experience/template experience cap')
    parser.add_argument('--recovery-actions', type=int, default=12)
    parser.add_argument('--experience-store', type=Path)
    args = parser.parse_args()
    if args.max_graph_nodes < 0 or not 0 < args.retention_half_life < float('inf'):
        parser.error('Invalid graph capacity or retrieval half-life')
    if args.framework_refresh_updates < 0:
        parser.error('Framework refresh interval must be nonnegative')
    if args.template_refresh_min < 0 or not 0 <= args.template_refresh_fraction <= 1 or not 0 <= args.template_tolerance <= 2:
        parser.error('Invalid global template parameters')
    if not 1 <= args.top_k <= 20 or not 1 <= args.max_steps <= 30 or args.max_calls < 1 or any(v is not None and v < 0 for v in (args.transition_calls, args.card_calls)):
        parser.error("Invalid top-k or execution budget")
    if args.out.exists():
        parser.error("Choose a new output directory; no automatic task retries")
    graph = Graph.load(args.graph.resolve())
    if not args.no_diagnosis and args.no_graph_update:
        parser.error('Diagnosis requires graph updates')
    if not args.no_diagnosis and (not 1 <= args.max_diagnoses <= 10 or args.diagnosis_calls < args.max_diagnoses or not 1 <= args.recovery_actions <= 30):
        parser.error('Invalid diagnosis budgets')
    framework = load_framework(graph, args.framework)
    memory = ExperienceStore(args.experience_store or graph.directory / 'experiences/memory.json', memory_namespace(graph))
    if args.cards is None and not args.without_cards:
        pointer = json.loads((graph.directory / 'function-cards/active.json').read_text(encoding='utf-8'))
        args.cards = graph.directory / pointer['cards']
    log = EventLog(args.out.resolve())
    encoder = LocalEncoder(graph, ROOT, args.encoder_python)
    client = OriginalResponses(args.env, log, args.max_calls, 2048, model=args.model, verified_protocol=False)
    client.configure_output_retries(args.max_output_retries)
    diagnosis = None if args.no_diagnosis else ResponsesClient(args.env, EventLog(log.directory / 'diagnosis-calls'),
        args.diagnosis_calls, 2048, model=args.diagnosis_model)
    knowledge = None if args.without_cards else TemplateKnowledge(graph, args.cards, log)
    descriptions = (ResponsesClient(args.env, EventLog(log.directory / 'graph-annotations'),
                    args.transition_calls, 2048, model='gpt-6-luna')
                    if not args.no_graph_update and args.transition_calls != 0 else None)
    card_client = (ResponsesClient(args.env, EventLog(log.directory / 'graph-cards'), args.card_calls, 2048,
                                  model='gpt-6-luna')
                   if not args.no_graph_update and not args.without_cards and args.card_calls != 0 else None)
    write_json(log.directory / "protocol.json", {"graph": str(graph.directory), "cards": str(args.cards.resolve()) if args.cards else None,
        "model": args.model, "top_k": args.top_k, "max_steps": args.max_steps, "max_calls": args.max_calls,
        "actor_prompt": "single-action-v2", "max_output_retries": args.max_output_retries,
        "ax_search": not args.no_ax_search,
        "ax_search_limits": None if args.no_ax_search else {"initial_lines": 800, "window_lines": 800, "max_searches": 3, "max_lines": 2500},
        "read_only": not args.allow_writes, "task": args.task, "template_context": not args.without_cards,
        "start_url": args.url or graph.data['start_url'], "home_url": args.home_url or args.url or graph.data['start_url'],
        "agent": "WebNavigator 15c06b96c6f65f8fc24a7ed739f801780d65eb88",
        "update_graph": not args.no_graph_update, "transition_calls": args.transition_calls if descriptions else 0,
        "transition_model": 'gpt-6-luna' if descriptions else None,
        "card_calls": args.card_calls if card_client else 0, "card_model": 'gpt-6-luna' if card_client else None,
        "finalize_graph_assets": not args.no_graph_update and not args.no_finalize_assets,
        'framework': framework['source'] if framework else None,
        'framework_refresh_updates': args.framework_refresh_updates,
        'template_clustering': args.template_clustering, 'template_tolerance': args.template_tolerance,
        'template_refresh_min': args.template_refresh_min, 'template_refresh_fraction': args.template_refresh_fraction,
        'max_graph_nodes': args.max_graph_nodes, 'retention_half_life': args.retention_half_life,
        'diagnosis_model': args.diagnosis_model if diagnosis else None, 'diagnosis_calls': args.diagnosis_calls if diagnosis else 0,
        'max_diagnoses': args.max_diagnoses, 'recovery_actions': args.recovery_actions, 'experience_store': str(memory.path),
        "max_total_model_calls": None if (descriptions and args.transition_calls is None) or (card_client and args.card_calls is None) else args.max_calls + (args.transition_calls if descriptions else 0) + (args.card_calls if card_client else 0) + (args.diagnosis_calls if diagnosis else 0),
        "scope": "Development smoke; not official benchmark success or paper replication"})
    try:
        encoder.ensure()
        embedding = JinaClient(encoder.index["endpoint"], log)
        result = run_task(graph, args.task, log, client, embedding,
            start_url=args.url, home_url=args.home_url, storage_state=args.storage_state,
            headed=args.headed, allow_writes=args.allow_writes, max_steps=args.max_steps,
            top_k=args.top_k, knowledge=knowledge, update_graph=not args.no_graph_update,
            description_client=descriptions, max_descriptions=args.transition_calls,
            card_client=card_client, max_cards=args.card_calls, finalize_graph_assets=not args.no_finalize_assets,
            framework=framework, diagnosis_client=diagnosis, experience_store=memory,
            framework_refresh_updates=args.framework_refresh_updates,
            template_clustering=args.template_clustering, template_tolerance=args.template_tolerance,
            template_refresh_min=args.template_refresh_min, template_refresh_fraction=args.template_refresh_fraction,
            max_graph_nodes=args.max_graph_nodes, retention_half_life=args.retention_half_life,
            max_diagnoses=args.max_diagnoses, max_replay_actions=args.recovery_actions,
            ax_search=not args.no_ax_search)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
