import argparse
import functools
import json
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .stage2.agent import run_task
from .visualization.annotations import annotate_graph
from .common.browser import Browser
from .common.config import model_config
from .stage1.crawl import CrawlBudget, crawl
from .evaluation.tasks import evaluate
from .evaluation.replay import replay_experiment
from .stage1.indexing import build_index
from .stage2.retrieval import choose, retrieve
from .common.models import ModelClient
from .stage2.navigation import navigate
from .common.storage import EventLog, Graph
from .visualization.viewer import export_viewer


def parser():
    root = argparse.ArgumentParser(description="Bounded interaction graphs and retrieval-assisted Web Agent")
    root.add_argument("--env-file", default=".env")
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check model access without displaying credentials")
    doctor.add_argument("--vision", action="store_true")
    crawl_cmd = commands.add_parser("crawl", help="Build a small graph on a resettable test site")
    crawl_cmd.add_argument("--url", required=True)
    crawl_cmd.add_argument("--test-site", action="store_true", required=True, help="Declare this is a resettable test environment")
    crawl_cmd.add_argument("--out", type=Path, required=True)
    crawl_cmd.add_argument("--max-states", type=int, default=25)
    crawl_cmd.add_argument("--max-depth", type=int, default=2)
    crawl_cmd.add_argument("--actions-per-page", type=int, default=12)
    crawl_cmd.add_argument("--per-group", type=int, default=1)
    crawl_cmd.add_argument("--max-attempts", type=int, default=100)
    crawl_cmd.add_argument("--resume", action="store_true", help="Continue an interrupted crawl with saved settings/rankings, skipping attempted actions")
    crawl_cmd.add_argument("--llm-review", action="store_true")
    crawl_cmd.add_argument("--exploration", choices=["grouped", "webnavigator", "llm_ranked"], default="grouped")
    crawl_cmd.add_argument("--llm-env", default="auth/llm-crawl.env")
    crawl_cmd.add_argument("--llm-max-calls", type=int, default=360)
    crawl_cmd.add_argument("--llm-output-tokens", type=int, default=2048)
    crawl_cmd.add_argument("--candidate-limit", type=int, default=160)
    crawl_cmd.add_argument('--serial-exploration', action='store_true', help='Use legacy synchronous text explanations')
    crawl_cmd.add_argument('--description-workers', type=int, default=2)
    crawl_cmd.add_argument('--max-pending-descriptions', type=int, default=4)
    crawl_cmd.add_argument('--ranking-provider', choices=['same', 'jev'], default='same')
    crawl_cmd.add_argument('--ranking-env', default='.env', help='Jev credentials; never copied to crawl outputs')
    crawl_cmd.add_argument('--ranking-model', default=None, help='Optional pinned Jev model')
    crawl_cmd.add_argument('--ranking-proxy', default=None, help='Explicit local HTTP proxy for Jev only')
    crawl_cmd.add_argument('--state-equivalence', action='store_true', help='Learn graph-local query-slot variations with Jev')
    crawl_cmd.add_argument("--identity-mode", choices=["semantic", "webnavigator", "interaction"],
                           help="Default: interaction for llm_ranked, webnavigator for its baseline, otherwise semantic")
    crawl_cmd.add_argument("--description-similarity", type=float, default=0.8,
                           help="Observation similarity threshold for extra AX context; does not merge states")
    index_cmd = commands.add_parser("index", help="Create text embeddings or an offline lexical index")
    index_cmd.add_argument("graph", type=Path)
    index_cmd.add_argument("--mode", choices=["embedding", "lexical", "jina_screenshot"], default="embedding")
    index_cmd.add_argument("--jina-endpoint", default="http://127.0.0.1:8793/v1/embeddings")
    index_cmd.add_argument("--jina-use-saved", choices=["whole", "tiled"], help="Switch saved index without re-encoding")
    index_cmd.add_argument("--jina-tiles", action="store_true", help="Opt in to whole-page plus overlapping crop retrieval")
    index_cmd.add_argument("--tile-height", type=int, default=900)
    index_cmd.add_argument("--tile-overlap", type=int, default=128)
    retrieve_cmd = commands.add_parser("retrieve")
    retrieve_cmd.add_argument("graph", type=Path)
    retrieve_cmd.add_argument("task")
    retrieve_cmd.add_argument("--top-k", type=int, default=5)
    retrieve_cmd.add_argument("--choose", action="store_true")
    nav_cmd = commands.add_parser("navigate")
    nav_cmd.add_argument("graph", type=Path)
    nav_cmd.add_argument("target")
    nav_cmd.add_argument("--strategy", choices=["xpath", "role"], default="role")
    replay_cmd = commands.add_parser("replay-experiment")
    replay_cmd.add_argument("graph", type=Path)
    replay_cmd.add_argument("--count", type=int, default=10)
    replay_cmd.add_argument("--repeats", type=int, default=3)
    run_cmd = commands.add_parser("run")
    run_cmd.add_argument("task")
    run_cmd.add_argument("--graph", type=Path)
    run_cmd.add_argument("--url", help="Required for graph-free baseline")
    run_cmd.add_argument("--baseline", action="store_true")
    run_cmd.add_argument("--max-steps", type=int, default=30)
    run_cmd.add_argument("--home-url")
    run_cmd.add_argument("--top-k", type=int, default=5)
    run_cmd.add_argument("--cards", type=Path)
    run_cmd.add_argument("--update-graph", action="store_true", help="Record new states/templates in a task-local graph")
    run_cmd.add_argument("--allow-writes", action="store_true", help="Explicitly authorize task execution to change test-site data")
    eval_cmd = commands.add_parser("evaluate")
    eval_cmd.add_argument("graph", type=Path)
    eval_cmd.add_argument("tasks", type=Path)
    eval_cmd.add_argument("--max-steps", type=int, default=30)
    eval_cmd.add_argument("--storage-state")
    for cmd in (run_cmd, eval_cmd):
        cmd.add_argument("--execution-env", default="auth/llm-crawl.env")
        cmd.add_argument("--model", default="gpt-5.6-terra")
        cmd.add_argument("--max-calls", type=int, default=31,
                         help="Actor action decisions; AX searches and page selection are accounted separately")
    serve_cmd = commands.add_parser("view")
    serve_cmd.add_argument("graph", type=Path)
    serve_cmd.add_argument("--port", type=int, default=8765)
    serve_cmd.add_argument("--export-only", action="store_true")
    serve_cmd.add_argument("--events", type=Path, help="Show another run's events.jsonl alongside this graph")
    annotate_cmd = commands.add_parser("annotate", help="Replay an existing graph and add numbered screenshots without changing originals")
    annotate_cmd.add_argument("graph", type=Path)
    manual_cmd = commands.add_parser("manual", help="Human retrieval review and verified discovery-path replay in two windows")
    manual_cmd.add_argument("graph", type=Path)
    manual_cmd.add_argument("--out", type=Path)
    manual_cmd.add_argument("--port", type=int, default=8768)
    manual_cmd.add_argument("--storage-state")
    manual_cmd.add_argument("--headless", action="store_true", help="For automated checks; normally two browser windows open")
    for cmd in (crawl_cmd, nav_cmd, replay_cmd, run_cmd, annotate_cmd):
        cmd.add_argument("--headed", action="store_true")
        cmd.add_argument("--storage-state", help="Playwright cookie/localStorage JSON; never stored in graph")
    for cmd in (crawl_cmd, nav_cmd, replay_cmd, annotate_cmd, manual_cmd):
        cmd.add_argument("--viewport-width", type=int, default=1280, help="Layout viewport width; screenshots cover the entire document")
        cmd.add_argument("--viewport-height", type=int, default=900, help="Layout viewport height, not a screenshot crop")
    for cmd in (nav_cmd, replay_cmd, run_cmd, eval_cmd):
        cmd.add_argument("--out", type=Path)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    graph = None
    def client(kind, log=None):
        return ModelClient(model_config(kind, args.env_file), log)

    def embedding_for(graph, log):
        data = json.loads((graph.directory / "index.json").read_text(encoding="utf-8"))
        if data["mode"] == "jina_screenshot":
            from .common.jina import JinaClient
            return JinaClient(data["endpoint"], log)
        return client("embed", log) if data["mode"] == "embedding" else None

    def new_log():
        directory = args.out or Path("runs") / f"{args.command}-{datetime.now():%Y%m%d-%H%M%S-%f}"
        log = EventLog(directory)
        print(f"Run: {directory.resolve()}", flush=True)
        return log

    def browser(url, log):
        mode = (graph.data.get("identity_mode", "semantic") if graph else
                getattr(args, "identity_mode", None) or
                {"webnavigator": "webnavigator", "llm_ranked": "interaction"}.get(
                    getattr(args, "exploration", None), "semantic"))
        return Browser(url, log, args.headed, args.storage_state, getattr(args, "allow_writes", False),
                       viewport={"width": args.viewport_width, "height": args.viewport_height},
                       identity_mode=mode)

    if args.command == "doctor":
        for kind in (["chat", "embed", "vision"] if args.vision else ["chat", "embed"]):
            model = client(kind)
            result = ({"dimension": model.embed(["Find a forum"]).shape[1]} if kind == "embed" else
                      model.json('Return {"ok":true}.', {}))
            print(json.dumps({"kind": kind, "provider": model.config.provider, "model": model.config.model, "result": result}))
        return
    if args.command == "crawl":
        if args.state_equivalence and (args.exploration != 'llm_ranked' or args.serial_exploration or args.resume):
            raise ValueError('State equivalence learning requires a fresh parallel llm_ranked crawl')
        if args.ranking_provider == 'jev' and (args.exploration != 'llm_ranked' or args.serial_exploration):
            raise ValueError('Jev ranking requires parallel llm_ranked exploration')
        if args.exploration in {"webnavigator", "llm_ranked"} and args.llm_review:
            raise ValueError("--llm-review applies only to grouped exploration; llm_ranked uses Responses directly")
        if not args.resume and (args.out / "graph.json").exists():
            raise ValueError("Graph already exists; choose a new --out directory")
        budget = CrawlBudget(args.max_states, args.max_depth, args.actions_per_page, args.per_group, args.max_attempts)
        log = new_log()
        model = client("chat", log) if args.llm_review else None
        explorer = judge = None
        if args.exploration == "llm_ranked":
            from .common.responses import ResponsesClient
            from .stage1.llm_exploration import LLMExplorer
            from .stage1.pipeline import PipelinedExplorer
            from .stage1.jev_ranking import JevRankingClient
            from .stage1.jev_equivalence import JevEquivalenceClient
            llm = ResponsesClient(args.llm_env, log, args.llm_max_calls, args.llm_output_tokens)
            ranking = (JevRankingClient(args.ranking_env, EventLog(args.out / 'ranking-calls'),
                args.llm_max_calls, model=args.ranking_model, proxy=args.ranking_proxy)
                if args.ranking_provider == 'jev' else llm)
            if args.state_equivalence:
                judge = JevEquivalenceClient(args.ranking_env, EventLog(args.out / 'equivalence-calls'),
                    args.llm_max_calls, model=args.ranking_model, proxy=args.ranking_proxy)
            explorer = (LLMExplorer(llm, args.out, args.candidate_limit, args.description_similarity)
                if args.serial_exploration else PipelinedExplorer(llm, args.out, args.candidate_limit,
                    description_workers=args.description_workers, max_pending=args.max_pending_descriptions,
                    ranking_client=ranking))
        try:
            with browser(args.url, log) as session:
                graph = crawl(session, args.out, log, budget, model, resume=args.resume, explorer=explorer, state_judge=judge)
        finally:
            if (args.out / "graph.json").exists():
                export_viewer(Graph.load(args.out))
        export_viewer(graph)
        print(json.dumps({"states": len(graph.nodes), "edges": len(graph.data["edges"])}))
        return
    if args.command == "run":
        graph = Graph.load(args.graph) if args.graph else None
        if not args.baseline and graph is None:
            raise ValueError("Supply --graph, or --baseline --url for a plain agent")
        url = args.url or (graph.data["start_url"] if graph else None)
        if not url:
            raise ValueError("Baseline requires --url or --graph")
        log = new_log()
        from .evaluation.webnavigator import OriginalResponses
        from .stage2.template_context import TemplateKnowledge
        chat = OriginalResponses(args.execution_env, log, args.max_calls, 2048, model=args.model, verified_protocol=False)
        embedding = embedding_for(graph, log) if graph and not args.baseline else None
        if graph is None:
            graph = Graph(log.directory / 'empty-graph')
            graph.data['start_url'] = url
        knowledge = TemplateKnowledge(graph, args.cards, log) if args.cards else None
        result = run_task(graph, args.task, log, chat, embedding, start_url=url,
            home_url=args.home_url, storage_state=args.storage_state, headed=args.headed,
            max_steps=args.max_steps, top_k=args.top_k, allow_writes=args.allow_writes,
            use_graph=not args.baseline, knowledge=knowledge, update_graph=args.update_graph)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    graph = Graph.load(args.graph)
    if args.command == "manual":
        from .visualization.manual import serve_manual
        directory = args.out or Path("runs") / f"manual-{datetime.now():%Y%m%d-%H%M%S}"
        serve_manual(graph, directory, args.env_file, args.port, args.storage_state,
                     {"width": args.viewport_width, "height": args.viewport_height}, args.headless)
    elif args.command == "annotate":
        log = EventLog(graph.directory)
        with browser(graph.data["start_url"], log) as session:
            result = annotate_graph(session, graph, log)
        export_viewer(graph)
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "index":
        log = EventLog(graph.directory)
        if args.mode == "jina_screenshot":
            from .common.jina import JinaClient
            from .stage1.visual_index import build_visual_index, activate_visual_index
            encoder = JinaClient(args.jina_endpoint, log)
            if args.jina_use_saved:
                data = activate_visual_index(graph, encoder, args.jina_use_saved)
            else:
                data = build_visual_index(graph, encoder, tiles=args.jina_tiles,
                                          tile_height=args.tile_height, overlap=args.tile_overlap, reuse_whole=True)
        else:
            data = build_index(graph, client("embed", log) if args.mode == "embedding" else None)
        log.emit("index_built", mode=data["mode"], nodes=len(data["ids"]))
        print(json.dumps({k: v for k, v in data.items() if k not in {"ids", "texts", "images_sha256"}}, indent=2))
    elif args.command == "retrieve":
        log = EventLog(graph.directory)
        candidates = retrieve(graph, args.task, embedding_for(graph, log), args.top_k)
        log.emit("retrieval", task=args.task, candidates=candidates)
        print(json.dumps(candidates, ensure_ascii=False, indent=2))
        if args.choose:
            print("Selected:", choose(args.task, candidates, client("chat", log), log))
    elif args.command in {"navigate", "replay-experiment"}:
        log = new_log()
        with browser(graph.data["start_url"], log) as session:
            if args.command == "navigate":
                result = navigate(session, graph, args.target, log, args.strategy)
                session.observe(log.directory)
            else:
                result = replay_experiment(session, graph, log, args.count, args.repeats)
        display = {k: v for k, v in result.items() if k != "trials"}
        print(json.dumps(display, ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        log = new_log()
        tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
        from .evaluation.webnavigator import OriginalResponses
        result = evaluate(graph, tasks, log.directory, lambda log: OriginalResponses(
                          args.execution_env, log, args.max_calls, 2048, model=args.model, verified_protocol=False),
                          lambda log: embedding_for(graph, log), args.max_steps, args.storage_state)
        print(json.dumps(result["summary"], indent=2))
    elif args.command == "view":
        if args.events and not args.events.is_file():
            raise ValueError("Events file does not exist")
        destination = export_viewer(graph, args.events)
        print(destination.resolve(), flush=True)
        if not args.export_only:
            handler = functools.partial(SimpleHTTPRequestHandler, directory=str(graph.directory.resolve()))
            server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
            print(f"http://127.0.0.1:{args.port}/viewer.html", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
