"""Open the project portal; launch manual review on demand."""
import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web2kg.common.storage import Graph
from web2kg.visualization.control import ReviewController, create_control_server
from web2kg.visualization.viewer import export_viewer


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root = ROOT
    parser = argparse.ArgumentParser(description="web2kg 项目总控：图谱、日志、使用说明与人工核验")
    pointer = root / "runs/active-graph.json"
    default_graph = Path(json.loads(pointer.read_text(encoding="utf-8"))["graph"]) if pointer.exists() else root / "runs/webnavigator-reddit-llm-120-v4"
    parser.add_argument("--graph", type=Path, default=default_graph)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--manual-port", type=int, default=8788)
    parser.add_argument("--storage-state", type=Path)
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=900)
    parser.add_argument("--no-open", action="store_true", help="不自动打开总控网页")
    parser.add_argument("--headless-review", action="store_true", help="自动化检查用；人工核验默认打开两个窗口")
    parser.add_argument("--encoder-python", type=Path, default=Path("D:/Anaconda/envs/web2kg-jina-gpu/python.exe"))
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    args.graph = args.graph.resolve()
    if not (args.graph / "graph.json").is_file():
        parser.error("图谱目录中没有 graph.json，请先建图或指定 --graph")
    if args.events and not args.events.is_file():
        parser.error("--events 文件不存在")
    if min(args.viewport_width, args.viewport_height) < 1:
        parser.error("视口尺寸必须为正数")
    url = f"http://127.0.0.1:{args.port}"
    http = requests.Session()
    http.trust_env = False
    try:
        existing = http.get(url + "/api/state", timeout=2)
    except requests.ConnectionError:
        existing = None
    except requests.Timeout:
        parser.exit(1, f"端口 {args.port} 的服务没有响应，请检查现有服务或使用其他端口。\n")
    if existing is not None:
        try:
            same_graph = Path(existing.json()["graph"]["directory"]).resolve() == args.graph
        except (ValueError, KeyError, TypeError):
            same_graph = False
        if not same_graph:
            parser.exit(1, f"端口 {args.port} 已被其他服务占用，请用 --port 选择空闲端口。\n")
        print(f"总控已在运行，打开现有服务: {url}", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        return
    if args.storage_state is None and args.graph == root / "runs/verified-reddit":
        args.storage_state = root / "auth/verified-reddit.json"
    graph = Graph.load(args.graph)
    if args.storage_state is None and urlsplit(graph.data["start_url"]).port == 10499:
        args.storage_state = root / "auth/dev-reddit.json"
    export_viewer(graph, args.events)
    controller = ReviewController(graph, root, args.manual_port,
        str(args.storage_state.resolve()) if args.storage_state else None,
        {"width": args.viewport_width, "height": args.viewport_height}, args.headless_review,
        encoder_config={"root": str(root), "python": str(args.encoder_python.resolve()),
                        "model_path": str(args.model_path.resolve()) if args.model_path else None})
    try:
        server = create_control_server(graph, root, controller, args.port)
    except OSError as error:
        parser.exit(1, f"总控服务启动失败：{error}。请用 --port 选择空闲端口。\n")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"项目总控: {url}\n图谱: {url}/graph/viewer.html\n日志: {url}/graph/viewer.html#logs", flush=True)
    print(f"当前图谱: {args.graph}\n总控日志: {controller.log.path}", flush=True)
    print("在主页点击“启动人工核验”打开两个窗口。按 Ctrl+C 关闭总控及本次核验。", flush=True)
    controller.log.emit("control_started", url=url, graph=str(args.graph))
    if not args.no_open:
        webbrowser.open(url)
    try:
        while thread.is_alive():
            thread.join(timeout=1)
            controller.snapshot()
    except KeyboardInterrupt:
        print("正在关闭总控；如人工核验正在执行操作，将等待当前操作结束并保留记录…", flush=True)
    finally:
        server.shutdown()
        server.server_close()
        controller.close()
        thread.join()
        controller.log.emit("control_stopped")


if __name__ == "__main__":
    main()
