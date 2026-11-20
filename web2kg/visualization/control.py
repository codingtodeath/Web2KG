"""Project portal and lifecycle for an independent Python review worker."""
import contextlib
import html
import json
import mimetypes
import multiprocessing
import queue
import secrets
import threading
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from importlib.resources import files
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import markdown

from ..common.storage import EventLog, Graph
from ..common.http import LocalHTTPServer
from ..common.jina_runtime import LocalEncoder
from .manual import serve_manual


def review_worker(graph_path, directory, env_file, port, storage_state, viewport, headless, stop, updates, encoder_config):
    with (Path(directory) / "console.log").open("w", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            encoder = None
            try:
                graph = Graph.load(graph_path)
                if encoder_config and (graph.directory / "index.json").is_file():
                    updates.put({"state": "starting", "message": "正在检查本地 Jina，首次加载可能需要等待…"})
                    encoder = LocalEncoder(graph, **encoder_config)
                    encoder.ensure(stop)
                if stop.is_set():
                    updates.put({"state": "stopped", "message": "启动已取消"})
                    return
                serve_manual(graph, directory, env_file, port, storage_state, viewport, headless,
                             stop_event=stop, on_ready=lambda url: updates.put({"state": "running", "url": url,
                                                                              "message": "人工核验已就绪"}))
            except Exception as error:
                traceback.print_exc()
                updates.put({"state": "failed", "message": str(error)})
            else:
                updates.put({"state": "stopped", "message": "人工核验已停止，记录已保留"})
            finally:
                if encoder:
                    encoder.close()


class ReviewController:
    def __init__(self, graph, root, port=8768, storage_state=None, viewport=None, headless=False, encoder_config=None):
        self.graph, self.root, self.port = graph, Path(root), port
        self.storage_state, self.viewport, self.headless = storage_state, viewport, headless
        self.encoder_config = encoder_config
        self.context = multiprocessing.get_context("spawn")
        self.lock = threading.RLock()
        self.process = self.stop_event = self.updates = None
        self.log = EventLog(self.root / "runs" / f"control-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}")
        self.status = {"state": "idle", "message": "按需启动，打开目标网站和调试界面两个窗口", "url": None,
                       "directory": None, "console_log": None}

    def snapshot(self):
        with self.lock:
            if self.updates:
                while True:
                    try:
                        update = self.updates.get_nowait()
                    except queue.Empty:
                        break
                    if update["state"] == "running" and self.stop_event.is_set():
                        update.update(state="stopping", message="等待当前操作完成后关闭")
                    self.status.update(update)
                    self.log.emit("review_status", **update)
            if self.process and not self.process.is_alive() and self.status["state"] in {"starting", "running", "stopping"}:
                state = "stopped" if self.process.exitcode == 0 else "failed"
                self.status.update(state=state, message=f"核验进程已退出（代码 {self.process.exitcode}），可查看启动日志")
                self.log.emit("review_exit", **self.status)
            return {**self.status, "pid": self.process.pid if self.process and self.process.is_alive() else None}

    def start(self):
        with self.lock:
            self.snapshot()
            if self.process and self.process.is_alive():
                return dict(self.status)
            if self.storage_state and not Path(self.storage_state).is_file():
                raise ValueError("登录态文件不存在，请通过 --storage-state 指定有效文件")
            if self.updates:
                self.updates.close()
                self.process.close()
            directory = self.root / "runs" / f"manual-{self.graph.directory.name}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
            directory.mkdir(parents=True)
            self.stop_event, self.updates = self.context.Event(), self.context.Queue()
            self.process = self.context.Process(target=review_worker, args=(str(self.graph.directory), str(directory),
                str(self.root / ".env"), self.port, self.storage_state, self.viewport, self.headless,
                self.stop_event, self.updates, self.encoder_config))
            self.status.update(state="starting", message="正在启动 Python 浏览器与核验窗口…", url=None,
                               directory=str(directory), console_log=str(directory / "console.log"))
            try:
                self.process.start()
            except OSError:
                self.process.close()
                self.updates.close()
                self.process = self.updates = None
                self.status.update(state="failed", message="核验进程未能启动")
                raise
            self.log.emit("review_start", graph=str(self.graph.directory), directory=str(directory), pid=self.process.pid)
            return dict(self.status)

    def stop(self):
        with self.lock:
            self.snapshot()
            if self.process and self.process.is_alive():
                self.stop_event.set()
                self.status.update(state="stopping", message="等待当前操作完成后关闭，已保存记录会保留")
                self.log.emit("review_stop_requested")
            return dict(self.status)

    def close(self):
        self.stop()
        if self.process:
            self.process.join()
            self.snapshot()
            self.process.close()
            self.updates.close()


def create_control_server(graph, root, controller, port=8769):
    root, graph_dir = Path(root).resolve(), graph.directory.resolve()
    token = secrets.token_urlsafe(24)
    index_path = graph_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    info = {"directory": str(graph_dir), "start_url": graph.data["start_url"], "nodes": len(graph.nodes),
            "edges": len(graph.data["edges"]), "attempts": graph.data.get("attempts", 0),
            "index": index.get("model", index.get("mode", "尚未构建")),
            "review_mode": graph.data.get("review_mode", "unknown")}
    links = {"graph": "/graph/viewer.html", "logs": "/graph/viewer.html#logs",
             "docs": "/docs/usage.md", "guide": "/docs/usage.md"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, mime="application/json; charset=utf-8", status=200):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            if route == "/":
                self.send(files("web2kg.visualization").joinpath("control.html").read_bytes(), "text/html; charset=utf-8")
                return
            if route == "/api/state":
                current_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
                info["index"] = current_index.get("model", current_index.get("mode", "尚未构建"))
                self.send({"graph": info, "links": links, "review": controller.snapshot(), "token": token,
                           "control_log": str(controller.log.path)})
                return
            if route == "/api/review-log":
                path = controller.snapshot().get("console_log")
                content = Path(path).read_text(encoding="utf-8") if path and Path(path).exists() else "尚无启动日志"
                self.send(content[-24000:].encode("utf-8"), "text/plain; charset=utf-8")
                return
            target = None
            # Only the selected graph and documentation are exposed, never auth or .env.
            mappings = [("/graph/", graph_dir), ("/docs/", root / "docs")]
            if graph_dir.is_relative_to(root / "runs"):
                mappings.append(("/" + graph_dir.relative_to(root).as_posix() + "/", graph_dir))
            for prefix, base in mappings:
                if route.startswith(prefix):
                    candidate = (base / route[len(prefix):]).resolve()
                    if candidate.is_relative_to(base) and candidate.is_file():
                        target = candidate
                    break
            if target:
                if target.suffix == ".md" and target.is_relative_to(root / "docs"):
                    body = markdown.markdown(target.read_text(encoding="utf-8"), extensions=["fenced_code", "tables"])
                    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                                f'<title>{html.escape(target.stem)} · web2kg</title>'
                                '<style>body{max-width:900px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui;}'
                                'pre{overflow:auto;padding:16px;background:#f3f5f7;}table{border-collapse:collapse;}'
                                'th,td{padding:8px;border:1px solid #ddd;}a{color:#185abd;}</style>'
                                f'<body><nav><a href="/">总控</a> · <a href="/docs/usage.md">使用说明</a></nav>{body}</body></html>')
                    self.send(document.encode("utf-8"), "text/html; charset=utf-8")
                    return
                mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                if mime.startswith("text/"):
                    mime += "; charset=utf-8"
                self.send(target.read_bytes(), mime)
            else:
                self.send({"error": "Not found"}, status=404)

        def do_POST(self):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if (self.headers.get("X-Control-Token") != token or self.headers.get("Origin") not in {None, origin}
                    or self.headers.get("Content-Type", "").split(";")[0] != "application/json"):
                self.send({"error": "请求来源或令牌无效"}, status=403)
                return
            route = urlsplit(self.path).path
            if route not in {"/api/review/start", "/api/review/stop"}:
                self.send({"error": "Not found"}, status=404)
                return
            try:
                result = controller.start() if route.endswith("/start") else controller.stop()
                self.send(result, status=202)
            except (OSError, ValueError) as error:
                controller.log.emit("review_control_failed", route=route, error=str(error))
                self.send({"error": str(error)}, status=400)

    return LocalHTTPServer(("127.0.0.1", port), Handler)
