"""Local command API. All Playwright calls stay on the Python main thread."""
import json
import mimetypes
import queue
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler
from importlib.resources import files
from urllib.parse import urlsplit
from uuid import uuid4
from playwright.sync_api import Error as PlaywrightError

from ..common.browser import Browser
from ..common.http import LocalHTTPServer
from ..common.config import model_config
from ..common.models import ModelClient
from ..evaluation.manual import ManualSession


def create_server(session, commands, port=8768):
    token = secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def blob(self, data, mime):
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/":
                self.blob(files("web2kg.visualization").joinpath("manual.html").read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/state":
                self.respond({**session.snapshot(), "token": token})
            elif route == "/api/nodes":
                self.respond(session.nodes())
            elif route == "/api/export":
                self.respond(session.snapshot()["session"])
            else:
                parts = route.strip("/").split("/")
                path = None
                if len(parts) == 3 and parts[0] == "assets" and parts[1] in session.graph.nodes and parts[2] in {"page.png", "page-annotated.png"}:
                    path = session.graph.directory / "snapshots" / parts[1] / parts[2]
                elif len(parts) == 4 and parts[0] == "attempt-assets" and parts[3] == "page.png":
                    with session.lock:
                        allowed = any(row["id"] == parts[1] and row.get("result", {}).get("observed_id") == parts[2]
                                      for row in session.data["attempts"])
                    if allowed:
                        path = session.directory / "attempts" / parts[1] / "snapshots" / parts[2] / "page.png"
                elif len(parts) == 4 and parts[0] == "failure-assets" and parts[3] == "page.png":
                    with session.lock:
                        allowed = any(row["id"] == parts[1] and row.get("result", {}).get("failure_page", {}).get("snapshot_dir") == "failures/" + parts[2]
                                      for row in session.data["attempts"])
                    if allowed:
                        path = session.directory / "attempts" / parts[1] / "failures" / parts[2] / "page.png"
                if path and path.is_file():
                    self.blob(path.read_bytes(), mimetypes.guess_type(path.name)[0])
                else:
                    self.respond({"error": "Not found"}, 404)

        def do_POST(self):
            if urlsplit(self.path).path != "/api/command":
                self.respond({"error": "Not found"}, 404)
                return
            origin = self.headers.get("Origin")
            expected_origin = f"http://127.0.0.1:{self.server.server_port}"
            if (self.headers.get("X-Review-Token") != token or origin not in {None, expected_origin}
                    or self.headers.get("Content-Type", "").split(";")[0] != "application/json"):
                self.respond({"error": "请求来源或令牌无效"}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("命令大小无效")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or payload.get("op") not in {"query", "visit", "judge"}:
                    raise ValueError("未知命令")
                with session.lock:
                    if session.status["busy"]:
                        self.respond({"error": "已有命令正在执行，请等待完成"}, 409)
                        return
                    job = uuid4().hex
                    session.update(busy=True, job=job, command=payload["op"], phase="queued", message="命令已接收", result=None, error=None)
                    commands.put(payload)
                self.respond({"job": job}, 202)
            except (ValueError, TypeError) as error:
                self.respond({"error": str(error)}, 400)

    return LocalHTTPServer(("127.0.0.1", port), Handler)


def execute_command(session, payload):
    try:
        result = getattr(session, payload["op"])(payload)
        message = "查询完成" if payload["op"] == "query" else "人工结论已保存" if payload["op"] == "judge" else (
            "已按记录路径到达并核验目标状态" if result["success"] else "到达失败，已停止后续动作")
        session.update(busy=False, phase="completed", message=message, result=result)
    except Exception as error:
        session.log.emit("manual_command_failed", command=payload["op"], error=str(error))
        session.update(busy=False, phase="failed", message="命令未完成", error=str(error))


def serve_manual(graph, directory, env_file=".env", port=8768, storage_state=None, viewport=None, headless=False,
                 stop_event=None, on_ready=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    session = ManualSession(graph, directory, lambda log: ModelClient(model_config("embed", env_file), log))
    commands = queue.Queue()
    server = create_server(session, commands, port)
    url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"人工核验调试窗口: {url}\n结果目录: {session.directory}\nCtrl+C 关闭服务与两个浏览器窗口", flush=True)
    try:
        width, height = 1920, 1080
        if sys.platform == "win32":
            import ctypes
            width, height = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
        half = width // 2
        bounds = {"left": 0, "top": 0, "width": half, "height": height - 50, "windowState": "normal"}
        with Browser(graph.data["start_url"], session.log, headed=not headless,
                     storage_state=storage_state, viewport=graph.data.get("viewport", viewport), window_bounds=bounds,
                     identity_mode=graph.data.get("identity_mode", "semantic")) as browser:
            session.browser = browser
            debug = None
            if not headless:
                # Separate browser process/window; this page never controls the target site directly.
                debug = browser.playwright.chromium.launch(headless=False)
                page = debug.new_page(no_viewport=True)
                page.goto(url)
                cdp = page.context.new_cdp_session(page)
                window = cdp.send("Browser.getWindowForTarget")
                cdp.send("Browser.setWindowBounds", {"windowId": window["windowId"], "bounds": {**bounds, "left": half}})
            try:
                if on_ready:
                    on_ready(url)
                while stop_event is None or not stop_event.is_set():
                    if (browser.page.is_closed() or not browser.browser.is_connected()
                            or (debug and (not debug.is_connected() or page.is_closed()))):
                        session.log.emit("manual_window_closed", message="核验窗口关闭，结束本次会话")
                        break
                    try:
                        command = commands.get_nowait()
                    except queue.Empty:
                        # Pump Playwright events so the target site's network handlers keep running.
                        try:
                            browser.page.wait_for_timeout(100)
                        except PlaywrightError:
                            if browser.page.is_closed() or not browser.browser.is_connected():
                                session.log.emit("manual_window_closed", message="目标窗口关闭")
                                break
                            raise
                    else:
                        execute_command(session, command)
            finally:
                if debug and debug.is_connected():
                    debug.close()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
