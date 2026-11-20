"""Human queries and judgments, with verified replay of the recorded discovery path."""
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from ..common.diagnostics import record_failure
from ..common.inspection import discovery_path, graph_inspection
from ..common.page_text import fingerprint
from ..common.replay import execute_path, verify
from ..common.storage import EventLog, write_json
from ..stage2.retrieval import retrieve


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ManualSession:
    def __init__(self, graph, directory, embedding_factory=None):
        self.graph, self.directory = graph, Path(directory).resolve()
        self.lock = RLock()
        self.log = EventLog(self.directory)
        self.embedding_factory = embedding_factory
        self.index_path = graph.directory / "index.json"
        self.index = json.loads(self.index_path.read_text(encoding="utf-8")) if self.index_path.exists() else None
        self.index_files = {"index.json": file_hash(self.index_path)} if self.index else {}
        if self.index and self.index.get("vectors"):
            vector = (graph.directory / self.index["vectors"]).resolve()
            if not vector.is_relative_to(graph.directory.resolve()):
                raise ValueError("Index vector path is outside the graph directory")
            self.index_files[self.index["vectors"]] = file_hash(vector)
        self.inspection = graph_inspection(graph)
        self.texts = dict(zip(self.index["ids"], self.index["texts"])) if self.index else {}
        self.metadata = {"graph_directory": str(graph.directory.resolve()), "root": graph.data["root"],
                         "start_url": graph.data["start_url"], "nodes": len(graph.nodes),
                         "graph_hash": file_hash(graph.directory / "graph.json"),
                         "index_files": self.index_files, "index_mode": self.index.get("mode") if self.index else None,
                         "index_model": self.index.get("model") if self.index else None,
                         "index_provider": self.index.get("provider") if self.index else None,
                         "index_valid": bool(self.index and self.index["fingerprint"] == fingerprint(graph))}
        self.result_path = self.directory / "manual_review.json"
        if self.result_path.exists():
            self.data = json.loads(self.result_path.read_text(encoding="utf-8"))
            if self.data["graph"] != self.metadata:
                raise ValueError("输出目录属于另一版本的图谱或索引，请选择新的 --out")
            for row in self.data["attempts"]:
                if row["status"] == "running":
                    row.update(status="completed", phase="interrupted", result={"success": False, "steps": 0,
                               "failed_step": None, "error": "上次进程在完成核验前结束，到达情况未知"})
            for row in self.data["queries"]:
                if "success" not in row:
                    row.update(success=False, error="上次查询在保存结果前中断")
        else:
            self.data = {"version": 1, "session_id": uuid4().hex, "created_at": now(),
                         "graph": self.metadata, "queries": [], "attempts": [], "judgments": []}
        self.status = {"busy": False, "phase": "idle", "message": "可以输入需求，或浏览全部节点", "job": None}
        self.browser = None
        self.save()

    def save(self):
        with self.lock:
            write_json(self.result_path, self.data)

    def update(self, **values):
        with self.lock:
            self.status.update(values)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({"status": self.status, "session": self.data})

    def nodes(self):
        return [{"id": node["id"], "title": node["title"], "url": node["url"], "depth": node["depth"],
                 "index_text": self.texts.get(node["id"]),
                 "inspection": self.inspection["nodes"][node["id"]],
                 "path": [{"kind": a["kind"], "name": a["name"], "number": a.get("number")} for a in node["path"]],
                 "image": f"/assets/{node['id']}/page.png", "annotated": f"/assets/{node['id']}/page-annotated.png"}
                for node in self.graph.nodes.values()]

    def query(self, payload):
        text, top_k = payload.get("text", "").strip(), payload.get("top_k", 5)
        if not text or len(text) > 6000 or type(top_k) is not int or not 1 <= top_k <= 50:
            raise ValueError("请输入需求，top-k 范围为 1–50")
        row = {"id": uuid4().hex, "time": now(), "text": text, "top_k": top_k, "candidates": []}
        with self.lock:
            self.data["queries"].append(row)
        self.save()
        self.update(phase="retrieval", message="正在检索已保存的页面索引")
        started = time.monotonic()
        try:
            if not self.metadata["index_valid"]:
                raise ValueError("索引缺失或与图谱不一致；可以浏览全部节点，请先使用 index 命令构建索引")
            if any(file_hash(self.graph.directory / name) != value for name, value in self.index_files.items()):
                raise ValueError("索引已被修改，请重新启动核验程序以记录新版本")
            model = self.embedding_factory(self.log) if self.index["mode"] == "embedding" else None
            if self.index["mode"] == "jina_screenshot":
                from ..common.jina import JinaClient
                model = JinaClient(self.index["endpoint"], self.log)
            candidates = retrieve(self.graph, text, model, top_k)
            with self.lock:
                row.update(candidates=[{**candidate, "rank": rank} for rank, candidate in enumerate(candidates, 1)], success=True)
        except Exception as error:
            with self.lock:
                row.update(success=False, error=str(error))
            raise
        finally:
            with self.lock:
                row["seconds"] = round(time.monotonic() - started, 3)
            self.log.emit("manual_query", **row)
            self.save()
        return {"query_id": row["id"]}

    def query_row(self, query_id):
        if query_id is None:
            return None
        row = next((row for row in self.data["queries"] if row["id"] == query_id), None)
        if row is None:
            raise ValueError("查询记录不存在")
        return row

    def visit(self, payload):
        target, source = payload.get("node_id"), payload.get("source")
        if target not in self.graph.nodes or source not in {"topk", "all"}:
            raise ValueError("请选择有效节点与候选来源")
        query = self.query_row(payload.get("query_id"))
        rank = next((c["rank"] for c in query["candidates"] if c["id"] == target), None) if query else None
        if source == "topk" and rank is None:
            raise ValueError("所选节点不在这次查询的候选中")
        row = {"id": uuid4().hex, "time": now(), "query_id": query["id"] if query else None,
               "node_id": target, "source": source, "rank": rank, "status": "running", "path": []}
        with self.lock:
            self.data["attempts"].append(row)
        self.save()
        log = EventLog(self.directory / "attempts" / row["id"])
        log.emit("manual_selection", **row)
        self.log.emit("manual_selection", **row)
        phase, result, completed = "path_validation", None, 0
        started = time.monotonic()
        try:
            path = discovery_path(self.graph, target)
            with self.lock:
                row["path"] = path
            self.save()
            phase = "root_restore"
            self.update(phase=phase, message="恢复起点，并核验起点状态", attempt_id=row["id"], steps=0, total=len(path))
            self.browser.log = log
            self.browser.reset()
            verify(self.browser, self.graph.data["root"])
            log.emit("manual_root_verified", node=self.graph.data["root"])
            phase = "replay"
            self.update(phase=phase, message=f"起点已确认，按记录路径执行 {len(path)} 步")
            # Reuse the existing checked executor, one recorded edge at a time for live progress.
            for index, edge in enumerate(path):
                self.update(message=f"第 {index + 1}/{len(path)} 步：{edge['action']['kind']} · {edge['action']['name']}", steps=index)
                step = execute_path(self.browser, self.graph, [edge], log)
                log.emit("manual_path_step", path_index=index, edge=edge["id"], **step)
                if not step["success"]:
                    result = {**step, "failed_step": index, "steps": index}
                    break
                self.update(steps=index + 1)
                completed = index + 1
            if result is None:
                phase = "target_check"
                observed = verify(self.browser, target)
                result = {"success": True, "steps": len(path), "failed_step": None, "observed_id": observed["id"]}
            try:
                observed = self.browser.observe(log.directory)
                result.update(observed_id=observed["id"], url=observed["url"], title=observed["title"],
                              screenshot=f"/attempt-assets/{row['id']}/{observed['id']}/page.png")
            except Exception as capture_error:
                result["capture_error"] = str(capture_error)
        except Exception as error:
            result = {"success": False, "steps": completed, "failed_step": None, "error": str(error)}
            if self.browser:
                record_failure(self.browser, log, "manual_visit_failed", self.graph.nodes[self.graph.data["root"]],
                               None, error, phase, False)
                failure = json.loads(log.path.read_text(encoding="utf-8").splitlines()[-1]).get("failure_page")
                if failure:
                    result.update(failure_page=failure, url=failure["url"], observed_id=failure["id"],
                                  screenshot=f"/failure-assets/{row['id']}/{Path(failure['snapshot_dir']).name}/page.png")
        finally:
            if result is None:
                phase = "interrupted"
                result = {"success": False, "steps": completed, "failed_step": None, "error": "核验被中断，到达情况未知"}
            with self.lock:
                row.update(status="completed", phase=phase, result=result, seconds=round(time.monotonic() - started, 3))
            log.emit("manual_arrival", **row)
            self.log.emit("manual_arrival", **row)
            self.save()
            if self.browser:
                self.browser.log = self.log
        return {"attempt_id": row["id"], **result}

    def judge(self, payload):
        query = self.query_row(payload.get("query_id"))
        verdict, expected = payload.get("verdict"), payload.get("expected_node_id") or None
        if query is None or not query.get("success"):
            raise ValueError("请先选择一次成功的检索记录")
        if verdict not in {"hit", "retrieval_miss", "coverage_miss", "uncertain"}:
            raise ValueError("请选择核验结论")
        ids = {row["id"] for row in query["candidates"]}
        if expected and expected not in self.graph.nodes:
            raise ValueError("目标节点不存在")
        if verdict == "hit" and expected not in ids:
            raise ValueError("请指定 top-k 中符合需求的节点")
        if verdict == "retrieval_miss" and (expected not in self.graph.nodes or expected in ids):
            raise ValueError("检索遗漏需要指定图中存在但未进入 top-k 的节点")
        if verdict == "coverage_miss" and expected:
            raise ValueError("建图未覆盖时不应指定已有目标节点")
        row = {"id": uuid4().hex, "time": now(), "query_id": query["id"], "verdict": verdict,
               "expected_node_id": expected, "note": str(payload.get("note", ""))[:4000]}
        with self.lock:
            self.data["judgments"].append(row)
        self.log.emit("manual_judgment", **row)
        self.save()
        return row
