"""Start a matching local encoder and clean up only the process we own."""
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

from .jina import JinaClient


class LocalEncoder:
    def __init__(self, graph, root, python, model_path=None):
        self.root, self.python = Path(root), Path(python)
        self.model_path = model_path
        self.index = json.loads((graph.directory / "index.json").read_text(encoding="utf-8"))
        self.process = self.stream = None

    def ensure(self, stop=None):
        if self.index.get("mode") != "jina_screenshot":
            return
        client = JinaClient(self.index["endpoint"])
        try:
            identity = client.identity()
        except OSError:
            identity = None
        if identity is None:
            if not self.python.is_file():
                raise ValueError("Jina Python 环境不存在，请指定 --encoder-python")
            log = self.root / "runs/jina-gpu/control-encoder.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            self.stream = log.open("a", encoding="utf-8", buffering=1)
            command = [str(self.python), str(self.root / "scripts/serve_jina_gpu.py"),
                       "--port", str(urlsplit(self.index["endpoint"]).port)]
            if self.index["encoder"].get("pipeline") == "retrieval-vision-fp32-v1":
                command.append("--vision-fp32")
            if self.model_path:
                command += ["--model-path", str(self.model_path)]
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            self.process = subprocess.Popen(command, cwd=self.root, env=env,
                stdout=self.stream, stderr=self.stream,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline and self.process.poll() is None:
                if stop and stop.is_set():
                    return
                try:
                    identity = client.identity()
                    break
                except OSError:
                    time.sleep(.5)
            if identity is None:
                raise RuntimeError(f"Jina 未就绪，请查看 {log}")
        if identity != self.index["encoder"]:
            raise ValueError("当前 Jina 配置与截图索引不一致，请关闭冲突服务后重开")

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=15)
        if self.stream:
            self.stream.close()
