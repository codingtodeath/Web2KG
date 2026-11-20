"""Loopback-only Jina multivector endpoint for text queries and local screenshots."""
import argparse
import hashlib
import json
import threading
import uuid
from importlib.metadata import version
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from jina_gpu import DEFAULT_MODEL, Encoder


def encoder_identity(encoder, model_path):
    digest = hashlib.sha256()
    for path in sorted(model_path.rglob("*")):
        if path.suffix not in {".json", ".py", ".safetensors"}:
            continue
        digest.update(path.relative_to(model_path).as_posix().encode())
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    digest.update(Path(__file__).with_name("jina_gpu.py").read_bytes())
    stable_vision = hasattr(encoder, "vision_dtype")
    if stable_vision:
        digest.update(Path(__file__).with_name("jina_vision_fp32.py").read_bytes())
    identity = {"model": "jina-embeddings-v4", "mode": encoder.mode, "max_pixels": 602112,
            "dtype": str(encoder.dtype), "pipeline": "retrieval-local-v1",
            "runtime": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
            "fingerprint": digest.hexdigest()}
    if stable_vision:
        identity.update(pipeline="retrieval-vision-fp32-v1", vision_dtype=str(encoder.vision_dtype))
    return identity


def handler_for(encoder, log_path, identity):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond(200 if self.path == "/health" else 404,
                         {"mode": encoder.mode, "ready": True, "load_seconds": encoder.load_seconds,
                          "encoder": identity})

        def do_POST(self):
            if self.path != "/v1/embeddings":
                self.respond(404, {"error": "Unknown endpoint"})
                return
            record = {"request_id": uuid.uuid4().hex}
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("Expected a JSON body of at most 1 MB")
                payload = json.loads(self.rfile.read(length))
                values = payload["input"]
                if not isinstance(values, list) or not 1 <= len(values) <= 8:
                    raise ValueError("input must contain 1 to 8 items")
                kind = "image" if isinstance(values[0], dict) and "image" in values[0] else "text"
                inputs = [v if isinstance(v, str) else v[kind] for v in values]
                if not all(isinstance(v, str) and v for v in inputs):
                    raise ValueError("Expected nonempty text strings or local image paths")
                if payload.get("task", "retrieval.query") not in ("retrieval.query", "retrieval.passage"):
                    raise ValueError("This endpoint supports retrieval query/passage only")
                vectors, timings = [], []
                with lock:
                    for value in inputs:
                        rows, stats = encoder.encode([value], kind, payload.get("task") != "retrieval.passage")
                        vectors.extend(rows)
                        timings.append(stats)
                record.update(status="passed", kind=kind, timings=timings)
                data = [{"index": i, "object": "embedding", "embeddings": row.tolist()} for i, row in enumerate(vectors)]
                self.respond(200, {"object": "list", "model": "jina-embeddings-v4", "data": data,
                                   "usage": {"total_tokens": sum(len(row) for row in vectors)},
                                   "local_config": {"mode": encoder.mode}, "encoder": identity,
                                   "request_id": record["request_id"]})
            except (ValueError, KeyError, TypeError, OSError) as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
                self.respond(400, {"error": record["error"], "request_id": record["request_id"]})
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
                self.respond(500, {"error": record["error"], "request_id": record["request_id"]})
            finally:
                with lock, log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        def log_message(self, *_):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["offload", "nf4", "int8", "cpu"], default="offload")
    parser.add_argument("--port", type=int, default=8793)
    parser.add_argument("--gpu-memory", default="4.8GiB")
    parser.add_argument("--vision-fp32", action="store_true", help="Use FP32 for every image's vision tower; text remains FP16")
    parser.add_argument("--log", type=Path, default=Path("runs/jina-gpu/service.jsonl"))
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    encoder_type = Encoder
    if args.vision_fp32:
        from jina_vision_fp32 import VisionFP32Encoder
        encoder_type = VisionFP32Encoder
    encoder = encoder_type(args.model_path, args.mode, gpu_memory=args.gpu_memory)
    identity = encoder_identity(encoder, args.model_path)
    print(f"Jina ready: http://127.0.0.1:{args.port}/v1/embeddings ({args.mode})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(encoder, args.log, identity)).serve_forever()


if __name__ == "__main__":
    main()
