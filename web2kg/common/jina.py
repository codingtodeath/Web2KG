"""Local Jina V4 screenshot/query multivectors and WebNavigator mean MaxSim."""
import hashlib
import time
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import requests


DEFAULT_ENDPOINT = "http://127.0.0.1:8793/v1/embeddings"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(rows):
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or not len(rows) or rows.shape[1] != 128 or not np.isfinite(rows).all():
        raise ValueError("Jina must return finite nonempty token × 128 multivectors")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Jina returned a zero token vector")
    return rows / norms


def maxsim(query, document):
    return float((normalize(query) @ normalize(document).T).max(axis=1).mean())


class JinaClient:
    def __init__(self, endpoint=DEFAULT_ENDPOINT, log=None, *, timeout=180, retry_connections=False, stop_file=None):
        parsed = urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username:
            raise ValueError("Jina screenshot service must be a local loopback HTTP endpoint")
        self.endpoint, self.log = endpoint, log
        self.timeout, self.retry_connections = timeout, retry_connections
        self.stop_file = Path(stop_file) if stop_file else None
        self.http = requests.Session()
        self.http.trust_env = False

    def identity(self):
        response = self.http.get(self.endpoint.rsplit("/v1/", 1)[0] + "/health", timeout=10)
        response.raise_for_status()
        identity = response.json().get("encoder")
        if not identity or not identity.get("fingerprint"):
            raise ValueError("Restart the dev Jina service: encoder identity is missing")
        return identity

    def encode(self, value, kind, expected):
        attempt = 0
        while True:
            if self.stop_file and self.stop_file.exists():
                raise RuntimeError('Local encoding cancelled by stop file')
            try:
                return self._encode_once(value, kind, expected)
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
                status = error.response.status_code if error.response is not None else None
                if not self.retry_connections or (status is not None and status not in {408, 429, 500, 502, 503, 504}):
                    raise
                attempt += 1
                delay = min(60, 2 ** min(attempt, 6))
                if self.log:
                    self.log.emit('jina_retry', attempt=attempt, kind=kind, delay_seconds=delay, http_status=status)
                for _ in range(delay):
                    if self.stop_file and self.stop_file.exists():
                        raise RuntimeError('Local encoding cancelled by stop file')
                    time.sleep(1)

    def _encode_once(self, value, kind, expected):
        started = time.monotonic()
        try:
            response = self.http.post(self.endpoint, json={"model": "jina-embeddings-v4",
                "task": "retrieval.query" if kind == "text" else "retrieval.passage",
                "input": [{kind: str(value)}], "return_multivector": True}, timeout=(10, self.timeout))
            response.raise_for_status()
            data = response.json()
            if data.get("encoder") != expected:
                raise ValueError("Jina encoder differs from the index; rebuild with matching settings")
            result = normalize(data["data"][0]["embeddings"])
            if self.log:
                self.log.emit("jina_encoded", kind=kind, tokens=len(result), seconds=time.monotonic()-started,
                              request_id=data.get("request_id"), encoder=expected["fingerprint"])
            return result
        except Exception as error:
            if self.log:
                self.log.emit("jina_failed", kind=kind, error=str(error), seconds=time.monotonic()-started)
            raise
