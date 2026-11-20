"""Map the benchmark origin to its local copy; otherwise relay normal traffic."""
import select
import socket
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import requests

from ..common.safety import origin


class LocalSiteProxy:
    """Map only the benchmark origin; preserve other HTTP/HTTPS destinations."""
    def __init__(self, config):
        self.config, self.context = config, None
        backend = urlsplit(config['backend'])
        if backend.scheme != 'http' or backend.hostname not in {'127.0.0.1', 'localhost'}:
            raise ValueError('Evaluation backend must be local HTTP')
        if urlsplit(config['allowed_origin']).scheme != 'http':
            raise ValueError('Isolated HTTP site requires an HTTP logical origin')

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_CONNECT(self):
                try:
                    host, port = self.path.rsplit(':', 1)
                    remote = socket.create_connection((host.strip('[]'), int(port)), timeout=20)
                except (OSError, ValueError):
                    self.send_error(502, 'Tunnel unavailable')
                    return
                with remote:
                    self.send_response(200, 'Connection established')
                    self.end_headers()
                    while True:
                        ready, _, _ = select.select([self.connection, remote], [], [], 30)
                        if not ready:
                            return
                        for source in ready:
                            data = source.recv(65536)
                            if not data:
                                return
                            (remote if source is self.connection else self.connection).sendall(data)

            def forward(self):
                if self.headers.get('Transfer-Encoding'):
                    self.send_error(400, 'Streaming request bodies are unsupported')
                    return
                parts = urlsplit(self.path)
                target = (config['backend'].rstrip('/') + (parts.path or '/') + ('?' + parts.query if parts.query else '')
                          if origin(self.path) == origin(config['allowed_origin']) else self.path)
                body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                headers = {k: v for k, v in self.headers.items() if k.lower() not in {
                    'host', 'content-length', 'connection', 'proxy-connection', 'proxy-authorization'}}
                # Symfony and other servers derive absolute URLs from the visible Host.
                headers['Host'] = parts.netloc
                if origin(self.path) == origin(config['allowed_origin']):
                    headers.update(config.get('extra_http_headers') or {})
                # Fresh sessions avoid shared cookie state: Chromium owns all cookies.
                try:
                    with requests.Session() as session:
                        session.trust_env = False
                        response = session.request(self.command, target, headers=headers, data=body or None,
                                                   allow_redirects=False, timeout=20)
                        content = response.content
                        self.send_response(response.status_code)
                        for key in response.raw.headers:
                            if key.lower() not in {'content-length', 'content-encoding', 'transfer-encoding', 'connection'}:
                                for value in response.raw.headers.getlist(key):
                                    self.send_header(key, value)
                        self.send_header('Content-Length', response.headers.get('Content-Length', '0')
                                         if self.command == 'HEAD' else str(len(content)))
                        self.end_headers()
                        if self.command != 'HEAD':
                            self.wfile.write(content)
                except requests.RequestException:
                    self.send_error(502, 'Isolated backend unavailable')

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = forward

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def route_request(route, config, context):
    backend = config.get("backend")
    if backend:
        raise ValueError('Backend forwarding requires LocalSiteProxy for redirects')
    if config.get('extra_http_headers') and origin(route.request.url) == origin(config['allowed_origin']):
        route.continue_(headers={**route.request.headers, **config['extra_http_headers']})
    else:
        route.continue_()
