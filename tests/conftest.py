import threading
from http.server import ThreadingHTTPServer

import pytest

from fixture_site import FixtureHandler


@pytest.fixture(scope="module")
def site():
    FixtureHandler.writes = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()
