"""Import shim for google.generativeai (see ../README.md)."""

from __future__ import annotations


def configure(*args, **kwargs):  # noqa: ANN002, ANN003 - signature parity only
    """No-op: callers configure at import time, and this run never uses the SDK."""


class GenerativeModel:  # noqa: D101 - stub
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError(
            "google.generativeai is stubbed in this environment; the run configured "
            "here uses the OpenAI-compatible provider instead."
        )
