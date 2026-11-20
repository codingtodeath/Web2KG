"""Import shim for boto3.

WebNavigator imports every LLM provider module at import time, including the
Bedrock-based ones (claude/llama/mistral/titan/cohere). Those providers are not
used in this experiment (``model_family: gpt`` over an OpenAI-compatible
endpoint), but the import still has to succeed. This module satisfies the import
and fails loudly if a Bedrock call is actually attempted.
"""

from __future__ import annotations


class _Unavailable:
    def __getattr__(self, name: str):
        raise RuntimeError(
            "boto3 is stubbed in this environment; Bedrock providers are not used "
            "by the WebNavigator run configured here."
        )


def client(*args, **kwargs):  # noqa: ANN002, ANN003 - signature parity only
    raise RuntimeError(
        "boto3 is stubbed in this environment; Bedrock providers are not used by "
        "the WebNavigator run configured here."
    )


def Session(*args, **kwargs):  # noqa: ANN002, ANN003 - signature parity only
    return _Unavailable()


__version__ = "0.0.0-stub"
