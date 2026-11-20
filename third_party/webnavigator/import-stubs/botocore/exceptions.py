"""Import shim for botocore.exceptions (see ../README.md)."""

from __future__ import annotations


class ClientError(Exception):
    """Stand-in for botocore.exceptions.ClientError."""


class BotoCoreError(Exception):
    """Stand-in for botocore.exceptions.BotoCoreError."""
