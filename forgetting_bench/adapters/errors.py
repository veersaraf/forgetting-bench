"""Errors for live incumbent adapters.

An adapter that is not actually talking to a backend must fail loudly. The
harness never invents mem0 / Letta numbers for a skipped or stubbed client.
"""

from __future__ import annotations


class AdapterUnavailable(RuntimeError):
    """Raised when a live backend cannot be constructed (missing SDK or credentials)."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        super().__init__(f"{name} is not runnable: {reason}")
