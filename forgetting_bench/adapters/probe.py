"""Honest availability probes for live incumbent backends.

``make bench`` does not report mem0 / Letta numbers unless a probe says the
backend is actually runnable. A missing package or API key is a skip, not a
zero, and not a re-implementation score.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    name: str
    runnable: bool
    reason: str

    def summary(self) -> str:
        state = "runnable" if self.runnable else "not runnable"
        return f"{self.name}: {state} ({self.reason})"


def mem0_probe() -> Probe:
    try:
        import mem0  # noqa: F401
    except ImportError:
        return Probe("mem0", False, "mem0ai is not installed")
    if os.environ.get("MEM0_API_KEY"):
        return Probe("mem0", True, "MEM0_API_KEY is set; will use MemoryClient")
    if os.environ.get("OPENAI_API_KEY"):
        return Probe("mem0", True, "OPENAI_API_KEY is set; will use OSS Memory()")
    return Probe(
        "mem0",
        False,
        "mem0ai is installed but neither MEM0_API_KEY nor OPENAI_API_KEY is set",
    )


def letta_probe() -> Probe:
    if _import_letta_client() is None:
        return Probe("letta", False, "letta-client (or letta) is not installed")
    if os.environ.get("LETTA_API_KEY"):
        return Probe("letta", True, "LETTA_API_KEY is set; will use Letta Cloud")
    if os.environ.get("LETTA_BASE_URL"):
        return Probe("letta", True, "LETTA_BASE_URL is set; will use a self-hosted server")
    return Probe(
        "letta",
        False,
        "Letta SDK is installed but neither LETTA_API_KEY nor LETTA_BASE_URL is set",
    )


def _import_letta_client():
    try:
        import letta_client  # noqa: F401

        return "letta_client"
    except ImportError:
        pass
    try:
        import letta  # noqa: F401

        return "letta"
    except ImportError:
        return None


def incumbent_probes() -> list[Probe]:
    return [mem0_probe(), letta_probe()]
