"""Defensive parsing of mem0 / Letta SDK payloads.

These libraries have shifted method names and response envelopes across
releases. The adapters call the real SDK; this module only normalises what
comes back so a version skew fails with a clear error instead of a KeyError.
"""

from __future__ import annotations

from typing import Any


def as_mapping(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(payload, attr, None)
        if callable(fn):
            dumped = fn()
            if isinstance(dumped, dict):
                return dumped
    return {}


def result_rows(payload: Any) -> list[Any]:
    """Unwrap ``{"results": [...]}``, a bare list, or an object with ``.results``."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    data = as_mapping(payload)
    for key in ("results", "memories", "passages", "data"):
        rows = data.get(key)
        if isinstance(rows, list):
            return rows
    rows = getattr(payload, "results", None)
    if isinstance(rows, list):
        return rows
    return []


def row_text(row: Any) -> str:
    data = as_mapping(row)
    for key in ("memory", "content", "text", "value", "passage"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    for attr in ("memory", "content", "text", "value"):
        value = getattr(row, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def row_id(row: Any) -> str | None:
    data = as_mapping(row)
    for key in ("id", "memory_id", "passage_id"):
        value = data.get(key)
        if value is not None:
            return str(value)
    value = getattr(row, "id", None)
    return str(value) if value is not None else None


def row_metadata(row: Any) -> dict[str, Any]:
    data = as_mapping(row)
    meta = data.get("metadata")
    if isinstance(meta, dict):
        return meta
    meta = getattr(row, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def row_tags(row: Any) -> list[str]:
    data = as_mapping(row)
    tags = data.get("tags")
    if isinstance(tags, list):
        return [str(t) for t in tags]
    tags = getattr(row, "tags", None)
    if isinstance(tags, list):
        return [str(t) for t in tags]
    return []


def whitespace_tokens(text: str) -> int:
    return len(text.split()) if text else 0
