"""Reproducible synthetic long-horizon workloads with ground truth."""

from .synthetic import (
    ATTRIBUTES,
    ENTITIES,
    AttrSpec,
    Observation,
    Query,
    Workload,
    default_extractor,
    generate_workload,
)

__all__ = [
    "ATTRIBUTES",
    "ENTITIES",
    "AttrSpec",
    "Observation",
    "Query",
    "Workload",
    "default_extractor",
    "generate_workload",
]
