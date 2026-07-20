"""Reproducible synthetic long-horizon workloads with ground truth."""

from .synthetic import (
    ATTRIBUTES,
    ENTITIES,
    Observation,
    Query,
    Workload,
    generate_workload,
)

__all__ = [
    "ATTRIBUTES",
    "ENTITIES",
    "Observation",
    "Query",
    "Workload",
    "generate_workload",
]
