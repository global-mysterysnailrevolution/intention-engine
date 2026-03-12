"""Intention Engine: Intention-driven hypergraph for cross-ontology knowledge discovery."""

from .models import (
    Node,
    Hyperedge,
    HyperedgeProvenance,
    Predicate,
    Intention,
    SearchScope,
    ScoredNode,
    SearchResult,
    SearchExplanation,
    ExploitStats,
    ExploreStats,
    EngineConfig,
    _make_id,
)
from .engine import IntentionEngine
from .hypergraph import HypergraphStore
from .encoder import HashEncoder

__all__ = [
    "IntentionEngine",
    "HypergraphStore",
    "Node",
    "Hyperedge",
    "HyperedgeProvenance",
    "Predicate",
    "Intention",
    "SearchScope",
    "ScoredNode",
    "SearchResult",
    "SearchExplanation",
    "ExploitStats",
    "ExploreStats",
    "EngineConfig",
    "HashEncoder",
    "_make_id",
]
