"""Intention Engine: Intention-driven hypergraph for cross-ontology knowledge discovery."""

from .models import (
    Node,
    Hyperedge,
    HyperedgeProvenance,
    IntentionEvent,
    Predicate,
    Intention,
    SearchScope,
    ScoredNode,
    SearchResult,
    SearchExplanation,
    ExploitStats,
    ExploreStats,
    EngineConfig,
    TemporalQuery,
    TemporalDiff,
    _make_id,
)
from .engine import IntentionEngine
from .hypergraph import HypergraphStore
from .encoder import HashEncoder
from .events import GraphEvent, EventLog, EventType
from .temporal import temporal_embedding, is_edge_valid_at, temporal_similarity
from .rag import IntentionRAG, RAGConfig
from .ingestion import IngestionPipeline, IngestConfig, IngestResult
from .context import ContextAssembler, ContextConfig
from .chunker import DocumentChunker, ChunkerConfig

__all__ = [
    "IntentionEngine",
    "HypergraphStore",
    "Node",
    "Hyperedge",
    "HyperedgeProvenance",
    "IntentionEvent",
    "Predicate",
    "Intention",
    "SearchScope",
    "ScoredNode",
    "SearchResult",
    "SearchExplanation",
    "ExploitStats",
    "ExploreStats",
    "EngineConfig",
    "TemporalQuery",
    "TemporalDiff",
    "HashEncoder",
    "_make_id",
    "GraphEvent",
    "EventLog",
    "EventType",
    "temporal_embedding",
    "is_edge_valid_at",
    "temporal_similarity",
    "IntentionRAG",
    "RAGConfig",
    "IngestionPipeline",
    "IngestConfig",
    "IngestResult",
    "ContextAssembler",
    "ContextConfig",
    "DocumentChunker",
    "ChunkerConfig",
]
