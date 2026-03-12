from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np
import time
import uuid


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Node:
    id: str
    features: np.ndarray | None = None  # Dense feature vector
    metadata: dict[str, Any] = field(default_factory=dict)
    ontology: str = "default"
    created_at: float = field(default_factory=time.time)
    embedding: np.ndarray | None = None  # Cached sentence embedding


@dataclass
class HyperedgeProvenance:
    source: Literal["manual", "extracted", "minted"]
    intention: str | None = None
    predicates: list[str] | None = None
    coherence_method: str = "geometric"
    parent_edges: list[str] | None = None


@dataclass
class Hyperedge:
    id: str
    members: frozenset[str]  # Set of member node IDs
    label: str = ""
    provenance: HyperedgeProvenance = field(
        default_factory=lambda: HyperedgeProvenance(source="manual")
    )
    coherence_score: float = 0.0
    utility_context: str = ""
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    weight: float = 1.0


@dataclass
class Predicate:
    text: str
    embedding: np.ndarray | None = None
    weight: float = 1.0


@dataclass
class SearchScope:
    ontologies: set[str] | None = None
    max_depth: int = 2
    min_coherence: float = 0.3
    explore_budget: int = 100


@dataclass
class Intention:
    raw: str
    predicates: list[Predicate] = field(default_factory=list)
    embedding: np.ndarray | None = None
    scope: SearchScope | None = None


@dataclass
class ScoredNode:
    node: Node
    score: float
    source: Literal["exploit", "explore", "both"] = "exploit"
    via_edges: list[str] = field(default_factory=list)


@dataclass
class ExploitStats:
    edges_scored: int = 0
    edges_activated: int = 0
    nodes_reached: int = 0
    elapsed_ms: float = 0.0


@dataclass
class ExploreStats:
    nodes_projected: int = 0
    clusters_found: int = 0
    candidates_evaluated: int = 0
    edges_minted: int = 0
    elapsed_ms: float = 0.0


@dataclass
class SearchExplanation:
    intention: Intention
    exploit_stats: ExploitStats = field(default_factory=ExploitStats)
    explore_stats: ExploreStats = field(default_factory=ExploreStats)


@dataclass
class SearchResult:
    nodes: list[ScoredNode] = field(default_factory=list)
    exploited_edges: list[Hyperedge] = field(default_factory=list)
    minted_edges: list[Hyperedge] = field(default_factory=list)
    explanation: SearchExplanation | None = None


@dataclass
class EngineConfig:
    encoder_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_exploit_depth: int = 2
    exploit_weight: float = 0.7
    explore_enabled: bool = True
    explore_budget: int = 100
    utility_threshold_percentile: float = 80.0
    min_coherence: float = 0.3
    coherence_weights: tuple[float, float, float] = (0.5, 0.4, 0.1)
    novelty_threshold: float = 0.8
    min_edge_size: int = 2
    max_edge_size: int = 50
    decay_half_life_days: float = 30.0
    prune_threshold: float = 0.01
    use_llm_decomposition: bool = False
    llm_model: str | None = None
    max_mints_per_query: int = 10
