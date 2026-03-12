"""Top-level IntentionEngine: orchestrates hypergraph search driven by intention."""

from __future__ import annotations

import numpy as np

from .models import (
    EngineConfig,
    Hyperedge,
    Node,
    SearchResult,
    SearchScope,
    _make_id,
)
from .hypergraph import HypergraphStore
from .decomposer import IntentionDecomposer
from .search import SearchEngine


class IntentionEngine:
    """
    Top-level orchestrator for intention-driven hypergraph search.

    The engine manages a hypergraph that grows through use.  Each search
    potentially discovers new multi-way relationships (hyperedges) by
    projecting node features through an intention-derived utility lens.
    """

    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        self.store = HypergraphStore()
        self.decomposer = IntentionDecomposer()
        self._search_engine: SearchEngine | None = None
        self._encoder = None          # Lazy-loaded sentence-transformers model
        self._encode_fn = None        # Custom override function

    # ------------------------------------------------------------------
    # Encoder management
    # ------------------------------------------------------------------

    def set_encoder(self, encode_fn) -> None:
        """Set a custom encoding function: str -> np.ndarray.

        Useful for testing without sentence-transformers.
        """
        self._encode_fn = encode_fn

    def _get_encoder(self):
        """Return the active encoding callable.

        Falls back to HashEncoder if sentence-transformers is not installed.
        """
        if self._encode_fn is not None:
            return self._encode_fn
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._encoder = SentenceTransformer(self.config.encoder_model)
                return lambda text: self._encoder.encode(text, normalize_embeddings=True)
            except ImportError:
                from .encoder import HashEncoder
                self._encode_fn = HashEncoder(dim=self.config.embedding_dim)
                return self._encode_fn
        return lambda text: self._encoder.encode(text, normalize_embeddings=True)

    def _encode(self, text: str) -> np.ndarray:
        """Encode *text* to a 1-D float32 embedding vector."""
        encoder = self._get_encoder()
        result = encoder(text)
        if isinstance(result, np.ndarray):
            return result.astype(np.float32)
        return np.array(result, dtype=np.float32)

    # ------------------------------------------------------------------
    # SearchEngine (lazy, so config changes are reflected on first use
    # after the engine is reset)
    # ------------------------------------------------------------------

    @property
    def search_engine(self) -> SearchEngine:
        if self._search_engine is None:
            self._search_engine = SearchEngine(self.store, self.config)
        return self._search_engine

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(
        self,
        id: str,
        description: str,
        ontology: str = "default",
        metadata: dict | None = None,
    ) -> Node:
        """Add a node.  Embedding is computed from *description*."""
        text = f"[{ontology}] {description}" if ontology != "default" else description
        embedding = self._encode(text)

        node = Node(
            id=id,
            metadata={"description": description, **(metadata or {})},
            ontology=ontology,
            embedding=embedding,
        )
        self.store.add_node(node)
        return node

    def add_nodes_batch(self, nodes: list[dict]) -> list[Node]:
        """Batch-add nodes.

        Each dict must have ``id`` and ``description``; ``ontology`` and
        ``metadata`` are optional.
        """
        result: list[Node] = []
        for nd in nodes:
            node = self.add_node(
                id=nd["id"],
                description=nd["description"],
                ontology=nd.get("ontology", "default"),
                metadata=nd.get("metadata"),
            )
            result.append(node)
        return result

    # ------------------------------------------------------------------
    # Hyperedge management
    # ------------------------------------------------------------------

    def add_hyperedge(
        self,
        member_ids: set[str],
        label: str,
        metadata: dict | None = None,
    ) -> Hyperedge:
        """Manually add a known hyperedge."""
        edge = Hyperedge(
            id=_make_id("he"),
            members=frozenset(member_ids),
            label=label,
        )
        self.store.add_hyperedge(edge)
        return edge

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        intention: str,
        max_results: int = 20,
        explore: bool = True,
        scope: SearchScope | None = None,
    ) -> SearchResult:
        """Search the hypergraph driven by *intention*.

        Phase 1 (exploit): traverse existing hyperedge structure.
        Phase 2 (explore): project nodes through the intention lens,
                           discover and persist new hyperedges.

        New hyperedges minted during explore are automatically stored
        and reused by subsequent searches.
        """
        # Decompose
        intent = self.decomposer.decompose(intention, scope)

        # Embed intention and each predicate
        intent.embedding = self._encode(intention)
        for pred in intent.predicates:
            pred.embedding = self._encode(pred.text)

        # Temporarily override explore flag for this call
        original_explore = self.config.explore_enabled
        self.config.explore_enabled = explore

        result = self.search_engine.search(intent, max_results)

        self.config.explore_enabled = original_explore
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save hypergraph state to *path* (a directory)."""
        self.store.save(path)

    def load(self, path: str) -> None:
        """Load hypergraph state from *path* (a directory)."""
        self.store.load(path)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return basic graph statistics."""
        minted = sum(
            1 for e in self.store.edges.values()
            if e.provenance.source == "minted"
        )
        return {
            "nodes": self.store.num_nodes,
            "edges": self.store.num_edges,
            "minted_edges": minted,
            "manual_edges": self.store.num_edges - minted,
        }

    def explain_edge(self, edge_id: str) -> dict | None:
        """Return a human-readable explanation of how *edge_id* was created."""
        edge = self.store.get_edge(edge_id)
        if not edge:
            return None
        return {
            "id": edge.id,
            "members": sorted(edge.members),
            "label": edge.label,
            "source": edge.provenance.source,
            "intention": edge.provenance.intention,
            "predicates": edge.provenance.predicates,
            "coherence_score": edge.coherence_score,
            "access_count": edge.access_count,
            "weight": edge.weight,
        }

    def decay_edges(self, threshold: float | None = None) -> int:
        """Apply temporal decay and prune low-weight edges.

        Returns the number of edges pruned.
        """
        return self.store.decay_edges(
            half_life_days=self.config.decay_half_life_days,
            prune_threshold=threshold or self.config.prune_threshold,
        )
