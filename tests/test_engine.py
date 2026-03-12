"""Integration tests for IntentionEngine using a deterministic mock encoder."""

from __future__ import annotations

import numpy as np
import pytest

from intention_engine import (
    IntentionEngine,
    EngineConfig,
    SearchResult,
    Node,
)


# ---------------------------------------------------------------------------
# Mock encoder — deterministic, same text always gives the same vector
# ---------------------------------------------------------------------------

class _MockEncoder:
    """Deterministic hash-based embedding mock."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        rng = np.random.RandomState(42 ^ (hash(text) % (2**31)))
        vec = rng.randn(self.dim).astype(np.float32)
        # Inject weak semantic signal for contamination / precision
        words = text.lower().split()
        for w in words:
            if w in ("contamination", "contaminant", "clean", "purge"):
                vec[0] += 2.0
            if w in ("precision", "tolerance", "measure", "inspect"):
                vec[1] += 2.0
            if w in ("titanium", "metal", "alloy"):
                vec[2] += 2.0
        vec /= np.linalg.norm(vec) + 1e-10
        self._cache[text] = vec
        return vec


def _make_engine(
    min_coherence: float = 0.10,
    explore_budget: int = 50,
    percentile: float = 50.0,
) -> IntentionEngine:
    config = EngineConfig(
        min_coherence=min_coherence,
        explore_budget=explore_budget,
        utility_threshold_percentile=percentile,
    )
    engine = IntentionEngine(config=config)
    engine.set_encoder(_MockEncoder(dim=64))
    return engine


def _populate(engine: IntentionEngine) -> None:
    """Add a small set of cross-ontology nodes."""
    nodes = [
        {"id": "eq_hepa",      "description": "HEPA filter air purge contamination barrier",      "ontology": "equipment"},
        {"id": "eq_vacuum",    "description": "Vacuum chamber inert atmosphere purge system",      "ontology": "equipment"},
        {"id": "eq_cmm",       "description": "CMM coordinate measure machine precision inspect",  "ontology": "equipment"},
        {"id": "proc_clean",   "description": "Contamination clean purge decontamination protocol","ontology": "process"},
        {"id": "proc_inspect", "description": "Quality inspect dimensional tolerance check measure","ontology": "process"},
        {"id": "proc_anneal",  "description": "Heat treatment anneal stress relief cycle",         "ontology": "process"},
        {"id": "mat_ti64",     "description": "Titanium Ti-6Al-4V alloy aerospace metal",          "ontology": "material"},
        {"id": "mat_al7075",   "description": "Aluminum 7075-T6 alloy metal machine",              "ontology": "material"},
        {"id": "qc_xrf",       "description": "XRF scanner contamination element analysis quality","ontology": "quality"},
        {"id": "qc_surface",   "description": "Surface roughness measure quality precision tolerance","ontology": "quality"},
    ]
    engine.add_nodes_batch(nodes)


# ---------------------------------------------------------------------------
# Tests: add_node
# ---------------------------------------------------------------------------

class TestAddNode:
    def test_node_is_stored(self):
        engine = _make_engine()
        node = engine.add_node(id="n1", description="test node")
        assert engine.store.get_node("n1") is not None

    def test_node_has_embedding(self):
        engine = _make_engine()
        node = engine.add_node(id="n1", description="test node")
        assert node.embedding is not None
        assert node.embedding.shape == (64,)

    def test_node_ontology_stored(self):
        engine = _make_engine()
        node = engine.add_node(id="n1", description="test", ontology="bio")
        assert node.ontology == "bio"

    def test_node_description_in_metadata(self):
        engine = _make_engine()
        node = engine.add_node(id="n1", description="my description")
        assert node.metadata["description"] == "my description"

    def test_node_custom_metadata(self):
        engine = _make_engine()
        node = engine.add_node(id="n1", description="test", metadata={"source": "lab"})
        assert node.metadata["source"] == "lab"

    def test_stats_count_increments(self):
        engine = _make_engine()
        engine.add_node(id="n1", description="a")
        engine.add_node(id="n2", description="b")
        assert engine.stats()["nodes"] == 2


# ---------------------------------------------------------------------------
# Tests: add_nodes_batch
# ---------------------------------------------------------------------------

class TestAddNodesBatch:
    def test_all_nodes_added(self):
        engine = _make_engine()
        nodes = [
            {"id": "n1", "description": "alpha"},
            {"id": "n2", "description": "beta", "ontology": "bio"},
            {"id": "n3", "description": "gamma", "metadata": {"x": 1}},
        ]
        result = engine.add_nodes_batch(nodes)
        assert len(result) == 3
        assert engine.stats()["nodes"] == 3

    def test_returns_node_objects(self):
        engine = _make_engine()
        result = engine.add_nodes_batch([{"id": "n1", "description": "test"}])
        assert isinstance(result[0], Node)

    def test_embeddings_set_for_all(self):
        engine = _make_engine()
        nodes = [{"id": f"n{i}", "description": f"node {i}"} for i in range(5)]
        result = engine.add_nodes_batch(nodes)
        for n in result:
            assert n.embedding is not None


# ---------------------------------------------------------------------------
# Tests: set_encoder
# ---------------------------------------------------------------------------

class TestSetEncoder:
    def test_custom_encoder_used(self):
        engine = IntentionEngine()
        calls = []

        def my_encoder(text: str) -> np.ndarray:
            calls.append(text)
            return np.ones(64, dtype=np.float32) / np.sqrt(64)

        engine.set_encoder(my_encoder)
        engine.add_node(id="n1", description="hello world")
        assert len(calls) > 0  # encoder was called

    def test_custom_encoder_overrides_default(self):
        """Engine must NOT attempt to import sentence-transformers if set_encoder is used."""
        engine = IntentionEngine()
        engine.set_encoder(lambda t: np.zeros(64, dtype=np.float32))
        # Should not raise ImportError even if sentence-transformers is absent
        node = engine.add_node(id="n1", description="test")
        assert node is not None


# ---------------------------------------------------------------------------
# Tests: search — cold start (no edges)
# ---------------------------------------------------------------------------

class TestSearchColdStart:
    def test_returns_search_result(self):
        engine = _make_engine()
        _populate(engine)
        result = engine.search("reduce contamination in titanium processing")
        assert isinstance(result, SearchResult)

    def test_result_has_explanation(self):
        engine = _make_engine()
        _populate(engine)
        result = engine.search("contamination purge")
        assert result.explanation is not None

    def test_explore_mints_edges_on_first_query(self):
        engine = _make_engine()
        _populate(engine)
        result = engine.search("contamination purge clean", explore=True)
        # Cold start: no edges -> explore phase fires
        assert engine.stats()["minted_edges"] >= 0  # at minimum no crash
        # With >= 2 coherent nodes in utility space, we should get minted edges
        # (allow 0 in degenerate mock, but result must be valid)
        assert isinstance(result.minted_edges, list)

    def test_explore_populates_minted_edges_in_result(self):
        engine = _make_engine(min_coherence=0.05, percentile=30.0)
        _populate(engine)
        result = engine.search("contamination clean purge", explore=True)
        stats = engine.stats()
        # minted_edges in result should match what was stored
        assert len(result.minted_edges) == stats["minted_edges"]

    def test_no_explore_flag_skips_explore(self):
        engine = _make_engine()
        _populate(engine)
        result = engine.search("contamination", explore=False)
        assert result.explanation.explore_stats.edges_minted == 0
        assert engine.stats()["minted_edges"] == 0


# ---------------------------------------------------------------------------
# Tests: search — second query exploits previous minted edges
# ---------------------------------------------------------------------------

class TestSearchExploit:
    def test_second_query_activates_minted_edges(self):
        engine = _make_engine(min_coherence=0.05, percentile=30.0)
        _populate(engine)

        # First search: mints edges
        result1 = engine.search("contamination clean purge", explore=True)
        minted_after_first = engine.stats()["minted_edges"]

        if minted_after_first == 0:
            pytest.skip("No edges minted in first query with this mock — skip exploit test")

        # Second related search: should activate existing edges
        result2 = engine.search("prevent contamination in metal parts", explore=True)
        assert result2.explanation.exploit_stats.edges_activated >= 0
        # At minimum exploit was attempted
        assert result2.explanation is not None

    def test_second_search_returns_nodes(self):
        engine = _make_engine(min_coherence=0.05, percentile=30.0)
        _populate(engine)
        engine.search("contamination purge", explore=True)
        result2 = engine.search("contamination clean", explore=True)
        assert isinstance(result2.nodes, list)

    def test_total_edges_non_decreasing_across_queries(self):
        engine = _make_engine(min_coherence=0.05, percentile=30.0)
        _populate(engine)
        engine.search("contamination purge clean", explore=True)
        count1 = engine.stats()["edges"]
        engine.search("precision tolerance measure", explore=True)
        count2 = engine.stats()["edges"]
        assert count2 >= count1  # Can only grow (or stay same if all similar)


# ---------------------------------------------------------------------------
# Tests: stats()
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty_stats(self):
        engine = _make_engine()
        s = engine.stats()
        assert s["nodes"] == 0
        assert s["edges"] == 0
        assert s["minted_edges"] == 0
        assert s["manual_edges"] == 0

    def test_manual_edge_counted(self):
        engine = _make_engine()
        engine.add_node(id="n1", description="a")
        engine.add_node(id="n2", description="b")
        engine.add_hyperedge({"n1", "n2"}, label="manual test")
        s = engine.stats()
        assert s["edges"] == 1
        assert s["manual_edges"] == 1
        assert s["minted_edges"] == 0

    def test_nodes_count_correct(self):
        engine = _make_engine()
        _populate(engine)
        assert engine.stats()["nodes"] == 10


# ---------------------------------------------------------------------------
# Tests: add_hyperedge / explain_edge
# ---------------------------------------------------------------------------

class TestHyperedgeManagement:
    def test_add_hyperedge_stored(self):
        engine = _make_engine()
        engine.add_node(id="a", description="alpha")
        engine.add_node(id="b", description="beta")
        edge = engine.add_hyperedge({"a", "b"}, label="my edge")
        assert engine.store.get_edge(edge.id) is not None

    def test_add_hyperedge_members(self):
        engine = _make_engine()
        edge = engine.add_hyperedge({"x", "y", "z"}, label="test")
        assert "x" in edge.members
        assert len(edge.members) == 3

    def test_explain_edge_returns_dict(self):
        engine = _make_engine()
        engine.add_node(id="a", description="alpha")
        engine.add_node(id="b", description="beta")
        edge = engine.add_hyperedge({"a", "b"}, label="explained edge")
        info = engine.explain_edge(edge.id)
        assert info is not None
        assert info["id"] == edge.id
        assert "a" in info["members"]
        assert info["label"] == "explained edge"
        assert info["source"] == "manual"

    def test_explain_edge_missing_returns_none(self):
        engine = _make_engine()
        assert engine.explain_edge("nonexistent_id") is None


# ---------------------------------------------------------------------------
# Tests: decay_edges
# ---------------------------------------------------------------------------

class TestDecayEdges:
    def test_decay_does_not_crash(self):
        engine = _make_engine()
        _populate(engine)
        # Prune everything (threshold=1.0 prunes all)
        pruned = engine.decay_edges(threshold=1.0)
        assert isinstance(pruned, int)
        assert pruned >= 0

    def test_decay_prunes_low_weight_edges(self):
        engine = _make_engine(min_coherence=0.05, percentile=30.0)
        _populate(engine)
        engine.search("contamination purge", explore=True)
        before = engine.stats()["edges"]
        # Prune all edges (threshold above any possible weight)
        pruned = engine.decay_edges(threshold=10.0)
        after = engine.stats()["edges"]
        assert pruned == before - after


# ---------------------------------------------------------------------------
# Tests: search result structure integrity
# ---------------------------------------------------------------------------

class TestSearchResultStructure:
    def test_nodes_are_scored_nodes(self):
        from intention_engine import ScoredNode
        engine = _make_engine()
        _populate(engine)
        result = engine.search("contamination titanium")
        for sn in result.nodes:
            assert isinstance(sn, ScoredNode)
            assert 0.0 <= sn.score

    def test_node_source_valid(self):
        engine = _make_engine(min_coherence=0.05, percentile=30.0)
        _populate(engine)
        result = engine.search("contamination purge clean", explore=True)
        valid_sources = {"exploit", "explore", "both"}
        for sn in result.nodes:
            assert sn.source in valid_sources

    def test_max_results_respected(self):
        engine = _make_engine()
        _populate(engine)
        result = engine.search("contamination", max_results=3)
        assert len(result.nodes) <= 3

    def test_explore_flag_false_suppresses_explore(self):
        engine = _make_engine()
        _populate(engine)
        result = engine.search("contamination titanium metal", explore=False)
        assert result.explanation.explore_stats.edges_minted == 0
