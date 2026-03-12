"""Tests for all data models in intention_engine.models."""

import time

import numpy as np
import pytest

from intention_engine.models import (
    EngineConfig,
    ExploitStats,
    ExploreStats,
    Hyperedge,
    HyperedgeProvenance,
    Intention,
    Node,
    Predicate,
    ScoredNode,
    SearchExplanation,
    SearchResult,
    SearchScope,
    _make_id,
)


# ---------------------------------------------------------------------------
# _make_id
# ---------------------------------------------------------------------------

class TestMakeId:
    def test_prefix_included(self):
        node_id = _make_id("node")
        assert node_id.startswith("node_")

    def test_hex_suffix_length(self):
        node_id = _make_id("he")
        # format: "he_<12 hex chars>"
        parts = node_id.split("_", 1)
        assert len(parts) == 2
        assert len(parts[1]) == 12

    def test_uniqueness(self):
        ids = {_make_id("x") for _ in range(1000)}
        assert len(ids) == 1000

    def test_different_prefixes(self):
        a = _make_id("node")
        b = _make_id("he")
        assert a.startswith("node_")
        assert b.startswith("he_")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TestNode:
    def test_minimal_creation(self):
        n = Node(id="n1")
        assert n.id == "n1"
        assert n.features is None
        assert n.metadata == {}
        assert n.ontology == "default"
        assert n.embedding is None

    def test_created_at_auto(self):
        before = time.time()
        n = Node(id="n2")
        after = time.time()
        assert before <= n.created_at <= after

    def test_with_features(self):
        feats = np.array([1.0, 2.0, 3.0])
        n = Node(id="n3", features=feats)
        assert np.array_equal(n.features, feats)

    def test_with_embedding(self):
        emb = np.random.rand(384)
        n = Node(id="n4", embedding=emb)
        assert n.embedding.shape == (384,)

    def test_metadata_is_independent(self):
        n1 = Node(id="n5")
        n2 = Node(id="n6")
        n1.metadata["key"] = "value"
        assert "key" not in n2.metadata

    def test_custom_ontology(self):
        n = Node(id="n7", ontology="bio")
        assert n.ontology == "bio"


# ---------------------------------------------------------------------------
# HyperedgeProvenance
# ---------------------------------------------------------------------------

class TestHyperedgeProvenance:
    def test_manual_source(self):
        p = HyperedgeProvenance(source="manual")
        assert p.source == "manual"
        assert p.intention is None
        assert p.predicates is None
        assert p.coherence_method == "geometric"
        assert p.parent_edges is None

    def test_minted_source(self):
        p = HyperedgeProvenance(
            source="minted",
            intention="find proteins",
            predicates=["binds_to", "inhibits"],
        )
        assert p.source == "minted"
        assert p.intention == "find proteins"
        assert p.predicates == ["binds_to", "inhibits"]

    def test_extracted_source(self):
        p = HyperedgeProvenance(source="extracted")
        assert p.source == "extracted"


# ---------------------------------------------------------------------------
# Hyperedge
# ---------------------------------------------------------------------------

class TestHyperedge:
    def _make_edge(self, members=None):
        return Hyperedge(
            id=_make_id("he"),
            members=frozenset(members or ["n1", "n2"]),
        )

    def test_defaults(self):
        e = self._make_edge()
        assert e.label == ""
        assert e.coherence_score == 0.0
        assert e.utility_context == ""
        assert e.access_count == 0
        assert e.weight == 1.0
        assert isinstance(e.members, frozenset)

    def test_provenance_default(self):
        e = self._make_edge()
        assert e.provenance.source == "manual"

    def test_created_at_auto(self):
        before = time.time()
        e = self._make_edge()
        after = time.time()
        assert before <= e.created_at <= after

    def test_last_accessed_auto(self):
        before = time.time()
        e = self._make_edge()
        after = time.time()
        assert before <= e.last_accessed <= after

    def test_members_frozenset(self):
        e = Hyperedge(id="e1", members=frozenset(["a", "b", "c"]))
        assert "a" in e.members
        assert len(e.members) == 3

    def test_custom_provenance(self):
        prov = HyperedgeProvenance(source="minted", intention="test intent")
        e = Hyperedge(id="e2", members=frozenset(["x"]), provenance=prov)
        assert e.provenance.source == "minted"
        assert e.provenance.intention == "test intent"


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------

class TestPredicate:
    def test_defaults(self):
        p = Predicate(text="binds_to")
        assert p.text == "binds_to"
        assert p.embedding is None
        assert p.weight == 1.0

    def test_with_embedding(self):
        emb = np.ones(128)
        p = Predicate(text="inhibits", embedding=emb, weight=0.5)
        assert p.weight == 0.5
        assert np.array_equal(p.embedding, emb)


# ---------------------------------------------------------------------------
# SearchScope
# ---------------------------------------------------------------------------

class TestSearchScope:
    def test_defaults(self):
        s = SearchScope()
        assert s.ontologies is None
        assert s.max_depth == 2
        assert s.min_coherence == 0.3
        assert s.explore_budget == 100

    def test_custom(self):
        s = SearchScope(ontologies={"bio", "chem"}, max_depth=5)
        assert "bio" in s.ontologies
        assert s.max_depth == 5


# ---------------------------------------------------------------------------
# Intention
# ---------------------------------------------------------------------------

class TestIntention:
    def test_minimal(self):
        i = Intention(raw="find proteins that inhibit kinase")
        assert i.raw == "find proteins that inhibit kinase"
        assert i.predicates == []
        assert i.embedding is None
        assert i.scope is None

    def test_with_predicates(self):
        preds = [Predicate(text="inhibit"), Predicate(text="kinase")]
        i = Intention(raw="query", predicates=preds)
        assert len(i.predicates) == 2

    def test_with_scope(self):
        scope = SearchScope(max_depth=3)
        i = Intention(raw="query", scope=scope)
        assert i.scope.max_depth == 3


# ---------------------------------------------------------------------------
# ScoredNode
# ---------------------------------------------------------------------------

class TestScoredNode:
    def test_defaults(self):
        node = Node(id="n1")
        sn = ScoredNode(node=node, score=0.95)
        assert sn.score == 0.95
        assert sn.source == "exploit"
        assert sn.via_edges == []

    def test_explore_source(self):
        node = Node(id="n2")
        sn = ScoredNode(node=node, score=0.7, source="explore", via_edges=["e1", "e2"])
        assert sn.source == "explore"
        assert len(sn.via_edges) == 2

    def test_both_source(self):
        node = Node(id="n3")
        sn = ScoredNode(node=node, score=0.5, source="both")
        assert sn.source == "both"


# ---------------------------------------------------------------------------
# ExploitStats / ExploreStats
# ---------------------------------------------------------------------------

class TestStats:
    def test_exploit_defaults(self):
        s = ExploitStats()
        assert s.edges_scored == 0
        assert s.edges_activated == 0
        assert s.nodes_reached == 0
        assert s.elapsed_ms == 0.0

    def test_explore_defaults(self):
        s = ExploreStats()
        assert s.nodes_projected == 0
        assert s.clusters_found == 0
        assert s.candidates_evaluated == 0
        assert s.edges_minted == 0
        assert s.elapsed_ms == 0.0

    def test_exploit_custom(self):
        s = ExploitStats(edges_scored=10, edges_activated=5, nodes_reached=20, elapsed_ms=12.5)
        assert s.edges_scored == 10
        assert s.elapsed_ms == 12.5


# ---------------------------------------------------------------------------
# SearchExplanation
# ---------------------------------------------------------------------------

class TestSearchExplanation:
    def test_defaults(self):
        intention = Intention(raw="test")
        ex = SearchExplanation(intention=intention)
        assert ex.exploit_stats.edges_scored == 0
        assert ex.explore_stats.edges_minted == 0

    def test_custom_stats(self):
        intention = Intention(raw="test")
        exploit = ExploitStats(edges_scored=5)
        explore = ExploreStats(edges_minted=2)
        ex = SearchExplanation(intention=intention, exploit_stats=exploit, explore_stats=explore)
        assert ex.exploit_stats.edges_scored == 5
        assert ex.explore_stats.edges_minted == 2


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_defaults(self):
        sr = SearchResult()
        assert sr.nodes == []
        assert sr.exploited_edges == []
        assert sr.minted_edges == []
        assert sr.explanation is None

    def test_with_data(self):
        node = Node(id="n1")
        sn = ScoredNode(node=node, score=0.9)
        edge = Hyperedge(id="e1", members=frozenset(["n1"]))
        sr = SearchResult(
            nodes=[sn],
            exploited_edges=[edge],
            minted_edges=[],
        )
        assert len(sr.nodes) == 1
        assert len(sr.exploited_edges) == 1


# ---------------------------------------------------------------------------
# EngineConfig
# ---------------------------------------------------------------------------

class TestEngineConfig:
    def test_defaults(self):
        cfg = EngineConfig()
        assert cfg.encoder_model == "all-MiniLM-L6-v2"
        assert cfg.embedding_dim == 384
        assert cfg.max_exploit_depth == 2
        assert cfg.exploit_weight == 0.7
        assert cfg.explore_enabled is True
        assert cfg.explore_budget == 100
        assert cfg.utility_threshold_percentile == 80.0
        assert cfg.min_coherence == 0.3
        assert cfg.coherence_weights == (0.5, 0.4, 0.1)
        assert cfg.novelty_threshold == 0.8
        assert cfg.min_edge_size == 2
        assert cfg.max_edge_size == 50
        assert cfg.decay_half_life_days == 30.0
        assert cfg.prune_threshold == 0.01
        assert cfg.use_llm_decomposition is False
        assert cfg.llm_model is None
        assert cfg.max_mints_per_query == 10

    def test_custom(self):
        cfg = EngineConfig(encoder_model="custom-model", embedding_dim=768, explore_enabled=False)
        assert cfg.encoder_model == "custom-model"
        assert cfg.embedding_dim == 768
        assert cfg.explore_enabled is False

    def test_coherence_weights_tuple(self):
        cfg = EngineConfig(coherence_weights=(0.6, 0.3, 0.1))
        assert isinstance(cfg.coherence_weights, tuple)
        assert len(cfg.coherence_weights) == 3
        assert abs(sum(cfg.coherence_weights) - 1.0) < 1e-9
