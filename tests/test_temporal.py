"""Tests for temporal features: events, temporal embedding, hypergraph time-scoping, engine temporal API."""

import json
import os
import time

import numpy as np
import pytest

from intention_engine.events import EventLog, GraphEvent
from intention_engine.temporal import is_edge_valid_at, temporal_embedding, temporal_similarity
from intention_engine.models import (
    Hyperedge,
    HyperedgeProvenance,
    IntentionEvent,
    TemporalDiff,
    TemporalQuery,
)
from intention_engine.hypergraph import HypergraphStore
from intention_engine.engine import IntentionEngine
from intention_engine.encoder import HashEncoder
from intention_engine.models import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_with_nodes(*descriptions: str) -> IntentionEngine:
    """Create an engine with HashEncoder and the given nodes (ids: n0, n1, ...)."""
    engine = IntentionEngine()
    engine.set_encoder(HashEncoder(dim=64))
    for i, desc in enumerate(descriptions):
        engine.add_node(id=f"n{i}", description=desc)
    return engine


def _make_store_with_edge(
    members: frozenset[str] | None = None,
    valid_from: float = 0.0,
    valid_until: float | None = None,
    label: str = "test-edge",
) -> tuple[HypergraphStore, str]:
    """Create a HypergraphStore containing one edge with the given temporal fields.

    Returns (store, edge_id).
    """
    from intention_engine.models import Node, _make_id

    store = HypergraphStore()
    if members is None:
        members = frozenset(["n1", "n2"])
    for nid in members:
        store.add_node(Node(id=nid))
    edge = Hyperedge(
        id=_make_id("he"),
        members=members,
        label=label,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    store.add_hyperedge(edge)
    return store, edge.id


# ===========================================================================
# GraphEvent
# ===========================================================================

class TestGraphEvent:
    def test_event_creation_defaults(self):
        ev = GraphEvent(event_type="node_added")
        assert ev.event_type == "node_added"
        assert ev.entity_id == ""
        assert ev.data == {}
        assert ev.intention == ""
        assert isinstance(ev.timestamp, float)

    def test_event_with_all_fields(self):
        ev = GraphEvent(
            event_type="edge_minted",
            timestamp=1000.0,
            entity_id="he_abc",
            data={"members": ["n1", "n2"]},
            intention="find proteins",
        )
        assert ev.event_type == "edge_minted"
        assert ev.timestamp == 1000.0
        assert ev.entity_id == "he_abc"
        assert ev.data == {"members": ["n1", "n2"]}
        assert ev.intention == "find proteins"

    def test_event_timestamp_auto_set(self):
        before = time.time()
        ev = GraphEvent(event_type="search_executed")
        after = time.time()
        assert before <= ev.timestamp <= after


# ===========================================================================
# EventLog
# ===========================================================================

class TestEventLog:
    def test_append_and_len(self):
        log = EventLog()
        assert len(log) == 0
        log.append(GraphEvent(event_type="node_added", entity_id="n1"))
        assert len(log) == 1
        log.append(GraphEvent(event_type="node_added", entity_id="n2"))
        assert len(log) == 2

    def test_events_for_entity(self):
        log = EventLog()
        log.append(GraphEvent(event_type="node_added", entity_id="n1"))
        log.append(GraphEvent(event_type="edge_minted", entity_id="he_1"))
        log.append(GraphEvent(event_type="edge_reinforced", entity_id="he_1"))
        result = log.events_for("he_1")
        assert len(result) == 2
        assert all(e.entity_id == "he_1" for e in result)

    def test_events_in_range(self):
        log = EventLog()
        log.append(GraphEvent(event_type="node_added", timestamp=10.0, entity_id="n1"))
        log.append(GraphEvent(event_type="node_added", timestamp=20.0, entity_id="n2"))
        log.append(GraphEvent(event_type="node_added", timestamp=30.0, entity_id="n3"))
        result = log.events_in_range(15.0, 25.0)
        assert len(result) == 1
        assert result[0].entity_id == "n2"

    def test_events_in_range_inclusive(self):
        log = EventLog()
        log.append(GraphEvent(event_type="node_added", timestamp=10.0, entity_id="n1"))
        log.append(GraphEvent(event_type="node_added", timestamp=20.0, entity_id="n2"))
        # Both endpoints are inclusive
        result = log.events_in_range(10.0, 20.0)
        assert len(result) == 2

    def test_events_by_type(self):
        log = EventLog()
        log.append(GraphEvent(event_type="node_added", entity_id="n1"))
        log.append(GraphEvent(event_type="edge_minted", entity_id="he_1"))
        log.append(GraphEvent(event_type="node_added", entity_id="n2"))
        result = log.events_by_type("node_added")
        assert len(result) == 2
        assert all(e.event_type == "node_added" for e in result)

    def test_save_and_load(self, tmp_path):
        log = EventLog()
        log.append(GraphEvent(event_type="edge_minted", timestamp=100.0, entity_id="he_1",
                              data={"label": "test"}, intention="intent"))
        log.append(GraphEvent(event_type="edge_closed", timestamp=200.0, entity_id="he_1"))
        path = str(tmp_path / "events.jsonl")
        log.save(path)

        log2 = EventLog()
        log2.load(path)
        assert len(log2) == 2
        events = list(log2)
        assert events[0].event_type == "edge_minted"
        assert events[0].timestamp == 100.0
        assert events[0].entity_id == "he_1"
        assert events[0].data == {"label": "test"}
        assert events[0].intention == "intent"
        assert events[1].event_type == "edge_closed"

    def test_load_nonexistent_file(self, tmp_path):
        log = EventLog()
        log.load(str(tmp_path / "does_not_exist.jsonl"))
        assert len(log) == 0

    def test_load_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        with open(path, "w") as f:
            f.write("")
        log = EventLog()
        log.load(path)
        assert len(log) == 0

    def test_iteration(self):
        log = EventLog()
        log.append(GraphEvent(event_type="node_added", entity_id="n1"))
        log.append(GraphEvent(event_type="edge_minted", entity_id="he_1"))
        events = list(log)
        assert len(events) == 2
        assert events[0].entity_id == "n1"
        assert events[1].entity_id == "he_1"

    def test_multiple_entities(self):
        log = EventLog()
        for i in range(5):
            log.append(GraphEvent(event_type="node_added", entity_id=f"n{i}"))
        assert len(log) == 5
        assert len(log.events_for("n0")) == 1
        assert len(log.events_for("n4")) == 1
        assert len(log.events_for("nonexistent")) == 0

    def test_save_preserves_order(self, tmp_path):
        log = EventLog()
        for i in range(10):
            log.append(GraphEvent(event_type="node_added", timestamp=float(i), entity_id=f"n{i}"))
        path = str(tmp_path / "ordered.jsonl")
        log.save(path)

        log2 = EventLog()
        log2.load(path)
        timestamps = [e.timestamp for e in log2]
        assert timestamps == list(range(10))

    def test_auto_flush_to_file(self, tmp_path):
        path = str(tmp_path / "auto.jsonl")
        log = EventLog(file_path=path)
        log.append(GraphEvent(event_type="node_added", entity_id="n1"))
        log.append(GraphEvent(event_type="edge_minted", entity_id="he_1"))
        # File should have 2 lines
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 2

    def test_load_file_with_blank_lines(self, tmp_path):
        path = str(tmp_path / "blanks.jsonl")
        with open(path, "w") as f:
            f.write('{"event_type": "node_added", "timestamp": 1.0, "entity_id": "n1", "data": {}, "intention": ""}\n')
            f.write("\n")
            f.write('{"event_type": "edge_minted", "timestamp": 2.0, "entity_id": "he_1", "data": {}, "intention": ""}\n')
            f.write("\n")
        log = EventLog()
        log.load(path)
        assert len(log) == 2


# ===========================================================================
# Temporal Embedding
# ===========================================================================

class TestTemporalEmbedding:
    def test_output_shape(self):
        text_emb = np.ones(64, dtype=np.float64)
        result = temporal_embedding(text_emb, time.time(), dim=16)
        assert result.shape == (64 + 16,)

    def test_different_timestamps_different_embeddings(self):
        text_emb = np.ones(32, dtype=np.float64)
        t1 = time.time()
        t2 = t1 - 86400 * 30  # 30 days ago
        emb1 = temporal_embedding(text_emb, t1, dim=16)
        emb2 = temporal_embedding(text_emb, t2, dim=16)
        # The text portion is the same but the temporal portion differs
        assert not np.allclose(emb1, emb2)
        # Text portions should match exactly
        assert np.array_equal(emb1[:32], emb2[:32])

    def test_same_text_same_time_deterministic(self):
        text_emb = np.ones(32, dtype=np.float64)
        t = 1700000000.0
        emb1 = temporal_embedding(text_emb, t, dim=16)
        emb2 = temporal_embedding(text_emb, t, dim=16)
        assert np.array_equal(emb1, emb2)

    def test_custom_dim(self):
        text_emb = np.zeros(10, dtype=np.float64)
        result = temporal_embedding(text_emb, time.time(), dim=32)
        assert result.shape == (10 + 32,)

    def test_dim_smaller_than_features(self):
        # Fourier features: 4 freqs * 2 (sin+cos) = 8, plus 2 recency = 10 total
        # With dim=4, should truncate to 4
        text_emb = np.zeros(8, dtype=np.float64)
        result = temporal_embedding(text_emb, time.time(), dim=4)
        assert result.shape == (8 + 4,)

    def test_recency_decay(self):
        text_emb = np.zeros(8, dtype=np.float64)
        now = time.time()
        recent = temporal_embedding(text_emb, now, dim=16)
        old = temporal_embedding(text_emb, now - 31_536_000 * 5, dim=16)  # 5 years ago
        # Recency feature (index 8 in temporal part = feature index 8)
        # Features: 8 fourier + recency_exp + age_years + padding
        # The exponential recency for 'now' should be close to 1.0
        # The exponential recency for 5 years ago should be near 0
        recent_temporal = recent[8:]  # temporal portion
        old_temporal = old[8:]
        # Feature index 8 = exp(-age_years): should be higher for recent
        assert recent_temporal[8] > old_temporal[8]

    def test_zero_dim_produces_text_only(self):
        text_emb = np.array([1.0, 2.0, 3.0])
        result = temporal_embedding(text_emb, time.time(), dim=0)
        assert result.shape == (3,)
        assert np.array_equal(result, text_emb)


# ===========================================================================
# is_edge_valid_at
# ===========================================================================

class TestIsEdgeValidAt:
    def test_valid_open_edge(self):
        # Edge with valid_from=100, valid_until=None (open) at t=200
        assert is_edge_valid_at(100.0, None, 200.0) is True

    def test_valid_closed_edge_within_interval(self):
        assert is_edge_valid_at(100.0, 300.0, 200.0) is True

    def test_invalid_before_valid_from(self):
        assert is_edge_valid_at(100.0, None, 50.0) is False

    def test_invalid_after_valid_until(self):
        # valid_until=300, query_time=300 -> invalid (>= boundary)
        assert is_edge_valid_at(100.0, 300.0, 400.0) is False

    def test_edge_at_exact_valid_from(self):
        # valid_from <= query_time -> valid
        assert is_edge_valid_at(100.0, None, 100.0) is True

    def test_edge_at_exact_valid_until(self):
        # query_time >= valid_until -> invalid
        assert is_edge_valid_at(100.0, 200.0, 200.0) is False

    def test_closed_edge_at_boundary_minus_epsilon(self):
        assert is_edge_valid_at(100.0, 200.0, 199.999) is True

    def test_open_edge_far_future(self):
        assert is_edge_valid_at(100.0, None, 1e12) is True


# ===========================================================================
# temporal_similarity
# ===========================================================================

class TestTemporalSimilarity:
    def test_same_time_returns_one(self):
        t = time.time()
        assert temporal_similarity(t, t) == pytest.approx(1.0)

    def test_distant_times_small_similarity(self):
        t1 = time.time()
        t2 = t1 - 86400 * 365  # 1 year apart
        sim = temporal_similarity(t1, t2)
        assert sim < 0.01  # very small for 1 year with 1-week half-life

    def test_half_life_effect(self):
        t1 = time.time()
        half_life = 604800.0  # 1 week
        t2 = t1 - half_life  # exactly one half-life apart
        sim = temporal_similarity(t1, t2, half_life=half_life)
        assert sim == pytest.approx(0.5, abs=0.01)

    def test_symmetric(self):
        t1 = time.time()
        t2 = t1 - 3600  # 1 hour apart
        assert temporal_similarity(t1, t2) == pytest.approx(temporal_similarity(t2, t1))

    def test_custom_half_life(self):
        t1 = time.time()
        t2 = t1 - 3600  # 1 hour apart
        # Shorter half-life -> faster decay -> lower similarity
        sim_short = temporal_similarity(t1, t2, half_life=1800)
        sim_long = temporal_similarity(t1, t2, half_life=86400)
        assert sim_short < sim_long

    def test_returns_positive(self):
        t1 = time.time()
        t2 = t1 - 86400 * 30  # 30 days apart (not so extreme it underflows)
        sim = temporal_similarity(t1, t2)
        assert sim > 0.0


# ===========================================================================
# HypergraphStore temporal
# ===========================================================================

class TestHypergraphTemporal:
    def test_mint_sets_valid_from(self):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        before = time.time()
        edge = store.mint_hyperedge(
            members=frozenset(["n1", "n2"]),
            label="test",
            coherence=0.8,
            intention="testing",
        )
        after = time.time()
        assert before <= edge.valid_from <= after
        assert edge.valid_until is None

    def test_mint_with_event_log(self):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.event_log = EventLog()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        edge = store.mint_hyperedge(
            members=frozenset(["n1", "n2"]),
            label="test",
            coherence=0.8,
            intention="testing",
        )
        assert len(store.event_log) == 1
        ev = list(store.event_log)[0]
        assert ev.event_type == "edge_minted"
        assert ev.entity_id == edge.id
        assert ev.intention == "testing"

    def test_reinforce_with_event_log(self):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.event_log = EventLog()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        edge = store.mint_hyperedge(
            members=frozenset(["n1", "n2"]),
            label="test",
            coherence=0.8,
        )
        store.reinforce_edge(edge.id)
        assert len(store.event_log) == 2
        events = list(store.event_log)
        assert events[1].event_type == "edge_reinforced"
        assert events[1].entity_id == edge.id

    def test_close_edge(self):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        edge = store.mint_hyperedge(
            members=frozenset(["n1", "n2"]),
            label="test",
            coherence=0.8,
        )
        eid = edge.id
        assert store.get_edge(eid) is not None
        store.close_edge(eid)
        # Edge removed from live layer
        assert store.get_edge(eid) is None
        # Edge archived in closed layer
        assert eid in store._closed_edges

    def test_close_edge_sets_valid_until(self):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        edge = store.mint_hyperedge(
            members=frozenset(["n1", "n2"]),
            label="test",
            coherence=0.8,
        )
        eid = edge.id
        before = time.time()
        store.close_edge(eid)
        after = time.time()
        closed = store._closed_edges[eid]
        assert closed.valid_until is not None
        assert before <= closed.valid_until <= after

    def test_close_edge_removes_from_live(self):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        edge = store.mint_hyperedge(
            members=frozenset(["n1", "n2"]),
            label="test",
            coherence=0.8,
        )
        assert store.num_edges == 1
        store.close_edge(edge.id)
        assert store.num_edges == 0

    def test_close_edge_with_event_log(self):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.event_log = EventLog()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        edge = store.mint_hyperedge(
            members=frozenset(["n1", "n2"]),
            label="test",
            coherence=0.8,
        )
        store.close_edge(edge.id)
        events = list(store.event_log)
        close_events = [e for e in events if e.event_type == "edge_closed"]
        assert len(close_events) == 1
        assert close_events[0].entity_id == edge.id

    def test_valid_edges_at(self):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        store.add_node(Node(id="n3"))
        # Edge 1: valid from 100 to 300
        e1 = Hyperedge(id="e1", members=frozenset(["n1", "n2"]), valid_from=100.0, valid_until=300.0)
        # Edge 2: valid from 200, still open
        e2 = Hyperedge(id="e2", members=frozenset(["n2", "n3"]), valid_from=200.0)
        store.add_hyperedge(e1)
        store.add_hyperedge(e2)

        # At t=150: only e1 valid
        valid_150 = store.valid_edges_at(150.0)
        assert "e1" in valid_150
        assert "e2" not in valid_150

        # At t=250: both valid
        valid_250 = store.valid_edges_at(250.0)
        assert "e1" in valid_250
        assert "e2" in valid_250

    def test_valid_edges_at_excludes_closed(self):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        # Use explicit timestamps to avoid race conditions from fast execution
        edge = Hyperedge(
            id="e_closed_test",
            members=frozenset(["n1", "n2"]),
            label="test",
            valid_from=1000.0,
            valid_until=2000.0,
        )
        edge.intention_history.append(IntentionEvent(
            timestamp=1000.0, intention="test", action="minted", score=0.8,
        ))
        # Put directly in closed archive (simulating a close)
        store._closed_edges[edge.id] = edge

        # At time between valid_from and valid_until: should be valid
        valid = store.valid_edges_at(1500.0)
        assert edge.id in valid

        # After valid_until: should not be valid
        valid_after = store.valid_edges_at(2500.0)
        assert edge.id not in valid_after

        # Before valid_from: should not be valid
        valid_before = store.valid_edges_at(500.0)
        assert edge.id not in valid_before

    def test_valid_edges_at_includes_open(self):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        edge = store.mint_hyperedge(
            members=frozenset(["n1", "n2"]),
            label="test",
            coherence=0.8,
        )
        # Open edge should be valid at any time after valid_from
        future = edge.valid_from + 86400 * 365 * 10
        valid = store.valid_edges_at(future)
        assert edge.id in valid

    def test_incidence_matrix_at(self):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        store.add_node(Node(id="n3"))
        # Edge only valid 100-200
        e1 = Hyperedge(id="e1", members=frozenset(["n1", "n2"]), valid_from=100.0, valid_until=200.0)
        # Edge valid from 150, open
        e2 = Hyperedge(id="e2", members=frozenset(["n2", "n3"]), valid_from=150.0)
        store.add_hyperedge(e1)
        store.add_hyperedge(e2)

        # At t=120: only e1 valid -> matrix has 1 edge column
        mat_120 = store.incidence_matrix_at(120.0)
        assert mat_120.shape[0] == 3  # 3 nodes
        assert mat_120.shape[1] == 1  # 1 valid edge

        # At t=175: both valid -> matrix has 2 edge columns
        mat_175 = store.incidence_matrix_at(175.0)
        assert mat_175.shape[0] == 3
        assert mat_175.shape[1] == 2

        # At t=250: only e2 valid -> 1 edge column
        mat_250 = store.incidence_matrix_at(250.0)
        assert mat_250.shape[0] == 3
        assert mat_250.shape[1] == 1

    def test_decay_closes_instead_of_deleting(self):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        edge = store.mint_hyperedge(
            members=frozenset(["n1", "n2"]),
            label="test",
            coherence=0.001,  # very low coherence
        )
        eid = edge.id
        # Set last_accessed far in the past to trigger decay
        edge.last_accessed = time.time() - 86400 * 365
        closed = store.decay_edges(half_life_days=30.0, prune_threshold=0.01)
        assert closed >= 1
        # Edge should be in closed archive, not deleted
        assert eid in store._closed_edges
        assert store.get_edge(eid) is None  # not in live

    def test_save_load_preserves_temporal_fields(self, tmp_path):
        from intention_engine.models import Node
        store = HypergraphStore()
        store.add_node(Node(id="n1"))
        store.add_node(Node(id="n2"))
        edge = store.mint_hyperedge(
            members=frozenset(["n1", "n2"]),
            label="temporal-test",
            coherence=0.9,
            intention="persist test",
        )
        eid = edge.id
        original_valid_from = edge.valid_from

        # Close the edge so it has valid_until
        store.close_edge(eid)
        closed = store._closed_edges[eid]
        original_valid_until = closed.valid_until

        path = str(tmp_path / "graph")
        store.save(path)

        store2 = HypergraphStore()
        store2.load(path)
        # Closed edge should be in _closed_edges
        assert eid in store2._closed_edges
        loaded = store2._closed_edges[eid]
        assert loaded.valid_from == pytest.approx(original_valid_from, abs=0.001)
        assert loaded.valid_until == pytest.approx(original_valid_until, abs=0.001)
        assert len(loaded.intention_history) >= 1

    def test_close_nonexistent_edge_is_noop(self):
        store = HypergraphStore()
        # Should not raise
        store.close_edge("nonexistent_edge")


# ===========================================================================
# Engine temporal
# ===========================================================================

class TestEngineTemporal:
    def test_enable_temporal(self):
        engine = _make_engine_with_nodes("node a", "node b")
        assert engine._temporal_enabled is False
        assert engine._event_log is None
        engine.enable_temporal()
        assert engine._temporal_enabled is True
        assert engine._event_log is not None
        assert engine.store.event_log is not None

    def test_temporal_diff_empty(self):
        engine = _make_engine_with_nodes()
        engine.enable_temporal()
        diff = engine.temporal_diff(0.0, time.time())
        assert isinstance(diff, TemporalDiff)
        assert diff.nodes_added == []
        assert diff.edges_minted == []
        assert diff.edges_closed == []
        assert diff.edges_reinforced == []
        assert diff.searches_executed == 0

    def test_temporal_diff_without_enable(self):
        engine = _make_engine_with_nodes()
        diff = engine.temporal_diff(0.0, time.time())
        assert isinstance(diff, TemporalDiff)
        # Should return empty diff when temporal not enabled
        assert diff.edges_minted == []

    def test_temporal_diff_with_events(self):
        engine = _make_engine_with_nodes("alpha", "beta", "gamma")
        engine.enable_temporal()
        t1 = time.time()
        # Mint an edge via the store
        edge = engine.store.mint_hyperedge(
            members=frozenset(["n0", "n1"]),
            label="test",
            coherence=0.8,
            intention="test intent",
        )
        # Reinforce it
        engine.store.reinforce_edge(edge.id)
        # Close it
        engine.store.close_edge(edge.id)
        t2 = time.time()

        diff = engine.temporal_diff(t1, t2)
        assert edge.id in diff.edges_minted
        assert edge.id in diff.edges_reinforced
        assert edge.id in diff.edges_closed

    def test_graph_at(self):
        engine = _make_engine_with_nodes("alpha", "beta")
        engine.enable_temporal()
        edge = engine.store.mint_hyperedge(
            members=frozenset(["n0", "n1"]),
            label="test",
            coherence=0.8,
        )
        mint_time = edge.valid_from
        # Before mint: no edges
        stats_before = engine.graph_at(mint_time - 1.0)
        assert stats_before["edges"] == 0
        # After mint: 1 edge
        stats_after = engine.graph_at(mint_time + 0.001)
        assert stats_after["edges"] == 1
        assert stats_after["nodes"] == 2

    def test_edge_history(self):
        engine = _make_engine_with_nodes("alpha", "beta")
        engine.enable_temporal()
        edge = engine.store.mint_hyperedge(
            members=frozenset(["n0", "n1"]),
            label="test",
            coherence=0.8,
            intention="creation",
        )
        history = engine.edge_history(edge.id)
        assert len(history) >= 1
        assert history[0]["action"] == "minted"
        assert history[0]["intention"] == "creation"

    def test_edge_history_after_close(self):
        engine = _make_engine_with_nodes("alpha", "beta")
        engine.enable_temporal()
        edge = engine.store.mint_hyperedge(
            members=frozenset(["n0", "n1"]),
            label="test",
            coherence=0.8,
            intention="creation",
        )
        eid = edge.id
        engine.store.close_edge(eid)
        # History should still be available via closed edges
        history = engine.edge_history(eid)
        assert len(history) >= 2
        actions = [h["action"] for h in history]
        assert "minted" in actions
        assert "closed" in actions

    def test_edge_history_not_found(self):
        engine = _make_engine_with_nodes()
        history = engine.edge_history("nonexistent")
        assert history == []

    def test_search_with_valid_at(self):
        engine = _make_engine_with_nodes("machine learning", "deep learning", "neural networks")
        # Add edge manually with specific valid_from
        e = engine.add_hyperedge({"n0", "n1"}, label="ML group")
        # Set valid_from to past
        edge_obj = engine.store.get_edge(e.id)
        edge_obj.valid_from = 1000.0
        edge_obj.valid_until = 2000.0

        # Search with valid_at within the interval
        result = engine.search("machine learning", valid_at=1500.0, explore=False)
        # Should work without error
        assert isinstance(result, SearchResult)

        # Search with valid_at outside the interval
        result2 = engine.search("machine learning", valid_at=3000.0, explore=False)
        assert isinstance(result2, SearchResult)

    def test_save_load_event_log(self, tmp_path):
        engine = _make_engine_with_nodes("alpha", "beta")
        engine.enable_temporal()
        edge = engine.store.mint_hyperedge(
            members=frozenset(["n0", "n1"]),
            label="test",
            coherence=0.8,
            intention="persist",
        )
        engine.store.reinforce_edge(edge.id)
        path = str(tmp_path / "graph")
        engine.save(path)

        engine2 = IntentionEngine()
        engine2.set_encoder(HashEncoder(dim=64))
        engine2.load(path)
        # Temporal should have been auto-enabled on load
        assert engine2._temporal_enabled is True
        assert engine2._event_log is not None
        assert len(engine2._event_log) >= 2

    def test_temporal_search_excludes_closed_edges(self):
        engine = _make_engine_with_nodes("machine learning", "deep learning")
        engine.enable_temporal()
        edge = engine.store.mint_hyperedge(
            members=frozenset(["n0", "n1"]),
            label="ML",
            coherence=0.8,
        )
        # Close the edge
        engine.store.close_edge(edge.id)
        # Search without valid_at: closed edges are not in live layer
        result = engine.search("machine learning", explore=False)
        # The exploited_edges should not contain the closed edge
        exploited_ids = [e.id for e in result.exploited_edges]
        assert edge.id not in exploited_ids


# ===========================================================================
# TemporalQuery and TemporalDiff models
# ===========================================================================

class TestTemporalModels:
    def test_temporal_query_defaults(self):
        tq = TemporalQuery()
        assert tq.as_of is None
        assert tq.valid_at is None
        assert tq.time_range is None

    def test_temporal_query_with_values(self):
        tq = TemporalQuery(as_of=100.0, valid_at=200.0, time_range=(50.0, 300.0))
        assert tq.as_of == 100.0
        assert tq.valid_at == 200.0
        assert tq.time_range == (50.0, 300.0)

    def test_temporal_diff_defaults(self):
        td = TemporalDiff(t1=10.0, t2=20.0)
        assert td.t1 == 10.0
        assert td.t2 == 20.0
        assert td.nodes_added == []
        assert td.nodes_removed == []
        assert td.edges_minted == []
        assert td.edges_closed == []
        assert td.edges_reinforced == []
        assert td.searches_executed == 0

    def test_intention_event_fields(self):
        ie = IntentionEvent(
            timestamp=100.0,
            intention="test",
            action="minted",
            score=0.9,
        )
        assert ie.timestamp == 100.0
        assert ie.intention == "test"
        assert ie.action == "minted"
        assert ie.score == 0.9
