"""Tests for HypergraphStore in intention_engine.hypergraph."""

import os
import tempfile
import time

import numpy as np
import pytest
import scipy.sparse as sp

from intention_engine.hypergraph import HypergraphStore
from intention_engine.models import Hyperedge, HyperedgeProvenance, Node, _make_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_node(nid: str, ontology: str = "default") -> Node:
    return Node(id=nid, ontology=ontology)


def make_edge(eid: str, members: list[str], label: str = "") -> Hyperedge:
    return Hyperedge(
        id=eid,
        members=frozenset(members),
        label=label,
        provenance=HyperedgeProvenance(source="manual"),
    )


# ---------------------------------------------------------------------------
# add_node / add_hyperedge / getters
# ---------------------------------------------------------------------------

class TestAddAndGet:
    def test_add_node(self):
        g = HypergraphStore()
        n = make_node("n1")
        g.add_node(n)
        assert g.num_nodes == 1
        assert g.get_node("n1") is n

    def test_add_multiple_nodes(self):
        g = HypergraphStore()
        for i in range(5):
            g.add_node(make_node(f"n{i}"))
        assert g.num_nodes == 5

    def test_get_nonexistent_node(self):
        g = HypergraphStore()
        assert g.get_node("missing") is None

    def test_add_hyperedge(self):
        g = HypergraphStore()
        g.add_node(make_node("n1"))
        g.add_node(make_node("n2"))
        e = make_edge("e1", ["n1", "n2"], label="test")
        g.add_hyperedge(e)
        assert g.num_edges == 1
        assert g.get_edge("e1") is e

    def test_hyperedge_updates_node_edges_index(self):
        g = HypergraphStore()
        g.add_node(make_node("n1"))
        g.add_node(make_node("n2"))
        g.add_hyperedge(make_edge("e1", ["n1", "n2"]))
        # Internal node_edges index updated
        assert "e1" in g._node_edges["n1"]
        assert "e1" in g._node_edges["n2"]

    def test_get_nonexistent_edge(self):
        g = HypergraphStore()
        assert g.get_edge("missing") is None

    def test_nodes_property_returns_dict(self):
        g = HypergraphStore()
        g.add_node(make_node("n1"))
        assert isinstance(g.nodes, dict)
        assert "n1" in g.nodes

    def test_edges_property_returns_dict(self):
        g = HypergraphStore()
        g.add_hyperedge(make_edge("e1", ["n1"]))
        assert isinstance(g.edges, dict)
        assert "e1" in g.edges

    def test_add_edge_without_pre_adding_nodes(self):
        """Edges referencing unknown nodes should still be stored."""
        g = HypergraphStore()
        e = make_edge("e1", ["ghost_node"])
        g.add_hyperedge(e)
        assert g.num_edges == 1
        # node_edges index created for ghost_node
        assert "ghost_node" in g._node_edges

    def test_overwrite_node(self):
        g = HypergraphStore()
        n1 = make_node("n1", ontology="bio")
        n2 = make_node("n1", ontology="chem")
        g.add_node(n1)
        g.add_node(n2)
        # Last write wins
        assert g.get_node("n1").ontology == "chem"
        assert g.num_nodes == 1


# ---------------------------------------------------------------------------
# incidence_matrix
# ---------------------------------------------------------------------------

class TestIncidenceMatrix:
    def test_empty_graph_returns_matrix(self):
        g = HypergraphStore()
        H = g.incidence_matrix()
        assert sp.issparse(H)
        # Empty graph returns a (1, 1) matrix per implementation
        assert H.shape == (1, 1)

    def test_shape_n_nodes_x_n_edges(self):
        g = HypergraphStore()
        for i in range(4):
            g.add_node(make_node(f"n{i}"))
        g.add_hyperedge(make_edge("e1", ["n0", "n1", "n2"]))
        g.add_hyperedge(make_edge("e2", ["n1", "n3"]))
        H = g.incidence_matrix()
        assert H.shape == (4, 2)

    def test_values_are_binary(self):
        g = HypergraphStore()
        for i in range(3):
            g.add_node(make_node(f"n{i}"))
        g.add_hyperedge(make_edge("e1", ["n0", "n1", "n2"]))
        H = g.incidence_matrix()
        data = H.data
        assert all(v == 1.0 for v in data)

    def test_membership_encoded_correctly(self):
        g = HypergraphStore()
        for i in range(3):
            g.add_node(make_node(f"n{i}"))
        e = make_edge("e1", ["n0", "n2"])
        g.add_hyperedge(e)
        H = g.incidence_matrix().toarray()

        # n_index maps node ids to row indices (sorted)
        g._ensure_indexed()
        n_idx = g._node_index
        e_idx = g._edge_index

        col = e_idx["e1"]
        assert H[n_idx["n0"], col] == 1.0
        assert H[n_idx["n1"], col] == 0.0
        assert H[n_idx["n2"], col] == 1.0

    def test_matrix_is_csr(self):
        g = HypergraphStore()
        g.add_node(make_node("n0"))
        g.add_hyperedge(make_edge("e1", ["n0"]))
        H = g.incidence_matrix()
        assert isinstance(H, sp.csr_matrix)

    def test_dirty_flag_clears_after_build(self):
        g = HypergraphStore()
        g.add_node(make_node("n0"))
        g.incidence_matrix()
        assert not g._dirty

    def test_matrix_rebuilt_after_new_node(self):
        g = HypergraphStore()
        g.add_node(make_node("n0"))
        g.add_hyperedge(make_edge("e1", ["n0"]))
        H1 = g.incidence_matrix()
        assert H1.shape == (1, 1)

        g.add_node(make_node("n1"))
        g.add_hyperedge(make_edge("e2", ["n0", "n1"]))
        H2 = g.incidence_matrix()
        assert H2.shape == (2, 2)

    def test_nnz_matches_total_memberships(self):
        g = HypergraphStore()
        for i in range(5):
            g.add_node(make_node(f"n{i}"))
        # e1 has 3 members, e2 has 2 members => nnz = 5
        g.add_hyperedge(make_edge("e1", ["n0", "n1", "n2"]))
        g.add_hyperedge(make_edge("e2", ["n3", "n4"]))
        H = g.incidence_matrix()
        assert H.nnz == 5


# ---------------------------------------------------------------------------
# has_similar_edge
# ---------------------------------------------------------------------------

class TestHasSimilarEdge:
    def test_exact_match_exceeds_threshold(self):
        g = HypergraphStore()
        g.add_hyperedge(make_edge("e1", ["n1", "n2", "n3"]))
        assert g.has_similar_edge(frozenset(["n1", "n2", "n3"]), threshold=0.8)

    def test_disjoint_sets_no_match(self):
        g = HypergraphStore()
        g.add_hyperedge(make_edge("e1", ["n1", "n2"]))
        assert not g.has_similar_edge(frozenset(["n3", "n4"]), threshold=0.8)

    def test_partial_overlap_below_threshold(self):
        g = HypergraphStore()
        # Jaccard({n1} | {n1,n2,n3,n4}) = 1/4 = 0.25
        g.add_hyperedge(make_edge("e1", ["n1", "n2", "n3", "n4"]))
        assert not g.has_similar_edge(frozenset(["n1"]), threshold=0.8)

    def test_partial_overlap_above_threshold(self):
        g = HypergraphStore()
        # Jaccard({n1,n2} | {n1,n2,n3}) = 2/3 ≈ 0.667
        g.add_hyperedge(make_edge("e1", ["n1", "n2", "n3"]))
        assert not g.has_similar_edge(frozenset(["n1", "n2"]), threshold=0.8)
        # threshold=0.5 => 0.667 > 0.5 => True
        assert g.has_similar_edge(frozenset(["n1", "n2"]), threshold=0.5)

    def test_empty_graph_no_match(self):
        g = HypergraphStore()
        assert not g.has_similar_edge(frozenset(["n1"]), threshold=0.5)

    def test_single_member_exact_match(self):
        g = HypergraphStore()
        g.add_hyperedge(make_edge("e1", ["solo"]))
        assert g.has_similar_edge(frozenset(["solo"]), threshold=0.9)


# ---------------------------------------------------------------------------
# mint_hyperedge
# ---------------------------------------------------------------------------

class TestMintHyperedge:
    def test_basic_mint(self):
        g = HypergraphStore()
        for nid in ["n1", "n2", "n3"]:
            g.add_node(make_node(nid))

        edge = g.mint_hyperedge(
            members=frozenset(["n1", "n2", "n3"]),
            label="protein cluster",
            coherence=0.85,
            intention="find binding partners",
            predicates=["binds_to"],
        )

        assert edge.id.startswith("he_")
        assert edge.provenance.source == "minted"
        assert edge.provenance.intention == "find binding partners"
        assert edge.provenance.predicates == ["binds_to"]
        assert edge.coherence_score == 0.85
        assert edge.utility_context == "find binding partners"

    def test_mint_adds_to_graph(self):
        g = HypergraphStore()
        before = g.num_edges
        g.mint_hyperedge(frozenset(["n1"]), "label", 0.5, "intent")
        assert g.num_edges == before + 1

    def test_mint_without_predicates(self):
        g = HypergraphStore()
        edge = g.mint_hyperedge(frozenset(["n1"]), "label", 0.6, "intent")
        assert edge.provenance.predicates is None

    def test_minted_edge_retrievable(self):
        g = HypergraphStore()
        edge = g.mint_hyperedge(frozenset(["n1", "n2"]), "lbl", 0.7, "intent")
        assert g.get_edge(edge.id) is edge


# ---------------------------------------------------------------------------
# reinforce_edge
# ---------------------------------------------------------------------------

class TestReinforceEdge:
    def test_increments_access_count(self):
        g = HypergraphStore()
        e = make_edge("e1", ["n1"])
        g.add_hyperedge(e)
        assert e.access_count == 0
        g.reinforce_edge("e1")
        assert e.access_count == 1
        g.reinforce_edge("e1")
        assert e.access_count == 2

    def test_updates_last_accessed(self):
        g = HypergraphStore()
        e = make_edge("e1", ["n1"])
        e.last_accessed = 0.0  # old timestamp
        g.add_hyperedge(e)
        before = time.time()
        g.reinforce_edge("e1")
        assert e.last_accessed >= before

    def test_weight_increases(self):
        g = HypergraphStore()
        e = make_edge("e1", ["n1"])
        e.weight = 0.5
        g.add_hyperedge(e)
        g.reinforce_edge("e1")
        assert e.weight > 0.5

    def test_weight_capped_at_one(self):
        g = HypergraphStore()
        e = make_edge("e1", ["n1"])
        e.weight = 1.0
        g.add_hyperedge(e)
        g.reinforce_edge("e1")
        assert e.weight <= 1.0

    def test_reinforce_nonexistent_edge_is_noop(self):
        g = HypergraphStore()
        # Should not raise
        g.reinforce_edge("nonexistent")

    def test_multiple_reinforcements(self):
        g = HypergraphStore()
        e = make_edge("e1", ["n1"])
        e.weight = 0.9  # start close to cap so it hits 1.0
        g.add_hyperedge(e)
        for _ in range(10):
            g.reinforce_edge("e1")
        # After enough reinforcements at 1.1x, weight should be capped at 1.0
        assert e.weight == 1.0
        assert e.access_count == 10


# ---------------------------------------------------------------------------
# decay_edges
# ---------------------------------------------------------------------------

class TestDecayEdges:
    def test_recent_edges_survive(self):
        g = HypergraphStore()
        e = make_edge("e1", ["n1"])
        e.coherence_score = 0.9
        e.weight = 0.9
        e.last_accessed = time.time()  # just now
        g.add_hyperedge(e)
        pruned = g.decay_edges(half_life_days=30.0, prune_threshold=0.01)
        assert pruned == 0
        assert g.get_edge("e1") is not None

    def test_ancient_low_coherence_edges_pruned(self):
        g = HypergraphStore()
        e = make_edge("e1", ["n1"])
        # Last accessed 1000 days ago, coherence very low
        e.coherence_score = 0.001
        e.access_count = 0
        e.last_accessed = time.time() - (1000 * 24 * 3600)
        g.add_hyperedge(e)
        pruned = g.decay_edges(half_life_days=30.0, prune_threshold=0.01)
        assert pruned == 1
        assert g.get_edge("e1") is None

    def test_pruned_edge_removed_from_node_index(self):
        g = HypergraphStore()
        g.add_node(make_node("n1"))
        e = make_edge("e1", ["n1"])
        e.coherence_score = 0.001
        e.last_accessed = time.time() - (1000 * 24 * 3600)
        g.add_hyperedge(e)
        g.decay_edges(half_life_days=30.0, prune_threshold=0.01)
        assert "e1" not in g._node_edges.get("n1", set())

    def test_high_access_count_protects_edge(self):
        g = HypergraphStore()
        e = make_edge("e1", ["n1"])
        # coherence=0.5, access_count=10 => access_boost=1.0
        # weight = 0.5 * (0.5 * decay + 0.5 * 1.0) >= 0.5 * 0.5 = 0.25 > 0.01
        e.coherence_score = 0.5
        e.access_count = 10
        e.last_accessed = time.time() - (1000 * 24 * 3600)  # very old
        g.add_hyperedge(e)
        pruned = g.decay_edges(half_life_days=30.0, prune_threshold=0.01)
        assert pruned == 0

    def test_empty_graph_returns_zero(self):
        g = HypergraphStore()
        assert g.decay_edges() == 0

    def test_prune_marks_dirty(self):
        g = HypergraphStore()
        e = make_edge("e1", ["n1"])
        e.coherence_score = 0.001
        e.last_accessed = 0
        g.add_hyperedge(e)
        g.incidence_matrix()  # clears dirty
        assert not g._dirty
        g.decay_edges(half_life_days=30.0, prune_threshold=0.01)
        assert g._dirty


# ---------------------------------------------------------------------------
# get_node_by_index
# ---------------------------------------------------------------------------

class TestGetNodeByIndex:
    def test_lookup_returns_correct_node(self):
        g = HypergraphStore()
        n_a = make_node("a")
        n_b = make_node("b")
        g.add_node(n_a)
        g.add_node(n_b)

        g._ensure_indexed()
        idx_a = g._node_index["a"]
        idx_b = g._node_index["b"]

        assert g.get_node_by_index(idx_a).id == "a"
        assert g.get_node_by_index(idx_b).id == "b"

    def test_out_of_range_returns_none(self):
        g = HypergraphStore()
        g.add_node(make_node("n1"))
        assert g.get_node_by_index(999) is None

    def test_empty_graph_returns_none(self):
        g = HypergraphStore()
        assert g.get_node_by_index(0) is None


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def _build_graph(self) -> HypergraphStore:
        g = HypergraphStore()
        for i in range(4):
            n = make_node(f"node_{i}", ontology="bio" if i % 2 == 0 else "chem")
            n.metadata = {"index": i}
            g.add_node(n)

        e1 = Hyperedge(
            id="edge_1",
            members=frozenset(["node_0", "node_1"]),
            label="pair",
            provenance=HyperedgeProvenance(
                source="minted", intention="test intent", predicates=["binds_to"]
            ),
            coherence_score=0.75,
            utility_context="test intent",
            access_count=3,
            weight=0.8,
        )
        e2 = Hyperedge(
            id="edge_2",
            members=frozenset(["node_2", "node_3"]),
            label="another",
            provenance=HyperedgeProvenance(source="manual"),
            coherence_score=0.5,
        )
        g.add_hyperedge(e1)
        g.add_hyperedge(e2)
        return g

    def test_save_creates_files(self):
        g = self._build_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "nodes.jsonl"))
            assert os.path.exists(os.path.join(tmpdir, "hyperedges.jsonl"))

    def test_roundtrip_node_count(self):
        g = self._build_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(tmpdir)
            g2 = HypergraphStore()
            g2.load(tmpdir)
            assert g2.num_nodes == g.num_nodes

    def test_roundtrip_edge_count(self):
        g = self._build_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(tmpdir)
            g2 = HypergraphStore()
            g2.load(tmpdir)
            assert g2.num_edges == g.num_edges

    def test_roundtrip_node_metadata(self):
        g = self._build_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(tmpdir)
            g2 = HypergraphStore()
            g2.load(tmpdir)
            for nid, node in g.nodes.items():
                loaded = g2.get_node(nid)
                assert loaded is not None, f"Node {nid} not found after load"
                assert loaded.ontology == node.ontology
                assert loaded.metadata == node.metadata

    def test_roundtrip_edge_fields(self):
        g = self._build_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(tmpdir)
            g2 = HypergraphStore()
            g2.load(tmpdir)

            e1 = g2.get_edge("edge_1")
            assert e1 is not None
            assert e1.label == "pair"
            assert e1.coherence_score == 0.75
            assert e1.access_count == 3
            assert abs(e1.weight - 0.8) < 1e-9
            assert e1.provenance.source == "minted"
            assert e1.provenance.intention == "test intent"
            assert e1.provenance.predicates == ["binds_to"]

    def test_roundtrip_edge_members(self):
        g = self._build_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(tmpdir)
            g2 = HypergraphStore()
            g2.load(tmpdir)
            e1 = g2.get_edge("edge_1")
            assert e1.members == frozenset(["node_0", "node_1"])

    def test_load_updates_dirty_flag(self):
        g = self._build_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(tmpdir)
            g2 = HypergraphStore()
            g2.load(tmpdir)
            # After loading, dirty should be True so matrix gets rebuilt
            assert g2._dirty

    def test_load_on_missing_dir_is_noop(self):
        g = HypergraphStore()
        # Loading from non-existent path should not raise
        g.load("/nonexistent/path/that/does/not/exist")
        assert g.num_nodes == 0
        assert g.num_edges == 0

    def test_roundtrip_incidence_matrix(self):
        g = self._build_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(tmpdir)
            g2 = HypergraphStore()
            g2.load(tmpdir)
            H = g2.incidence_matrix()
            assert H.shape == (4, 2)
            assert H.nnz == 4  # 2 edges x 2 members each


# ---------------------------------------------------------------------------
# _remove_edge (internal)
# ---------------------------------------------------------------------------

class TestRemoveEdge:
    def test_remove_decrements_count(self):
        g = HypergraphStore()
        g.add_hyperedge(make_edge("e1", ["n1", "n2"]))
        assert g.num_edges == 1
        g._remove_edge("e1")
        assert g.num_edges == 0

    def test_remove_cleans_node_index(self):
        g = HypergraphStore()
        g.add_node(make_node("n1"))
        g.add_hyperedge(make_edge("e1", ["n1"]))
        g._remove_edge("e1")
        assert "e1" not in g._node_edges.get("n1", set())

    def test_remove_marks_dirty(self):
        g = HypergraphStore()
        g.add_hyperedge(make_edge("e1", ["n1"]))
        g.incidence_matrix()
        assert not g._dirty
        g._remove_edge("e1")
        assert g._dirty

    def test_remove_nonexistent_is_noop(self):
        g = HypergraphStore()
        g._remove_edge("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# Empty graph edge cases
# ---------------------------------------------------------------------------

class TestEmptyGraph:
    def test_num_nodes_zero(self):
        assert HypergraphStore().num_nodes == 0

    def test_num_edges_zero(self):
        assert HypergraphStore().num_edges == 0

    def test_has_similar_edge_empty(self):
        g = HypergraphStore()
        assert not g.has_similar_edge(frozenset(["n1"]))

    def test_reinforce_nonexistent(self):
        g = HypergraphStore()
        g.reinforce_edge("e_missing")  # should not raise

    def test_decay_empty(self):
        g = HypergraphStore()
        assert g.decay_edges() == 0

    def test_incidence_matrix_empty(self):
        g = HypergraphStore()
        H = g.incidence_matrix()
        assert sp.issparse(H)
