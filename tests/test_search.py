"""
Comprehensive tests for the search and projection engine.
All embeddings are mocked with random numpy vectors — no sentence-transformers required.
"""
import numpy as np
import pytest

# Fixed seed for all random vectors
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_embedding(dim: int = 16, rng=None) -> np.ndarray:
    r = rng or RNG
    v = r.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def make_node(node_id: str, dim: int = 16, ontology: str = "default", rng=None):
    from intention_engine.models import Node
    return Node(id=node_id, embedding=make_embedding(dim, rng), ontology=ontology)


def make_store_with_nodes(n: int = 10, dim: int = 16, ontology: str = "default"):
    """Build a HypergraphStore with n nodes, all having embeddings."""
    from intention_engine.hypergraph import HypergraphStore
    store = HypergraphStore()
    for i in range(n):
        store.add_node(make_node(f"n{i:04d}", dim=dim, ontology=ontology))
    return store


def make_intention_with_embeddings(raw: str = "find similar concepts", dim: int = 16):
    """Create an Intention whose predicates have fake embeddings."""
    from intention_engine.models import Intention, Predicate, SearchScope
    pred1 = Predicate(text="find similar", embedding=make_embedding(dim), weight=0.5)
    pred2 = Predicate(text="concepts knowledge", embedding=make_embedding(dim), weight=0.5)
    return Intention(
        raw=raw,
        predicates=[pred1, pred2],
        embedding=make_embedding(dim),
        scope=SearchScope(),
    )


# ---------------------------------------------------------------------------
# UtilityProjector tests
# ---------------------------------------------------------------------------

class TestUtilityProjector:
    def test_project_shape(self):
        from intention_engine.projection import UtilityProjector
        proj = UtilityProjector()
        n, k, d = 10, 3, 16
        node_emb = RNG.standard_normal((n, d)).astype(np.float32)
        pred_emb = RNG.standard_normal((k, d)).astype(np.float32)
        pred_w = np.ones(k, dtype=np.float32) / k
        U = proj.project(node_emb, pred_emb, pred_w)
        assert U.shape == (n, k), f"Expected ({n},{k}), got {U.shape}"

    def test_project_cosine_range(self):
        """Weighted cosine similarities should stay in [-1, 1] * max_weight."""
        from intention_engine.projection import UtilityProjector
        proj = UtilityProjector()
        n, k, d = 20, 4, 32
        node_emb = RNG.standard_normal((n, d)).astype(np.float32)
        pred_emb = RNG.standard_normal((k, d)).astype(np.float32)
        pred_w = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        U = proj.project(node_emb, pred_emb, pred_w)
        # Each column j has values in [-pred_w[j], pred_w[j]]
        for j in range(k):
            assert np.all(U[:, j] <= pred_w[j] + 1e-5), f"col {j} exceeds weight"
            assert np.all(U[:, j] >= -pred_w[j] - 1e-5), f"col {j} below -weight"

    def test_project_identical_embeddings(self):
        """If a node embedding == predicate embedding, cosine sim should be 1.0."""
        from intention_engine.projection import UtilityProjector
        proj = UtilityProjector()
        d = 16
        v = make_embedding(d)
        node_emb = v[np.newaxis, :]  # (1, d)
        pred_emb = v[np.newaxis, :]  # (1, d)
        pred_w = np.array([1.0])
        U = proj.project(node_emb, pred_emb, pred_w)
        assert abs(U[0, 0] - 1.0) < 1e-5, f"Expected 1.0, got {U[0,0]}"

    def test_project_orthogonal_embeddings(self):
        """Orthogonal embeddings => cosine sim ≈ 0."""
        from intention_engine.projection import UtilityProjector
        proj = UtilityProjector()
        node_emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        pred_emb = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        pred_w = np.array([1.0])
        U = proj.project(node_emb, pred_emb, pred_w)
        assert abs(U[0, 0]) < 1e-5, f"Expected ~0, got {U[0,0]}"

    def test_filter_by_threshold_shape(self):
        from intention_engine.projection import UtilityProjector
        proj = UtilityProjector()
        n, k = 50, 4
        U = RNG.standard_normal((n, k)).astype(np.float32)
        U_active, indices = proj.filter_by_threshold(U, percentile=80.0)
        # 80th percentile → roughly 20% should be retained (at least 1)
        assert U_active.shape[0] == len(indices)
        assert len(indices) >= 1
        assert len(indices) <= n

    def test_filter_by_threshold_top_nodes_included(self):
        """The highest-magnitude node must always be in the result."""
        from intention_engine.projection import UtilityProjector
        proj = UtilityProjector()
        n, k = 30, 3
        U = RNG.standard_normal((n, k)).astype(np.float32)
        mags = proj.utility_magnitudes(U)
        best_idx = int(np.argmax(mags))
        _, indices = proj.filter_by_threshold(U, percentile=90.0)
        assert best_idx in indices, f"Top node {best_idx} not in active set"

    def test_utility_magnitudes_shape(self):
        from intention_engine.projection import UtilityProjector
        proj = UtilityProjector()
        U = RNG.standard_normal((15, 5)).astype(np.float32)
        mags = proj.utility_magnitudes(U)
        assert mags.shape == (15,)
        assert np.all(mags >= 0)


# ---------------------------------------------------------------------------
# CoherenceScorer tests
# ---------------------------------------------------------------------------

class TestCoherenceScorer:
    def test_tight_cluster_high_score(self):
        """Vectors all pointing in the same direction → high coherence."""
        from intention_engine.coherence import CoherenceScorer
        scorer = CoherenceScorer()
        base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # 5 nearly identical vectors
        U = np.tile(base, (5, 1)) + RNG.standard_normal((5, 4)).astype(np.float32) * 0.01
        score = scorer.score(U)
        assert score > 0.7, f"Expected high score for tight cluster, got {score}"

    def test_spread_low_score(self):
        """Vectors pointing in random directions → lower coherence."""
        from intention_engine.coherence import CoherenceScorer
        scorer = CoherenceScorer()
        # 10 random unit vectors in high-dim space
        local_rng = np.random.default_rng(99)
        U = local_rng.standard_normal((10, 64)).astype(np.float32)
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / np.maximum(norms, 1e-10)
        score = scorer.score(U)
        # Random vectors in high-dim will have low pairwise similarity
        assert score < 0.8, f"Spread vectors should not score very high, got {score}"

    def test_single_vector_returns_zero(self):
        """m < 2 should return 0.0."""
        from intention_engine.coherence import CoherenceScorer
        scorer = CoherenceScorer()
        U = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        score = scorer.score(U)
        assert score == 0.0

    def test_score_range(self):
        """Score must always be in [0, 1]."""
        from intention_engine.coherence import CoherenceScorer
        scorer = CoherenceScorer()
        for _ in range(20):
            m = RNG.integers(2, 15)
            k = RNG.integers(2, 10)
            U = RNG.standard_normal((m, k)).astype(np.float32)
            score = scorer.score(U)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1]"

    def test_ontology_diversity_bonus(self):
        """Cross-ontology group should score higher than same-ontology group."""
        from intention_engine.coherence import CoherenceScorer
        scorer = CoherenceScorer(weights=(0.4, 0.4, 0.2))
        # Tight cluster
        base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        U = np.tile(base, (6, 1)) + RNG.standard_normal((6, 4)).astype(np.float32) * 0.01

        same_ontology = ["physics"] * 6
        diverse_ontology = ["physics", "biology", "chemistry", "math", "cs", "art"]

        score_same = scorer.score(U, same_ontology)
        score_diverse = scorer.score(U, diverse_ontology)
        assert score_diverse >= score_same, (
            f"Diverse {score_diverse} should be >= same {score_same}"
        )

    def test_diversity_bonus_no_ontologies(self):
        """No ontologies provided → diversity bonus = 0, no crash."""
        from intention_engine.coherence import CoherenceScorer
        scorer = CoherenceScorer()
        U = RNG.standard_normal((4, 4)).astype(np.float32)
        score = scorer.score(U, ontologies=None)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# cluster_utility_space tests
# ---------------------------------------------------------------------------

class TestClusterUtilitySpace:
    def test_two_distinct_groups(self):
        """Two well-separated groups in utility space → 2 clusters."""
        from intention_engine.clustering import cluster_utility_space
        local_rng = np.random.default_rng(7)
        # Group A: near [1, 0]
        A = local_rng.standard_normal((8, 2)).astype(np.float32) * 0.05
        A += np.array([1.0, 0.0])
        # Group B: near [0, 1]
        B = local_rng.standard_normal((8, 2)).astype(np.float32) * 0.05
        B += np.array([0.0, 1.0])
        U = np.vstack([A, B])
        clusters = cluster_utility_space(U, min_cluster_size=2, max_cluster_size=50)
        assert len(clusters) >= 2, f"Expected >=2 clusters, got {len(clusters)}"

    def test_single_tight_group(self):
        """One tight group → 1 cluster."""
        from intention_engine.clustering import cluster_utility_space
        local_rng = np.random.default_rng(13)
        U = np.tile([1.0, 0.0], (10, 1)).astype(np.float32)
        U += local_rng.standard_normal((10, 2)).astype(np.float32) * 0.01
        clusters = cluster_utility_space(U, min_cluster_size=2, max_cluster_size=50)
        assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"

    def test_too_few_points(self):
        """Fewer points than min_cluster_size → empty list."""
        from intention_engine.clustering import cluster_utility_space
        U = np.array([[1.0, 0.0]], dtype=np.float32)
        clusters = cluster_utility_space(U, min_cluster_size=2)
        assert clusters == []

    def test_cluster_indices_valid(self):
        """All cluster indices must be valid indices into the input array."""
        from intention_engine.clustering import cluster_utility_space
        n, k = 20, 4
        U = RNG.standard_normal((n, k)).astype(np.float32)
        clusters = cluster_utility_space(U, min_cluster_size=2, max_cluster_size=50)
        for c in clusters:
            assert np.all(c.indices >= 0)
            assert np.all(c.indices < n)

    def test_cluster_centroid_shape(self):
        """Cluster centroid must have same dim as utility vectors."""
        from intention_engine.clustering import cluster_utility_space
        k = 5
        U = RNG.standard_normal((15, k)).astype(np.float32)
        clusters = cluster_utility_space(U, min_cluster_size=2, max_cluster_size=50)
        for c in clusters:
            assert c.centroid.shape == (k,), f"Centroid shape {c.centroid.shape} != ({k},)"

    def test_cluster_size_bounds(self):
        """All returned clusters must respect size bounds."""
        from intention_engine.clustering import cluster_utility_space
        U = RNG.standard_normal((30, 4)).astype(np.float32)
        min_s, max_s = 3, 10
        clusters = cluster_utility_space(U, min_cluster_size=min_s, max_cluster_size=max_s)
        for c in clusters:
            sz = len(c.indices)
            assert min_s <= sz <= max_s, f"Cluster size {sz} out of [{min_s},{max_s}]"


# ---------------------------------------------------------------------------
# SearchEngine tests — exploit phase
# ---------------------------------------------------------------------------

class TestSearchEngineExploit:
    def _make_engine(self, dim: int = 16):
        from intention_engine.hypergraph import HypergraphStore
        from intention_engine.models import EngineConfig
        from intention_engine.search import SearchEngine
        store = HypergraphStore()
        config = EngineConfig(
            embedding_dim=dim,
            explore_enabled=False,  # test exploit only
        )
        return SearchEngine(store, config), store, dim

    def test_empty_store_returns_empty(self):
        engine, store, dim = self._make_engine()
        intention = make_intention_with_embeddings(dim=dim)
        result = engine.search(intention)
        assert result.nodes == []
        assert result.exploited_edges == []

    def test_exploit_returns_scored_nodes(self):
        from intention_engine.models import Hyperedge, HyperedgeProvenance
        engine, store, dim = self._make_engine()
        # Add nodes + one edge
        nodes = [make_node(f"n{i}", dim=dim) for i in range(5)]
        for n in nodes:
            store.add_node(n)
        edge = Hyperedge(
            id="he_test_001",
            members=frozenset([n.id for n in nodes[:3]]),
            label="test edge",
        )
        store.add_hyperedge(edge)

        intention = make_intention_with_embeddings(dim=dim)
        result = engine.search(intention)
        assert len(result.nodes) > 0, "Expected some scored nodes"

    def test_exploit_scores_are_positive(self):
        from intention_engine.models import Hyperedge
        engine, store, dim = self._make_engine()
        for i in range(8):
            store.add_node(make_node(f"n{i}", dim=dim))
        node_ids = list(store.nodes.keys())
        edge = Hyperedge(id="he_t002", members=frozenset(node_ids[:4]))
        store.add_hyperedge(edge)

        intention = make_intention_with_embeddings(dim=dim)
        result = engine.search(intention)
        for sn in result.nodes:
            assert sn.score > 0.0, f"Non-positive score: {sn.score}"

    def test_exploit_activates_relevant_edges(self):
        """Edges with high cosine sim to intent should be activated and reinforced."""
        from intention_engine.models import Hyperedge
        engine, store, dim = self._make_engine()
        intent_vec = make_embedding(dim)

        # Create nodes whose embeddings are close to intent
        from intention_engine.models import Node
        for i in range(4):
            n = Node(id=f"close_{i}", embedding=intent_vec + RNG.standard_normal(dim).astype(np.float32) * 0.05)
            store.add_node(n)

        close_ids = [f"close_{i}" for i in range(4)]
        edge = Hyperedge(id="he_close", members=frozenset(close_ids))
        store.add_hyperedge(edge)

        from intention_engine.models import Intention, Predicate, SearchScope
        intention = Intention(
            raw="test",
            predicates=[Predicate(text="test", embedding=intent_vec, weight=1.0)],
            embedding=intent_vec,
            scope=SearchScope(),
        )
        result = engine.search(intention)
        assert len(result.exploited_edges) > 0, "Expected activated edges"

    def test_exploit_result_sorted_descending(self):
        from intention_engine.models import Hyperedge
        engine, store, dim = self._make_engine()
        for i in range(10):
            store.add_node(make_node(f"n{i}", dim=dim))
        node_ids = list(store.nodes.keys())
        edge = Hyperedge(id="he_t003", members=frozenset(node_ids))
        store.add_hyperedge(edge)

        intention = make_intention_with_embeddings(dim=dim)
        result = engine.search(intention)
        scores = [sn.score for sn in result.nodes]
        assert scores == sorted(scores, reverse=True), "Results not sorted descending"

    def test_exploit_respects_max_results(self):
        from intention_engine.models import Hyperedge
        engine, store, dim = self._make_engine()
        for i in range(50):
            store.add_node(make_node(f"n{i:04d}", dim=dim))
        node_ids = sorted(store.nodes.keys())
        edge = Hyperedge(id="he_t004", members=frozenset(node_ids))
        store.add_hyperedge(edge)

        intention = make_intention_with_embeddings(dim=dim)
        result = engine.search(intention, max_results=5)
        assert len(result.nodes) <= 5


# ---------------------------------------------------------------------------
# SearchEngine tests — explore phase
# ---------------------------------------------------------------------------

class TestSearchEngineExplore:
    def _make_engine(self, dim: int = 16):
        from intention_engine.hypergraph import HypergraphStore
        from intention_engine.models import EngineConfig
        from intention_engine.search import SearchEngine
        store = HypergraphStore()
        config = EngineConfig(
            embedding_dim=dim,
            explore_enabled=True,
            min_coherence=0.1,  # low threshold for test reliability
            utility_threshold_percentile=50.0,
            min_edge_size=2,
            max_edge_size=20,
            max_mints_per_query=5,
        )
        return SearchEngine(store, config), store

    def test_explore_cold_start_no_edges(self):
        """With no edges but nodes with embeddings, explore should still run."""
        engine, store = self._make_engine(dim=8)
        # Two tight clusters of nodes
        local_rng = np.random.default_rng(55)
        from intention_engine.models import Node
        for i in range(5):
            v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            v += local_rng.standard_normal(8).astype(np.float32) * 0.05
            store.add_node(Node(id=f"a{i}", embedding=v, ontology="A"))
        for i in range(5):
            v = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            v += local_rng.standard_normal(8).astype(np.float32) * 0.05
            store.add_node(Node(id=f"b{i}", embedding=v, ontology="B"))

        from intention_engine.models import Intention, Predicate, SearchScope
        pred_emb = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        intention = Intention(
            raw="find A cluster",
            predicates=[Predicate(text="find", embedding=pred_emb, weight=1.0)],
            embedding=pred_emb,
            scope=SearchScope(min_coherence=0.1),
        )
        result = engine.search(intention)
        # Should produce at least some nodes (either from explore or both phases)
        # Not asserting exact count — just no crash and explanation present
        assert result.explanation is not None
        assert isinstance(result.nodes, list)

    def test_explore_mints_edges(self):
        """Tight cluster of nodes with no prior edges → explore should mint at least 1 edge."""
        engine, store = self._make_engine(dim=4)
        local_rng = np.random.default_rng(77)
        from intention_engine.models import Node
        # Tight group
        base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(8):
            v = base + local_rng.standard_normal(4).astype(np.float32) * 0.02
            store.add_node(Node(id=f"tight_{i}", embedding=v, ontology="X"))

        pred_emb = base.copy()
        from intention_engine.models import Intention, Predicate, SearchScope
        intention = Intention(
            raw="find tight",
            predicates=[Predicate(text="tight", embedding=pred_emb, weight=1.0)],
            embedding=pred_emb,
            scope=SearchScope(min_coherence=0.05),
        )
        result = engine.search(intention)
        assert len(result.minted_edges) >= 1, (
            f"Expected minted edges, got 0. explore_stats: {result.explanation.explore_stats}"
        )

    def test_explore_no_double_mint(self):
        """Running search twice should not mint the same cluster again (Jaccard filter)."""
        engine, store = self._make_engine(dim=4)
        local_rng = np.random.default_rng(88)
        from intention_engine.models import Node
        base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(6):
            v = base + local_rng.standard_normal(4).astype(np.float32) * 0.01
            store.add_node(Node(id=f"c{i}", embedding=v))

        from intention_engine.models import Intention, Predicate, SearchScope
        pred_emb = base.copy()
        intention = Intention(
            raw="find c",
            predicates=[Predicate(text="find", embedding=pred_emb, weight=1.0)],
            embedding=pred_emb,
            scope=SearchScope(min_coherence=0.05),
        )
        result1 = engine.search(intention)
        minted1 = len(result1.minted_edges)
        result2 = engine.search(intention)
        minted2 = len(result2.minted_edges)
        # Second search should mint fewer or equal edges (similar edge exists)
        assert minted2 <= minted1, (
            f"Second search minted more edges ({minted2}) than first ({minted1})"
        )


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def _make_full_store(self, dim: int = 16):
        """Build a store with nodes + edges for full pipeline testing."""
        from intention_engine.hypergraph import HypergraphStore
        from intention_engine.models import Hyperedge, Node
        store = HypergraphStore()
        local_rng = np.random.default_rng(42)

        # 20 nodes
        for i in range(20):
            v = local_rng.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-10
            store.add_node(Node(id=f"node_{i:03d}", embedding=v, ontology=f"ont_{i % 3}"))

        node_ids = sorted(store.nodes.keys())

        # 4 edges covering subsets
        edge1 = Hyperedge(id="he_001", members=frozenset(node_ids[:5]))
        edge2 = Hyperedge(id="he_002", members=frozenset(node_ids[5:10]))
        edge3 = Hyperedge(id="he_003", members=frozenset(node_ids[10:15]))
        edge4 = Hyperedge(id="he_004", members=frozenset(node_ids[15:]))
        for e in [edge1, edge2, edge3, edge4]:
            store.add_hyperedge(e)

        return store

    def test_full_search_returns_result(self):
        from intention_engine.models import EngineConfig
        from intention_engine.search import SearchEngine
        store = self._make_full_store()
        config = EngineConfig(explore_enabled=True, min_coherence=0.1)
        engine = SearchEngine(store, config)
        intention = make_intention_with_embeddings(dim=16)
        result = engine.search(intention, max_results=10)
        assert result is not None
        assert result.explanation is not None
        assert len(result.nodes) <= 10

    def test_full_search_explanation_populated(self):
        from intention_engine.models import EngineConfig
        from intention_engine.search import SearchEngine
        store = self._make_full_store()
        config = EngineConfig(explore_enabled=True, min_coherence=0.1)
        engine = SearchEngine(store, config)
        intention = make_intention_with_embeddings(dim=16)
        result = engine.search(intention)
        exp = result.explanation
        assert exp.intention.raw == intention.raw
        assert exp.exploit_stats.elapsed_ms >= 0
        assert exp.explore_stats.elapsed_ms >= 0

    def test_cold_start_explore_only(self):
        """No edges at all — search falls through to explore phase."""
        from intention_engine.hypergraph import HypergraphStore
        from intention_engine.models import EngineConfig, Node
        from intention_engine.search import SearchEngine
        store = HypergraphStore()
        local_rng = np.random.default_rng(101)
        for i in range(12):
            v = local_rng.standard_normal(8).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-10
            store.add_node(Node(id=f"cold_{i}", embedding=v))

        config = EngineConfig(
            explore_enabled=True,
            embedding_dim=8,
            min_coherence=0.05,
            utility_threshold_percentile=50.0,
        )
        engine = SearchEngine(store, config)
        intention = make_intention_with_embeddings(dim=8)
        result = engine.search(intention)
        # Just verify it doesn't crash
        assert isinstance(result.nodes, list)

    def test_search_source_labels(self):
        """All scored nodes must have a valid source label."""
        from intention_engine.models import EngineConfig
        from intention_engine.search import SearchEngine
        store = self._make_full_store()
        config = EngineConfig(explore_enabled=True, min_coherence=0.1)
        engine = SearchEngine(store, config)
        intention = make_intention_with_embeddings(dim=16)
        result = engine.search(intention)
        valid_sources = {"exploit", "explore", "both"}
        for sn in result.nodes:
            assert sn.source in valid_sources, f"Invalid source: {sn.source}"


# ---------------------------------------------------------------------------
# Coverage computation tests
# ---------------------------------------------------------------------------

class TestCoverage:
    def _make_engine(self):
        from intention_engine.hypergraph import HypergraphStore
        from intention_engine.models import EngineConfig
        from intention_engine.search import SearchEngine
        store = HypergraphStore()
        config = EngineConfig()
        return SearchEngine(store, config)

    def test_empty_exploit_nodes_returns_zero(self):
        engine = self._make_engine()
        intention = make_intention_with_embeddings(dim=8)
        coverage = engine._coverage([], intention)
        assert coverage == 0.0

    def test_no_predicates_returns_zero(self):
        from intention_engine.models import Intention, SearchScope
        engine = self._make_engine()
        intention = Intention(raw="no predicates", predicates=[], scope=SearchScope())
        scored = [make_node("n0", dim=8)]
        from intention_engine.models import ScoredNode
        sn = ScoredNode(node=scored[0], score=0.9)
        coverage = engine._coverage([sn], intention)
        assert coverage == 0.0

    def test_perfect_coverage(self):
        """Node embedding = predicate embedding → coverage = 1.0."""
        from intention_engine.models import Intention, Predicate, SearchScope, ScoredNode, Node
        engine = self._make_engine()
        d = 8
        v = make_embedding(d)
        node = Node(id="n_perfect", embedding=v)
        pred = Predicate(text="perfect", embedding=v, weight=1.0)
        intention = Intention(raw="perfect", predicates=[pred], scope=SearchScope())
        sn = ScoredNode(node=node, score=1.0)
        coverage = engine._coverage([sn], intention)
        assert coverage == 1.0, f"Expected 1.0, got {coverage}"

    def test_no_embeddings_on_predicates(self):
        """Predicates without embeddings should be skipped gracefully."""
        from intention_engine.models import Intention, Predicate, SearchScope, ScoredNode, Node
        engine = self._make_engine()
        pred = Predicate(text="no embedding", embedding=None, weight=1.0)
        intention = Intention(raw="test", predicates=[pred], scope=SearchScope())
        node = Node(id="n0", embedding=make_embedding(8))
        sn = ScoredNode(node=node, score=0.5)
        coverage = engine._coverage([sn], intention)
        assert coverage == 0.0  # no predicates with embeddings


# ---------------------------------------------------------------------------
# Fusion tests
# ---------------------------------------------------------------------------

class TestFusion:
    def _make_engine(self):
        from intention_engine.hypergraph import HypergraphStore
        from intention_engine.models import EngineConfig
        from intention_engine.search import SearchEngine
        store = HypergraphStore()
        config = EngineConfig()
        return SearchEngine(store, config)

    def test_no_overlap_all_nodes_present(self):
        from intention_engine.models import ScoredNode
        engine = self._make_engine()
        n1 = make_node("n1")
        n2 = make_node("n2")
        exploit = [ScoredNode(node=n1, score=0.5, source="exploit")]
        explore = [ScoredNode(node=n2, score=0.6, source="explore")]
        fused = engine._fuse(exploit, explore, [])
        ids = {sn.node.id for sn in fused}
        assert "n1" in ids
        assert "n2" in ids
        assert len(fused) == 2

    def test_overlap_gets_boost(self):
        """Node found by both phases should get 1.2x boost."""
        from intention_engine.models import ScoredNode
        engine = self._make_engine()
        n = make_node("shared")
        exploit_score = 0.5
        explore_score = 0.4
        exploit = [ScoredNode(node=n, score=exploit_score, source="exploit")]
        explore = [ScoredNode(node=n, score=explore_score, source="explore")]
        fused = engine._fuse(exploit, explore, [])
        assert len(fused) == 1
        expected_score = max(exploit_score, explore_score) * 1.2
        assert abs(fused[0].score - expected_score) < 1e-5
        assert fused[0].source == "both"

    def test_overlap_merges_via_edges(self):
        """Via-edges from both phases should be combined."""
        from intention_engine.models import ScoredNode
        engine = self._make_engine()
        n = make_node("shared")
        exploit = [ScoredNode(node=n, score=0.5, source="exploit", via_edges=["e1"])]
        explore = [ScoredNode(node=n, score=0.4, source="explore", via_edges=["e2"])]
        fused = engine._fuse(exploit, explore, [])
        assert "e1" in fused[0].via_edges
        assert "e2" in fused[0].via_edges

    def test_empty_inputs(self):
        engine = self._make_engine()
        fused = engine._fuse([], [], [])
        assert fused == []

    def test_exploit_only_no_change(self):
        """With no explore nodes, fusion = identity."""
        from intention_engine.models import ScoredNode
        engine = self._make_engine()
        n1 = make_node("n1")
        exploit = [ScoredNode(node=n1, score=0.7, source="exploit")]
        fused = engine._fuse(exploit, [], [])
        assert len(fused) == 1
        assert fused[0].score == 0.7
        assert fused[0].source == "exploit"


# ---------------------------------------------------------------------------
# IntentionDecomposer tests
# ---------------------------------------------------------------------------

class TestIntentionDecomposer:
    def test_basic_decomposition(self):
        from intention_engine.decomposer import IntentionDecomposer
        d = IntentionDecomposer()
        intention = d.decompose("find proteins that interact with membrane")
        assert len(intention.predicates) >= 1
        assert intention.raw == "find proteins that interact with membrane"

    def test_equal_weights(self):
        """All predicates should have equal weights summing to 1."""
        from intention_engine.decomposer import IntentionDecomposer
        d = IntentionDecomposer()
        intention = d.decompose("alpha and beta and gamma")
        weights = [p.weight for p in intention.predicates]
        n = len(weights)
        for w in weights:
            assert abs(w - 1.0 / n) < 1e-5, f"Weight {w} != 1/{n}"

    def test_scope_passed_through(self):
        from intention_engine.decomposer import IntentionDecomposer
        from intention_engine.models import SearchScope
        d = IntentionDecomposer()
        scope = SearchScope(max_depth=5)
        intention = d.decompose("test", scope=scope)
        assert intention.scope is scope

    def test_default_scope_created(self):
        from intention_engine.decomposer import IntentionDecomposer
        d = IntentionDecomposer()
        intention = d.decompose("test query")
        assert intention.scope is not None

    def test_empty_string_fallback(self):
        """Empty or very short input should not crash."""
        from intention_engine.decomposer import IntentionDecomposer
        d = IntentionDecomposer()
        intention = d.decompose("")
        assert isinstance(intention.predicates, list)

    def test_no_embeddings_set(self):
        """Decomposer should leave embeddings as None (set later by pipeline)."""
        from intention_engine.decomposer import IntentionDecomposer
        d = IntentionDecomposer()
        intention = d.decompose("some raw text")
        assert intention.embedding is None
        for p in intention.predicates:
            assert p.embedding is None
