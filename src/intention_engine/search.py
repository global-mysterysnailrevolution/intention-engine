import time
import numpy as np
import scipy.sparse as sp

from .models import (
    Intention,
    SearchResult,
    ScoredNode,
    SearchExplanation,
    ExploitStats,
    ExploreStats,
    EngineConfig,
    Hyperedge,
    _make_id,
)
from .hypergraph import HypergraphStore
from .projection import UtilityProjector
from .coherence import CoherenceScorer
from .clustering import cluster_utility_space


class SearchEngine:
    """Two-phase search: exploit existing structure + explore for new connections."""

    def __init__(self, store: HypergraphStore, config: EngineConfig):
        self.store = store
        self.config = config
        self.projector = UtilityProjector()
        self.scorer = CoherenceScorer(weights=config.coherence_weights)

    def search(self, intention: Intention, max_results: int = 20, valid_at: float | None = None) -> SearchResult:
        """Execute two-phase search.

        Parameters
        ----------
        valid_at : float | None
            If given, restrict exploit phase to edges valid at this timestamp.
        """
        # Phase 1: Exploit
        exploit_nodes, exploit_edges, exploit_stats = self._exploit(intention, valid_at=valid_at)

        # Decide whether to explore
        minted_edges: list[Hyperedge] = []
        explore_stats = ExploreStats()

        if self.config.explore_enabled:
            coverage = self._coverage(exploit_nodes, intention)
            if coverage < 0.9:
                explore_nodes, minted_edges, explore_stats = self._explore(intention)
                # Fuse results
                exploit_nodes = self._fuse(exploit_nodes, explore_nodes, minted_edges)

        # Sort and limit
        exploit_nodes.sort(key=lambda x: x.score, reverse=True)
        exploit_nodes = exploit_nodes[:max_results]

        return SearchResult(
            nodes=exploit_nodes,
            exploited_edges=exploit_edges,
            minted_edges=minted_edges,
            explanation=SearchExplanation(
                intention=intention,
                exploit_stats=exploit_stats,
                explore_stats=explore_stats,
            ),
        )

    def _exploit(
        self, intention: Intention, valid_at: float | None = None,
    ) -> tuple[list[ScoredNode], list[Hyperedge], ExploitStats]:
        """Phase 1: Traverse existing structure via SpMV over the incidence matrix."""
        t0 = time.perf_counter()

        if self.store.num_edges == 0 or self.store.num_nodes == 0:
            return [], [], ExploitStats(elapsed_ms=(time.perf_counter() - t0) * 1000)

        if valid_at is not None:
            H = self.store.incidence_matrix_at(valid_at)
        else:
            H = self.store.incidence_matrix()  # (|V|, |E|)

        # Resolve intent vector: prefer full embedding, fall back to mean of predicate embeddings
        intent_vec: np.ndarray | None = None
        if intention.embedding is not None:
            intent_vec = intention.embedding
        elif intention.predicates:
            vecs = [p.embedding for p in intention.predicates if p.embedding is not None]
            if vecs:
                intent_vec = np.mean(vecs, axis=0)

        # Score each edge by cosine similarity to intent vector
        self.store._ensure_indexed()
        if valid_at is not None:
            valid_edges = self.store.valid_edges_at(valid_at)
            edge_ids_sorted = sorted(valid_edges.keys())
        else:
            edge_ids_sorted = sorted(self.store.edges.keys())
        edge_scores = np.zeros(len(edge_ids_sorted))

        if intent_vec is not None:
            norm_i = np.linalg.norm(intent_vec)
            for i, eid in enumerate(edge_ids_sorted):
                edge = self.store.get_edge(eid)
                if edge is None:
                    continue
                member_embeddings = []
                for nid in edge.members:
                    node = self.store.get_node(nid)
                    if node and node.embedding is not None:
                        member_embeddings.append(node.embedding)
                if member_embeddings and norm_i > 1e-10:
                    edge_emb = np.mean(member_embeddings, axis=0)
                    norm_e = np.linalg.norm(edge_emb)
                    if norm_e > 1e-10:
                        edge_scores[i] = np.dot(edge_emb, intent_vec) / (norm_e * norm_i)

        # ReLU
        edge_scores = np.maximum(edge_scores, 0.0)
        score_sum = edge_scores.sum()
        if score_sum > 1e-10:
            w = edge_scores / score_sum
        else:
            w = np.ones(len(edge_ids_sorted)) / max(len(edge_ids_sorted), 1)

        # Two-hop SpMV: node_scores -> edge_weights -> expanded node scores
        # Ensure w has correct shape matching H's columns
        n_cols = H.shape[1]
        if len(w) < n_cols:
            w = np.pad(w, (0, n_cols - len(w)))
        elif len(w) > n_cols:
            w = w[:n_cols]

        node_scores = H @ w  # (|V|,)
        total = node_scores.sum()
        node_weights = node_scores / max(total, 1e-10)

        emergent = H.T @ node_weights  # (|E|,)
        expanded = H @ emergent  # (|V|,)

        alpha = self.config.exploit_weight
        final_scores = alpha * node_scores + (1 - alpha) * expanded

        # Reinforce accessed edges
        activated_edges: list[Hyperedge] = []
        for i, eid in enumerate(edge_ids_sorted):
            if i < len(edge_scores) and edge_scores[i] > 0.1:
                self.store.reinforce_edge(eid)
                edge = self.store.get_edge(eid)
                if edge:
                    activated_edges.append(edge)

        # Build scored nodes in sorted order
        node_ids_sorted = sorted(self.store.nodes.keys())
        scored: list[ScoredNode] = []
        for i, nid in enumerate(node_ids_sorted):
            if i < len(final_scores) and final_scores[i] > 0.01:
                node = self.store.get_node(nid)
                if node:
                    scored.append(ScoredNode(node=node, score=float(final_scores[i]), source="exploit"))

        elapsed = (time.perf_counter() - t0) * 1000
        stats = ExploitStats(
            edges_scored=len(edge_ids_sorted),
            edges_activated=len(activated_edges),
            nodes_reached=len(scored),
            elapsed_ms=elapsed,
        )
        return scored, activated_edges, stats

    def _explore(
        self, intention: Intention
    ) -> tuple[list[ScoredNode], list[Hyperedge], ExploreStats]:
        """Phase 2: Project into utility space, cluster, and mint new hyperedges."""
        t0 = time.perf_counter()

        # Collect node embeddings
        node_ids = sorted(self.store.nodes.keys())
        embeddings: list[np.ndarray] = []
        valid_ids: list[str] = []
        for nid in node_ids:
            node = self.store.get_node(nid)
            if node and node.embedding is not None:
                embeddings.append(node.embedding)
                valid_ids.append(nid)

        if not embeddings or not intention.predicates:
            return [], [], ExploreStats(elapsed_ms=(time.perf_counter() - t0) * 1000)

        node_emb = np.array(embeddings)  # (n, d)

        # Collect predicate embeddings
        pred_embs: list[np.ndarray] = []
        pred_weights: list[float] = []
        for p in intention.predicates:
            if p.embedding is not None:
                pred_embs.append(p.embedding)
                pred_weights.append(p.weight)

        if not pred_embs:
            return [], [], ExploreStats(elapsed_ms=(time.perf_counter() - t0) * 1000)

        pred_emb = np.array(pred_embs)  # (k, d)
        pred_w = np.array(pred_weights)  # (k,)

        # Project into utility space
        U = self.projector.project(node_emb, pred_emb, pred_w)  # (n, k)

        # Filter to high-utility nodes
        U_active, active_indices = self.projector.filter_by_threshold(
            U, self.config.utility_threshold_percentile
        )

        if len(active_indices) < 2:
            return [], [], ExploreStats(
                nodes_projected=len(node_emb),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # Cluster in utility space
        clusters = cluster_utility_space(
            U_active,
            min_cluster_size=self.config.min_edge_size,
            max_cluster_size=self.config.max_edge_size,
        )

        # Score and mint
        minted: list[Hyperedge] = []
        scored_nodes: list[ScoredNode] = []
        candidates_evaluated = 0

        for cluster in clusters[: self.config.explore_budget]:
            candidates_evaluated += 1

            # Map cluster indices back to node IDs
            member_indices = active_indices[cluster.indices]
            member_ids = frozenset(valid_ids[i] for i in member_indices)

            # Skip if a similar edge already exists
            if self.store.has_similar_edge(member_ids, self.config.novelty_threshold):
                continue

            # Collect ontologies for diversity scoring
            ontologies: list[str] = []
            for nid in member_ids:
                node = self.store.get_node(nid)
                if node:
                    ontologies.append(node.ontology)

            # Score coherence of the cluster
            score = self.scorer.score(U_active[cluster.indices], ontologies)

            min_coherence = (
                intention.scope.min_coherence if intention.scope else self.config.min_coherence
            )
            if score >= min_coherence and len(minted) < self.config.max_mints_per_query:
                edge = self.store.mint_hyperedge(
                    members=member_ids,
                    label=f"Discovered: {intention.raw[:50]}",
                    coherence=score,
                    intention=intention.raw,
                    predicates=[p.text for p in intention.predicates],
                )
                minted.append(edge)

                for nid in member_ids:
                    node = self.store.get_node(nid)
                    if node:
                        scored_nodes.append(
                            ScoredNode(
                                node=node, score=score, source="explore", via_edges=[edge.id]
                            )
                        )

        elapsed = (time.perf_counter() - t0) * 1000
        stats = ExploreStats(
            nodes_projected=len(node_emb),
            clusters_found=len(clusters),
            candidates_evaluated=candidates_evaluated,
            edges_minted=len(minted),
            elapsed_ms=elapsed,
        )
        return scored_nodes, minted, stats

    def _coverage(self, exploit_nodes: list[ScoredNode], intention: Intention) -> float:
        """Estimate how well exploit results cover the intention predicates."""
        if not intention.predicates or not exploit_nodes:
            return 0.0

        covered = 0
        for p in intention.predicates:
            if p.embedding is None:
                continue
            p_norm = np.linalg.norm(p.embedding)
            if p_norm < 1e-10:
                continue
            best = 0.0
            for sn in exploit_nodes:
                if sn.node.embedding is not None:
                    n_norm = np.linalg.norm(sn.node.embedding)
                    if n_norm > 1e-10:
                        sim = np.dot(sn.node.embedding, p.embedding) / (n_norm * p_norm)
                        best = max(best, float(sim))
            if best >= 0.3:
                covered += 1

        total = sum(1 for p in intention.predicates if p.embedding is not None)
        return covered / total if total > 0 else 0.0

    def _fuse(
        self,
        exploit_nodes: list[ScoredNode],
        explore_nodes: list[ScoredNode],
        minted_edges: list[Hyperedge],
    ) -> list[ScoredNode]:
        """Merge exploit + explore results. Nodes found by both get a 1.2x boost."""
        by_id: dict[str, ScoredNode] = {}

        for sn in exploit_nodes:
            by_id[sn.node.id] = sn

        for sn in explore_nodes:
            if sn.node.id in by_id:
                existing = by_id[sn.node.id]
                by_id[sn.node.id] = ScoredNode(
                    node=existing.node,
                    score=max(existing.score, sn.score) * 1.2,
                    source="both",
                    via_edges=existing.via_edges + sn.via_edges,
                )
            else:
                by_id[sn.node.id] = sn

        return list(by_id.values())
