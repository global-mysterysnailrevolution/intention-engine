import time as _time

import numpy as np
import scipy.sparse as sp

from .models import Hyperedge, HyperedgeProvenance, IntentionEvent, Node, _make_id


class HypergraphStore:
    """Hypergraph with two-layer storage: dict-of-sets (live) + CSR (snapshot)."""

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Hyperedge] = {}
        self._closed_edges: dict[str, Hyperedge] = {}  # archived closed edges
        # Live layer: O(1) insert/delete
        self._edge_members: dict[str, set[str]] = {}  # edge_id -> node_ids
        self._node_edges: dict[str, set[str]] = {}    # node_id -> edge_ids
        # Snapshot layer: rebuilt on demand
        self._csr: sp.csr_matrix | None = None
        self._node_index: dict[str, int] = {}  # node_id -> row index
        self._edge_index: dict[str, int] = {}  # edge_id -> col index
        self._dirty = True
        # Temporal event log (set by engine when temporal mode enabled)
        self.event_log = None

    def add_node(self, node: Node) -> None:
        self._nodes[node.id] = node
        if node.id not in self._node_edges:
            self._node_edges[node.id] = set()
        self._dirty = True

    def add_hyperedge(self, edge: Hyperedge) -> None:
        self._edges[edge.id] = edge
        self._edge_members[edge.id] = set(edge.members)
        for nid in edge.members:
            if nid not in self._node_edges:
                self._node_edges[nid] = set()
            self._node_edges[nid].add(edge.id)
        self._dirty = True

    def get_node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Hyperedge | None:
        return self._edges.get(edge_id)

    def get_node_by_index(self, idx: int) -> Node | None:
        """Reverse lookup from matrix index to node."""
        self._ensure_indexed()
        for nid, i in self._node_index.items():
            if i == idx:
                return self._nodes.get(nid)
        return None

    @property
    def nodes(self) -> dict[str, Node]:
        return self._nodes

    @property
    def edges(self) -> dict[str, Hyperedge]:
        return self._edges

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def has_similar_edge(self, member_ids: frozenset[str], threshold: float = 0.8) -> bool:
        """Check if an edge with similar membership exists (Jaccard > threshold)."""
        for eid, members in self._edge_members.items():
            jaccard = len(member_ids & members) / max(len(member_ids | members), 1)
            if jaccard > threshold:
                return True
        return False

    def _ensure_indexed(self):
        """Rebuild index mappings if needed."""
        if not self._node_index or self._dirty:
            self._node_index = {nid: i for i, nid in enumerate(sorted(self._nodes.keys()))}
            self._edge_index = {eid: i for i, eid in enumerate(sorted(self._edges.keys()))}

    def incidence_matrix(self) -> sp.csr_matrix:
        """Get the incidence matrix H (|V| x |E|). Rebuilt lazily when dirty."""
        if self._csr is not None and not self._dirty:
            return self._csr

        self._ensure_indexed()
        n_nodes = len(self._nodes)
        n_edges = len(self._edges)

        if n_nodes == 0 or n_edges == 0:
            self._csr = sp.csr_matrix((max(n_nodes, 1), max(n_edges, 1)))
            self._dirty = False
            return self._csr

        rows, cols = [], []
        for eid, members in self._edge_members.items():
            col = self._edge_index.get(eid)
            if col is None:
                continue
            for nid in members:
                row = self._node_index.get(nid)
                if row is not None:
                    rows.append(row)
                    cols.append(col)

        data = np.ones(len(rows), dtype=np.float64)
        self._csr = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_edges))
        self._dirty = False
        return self._csr

    def valid_edges_at(self, t: float) -> dict[str, Hyperedge]:
        """Return only edges valid at time *t*.

        An edge is valid when ``valid_from <= t`` and either ``valid_until``
        is ``None`` (still open) or ``valid_until > t``.  Both live and
        closed (archived) edges are considered.
        """
        result: dict[str, Hyperedge] = {}
        # Check live edges
        for eid, edge in self._edges.items():
            if edge.valid_from <= t and (edge.valid_until is None or edge.valid_until > t):
                result[eid] = edge
        # Check closed edges (they have valid_until set)
        for eid, edge in self._closed_edges.items():
            if edge.valid_from <= t and (edge.valid_until is None or edge.valid_until > t):
                result[eid] = edge
        return result

    def incidence_matrix_at(self, t: float) -> sp.csr_matrix:
        """Build incidence matrix using only edges valid at time *t*."""
        valid = self.valid_edges_at(t)

        n_nodes = len(self._nodes)
        n_valid = len(valid)

        if n_nodes == 0 or n_valid == 0:
            return sp.csr_matrix((max(n_nodes, 1), max(n_valid, 1)))

        # Build fresh index mappings for the time-scoped view
        node_index = {nid: i for i, nid in enumerate(sorted(self._nodes.keys()))}
        edge_ids_sorted = sorted(valid.keys())
        edge_index = {eid: i for i, eid in enumerate(edge_ids_sorted)}

        rows, cols = [], []
        for eid, edge in valid.items():
            col = edge_index[eid]
            for nid in edge.members:
                row = node_index.get(nid)
                if row is not None:
                    rows.append(row)
                    cols.append(col)

        data = np.ones(len(rows), dtype=np.float64)
        return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_valid))

    def mint_hyperedge(
        self,
        members: frozenset[str],
        label: str,
        coherence: float,
        intention: str = "",
        predicates: list[str] | None = None,
    ) -> Hyperedge:
        """Mint a new hyperedge from the explore phase."""
        now = _time.time()
        edge = Hyperedge(
            id=_make_id("he"),
            members=members,
            label=label,
            provenance=HyperedgeProvenance(
                source="minted",
                intention=intention,
                predicates=predicates,
            ),
            coherence_score=coherence,
            utility_context=intention,
            valid_from=now,
        )
        edge.intention_history.append(IntentionEvent(
            timestamp=now,
            intention=intention,
            action="minted",
            score=coherence,
        ))
        self.add_hyperedge(edge)
        if self.event_log is not None:
            from intention_engine.events import GraphEvent
            self.event_log.append(GraphEvent(
                event_type="edge_minted",
                entity_id=edge.id,
                data={"members": sorted(edge.members), "label": edge.label},
                intention=intention or "",
            ))
        return edge

    def reinforce_edge(self, edge_id: str) -> None:
        """Called when an edge is accessed during exploit."""
        edge = self._edges.get(edge_id)
        if edge:
            edge.access_count += 1
            edge.last_accessed = _time.time()
            edge.weight = min(1.0, edge.weight * 1.1)
            if self.event_log is not None:
                from intention_engine.events import GraphEvent
                self.event_log.append(GraphEvent(
                    event_type="edge_reinforced",
                    entity_id=edge.id,
                    data={"access_count": edge.access_count, "weight": edge.weight},
                ))

    def close_edge(self, edge_id: str) -> None:
        """Close a hyperedge by setting its valid_until timestamp.

        The edge is moved from the live ``_edges`` dict into the
        ``_closed_edges`` archive so that historical / temporal queries
        can still find it while the live layer behaves as if the edge
        was removed.
        """
        edge = self._edges.get(edge_id)
        if edge is None:
            return
        edge.valid_until = _time.time()
        edge.intention_history.append(IntentionEvent(
            timestamp=edge.valid_until,
            intention="",
            action="closed",
        ))
        # Archive before removing from live layer
        self._closed_edges[edge_id] = edge
        # Remove from live layer (same as _remove_edge)
        members = self._edge_members.pop(edge_id, set())
        for nid in members:
            if nid in self._node_edges:
                self._node_edges[nid].discard(edge_id)
        self._edges.pop(edge_id, None)
        self._dirty = True
        if self.event_log is not None:
            from intention_engine.events import GraphEvent
            self.event_log.append(GraphEvent(
                event_type="edge_closed",
                entity_id=edge_id,
                data={"valid_until": edge.valid_until},
            ))

    def decay_edges(self, half_life_days: float = 30.0, prune_threshold: float = 0.01) -> int:
        """Apply temporal decay and close low-weight edges. Returns count closed."""
        now = _time.time()
        half_life = half_life_days * 24 * 3600
        to_close = []

        for eid, edge in self._edges.items():
            # Skip already-closed edges
            if edge.valid_until is not None:
                continue
            age = now - edge.last_accessed
            base_decay = 0.5 ** (age / half_life) if half_life > 0 else 1.0
            access_boost = min(1.0, edge.access_count / 10)
            edge.weight = edge.coherence_score * (0.5 * base_decay + 0.5 * access_boost)

            if edge.weight < prune_threshold:
                to_close.append(eid)

        for eid in to_close:
            self.close_edge(eid)

        return len(to_close)

    def _remove_edge(self, edge_id: str) -> None:
        """Remove a hyperedge."""
        members = self._edge_members.pop(edge_id, set())
        for nid in members:
            if nid in self._node_edges:
                self._node_edges[nid].discard(edge_id)
        self._edges.pop(edge_id, None)
        self._dirty = True

    def save(self, path: str) -> None:
        """Save to directory."""
        import json
        import os

        os.makedirs(path, exist_ok=True)

        # Save nodes
        nodes_data = []
        for n in self._nodes.values():
            nd = {
                "id": n.id,
                "ontology": n.ontology,
                "metadata": n.metadata,
                "created_at": n.created_at,
            }
            nodes_data.append(nd)
        with open(os.path.join(path, "nodes.jsonl"), "w") as f:
            for nd in nodes_data:
                f.write(json.dumps(nd) + "\n")

        # Save edges (both live and closed)
        edges_data = []
        all_edges = list(self._edges.values()) + list(self._closed_edges.values())
        for e in all_edges:
            ed = {
                "id": e.id,
                "members": sorted(e.members),
                "label": e.label,
                "provenance": {
                    "source": e.provenance.source,
                    "intention": e.provenance.intention,
                    "predicates": e.provenance.predicates,
                },
                "coherence_score": e.coherence_score,
                "utility_context": e.utility_context,
                "access_count": e.access_count,
                "weight": e.weight,
                "last_accessed": e.last_accessed,
                "created_at": e.created_at,
                "valid_from": e.valid_from,
                "valid_until": e.valid_until,
                "intention_history": [
                    {
                        "timestamp": ie.timestamp,
                        "intention": ie.intention,
                        "action": ie.action,
                        "score": ie.score,
                    }
                    for ie in e.intention_history
                ],
            }
            edges_data.append(ed)
        with open(os.path.join(path, "hyperedges.jsonl"), "w") as f:
            for ed in edges_data:
                f.write(json.dumps(ed) + "\n")

        # Save embeddings
        node_ids = sorted(self._nodes.keys())
        embeddings = []
        for nid in node_ids:
            emb = self._nodes[nid].embedding
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(1))  # placeholder
        if embeddings:
            np.save(
                os.path.join(path, "embeddings.npy"),
                np.array(embeddings, dtype=object),
                allow_pickle=True,
            )

    def load(self, path: str) -> None:
        """Load from directory."""
        import json
        import os

        nodes_path = os.path.join(path, "nodes.jsonl")
        if os.path.exists(nodes_path):
            with open(nodes_path) as f:
                for line in f:
                    nd = json.loads(line)
                    node = Node(
                        id=nd["id"],
                        ontology=nd.get("ontology", "default"),
                        metadata=nd.get("metadata", {}),
                        created_at=nd.get("created_at", 0),
                    )
                    self.add_node(node)

        edges_path = os.path.join(path, "hyperedges.jsonl")
        if os.path.exists(edges_path):
            with open(edges_path) as f:
                for line in f:
                    ed = json.loads(line)
                    prov_data = ed.get("provenance", {})
                    prov = HyperedgeProvenance(
                        source=prov_data.get("source", "manual"),
                        intention=prov_data.get("intention"),
                        predicates=prov_data.get("predicates"),
                    )
                    # Restore intention history
                    history = []
                    for ih in ed.get("intention_history", []):
                        history.append(IntentionEvent(
                            timestamp=ih.get("timestamp", 0.0),
                            intention=ih.get("intention", ""),
                            action=ih.get("action", "minted"),
                            score=ih.get("score", 0.0),
                        ))
                    edge = Hyperedge(
                        id=ed["id"],
                        members=frozenset(ed["members"]),
                        label=ed.get("label", ""),
                        provenance=prov,
                        coherence_score=ed.get("coherence_score", 0),
                        utility_context=ed.get("utility_context", ""),
                        access_count=ed.get("access_count", 0),
                        weight=ed.get("weight", 1.0),
                        last_accessed=ed.get("last_accessed", 0),
                        created_at=ed.get("created_at", 0),
                        valid_from=ed.get("valid_from", 0.0),
                        valid_until=ed.get("valid_until", None),
                        intention_history=history,
                    )
                    if edge.valid_until is not None:
                        # Closed edge: archive only, do not add to live layer
                        self._closed_edges[edge.id] = edge
                    else:
                        self.add_hyperedge(edge)
