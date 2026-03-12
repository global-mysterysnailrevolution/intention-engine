"""MCP server exposing IntentionEngine as agent tools.

Run with:
    python -m intention_engine.mcp_server

Provides tools for building, querying, and managing hypergraphs.
All state auto-persists to ~/.intention-engine/<graph_name>/.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the package is importable when run as a module
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _src not in sys.path:
    sys.path.insert(0, _src)


STORE_ROOT = os.path.join(os.path.expanduser("~"), ".intention-engine")

# Cache engines by graph name so tools can share state within a session
_engines: dict[str, object] = {}


def _store_path(name: str) -> str:
    return os.path.join(STORE_ROOT, name)


def _get_engine(name: str):
    """Load or create an engine for the named graph."""
    if name in _engines:
        return _engines[name]

    from intention_engine import IntentionEngine, EngineConfig
    from intention_engine.encoder import HashEncoder

    config = EngineConfig(
        min_coherence=0.2,
        explore_budget=50,
        utility_threshold_percentile=70.0,
    )
    engine = IntentionEngine(config=config)
    engine.set_encoder(HashEncoder(dim=384))

    path = _store_path(name)
    if os.path.exists(os.path.join(path, "nodes.jsonl")):
        engine.load(path)
        for node in engine.store.nodes.values():
            if node.embedding is None:
                desc = node.metadata.get("description", node.id)
                text = f"[{node.ontology}] {desc}" if node.ontology != "default" else desc
                node.embedding = engine._encode(text)

    _engines[name] = engine
    return engine


def _save(engine, name: str):
    path = _store_path(name)
    os.makedirs(path, exist_ok=True)
    engine.save(path)


def main():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print(
            "MCP server requires the 'mcp' package. Install with:\n"
            "  pip install mcp[cli]\n\n"
            "Alternatively, use the CLI interface:\n"
            "  python -m intention_engine <command> <graph> [args]",
            file=sys.stderr,
        )
        sys.exit(1)

    mcp = FastMCP(
        "intention-engine",
        description="Intention-driven hypergraph for cross-ontology knowledge discovery. "
        "Agents can build knowledge graphs where searching creates new structure.",
    )

    @mcp.tool()
    def intention_add_node(
        graph: str,
        node_id: str,
        description: str,
        ontology: str = "default",
    ) -> str:
        """Add a node to a named hypergraph. The node is embedded from its description.

        Args:
            graph: Name of the hypergraph (created if it doesn't exist)
            node_id: Unique identifier for the node
            description: Natural language description (used for embedding)
            ontology: Source ontology/category tag (e.g., "equipment", "process")
        """
        engine = _get_engine(graph)
        node = engine.add_node(id=node_id, description=description, ontology=ontology)
        _save(engine, graph)
        return json.dumps({
            "status": "ok",
            "node": {"id": node.id, "ontology": node.ontology},
            "stats": engine.stats(),
        })

    @mcp.tool()
    def intention_add_nodes(graph: str, nodes_json: str) -> str:
        """Batch add multiple nodes to a hypergraph.

        Args:
            graph: Name of the hypergraph
            nodes_json: JSON array of objects, each with "id", "description", and optional "ontology"
                Example: [{"id": "cnc_1", "description": "CNC mill", "ontology": "equipment"}]
        """
        engine = _get_engine(graph)
        nodes_data = json.loads(nodes_json)
        added = engine.add_nodes_batch(nodes_data)
        _save(engine, graph)
        return json.dumps({
            "status": "ok",
            "added": len(added),
            "nodes": [{"id": n.id, "ontology": n.ontology} for n in added],
        })

    @mcp.tool()
    def intention_add_edge(
        graph: str,
        label: str,
        member_ids: str,
    ) -> str:
        """Manually add a known hyperedge connecting multiple nodes.

        Args:
            graph: Name of the hypergraph
            label: Human-readable label for the relationship
            member_ids: Comma-separated node IDs to connect (e.g., "node1,node2,node3")
        """
        engine = _get_engine(graph)
        ids = {m.strip() for m in member_ids.split(",")}
        edge = engine.add_hyperedge(member_ids=ids, label=label)
        _save(engine, graph)
        return json.dumps({
            "status": "ok",
            "edge": {"id": edge.id, "label": edge.label, "members": sorted(edge.members)},
        })

    @mcp.tool()
    def intention_search(
        graph: str,
        intention: str,
        max_results: int = 20,
        explore: bool = True,
        valid_at: float | None = None,
    ) -> str:
        """Search the hypergraph driven by an intention.

        This is the core operation. It:
        1. EXPLOITS existing hyperedge structure (fast traversal)
        2. EXPLORES for new connections by projecting nodes through the intention lens
        3. MINTS new hyperedges when coherent groups are discovered
        4. Returns ranked results from both phases

        The graph grows richer with each search — discovered connections persist.

        Args:
            graph: Name of the hypergraph
            intention: Natural language description of what you're looking for
            max_results: Maximum number of results to return
            explore: Whether to run the explore phase (set False for exploit-only)
            valid_at: Restrict exploit phase to edges valid at this Unix timestamp (optional)
        """
        engine = _get_engine(graph)
        result = engine.search(intention=intention, max_results=max_results, explore=explore, valid_at=valid_at)
        _save(engine, graph)

        return json.dumps({
            "intention": intention,
            "results": [
                {
                    "id": sn.node.id,
                    "ontology": sn.node.ontology,
                    "description": sn.node.metadata.get("description", ""),
                    "score": round(sn.score, 4),
                    "source": sn.source,
                }
                for sn in result.nodes
            ],
            "minted_edges": [
                {
                    "id": e.id,
                    "label": e.label,
                    "members": sorted(e.members),
                    "coherence": round(e.coherence_score, 4),
                }
                for e in result.minted_edges
            ],
            "stats": {
                "exploit_edges_activated": result.explanation.exploit_stats.edges_activated if result.explanation else 0,
                "explore_edges_minted": result.explanation.explore_stats.edges_minted if result.explanation else 0,
            },
        })

    @mcp.tool()
    def intention_stats(graph: str) -> str:
        """Get statistics for a named hypergraph.

        Args:
            graph: Name of the hypergraph
        """
        engine = _get_engine(graph)
        stats = engine.stats()
        ontologies: dict[str, int] = {}
        for node in engine.store.nodes.values():
            ontologies[node.ontology] = ontologies.get(node.ontology, 0) + 1
        stats["ontologies"] = ontologies
        return json.dumps(stats)

    @mcp.tool()
    def intention_list_nodes(graph: str, ontology: str = "") -> str:
        """List all nodes in a hypergraph, optionally filtered by ontology.

        Args:
            graph: Name of the hypergraph
            ontology: Filter by this ontology (empty = all)
        """
        engine = _get_engine(graph)
        nodes = []
        for n in engine.store.nodes.values():
            if ontology and n.ontology != ontology:
                continue
            nodes.append({
                "id": n.id,
                "ontology": n.ontology,
                "description": n.metadata.get("description", ""),
            })
        nodes.sort(key=lambda x: x["id"])
        return json.dumps({"count": len(nodes), "nodes": nodes})

    @mcp.tool()
    def intention_list_edges(graph: str, source: str = "") -> str:
        """List all hyperedges, optionally filtered by source (manual/minted/extracted).

        Args:
            graph: Name of the hypergraph
            source: Filter by provenance source (empty = all)
        """
        engine = _get_engine(graph)
        edges = []
        for e in engine.store.edges.values():
            if source and e.provenance.source != source:
                continue
            edges.append({
                "id": e.id,
                "label": e.label,
                "members": sorted(e.members),
                "source": e.provenance.source,
                "coherence": round(e.coherence_score, 4),
                "weight": round(e.weight, 4),
                "access_count": e.access_count,
            })
        edges.sort(key=lambda x: x["weight"], reverse=True)
        return json.dumps({"count": len(edges), "edges": edges})

    @mcp.tool()
    def intention_explain(graph: str, edge_id: str) -> str:
        """Explain how a specific hyperedge was created — its provenance, intention, and coherence.

        Args:
            graph: Name of the hypergraph
            edge_id: The hyperedge ID to explain
        """
        engine = _get_engine(graph)
        info = engine.explain_edge(edge_id)
        if info is None:
            return json.dumps({"error": f"Edge {edge_id} not found"})
        return json.dumps(info)

    @mcp.tool()
    def intention_graphs() -> str:
        """List all named hypergraphs that have been created."""
        if not os.path.exists(STORE_ROOT):
            return json.dumps({"graphs": []})
        graphs = []
        for name in sorted(os.listdir(STORE_ROOT)):
            path = os.path.join(STORE_ROOT, name)
            if os.path.isdir(path):
                nodes_file = os.path.join(path, "nodes.jsonl")
                edges_file = os.path.join(path, "hyperedges.jsonl")
                n_nodes = sum(1 for _ in open(nodes_file)) if os.path.exists(nodes_file) else 0
                n_edges = sum(1 for _ in open(edges_file)) if os.path.exists(edges_file) else 0
                graphs.append({"name": name, "nodes": n_nodes, "edges": n_edges})
        return json.dumps({"graphs": graphs})

    @mcp.tool()
    def intention_temporal_diff(graph: str, t1: float, t2: float) -> str:
        """Show what changed in the graph between two timestamps.

        Args:
            graph: Name of the hypergraph
            t1: Start timestamp (Unix epoch seconds)
            t2: End timestamp (Unix epoch seconds)
        """
        engine = _get_engine(graph)
        engine.enable_temporal()
        # Load event log
        path = _store_path(graph)
        events_path = os.path.join(path, "events.jsonl")
        if os.path.exists(events_path):
            engine._event_log.load(events_path)
        diff = engine.temporal_diff(t1, t2)
        return json.dumps({
            "t1": diff.t1, "t2": diff.t2,
            "nodes_added": diff.nodes_added,
            "nodes_removed": diff.nodes_removed,
            "edges_minted": diff.edges_minted,
            "edges_closed": diff.edges_closed,
            "edges_reinforced": diff.edges_reinforced,
            "searches_executed": diff.searches_executed,
        })

    @mcp.tool()
    def intention_edge_history(graph: str, edge_id: str) -> str:
        """Get the full intention history for a hyperedge — every intention that touched it.

        Args:
            graph: Name of the hypergraph
            edge_id: The hyperedge ID
        """
        engine = _get_engine(graph)
        history = engine.edge_history(edge_id)
        return json.dumps({"edge_id": edge_id, "history": history})

    @mcp.tool()
    def intention_graph_at(graph: str, timestamp: float) -> str:
        """Get graph statistics as they would have appeared at a specific point in time.

        Args:
            graph: Name of the hypergraph
            timestamp: Unix epoch seconds
        """
        engine = _get_engine(graph)
        stats = engine.graph_at(timestamp)
        return json.dumps(stats)

    @mcp.tool()
    def intention_ingest(graph: str, path: str, recursive: bool = True, chunk_size: int = 512) -> str:
        """Ingest a file or directory into the knowledge graph.

        Creates document, section, and chunk nodes with structural hyperedges.
        Files are chunked with structural awareness (headings, functions, paragraphs).

        Args:
            graph: Name of the hypergraph
            path: Absolute path to a file or directory to ingest
            recursive: Whether to recurse into subdirectories
            chunk_size: Target chunk size in characters
        """
        from intention_engine.rag import IntentionRAG, RAGConfig
        config = RAGConfig(graph_name=graph, chunk_size=chunk_size)
        rag = IntentionRAG(config=config)
        result = rag.ingest(path, recursive=recursive)
        return json.dumps({
            "status": "ok",
            "documents": result.documents,
            "chunks": result.chunks,
            "nodes_created": result.nodes_created,
            "edges_created": result.edges_created,
            "files": result.files,
            **rag.stats(),
        })

    @mcp.tool()
    def intention_ingest_text(graph: str, text: str, name: str = "inline", ontology: str = "text") -> str:
        """Ingest raw text into the knowledge graph.

        Args:
            graph: Name of the hypergraph
            text: The text content to ingest
            name: A label for the text
            ontology: Category tag
        """
        from intention_engine.rag import IntentionRAG, RAGConfig
        config = RAGConfig(graph_name=graph)
        rag = IntentionRAG(config=config)
        result = rag.ingest_text(text, name, ontology)
        return json.dumps({
            "status": "ok",
            "documents": result.documents,
            "chunks": result.chunks,
            **rag.stats(),
        })

    @mcp.tool()
    def intention_retrieve(
        graph: str,
        query: str,
        max_results: int = 10,
        format: str = "text",
        explore: bool = True,
    ) -> str:
        """Retrieve relevant context from the knowledge graph for a query.

        This is the core RAG replacement operation. Returns formatted context
        ready for LLM consumption. Each call potentially discovers and persists
        new cross-document connections.

        Args:
            graph: Name of the hypergraph
            query: Natural language query/intention
            max_results: Maximum chunks to retrieve
            format: Output format — "text", "markdown", or "xml"
            explore: Whether to discover new connections (True) or just use existing (False)
        """
        from intention_engine.rag import IntentionRAG, RAGConfig
        config = RAGConfig(
            graph_name=graph,
            max_results=max_results,
            context_format=format,
        )
        rag = IntentionRAG(config=config)
        context = rag.retrieve(query=query, explore=explore)
        return json.dumps({
            "query": query,
            "context": context,
            "stats": rag.stats(),
        })

    @mcp.tool()
    def intention_documents(graph: str) -> str:
        """List all documents that have been ingested into a knowledge graph.

        Args:
            graph: Name of the hypergraph
        """
        from intention_engine.rag import IntentionRAG, RAGConfig
        rag = IntentionRAG(RAGConfig(graph_name=graph))
        docs = rag.list_documents()
        return json.dumps({"count": len(docs), "documents": docs})

    mcp.run()


if __name__ == "__main__":
    main()
