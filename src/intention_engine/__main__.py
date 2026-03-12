"""CLI for agent-native hypergraph operations.

All commands output JSON for easy parsing by agents.
Graphs auto-persist to ~/.intention-engine/<name>/.

Usage:
    python -m intention_engine <command> <graph> [args...]

Commands:
    init <graph>                              Create or load a named graph
    add-node <graph> <id> <description> [--ontology ONT]
    add-nodes <graph> --json '<json_array>'   Batch add nodes
    add-edge <graph> <label> <id1> <id2> [id3...]
    search <graph> <intention> [--top N]      Search with intention
    stats <graph>                             Graph statistics
    list-nodes <graph> [--ontology ONT]       List all nodes
    list-edges <graph> [--source SOURCE]      List all hyperedges
    explain <graph> <edge_id>                 Explain a hyperedge
    decay <graph> [--threshold T]             Prune low-weight edges
    graphs                                    List all named graphs

RAG Commands:
    ingest <graph> <path> [--recursive] [--chunk-size N]
                                              Ingest a file or directory
    ingest-text <graph> <text> [--name NAME] [--ontology ONT]
                                              Ingest raw text
    retrieve <graph> <query> [--top N] [--format text|markdown|xml] [--no-explore]
                                              Retrieve context for a query (RAG)
    documents <graph>                         List ingested documents
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure the package is importable
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _src not in sys.path:
    sys.path.insert(0, _src)


STORE_ROOT = os.path.join(os.path.expanduser("~"), ".intention-engine")


def _store_path(name: str) -> str:
    return os.path.join(STORE_ROOT, name)


def _get_engine(name: str):
    """Load or create an engine for the named graph."""
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
        # Restore embeddings — re-encode from metadata descriptions
        for node in engine.store.nodes.values():
            if node.embedding is None:
                desc = node.metadata.get("description", node.id)
                text = f"[{node.ontology}] {desc}" if node.ontology != "default" else desc
                node.embedding = engine._encode(text)

    return engine


def _save(engine, name: str):
    """Auto-save after mutations."""
    path = _store_path(name)
    os.makedirs(path, exist_ok=True)
    engine.save(path)


def _json_out(data):
    """Print JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


# === Commands ===


def cmd_init(args):
    path = _store_path(args.graph)
    os.makedirs(path, exist_ok=True)
    engine = _get_engine(args.graph)
    _json_out({
        "status": "ok",
        "graph": args.graph,
        "path": path,
        **engine.stats(),
    })


def cmd_add_node(args):
    engine = _get_engine(args.graph)
    node = engine.add_node(
        id=args.id,
        description=args.description,
        ontology=args.ontology,
        metadata=json.loads(args.metadata) if args.metadata else None,
    )
    _save(engine, args.graph)
    _json_out({
        "status": "ok",
        "node": {"id": node.id, "ontology": node.ontology, "description": args.description},
    })


def cmd_add_nodes(args):
    engine = _get_engine(args.graph)
    nodes_data = json.loads(args.json)
    if not isinstance(nodes_data, list):
        _json_out({"status": "error", "message": "Expected JSON array"})
        sys.exit(1)
    added = engine.add_nodes_batch(nodes_data)
    _save(engine, args.graph)
    _json_out({
        "status": "ok",
        "added": len(added),
        "nodes": [{"id": n.id, "ontology": n.ontology} for n in added],
    })


def cmd_add_edge(args):
    engine = _get_engine(args.graph)
    edge = engine.add_hyperedge(
        member_ids=set(args.members),
        label=args.label,
    )
    _save(engine, args.graph)
    _json_out({
        "status": "ok",
        "edge": {"id": edge.id, "label": edge.label, "members": sorted(edge.members)},
    })


def cmd_search(args):
    engine = _get_engine(args.graph)
    result = engine.search(
        intention=args.intention,
        max_results=args.top,
        explore=not args.no_explore,
    )
    _save(engine, args.graph)  # Save newly minted edges

    _json_out({
        "intention": args.intention,
        "results": [
            {
                "id": sn.node.id,
                "ontology": sn.node.ontology,
                "description": sn.node.metadata.get("description", ""),
                "score": round(sn.score, 4),
                "source": sn.source,
                "via_edges": sn.via_edges,
            }
            for sn in result.nodes
        ],
        "exploit": {
            "edges_activated": result.explanation.exploit_stats.edges_activated if result.explanation else 0,
            "nodes_reached": result.explanation.exploit_stats.nodes_reached if result.explanation else 0,
        },
        "explore": {
            "clusters_found": result.explanation.explore_stats.clusters_found if result.explanation else 0,
            "edges_minted": result.explanation.explore_stats.edges_minted if result.explanation else 0,
        },
        "minted_edges": [
            {"id": e.id, "label": e.label, "members": sorted(e.members), "coherence": round(e.coherence_score, 4)}
            for e in result.minted_edges
        ],
    })


def cmd_stats(args):
    engine = _get_engine(args.graph)
    stats = engine.stats()
    # Add ontology breakdown
    ontologies: dict[str, int] = {}
    for node in engine.store.nodes.values():
        ontologies[node.ontology] = ontologies.get(node.ontology, 0) + 1
    stats["ontologies"] = ontologies
    _json_out(stats)


def cmd_list_nodes(args):
    engine = _get_engine(args.graph)
    nodes = []
    for n in engine.store.nodes.values():
        if args.ontology and n.ontology != args.ontology:
            continue
        nodes.append({
            "id": n.id,
            "ontology": n.ontology,
            "description": n.metadata.get("description", ""),
        })
    nodes.sort(key=lambda x: x["id"])
    _json_out({"count": len(nodes), "nodes": nodes})


def cmd_list_edges(args):
    engine = _get_engine(args.graph)
    edges = []
    for e in engine.store.edges.values():
        if args.source and e.provenance.source != args.source:
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
    _json_out({"count": len(edges), "edges": edges})


def cmd_explain(args):
    engine = _get_engine(args.graph)
    info = engine.explain_edge(args.edge_id)
    if info is None:
        _json_out({"status": "error", "message": f"Edge {args.edge_id} not found"})
        sys.exit(1)
    _json_out(info)


def cmd_decay(args):
    engine = _get_engine(args.graph)
    pruned = engine.decay_edges(threshold=args.threshold)
    _save(engine, args.graph)
    _json_out({"status": "ok", "pruned": pruned, **engine.stats()})


def cmd_graphs(args):
    if not os.path.exists(STORE_ROOT):
        _json_out({"graphs": []})
        return
    graphs = []
    for name in sorted(os.listdir(STORE_ROOT)):
        path = os.path.join(STORE_ROOT, name)
        if os.path.isdir(path):
            nodes_file = os.path.join(path, "nodes.jsonl")
            edges_file = os.path.join(path, "hyperedges.jsonl")
            n_nodes = sum(1 for _ in open(nodes_file)) if os.path.exists(nodes_file) else 0
            n_edges = sum(1 for _ in open(edges_file)) if os.path.exists(edges_file) else 0
            graphs.append({"name": name, "nodes": n_nodes, "edges": n_edges})
    _json_out({"graphs": graphs})


# === RAG Commands ===


def cmd_ingest(args):
    from intention_engine.rag import IntentionRAG, RAGConfig
    config = RAGConfig(graph_name=args.graph, chunk_size=args.chunk_size)
    rag = IntentionRAG(config=config)
    result = rag.ingest(args.path, recursive=args.recursive)
    _json_out({
        "status": "ok",
        "documents": result.documents,
        "chunks": result.chunks,
        "nodes_created": result.nodes_created,
        "edges_created": result.edges_created,
        "files": result.files,
        **rag.stats(),
    })


def cmd_ingest_text(args):
    from intention_engine.rag import IntentionRAG, RAGConfig
    config = RAGConfig(graph_name=args.graph)
    rag = IntentionRAG(config=config)
    result = rag.ingest_text(args.text, name=args.name, ontology=args.ontology)
    _json_out({
        "status": "ok",
        "documents": result.documents,
        "chunks": result.chunks,
        "nodes_created": result.nodes_created,
        "edges_created": result.edges_created,
        **rag.stats(),
    })


def cmd_retrieve(args):
    from intention_engine.rag import IntentionRAG, RAGConfig
    config = RAGConfig(
        graph_name=args.graph,
        max_results=args.top,
        context_format=args.format,
    )
    rag = IntentionRAG(config=config)
    context = rag.retrieve(
        query=args.query,
        explore=not args.no_explore,
    )
    _json_out({
        "query": args.query,
        "context": context,
        "stats": rag.stats(),
    })


def cmd_documents(args):
    from intention_engine.rag import IntentionRAG, RAGConfig
    rag = IntentionRAG(RAGConfig(graph_name=args.graph))
    docs = rag.list_documents()
    _json_out({"count": len(docs), "documents": docs})


# === Argument Parsing ===


def main():
    parser = argparse.ArgumentParser(
        prog="intention_engine",
        description="Agent-native intention-driven hypergraph CLI. All output is JSON.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p = sub.add_parser("init", help="Create or load a named graph")
    p.add_argument("graph")

    # add-node
    p = sub.add_parser("add-node", help="Add a single node")
    p.add_argument("graph")
    p.add_argument("id")
    p.add_argument("description")
    p.add_argument("--ontology", default="default")
    p.add_argument("--metadata", default=None, help="JSON object of extra metadata")

    # add-nodes
    p = sub.add_parser("add-nodes", help="Batch add nodes from JSON")
    p.add_argument("graph")
    p.add_argument("--json", required=True, help='JSON array: [{"id":"...","description":"..."}]')

    # add-edge
    p = sub.add_parser("add-edge", help="Add a manual hyperedge")
    p.add_argument("graph")
    p.add_argument("label")
    p.add_argument("members", nargs="+", help="Node IDs to connect")

    # search
    p = sub.add_parser("search", help="Search with intention")
    p.add_argument("graph")
    p.add_argument("intention")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--no-explore", action="store_true", help="Exploit only, skip explore")

    # stats
    p = sub.add_parser("stats", help="Graph statistics")
    p.add_argument("graph")

    # list-nodes
    p = sub.add_parser("list-nodes", help="List all nodes")
    p.add_argument("graph")
    p.add_argument("--ontology", default=None, help="Filter by ontology")

    # list-edges
    p = sub.add_parser("list-edges", help="List all hyperedges")
    p.add_argument("graph")
    p.add_argument("--source", default=None, choices=["manual", "minted", "extracted"])

    # explain
    p = sub.add_parser("explain", help="Explain a hyperedge")
    p.add_argument("graph")
    p.add_argument("edge_id")

    # decay
    p = sub.add_parser("decay", help="Prune low-weight edges")
    p.add_argument("graph")
    p.add_argument("--threshold", type=float, default=0.01)

    # graphs
    sub.add_parser("graphs", help="List all named graphs")

    # ingest
    p = sub.add_parser("ingest", help="Ingest a file or directory")
    p.add_argument("graph")
    p.add_argument("path")
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument("--no-recursive", dest="recursive", action="store_false")
    p.add_argument("--chunk-size", type=int, default=512)

    # ingest-text
    p = sub.add_parser("ingest-text", help="Ingest raw text")
    p.add_argument("graph")
    p.add_argument("text")
    p.add_argument("--name", default="inline")
    p.add_argument("--ontology", default="text")

    # retrieve
    p = sub.add_parser("retrieve", help="Retrieve context for a query (RAG)")
    p.add_argument("graph")
    p.add_argument("query")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--format", choices=["text", "markdown", "xml"], default="text")
    p.add_argument("--no-explore", action="store_true")

    # documents
    p = sub.add_parser("documents", help="List ingested documents")
    p.add_argument("graph")

    args = parser.parse_args()

    cmd_map = {
        "init": cmd_init,
        "add-node": cmd_add_node,
        "add-nodes": cmd_add_nodes,
        "add-edge": cmd_add_edge,
        "search": cmd_search,
        "stats": cmd_stats,
        "list-nodes": cmd_list_nodes,
        "list-edges": cmd_list_edges,
        "explain": cmd_explain,
        "decay": cmd_decay,
        "graphs": cmd_graphs,
        "ingest": cmd_ingest,
        "ingest-text": cmd_ingest_text,
        "retrieve": cmd_retrieve,
        "documents": cmd_documents,
    }

    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
