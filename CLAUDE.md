# Intention Engine — Agent Instructions

## What This Is

A drop-in RAG replacement using intention-driven hypergraph search. Instead of
embedding chunks into a flat vector store, it builds a hypergraph where documents,
sections, and chunks are connected by structural and semantic hyperedges. Each
search discovers and persists new cross-document connections. The graph gets
smarter with use.

## RAG Usage (Primary)

```bash
# Ingest files or directories — auto-chunks with structural awareness
python -m intention_engine ingest <graph> ./docs/
python -m intention_engine ingest <graph> ./src/auth.py --chunk-size 1024
python -m intention_engine ingest-text <graph> "raw text" --name "label"

# Retrieve context for LLM — the core RAG operation
python -m intention_engine retrieve <graph> "how does auth work?" --top 10
python -m intention_engine retrieve <graph> "security policies" --format xml
python -m intention_engine retrieve <graph> "fast lookup" --no-explore

# Inspect
python -m intention_engine documents <graph>
python -m intention_engine stats <graph>
python -m intention_engine list-edges <graph> --source minted
```

## Low-Level Graph Commands

```bash
python -m intention_engine graphs
python -m intention_engine init <graph>
python -m intention_engine add-node <graph> <id> "<description>" --ontology <type>
python -m intention_engine add-nodes <graph> --json '[...]'
python -m intention_engine add-edge <graph> "<label>" <id1> <id2> [id3...]
python -m intention_engine search <graph> "<intention>" --top 20
python -m intention_engine list-nodes <graph> [--ontology <type>]
python -m intention_engine explain <graph> <edge_id>
python -m intention_engine decay <graph>
```

## MCP Server

```bash
python -m intention_engine.mcp_server  # requires: pip install mcp[cli]
```

13 tools: `intention_add_node`, `intention_add_nodes`, `intention_add_edge`,
`intention_search`, `intention_stats`, `intention_list_nodes`, `intention_list_edges`,
`intention_explain`, `intention_graphs`, `intention_ingest`, `intention_ingest_text`,
`intention_retrieve`, `intention_documents`

## Why This Replaces RAG

| Traditional RAG | Intention Engine |
|----------------|-----------------|
| Chunks retrieved independently | Hyperedges connect related chunks across documents |
| Same query → same results forever | Each query mints new structure; search improves |
| Chunking destroys document hierarchy | Multi-level: document → section → chunk nodes |
| No cross-document awareness | Explore phase discovers cross-doc connections |
| "Lost in the middle" redundancy | Coherence scoring surfaces diverse results |

## How Search Works

1. **EXPLOIT**: Traverse existing hyperedge structure (2 SpMV operations, <1ms)
2. **EXPLORE**: Project nodes through intention lens → cluster → score coherence → mint
3. **FUSE**: Combine results from both phases, boost nodes found by both
4. **PERSIST**: Minted hyperedges are saved — future queries benefit

## Context Formats

- `text`: `[filename:lines (section)]` headers with plain text (default)
- `markdown`: `### Source N: filename > section` with content
- `xml`: `<context><chunk source="..." section="..." lines="...">` tags

## Test & Dev

```bash
python -m pytest tests/ -v             # 353 tests
python examples/manufacturing_demo.py  # Discovery demo
```

## No ML Dependencies Required

Built-in `HashEncoder` uses deterministic word-similarity embeddings.
No GPU, no model downloads. Semantic signal comes from descriptions.
For production: `engine.set_encoder(SentenceTransformer(...).encode)`
