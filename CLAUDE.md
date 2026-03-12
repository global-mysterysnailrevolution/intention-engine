# Intention Engine — Agent Instructions

## What This Is

An intention-driven hypergraph where searching creates knowledge. Unlike static indices,
each query potentially discovers new multi-way relationships (hyperedges) by projecting
node features through a utility lens. The graph grows richer through use.

## Agent Usage (CLI)

All commands output JSON. Graphs auto-persist to `~/.intention-engine/<name>/`.

```bash
# Create a graph and add nodes
python -m intention_engine init myproject
python -m intention_engine add-node myproject sensor_1 "Temperature sensor for CNC monitoring" --ontology equipment
python -m intention_engine add-node myproject proc_mill "5-axis CNC milling process" --ontology process

# Batch add
python -m intention_engine add-nodes myproject --json '[{"id":"a","description":"thing","ontology":"type"}]'

# Search — this is where discovery happens
python -m intention_engine search myproject "reduce defect rate in precision machining"

# Inspect
python -m intention_engine stats myproject
python -m intention_engine list-nodes myproject --ontology equipment
python -m intention_engine list-edges myproject --source minted
python -m intention_engine explain myproject he_abc123def456
python -m intention_engine graphs
```

## MCP Server

If `mcp[cli]` is installed, run as MCP server:
```bash
python -m intention_engine.mcp_server
```

Tools exposed: `intention_add_node`, `intention_add_nodes`, `intention_add_edge`,
`intention_search`, `intention_stats`, `intention_list_nodes`, `intention_list_edges`,
`intention_explain`, `intention_graphs`

## Key Concepts

- **Nodes** = entities from any ontology, embedded from their text description
- **Hyperedges** = multi-way relationships connecting 2+ nodes across any ontology boundaries
- **Intention** = natural language goal that drives search and discovery
- **Exploit** = traverse existing hyperedge structure (fast, 2 SpMV operations)
- **Explore** = project nodes through intention lens → cluster → score coherence → mint new hyperedges
- **Minting** = when explore discovers a coherent group, it creates a persistent hyperedge

## How It Works

1. You add nodes with descriptions and ontology tags
2. You search with natural language intentions
3. First search: no edges exist, so EXPLORE fires and discovers structure from raw embeddings
4. EXPLORE mints hyperedges for coherent groups it discovers
5. Next search: EXPLOIT traverses the minted edges (fast), EXPLORE may find more
6. Over time the graph gets richer — more structure means faster, more accurate searches

## Test & Dev

```bash
python -m pytest tests/ -v          # 180 tests
python examples/manufacturing_demo.py  # End-to-end demo
```

## No ML Dependencies Required

The built-in `HashEncoder` provides deterministic, word-similarity-based embeddings.
No GPU, no model downloads. The semantic signal comes from how you describe nodes.
For production use, swap in sentence-transformers via `engine.set_encoder()`.
