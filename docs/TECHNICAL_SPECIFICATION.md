# Intention Engine: Technical Specification

**Version**: 0.1.0
**Status**: Design Complete
**Date**: 2026-03-11

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Core Thesis](#2-core-thesis)
3. [Foundational Concepts](#3-foundational-concepts)
4. [System Architecture](#4-system-architecture)
5. [Data Model](#5-data-model)
6. [Algorithm Specification](#6-algorithm-specification)
7. [Embedding Pipeline](#7-embedding-pipeline)
8. [Two-Phase Search](#8-two-phase-search)
9. [Hyperedge Minting Protocol](#9-hyperedge-minting-protocol)
10. [Cold Start Protocol](#10-cold-start-protocol)
11. [Learning Loop](#11-learning-loop)
12. [Computational Complexity](#12-computational-complexity)
13. [API Surface](#13-api-surface)
14. [Dependencies & Building Blocks](#14-dependencies--building-blocks)
15. [Failure Modes & Mitigations](#15-failure-modes--mitigations)

---

## 1. Problem Statement

### 1.1 The Attention-Intention Gap

Transformer attention operates over a fixed topology. Whether that topology is token
positions in a sequence, documents in a vector store, or nodes in a knowledge graph,
the structure was determined at write-time — at indexing. At query-time, attention can
weight what's reachable, but it cannot restructure what's reachable.

This creates three failure modes:

1. **Cross-ontology blindness**: Entities in different taxonomic branches (e.g.,
   "annealing" in metallurgy vs. "simulated annealing" in optimization) are invisible
   to each other if the index places them in separate partitions. No amount of
   attention weighting can bridge a structural gap.

2. **Hierarchical rigidity**: Fixed hierarchies force traversal through intermediate
   levels. A CEO-level goal ("reduce defect rate") cannot directly surface a specific
   sensor calibration parameter without traversing org → department → team → process →
   parameter. The utility gradient is steep, but the index path is long.

3. **Static granularity**: The same entity has different relevant representations
   depending on the query context. A CNC mill is simultaneously a "capability" (to a
   buyer), a "maintenance liability" (to an operator), and a "revenue unit" (to
   finance). Fixed indices commit to one granularity at write-time.

### 1.2 What's Missing

**Attention** answers: "Given what's visible, what's relevant?"
**Intention** answers: "Given what I'm trying to do, what *should be* visible?"

The gap between these two questions is where useful knowledge gets stranded. Current
retrieval-augmented systems (RAG, GraphRAG, HyperGraphRAG) partially address this via
query reformulation and multi-hop traversal, but they are fundamentally patches on a
static index. The index structure itself never changes in response to use.

### 1.3 The Goal

Build a system where:

- **Intention is not a query over an index — it is an operator that generates new
  structure within the index.**
- The knowledge substrate (a hypergraph) grows richer through use.
- Cold-start is possible: even with zero prior structure, intention can discover
  connections from raw node features.
- Cross-ontology bridging, hierarchical level-jumping, and dynamic granularity are
  not special cases — they are the default mode of operation.

---

## 2. Core Thesis

> **Intention, operating on a hypergraph substrate, generates new multi-way
> relationships (hyperedges) by projecting node features through utility lenses and
> detecting coherence. The act of searching is itself the act of learning.**

This thesis has four components:

1. **Hypergraph substrate**: The knowledge base is a hypergraph where nodes are
   entities and hyperedges are multi-way relationships. Hyperedges can span arbitrary
   ontology boundaries and hierarchy levels.

2. **Intention as generative operator**: Intention is not a selector that filters
   pre-existing structure. It is an operator that projects node features into a
   utility-aligned space, discovers multi-way coherence, and mints new hyperedges.

3. **Two-phase search**: Every query has an EXPLOIT phase (traverse existing
   structure) and an EXPLORE phase (generate new structure). Phase 1 is fast; Phase 2
   is where discovery happens.

4. **Learning through use**: Validated hyperedges from the EXPLORE phase are committed
   to persistent storage. The hypergraph grows richer with each query. Over time, the
   EXPLOIT phase covers more cases and the EXPLORE phase fires less frequently — but
   it is always available for genuinely novel intentions.

---

## 3. Foundational Concepts

### 3.1 Hypergraphs

A **hypergraph** H = (V, E) consists of:
- V: a finite set of **vertices** (nodes)
- E: a family of non-empty subsets of V, called **hyperedges**

Unlike a graph where each edge connects exactly 2 nodes, a hyperedge can connect any
number of nodes. This is the critical property: multi-way relationships are first-class.

**Incidence matrix**: H ∈ {0,1}^{|V|×|E|} where H[v,e] = 1 iff v ∈ e.

**Dual hypergraph**: H* = (E, V*) where V*_e = {e' ∈ E : v ∈ e'} for each v ∈ V.
The dual swaps the roles of nodes and hyperedges. Querying the primal by intention
selects hyperedges; querying the dual selects nodes.

### 3.2 Intention

An **intention** is a goal-directed signal expressed as natural language that can be
decomposed into component predicates. Each predicate defines a utility dimension.

```
Intention: "reduce contamination in titanium processing"

Predicates:
  p₁: "interacts with contaminants"    → utility dimension 1
  p₂: "compatible with titanium"       → utility dimension 2
  p₃: "acts as barrier or filter"      → utility dimension 3
```

Intention creates a **projection function** that maps node features into a
utility-aligned subspace. In this subspace, nodes that are "useful for this intention"
cluster together regardless of their original ontology.

### 3.3 Utility Projection

Given:
- Node feature vectors: {f_v ∈ ℝ^d : v ∈ V}
- Intention predicates: {p₁, ..., p_k} with embeddings {e₁, ..., e_k} ∈ ℝ^d

The **utility vector** of node v under intention I is:

```
u(v | I) = [sim(f_v, e₁), sim(f_v, e₂), ..., sim(f_v, e_k)] ∈ ℝ^k
```

where sim(·,·) is cosine similarity in embedding space.

This maps each node from its d-dimensional feature space into a k-dimensional utility
space defined by the intention. Clustering in utility space reveals multi-way coherence.

### 3.4 Multi-Way Coherence

Pairwise similarity is **provably insufficient** for detecting multi-way coherence
(DHNE, AAAI 2018). A set of nodes {A, B, C} may form a coherent group even when no
individual pair is unusually similar.

Multi-way coherence requires a **set function** that scores a candidate set of nodes
as a whole, not as pairs:

```
coherence({v₁, ..., v_m}) = SetScore(u(v₁|I), ..., u(v_m|I))
```

where SetScore is a learned permutation-invariant function (attention-based
aggregation, DeepSets, or a simpler geometric measure like minimum enclosing ball
diameter in utility space).

### 3.5 Hyperedge Minting

When the EXPLORE phase discovers a coherent group of nodes that is not already
represented by an existing hyperedge, it **mints** a new hyperedge:

1. The candidate node set is scored for coherence
2. If above threshold, a new hyperedge is created
3. The hyperedge is tagged with the intention that generated it (provenance)
4. The hyperedge is committed to persistent storage
5. Future queries benefit from this structure existing

This is the mechanism by which searching creates knowledge.

---

## 4. System Architecture

### 4.1 Component Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         IntentionEngine                              │
│                                                                      │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Intention     │  │  Embedding   │  │  HypergraphStore         │  │
│  │  Decomposer   │  │  Pipeline    │  │                          │  │
│  │               │  │              │  │  ┌──────────────────┐    │  │
│  │  NL → pred-   │  │  Node enc.   │  │  │ NodeStore        │    │  │
│  │  icates via   │  │  Predicate   │  │  │ (features+meta)  │    │  │
│  │  LLM or rule  │  │  encoding    │  │  └──────────────────┘    │  │
│  │  decomposition│  │  Similarity  │  │  ┌──────────────────┐    │  │
│  └───────┬───────┘  └──────┬───────┘  │  │ HyperedgeStore   │    │  │
│          │                 │          │  │ (members+prov.)  │    │  │
│          ▼                 ▼          │  └──────────────────┘    │  │
│  ┌──────────────────────────────┐    │  ┌──────────────────┐    │  │
│  │       Utility Projector      │    │  │ IncidenceMatrix  │    │  │
│  │                              │    │  │ (sparse CSR)     │    │  │
│  │  intention → predicates →    │    │  └──────────────────┘    │  │
│  │  project nodes into utility  │    └──────────────────────────┘  │
│  │  space                       │                                   │
│  └──────────────┬───────────────┘                                   │
│                 │                                                    │
│                 ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Two-Phase Search                           │   │
│  │                                                              │   │
│  │  ┌─────────────────────┐    ┌─────────────────────────────┐  │   │
│  │  │   Phase 1: EXPLOIT  │    │   Phase 2: EXPLORE          │  │   │
│  │  │                     │    │                             │  │   │
│  │  │  PPR over existing  │    │  Project → Cluster →       │  │   │
│  │  │  hypergraph using   │    │  Score coherence →          │  │   │
│  │  │  intention weights  │    │  Mint new hyperedges        │  │   │
│  │  │                     │    │                             │  │   │
│  │  │  Fast: 2 SpMV ops   │    │  Slower: embedding +       │  │   │
│  │  │  Returns known       │    │  clustering + scoring      │  │   │
│  │  │  connections         │    │  Returns novel connections │  │   │
│  │  └─────────┬───────────┘    └──────────────┬──────────────┘  │   │
│  │            │                                │                 │   │
│  │            └──────────┬─────────────────────┘                 │   │
│  │                       ▼                                       │   │
│  │              ┌────────────────┐                                │   │
│  │              │  Result Fusion │                                │   │
│  │              └────────┬───────┘                                │   │
│  └───────────────────────┼───────────────────────────────────────┘   │
│                          │                                           │
│                          ▼                                           │
│                 ┌────────────────┐                                    │
│                 │ Hyperedge      │                                    │
│                 │ Minting &      │──────► back to HypergraphStore    │
│                 │ Validation     │        (learning loop)             │
│                 └────────────────┘                                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

1. **Input**: Intention (natural language string) + optional scope constraints
2. **Decompose**: Intention → predicate list via LLM or rule-based decomposition
3. **Encode**: Predicates → embedding vectors via sentence encoder
4. **Project**: Node features × predicate embeddings → utility vectors
5. **Exploit**: Score existing hyperedges by intention relevance → traverse
6. **Explore**: Cluster nodes in utility space → detect coherence → generate candidates
7. **Score**: Evaluate candidate hyperedges for multi-way coherence
8. **Mint**: Commit validated hyperedges to the hypergraph
9. **Fuse**: Combine exploit + explore results, deduplicate, rank
10. **Output**: Ranked list of relevant nodes + discovered hyperedges + explanation

### 4.3 Storage Model

```
PERSISTENT (survives restarts):
  ├── nodes.jsonl          # Node ID, features, metadata, ontology tag
  ├── hyperedges.jsonl     # Hyperedge ID, member node IDs, provenance, score
  ├── incidence.npz        # Sparse incidence matrix (scipy CSR)
  └── embeddings.npy       # Node embedding matrix (dense, n × d)

EPHEMERAL (per-query):
  ├── utility_vectors      # Projected node features for current intention
  ├── candidate_clusters   # Potential new hyperedges
  └── coherence_scores     # Scores for candidates

INDEXES (rebuilt from persistent):
  ├── faiss_index          # ANN index over node embeddings
  └── edge_index           # Mapping from node pairs to shared hyperedges
```

---

## 5. Data Model

### 5.1 Node

```python
@dataclass
class Node:
    id: str                          # Unique identifier
    features: np.ndarray             # Dense feature vector (d-dimensional)
    metadata: dict[str, Any]         # Arbitrary key-value metadata
    ontology: str                    # Source ontology identifier
    created_at: float                # Unix timestamp
    embedding: np.ndarray | None     # Cached sentence embedding (set by pipeline)
```

Nodes represent entities from any ontology. The `ontology` field tags the source
taxonomy but does not constrain which hyperedges a node can participate in. Nodes
from different ontologies can share hyperedges — this is the cross-ontology bridging
mechanism.

The `features` vector is the node's intrinsic representation. It can be:
- A pre-computed embedding from a domain encoder
- A bag-of-properties vector
- A concatenation of heterogeneous features with masking

The `embedding` field is a cached sentence-transformer encoding of the node's textual
description, used for similarity operations. It is populated lazily by the embedding
pipeline.

### 5.2 Hyperedge

```python
@dataclass
class Hyperedge:
    id: str                          # Unique identifier
    members: frozenset[str]          # Set of member node IDs
    label: str                       # Human-readable label
    provenance: HyperedgeProvenance  # How this edge was created
    coherence_score: float           # Multi-way coherence score at creation
    utility_context: str             # Intention that generated this edge
    access_count: int                # Times accessed in exploit phase
    last_accessed: float             # Unix timestamp
    created_at: float                # Unix timestamp
    weight: float                    # Current weight (decays if unused)

@dataclass
class HyperedgeProvenance:
    source: Literal["manual", "extracted", "minted"]
    intention: str | None            # The intention that created this (if minted)
    predicates: list[str] | None     # Decomposed predicates (if minted)
    coherence_method: str            # How coherence was scored
    parent_edges: list[str] | None   # If this was composed from other edges
```

Hyperedges are the primary unit of learned knowledge. The `provenance` field tracks
exactly how each hyperedge was created:
- **manual**: Inserted by a human or external system
- **extracted**: Generated by an LLM extraction pipeline (like HyperGraphRAG)
- **minted**: Generated during the EXPLORE phase of a search

The `weight` field implements temporal decay: hyperedges that are accessed frequently
maintain high weight; unused hyperedges decay toward zero and are eventually pruned.

### 5.3 Intention

```python
@dataclass
class Intention:
    raw: str                         # Original natural language
    predicates: list[Predicate]      # Decomposed utility predicates
    embedding: np.ndarray            # Sentence embedding of raw intention
    scope: SearchScope | None        # Optional constraints

@dataclass
class Predicate:
    text: str                        # Natural language predicate
    embedding: np.ndarray            # Sentence embedding
    weight: float                    # Relative importance (0-1)

@dataclass
class SearchScope:
    ontologies: set[str] | None      # Limit to specific ontologies (None = all)
    max_depth: int                   # Max traversal depth in exploit phase
    min_coherence: float             # Minimum coherence threshold for minting
    explore_budget: int              # Max candidate hyperedges to evaluate
```

### 5.4 Incidence Matrix

The incidence matrix H ∈ {0,1}^{|V|×|E|} is the core computational structure.
Stored as a scipy CSR sparse matrix for efficient arithmetic.

```python
class IncidenceMatrix:
    """Sparse incidence matrix with dynamic updates."""

    _live: dict[str, set[str]]       # edge_id → set(node_ids) — live updates
    _csr: sp.csr_matrix | None       # Cached CSR matrix — rebuilt on demand
    _node_index: dict[str, int]      # node_id → matrix row index
    _edge_index: dict[str, int]      # edge_id → matrix column index
    _dirty: bool                     # Whether _csr needs rebuilding
```

The two-layer architecture:
1. **Live layer**: dict-of-sets for O(1) insertion/deletion
2. **Snapshot layer**: CSR matrix for O(nnz) sparse algebra

The CSR is rebuilt lazily — only when arithmetic operations are requested. This avoids
the O(nnz log nnz) conversion cost on every insertion.

### 5.5 Search Result

```python
@dataclass
class SearchResult:
    nodes: list[ScoredNode]          # Ranked relevant nodes
    exploited_edges: list[Hyperedge] # Existing edges that were traversed
    minted_edges: list[Hyperedge]    # New edges discovered during explore
    explanation: SearchExplanation    # How results were found

@dataclass
class ScoredNode:
    node: Node
    score: float                     # Relevance score
    source: Literal["exploit", "explore", "both"]
    via_edges: list[str]             # Which hyperedges connected this node

@dataclass
class SearchExplanation:
    intention: Intention
    exploit_stats: ExploitStats
    explore_stats: ExploreStats

@dataclass
class ExploitStats:
    edges_scored: int
    edges_activated: int
    nodes_reached: int
    elapsed_ms: float

@dataclass
class ExploreStats:
    nodes_projected: int
    clusters_found: int
    candidates_evaluated: int
    edges_minted: int
    elapsed_ms: float
```

---

## 6. Algorithm Specification

### 6.1 Intention Decomposition

**Input**: Raw natural language intention string
**Output**: List of weighted predicates

**Method**: Compositional decomposition via either:

1. **LLM-based** (higher quality, higher latency ~200ms):
   - Prompt an instruction-tuned LLM to decompose the intention into orthogonal
     utility predicates
   - Each predicate should describe a single dimension of utility
   - LLM assigns relative weights

2. **Rule-based** (lower quality, <1ms):
   - Parse intention into noun phrases and verb phrases
   - Each noun phrase becomes a domain predicate
   - Each verb phrase becomes an action predicate
   - Equal weights

**Formal specification**:

```
DECOMPOSE(intention: str) → list[Predicate]:
  IF llm_available:
    response = LLM.complete(
      system: "Decompose the following intention into 2-6 orthogonal utility
               predicates. Each predicate should describe one dimension of what
               makes something useful for this intention. Return JSON array of
               {text, weight} where weights sum to 1.0."
      user: intention
    )
    RETURN parse_predicates(response)
  ELSE:
    chunks = extract_noun_and_verb_phrases(intention)
    RETURN [Predicate(text=c, weight=1.0/len(chunks)) for c in chunks]
```

### 6.2 Utility Projection

**Input**: Set of nodes V, list of predicates P with embeddings
**Output**: Utility matrix U ∈ ℝ^{|V|×|P|}

**Method**:

```
PROJECT(V, P) → U:
  # Encode predicates if not already done
  FOR p in P:
    IF p.embedding is None:
      p.embedding = sentence_encoder.encode(p.text)

  # Build node embedding matrix (n × d)
  F = stack([v.embedding for v in V])       # (n, d)

  # Build predicate embedding matrix (k × d)
  E = stack([p.embedding for p in P])       # (k, d)

  # Compute utility matrix via batched cosine similarity
  # Normalize both matrices
  F_norm = F / norm(F, axis=1, keepdims=True)
  E_norm = E / norm(E, axis=1, keepdims=True)

  # U[i,j] = cosine_similarity(node_i, predicate_j)
  U = F_norm @ E_norm.T                     # (n, k) — single matmul

  # Apply predicate weights
  W = array([p.weight for p in P])           # (k,)
  U_weighted = U * W                         # (n, k) broadcast

  RETURN U_weighted
```

**Complexity**: O(n·d·k) for the matmul. With n=100K nodes, d=384, k=5 predicates:
~192M FLOPs, roughly 1ms on modern hardware.

### 6.3 Exploit Phase: Weighted Hypergraph Traversal

**Input**: Intention I, HypergraphStore G, max results
**Output**: Set of scored nodes from existing structure

**Method**: Intention-weighted Personalized PageRank over the hypergraph.

```
EXPLOIT(I, G) → list[ScoredNode]:
  H = G.incidence_matrix()                  # (|V|, |E|) sparse CSR

  # Step 1: Score hyperedges by relevance to intention
  edge_embeddings = G.get_edge_embeddings()  # (|E|, d)
  w = cosine_similarity(I.embedding, edge_embeddings)  # (|E|,)
  w = relu(w)                                # Zero out negative scores
  w = w / sum(w)                             # Normalize to distribution

  # Step 2: Compute node relevance via weighted hyperedge membership
  #   node_scores[v] = sum over edges e: H[v,e] * w[e]
  #   This is a single SpMV: H @ w
  node_scores = H @ w                        # (|V|,) — sparse matmul

  # Step 3: Two-hop expansion (primal → dual → primal)
  #   "What other hyperedges contain my high-scoring nodes?"
  #   This discovers edges the intention didn't directly match
  #   but that share structure with relevant nodes
  node_weights = node_scores / max(sum(node_scores), 1e-10)
  emergent_edge_scores = H.T @ node_weights  # (|E|,) — dual projection

  #   Expand back: nodes in these emergent edges
  expanded_scores = H @ emergent_edge_scores # (|V|,)

  # Step 4: Combine direct and emergent scores
  alpha = 0.7  # Weight toward direct matches
  final_scores = alpha * node_scores + (1 - alpha) * expanded_scores

  # Step 5: Rank and return top-k
  top_k = argsort(final_scores)[-max_results:][::-1]
  RETURN [ScoredNode(G.get_node(i), final_scores[i], "exploit") for i in top_k]
```

**Complexity**: Two SpMV operations, each O(nnz). Total: O(2 · nnz).
For 100K nodes, 50K hyperedges, avg 5 members each → nnz ≈ 250K → ~0.5ms.

### 6.4 Explore Phase: Intention-Driven Discovery

**Input**: Intention I with predicates, all nodes V, existing hyperedges E
**Output**: Set of candidate new hyperedges with coherence scores

This is where intention operates as a generative operator.

```
EXPLORE(I, V, E) → list[CandidateHyperedge]:
  # Step 1: Project all nodes into utility space
  U = PROJECT(V, I.predicates)               # (n, k)

  # Step 2: Identify high-utility nodes (threshold or top-n)
  utility_magnitude = norm(U, axis=1)        # (n,)
  threshold = percentile(utility_magnitude, 80)  # Top 20% by utility
  active_mask = utility_magnitude >= threshold
  U_active = U[active_mask]                  # (m, k) where m << n
  active_nodes = V[active_mask]

  # Step 3: Cluster in utility space
  #   Nodes that are close in utility space are candidates for
  #   belonging to the same hyperedge
  clusters = cluster_utility_space(U_active)

  # Step 4: Score each cluster for multi-way coherence
  candidates = []
  FOR cluster in clusters:
    members = active_nodes[cluster.indices]

    # Skip if this group already exists as a hyperedge
    IF G.has_similar_edge(members, similarity_threshold=0.8):
      CONTINUE

    # Score multi-way coherence
    score = coherence_score(U_active[cluster.indices])

    IF score >= I.scope.min_coherence:
      candidate = CandidateHyperedge(
        members=frozenset(m.id for m in members),
        coherence=score,
        label=generate_label(members, I),
        intention=I.raw,
        predicates=[p.text for p in I.predicates]
      )
      candidates.append(candidate)

  RETURN candidates
```

### 6.5 Clustering in Utility Space

**Input**: Utility vectors U_active ∈ ℝ^{m×k}
**Output**: Set of clusters (each a set of indices)

Multiple strategies, selected based on data characteristics:

```
cluster_utility_space(U) → list[Cluster]:
  m, k = U.shape

  IF m < 50:
    # Small: exhaustive agglomerative
    RETURN agglomerative_clustering(U, distance='cosine',
                                     linkage='average',
                                     distance_threshold=0.5)

  ELIF m < 5000:
    # Medium: HDBSCAN (density-based, finds variable-size clusters)
    RETURN hdbscan(U, min_cluster_size=3, min_samples=2)

  ELSE:
    # Large: FAISS k-means → refine with HDBSCAN per partition
    k_coarse = max(10, int(sqrt(m)))
    coarse = faiss_kmeans(U, k_coarse)
    refined = []
    FOR partition in coarse.partitions:
      IF len(partition) >= 3:
        sub = hdbscan(U[partition], min_cluster_size=3)
        refined.extend(sub)
    RETURN refined
```

### 6.6 Multi-Way Coherence Scoring

**Input**: Utility vectors of a candidate group ∈ ℝ^{m×k}
**Output**: Coherence score ∈ [0, 1]

**Method**: Geometric coherence in utility space. Three complementary measures:

```
coherence_score(U_group) → float:
  m, k = U_group.shape

  # Measure 1: Compactness — how tightly grouped in utility space
  centroid = mean(U_group, axis=0)                    # (k,)
  distances = [cosine_distance(u, centroid) for u in U_group]
  compactness = 1.0 - mean(distances)                 # Higher = tighter

  # Measure 2: Distinctiveness — how different from random background
  # Compare group diameter to expected diameter of random m-subset
  group_diameter = max(pairwise_cosine_distances(U_group))
  # Expected random diameter in k-dimensional space ≈ 1.0 for unit vectors
  distinctiveness = 1.0 - group_diameter              # Higher = more distinct

  # Measure 3: Ontology diversity — bonus for cross-ontology groups
  ontologies = set(node.ontology for node in group_nodes)
  diversity = len(ontologies) / max(len(ontologies), m)  # Fraction of unique
  diversity_bonus = 0.1 * (diversity - 0.5) if diversity > 0.5 else 0

  # Weighted combination
  score = (0.5 * compactness + 0.4 * distinctiveness + 0.1 * diversity_bonus)
  RETURN clip(score, 0.0, 1.0)
```

**Why three measures?**
- Compactness alone would favor trivially similar nodes (same ontology, same type)
- Distinctiveness alone would favor outlier groups
- Ontology diversity provides a controlled bonus for cross-domain coherence —
  which is the specific property we want to discover

**Note on learned scoring**: For production systems, replace the geometric scorer with
a trained permutation-invariant set function (DeepSets or attention-based aggregator).
The geometric approach provides a training-free baseline for cold start.

### 6.7 Result Fusion

**Input**: Exploit results (scored nodes), Explore results (candidate hyperedges)
**Output**: Unified ranked result

```
FUSE(exploit_results, explore_candidates, G) → SearchResult:
  # All nodes from exploit phase
  node_scores = {n.node.id: n for n in exploit_results}

  # Add nodes from minted hyperedges (explore phase)
  FOR candidate in explore_candidates:
    FOR node_id in candidate.members:
      node = G.get_node(node_id)
      explore_score = candidate.coherence
      IF node_id in node_scores:
        # Node found by both phases — boost
        existing = node_scores[node_id]
        node_scores[node_id] = ScoredNode(
          node=existing.node,
          score=max(existing.score, explore_score) * 1.2,  # Boost
          source="both",
          via_edges=existing.via_edges + [candidate.id]
        )
      ELSE:
        node_scores[node_id] = ScoredNode(
          node=node,
          score=explore_score,
          source="explore",
          via_edges=[candidate.id]
        )

  # Sort by score descending
  ranked = sorted(node_scores.values(), key=lambda x: x.score, reverse=True)

  RETURN SearchResult(
    nodes=ranked,
    exploited_edges=[...],
    minted_edges=[validated candidates],
    explanation=SearchExplanation(...)
  )
```

---

## 7. Embedding Pipeline

### 7.1 Encoder Selection

The embedding pipeline maps text descriptions to dense vectors. Requirements:
- Shared embedding space for nodes and predicates (so cosine similarity is meaningful)
- Fast encoding (batch encoding of predicates at query time)
- Good coverage of technical/domain vocabulary

**Default**: `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional embeddings
- 80M parameters, ~14ms per encoding on CPU
- Good general-purpose coverage

**Upgrade path**: `BAAI/bge-large-en-v1.5` or domain-specific fine-tuned models.

### 7.2 Node Encoding

Nodes are encoded from their textual description:

```python
def encode_node(node: Node, encoder: SentenceTransformer) -> np.ndarray:
    """Encode a node's textual description into embedding space."""
    # Build a description string from node metadata
    text = node.metadata.get("description", "")
    if "name" in node.metadata:
        text = f"{node.metadata['name']}: {text}"
    if "ontology" in node.metadata:
        text = f"[{node.ontology}] {text}"

    return encoder.encode(text, normalize_embeddings=True)
```

Encoding includes the ontology tag as a prefix so the encoder can learn
domain-specific nuances, but the embedding space is shared across all ontologies.

### 7.3 Predicate Encoding

Predicates are encoded identically to nodes — same model, same space. This is
what makes cosine similarity between node embeddings and predicate embeddings
meaningful. No additional alignment step is needed.

### 7.4 Batch Encoding

For efficiency, encode all nodes at ingest time and cache. Predicates are encoded
at query time (typically 2-6 predicates, ~100ms total).

```python
def encode_all_nodes(nodes: list[Node], encoder: SentenceTransformer) -> np.ndarray:
    """Batch encode all nodes. Returns (n, d) matrix."""
    texts = [build_node_text(n) for n in nodes]
    return encoder.encode(texts, normalize_embeddings=True, batch_size=256)
```

---

## 8. Two-Phase Search

### 8.1 Phase 1: EXPLOIT

Traverse existing hypergraph structure using intention as a weighting signal.

**Algorithm**: Intention-weighted two-hop expansion (Section 6.3)

**Properties**:
- Uses only existing structure — no new hyperedges generated
- Extremely fast: two sparse matrix-vector multiplications
- Finds nodes that are connected to the intention through known relationships
- Limited by what structure exists: cannot discover truly novel connections

**When it dominates**: Mature hypergraphs with rich structure. Most queries on
well-explored topics will be fully answered by the exploit phase.

### 8.2 Phase 2: EXPLORE

Generate new structure by projecting node features through the intention lens.

**Algorithm**: Project → Cluster → Score → Mint (Section 6.4-6.6)

**Properties**:
- Operates on raw node features, not existing structure
- Can discover connections that no prior query found
- More expensive than exploit (embedding operations + clustering)
- Returns novel hyperedges that enrich the graph for future queries

**When it dominates**: Cold start, novel intentions, cross-ontology queries that
require connections no existing hyperedge represents.

### 8.3 Phase Control

Not every query needs both phases. The engine adaptively controls phase allocation:

```
SEARCH(intention, budget) → SearchResult:
  # Always run exploit first (it's cheap)
  exploit_result = EXPLOIT(intention, G)

  # Decide whether to explore
  IF exploit_result.coverage >= 0.9:
    # Exploit found plenty — skip explore
    RETURN exploit_result

  IF budget.explore_allowed:
    explore_budget = budget.total - exploit_elapsed
    explore_result = EXPLORE(intention, V, E, budget=explore_budget)

    # Mint validated hyperedges
    FOR candidate in explore_result:
      IF candidate.coherence >= threshold:
        G.mint_hyperedge(candidate)

    RETURN FUSE(exploit_result, explore_result)

  RETURN exploit_result
```

**Coverage heuristic**: How well does the exploit result cover the intention
predicates? If all predicates have at least one highly-scored node, coverage is high.
If some predicates have no matches, the exploit phase missed something — trigger
explore.

```
coverage(exploit_result, predicates) → float:
  FOR p in predicates:
    best_match = max(sim(n.embedding, p.embedding) for n in exploit_result.nodes)
    IF best_match < 0.3:
      RETURN partial_coverage  # This predicate wasn't covered
  RETURN full_coverage
```

---

## 9. Hyperedge Minting Protocol

### 9.1 Validation Pipeline

Not every candidate cluster should become a persistent hyperedge. The minting
protocol applies three validation gates:

```
VALIDATE(candidate, G) → bool:
  # Gate 1: Minimum coherence
  IF candidate.coherence < G.config.min_coherence_threshold:
    RETURN False

  # Gate 2: Novelty — is this genuinely new?
  # Check overlap with existing hyperedges
  FOR existing in G.hyperedges:
    jaccard = len(candidate.members & existing.members) / len(candidate.members | existing.members)
    IF jaccard > 0.8:
      # Too similar to existing — update existing edge weight instead
      existing.weight = max(existing.weight, candidate.coherence)
      existing.access_count += 1
      RETURN False

  # Gate 3: Minimum size
  IF len(candidate.members) < 2:
    RETURN False

  RETURN True
```

### 9.2 Labeling

New hyperedges need human-readable labels for interpretability.

```
generate_label(members, intention) → str:
  IF llm_available:
    names = [m.metadata.get("name", m.id) for m in members]
    response = LLM.complete(
      "Given these entities: {names}, found together under the intention "
      "'{intention.raw}', generate a concise label (3-8 words) describing "
      "what they have in common."
    )
    RETURN response
  ELSE:
    # Fallback: use the most specific common predicate
    RETURN f"Group: {intention.predicates[0].text}"
```

### 9.3 Provenance Tracking

Every minted hyperedge records its full provenance:
- Which intention triggered its creation
- Which predicates were active
- What coherence score it received
- When it was created
- How it has been accessed since

This enables:
- Debugging: "Why does this edge exist?"
- Decay: Edges that were minted but never accessed again can be pruned
- Meta-learning: Patterns in successful mintings can inform future exploration

---

## 10. Cold Start Protocol

### 10.1 The Problem

On first use, the hypergraph has nodes but zero hyperedges. The exploit phase has
nothing to traverse. All discovery must come from the explore phase.

### 10.2 The Solution

Cold start is not a special case — it is just the explore phase operating without
exploit results.

```
COLD_START_SEARCH(intention, nodes) → SearchResult:
  # Exploit phase returns empty (no structure to traverse)
  exploit_result = EXPLOIT(intention, empty_graph)  # → empty

  # Explore phase operates on raw node features
  explore_result = EXPLORE(intention, nodes, existing_edges=∅)

  # Mint discovered hyperedges → first structure in the graph
  FOR candidate in explore_result:
    IF VALIDATE(candidate, G):
      G.mint_hyperedge(candidate)

  RETURN SearchResult(
    nodes=explore_result.nodes,
    exploited_edges=[],
    minted_edges=[newly minted],
    explanation=...
  )
```

### 10.3 Bootstrap Strategy

For a genuinely empty graph, the first few queries are critical — they create the
initial structure that all future exploit phases will build on.

**Recommended bootstrap sequence**:
1. Register nodes with rich textual metadata
2. Issue a diverse set of intentions covering different utility dimensions
3. Each query mints hyperedges → graph structure grows
4. After 10-20 queries, the exploit phase begins to fire
5. After 50-100 queries, the graph is rich enough for most queries to be
   exploit-only

**Automatic seeding**: Optionally, at ingest time, run a set of generic
intentions (e.g., "what things serve similar functions?", "what things are used
together?") to pre-populate a basic structure.

---

## 11. Learning Loop

### 11.1 Knowledge Accumulation

The hypergraph is a living structure. Each query potentially enriches it:

```
t=0:  Graph has N nodes, 0 hyperedges
t=1:  Query Q1 → EXPLORE mints 3 hyperedges
t=2:  Query Q2 → EXPLOIT uses 1 of Q1's edges, EXPLORE mints 2 more
t=5:  Query Q5 → EXPLOIT covers 80% of the intention, EXPLORE adds 1 edge
t=50: Query Q50 → EXPLOIT covers 100%, EXPLORE doesn't fire
```

The system exhibits **diminishing exploration** — it learns quickly from early
queries and gradually shifts toward exploitation as the graph matures.

### 11.2 Temporal Decay

Hyperedges have a `weight` that decays over time if not accessed:

```
decay(edge, current_time) → float:
  age = current_time - edge.last_accessed
  half_life = 30 * 24 * 3600  # 30 days in seconds

  # Exponential decay with access-count boost
  base_decay = 0.5 ** (age / half_life)
  access_boost = min(1.0, edge.access_count / 10)  # Caps at 10 accesses

  RETURN edge.coherence_score * (0.5 * base_decay + 0.5 * access_boost)
```

Edges with high coherence AND frequent access persist. Edges that were minted but
never re-accessed fade. Pruning happens at configurable thresholds.

### 11.3 Weight Updates

When an exploit phase traverses a hyperedge, its weight is reinforced:

```
ON_EXPLOIT_ACCESS(edge):
  edge.access_count += 1
  edge.last_accessed = now()
  edge.weight = min(1.0, edge.weight * 1.1)  # Gradual reinforcement
```

### 11.4 Edge Composition

Over time, the system may discover that two hyperedges share members and are often
co-activated. These can be composed into a higher-order hyperedge:

```
COMPOSE(edge_a, edge_b) → Hyperedge | None:
  overlap = edge_a.members & edge_b.members
  IF len(overlap) >= 2:
    combined = edge_a.members | edge_b.members
    # Score the combined group for coherence
    score = coherence_score(combined)
    IF score >= threshold:
      RETURN Hyperedge(
        members=combined,
        provenance=HyperedgeProvenance(
          source="minted",
          parent_edges=[edge_a.id, edge_b.id]
        ),
        coherence_score=score
      )
  RETURN None
```

---

## 12. Computational Complexity

### 12.1 Per-Query Costs

| Operation | Complexity | Typical Time (100K nodes) |
|-----------|-----------|--------------------------|
| Intention decomposition (LLM) | O(1) | ~200ms |
| Intention decomposition (rules) | O(n_words) | <1ms |
| Predicate encoding | O(k · d) | ~70ms (k=5) |
| Exploit: edge scoring | O(|E| · d) | ~5ms |
| Exploit: two-hop SpMV | O(2 · nnz) | ~0.5ms |
| Explore: utility projection | O(n · d · k) | ~1ms |
| Explore: threshold + filter | O(n) | <0.1ms |
| Explore: clustering (HDBSCAN) | O(m · log m) | ~10ms (m=20K) |
| Explore: coherence scoring | O(c · m_avg²) | ~5ms (c=50 clusters) |
| Total (with LLM decomposition) | - | ~290ms |
| Total (rule-based decomposition) | - | ~90ms |

### 12.2 Storage Costs

| Component | Space | Typical (100K nodes, 50K edges) |
|-----------|-------|---------------------------------|
| Node features | O(n · d) | ~150MB (n=100K, d=384, float32) |
| Incidence matrix | O(nnz) | ~2MB (nnz=250K, CSR) |
| Hyperedge metadata | O(|E|) | ~5MB |
| FAISS index | O(n · d) | ~150MB |
| Total | - | ~310MB |

### 12.3 Scaling Characteristics

- **Exploit phase scales with nnz**, not with |V| or |E| separately
- **Explore phase scales with n** (all nodes) but filters aggressively (top 20%)
- **Clustering scales with m·log(m)** where m is the filtered node count
- **Storage scales linearly** with nodes + edges

For 1M nodes: ~3GB storage, ~500ms query time. For 10M nodes: ~30GB, ~2s.

---

## 13. API Surface

### 13.1 Core Engine

```python
class IntentionEngine:
    """Top-level orchestrator for intention-driven hypergraph search."""

    def __init__(
        self,
        encoder_model: str = "all-MiniLM-L6-v2",
        min_coherence: float = 0.3,
        explore_budget: int = 100,
        decay_half_life_days: float = 30.0,
    ): ...

    # === Node Management ===

    def add_node(
        self,
        id: str,
        description: str,
        ontology: str = "default",
        metadata: dict | None = None,
    ) -> Node:
        """Add a node to the hypergraph. Embedding computed lazily."""

    def add_nodes_batch(
        self,
        nodes: list[dict],
    ) -> list[Node]:
        """Batch add nodes with automatic embedding."""

    # === Hyperedge Management ===

    def add_hyperedge(
        self,
        member_ids: set[str],
        label: str,
        metadata: dict | None = None,
    ) -> Hyperedge:
        """Manually add a hyperedge (for known relationships)."""

    # === Search ===

    def search(
        self,
        intention: str,
        max_results: int = 20,
        explore: bool = True,
        scope: SearchScope | None = None,
    ) -> SearchResult:
        """
        Search the hypergraph driven by intention.

        Phase 1 (exploit): Traverse existing structure.
        Phase 2 (explore): Project nodes through intention lens,
                           discover new hyperedges.

        New hyperedges are automatically minted and persisted.
        """

    # === Persistence ===

    def save(self, path: str) -> None:
        """Save hypergraph state to disk."""

    def load(self, path: str) -> None:
        """Load hypergraph state from disk."""

    # === Introspection ===

    def stats(self) -> dict:
        """Return graph statistics: node count, edge count, etc."""

    def explain_edge(self, edge_id: str) -> dict:
        """Explain how a hyperedge was created (provenance)."""

    def decay_edges(self, threshold: float = 0.01) -> int:
        """Prune edges below weight threshold. Returns count pruned."""
```

### 13.2 Configuration

```python
@dataclass
class EngineConfig:
    # Embedding
    encoder_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Search
    max_exploit_depth: int = 2          # Two-hop by default
    exploit_weight: float = 0.7         # Weight exploit vs explore

    # Exploration
    explore_enabled: bool = True
    explore_budget: int = 100           # Max candidates to evaluate
    utility_threshold_percentile: float = 80.0  # Top 20% by utility

    # Coherence
    min_coherence: float = 0.3          # Minimum score to mint
    coherence_weights: tuple = (0.5, 0.4, 0.1)  # compact, distinct, diverse

    # Minting
    novelty_threshold: float = 0.8      # Jaccard threshold for dedup
    min_edge_size: int = 2
    max_edge_size: int = 50

    # Decay
    decay_half_life_days: float = 30.0
    prune_threshold: float = 0.01

    # Decomposition
    use_llm_decomposition: bool = False # False = rule-based (fast)
    llm_model: str | None = None
```

---

## 14. Dependencies & Building Blocks

### 14.1 Required Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.26 | Array operations, utility matrices |
| scipy | ≥1.12 | Sparse matrices (CSR, COO), eigensolvers |
| sentence-transformers | ≥3.0 | Text embedding (nodes + predicates) |
| faiss-cpu | ≥1.8 | Approximate nearest neighbor search |
| hypernetx | ≥2.3 | Hypergraph visualization, validation |

### 14.2 Optional Dependencies

| Library | Purpose |
|---------|---------|
| hdbscan | Density-based clustering (explore phase) |
| torch | GPU-accelerated embeddings and sparse ops |
| faiss-gpu | GPU-accelerated ANN search |
| anthropic / openai | LLM-based intention decomposition |

### 14.3 From Landscape Report

The following existing tools inform the design but are NOT direct dependencies:

| Tool | Influence |
|------|-----------|
| HyperGraphRAG | Validated that LLM→hyperedge extraction works at scale |
| DeepHypergraph (DHG) | Validated vertex↔hyperedge message passing architecture |
| DHGNN | Inspired dynamic per-query hyperedge construction from features |
| HyperSearch | Informed the candidate hyperedge scoring approach |
| CAt-Walk | Informed temporal hyperedge dynamics |

---

## 15. Failure Modes & Mitigations

### 15.1 Degenerate Clustering

**Problem**: Utility projection maps all nodes to similar vectors → one giant cluster.
**Cause**: Predicates are too generic ("things that are useful").
**Mitigation**: If the largest cluster contains >50% of active nodes, re-decompose
the intention with more specific predicates. Or raise the coherence threshold.

### 15.2 Hyperedge Explosion

**Problem**: Too many new hyperedges minted per query → storage bloat.
**Cause**: Low coherence threshold + high explore budget.
**Mitigation**:
- Hard cap on mints per query (default: 10)
- Jaccard-based deduplication against existing edges
- Temporal decay prunes low-value edges automatically

### 15.3 Cold Start Quality

**Problem**: First few queries mint low-quality hyperedges that pollute future exploit.
**Cause**: Insufficient structure to distinguish signal from noise.
**Mitigation**:
- Higher coherence threshold during bootstrap (0.5 vs 0.3 steady-state)
- Mark bootstrap edges as provisional (lower initial weight)
- Re-evaluate provisional edges after 50 queries; prune if never accessed

### 15.4 Embedding Space Limitations

**Problem**: Cosine similarity between node embeddings and predicate embeddings is
not always meaningful (especially for highly technical domain terms).
**Cause**: General-purpose encoders don't understand domain vocabulary.
**Mitigation**:
- Fine-tune the encoder on domain-specific text pairs
- Use domain-specific encoders when available
- Fall back to keyword-based matching for OOV terms

### 15.5 Ontology Collapse

**Problem**: Cross-ontology bonus causes spurious bridges between unrelated domains.
**Cause**: Diversity bonus in coherence scoring is too aggressive.
**Mitigation**: Diversity bonus is capped at 0.1 (10% of score). It cannot override
low compactness or distinctiveness. Adjust weight in config if needed.

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| V | Set of vertices (nodes) |
| E | Set of hyperedges |
| H | Incidence matrix ∈ {0,1}^{\|V\|×\|E\|} |
| D_v | Diagonal vertex degree matrix |
| D_e | Diagonal edge degree matrix |
| W | Diagonal edge weight matrix |
| f_v | Feature vector of node v |
| e_p | Embedding of predicate p |
| u(v\|I) | Utility vector of node v under intention I |
| Δ | Normalized hypergraph Laplacian |

## Appendix B: Relationship to Prior Art

| Concept in This System | Prior Art | Key Difference |
|----------------------|-----------|----------------|
| Utility projection | Multi-view embeddings | Views are dynamic (per-query), not pre-computed |
| Hyperedge minting | HyperGraphRAG extraction | Happens at search-time, not index-time |
| Two-phase search | Explore/exploit in RL | Applied to information retrieval, not action selection |
| Coherence scoring | DHNE tuplewise similarity | Training-free geometric baseline, not learned |
| Temporal decay | PageRank damping | Applied to edge persistence, not random walk |
| Compositional predicates | HyDE (hypothetical documents) | Decomposes into orthogonal utility dimensions |

---

*End of Technical Specification*
