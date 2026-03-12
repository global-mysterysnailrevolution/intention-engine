"""IntentionRAG: Drop-in replacement for traditional RAG systems.

Instead of: embed chunks → vector store → retrieve by similarity
This does:  embed chunks → hypergraph → intention-driven two-phase search
            where each query discovers and persists new structure.

Usage:
    rag = IntentionRAG("myproject")
    rag.ingest("./docs/")
    context = rag.retrieve("how does authentication work?")
    # context is a formatted string ready for LLM consumption
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .models import EngineConfig, SearchScope
from .engine import IntentionEngine
from .encoder import HashEncoder
from .ingestion import IngestionPipeline, IngestConfig, IngestResult
from .context import ContextAssembler, ContextConfig


STORE_ROOT = os.path.join(os.path.expanduser("~"), ".intention-engine")


@dataclass
class RAGConfig:
    """Configuration for the full RAG pipeline."""
    # Graph
    graph_name: str = "default"
    store_path: str | None = None       # Override auto path

    # Ingestion
    chunk_size: int = 512
    chunk_overlap: int = 64
    extract_term_edges: bool = True

    # Search
    max_results: int = 10
    explore: bool = True
    min_coherence: float = 0.2

    # Context
    max_context_chars: int = 16000
    context_format: str = "text"        # "text", "markdown", "xml"
    include_sources: bool = True
    expand_adjacent: bool = True


class IntentionRAG:
    """Drop-in RAG replacement using intention-driven hypergraph search.

    Key differences from traditional RAG:
    - Structure accumulates: each query enriches the graph for future queries
    - Cross-document discovery: hyperedges bridge related chunks across files
    - Multi-level: document, section, and chunk nodes with structural edges
    - Intention decomposition: queries are split into utility predicates
    """

    def __init__(self, config: RAGConfig | None = None, graph_name: str | None = None):
        """Initialize or load a RAG instance.

        Args:
            config: Full RAG configuration
            graph_name: Shorthand — just set the graph name (creates default config)
        """
        if config is None:
            config = RAGConfig()
        if graph_name is not None:
            config.graph_name = graph_name
        self.config = config

        # Store path
        self._store_path = config.store_path or os.path.join(STORE_ROOT, config.graph_name)

        # Engine
        engine_config = EngineConfig(
            min_coherence=config.min_coherence,
            explore_budget=50,
            utility_threshold_percentile=70.0,
        )
        self.engine = IntentionEngine(config=engine_config)
        self.engine.set_encoder(HashEncoder(dim=384))

        # Load existing graph
        if os.path.exists(os.path.join(self._store_path, "nodes.jsonl")):
            self.engine.load(self._store_path)
            self._restore_embeddings()

        # Ingestion pipeline
        ingest_config = IngestConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            extract_term_edges=config.extract_term_edges,
        )
        self.ingestion = IngestionPipeline(self.engine, ingest_config)

        # Context assembler
        context_config = ContextConfig(
            max_chars=config.max_context_chars,
            format=config.context_format,
            include_sources=config.include_sources,
            expand_adjacent=config.expand_adjacent,
        )
        self.context = ContextAssembler(self.engine, context_config)

    def _restore_embeddings(self):
        """Re-encode nodes that lost embeddings during save/load."""
        for node in self.engine.store.nodes.values():
            if node.embedding is None:
                desc = node.metadata.get("description", node.id)
                text = f"[{node.ontology}] {desc}" if node.ontology != "default" else desc
                node.embedding = self.engine._encode(text)

    def _save(self):
        """Persist graph state."""
        os.makedirs(self._store_path, exist_ok=True)
        self.engine.save(self._store_path)

    # === Ingestion ===

    def ingest(self, path: str, recursive: bool = True) -> IngestResult:
        """Ingest a file or directory into the knowledge graph.

        Creates document-level, section-level, and chunk-level nodes.
        Extracts structural relationships as hyperedges.

        Args:
            path: File path or directory to ingest
            recursive: Whether to recurse into subdirectories

        Returns:
            IngestResult with counts of created nodes/edges
        """
        path = os.path.abspath(path)

        if os.path.isfile(path):
            result = self.ingestion.ingest_file(path)
        elif os.path.isdir(path):
            result = self.ingestion.ingest_directory(path, recursive=recursive)
        else:
            result = IngestResult()

        self._save()
        return result

    def ingest_text(self, text: str, name: str = "inline", ontology: str = "text") -> IngestResult:
        """Ingest raw text (not from a file).

        Args:
            text: The text content to ingest
            name: A name/label for the text
            ontology: Category tag
        """
        result = self.ingestion.ingest_text(text, name, ontology)
        self._save()
        return result

    # === Retrieval ===

    def retrieve(
        self,
        query: str,
        max_results: int | None = None,
        explore: bool | None = None,
        format: str | None = None,
    ) -> str:
        """Retrieve relevant context for a query.

        This is the main RAG operation. Returns a formatted string of
        relevant chunks from the knowledge graph, ready for LLM consumption.

        Each call potentially discovers and persists new hyperedges.

        Args:
            query: Natural language query/intention
            max_results: Override default max results
            explore: Override default explore setting
            format: Override context format ("text", "markdown", "xml")

        Returns:
            Formatted context string
        """
        result = self.engine.search(
            intention=query,
            max_results=max_results or self.config.max_results,
            explore=explore if explore is not None else self.config.explore,
        )

        # Override format temporarily if specified
        if format is not None:
            old_fmt = self.context.config.format
            self.context.config.format = format
            ctx = self.context.assemble(result, query)
            self.context.config.format = old_fmt
        else:
            ctx = self.context.assemble(result, query)

        self._save()  # Persist any newly minted edges
        return ctx

    def search(self, query: str, max_results: int | None = None):
        """Raw search — returns SearchResult object instead of formatted context.

        Use this when you need programmatic access to scores, hyperedges, etc.
        """
        return self.engine.search(
            intention=query,
            max_results=max_results or self.config.max_results,
            explore=self.config.explore,
        )

    # === Introspection ===

    def stats(self) -> dict:
        """Return graph statistics."""
        base = self.engine.stats()
        # Count document vs chunk nodes
        docs = sum(1 for n in self.engine.store.nodes.values()
                   if n.metadata.get("type") == "document")
        chunks = sum(1 for n in self.engine.store.nodes.values()
                    if n.metadata.get("type") == "chunk")
        base["documents"] = docs
        base["chunks"] = chunks
        base["store_path"] = self._store_path
        return base

    def list_documents(self) -> list[dict]:
        """List all ingested documents."""
        docs = []
        for n in self.engine.store.nodes.values():
            if n.metadata.get("type") == "document":
                docs.append({
                    "id": n.id,
                    "path": n.metadata.get("path", ""),
                    "filename": n.metadata.get("filename", n.metadata.get("name", "")),
                    "ontology": n.ontology,
                })
        return sorted(docs, key=lambda x: x.get("filename", ""))
