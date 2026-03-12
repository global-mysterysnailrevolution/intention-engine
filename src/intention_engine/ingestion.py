"""Document ingestion pipeline.

Takes files or directories, chunks them, creates nodes (document + section + chunk levels),
and extracts structural relationships as hyperedges.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .models import _make_id
from .chunker import DocumentChunker, ChunkerConfig, Chunk, detect_file_type


@dataclass
class IngestResult:
    """Summary of an ingestion operation."""
    documents: int = 0
    chunks: int = 0
    nodes_created: int = 0
    edges_created: int = 0
    files: list[str] = field(default_factory=list)


@dataclass
class IngestConfig:
    """Configuration for the ingestion pipeline."""
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    # File filtering
    include_extensions: set[str] | None = None  # None = all supported
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "__pycache__", "node_modules", ".git", ".venv", "venv",
        "dist", "build", ".egg-info", ".pytest_cache",
    ])
    max_file_size: int = 1_000_000  # 1MB max per file
    # Relationship extraction
    extract_term_edges: bool = True
    min_shared_terms: int = 3  # Minimum shared significant terms for a term edge
    # Node ID prefix
    doc_prefix: str = "doc"
    chunk_prefix: str = "chunk"


class IngestionPipeline:
    """Ingests documents into an IntentionEngine's hypergraph."""

    def __init__(self, engine, config: IngestConfig | None = None):
        """
        Args:
            engine: An IntentionEngine instance (with add_node, add_hyperedge, store)
            config: Ingestion configuration
        """
        self.engine = engine
        self.config = config or IngestConfig()
        self.chunker = DocumentChunker(ChunkerConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
        ))

    def ingest_file(self, file_path: str) -> IngestResult:
        """Ingest a single file into the hypergraph."""
        result = IngestResult()
        file_path = os.path.abspath(file_path)

        if not os.path.isfile(file_path):
            return result

        # Check file size
        if os.path.getsize(file_path) > self.config.max_file_size:
            return result

        # Check extension
        ext = os.path.splitext(file_path)[1].lower()
        if self.config.include_extensions and ext not in self.config.include_extensions:
            return result

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            return result

        if not text.strip():
            return result

        file_type = detect_file_type(file_path)
        basename = os.path.basename(file_path)

        # Create document-level node
        doc_id = f"{self.config.doc_prefix}_{_make_id('f')}"
        doc_description = f"Document: {basename}"
        # Add first few lines as context
        first_lines = text[:200].replace('\n', ' ').strip()
        if first_lines:
            doc_description += f" — {first_lines}"

        self.engine.add_node(
            id=doc_id,
            description=doc_description,
            ontology=file_type,
            metadata={"path": file_path, "filename": basename, "type": "document"},
        )
        result.documents += 1
        result.nodes_created += 1
        result.files.append(file_path)

        # Chunk the document
        chunks = self.chunker.chunk_text(text, file_path)
        if not chunks:
            return result

        chunk_ids = []
        section_groups: dict[str, list[str]] = {}  # section_name -> chunk_ids

        for chunk in chunks:
            chunk_id = f"{self.config.chunk_prefix}_{_make_id('c')}"

            # Build a rich description for the chunk
            desc_parts = [chunk.text[:300]]  # First 300 chars of chunk content
            if chunk.section:
                desc_parts.insert(0, f"[Section: {chunk.section}]")

            self.engine.add_node(
                id=chunk_id,
                description=" ".join(desc_parts),
                ontology=file_type,
                metadata={
                    "path": file_path,
                    "filename": basename,
                    "type": "chunk",
                    "chunk_index": chunk.index,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "section": chunk.section,
                    "full_text": chunk.text,  # Store full text for context assembly
                    "parent_doc": doc_id,
                },
            )
            chunk_ids.append(chunk_id)
            result.chunks += 1
            result.nodes_created += 1

            # Track sections
            section = chunk.section or "__root__"
            if section not in section_groups:
                section_groups[section] = []
            section_groups[section].append(chunk_id)

        # Create structural hyperedges

        # 1. Document-to-chunks edge
        if chunk_ids:
            self.engine.add_hyperedge(
                member_ids={doc_id} | set(chunk_ids),
                label=f"Document structure: {basename}",
            )
            result.edges_created += 1

        # 2. Section edges (chunks in same section)
        for section_name, section_chunk_ids in section_groups.items():
            if len(section_chunk_ids) >= 2:
                label = (
                    f"Section: {section_name}"
                    if section_name != "__root__"
                    else f"Root section: {basename}"
                )
                self.engine.add_hyperedge(
                    member_ids=set(section_chunk_ids),
                    label=label,
                )
                result.edges_created += 1

        # 3. Adjacent chunk edges (sequential pairs for continuity)
        for i in range(len(chunk_ids) - 1):
            self.engine.add_hyperedge(
                member_ids={chunk_ids[i], chunk_ids[i + 1]},
                label=f"Adjacent chunks {i}-{i+1} in {basename}",
            )
            result.edges_created += 1

        # 4. Term-based cross-chunk edges (chunks sharing significant terms)
        if self.config.extract_term_edges and len(chunks) >= 2:
            term_edges = self._extract_term_edges(chunks, chunk_ids)
            for member_ids, label in term_edges:
                self.engine.add_hyperedge(member_ids=member_ids, label=label)
                result.edges_created += 1

        return result

    def ingest_directory(self, dir_path: str, recursive: bool = True) -> IngestResult:
        """Ingest all supported files in a directory."""
        total = IngestResult()
        dir_path = os.path.abspath(dir_path)

        for root, dirs, files in os.walk(dir_path):
            # Filter excluded directories
            dirs[:] = [d for d in dirs if not any(
                p in os.path.join(root, d) for p in self.config.exclude_patterns
            )]

            if not recursive and root != dir_path:
                break

            for fname in sorted(files):
                fpath = os.path.join(root, fname)

                # Check exclusion patterns
                if any(p in fpath for p in self.config.exclude_patterns):
                    continue

                r = self.ingest_file(fpath)
                total.documents += r.documents
                total.chunks += r.chunks
                total.nodes_created += r.nodes_created
                total.edges_created += r.edges_created
                total.files.extend(r.files)

        return total

    def ingest_text(self, text: str, name: str = "inline", ontology: str = "text") -> IngestResult:
        """Ingest raw text (not from a file) into the hypergraph."""
        result = IngestResult()

        doc_id = f"{self.config.doc_prefix}_{_make_id('t')}"
        first_lines = text[:200].replace('\n', ' ').strip()
        self.engine.add_node(
            id=doc_id,
            description=f"Text: {name} — {first_lines}",
            ontology=ontology,
            metadata={"name": name, "type": "document"},
        )
        result.documents += 1
        result.nodes_created += 1

        chunks = self.chunker.chunk_text(text, "")
        chunk_ids = []

        for chunk in chunks:
            chunk_id = f"{self.config.chunk_prefix}_{_make_id('c')}"
            self.engine.add_node(
                id=chunk_id,
                description=chunk.text[:300],
                ontology=ontology,
                metadata={
                    "type": "chunk",
                    "chunk_index": chunk.index,
                    "full_text": chunk.text,
                    "parent_doc": doc_id,
                    "name": name,
                },
            )
            chunk_ids.append(chunk_id)
            result.chunks += 1
            result.nodes_created += 1

        if chunk_ids:
            self.engine.add_hyperedge(
                member_ids={doc_id} | set(chunk_ids),
                label=f"Document structure: {name}",
            )
            result.edges_created += 1

        for i in range(len(chunk_ids) - 1):
            self.engine.add_hyperedge(
                member_ids={chunk_ids[i], chunk_ids[i + 1]},
                label=f"Adjacent chunks {i}-{i+1} in {name}",
            )
            result.edges_created += 1

        return result

    def _extract_term_edges(
        self,
        chunks: list[Chunk],
        chunk_ids: list[str],
    ) -> list[tuple[set[str], str]]:
        """Extract term-based relationships between chunks.

        Chunks sharing significant terms (beyond stop words) get a hyperedge.
        """
        # Simple stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "must", "need",
            "not", "no", "nor", "and", "or", "but", "if", "then", "else",
            "when", "where", "how", "what", "which", "who", "whom", "why",
            "this", "that", "these", "those", "it", "its", "their", "them",
            "we", "you", "he", "she", "they", "i", "me", "my", "your",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "all", "each",
            "every", "both", "few", "more", "most", "other", "some", "such",
            "only", "own", "same", "so", "than", "too", "very",
            "just", "also", "about", "up", "down",
            # Code-common
            "def", "class", "function", "return", "import", "from", "self",
            "true", "false", "none", "null", "var", "let", "const",
        }

        # Extract significant terms per chunk
        chunk_terms: list[set[str]] = []
        for chunk in chunks:
            words = set(re.findall(r'[a-zA-Z_]\w{2,}', chunk.text.lower()))
            significant = words - stop_words
            chunk_terms.append(significant)

        # Find pairs/groups with significant overlap
        edges = []
        seen = set()

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                shared = chunk_terms[i] & chunk_terms[j]
                if len(shared) >= self.config.min_shared_terms:
                    key = frozenset([chunk_ids[i], chunk_ids[j]])
                    if key not in seen:
                        seen.add(key)
                        top_terms = sorted(shared)[:5]
                        edges.append((
                            {chunk_ids[i], chunk_ids[j]},
                            f"Shared terms: {', '.join(top_terms)}",
                        ))

        return edges
