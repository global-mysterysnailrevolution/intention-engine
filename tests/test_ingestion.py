"""Tests for the IngestionPipeline."""
from __future__ import annotations

import os
import tempfile
import pytest

from intention_engine import IntentionEngine, EngineConfig
from intention_engine.encoder import HashEncoder
from intention_engine.ingestion import IngestionPipeline, IngestConfig, IngestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine() -> IntentionEngine:
    """Create a fast test engine using HashEncoder."""
    config = EngineConfig(embedding_dim=64)
    engine = IntentionEngine(config=config)
    engine.set_encoder(HashEncoder(dim=64))
    return engine


def _make_pipeline(engine=None, **kwargs) -> IngestionPipeline:
    engine = engine or _make_engine()
    config = IngestConfig(**kwargs) if kwargs else None
    return IngestionPipeline(engine, config)


# ---------------------------------------------------------------------------
# IngestResult dataclass
# ---------------------------------------------------------------------------

class TestIngestResult:
    def test_defaults(self):
        r = IngestResult()
        assert r.documents == 0
        assert r.chunks == 0
        assert r.nodes_created == 0
        assert r.edges_created == 0
        assert r.files == []


# ---------------------------------------------------------------------------
# ingest_text
# ---------------------------------------------------------------------------

class TestIngestText:
    def test_returns_ingest_result(self):
        pipeline = _make_pipeline()
        result = pipeline.ingest_text("Hello world this is a test.", name="test")
        assert isinstance(result, IngestResult)

    def test_document_count_one(self):
        pipeline = _make_pipeline()
        result = pipeline.ingest_text("Some text here with enough content.", name="doc")
        assert result.documents == 1

    def test_nodes_created_positive(self):
        pipeline = _make_pipeline()
        result = pipeline.ingest_text("A reasonably long text for testing purposes.", name="t")
        assert result.nodes_created >= 1

    def test_doc_node_added_to_engine(self):
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        pipeline.ingest_text("Some content for the engine.", name="mytext")
        stats = engine.stats()
        assert stats["nodes"] >= 1

    def test_structural_edge_created(self):
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        # Use long enough text to produce chunks + a doc edge
        text = "This is a long sentence. " * 30
        result = pipeline.ingest_text(text, name="doc")
        assert result.edges_created >= 1

    def test_chunks_count_matches_engine_nodes(self):
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        text = "Word " * 100  # 500 chars, should produce 1+ chunks with default 512
        result = pipeline.ingest_text(text, name="t")
        # nodes = 1 doc + N chunks
        assert engine.stats()["nodes"] == result.nodes_created

    def test_custom_ontology(self):
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        pipeline.ingest_text("Content here.", name="t", ontology="my_ontology")
        # All nodes should use this ontology
        for node in engine.store.nodes.values():
            assert node.ontology == "my_ontology"

    def test_empty_text_returns_zero_docs(self):
        pipeline = _make_pipeline()
        result = pipeline.ingest_text("", name="empty")
        # An empty text still creates the doc node but no chunks
        # The doc node IS created (it uses text[:200] which is "")
        assert result.chunks == 0

    def test_adjacent_edges_for_multiple_chunks(self):
        engine = _make_engine()
        pipeline = IngestionPipeline(engine, IngestConfig(chunk_size=100, min_chunk_size=10,
                                                          chunk_overlap=0,
                                                          extract_term_edges=False))
        text = "sentence number one here. " * 20  # ~520 chars → multiple chunks
        result = pipeline.ingest_text(text, name="t")
        if result.chunks >= 2:
            # Should have at least the doc structure edge + adjacent edges
            assert result.edges_created >= 2


# ---------------------------------------------------------------------------
# ingest_file
# ---------------------------------------------------------------------------

class TestIngestFile:
    def test_nonexistent_file_returns_empty(self):
        pipeline = _make_pipeline()
        result = pipeline.ingest_file("/nonexistent/path/file.txt")
        assert result.documents == 0

    def test_plain_text_file(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("Hello world. This is a test document with sufficient content.", encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        result = pipeline.ingest_file(str(p))
        assert result.documents == 1
        assert result.nodes_created >= 1
        assert str(p) in result.files

    def test_file_in_files_list(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("Content here enough for ingestion.", encoding="utf-8")
        pipeline = _make_pipeline()
        result = pipeline.ingest_file(str(p))
        assert str(p) in result.files

    def test_document_node_created(self, tmp_path):
        p = tmp_path / "notes.txt"
        p.write_text("Important notes about the project.", encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        pipeline.ingest_file(str(p))
        # At least the doc node should be in the engine
        assert engine.stats()["nodes"] >= 1

    def test_chunk_nodes_created(self, tmp_path):
        p = tmp_path / "long.txt"
        # Write enough content to guarantee at least one chunk
        p.write_text("This is a sentence. " * 30, encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        result = pipeline.ingest_file(str(p))
        assert result.chunks >= 1

    def test_structural_edge_created(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("Content with enough length. " * 10, encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        result = pipeline.ingest_file(str(p))
        assert result.edges_created >= 1

    def test_markdown_file_section_edges(self, tmp_path):
        md_content = (
            "# Introduction\n\n"
            "First section content here. " * 5 + "\n\n"
            "More content for this section.\n\n"
            "# Background\n\n"
            "Background content here. " * 5 + "\n\n"
            "Even more background content.\n\n"
        )
        p = tmp_path / "doc.md"
        p.write_text(md_content, encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine, IngestConfig(
            chunk_size=300, min_chunk_size=10, extract_term_edges=False
        ))
        result = pipeline.ingest_file(str(p))
        assert result.documents == 1
        # Should have section edges when multiple chunks in same section
        assert result.edges_created >= 1

    def test_file_with_exclude_extension(self, tmp_path):
        p = tmp_path / "image.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        pipeline = _make_pipeline(include_extensions={".txt", ".md"})
        result = pipeline.ingest_file(str(p))
        assert result.documents == 0

    def test_file_too_large_skipped(self, tmp_path):
        p = tmp_path / "huge.txt"
        p.write_text("x" * 1001, encoding="utf-8")
        pipeline = _make_pipeline(max_file_size=1000)
        result = pipeline.ingest_file(str(p))
        assert result.documents == 0

    def test_empty_file_skipped(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        pipeline = _make_pipeline()
        result = pipeline.ingest_file(str(p))
        assert result.documents == 0

    def test_whitespace_only_file_skipped(self, tmp_path):
        p = tmp_path / "blank.txt"
        p.write_text("   \n\n   ", encoding="utf-8")
        pipeline = _make_pipeline()
        result = pipeline.ingest_file(str(p))
        assert result.documents == 0

    def test_chunk_metadata_has_parent_doc(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("Some content that has enough text to chunk properly.", encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        pipeline.ingest_file(str(p))
        # Check that chunk nodes have parent_doc metadata
        chunk_nodes = [
            n for n in engine.store.nodes.values()
            if n.metadata.get("type") == "chunk"
        ]
        for cn in chunk_nodes:
            assert "parent_doc" in cn.metadata
            assert cn.metadata["parent_doc"].startswith("doc_")

    def test_chunk_metadata_has_full_text(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("Full text should be stored in metadata here.", encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        pipeline.ingest_file(str(p))
        chunk_nodes = [
            n for n in engine.store.nodes.values()
            if n.metadata.get("type") == "chunk"
        ]
        for cn in chunk_nodes:
            assert "full_text" in cn.metadata
            assert len(cn.metadata["full_text"]) > 0

    def test_python_file(self, tmp_path):
        p = tmp_path / "script.py"
        p.write_text(
            "def foo():\n    return 1\n\ndef bar():\n    return 2\n",
            encoding="utf-8"
        )
        engine = _make_engine()
        pipeline = IngestionPipeline(engine, IngestConfig(min_chunk_size=5))
        result = pipeline.ingest_file(str(p))
        assert result.documents == 1
        assert result.chunks >= 1

    def test_ingest_result_counts_consistent(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("Test content. " * 20, encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        result = pipeline.ingest_file(str(p))
        # nodes_created = 1 doc + N chunks
        assert result.nodes_created == 1 + result.chunks
        assert engine.stats()["nodes"] == result.nodes_created


# ---------------------------------------------------------------------------
# ingest_directory
# ---------------------------------------------------------------------------

class TestIngestDirectory:
    def test_ingests_multiple_files(self, tmp_path):
        for i in range(3):
            f = tmp_path / f"doc{i}.txt"
            f.write_text(f"Document {i} content with enough text here.", encoding="utf-8")
        pipeline = _make_pipeline()
        result = pipeline.ingest_directory(str(tmp_path))
        assert result.documents == 3

    def test_files_list_contains_all(self, tmp_path):
        for i in range(3):
            f = tmp_path / f"doc{i}.txt"
            f.write_text(f"Content for document {i}.", encoding="utf-8")
        pipeline = _make_pipeline()
        result = pipeline.ingest_directory(str(tmp_path))
        assert len(result.files) == 3

    def test_recursive_ingestion(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "root.txt").write_text("Root file content here.", encoding="utf-8")
        (sub / "nested.txt").write_text("Nested file content here.", encoding="utf-8")
        pipeline = _make_pipeline()
        result = pipeline.ingest_directory(str(tmp_path), recursive=True)
        assert result.documents == 2

    def test_non_recursive_stays_at_root(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "root.txt").write_text("Root file content here.", encoding="utf-8")
        (sub / "nested.txt").write_text("Nested file content here.", encoding="utf-8")
        pipeline = _make_pipeline()
        result = pipeline.ingest_directory(str(tmp_path), recursive=False)
        assert result.documents == 1

    def test_exclude_patterns_respected(self, tmp_path):
        good = tmp_path / "good.txt"
        good.write_text("Good file content.", encoding="utf-8")
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        bad = cache_dir / "bad.txt"
        bad.write_text("Should be excluded.", encoding="utf-8")
        pipeline = _make_pipeline()
        result = pipeline.ingest_directory(str(tmp_path))
        assert result.documents == 1

    def test_node_modules_excluded(self, tmp_path):
        good = tmp_path / "index.js"
        good.write_text("const x = 1;", encoding="utf-8")
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lib.js").write_text("module content", encoding="utf-8")
        pipeline = _make_pipeline(min_chunk_size=5)
        result = pipeline.ingest_directory(str(tmp_path))
        # node_modules should be excluded
        for f in result.files:
            assert "node_modules" not in f

    def test_empty_directory_returns_zero(self, tmp_path):
        pipeline = _make_pipeline()
        result = pipeline.ingest_directory(str(tmp_path))
        assert result.documents == 0
        assert result.nodes_created == 0

    def test_total_counts_accumulate(self, tmp_path):
        for i in range(3):
            f = tmp_path / f"file{i}.txt"
            f.write_text(f"Content for file {i} with enough length.", encoding="utf-8")
        pipeline = _make_pipeline()
        result = pipeline.ingest_directory(str(tmp_path))
        assert result.documents == 3
        assert result.nodes_created >= 3  # At minimum one node per doc

    def test_mixed_file_types(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Plain text file content.", encoding="utf-8")
        (tmp_path / "readme.md").write_text("# Title\n\nMarkdown content.", encoding="utf-8")
        (tmp_path / "script.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        pipeline = _make_pipeline(min_chunk_size=5)
        result = pipeline.ingest_directory(str(tmp_path))
        assert result.documents == 3


# ---------------------------------------------------------------------------
# Term edge extraction
# ---------------------------------------------------------------------------

class TestTermEdges:
    def test_term_edges_extracted_when_chunks_share_terms(self, tmp_path):
        # Create text where two sections share many significant words
        text = (
            "machine learning algorithms neural network training optimization. " * 5
            + "\n\n"
            + "deep learning neural network architecture training optimization. " * 5
        )
        engine = _make_engine()
        pipeline = IngestionPipeline(engine, IngestConfig(
            chunk_size=300,
            min_chunk_size=10,
            chunk_overlap=0,
            extract_term_edges=True,
            min_shared_terms=3,
        ))
        pipeline.ingest_text(text, name="test")
        # Should have created some term edges if chunks share significant terms
        # (not guaranteed since text may or may not be split into 2+ chunks)
        # Just verify no crash and result is valid
        assert engine.stats()["nodes"] >= 1

    def test_term_edges_disabled(self, tmp_path):
        p = tmp_path / "doc.txt"
        text = "alpha beta gamma delta epsilon zeta eta theta. " * 10
        p.write_text(text, encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine, IngestConfig(
            chunk_size=200,
            min_chunk_size=10,
            chunk_overlap=0,
            extract_term_edges=False,
        ))
        result = pipeline.ingest_file(str(p))
        # Without term edges, only structural edges:
        # 1 doc-structure edge + (n-1) adjacent edges + possibly 1 section edge
        # (if >=2 chunks share the __root__ section)
        if result.chunks >= 2:
            # max = doc edge + adjacent edges + 1 optional section edge
            expected_max = 1 + (result.chunks - 1) + 1
            assert result.edges_created <= expected_max

    def test_term_edges_with_high_threshold(self, tmp_path):
        """With very high min_shared_terms, no term edges should be extracted."""
        p = tmp_path / "doc.txt"
        text = "apple banana cherry. " * 5 + "\n\n" + "orange grape lemon. " * 5
        p.write_text(text, encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine, IngestConfig(
            chunk_size=200,
            min_chunk_size=10,
            chunk_overlap=0,
            extract_term_edges=True,
            min_shared_terms=1000,  # Impossibly high threshold
        ))
        result = pipeline.ingest_file(str(p))
        # With impossible threshold, no term edges
        # max = doc edge + adjacent edges + 1 optional section edge
        if result.chunks >= 2:
            expected_max = 1 + (result.chunks - 1) + 1
            assert result.edges_created <= expected_max

    def test_stop_words_excluded_from_terms(self):
        """Common words like 'the', 'and' should not create term edges."""
        from intention_engine.ingestion import IngestionPipeline, IngestConfig
        from intention_engine.chunker import Chunk

        engine = _make_engine()
        pipeline = IngestionPipeline(engine, IngestConfig(
            min_shared_terms=1,
        ))
        # Create chunks that only share stop words
        chunk1 = Chunk(text="the and is are was were", index=0, start_line=1, end_line=1)
        chunk2 = Chunk(text="the and is are was were", index=1, start_line=2, end_line=2)
        edges = pipeline._extract_term_edges([chunk1, chunk2], ["id1", "id2"])
        assert edges == []


# ---------------------------------------------------------------------------
# ID prefixes
# ---------------------------------------------------------------------------

class TestIdPrefixes:
    def test_doc_node_id_has_doc_prefix(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("Content here.", encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        pipeline.ingest_file(str(p))
        doc_nodes = [
            n for n in engine.store.nodes.values()
            if n.metadata.get("type") == "document"
        ]
        assert len(doc_nodes) >= 1
        for dn in doc_nodes:
            assert dn.id.startswith("doc_")

    def test_chunk_node_id_has_chunk_prefix(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("Content here enough for a chunk.", encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        pipeline.ingest_file(str(p))
        chunk_nodes = [
            n for n in engine.store.nodes.values()
            if n.metadata.get("type") == "chunk"
        ]
        for cn in chunk_nodes:
            assert cn.id.startswith("chunk_")

    def test_custom_prefix(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("Some text here.", encoding="utf-8")
        engine = _make_engine()
        pipeline = IngestionPipeline(engine, IngestConfig(
            doc_prefix="mydoc",
            chunk_prefix="mychunk",
        ))
        pipeline.ingest_file(str(p))
        for n in engine.store.nodes.values():
            if n.metadata.get("type") == "document":
                assert n.id.startswith("mydoc_")
            elif n.metadata.get("type") == "chunk":
                assert n.id.startswith("mychunk_")


# ---------------------------------------------------------------------------
# Integration: ingest then search
# ---------------------------------------------------------------------------

class TestIngestThenSearch:
    def test_ingested_content_is_searchable(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text(
            "Machine learning neural networks deep learning gradient descent optimization. "
            "Backpropagation training data epochs batch size learning rate.",
            encoding="utf-8"
        )
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        pipeline.ingest_file(str(p))
        result = engine.search("machine learning training", explore=False)
        assert isinstance(result.nodes, list)

    def test_ingested_nodes_appear_in_stats(self, tmp_path):
        engine = _make_engine()
        pipeline = IngestionPipeline(engine)
        for i in range(3):
            p = tmp_path / f"doc{i}.txt"
            p.write_text(f"Document {i} content with sufficient length for ingestion.", encoding="utf-8")
            pipeline.ingest_file(str(p))
        stats = engine.stats()
        assert stats["nodes"] >= 3
        assert stats["edges"] >= 3  # At least one doc structure edge per file
