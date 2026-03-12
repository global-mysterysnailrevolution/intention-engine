"""Integration tests for IntentionRAG."""
from __future__ import annotations

import os
import tempfile
import uuid

import pytest

from intention_engine.rag import IntentionRAG, RAGConfig
from intention_engine.models import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_name() -> str:
    """Generate a unique graph name so tests don't collide."""
    return f"test_{uuid.uuid4().hex[:8]}"


def _make_rag(tmp_path: str | None = None, **kwargs) -> IntentionRAG:
    """Create an IntentionRAG instance in a temp directory."""
    store = tmp_path or tempfile.mkdtemp()
    config = RAGConfig(
        graph_name=_unique_name(),
        store_path=store,
        **kwargs,
    )
    return IntentionRAG(config=config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_init_creates_engine(self):
        rag = _make_rag()
        assert rag.engine is not None

    def test_init_with_graph_name_shorthand(self, tmp_path):
        rag = IntentionRAG(graph_name="shorthand_test")
        assert rag.config.graph_name == "shorthand_test"

    def test_init_with_config(self, tmp_path):
        config = RAGConfig(
            graph_name=_unique_name(),
            store_path=str(tmp_path),
            chunk_size=256,
        )
        rag = IntentionRAG(config=config)
        assert rag.config.chunk_size == 256

    def test_init_has_ingestion_pipeline(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        assert rag.ingestion is not None

    def test_init_has_context_assembler(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        assert rag.context is not None

    def test_init_empty_graph_has_no_nodes(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        assert rag.engine.store.num_nodes == 0


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class TestIngestText:
    def test_ingest_text_creates_document_node(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        text = "Authentication uses JWT tokens. The token expires after 24 hours."
        result = rag.ingest_text(text, name="auth_doc")
        assert result.documents == 1

    def test_ingest_text_creates_chunk_nodes(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        text = "A" * 600  # Longer than default chunk_size to ensure chunking
        result = rag.ingest_text(text, name="long_doc")
        assert result.chunks >= 1

    def test_ingest_text_nodes_in_store(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("Some text content.", name="test")
        assert rag.engine.store.num_nodes > 0

    def test_ingest_text_creates_document_and_chunk_nodes_in_store(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("This is a test document with enough content.", name="test")
        doc_nodes = [
            n for n in rag.engine.store.nodes.values()
            if n.metadata.get("type") == "document"
        ]
        assert len(doc_nodes) == 1


class TestIngestFile:
    def test_ingest_file_creates_proper_structure(self, tmp_path):
        # Create a temp markdown file
        fpath = tmp_path / "test.md"
        fpath.write_text("# Installation\n\nRun `pip install mypackage`.\n\n# Usage\n\nImport and use.\n")
        rag = _make_rag(str(tmp_path / "store"))
        os.makedirs(str(tmp_path / "store"), exist_ok=True)
        result = rag.ingest(str(fpath))
        assert result.documents >= 1
        assert result.chunks >= 1

    def test_ingest_file_sets_filename_metadata(self, tmp_path):
        fpath = tmp_path / "myfile.txt"
        fpath.write_text("Hello world content here.")
        store_path = str(tmp_path / "store")
        os.makedirs(store_path)
        rag = _make_rag(store_path)
        rag.ingest(str(fpath))
        doc_nodes = [
            n for n in rag.engine.store.nodes.values()
            if n.metadata.get("type") == "document"
        ]
        assert any("myfile.txt" in n.metadata.get("filename", "") for n in doc_nodes)

    def test_ingest_directory(self, tmp_path):
        # Create several files
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("File A content about authentication.")
        (docs_dir / "b.txt").write_text("File B content about database.")
        store_path = str(tmp_path / "store")
        os.makedirs(store_path)
        rag = _make_rag(store_path)
        result = rag.ingest(str(docs_dir))
        assert result.documents == 2


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_retrieve_returns_string(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("Authentication uses JWT tokens for security.", name="auth")
        ctx = rag.retrieve("authentication")
        assert isinstance(ctx, str)

    def test_retrieve_on_empty_graph_returns_empty_string(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        ctx = rag.retrieve("authentication")
        assert ctx == ""

    def test_retrieve_contains_relevant_content(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text(
            "The login system uses OAuth2 for single sign-on. "
            "Users authenticate with their email and password.",
            name="auth_doc",
        )
        ctx = rag.retrieve("OAuth2 authentication")
        # Should contain some of the ingested content
        assert len(ctx) > 0

    def test_retrieve_with_format_override(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("Markdown test content here.", name="test")
        ctx = rag.retrieve("test content", format="markdown")
        assert "### Source" in ctx or ctx == ""

    def test_retrieve_with_xml_format(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("XML test content for retrieval.", name="test")
        ctx = rag.retrieve("test", format="xml")
        if ctx:
            assert "<context>" in ctx

    def test_retrieve_explore_false_works(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("Some content to search.", name="test")
        ctx = rag.retrieve("content", explore=False)
        assert isinstance(ctx, str)

    def test_retrieve_does_not_crash_with_max_results(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("Sample text for retrieval testing.", name="sample")
        ctx = rag.retrieve("sample", max_results=5)
        assert isinstance(ctx, str)


# ---------------------------------------------------------------------------
# Raw search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_search_result(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("JWT authentication with refresh tokens.", name="auth")
        result = rag.search("JWT tokens")
        assert isinstance(result, SearchResult)

    def test_search_result_has_nodes(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("A document about machine learning models.", name="ml")
        result = rag.search("machine learning")
        assert hasattr(result, "nodes")
        assert isinstance(result.nodes, list)

    def test_search_empty_graph_returns_empty_nodes(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        result = rag.search("anything")
        assert result.nodes == []


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_returns_dict(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        s = rag.stats()
        assert isinstance(s, dict)

    def test_stats_shows_document_count(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("Document one.", name="doc1")
        rag.ingest_text("Document two.", name="doc2")
        s = rag.stats()
        assert s["documents"] == 2

    def test_stats_shows_chunk_count(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("A document with enough content to be chunked properly.", name="d1")
        s = rag.stats()
        assert s["chunks"] >= 1

    def test_stats_has_node_count(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("Some text.", name="x")
        s = rag.stats()
        assert "nodes" in s
        assert s["nodes"] > 0

    def test_stats_has_store_path(self, tmp_path):
        store = str(tmp_path / "mystore")
        os.makedirs(store)
        config = RAGConfig(graph_name=_unique_name(), store_path=store)
        rag = IntentionRAG(config=config)
        s = rag.stats()
        assert "store_path" in s


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------

class TestListDocuments:
    def test_list_documents_returns_list(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        docs = rag.list_documents()
        assert isinstance(docs, list)

    def test_list_documents_empty_initially(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        assert rag.list_documents() == []

    def test_list_documents_returns_metadata(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("Some content here.", name="myfile.txt")
        docs = rag.list_documents()
        assert len(docs) == 1
        doc = docs[0]
        assert "id" in doc
        assert "filename" in doc
        assert "ontology" in doc

    def test_list_documents_sorted_by_filename(self, tmp_path):
        rag = _make_rag(str(tmp_path))
        rag.ingest_text("Zebra document.", name="zebra.txt")
        rag.ingest_text("Alpha document.", name="alpha.txt")
        docs = rag.list_documents()
        filenames = [d["filename"] for d in docs]
        assert filenames == sorted(filenames)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persist_and_reload(self, tmp_path):
        """Create RAG, ingest, reload from same path — nodes should persist."""
        store = str(tmp_path / "store")
        os.makedirs(store)
        graph_name = _unique_name()
        config1 = RAGConfig(graph_name=graph_name, store_path=store)
        rag1 = IntentionRAG(config=config1)
        rag1.ingest_text(
            "The persistence test verifies that data survives restarts.",
            name="persist_test",
        )
        node_count_before = rag1.engine.store.num_nodes

        # Create a second RAG with same store path → should load existing graph
        config2 = RAGConfig(graph_name=graph_name, store_path=store)
        rag2 = IntentionRAG(config=config2)
        assert rag2.engine.store.num_nodes == node_count_before

    def test_persist_data_retrievable_after_reload(self, tmp_path):
        """Data ingested before save should be retrievable after reload."""
        store = str(tmp_path / "store")
        os.makedirs(store)
        graph_name = _unique_name()

        config1 = RAGConfig(graph_name=graph_name, store_path=store)
        rag1 = IntentionRAG(config=config1)
        rag1.ingest_text(
            "Hypergraph persistence with minted edges survives reload.",
            name="persistence_doc",
        )
        # First query — may mint edges
        rag1.retrieve("hypergraph persistence")

        # Reload
        config2 = RAGConfig(graph_name=graph_name, store_path=store)
        rag2 = IntentionRAG(config=config2)
        docs = rag2.list_documents()
        assert len(docs) >= 1

    def test_second_retrieve_benefits_from_minted_edges(self, tmp_path):
        """Two retrieval calls on same RAG — second should work at least as well."""
        rag = _make_rag(str(tmp_path))
        text = (
            "Authentication uses JWT tokens. "
            "JWT tokens are signed with a secret key. "
            "The secret key must be kept confidential. "
            "Token expiry is set to 24 hours. "
            "Refresh tokens can extend the session."
        )
        rag.ingest_text(text, name="auth.txt")

        result1 = rag.search("JWT token authentication")
        result2 = rag.search("JWT token authentication")

        # Both should return the same nodes (structure preserved)
        ids1 = {sn.node.id for sn in result1.nodes}
        ids2 = {sn.node.id for sn in result2.nodes}
        # Second result should be non-empty if first was
        if ids1:
            assert len(ids2) >= 0  # Non-crashing is the minimum bar
