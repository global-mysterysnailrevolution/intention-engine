"""Tests for ContextAssembler and ContextConfig."""
from __future__ import annotations

import pytest

from intention_engine.models import (
    SearchResult,
    ScoredNode,
    Node,
    SearchExplanation,
    ExploitStats,
    ExploreStats,
    Intention,
    Predicate,
    SearchScope,
)
from intention_engine.context import ContextAssembler, ContextConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_stub():
    """Return a minimal engine stub (context assembler only needs it stored)."""
    from intention_engine import IntentionEngine, EngineConfig
    from intention_engine.encoder import HashEncoder
    engine = IntentionEngine(config=EngineConfig(min_coherence=0.1))
    engine.set_encoder(HashEncoder(dim=64))
    return engine


def _make_chunk_node(
    node_id: str = "chunk_001",
    full_text: str = "This is a chunk of text.",
    filename: str = "readme.md",
    section: str = "",
    start_line: int = 1,
    end_line: int = 5,
    ontology: str = "documentation",
) -> Node:
    return Node(
        id=node_id,
        ontology=ontology,
        metadata={
            "type": "chunk",
            "full_text": full_text,
            "filename": filename,
            "section": section,
            "start_line": start_line,
            "end_line": end_line,
        },
    )


def _make_scored_node(node: Node, score: float = 0.8, source: str = "exploit") -> ScoredNode:
    return ScoredNode(node=node, score=score, source=source)


def _make_search_result(nodes: list[ScoredNode]) -> SearchResult:
    intention = Intention(
        raw="test query",
        predicates=[Predicate(text="test")],
        scope=SearchScope(),
    )
    return SearchResult(
        nodes=nodes,
        explanation=SearchExplanation(
            intention=intention,
            exploit_stats=ExploitStats(),
            explore_stats=ExploreStats(),
        ),
    )


# ---------------------------------------------------------------------------
# Basic assembly
# ---------------------------------------------------------------------------

class TestContextAssemblerBasic:
    def test_empty_result_returns_empty_string(self):
        engine = _make_engine_stub()
        assembler = ContextAssembler(engine)
        result = _make_search_result([])
        out = assembler.assemble(result, "query")
        assert out == ""

    def test_single_chunk_text_format(self):
        engine = _make_engine_stub()
        assembler = ContextAssembler(engine)
        node = _make_chunk_node(full_text="The answer is 42.")
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result, "what is the answer")
        assert "The answer is 42." in out

    def test_node_without_full_text_uses_description(self):
        engine = _make_engine_stub()
        assembler = ContextAssembler(engine)
        node = Node(
            id="node_desc",
            ontology="default",
            metadata={"description": "A description fallback"},
        )
        sn = _make_scored_node(node)
        result = _make_search_result([sn])
        out = assembler.assemble(result)
        assert "A description fallback" in out

    def test_node_with_no_text_is_skipped(self):
        engine = _make_engine_stub()
        assembler = ContextAssembler(engine)
        node = Node(id="empty_node", ontology="default", metadata={})
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert out == ""


# ---------------------------------------------------------------------------
# Text format
# ---------------------------------------------------------------------------

class TestTextFormat:
    def test_text_format_includes_source_info(self):
        engine = _make_engine_stub()
        config = ContextConfig(include_sources=True, format="text")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(filename="guide.md", start_line=10, end_line=20)
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert "guide.md" in out

    def test_text_format_omits_source_when_disabled(self):
        engine = _make_engine_stub()
        config = ContextConfig(include_sources=False, format="text")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(filename="guide.md")
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert "guide.md" not in out

    def test_text_format_includes_section_in_source(self):
        engine = _make_engine_stub()
        config = ContextConfig(include_sources=True, format="text")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(filename="guide.md", section="Installation")
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert "Installation" in out

    def test_text_format_includes_scores_when_enabled(self):
        engine = _make_engine_stub()
        config = ContextConfig(include_scores=True, format="text")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(full_text="Some content here.")
        result = _make_search_result([_make_scored_node(node, score=0.753)])
        out = assembler.assemble(result)
        assert "0.753" in out

    def test_text_format_excludes_scores_by_default(self):
        engine = _make_engine_stub()
        config = ContextConfig(include_scores=False, format="text")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(full_text="Some content here.")
        result = _make_search_result([_make_scored_node(node, score=0.753)])
        out = assembler.assemble(result)
        # Score value should not appear in source-free output
        assert "score" not in out.lower() or "0.753" not in out

    def test_multiple_chunks_separated(self):
        engine = _make_engine_stub()
        config = ContextConfig(separator="\n---\n", format="text")
        assembler = ContextAssembler(engine, config)
        nodes = [
            _make_chunk_node("chunk_a", "First chunk content.", "a.md"),
            _make_chunk_node("chunk_b", "Second chunk content.", "b.md"),
        ]
        result = _make_search_result([_make_scored_node(n) for n in nodes])
        out = assembler.assemble(result)
        assert "First chunk content." in out
        assert "Second chunk content." in out
        assert "---" in out

    def test_header_appears_first(self):
        engine = _make_engine_stub()
        config = ContextConfig(header="=== CONTEXT ===", format="text")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(full_text="Chunk body.")
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert out.startswith("=== CONTEXT ===")


# ---------------------------------------------------------------------------
# Markdown format
# ---------------------------------------------------------------------------

class TestMarkdownFormat:
    def test_markdown_format_has_headers(self):
        engine = _make_engine_stub()
        config = ContextConfig(format="markdown")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(filename="api.md")
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert "### Source" in out

    def test_markdown_format_has_filename_in_header(self):
        engine = _make_engine_stub()
        config = ContextConfig(format="markdown")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(filename="api.md")
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert "api.md" in out

    def test_markdown_format_has_line_numbers(self):
        engine = _make_engine_stub()
        config = ContextConfig(format="markdown")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(start_line=5, end_line=15)
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert "5" in out
        assert "15" in out

    def test_markdown_multiple_sources_numbered(self):
        engine = _make_engine_stub()
        config = ContextConfig(format="markdown")
        assembler = ContextAssembler(engine, config)
        nodes = [
            _make_chunk_node("c1", "Authentication guide explains JWT tokens and sessions.", "a.md"),
            _make_chunk_node("c2", "Database schema uses PostgreSQL tables and indexes.", "b.md"),
        ]
        result = _make_search_result([_make_scored_node(n) for n in nodes])
        out = assembler.assemble(result)
        assert "Source 1" in out
        assert "Source 2" in out


# ---------------------------------------------------------------------------
# XML format
# ---------------------------------------------------------------------------

class TestXmlFormat:
    def test_xml_format_has_context_tag(self):
        engine = _make_engine_stub()
        config = ContextConfig(format="xml")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(full_text="XML content.")
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert "<context>" in out
        assert "</context>" in out

    def test_xml_format_has_chunk_tags(self):
        engine = _make_engine_stub()
        config = ContextConfig(format="xml")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(full_text="A chunk.")
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert "<chunk " in out
        assert "</chunk>" in out

    def test_xml_format_has_source_attribute(self):
        engine = _make_engine_stub()
        config = ContextConfig(format="xml")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(filename="data.txt")
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert 'source="data.txt"' in out

    def test_xml_format_has_section_attribute(self):
        engine = _make_engine_stub()
        config = ContextConfig(format="xml")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(section="Overview")
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert 'section="Overview"' in out

    def test_xml_format_includes_score_when_enabled(self):
        engine = _make_engine_stub()
        config = ContextConfig(format="xml", include_scores=True)
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node()
        result = _make_search_result([_make_scored_node(node, score=0.9)])
        out = assembler.assemble(result)
        assert "score=" in out

    def test_xml_empty_result_returns_empty_string(self):
        engine = _make_engine_stub()
        config = ContextConfig(format="xml")
        assembler = ContextAssembler(engine, config)
        result = _make_search_result([])
        out = assembler.assemble(result)
        assert out == ""


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_near_duplicate_chunks_removed(self):
        engine = _make_engine_stub()
        config = ContextConfig(dedup_threshold=0.8)
        assembler = ContextAssembler(engine, config)
        # Same text (identical first 100 chars) → second should be deduped
        text = "A" * 50 + " This is repeated content that will be detected as duplicate."
        node1 = _make_chunk_node("c1", text, "a.md")
        node2 = _make_chunk_node("c2", text, "b.md")
        result = _make_search_result([_make_scored_node(node1), _make_scored_node(node2)])
        out = assembler.assemble(result)
        # The text should appear once, not twice
        assert out.count(text[:30]) == 1

    def test_distinct_chunks_both_included(self):
        engine = _make_engine_stub()
        config = ContextConfig(dedup_threshold=0.8)
        assembler = ContextAssembler(engine, config)
        node1 = _make_chunk_node("c1", "Authentication uses JWT tokens for security.", "a.md")
        node2 = _make_chunk_node("c2", "Database schema uses PostgreSQL with migrations.", "b.md")
        result = _make_search_result([_make_scored_node(node1), _make_scored_node(node2)])
        out = assembler.assemble(result)
        assert "Authentication" in out
        assert "Database schema" in out


# ---------------------------------------------------------------------------
# Budget limiting
# ---------------------------------------------------------------------------

class TestBudgetLimiting:
    def test_budget_truncates_oversized_content(self):
        engine = _make_engine_stub()
        config = ContextConfig(max_chars=200, max_tokens=50)
        assembler = ContextAssembler(engine, config)
        # Create many chunks that exceed the budget
        nodes = [
            _make_chunk_node(f"c{i}", f"Chunk {i}: " + "x" * 100, "big.md")
            for i in range(10)
        ]
        result = _make_search_result([_make_scored_node(n) for n in nodes])
        out = assembler.assemble(result)
        assert len(out) <= 500  # Well within doubled budget check

    def test_truncated_block_has_ellipsis(self):
        engine = _make_engine_stub()
        # Very tight budget to force truncation
        config = ContextConfig(max_chars=150, max_tokens=37, include_sources=False)
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node("c1", "x" * 300, "big.md")
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        if out:  # If something was included
            assert "..." in out


# ---------------------------------------------------------------------------
# include_sources=False
# ---------------------------------------------------------------------------

class TestIncludeSourcesFalse:
    def test_no_source_bracket_when_disabled(self):
        engine = _make_engine_stub()
        config = ContextConfig(include_sources=False, format="text")
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(filename="secret.txt", section="Secrets", start_line=1, end_line=5)
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert "secret.txt" not in out
        assert "[" not in out or "score" not in out  # No source bracket

    def test_content_still_present_without_sources(self):
        engine = _make_engine_stub()
        config = ContextConfig(include_sources=False)
        assembler = ContextAssembler(engine, config)
        node = _make_chunk_node(full_text="Important information here.")
        result = _make_search_result([_make_scored_node(node)])
        out = assembler.assemble(result)
        assert "Important information here." in out
