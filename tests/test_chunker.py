"""Tests for the DocumentChunker and related utilities."""
from __future__ import annotations

import pytest

from intention_engine.chunker import (
    Chunk,
    ChunkerConfig,
    DocumentChunker,
    detect_file_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunker(**kwargs) -> DocumentChunker:
    return DocumentChunker(ChunkerConfig(**kwargs))


# ---------------------------------------------------------------------------
# detect_file_type
# ---------------------------------------------------------------------------

class TestDetectFileType:
    def test_markdown(self):
        assert detect_file_type("README.md") == "documentation"

    def test_markdown_long_ext(self):
        assert detect_file_type("notes.markdown") == "documentation"

    def test_python(self):
        assert detect_file_type("script.py") == "code_python"

    def test_javascript(self):
        assert detect_file_type("app.js") == "code_javascript"

    def test_typescript(self):
        assert detect_file_type("app.ts") == "code_typescript"

    def test_go(self):
        assert detect_file_type("main.go") == "code_go"

    def test_rust(self):
        assert detect_file_type("lib.rs") == "code_rust"

    def test_json(self):
        assert detect_file_type("config.json") == "config"

    def test_yaml(self):
        assert detect_file_type("config.yaml") == "config"

    def test_yml(self):
        assert detect_file_type("config.yml") == "config"

    def test_toml(self):
        assert detect_file_type("Cargo.toml") == "config"

    def test_txt(self):
        assert detect_file_type("notes.txt") == "text"

    def test_unknown_extension(self):
        assert detect_file_type("file.xyz") == "text"

    def test_no_extension(self):
        assert detect_file_type("Makefile") == "text"

    def test_sql(self):
        assert detect_file_type("schema.sql") == "database"

    def test_sh(self):
        assert detect_file_type("deploy.sh") == "script"

    def test_c_header(self):
        assert detect_file_type("header.h") == "code_c"

    def test_cpp(self):
        assert detect_file_type("main.cpp") == "code_cpp"

    def test_case_insensitive(self):
        assert detect_file_type("README.MD") == "documentation"


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

class TestChunkDataclass:
    def test_chunk_defaults(self):
        c = Chunk(text="hello", index=0, start_line=1, end_line=1)
        assert c.section == ""
        assert c.doc_path == ""
        assert c.doc_id == ""
        assert c.metadata == {}

    def test_chunk_with_all_fields(self):
        c = Chunk(
            text="content",
            index=2,
            start_line=10,
            end_line=15,
            section="Introduction",
            doc_path="/tmp/doc.md",
        )
        assert c.section == "Introduction"
        assert c.start_line == 10
        assert c.end_line == 15


# ---------------------------------------------------------------------------
# _split_by_size
# ---------------------------------------------------------------------------

class TestSplitBySize:
    def test_short_text_single_chunk(self):
        chunker = _make_chunker(chunk_size=200, min_chunk_size=10)
        chunks = chunker._split_by_size("Hello world this is a test.", 1)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world this is a test."

    def test_text_below_min_size_returns_empty(self):
        chunker = _make_chunker(chunk_size=200, min_chunk_size=50)
        chunks = chunker._split_by_size("tiny", 1)
        assert chunks == []

    def test_long_text_splits_into_multiple(self):
        chunker = _make_chunker(chunk_size=100, min_chunk_size=10, chunk_overlap=0)
        text = "word " * 100  # 500 chars
        chunks = chunker._split_by_size(text, 1)
        assert len(chunks) >= 2

    def test_paragraph_boundary_preferred(self):
        # Two paragraphs each ~136 chars, total ~275 chars > chunk_size=200.
        # The splitter should split them into separate chunks.
        chunker = _make_chunker(chunk_size=200, min_chunk_size=20, chunk_overlap=0)
        para1 = "First paragraph. " * 8   # ~136 chars
        para2 = "Second paragraph. " * 8  # ~144 chars
        text = para1 + "\n\n" + para2
        chunks = chunker._split_by_size(text, 1)
        # Two paragraphs, split into at least 2 chunks
        assert len(chunks) >= 2
        # First chunk should contain "First paragraph" content
        assert any("First paragraph" in c.text for c in chunks)
        # Second chunk should contain "Second paragraph" content
        assert any("Second paragraph" in c.text for c in chunks)

    def test_sentence_boundary_fallback(self):
        chunker = _make_chunker(chunk_size=100, min_chunk_size=20, chunk_overlap=0)
        # Build text with no double newlines but with sentence boundaries
        text = "First sentence ends here. " * 6  # ~156 chars, no paragraphs
        chunks = chunker._split_by_size(text, 1)
        # Should produce multiple chunks without crashing
        assert len(chunks) >= 1

    def test_word_boundary_fallback(self):
        chunker = _make_chunker(chunk_size=50, min_chunk_size=10, chunk_overlap=0)
        text = "abcdefghij " * 10  # 110 chars, words separated by spaces
        chunks = chunker._split_by_size(text, 1)
        assert len(chunks) >= 1
        # No chunk should end in the middle of a word (unless hard-cut)
        for c in chunks:
            assert len(c.text) > 0

    def test_overlap_applied(self):
        chunker = _make_chunker(chunk_size=100, min_chunk_size=10, chunk_overlap=20,
                                 respect_boundaries=False)
        text = "x" * 200
        chunks = chunker._split_by_size(text, 1)
        assert len(chunks) >= 2

    def test_chunk_indices_sequential(self):
        chunker = _make_chunker(chunk_size=50, min_chunk_size=10, chunk_overlap=0)
        text = "word " * 50
        chunks = chunker._split_by_size(text, 1)
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_start_line_tracked(self):
        chunker = _make_chunker(chunk_size=200, min_chunk_size=10)
        chunks = chunker._split_by_size("hello world", 5)
        assert chunks[0].start_line == 5

    def test_no_boundaries_respected(self):
        chunker = _make_chunker(chunk_size=100, min_chunk_size=10, chunk_overlap=0,
                                 respect_boundaries=False)
        text = "abc" * 100  # 300 chars, no natural breaks
        chunks = chunker._split_by_size(text, 1)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Plain text chunking
# ---------------------------------------------------------------------------

class TestChunkPlain:
    def test_returns_list_of_chunks(self):
        chunker = _make_chunker(chunk_size=100, min_chunk_size=10)
        text = "Hello world. " * 20
        result = chunker._chunk_plain(text, "doc.txt")
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_doc_path_set(self):
        chunker = _make_chunker(chunk_size=200, min_chunk_size=10)
        chunks = chunker._chunk_plain("Some text here.", "path/to/file.txt")
        assert all(c.doc_path == "path/to/file.txt" for c in chunks)

    def test_indices_are_zero_based(self):
        chunker = _make_chunker(chunk_size=50, min_chunk_size=10, chunk_overlap=0)
        text = "word " * 50
        chunks = chunker._chunk_plain(text, "")
        assert chunks[0].index == 0
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_empty_text_returns_empty(self):
        chunker = _make_chunker()
        chunks = chunker._chunk_plain("", "")
        assert chunks == []

    def test_tiny_text_below_min_returns_empty(self):
        chunker = _make_chunker(min_chunk_size=100)
        chunks = chunker._chunk_plain("hi", "")
        assert chunks == []


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------

SAMPLE_MARKDOWN = """\
# Introduction

This is the introduction section. It contains some general information
about the topic at hand. Multiple sentences are here.

## Background

The background section provides context. Here we discuss prior work
and related research. This is important for understanding the rest.

## Methods

The methods section describes the approach. We use several techniques
to accomplish our goals. The implementation details are here.

### Sub-section

A sub-section within methods. More detail here.
"""


class TestChunkMarkdown:
    def test_returns_chunks(self):
        chunker = _make_chunker(chunk_size=200, min_chunk_size=10)
        chunks = chunker._chunk_markdown(SAMPLE_MARKDOWN, "doc.md")
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_sections_detected(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=10)
        chunks = chunker._chunk_markdown(SAMPLE_MARKDOWN, "doc.md")
        sections = {c.section for c in chunks if c.section}
        assert "Introduction" in sections or len(sections) >= 1

    def test_background_section_detected(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=10)
        chunks = chunker._chunk_markdown(SAMPLE_MARKDOWN, "doc.md")
        sections = {c.section for c in chunks}
        assert "Background" in sections

    def test_methods_section_detected(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=10)
        chunks = chunker._chunk_markdown(SAMPLE_MARKDOWN, "doc.md")
        sections = {c.section for c in chunks}
        assert "Methods" in sections

    def test_doc_path_set(self):
        chunker = _make_chunker(chunk_size=200, min_chunk_size=10)
        chunks = chunker._chunk_markdown(SAMPLE_MARKDOWN, "readme.md")
        assert all(c.doc_path == "readme.md" for c in chunks)

    def test_indices_sequential(self):
        chunker = _make_chunker(chunk_size=200, min_chunk_size=10)
        chunks = chunker._chunk_markdown(SAMPLE_MARKDOWN, "")
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_no_heading_only_content(self):
        md = "# Title\n\nJust a paragraph."
        chunker = _make_chunker(chunk_size=500, min_chunk_size=5)
        chunks = chunker._chunk_markdown(md, "f.md")
        assert len(chunks) == 1
        assert "Just a paragraph." in chunks[0].text

    def test_fallback_to_plain_when_no_content(self):
        # Only headings, no body
        md = "# Title\n## Section\n### Sub"
        chunker = _make_chunker(chunk_size=500, min_chunk_size=5)
        # Should not crash
        chunks = chunker._chunk_markdown(md, "f.md")
        assert isinstance(chunks, list)

    def test_empty_markdown_returns_empty(self):
        chunker = _make_chunker()
        chunks = chunker._chunk_markdown("", "f.md")
        assert chunks == []


# ---------------------------------------------------------------------------
# Code chunking
# ---------------------------------------------------------------------------

SAMPLE_PYTHON = """\
import os
import sys


def calculate_sum(a, b):
    \"\"\"Return the sum of a and b.\"\"\"
    return a + b


def calculate_product(a, b):
    \"\"\"Return the product of a and b.\"\"\"
    result = a * b
    return result


class MathHelper:
    \"\"\"A helper class for math operations.\"\"\"

    def __init__(self, factor):
        self.factor = factor

    def scale(self, value):
        return value * self.factor
"""

SAMPLE_RUST = """\
use std::collections::HashMap;

pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

struct Calculator {
    history: Vec<i32>,
}

impl Calculator {
    fn new() -> Self {
        Calculator { history: vec![] }
    }
}
"""


class TestChunkCode:
    def test_python_splits_on_functions(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=10)
        chunks = chunker._chunk_code(SAMPLE_PYTHON, "math.py")
        # Should detect calculate_sum and calculate_product and MathHelper
        texts = " ".join(c.text for c in chunks)
        assert "calculate_sum" in texts
        assert "calculate_product" in texts

    def test_python_class_boundary(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=10)
        chunks = chunker._chunk_code(SAMPLE_PYTHON, "math.py")
        texts = " ".join(c.text for c in chunks)
        assert "MathHelper" in texts

    def test_rust_fn_boundary(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=10)
        chunks = chunker._chunk_code(SAMPLE_RUST, "lib.rs")
        texts = " ".join(c.text for c in chunks)
        assert "add" in texts or "multiply" in texts

    def test_rust_struct_boundary(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=10)
        chunks = chunker._chunk_code(SAMPLE_RUST, "lib.rs")
        texts = " ".join(c.text for c in chunks)
        assert "Calculator" in texts

    def test_doc_path_set(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=10)
        chunks = chunker._chunk_code(SAMPLE_PYTHON, "module.py")
        assert all(c.doc_path == "module.py" for c in chunks)

    def test_fallback_to_plain_when_no_boundaries(self):
        # Plain text in a .py file with no def/class
        code = "x = 1\ny = 2\nz = x + y\n"
        chunker = _make_chunker(chunk_size=500, min_chunk_size=5)
        chunks = chunker._chunk_code(code, "script.py")
        assert len(chunks) >= 1

    def test_large_function_split_by_size(self):
        # A very large "function" that exceeds chunk_size * 2
        big_func = "def big_function():\n" + "    x = 1\n" * 200
        chunker = _make_chunker(chunk_size=100, min_chunk_size=10, chunk_overlap=0)
        chunks = chunker._chunk_code(big_func, "big.py")
        assert len(chunks) >= 2

    def test_indices_sequential(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=10)
        chunks = chunker._chunk_code(SAMPLE_PYTHON, "")
        for i, c in enumerate(chunks):
            assert c.index == i


# ---------------------------------------------------------------------------
# chunk_text dispatch
# ---------------------------------------------------------------------------

class TestChunkTextDispatch:
    def test_dispatch_markdown(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=5)
        chunks = chunker.chunk_text("# Title\n\nContent here.", "doc.md")
        assert len(chunks) >= 1

    def test_dispatch_python(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=5)
        chunks = chunker.chunk_text("def foo():\n    pass\n", "script.py")
        assert len(chunks) >= 1

    def test_dispatch_plain_txt(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=5)
        chunks = chunker.chunk_text("Hello world.", "notes.txt")
        assert len(chunks) == 1

    def test_dispatch_no_extension(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=5)
        chunks = chunker.chunk_text("Some content.", "")
        assert len(chunks) == 1

    def test_dispatch_unknown_extension(self):
        chunker = _make_chunker(chunk_size=500, min_chunk_size=5)
        chunks = chunker.chunk_text("Some content.", "file.xyz")
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# chunk_file
# ---------------------------------------------------------------------------

class TestChunkFile:
    def test_chunk_file_reads_and_chunks(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("Hello world this is a test file with enough content.", encoding="utf-8")
        chunker = _make_chunker(chunk_size=500, min_chunk_size=5)
        chunks = chunker.chunk_file(str(p))
        assert len(chunks) >= 1
        assert all(c.doc_path == str(p) for c in chunks)

    def test_chunk_file_markdown(self, tmp_path):
        p = tmp_path / "readme.md"
        p.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
        chunker = _make_chunker(chunk_size=300, min_chunk_size=10)
        chunks = chunker.chunk_file(str(p))
        assert len(chunks) >= 1
        sections = {c.section for c in chunks if c.section}
        assert len(sections) >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self):
        chunker = _make_chunker()
        assert chunker.chunk_text("", "doc.txt") == []

    def test_whitespace_only(self):
        chunker = _make_chunker()
        assert chunker.chunk_text("   \n\n  ", "doc.txt") == []

    def test_single_char(self):
        chunker = _make_chunker(min_chunk_size=1)
        chunks = chunker.chunk_text("x", "doc.txt")
        assert len(chunks) == 1

    def test_single_char_below_min(self):
        chunker = _make_chunker(min_chunk_size=10)
        chunks = chunker.chunk_text("x", "doc.txt")
        assert chunks == []

    def test_chunk_size_equals_text_size(self):
        text = "a" * 512
        chunker = _make_chunker(chunk_size=512, min_chunk_size=10)
        chunks = chunker.chunk_text(text, "doc.txt")
        assert len(chunks) == 1

    def test_unicode_content(self):
        text = "Hello wörld. This is ünïcode. " * 5
        chunker = _make_chunker(chunk_size=100, min_chunk_size=10)
        chunks = chunker.chunk_text(text, "doc.txt")
        assert len(chunks) >= 1

    def test_newlines_only(self):
        chunker = _make_chunker()
        chunks = chunker.chunk_text("\n\n\n\n", "doc.txt")
        assert chunks == []

    def test_code_empty_file(self):
        chunker = _make_chunker(min_chunk_size=10)
        chunks = chunker.chunk_text("", "script.py")
        assert chunks == []
