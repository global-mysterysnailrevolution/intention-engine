"""Document chunking with structural awareness.

Supports: plain text, markdown, and code files.
Chunks respect paragraph/heading/function boundaries when possible.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re
import os


@dataclass
class Chunk:
    """A chunk of text from a document."""
    text: str
    index: int                   # Position in document (0-based)
    start_line: int              # Line number in source
    end_line: int                # Line number in source
    metadata: dict = field(default_factory=dict)
    # Populated by ingestion:
    section: str = ""            # Heading/section this chunk belongs to
    doc_path: str = ""           # Source file path
    doc_id: str = ""             # Parent document node ID


@dataclass
class ChunkerConfig:
    chunk_size: int = 512        # Target chunk size in characters
    chunk_overlap: int = 64      # Overlap between consecutive chunks
    min_chunk_size: int = 50     # Minimum chunk size (don't create tiny fragments)
    respect_boundaries: bool = True  # Try to break at paragraph/heading boundaries


class DocumentChunker:
    """Chunks documents with structural awareness."""

    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()

    def chunk_text(self, text: str, file_path: str = "") -> list[Chunk]:
        """Chunk a text string. Detects format from file extension."""
        ext = os.path.splitext(file_path)[1].lower() if file_path else ""

        if ext in (".md", ".markdown"):
            return self._chunk_markdown(text, file_path)
        elif ext in (".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h"):
            return self._chunk_code(text, file_path)
        else:
            return self._chunk_plain(text, file_path)

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """Read and chunk a file."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return self.chunk_text(text, file_path)

    def _chunk_markdown(self, text: str, file_path: str) -> list[Chunk]:
        """Chunk markdown: split on headings first, then by size."""
        # Split into sections by headings
        sections = re.split(r'^(#{1,6}\s+.+)$', text, flags=re.MULTILINE)

        chunks = []
        current_section = ""
        current_line = 1

        for part in sections:
            part = part.strip()
            if not part:
                continue

            if re.match(r'^#{1,6}\s+', part):
                current_section = part.lstrip('#').strip()
                continue

            # Split this section's text into chunks
            sub_chunks = self._split_by_size(part, current_line)
            for sc in sub_chunks:
                sc.section = current_section
                sc.doc_path = file_path
                sc.index = len(chunks)
                chunks.append(sc)

            current_line += part.count('\n') + 1

        return chunks if chunks else self._chunk_plain(text, file_path)

    def _chunk_code(self, text: str, file_path: str) -> list[Chunk]:
        """Chunk code: split on function/class definitions, then by size."""
        # Split on function/class boundaries
        # Match common patterns: def, function, class, fn, func, pub fn, etc.
        boundary_pattern = re.compile(
            r'^(?:(?:pub\s+)?(?:async\s+)?(?:def|fn|func|function)\s+\w+|'
            r'(?:export\s+)?(?:default\s+)?class\s+\w+|'
            r'(?:pub\s+)?(?:impl|struct|enum|trait|interface)\s+\w+)',
            re.MULTILINE
        )

        boundaries = [m.start() for m in boundary_pattern.finditer(text)]

        if not boundaries:
            return self._chunk_plain(text, file_path)

        # Add start and end
        if boundaries[0] != 0:
            boundaries.insert(0, 0)

        chunks = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            segment = text[start:end].strip()

            if not segment:
                continue

            # If segment is too large, split further by size
            if len(segment) > self.config.chunk_size * 2:
                line_offset = text[:start].count('\n') + 1
                sub_chunks = self._split_by_size(segment, line_offset)
            else:
                line_offset = text[:start].count('\n') + 1
                sub_chunks = [Chunk(
                    text=segment,
                    index=0,
                    start_line=line_offset,
                    end_line=line_offset + segment.count('\n'),
                )]

            for sc in sub_chunks:
                sc.doc_path = file_path
                sc.index = len(chunks)
                chunks.append(sc)

        return chunks if chunks else self._chunk_plain(text, file_path)

    def _chunk_plain(self, text: str, file_path: str) -> list[Chunk]:
        """Chunk plain text: split on paragraphs first, then by size."""
        chunks = self._split_by_size(text, 1)
        for i, c in enumerate(chunks):
            c.doc_path = file_path
            c.index = i
        return chunks

    def _split_by_size(self, text: str, start_line: int = 1) -> list[Chunk]:
        """Split text into chunks of approximately chunk_size characters.

        Tries to break at paragraph boundaries (\\n\\n), then sentence boundaries (.),
        then word boundaries (space), falling back to hard cut.
        """
        if len(text) <= self.config.chunk_size:
            if len(text) >= self.config.min_chunk_size:
                return [Chunk(
                    text=text.strip(),
                    index=0,
                    start_line=start_line,
                    end_line=start_line + text.count('\n'),
                )]
            return []

        chunks = []
        pos = 0
        current_line = start_line

        while pos < len(text):
            end = min(pos + self.config.chunk_size, len(text))

            if end < len(text) and self.config.respect_boundaries:
                # Try to find a good break point
                segment = text[pos:end]

                # Try paragraph boundary
                break_idx = segment.rfind('\n\n')
                if break_idx > self.config.min_chunk_size:
                    end = pos + break_idx + 2
                else:
                    # Try sentence boundary
                    break_idx = segment.rfind('. ')
                    if break_idx > self.config.min_chunk_size:
                        end = pos + break_idx + 2
                    else:
                        # Try word boundary
                        break_idx = segment.rfind(' ')
                        if break_idx > self.config.min_chunk_size:
                            end = pos + break_idx + 1

            chunk_text = text[pos:end].strip()
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_line=current_line,
                    end_line=current_line + chunk_text.count('\n'),
                ))

            current_line += text[pos:end].count('\n')
            # Apply overlap
            pos = end - self.config.chunk_overlap if end < len(text) else end

        return chunks


def detect_file_type(path: str) -> str:
    """Detect the type/ontology of a file from its extension."""
    ext = os.path.splitext(path)[1].lower()
    type_map = {
        ".md": "documentation", ".markdown": "documentation",
        ".txt": "text", ".rst": "documentation",
        ".py": "code_python", ".js": "code_javascript", ".ts": "code_typescript",
        ".go": "code_go", ".rs": "code_rust", ".java": "code_java",
        ".c": "code_c", ".cpp": "code_cpp", ".h": "code_c",
        ".json": "config", ".yaml": "config", ".yml": "config", ".toml": "config",
        ".html": "web", ".css": "web",
        ".sql": "database",
        ".sh": "script", ".bash": "script",
    }
    return type_map.get(ext, "text")
