"""Context assembly for LLM consumption.

Takes search results from the IntentionEngine and formats them into
a context string suitable for insertion into an LLM prompt.
Handles deduplication, chunk expansion, source attribution, and token budgeting.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ContextConfig:
    """Configuration for context assembly."""
    max_tokens: int = 4000          # Approximate token budget (chars / 4)
    max_chars: int = 16000          # Hard character limit
    include_sources: bool = True    # Include source file/line metadata
    include_scores: bool = False    # Include relevance scores
    expand_adjacent: bool = True    # Include adjacent chunks for continuity
    dedup_threshold: float = 0.8    # Text overlap threshold for deduplication
    separator: str = "\n---\n"      # Separator between context blocks
    header: str = ""                # Optional header before context
    format: str = "text"            # "text", "markdown", "xml"


class ContextAssembler:
    """Assembles search results into LLM-ready context."""

    def __init__(self, engine, config: ContextConfig | None = None):
        """
        Args:
            engine: IntentionEngine instance (for node lookups and adjacent chunk expansion)
            config: Context assembly configuration
        """
        self.engine = engine
        self.config = config or ContextConfig()

    def assemble(self, result, query: str = "") -> str:
        """Assemble a SearchResult into a formatted context string.

        Args:
            result: SearchResult from engine.search()
            query: Original query (for header)

        Returns:
            Formatted context string ready for LLM consumption
        """
        # Collect chunk texts with metadata
        blocks = []
        seen_texts: set[str] = set()  # For deduplication

        for scored_node in result.nodes:
            node = scored_node.node
            meta = node.metadata

            # Get the full text (chunks store full_text in metadata)
            full_text = meta.get("full_text", meta.get("description", ""))
            if not full_text:
                continue

            # Deduplication: skip if too similar to already-included text
            if self._is_duplicate(full_text, seen_texts):
                continue
            seen_texts.add(full_text[:100])  # Use first 100 chars as fingerprint

            # Build the block
            block = self._format_block(
                text=full_text,
                source=meta.get("filename", meta.get("path", "")),
                section=meta.get("section", ""),
                lines=(meta.get("start_line", 0), meta.get("end_line", 0)),
                score=scored_node.score,
                source_type=scored_node.source,
                ontology=node.ontology,
            )
            blocks.append(block)

        # Expand with adjacent chunks if configured
        if self.config.expand_adjacent:
            blocks = self._expand_adjacent(blocks, result)

        # Apply token budget
        blocks = self._apply_budget(blocks)

        # Format output
        return self._format_output(blocks, query)

    def _format_block(
        self,
        text: str,
        source: str = "",
        section: str = "",
        lines: tuple[int, int] = (0, 0),
        score: float = 0.0,
        source_type: str = "",
        ontology: str = "",
    ) -> dict:
        """Format a single context block."""
        block = {
            "text": text,
            "source": source,
            "section": section,
            "lines": lines,
            "score": score,
            "source_type": source_type,
            "ontology": ontology,
        }
        return block

    def _is_duplicate(self, text: str, seen: set[str]) -> bool:
        """Check if text is too similar to already-seen content.

        Uses exact match on the first 100 characters as the fingerprint.
        Only texts that share an identical prefix are considered duplicates.
        """
        fingerprint = text[:100]
        return fingerprint in seen

    def _expand_adjacent(self, blocks: list[dict], result) -> list[dict]:
        """Add adjacent chunks for continuity when a chunk is included.

        For now, adjacent chunks are already connected via hyperedges
        and will be found by the exploit phase. This is a placeholder
        for more aggressive expansion if needed.
        """
        return blocks

    def _apply_budget(self, blocks: list[dict]) -> list[dict]:
        """Trim blocks to fit within token/char budget."""
        max_chars = min(self.config.max_chars, self.config.max_tokens * 4)
        total = 0
        kept = []
        for block in blocks:
            block_size = len(block["text"]) + 50  # Overhead for metadata
            if total + block_size > max_chars:
                # Try to include a truncated version
                remaining = max_chars - total - 50
                if remaining > 100:
                    block = dict(block)  # shallow copy before mutation
                    block["text"] = block["text"][:remaining] + "..."
                    kept.append(block)
                break
            total += block_size
            kept.append(block)
        return kept

    def _format_output(self, blocks: list[dict], query: str) -> str:
        """Format all blocks into final output string."""
        if not blocks:
            return ""

        fmt = self.config.format

        if fmt == "xml":
            return self._format_xml(blocks, query)
        elif fmt == "markdown":
            return self._format_markdown(blocks, query)
        else:
            return self._format_text(blocks, query)

    def _format_text(self, blocks: list[dict], query: str) -> str:
        parts = []
        if self.config.header:
            parts.append(self.config.header)

        for block in blocks:
            lines = []
            if self.config.include_sources and block["source"]:
                src = block["source"]
                if block["lines"][0]:
                    src += f":{block['lines'][0]}-{block['lines'][1]}"
                if block["section"]:
                    src += f" ({block['section']})"
                lines.append(f"[{src}]")
            if self.config.include_scores:
                lines.append(f"[score: {block['score']:.3f}, via: {block['source_type']}]")
            lines.append(block["text"])
            parts.append("\n".join(lines))

        return self.config.separator.join(parts)

    def _format_markdown(self, blocks: list[dict], query: str) -> str:
        parts = []
        if self.config.header:
            parts.append(self.config.header)

        for i, block in enumerate(blocks):
            lines = []
            src = block["source"] or "inline"
            if block["section"]:
                src += f" > {block['section']}"
            lines.append(f"### Source {i+1}: {src}")
            if block["lines"][0]:
                lines.append(f"*Lines {block['lines'][0]}-{block['lines'][1]}*")
            lines.append("")
            lines.append(block["text"])
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def _format_xml(self, blocks: list[dict], query: str) -> str:
        parts = []
        parts.append("<context>")

        for i, block in enumerate(blocks):
            src = block["source"] or "inline"
            attrs = f'source="{src}"'
            if block["section"]:
                attrs += f' section="{block["section"]}"'
            if block["lines"][0]:
                attrs += f' lines="{block["lines"][0]}-{block["lines"][1]}"'
            if self.config.include_scores:
                attrs += f' score="{block["score"]:.3f}"'
            parts.append(f"  <chunk {attrs}>")
            parts.append(f"    {block['text']}")
            parts.append("  </chunk>")

        parts.append("</context>")
        return "\n".join(parts)
