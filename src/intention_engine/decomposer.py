import re
import numpy as np
from .models import Intention, Predicate, SearchScope


class IntentionDecomposer:
    """Decomposes natural language intentions into utility predicates."""

    def decompose(self, raw: str, scope: SearchScope | None = None) -> Intention:
        """Rule-based decomposition: extract noun/verb phrases as predicates."""
        predicates = self._extract_predicates(raw)
        return Intention(
            raw=raw,
            predicates=predicates,
            embedding=None,  # Set later by embedding pipeline
            scope=scope or SearchScope(),
        )

    def _extract_predicates(self, text: str) -> list[Predicate]:
        # Normalize
        text = text.lower().strip()
        # Split on common delimiters
        chunks = re.split(r'\b(?:and|or|for|in|with|that|which|to|the|a|an)\b|[,;]', text)
        # Filter empty/short chunks
        chunks = [c.strip() for c in chunks if len(c.strip()) > 3]
        if not chunks:
            chunks = [text]  # Fallback: use whole text
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for c in chunks:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        # Equal weights
        w = 1.0 / len(unique) if unique else 1.0
        return [Predicate(text=c, weight=w) for c in unique]
