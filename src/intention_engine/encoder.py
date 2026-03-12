"""Built-in mock encoder for agent use without ML dependencies.

Uses deterministic hash-based projection with word-level semantic boosting.
Same text always produces the same embedding. Texts sharing words produce
similar embeddings — good enough for structural discovery. The real semantic
signal comes from how the agent describes nodes, not from model sophistication.
"""

from __future__ import annotations

import numpy as np


class HashEncoder:
    """Deterministic text encoder using hash-based projection.

    Maps text to a fixed-dimensional vector where:
    - Same text → same vector (deterministic)
    - Shared words → overlapping boosted dimensions → higher cosine similarity
    - No ML model required
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]

        # Deterministic base vector from text hash
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        vec = rng.randn(self.dim).astype(np.float32)

        # Boost dimensions based on individual words.
        # Each unique word hashes to a small set of dimensions.
        # Texts sharing words will have overlapping boosts → higher similarity.
        words = set(text.lower().split())
        for word in words:
            if len(word) < 2:
                continue
            word_hash = abs(hash(word)) % (2**31)
            # Each word boosts 3 dimensions
            for offset in range(3):
                dim_idx = (word_hash + offset * 7) % self.dim
                boost = 3.0 / (1 + offset)  # Decreasing boost
                vec[dim_idx] += boost

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        self._cache[text] = vec
        return vec

    def clear_cache(self) -> None:
        self._cache.clear()
