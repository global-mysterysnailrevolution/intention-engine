"""Temporal embedding and time-scoped utilities."""
from __future__ import annotations

import time as _time

import numpy as np

# Fourier frequency bases (in Hz): daily, weekly, monthly, yearly cycles
_FREQ_BASES = [
    1 / 86400,  # day
    1 / 604800,  # week
    1 / 2_592_000,  # month (~30 days)
    1 / 31_536_000,  # year
]


def temporal_embedding(
    text_embedding: np.ndarray,
    timestamp: float,
    dim: int = 16,
) -> np.ndarray:
    """Concatenate text embedding with Fourier temporal features.

    Encodes time as sin/cos pairs at multiple frequency bases,
    plus exponential recency decay and raw age.  This lets the system
    assess temporal differences structurally, not just as timestamps.

    Args:
        text_embedding: Content embedding vector.
        timestamp: Unix epoch seconds.
        dim: Number of temporal dimensions (default 16).

    Returns:
        Concatenated vector of shape ``(len(text_embedding) + dim,)``.
    """
    features: list[float] = []
    for freq in _FREQ_BASES:
        features.append(float(np.sin(2 * np.pi * freq * timestamp)))
        features.append(float(np.cos(2 * np.pi * freq * timestamp)))

    # Recency features
    age_seconds = max(0.0, _time.time() - timestamp)
    age_years = age_seconds / 31_536_000
    features.append(float(np.exp(-age_years)))  # exponential recency (1.0 = now, decays)
    features.append(min(age_years, 10.0))  # capped linear age

    # Pad or truncate to dim
    t_vec = np.array(features[:dim], dtype=np.float64)
    if len(t_vec) < dim:
        t_vec = np.pad(t_vec, (0, dim - len(t_vec)))

    return np.concatenate([text_embedding, t_vec])


def is_edge_valid_at(
    valid_from: float,
    valid_until: float | None,
    query_time: float,
) -> bool:
    """Check if an edge's validity interval contains the query time."""
    if query_time < valid_from:
        return False
    if valid_until is not None and query_time >= valid_until:
        return False
    return True


def temporal_similarity(
    t1: float,
    t2: float,
    half_life: float = 604800,
) -> float:
    """Compute temporal proximity score between two timestamps.

    Returns value in ``(0, 1]`` where ``1.0`` means same time.
    *half_life* is in seconds (default = 1 week).
    """
    dt = abs(t1 - t2)
    return float(np.exp(-0.693 * dt / half_life))  # ln(2) ~ 0.693
