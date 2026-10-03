"""Metric implementations. Every metric is computed @K and averaged over users.

Definitions follow CLAUDE.md section 7 exactly, including the truncated denominator
`min(K, |relevant|)` on both recall variants.

A list containing duplicate items is an error, not something to silently dedupe: a
recommender that fills a slate with the same track twice should fail loudly.

Where a user has nothing to be measured against, the metric returns None rather than 0.
A loyalist with no discoveries did not score zero on discovery recall; the question simply
does not apply to them, and averaging a 0 in would understate every model equally.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np


def check_no_duplicates(recs: Sequence[str]) -> None:
    """Raise if a recommendation list repeats an item."""
    if len(set(recs)) != len(recs):
        repeated = [item for item, count in Counter(recs).items() if count > 1]
        raise ValueError(f"recommendation list contains duplicates: {sorted(repeated)}")


def recall_at_k(recs: Sequence[str], relevant: set[str], k: int) -> float | None:
    """Hits in the relevant set divided by min(k, len(relevant))."""
    check_no_duplicates(recs)
    if not relevant:
        return None
    hits = sum(1 for item in recs[:k] if item in relevant)
    return hits / min(k, len(relevant))


def ndcg_at_k(recs: Sequence[str], relevant: set[str], k: int) -> float | None:
    """Binary relevance with a log2 discount."""
    check_no_duplicates(recs)
    if not relevant:
        return None
    gains = [1.0 if item in relevant else 0.0 for item in recs[:k]]
    dcg = sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
    return dcg / idcg if idcg > 0 else None


def discovery_recall_at_k(rec_artists: Sequence[str], discovery: set[str], k: int) -> float | None:
    """Artist-level hits among artists absent from the user's train history."""
    if not discovery:
        return None
    found = {artist for artist in rec_artists[:k] if artist in discovery}
    return len(found) / min(k, len(discovery))


def novelty_at_k(recs: Sequence[str], popularity: dict[str, float], k: int) -> float | None:
    """Mean of -log2(p_i), where p_i is the share of train users who played item i."""
    check_no_duplicates(recs)
    window = recs[:k]
    if not window:
        return None
    values = [-math.log2(max(popularity.get(item, 1e-6), 1e-12)) for item in window]
    return sum(values) / len(values)


def ild_at_k(recs: Sequence[str], factors: dict[str, np.ndarray], k: int) -> float | None:
    """Mean pairwise cosine distance of the recommended items."""
    window = [item for item in recs[:k] if item in factors]
    if len(window) < 2:
        return None

    vectors = []
    for item in window:
        vector = np.asarray(factors[item], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            vectors.append(vector / norm)
    if len(vectors) < 2:
        return None

    distances = [
        1.0 - float(np.dot(vectors[i], vectors[j]))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    return sum(distances) / len(distances)


def serendipity_at_k(
    recs: Sequence[str], relevant: set[str], popular_top: set[str], k: int
) -> float | None:
    """Share of recs that are relevant AND outside the popularity baseline's top-50."""
    check_no_duplicates(recs)
    window = recs[:k]
    if not window:
        return None
    hits = sum(1 for item in window if item in relevant and item not in popular_top)
    return hits / len(window)


def familiarity_anchor_rate(rec_artists: Sequence[str], known: set[str], k: int) -> float | None:
    """Share of recs from artists already in the user's train history."""
    window = rec_artists[:k]
    if not window:
        return None
    return sum(1 for artist in window if artist in known) / len(window)


def catalogue_coverage(all_recs: Sequence[Sequence[str]], catalogue_size: int) -> float:
    """Distinct recommended items across all users, over the catalogue size."""
    if catalogue_size <= 0:
        return 0.0
    distinct = {item for recs in all_recs for item in recs}
    return len(distinct) / catalogue_size


def exposure_gini(all_recs: Sequence[Sequence[str]], catalogue: Sequence[str]) -> float:
    """Gini coefficient of recommendation counts across the whole catalogue.

    0 means every item is recommended equally often, 1 means a single item takes
    everything. Items that are never recommended count as zeros, which is the point:
    a model that only ever surfaces the same hits should score near 1.
    """
    if not catalogue:
        return 0.0
    counts = Counter(item for recs in all_recs for item in recs)
    values = np.array([counts.get(item, 0) for item in catalogue], dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0

    values.sort()
    n = values.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (index * values).sum()) / (n * total) - (n + 1.0) / n)


def mean_ignoring_none(values: Sequence[float | None]) -> float | None:
    """Average the users the metric applies to, ignoring the ones it does not."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)
