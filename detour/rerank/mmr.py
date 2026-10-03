"""MMR-style re-ranking, the mechanism the whole project turns on.

    score(i) = (1 - lam) * relevance(i) + lam * novelty(i) - beta * max_sim(i, selected)

Relevance and novelty are min-max normalised per user before they are mixed, otherwise the
two terms are on different scales and `lam` would not mean what it looks like it means.
Similarity uses the ALS item factors until item2vec lands.

Items are picked greedily. Afterwards the familiarity anchor rule is enforced: every list
of k keeps at least `anchor_min` tracks by artists the user already knows, so turning the
dial up never produces a list of pure strangers.

Note on `lam = 0`: the list matches relevance order exactly only when `beta` is also 0,
since the similarity penalty is independent of `lam` in the formula above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Candidate:
    """One scored candidate for one user, ready to re-rank."""

    item_id: str
    relevance: float
    novelty: float
    is_anchor: bool
    factors: np.ndarray | None = field(default=None, repr=False)


def minmax(values: list[float]) -> list[float]:
    """Scale to [0, 1]. A flat list maps to all zeros rather than dividing by zero."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def novelty_from_popularity(p: float) -> float:
    """-log2(p), the self-information of an item. Rarer items score higher."""
    return float(-np.log2(max(p, 1e-12)))


def unit(vector: np.ndarray | None) -> np.ndarray | None:
    """Normalise a factor vector so dot products are cosines."""
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else None


def max_similarity(candidate: Candidate, selected: list[Candidate]) -> float:
    """Largest cosine similarity between a candidate and anything already picked."""
    left = unit(candidate.factors)
    if left is None:
        return 0.0
    best = 0.0
    for chosen in selected:
        right = unit(chosen.factors)
        if right is None:
            continue
        best = max(best, float(np.dot(left, right)))
    return best


def score(candidate: Candidate, selected: list[Candidate], lam: float, beta: float) -> float:
    """The MMR score for one candidate given what is already in the list."""
    relevance = (1.0 - lam) * candidate.relevance
    novelty = lam * candidate.novelty
    penalty = beta * max_similarity(candidate, selected)
    return relevance + novelty - penalty


def greedy_select(candidates: list[Candidate], lam: float, beta: float, k: int) -> list[Candidate]:
    """Pick k items one at a time, each time taking the current best score."""
    remaining = list(candidates)
    selected: list[Candidate] = []
    while len(selected) < k and remaining:
        best = max(remaining, key=lambda c: score(c, selected, lam, beta))
        selected.append(best)
        remaining.remove(best)
    return selected


def enforce_anchors(
    selected: list[Candidate],
    candidates: list[Candidate],
    lam: float,
    beta: float,
    anchor_min: int,
) -> list[Candidate]:
    """Swap in known-artist candidates until the list holds at least `anchor_min` of them.

    The lowest-scoring non-anchor picks are the ones replaced, so the cost of meeting the
    floor falls on the weakest part of the list.
    """
    chosen_ids = {c.item_id for c in selected}
    anchors = [c for c in selected if c.is_anchor]
    if len(anchors) >= anchor_min:
        return selected

    spare_anchors = [c for c in candidates if c.is_anchor and c.item_id not in chosen_ids]
    spare_anchors.sort(key=lambda c: c.relevance, reverse=True)

    result = list(selected)
    while len([c for c in result if c.is_anchor]) < anchor_min and spare_anchors:
        swappable = [c for c in result if not c.is_anchor]
        if not swappable:
            break
        weakest = min(swappable, key=lambda c: score(c, result, lam, beta))
        result[result.index(weakest)] = spare_anchors.pop(0)
    return result


def rerank(
    candidates: list[Candidate],
    lam: float,
    beta: float,
    k: int,
    anchor_min: int,
) -> list[Candidate]:
    """Return exactly k items, no duplicates, respecting the familiarity anchor floor."""
    seen = {c.item_id for c in candidates}
    if len(seen) != len(candidates):
        raise ValueError("candidate list contains duplicate item_id values")

    selected = greedy_select(candidates, lam, beta, k)
    return enforce_anchors(selected, candidates, lam, beta, anchor_min)


def prepare(
    item_ids: list[str],
    relevances: list[float],
    popularities: list[float],
    anchors: list[bool],
    factors: list[np.ndarray] | None = None,
) -> list[Candidate]:
    """Build normalised candidates from raw per-user columns."""
    scaled_relevance = minmax(relevances)
    scaled_novelty = minmax([novelty_from_popularity(p) for p in popularities])
    return [
        Candidate(
            item_id=item_ids[i],
            relevance=scaled_relevance[i],
            novelty=scaled_novelty[i],
            is_anchor=anchors[i],
            factors=None if factors is None else factors[i],
        )
        for i in range(len(item_ids))
    ]
