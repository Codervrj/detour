"""Does the map predict what someone goes on to listen to?

The claim under test: artists near a listener's territory are ones they would actually
adopt. If the map is meaningless, the artists they really adopted will sit no closer than
anything else.

Protocol. Train the embedding on months 1-9. For each held-out listener, place them from
their train history alone, then take the artists they first played in months 10-12. Rank
every candidate artist by distance from that position, and see where the real adoptions
landed.

**Adoption Rank Percentile** is the headline. 0.5 is chance. Lower is better.

The confound that would otherwise sink this
-------------------------------------------
Popular artists are adopted more often *and* sit centrally in embedding space, because
they appear in many sessions. So a model that learned nothing except popularity would
still score well on the naive version of this test.

`matched_percentile` is the honest version: each real adoption is compared only against
artists of **similar global popularity** that the listener did not adopt. If a model beats
chance on the naive metric but not on the matched one, it learned popularity and nothing
else, and the report must say so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from detour.foldin import unit

MATCHED_NEGATIVES = 50


@dataclass(frozen=True)
class AdoptionResult:
    """Scores for one listener."""

    user_id: str
    adoptions: int
    percentile: float | None
    matched_percentile: float | None
    hit_at_k: dict[int, float]
    median_rank: float | None


class ArtistSpace:
    """Artist vectors and popularity, prepared once and reused for every listener."""

    def __init__(self, vectors: dict[str, np.ndarray], popularity: dict[str, int]) -> None:
        self.artists = sorted(vectors)
        self.index = {artist: i for i, artist in enumerate(self.artists)}
        self.matrix = np.vstack([unit(vectors[a]) for a in self.artists])
        self.popularity = np.array(
            [float(popularity.get(a, 0)) for a in self.artists], dtype=np.float64
        )
        self._popularity_order = np.argsort(self.popularity)

    def distances(self, position: np.ndarray) -> np.ndarray:
        """Cosine distance from a position to every artist in the space."""
        return np.asarray(1.0 - self.matrix @ unit(position), dtype=np.float64)

    def popularity_matched(
        self, target: int, count: int, exclude: set[int], rng: np.random.Generator
    ) -> list[int]:
        """Indices of artists with popularity closest to `target`, excluding some.

        Drawn from a neighbourhood in the popularity ordering rather than the whole
        catalogue, so the comparison holds popularity roughly constant.
        """
        position = int(
            np.searchsorted(self.popularity[self._popularity_order], self.popularity[target])
        )
        window = max(count * 4, 200)
        low = max(0, position - window // 2)
        high = min(len(self._popularity_order), low + window)
        pool = [
            int(i)
            for i in self._popularity_order[low:high]
            if int(i) not in exclude and int(i) != target
        ]
        if not pool:
            return []
        if len(pool) <= count:
            return pool
        return [int(i) for i in rng.choice(np.array(pool), size=count, replace=False)]


def percentile_of(distances: np.ndarray, target: int) -> float:
    """Where `target` ranks by distance, as a fraction. 0 is nearest, 1 is furthest."""
    rank = int((distances < distances[target]).sum())
    return rank / max(len(distances) - 1, 1)


def score_listener(
    user_id: str,
    position: np.ndarray,
    adopted: Sequence[str],
    known: set[str],
    space: ArtistSpace,
    k_values: Sequence[int],
    rng: np.random.Generator,
    matched_negatives: int = MATCHED_NEGATIVES,
) -> AdoptionResult:
    """Score one listener's real adoptions against the map."""
    targets = [space.index[a] for a in adopted if a in space.index]
    empty_hits = dict.fromkeys(k_values, 0.0)
    if not targets or not position.any():
        return AdoptionResult(user_id, 0, None, None, empty_hits, None)

    distances = space.distances(position)

    # Artists already in the listener's history are not available to be adopted, so they
    # must not sit in the ranking competing with the real adoptions.
    known_indices = {space.index[a] for a in known if a in space.index}
    mask = np.ones(len(distances), dtype=bool)
    mask[list(known_indices)] = False

    candidate_indices = np.flatnonzero(mask)
    candidate_distances = distances[candidate_indices]
    order = np.argsort(candidate_distances)
    ranked = candidate_indices[order]
    rank_of = {int(artist): position for position, artist in enumerate(ranked)}

    percentiles: list[float] = []
    matched: list[float] = []
    ranks: list[int] = []

    for target in targets:
        if target not in rank_of:
            continue
        rank = rank_of[target]
        ranks.append(rank)
        percentiles.append(rank / max(len(ranked) - 1, 1))

        negatives = space.popularity_matched(target, matched_negatives, known_indices, rng)
        if negatives:
            beaten = sum(1 for n in negatives if distances[target] < distances[n])
            matched.append(1.0 - beaten / len(negatives))

    if not ranks:
        return AdoptionResult(user_id, 0, None, None, empty_hits, None)

    hits = {k: sum(1 for rank in ranks if rank < k) / len(ranks) for k in k_values}
    return AdoptionResult(
        user_id=user_id,
        adoptions=len(ranks),
        percentile=float(np.mean(percentiles)),
        matched_percentile=float(np.mean(matched)) if matched else None,
        hit_at_k=hits,
        median_rank=float(np.median(ranks)),
    )


def aggregate(
    results: Sequence[AdoptionResult], k_values: Sequence[int]
) -> dict[str, float | None]:
    """Average over the listeners the metric applies to."""

    def mean_of(values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return float(np.mean(present)) if present else None

    scored = [r for r in results if r.adoptions > 0]
    return {
        "listeners_scored": len(scored),
        "adoptions_total": sum(r.adoptions for r in scored),
        "adoption_percentile": mean_of([r.percentile for r in scored]),
        "matched_percentile": mean_of([r.matched_percentile for r in scored]),
        "median_rank": mean_of([r.median_rank for r in scored]),
        **{f"hit@{k}": mean_of([r.hit_at_k.get(k) for r in scored]) for k in k_values},
    }
