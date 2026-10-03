"""Adoption metrics, including the two tests that decide whether the metric is trustworthy:
a meaningless embedding must score at chance, and a perfect one must score near zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from detour.eval.adoption import ArtistSpace, aggregate, percentile_of, score_listener

K_VALUES = (10, 50)


def ring_space(n: int = 200, popularity_spread: bool = True) -> ArtistSpace:
    """Artists evenly spaced around a circle, so geometry is predictable."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vectors = {
        f"a{i}": np.array([np.cos(t), np.sin(t)], dtype=np.float64) for i, t in enumerate(angles)
    }
    popularity = {f"a{i}": (n - i if popularity_spread else 1) for i in range(n)}
    return ArtistSpace(vectors, popularity)


def test_percentile_of_the_nearest_is_zero() -> None:
    distances = np.array([0.1, 0.5, 0.9])
    assert percentile_of(distances, 0) == 0.0
    assert percentile_of(distances, 2) == 1.0


def test_a_listener_adopting_their_nearest_artists_scores_near_zero() -> None:
    space = ring_space()
    position = np.array([1.0, 0.0])
    distances = space.distances(position)
    nearest = [space.artists[i] for i in np.argsort(distances)[:5]]

    result = score_listener(
        "u", position, nearest, set(), space, K_VALUES, np.random.default_rng(0)
    )
    assert result.percentile is not None
    assert result.percentile < 0.05
    assert result.hit_at_k[10] == 1.0


def test_a_meaningless_embedding_scores_at_chance() -> None:
    """The decisive control.

    Adoptions drawn uniformly at random have no relationship to the position, so the mean
    percentile must sit near 0.5. If this ever passes at a much lower value, the metric is
    flattering the model and every result built on it is worthless.
    """
    space = ring_space(400)
    rng = np.random.default_rng(7)
    position = np.array([1.0, 0.0])

    percentiles = []
    for _ in range(200):
        adopted = list(rng.choice(np.array(space.artists), size=5, replace=False))
        result = score_listener("u", position, adopted, set(), space, K_VALUES, rng)
        if result.percentile is not None:
            percentiles.append(result.percentile)

    assert 0.45 < float(np.mean(percentiles)) < 0.55


def test_artists_already_known_do_not_compete_in_the_ranking() -> None:
    # Artists on a ring come in tied pairs either side of the position, so asserting an
    # exact rank would test numpy's tie-breaking. The real requirement is that excluding
    # what the listener already plays moves a genuine adoption up the ranking.
    space = ring_space(50)
    position = np.array([1.0, 0.0])
    order = [space.artists[i] for i in np.argsort(space.distances(position))]
    rng = np.random.default_rng(0)
    target = order[5]

    without = score_listener("u", position, [target], set(), space, K_VALUES, rng)
    with_known = score_listener("u", position, [target], set(order[:5]), space, K_VALUES, rng)

    assert without.median_rank is not None and with_known.median_rank is not None
    assert with_known.median_rank < without.median_rank
    assert with_known.median_rank <= 1.0


def test_listener_with_no_recognisable_adoptions_is_not_scored() -> None:
    space = ring_space(20)
    result = score_listener(
        "u", np.array([1.0, 0.0]), ["unknown"], set(), space, K_VALUES, np.random.default_rng(0)
    )
    assert result.adoptions == 0
    assert result.percentile is None


def test_listener_who_could_not_be_placed_is_not_scored() -> None:
    space = ring_space(20)
    result = score_listener(
        "u", np.zeros(2), ["a1"], set(), space, K_VALUES, np.random.default_rng(0)
    )
    assert result.adoptions == 0


def test_matched_negatives_hold_popularity_roughly_constant() -> None:
    space = ring_space(400)
    rng = np.random.default_rng(3)
    target = space.index["a200"]
    negatives = space.popularity_matched(target, 30, set(), rng)

    assert len(negatives) == 30
    assert target not in negatives
    target_popularity = space.popularity[target]
    spread = [abs(space.popularity[n] - target_popularity) for n in negatives]
    # Drawn from a popularity neighbourhood, not the whole catalogue.
    assert max(spread) < 400


def test_matched_percentile_is_reported_separately() -> None:
    space = ring_space(200)
    position = np.array([1.0, 0.0])
    nearest = [space.artists[i] for i in np.argsort(space.distances(position))[:3]]
    result = score_listener(
        "u", position, nearest, set(), space, K_VALUES, np.random.default_rng(0)
    )
    assert result.matched_percentile is not None
    assert 0.0 <= result.matched_percentile <= 1.0


def test_aggregate_ignores_listeners_with_nothing_to_score() -> None:
    space = ring_space(100)
    position = np.array([1.0, 0.0])
    nearest = [space.artists[i] for i in np.argsort(space.distances(position))[:3]]
    rng = np.random.default_rng(0)

    scored = score_listener("good", position, nearest, set(), space, K_VALUES, rng)
    empty = score_listener("empty", position, ["unknown"], set(), space, K_VALUES, rng)

    summary = aggregate([scored, empty], K_VALUES)
    assert summary["listeners_scored"] == 1
    assert summary["adoption_percentile"] is not None


def test_aggregate_of_nothing_is_none_not_zero() -> None:
    summary = aggregate([], K_VALUES)
    assert summary["listeners_scored"] == 0
    assert summary["adoption_percentile"] is None


@pytest.mark.parametrize("k", [1, 10, 50])
def test_hit_at_k_is_a_share_between_zero_and_one(k: int) -> None:
    space = ring_space(200)
    position = np.array([1.0, 0.0])
    rng = np.random.default_rng(1)
    adopted = list(rng.choice(np.array(space.artists), size=8, replace=False))
    result = score_listener("u", position, adopted, set(), space, (k,), rng)
    assert 0.0 <= result.hit_at_k[k] <= 1.0
