"""Re-ranking rules and properties from CLAUDE.md section 8."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from detour.rerank.mmr import Candidate, minmax, novelty_from_popularity, prepare, rerank
from detour.rerank.personalized_lambda import lambda_for

LAMBDA_CONFIG = {"lam_min": 0.1, "lam_max": 0.8, "gamma": 1.0}

# The monotonicity and ordering properties below hold for the pure blend, so they are
# checked with beta = 0 and no anchor floor. The similarity penalty is independent of
# lambda in the scoring formula, and the anchor floor deliberately overrides the blend.


@st.composite
def candidate_lists(draw: st.DrawFn) -> list[Candidate]:
    """A list of distinct candidates with relevances, popularities and anchor flags."""
    size = draw(st.integers(min_value=4, max_value=18))
    floats = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    relevances = draw(st.lists(floats, min_size=size, max_size=size))
    popularities = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )
    anchors = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    ids = [f"item-{i}" for i in range(size)]
    return prepare(ids, relevances, popularities, anchors)


def simple(n: int = 10, anchors: int = 0) -> list[Candidate]:
    """Candidates with strictly decreasing relevance and strictly increasing novelty."""
    ids = [f"item-{i}" for i in range(n)]
    relevances = [float(n - i) for i in range(n)]
    popularities = [1.0 / (i + 1) for i in range(n)]
    flags = [i < anchors for i in range(n)]
    return prepare(ids, relevances, popularities, flags)


def test_lambda_zero_matches_relevance_order() -> None:
    candidates = simple(10)
    picked = rerank(candidates, lam=0.0, k=5, beta=0.0, anchor_min=0)
    expected = [c.item_id for c in sorted(candidates, key=lambda c: -c.relevance)][:5]
    assert [c.item_id for c in picked] == expected


def test_lambda_one_matches_novelty_order() -> None:
    candidates = simple(10)
    picked = rerank(candidates, lam=1.0, k=5, beta=0.0, anchor_min=0)
    expected = [c.item_id for c in sorted(candidates, key=lambda c: -c.novelty)][:5]
    assert [c.item_id for c in picked] == expected


def test_duplicate_items_raise() -> None:
    candidates = simple(5)
    candidates.append(candidates[0])
    with pytest.raises(ValueError, match="duplicate"):
        rerank(candidates, lam=0.5, k=3, beta=0.0, anchor_min=0)


def test_k_larger_than_candidate_list_returns_everything() -> None:
    candidates = simple(4)
    picked = rerank(candidates, lam=0.5, k=10, beta=0.0, anchor_min=0)
    assert len(picked) == 4


@given(candidate_lists())
@settings(max_examples=50, deadline=None)
def test_output_has_no_duplicates_and_respects_k(candidates: list[Candidate]) -> None:
    k = min(5, len(candidates))
    picked = rerank(candidates, lam=0.5, k=k, beta=0.0, anchor_min=0)
    ids = [c.item_id for c in picked]
    assert len(ids) == k
    assert len(set(ids)) == k


@given(candidate_lists())
@settings(max_examples=50, deadline=None)
def test_higher_lambda_never_lowers_mean_novelty(candidates: list[Candidate]) -> None:
    k = min(5, len(candidates))
    previous = -1.0
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        picked = rerank(candidates, lam=lam, k=k, beta=0.0, anchor_min=0)
        mean_novelty = sum(c.novelty for c in picked) / len(picked)
        assert mean_novelty >= previous - 1e-9
        previous = mean_novelty


@given(candidate_lists())
@settings(max_examples=50, deadline=None)
def test_anchor_floor_is_respected_when_anchors_exist(candidates: list[Candidate]) -> None:
    k = min(6, len(candidates))
    available = sum(1 for c in candidates if c.is_anchor)
    anchor_min = min(2, available, k)
    picked = rerank(candidates, lam=1.0, beta=0.0, k=k, anchor_min=anchor_min)
    assert sum(1 for c in picked if c.is_anchor) >= anchor_min


def test_anchor_floor_pulls_known_artists_into_a_novelty_heavy_list() -> None:
    # Anchors are the least novel items here, so a lam=1 list would hold none of them.
    candidates = simple(10, anchors=3)
    without = rerank(candidates, lam=1.0, beta=0.0, k=5, anchor_min=0)
    with_floor = rerank(candidates, lam=1.0, beta=0.0, k=5, anchor_min=2)
    assert sum(1 for c in without if c.is_anchor) == 0
    assert sum(1 for c in with_floor if c.is_anchor) == 2
    assert len(with_floor) == 5
    assert len({c.item_id for c in with_floor}) == 5


def test_similarity_penalty_pushes_apart_near_identical_items() -> None:
    # Two items share a direction; a third is orthogonal. With a penalty the list should
    # prefer the orthogonal item over the twin.
    twin_a = Candidate(
        "a", relevance=1.0, novelty=0.0, is_anchor=False, factors=np.array([1.0, 0.0])
    )
    twin_b = Candidate(
        "b", relevance=0.9, novelty=0.0, is_anchor=False, factors=np.array([1.0, 0.0])
    )
    other = Candidate(
        "c", relevance=0.8, novelty=0.0, is_anchor=False, factors=np.array([0.0, 1.0])
    )

    without = rerank([twin_a, twin_b, other], lam=0.0, beta=0.0, k=2, anchor_min=0)
    with_penalty = rerank([twin_a, twin_b, other], lam=0.0, beta=0.5, k=2, anchor_min=0)
    assert [c.item_id for c in without] == ["a", "b"]
    assert [c.item_id for c in with_penalty] == ["a", "c"]


def test_novelty_is_higher_for_rarer_items() -> None:
    # p = 0.5 -> 1 bit; p = 0.25 -> 2 bits.
    assert novelty_from_popularity(0.5) == pytest.approx(1.0)
    assert novelty_from_popularity(0.25) == pytest.approx(2.0)
    assert novelty_from_popularity(0.01) > novelty_from_popularity(0.9)


def test_minmax_handles_a_flat_list() -> None:
    assert minmax([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]
    assert minmax([]) == []
    assert minmax([1.0, 3.0, 2.0]) == [0.0, 1.0, 0.5]


def test_personalised_lambda_endpoints_and_range() -> None:
    assert lambda_for(0.0, LAMBDA_CONFIG) == pytest.approx(0.1)
    assert lambda_for(1.0, LAMBDA_CONFIG) == pytest.approx(0.8)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_personalised_lambda_stays_in_configured_range(explorer_score: float) -> None:
    value = lambda_for(explorer_score, LAMBDA_CONFIG)
    assert 0.1 <= value <= 0.8


@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_personalised_lambda_is_monotonic(first: float, second: float) -> None:
    low, high = sorted((first, second))
    assert lambda_for(low, LAMBDA_CONFIG) <= lambda_for(high, LAMBDA_CONFIG) + 1e-12


def test_personalised_lambda_clamps_out_of_range_scores() -> None:
    assert lambda_for(-0.5, LAMBDA_CONFIG) == pytest.approx(0.1)
    assert lambda_for(1.5, LAMBDA_CONFIG) == pytest.approx(0.8)
