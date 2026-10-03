"""Golden metric cases from CLAUDE.md section 8.

Every case carries the arithmetic in a comment, so a failure tells you whether the code
drifted or the definition did.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from detour.eval.metrics import (
    catalogue_coverage,
    check_no_duplicates,
    discovery_recall_at_k,
    exposure_gini,
    familiarity_anchor_rate,
    ild_at_k,
    mean_ignoring_none,
    ndcg_at_k,
    novelty_at_k,
    recall_at_k,
    serendipity_at_k,
)

# --- recall ----------------------------------------------------------------------------


def test_recall_hits_two_of_four_relevant() -> None:
    # recs a,b,c,d,e; relevant {a,c,x,y}; hits = a,c = 2
    # denominator = min(5, 4) = 4  ->  2/4 = 0.5
    recs = ["a", "b", "c", "d", "e"]
    assert recall_at_k(recs, {"a", "c", "x", "y"}, 5) == 0.5


def test_recall_denominator_is_truncated_by_k() -> None:
    # relevant has 10 items but k = 2, so denominator = min(2, 10) = 2
    # recs[:2] = a,b, both relevant -> 2/2 = 1.0
    recs = ["a", "b", "c"]
    relevant = {f"r{i}" for i in range(8)} | {"a", "b"}
    assert recall_at_k(recs, relevant, 2) == 1.0


def test_recall_with_empty_relevant_set_is_none() -> None:
    assert recall_at_k(["a", "b"], set(), 2) is None


def test_recall_with_k_larger_than_list() -> None:
    # recs a,b; relevant {a}; hits = 1; denominator = min(10, 1) = 1 -> 1.0
    assert recall_at_k(["a", "b"], {"a"}, 10) == 1.0


# --- ndcg ------------------------------------------------------------------------------


def test_ndcg_all_relevant_is_one() -> None:
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)


def test_ndcg_single_hit_at_second_position() -> None:
    # gains 0,1,0 -> dcg = 1/log2(3) = 0.63093
    # relevant has 1 item -> idcg = 1/log2(2) = 1.0
    # ndcg = 0.63093
    expected = 1.0 / math.log2(3)
    assert ndcg_at_k(["x", "a", "y"], {"a"}, 3) == pytest.approx(expected)


def test_ndcg_rewards_earlier_hits() -> None:
    early = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
    late = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
    assert early is not None and late is not None
    assert early > late


def test_ndcg_two_hits_hand_computed() -> None:
    # recs a,x,b ; relevant {a,b}
    # dcg = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    # idcg = 1/log2(2) + 1/log2(3) = 1.0 + 0.63093 = 1.63093
    # ndcg = 1.5 / 1.63093 = 0.91972
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(expected)


def test_ndcg_with_empty_relevant_set_is_none() -> None:
    assert ndcg_at_k(["a"], set(), 1) is None


# --- discovery recall ------------------------------------------------------------------


def test_discovery_recall_is_artist_level() -> None:
    # rec artists A1,A1,A2; discovery {A1,A2,A3}
    # distinct found = {A1,A2} = 2; denominator = min(3, 3) = 3 -> 2/3
    artists = ["A1", "A1", "A2"]
    assert discovery_recall_at_k(artists, {"A1", "A2", "A3"}, 3) == pytest.approx(2 / 3)


def test_discovery_recall_with_empty_discovery_set_is_none() -> None:
    # A loyalist discovered nothing, so the question does not apply.
    assert discovery_recall_at_k(["A1"], set(), 10) is None


# --- novelty ---------------------------------------------------------------------------


def test_novelty_is_mean_self_information() -> None:
    # p = 0.5 -> 1 bit; p = 0.25 -> 2 bits; mean = 1.5
    popularity = {"a": 0.5, "b": 0.25}
    assert novelty_at_k(["a", "b"], popularity, 2) == pytest.approx(1.5)


def test_novelty_treats_unknown_items_as_very_rare() -> None:
    value = novelty_at_k(["unseen"], {}, 1)
    assert value is not None and value > 15.0


# --- ild -------------------------------------------------------------------------------


def test_ild_of_orthogonal_items_is_one() -> None:
    # cosine 0 -> distance 1
    factors = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    assert ild_at_k(["a", "b"], factors, 2) == pytest.approx(1.0)


def test_ild_of_identical_items_is_zero() -> None:
    factors = {"a": np.array([1.0, 0.0]), "b": np.array([2.0, 0.0])}
    assert ild_at_k(["a", "b"], factors, 2) == pytest.approx(0.0)


def test_ild_needs_at_least_two_items() -> None:
    assert ild_at_k(["a"], {"a": np.array([1.0, 0.0])}, 2) is None


# --- serendipity -----------------------------------------------------------------------


def test_serendipity_counts_relevant_but_unpopular() -> None:
    # recs a,b,c,d; relevant {a,b,c}; popular top {a}
    # relevant and not popular = b,c = 2; over list length 4 -> 0.5
    recs = ["a", "b", "c", "d"]
    assert serendipity_at_k(recs, {"a", "b", "c"}, {"a"}, 4) == 0.5


# --- familiarity anchors ---------------------------------------------------------------


def test_familiarity_anchor_rate() -> None:
    # 2 of 4 rec artists are already known -> 0.5
    assert familiarity_anchor_rate(["A1", "A2", "A3", "A4"], {"A1", "A3"}, 4) == 0.5


# --- coverage and gini -----------------------------------------------------------------


def test_catalogue_coverage() -> None:
    # distinct recommended = {a,b,c} = 3 of a catalogue of 10 -> 0.3
    assert catalogue_coverage([["a", "b"], ["b", "c"]], 10) == pytest.approx(0.3)


def test_gini_is_zero_when_exposure_is_perfectly_even() -> None:
    catalogue = ["a", "b", "c"]
    assert exposure_gini([["a"], ["b"], ["c"]], catalogue) == pytest.approx(0.0, abs=1e-9)


def test_gini_is_high_when_one_item_takes_everything() -> None:
    catalogue = [f"i{n}" for n in range(10)]
    value = exposure_gini([["i0"]] * 10, catalogue)
    assert value > 0.85


def test_gini_of_empty_catalogue_is_zero() -> None:
    assert exposure_gini([["a"]], []) == 0.0


# --- shared rules ----------------------------------------------------------------------


def test_duplicate_recommendations_raise() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        check_no_duplicates(["a", "b", "a"])


def test_every_metric_rejects_duplicate_lists() -> None:
    recs = ["a", "a"]
    calls: list[Callable[[], float | None]] = [
        lambda: recall_at_k(recs, {"a"}, 2),
        lambda: ndcg_at_k(recs, {"a"}, 2),
        lambda: novelty_at_k(recs, {"a": 0.5}, 2),
        lambda: serendipity_at_k(recs, {"a"}, set(), 2),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="duplicates"):
            call()


def test_mean_ignores_users_the_metric_does_not_apply_to() -> None:
    assert mean_ignoring_none([1.0, None, 0.0]) == pytest.approx(0.5)
    assert mean_ignoring_none([None, None]) is None
    assert mean_ignoring_none([]) is None
