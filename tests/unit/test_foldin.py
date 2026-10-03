"""Fold-in properties: placing a listener the model has never seen."""

from __future__ import annotations

import numpy as np
import pytest

from detour.foldin import Placement, edge_artists, nearest, place, unit

A = np.array([1.0, 0.0, 0.0])
B = np.array([0.0, 1.0, 0.0])
C = np.array([0.0, 0.0, 1.0])
VECTORS = {"a": A, "b": B, "c": C}


def test_single_artist_listener_lands_on_that_artist() -> None:
    placement = place({"a": 50}, VECTORS)
    assert np.allclose(placement.position, unit(A))


def test_position_moves_toward_an_artist_as_plays_grow() -> None:
    light = place({"a": 100, "b": 1}, VECTORS)
    heavy = place({"a": 100, "b": 90}, VECTORS)
    # More plays of b should pull the position closer to b.
    assert float(heavy.position @ unit(B)) > float(light.position @ unit(B))


def test_position_is_unit_length() -> None:
    placement = place({"a": 10, "b": 4, "c": 1}, VECTORS)
    assert float(np.linalg.norm(placement.position)) == pytest.approx(1.0)


def test_one_heavy_rotation_does_not_drown_everything_else() -> None:
    # 4,000 plays of one band against 100 each of two others. With raw-count weighting the
    # position would sit almost exactly on "a"; log weighting must keep the others visible.
    placement = place({"a": 4000, "b": 100, "c": 100}, VECTORS)
    similarity_to_a = float(placement.position @ unit(A))
    assert similarity_to_a < 0.9


def test_unknown_artists_are_counted_not_silently_dropped() -> None:
    placement = place({"a": 10, "mystery": 5, "other": 1}, VECTORS)
    assert placement.matched_artists == 1
    assert placement.unmatched_artists == 2
    assert placement.artist_coverage == pytest.approx(1 / 3)


def test_play_coverage_reflects_plays_not_artists() -> None:
    # One known artist with most of the plays: artist coverage is low, play coverage high.
    placement = place({"a": 900, "mystery": 50, "other": 50}, VECTORS)
    assert placement.artist_coverage == pytest.approx(1 / 3)
    assert placement.play_coverage == pytest.approx(0.9)


def test_listener_with_nothing_recognisable_is_handled() -> None:
    placement = place({"mystery": 10}, VECTORS)
    assert placement.matched_artists == 0
    assert placement.artist_coverage == 0.0
    assert placement.position.shape == (3,)
    assert not placement.position.any()


def test_empty_history_does_not_raise() -> None:
    placement = place({}, VECTORS)
    assert placement.artist_coverage == 0.0
    assert placement.play_coverage == 0.0


def test_nearest_can_exclude_artists_already_known() -> None:
    position = unit(A)
    assert nearest(position, VECTORS, k=1)[0][0] == "a"
    assert nearest(position, VECTORS, k=1, exclude={"a"})[0][0] != "a"


def test_nearest_returns_at_most_k() -> None:
    assert len(nearest(unit(A), VECTORS, k=2)) == 2
    assert len(nearest(unit(A), VECTORS, k=99)) == 3


def test_edge_artists_are_the_ones_furthest_from_the_centre() -> None:
    # Heavy on a and b, one play of c: c is the outlier at the edge of this taste.
    placement = place({"a": 500, "b": 500, "c": 1}, VECTORS)
    edges = edge_artists(placement, k=1)
    assert edges[0][0] == "c"


def test_edge_artists_of_an_unreadable_listener_is_empty() -> None:
    placement = Placement(position=np.zeros(3), artist_vectors={}, weights={})
    assert edge_artists(placement) == []
