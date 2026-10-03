"""Explorer score and popularity, including the section 8 edge cases."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from detour.features.explorer_score import explorer_scores, monthly_new_artist_share
from detour.features.popularity import popularity_table

CLIP = (0.0, 1.0)


def listen(user: str, artist: str, month: int, day: int = 10) -> dict[str, object]:
    """One listen by `user` to `artist` in month `month` of 2024."""
    return {
        "user_id": user,
        "artist_mbid": artist,
        "item_id": f"{artist}-track",
        "listened_at": datetime(2024, month, day, 12, 0, tzinfo=UTC),
    }


def frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_single_artist_replayer_scores_zero() -> None:
    # One artist, nine months. After the first month nothing is ever new.
    rows = [listen("loyalist", "a1", m) for m in range(1, 10)]
    scores = explorer_scores(frame(rows), window=3, clip=CLIP)
    assert scores["explorer_score"].to_list() == [0.0]


def test_new_artist_every_month_scores_high() -> None:
    # Every month brings exactly one listen to an artist never heard before, so every
    # counted month has a new-artist share of 1.0.
    rows = [listen("explorer", f"a{m}", m) for m in range(1, 10)]
    scores = explorer_scores(frame(rows), window=3, clip=CLIP)
    assert scores["explorer_score"].to_list() == [1.0]


def test_score_is_always_within_zero_and_one() -> None:
    rows = [listen("u", f"a{m % 3}", m) for m in range(1, 10)]
    rows += [listen("v", "a1", m) for m in range(1, 10)]
    scores = explorer_scores(frame(rows), window=3, clip=CLIP)
    for value in scores["explorer_score"].to_list():
        assert 0.0 <= value <= 1.0


def test_first_active_month_is_excluded() -> None:
    # In month 1 every artist is new. Only month 2 is counted, and it is all repeats,
    # so the score is 0 rather than 0.5.
    rows = [listen("u", "a1", 1), listen("u", "a2", 1), listen("u", "a1", 2)]
    scores = explorer_scores(frame(rows), window=1, clip=CLIP)
    assert scores["explorer_score"].to_list() == [0.0]


def test_user_with_one_active_month_scores_zero() -> None:
    rows = [listen("u", "a1", 1), listen("u", "a2", 1)]
    scores = explorer_scores(frame(rows), window=3, clip=CLIP)
    assert scores["explorer_score"].to_list() == [0.0]
    assert scores["months_counted"].to_list() == [0]


def test_every_train_user_appears_in_the_output() -> None:
    rows = [listen("quiet", "a1", 1)]
    rows += [listen("busy", f"a{m}", m) for m in range(1, 5)]
    scores = explorer_scores(frame(rows), window=3, clip=CLIP)
    assert set(scores["user_id"].to_list()) == {"quiet", "busy"}


def test_monthly_share_is_hand_checkable() -> None:
    # Month 2: three listens, two of them to a3 which is new that month -> 2/3.
    rows = [listen("u", "a1", 1), listen("u", "a2", 1)]
    rows += [listen("u", "a1", 2), listen("u", "a3", 2), listen("u", "a3", 2, day=11)]
    monthly = monthly_new_artist_share(frame(rows)).sort("month")
    shares = monthly["share"].to_list()
    assert shares[0] == 1.0  # month 1: both artists new
    assert abs(shares[1] - 2 / 3) < 1e-12


def test_popularity_p_is_listeners_over_train_users() -> None:
    # Four users. a1 heard by 3 of them -> p = 0.75; a2 by 1 -> p = 0.25.
    rows = [listen(f"u{i}", "a1", 1) for i in range(3)]
    rows += [listen("u3", "a2", 1)]
    table = popularity_table(frame(rows)).sort("item_id")
    assert table["p"].to_list() == [0.75, 0.25]
    assert table["listeners"].to_list() == [3, 1]


def test_popularity_counts_distinct_listeners_not_plays() -> None:
    # One user playing the same item three times is still one listener.
    rows = [listen("u0", "a1", 1, day=d) for d in (1, 2, 3)]
    table = popularity_table(frame(rows))
    assert table["listeners"].to_list() == [1]
    assert table["plays"].to_list() == [3]
    assert table["p"].to_list() == [1.0]
