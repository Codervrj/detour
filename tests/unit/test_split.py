"""Split and filter rules from CLAUDE.md section 5: time-based, train-only, no leakage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from detour.split import apply_filters, boundaries, eligible, split_frames

ORIGIN = datetime(2024, 1, 1, tzinfo=UTC)


def frame_from(rows: list[tuple[str, str, datetime]]) -> pl.DataFrame:
    """Build a minimal cleaned-listens frame from (user, item, timestamp) triples."""
    return pl.DataFrame(
        {
            "user_id": [r[0] for r in rows],
            "item_id": [r[1] for r in rows],
            "artist_mbid": [f"a-{r[1]}" for r in rows],
            "listened_at": [r[2] for r in rows],
        }
    )


def month(index: int, day: int = 15) -> datetime:
    """A timestamp inside month `index` (0-based) of 2024."""
    return datetime(2024, index + 1, day, 12, 0, tzinfo=UTC)


def twelve_month_frame() -> pl.DataFrame:
    """One user, one listen in the middle of each of twelve consecutive months."""
    return frame_from([("u1", f"i{m}", month(m)) for m in range(12)])


def test_split_is_nine_one_two_by_time() -> None:
    parts = split_frames(twelve_month_frame(), train_months=9, val_months=1)
    assert parts["train"].height == 9
    assert parts["val"].height == 1
    assert parts["test"].height == 2


def extreme(frame: pl.DataFrame, which: str) -> datetime:
    """Earliest or latest timestamp in a frame, narrowed for the type checker."""
    series = frame["listened_at"]
    value = series.max() if which == "max" else series.min()
    assert isinstance(value, datetime)
    return value


def test_no_leakage_across_split_boundaries() -> None:
    parts = split_frames(twelve_month_frame(), train_months=9, val_months=1)
    train_max = extreme(parts["train"], "max")
    val_min = extreme(parts["val"], "min")
    val_max = extreme(parts["val"], "max")
    test_min = extreme(parts["test"], "min")
    assert train_max < val_min <= val_max < test_min


def test_boundaries_land_on_month_starts() -> None:
    train_end, val_end = boundaries(twelve_month_frame(), train_months=9, val_months=1)
    assert train_end == datetime(2024, 10, 1, tzinfo=UTC)
    assert val_end == datetime(2024, 11, 1, tzinfo=UTC)


def test_boundaries_ignore_the_day_of_the_first_listen() -> None:
    # The first listen is late in January, but month 1 still starts on the 1st.
    frame = frame_from([("u1", "i0", datetime(2024, 1, 27, tzinfo=UTC))])
    train_end, _ = boundaries(frame, train_months=9, val_months=1)
    assert train_end == datetime(2024, 10, 1, tzinfo=UTC)


def test_item_filter_uses_train_counts_only() -> None:
    # "popular-later" has one listener in train but three in the test period.
    # With min_item_listeners=2 it must still be excluded, because only train counts.
    rows = [("u1", "popular-later", month(0))]
    rows += [(f"u{i}", "popular-later", month(10)) for i in range(2, 5)]
    rows += [(u, "steady", month(0)) for u in ("u1", "u2")]
    frame = frame_from(rows)

    parts = split_frames(frame, train_months=9, val_months=1)
    _, items = eligible(parts["train"], min_user_listens=1, min_item_listeners=2)

    assert "popular-later" not in items
    assert "steady" in items


def test_user_filter_uses_train_counts_only() -> None:
    # "late-bloomer" has one train listen and five in the test period.
    rows = [("late-bloomer", "i0", month(0))]
    rows += [("late-bloomer", f"i{i}", month(10)) for i in range(1, 6)]
    rows += [("regular", f"i{i}", month(0)) for i in range(3)]
    frame = frame_from(rows)

    parts = split_frames(frame, train_months=9, val_months=1)
    users, _ = eligible(parts["train"], min_user_listens=3, min_item_listeners=1)

    assert "late-bloomer" not in users
    assert "regular" in users


def test_apply_filters_drops_rows_failing_either_filter() -> None:
    frame = frame_from(
        [("keep", "keep", month(0)), ("keep", "drop", month(0)), ("drop", "keep", month(0))]
    )
    filtered = apply_filters(frame, users={"keep"}, items={"keep"})
    assert filtered.height == 1
    assert filtered["user_id"].to_list() == ["keep"]
    assert filtered["item_id"].to_list() == ["keep"]


def test_split_handles_a_user_with_no_val_listens() -> None:
    frame = frame_from([("u1", "i0", month(0)), ("u1", "i1", month(10))])
    parts = split_frames(frame, train_months=9, val_months=1)
    assert parts["val"].height == 0
    assert parts["test"].height == 1


def test_dedupe_window_boundary_does_not_shift_the_split() -> None:
    # A listen one second before the train cut stays in train.
    just_before = datetime(2024, 10, 1, tzinfo=UTC) - timedelta(seconds=1)
    frame = frame_from([("u1", "i0", month(0)), ("u1", "i1", just_before)])
    parts = split_frames(frame, train_months=9, val_months=1)
    assert parts["train"].height == 2
    assert parts["val"].height == 0
