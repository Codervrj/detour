"""Time-based train/val/test split. Never random.

Train is months 1-9, val is month 10, test is months 11-12, counted from the first
listen in the cleaned data.

The user and item filters live here rather than in `clean` because they must be computed
from the train period only. Eligible users and items are derived from train, then applied
to all three splits, so no val or test row can influence who gets evaluated. Val and test
are also restricted to train users, since a user must appear in train to be evaluated;
cold-start users are reported separately, not mixed in.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from detour.config import load_config, write_sidecar

SPLIT_NAMES = ("train", "val", "test")


def add_months(moment: datetime, months: int) -> datetime:
    """Shift a datetime by whole calendar months, keeping day 1 semantics."""
    total = moment.month - 1 + months
    year = moment.year + total // 12
    month = total % 12 + 1
    return moment.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)


def boundaries(
    frame: pl.DataFrame, train_months: int, val_months: int
) -> tuple[datetime, datetime]:
    """Return the two cut points: train ends at the first, val ends at the second."""
    first = frame["listened_at"].min()
    assert isinstance(first, datetime)
    origin = first.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    train_end = add_months(origin, train_months)
    return train_end, add_months(train_end, val_months)


def eligible(
    train: pl.DataFrame,
    min_user_listens: int,
    min_item_listeners: int,
    unit: str = "item_id",
) -> tuple[set[str], set[str]]:
    """Users and items that clear the filters, computed from train only.

    `unit` is the column the item filter counts listeners on. It must match the unit of
    analysis: filtering rare *recordings* while building a map of *artists* throws away
    most of the catalogue, because a well-known artist can have many rarely-played tracks.
    """
    users = (
        train.group_by("user_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_user_listens)["user_id"]
    )
    items = (
        train.group_by(unit)
        .agg(pl.col("user_id").n_unique().alias("listeners"))
        .filter(pl.col("listeners") >= min_item_listeners)[unit]
    )
    return set(users.to_list()), set(items.to_list())


def apply_filters(
    frame: pl.DataFrame, users: set[str], items: set[str], unit: str = "item_id"
) -> pl.DataFrame:
    """Keep only rows whose user and item both cleared the train-derived filters."""
    return frame.filter(pl.col("user_id").is_in(list(users)) & pl.col(unit).is_in(list(items)))


def split_frames(
    frame: pl.DataFrame, train_months: int, val_months: int
) -> dict[str, pl.DataFrame]:
    """Cut the cleaned listens into train, val and test by time."""
    train_end, val_end = boundaries(frame, train_months, val_months)
    return {
        "train": frame.filter(pl.col("listened_at") < train_end),
        "val": frame.filter(
            (pl.col("listened_at") >= train_end) & (pl.col("listened_at") < val_end)
        ),
        "test": frame.filter(pl.col("listened_at") >= val_end),
    }


def stats_for(name: str, frame: pl.DataFrame, extra: dict[str, Any]) -> dict[str, Any]:
    """Sidecar stats for one split."""
    return {
        "stage": "split",
        "split": name,
        "rows": frame.height,
        "users": frame["user_id"].n_unique(),
        "items": frame["item_id"].n_unique(),
        "date_min": frame["listened_at"].min(),
        "date_max": frame["listened_at"].max(),
        **extra,
    }


def main(
    config_path: str = "evals/configs/smoke.yaml",
    source: str = "data/processed/listens.parquet",
) -> None:
    """Split the cleaned listens by time and write one Parquet file per split."""
    config = load_config(config_path)
    split_cfg = config["split"]
    filter_cfg = config["filters"]

    frame = pl.read_parquet(source)
    parts = split_frames(frame, split_cfg["train_months"], split_cfg["val_months"])

    unit = filter_cfg.get("unit", "item_id")
    users, items = eligible(
        parts["train"],
        filter_cfg["min_user_listens"],
        filter_cfg["min_item_listeners"],
        unit,
    )
    filter_stats = {
        "min_user_listens": filter_cfg["min_user_listens"],
        "min_item_listeners": filter_cfg["min_item_listeners"],
        "filter_unit": unit,
        "eligible_users": len(users),
        "eligible_items": len(items),
        "filters_computed_on": "train",
    }

    out_dir = Path(source).parent
    for name in SPLIT_NAMES:
        filtered = apply_filters(parts[name], users, items, unit)
        target = out_dir / f"{name}.parquet"
        filtered.write_parquet(target)
        write_sidecar(target, stats_for(name, filtered, filter_stats))
        print(f"{name:5s} {filtered.height:6d} rows  {filtered['user_id'].n_unique():3d} users")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-based train/val/test split.")
    parser.add_argument("--config", default="evals/configs/smoke.yaml")
    parser.add_argument("--source", default="data/processed/listens.parquet")
    args = parser.parse_args()
    main(args.config, args.source)
