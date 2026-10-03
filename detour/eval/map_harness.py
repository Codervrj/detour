"""Evaluate the map: do artists near a listener's territory get adopted?

Every model is placed in the same position: trained on months 1-9, asked about artists each
listener first played in the eval period, and scored on where those real adoptions landed
in a ranking of the whole catalogue.

Popularity is the model to watch. It is the confound in person: if item2vec cannot beat a
ranking that ignores the listener entirely, then the map has learned popularity and nothing
about taste, and the report must say so.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from detour.config import load_config
from detour.eval import report as report_writer
from detour.eval.adoption import ArtistSpace, aggregate, score_listener
from detour.foldin import place
from detour.models.baselines import (
    als_vectors,
    artist_plays,
    ppmi_svd_vectors,
    random_vectors,
)

MODEL_ORDER = ["random", "popularity", "ppmi_svd", "als", "item2vec"]


def listener_artists(frame: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Play counts per artist, per listener."""
    counted = frame.group_by(["user_id", "artist_mbid"]).agg(pl.len().alias("plays"))
    out: dict[str, dict[str, int]] = {}
    for row in counted.iter_rows(named=True):
        out.setdefault(str(row["user_id"]), {})[str(row["artist_mbid"])] = int(row["plays"])
    return out


def adoptions(
    train_artists: dict[str, dict[str, int]], eval_artists: dict[str, dict[str, int]]
) -> dict[str, list[str]]:
    """Artists each listener played in the eval period and never before.

    This is the ground truth: a real discovery, made by a real person, after the model's
    training data ended.
    """
    out: dict[str, list[str]] = {}
    for user, played in eval_artists.items():
        known = train_artists.get(user, {})
        new = [artist for artist in played if artist not in known]
        if new:
            out[user] = new
    return out


def popularity_vectors(plays: dict[str, int], artists: Sequence[str]) -> dict[str, np.ndarray]:
    """A one-dimensional space where distance is purely global popularity.

    Every listener sits at the same point, so the ranking ignores who is asking. That is
    exactly the baseline we need: it is what a model looks like when it has learned the
    charts and nothing else.
    """
    counts = np.array([float(plays.get(a, 0)) for a in artists])
    scaled = counts / (counts.max() or 1.0)
    # A single positive axis: the more popular, the closer to the shared position.
    return {a: np.array([v, 0.0]) for a, v in zip(artists, scaled, strict=True)}


def build_spaces(
    train: pl.DataFrame, config: dict[str, Any], seed: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """Artist vectors for every model, plus the play counts used by the control."""
    plays = artist_plays(train)
    vectors = pl.read_parquet("artifacts/artist_vectors.parquet")
    item2vec = {
        str(row["artist_mbid"]): np.asarray(row["vector"], dtype=np.float64)
        for row in vectors.iter_rows(named=True)
    }
    # Every model is restricted to the same vocabulary, so the comparison is like for like.
    vocabulary = sorted(item2vec)
    dimensions = config["item2vec"]["dimensions"]

    print("building baselines")
    spaces: dict[str, dict[str, np.ndarray]] = {
        "item2vec": item2vec,
        "random": random_vectors(vocabulary, dimensions, seed),
        "popularity": popularity_vectors(plays, vocabulary),
    }

    restricted = train.filter(pl.col("artist_mbid").is_in(vocabulary))
    for name, builder in (
        ("ppmi_svd", lambda: ppmi_svd_vectors(restricted, dimensions)),
        ("als", lambda: als_vectors(restricted, dimensions, seed)),
    ):
        print(f"  {name}")
        built = builder()  # type: ignore[no-untyped-call]
        spaces[name] = {a: v for a, v in built.items() if a in item2vec}

    return spaces, plays


def evaluate_model(
    name: str,
    vectors: dict[str, np.ndarray],
    train_artists: dict[str, dict[str, int]],
    targets: dict[str, list[str]],
    plays: dict[str, int],
    k_values: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    """Score one model over every evaluable listener."""
    space = ArtistSpace(vectors, plays)
    rng = np.random.default_rng(seed)

    results = []
    for user, adopted in targets.items():
        history = train_artists.get(user, {})
        if not history:
            continue
        placement = place(history, vectors)
        results.append(
            score_listener(
                user,
                placement.position,
                adopted,
                set(history),
                space,
                k_values,
                rng,
            )
        )
    summary = aggregate(results, k_values)
    print(
        f"  {name:<12} percentile {summary['adoption_percentile']:.4f}"
        if summary["adoption_percentile"] is not None
        else f"  {name:<12} not scorable"
    )
    return summary


def run(config_path: str, split: str) -> dict[str, Any]:
    """Evaluate every model on the held-out adoption question."""
    config = load_config(config_path)
    seed = config["eval"]["seeds"][0]
    k_values = config["eval"]["k_values"]
    processed = Path(config["data"]["processed_dir"])

    train = pl.read_parquet(processed / "train.parquet")
    evaluation = pl.read_parquet(processed / f"{split}.parquet")

    train_artists = listener_artists(train)
    eval_artists = listener_artists(evaluation)
    targets = adoptions(train_artists, eval_artists)
    print(f"{len(targets):,} listeners adopted a new artist in {split}")

    spaces, plays = build_spaces(train, config, seed)

    print("\nscoring")
    models = {
        name: evaluate_model(name, spaces[name], train_artists, targets, plays, k_values, seed)
        for name in MODEL_ORDER
        if name in spaces
    }

    return {
        "split": split,
        "models": models,
        "listeners_with_adoptions": len(targets),
        "vocabulary": len(spaces["item2vec"]),
        "k_values": list(k_values),
    }


def main(config_path: str = "evals/configs/main.yaml", split: str = "val") -> None:
    """Run the adoption eval and write a report."""
    results = run(config_path, split)
    path = report_writer.write_map_report(results, load_config(config_path), config_path)
    print(f"\nreport: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the artist map.")
    parser.add_argument("--config", default="evals/configs/main.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()
    main(args.config, args.split)
