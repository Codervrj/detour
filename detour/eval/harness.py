"""Run every baseline and our model over the eval split and collect metrics.

Baselines, all required in every report: random, popularity, item-kNN, ALS,
ALS + MMR with a fixed global lambda, and ALS + personalised lambda (ours).

Ground truth, per CLAUDE.md section 7:
  relevant set  = items the user listened to in the eval period
  discovery set = artists in the eval period that never appear in their train history

Only users present in train are evaluated. Cold-ish users (under 100 train listens) are
reported as their own segment rather than being mixed into the headline numbers.

Report whatever the numbers are. No seed picking, no segment picking, no K picking.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from detour.config import load_config
from detour.eval import report as report_writer
from detour.eval.metrics import (
    catalogue_coverage,
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
from detour.recommend import Recommender

COLD_LISTEN_THRESHOLD = 100
POPULAR_TOP_N = 50
BASELINE_FILES = {
    "random": "random_candidates.parquet",
    "popularity": "popularity_candidates.parquet",
    "itemknn": "itemknn_candidates.parquet",
    "als": "als_candidates.parquet",
}


@dataclass
class UserTruth:
    """Everything needed to score one user."""

    user_id: str
    relevant_items: set[str]
    discovery_artists: set[str]
    known_artists: set[str]
    train_listens: int
    explorer_score: float
    segment: str
    is_cold: bool


def anchor_floor(anchor_min: int, k: int) -> int:
    """Scale the anchor floor, which section 6 defines for lists of 20, to this k."""
    return max(1, round(anchor_min * k / 20)) if anchor_min > 0 else 0


def terciles(scores: list[float]) -> tuple[float, float]:
    """The two explorer-score cut points that split users into three segments."""
    array = np.asarray(scores, dtype=np.float64)
    return float(np.quantile(array, 1 / 3)), float(np.quantile(array, 2 / 3))


def segment_for(score: float, cuts: tuple[float, float]) -> str:
    """Name the explorer tercile a score falls in."""
    if score <= cuts[0]:
        return "loyalists"
    if score <= cuts[1]:
        return "middle"
    return "explorers"


def build_truth(
    train: pl.DataFrame, evaluation: pl.DataFrame, explorer: pl.DataFrame
) -> list[UserTruth]:
    """Assemble per-user ground truth for every train user with eval activity."""
    train_items = _grouped_sets(train, "item_id")
    train_artists = _grouped_sets(train, "artist_mbid")
    eval_items = _grouped_sets(evaluation, "item_id")
    eval_artists = _grouped_sets(evaluation, "artist_mbid")
    listen_counts = dict(train.group_by("user_id").agg(pl.len().alias("n")).iter_rows())
    scores = dict(zip(explorer["user_id"], explorer["explorer_score"], strict=True))

    evaluable = sorted(set(train_items) & set(eval_items))
    cuts = terciles([float(scores.get(u, 0.0)) for u in evaluable])

    truths: list[UserTruth] = []
    for user_id in evaluable:
        score = float(scores.get(user_id, 0.0))
        listens = int(listen_counts.get(user_id, 0))
        truths.append(
            UserTruth(
                user_id=user_id,
                relevant_items=eval_items[user_id],
                discovery_artists=eval_artists[user_id] - train_artists.get(user_id, set()),
                known_artists=train_artists.get(user_id, set()),
                train_listens=listens,
                explorer_score=score,
                segment=segment_for(score, cuts),
                is_cold=listens < COLD_LISTEN_THRESHOLD,
            )
        )
    return truths


def _grouped_sets(frame: pl.DataFrame, column: str) -> dict[str, set[str]]:
    """user_id -> set of values in `column`."""
    grouped = frame.group_by("user_id").agg(pl.col(column).unique().alias("values"))
    return {row["user_id"]: set(row["values"]) for row in grouped.iter_rows(named=True)}


@dataclass
class Context:
    """Lookups shared by every metric computation."""

    popularity: dict[str, float]
    factors: dict[str, np.ndarray]
    item_artist: dict[str, str]
    popular_top: set[str]
    catalogue: list[str]


def build_context(recommender: Recommender, train: pl.DataFrame) -> Context:
    """Collect the tables the metrics need, all derived from train."""
    popularity = {
        row["item_id"]: float(row["p"]) for row in recommender.popularity.iter_rows(named=True)
    }
    top = (
        recommender.popularity.sort("listeners", descending=True)
        .head(POPULAR_TOP_N)["item_id"]
        .to_list()
    )
    return Context(
        popularity=popularity,
        factors=recommender._factors,
        item_artist={
            item: str(meta.get("artist_mbid", "")) for item, meta in recommender.catalogue.items()
        },
        popular_top=set(top),
        catalogue=sorted(train["item_id"].unique().to_list()),
    )


def score_lists(
    lists: dict[str, list[str]], truths: Sequence[UserTruth], context: Context, k: int
) -> dict[str, Any]:
    """Compute every metric at k, averaged over the users each metric applies to."""
    per_user: dict[str, list[float | None]] = {
        f"recall@{k}": [],
        f"ndcg@{k}": [],
        f"discovery_recall@{k}": [],
        f"novelty@{k}": [],
        f"ild@{k}": [],
        f"serendipity@{k}": [],
        f"familiarity_anchor_rate@{k}": [],
    }
    all_recs: list[list[str]] = []

    for truth in truths:
        recs = lists.get(truth.user_id, [])[:k]
        all_recs.append(recs)
        artists = [context.item_artist.get(item, "") for item in recs]

        per_user[f"recall@{k}"].append(recall_at_k(recs, truth.relevant_items, k))
        per_user[f"ndcg@{k}"].append(ndcg_at_k(recs, truth.relevant_items, k))
        per_user[f"discovery_recall@{k}"].append(
            discovery_recall_at_k(artists, truth.discovery_artists, k)
        )
        per_user[f"novelty@{k}"].append(novelty_at_k(recs, context.popularity, k))
        per_user[f"ild@{k}"].append(ild_at_k(recs, context.factors, k))
        per_user[f"serendipity@{k}"].append(
            serendipity_at_k(recs, truth.relevant_items, context.popular_top, k)
        )
        per_user[f"familiarity_anchor_rate@{k}"].append(
            familiarity_anchor_rate(artists, truth.known_artists, k)
        )

    metrics: dict[str, Any] = {
        name: mean_ignoring_none(values) for name, values in per_user.items()
    }
    metrics[f"catalogue_coverage@{k}"] = catalogue_coverage(all_recs, len(context.catalogue))
    metrics[f"exposure_gini@{k}"] = exposure_gini(all_recs, context.catalogue)
    metrics["_per_user"] = per_user
    return metrics


def segment_breakdown(
    lists: dict[str, list[str]], truths: Sequence[UserTruth], context: Context, k: int
) -> dict[str, dict[str, Any]]:
    """The same metrics, split by explorer tercile and for cold-ish users."""
    groups: dict[str, list[UserTruth]] = {
        "loyalists": [],
        "middle": [],
        "explorers": [],
        "cold": [],
    }
    for truth in truths:
        groups[truth.segment].append(truth)
        if truth.is_cold:
            groups["cold"].append(truth)

    out: dict[str, dict[str, Any]] = {}
    for name, members in groups.items():
        if not members:
            out[name] = {"users": 0}
            continue
        scored = score_lists(lists, members, context, k)
        scored.pop("_per_user", None)
        scored["users"] = len(members)
        out[name] = scored
    return out


def baseline_lists(path: Path, users: Sequence[str], k_max: int) -> dict[str, list[str]]:
    """Read a candidate Parquet into per-user ranked item lists."""
    frame = pl.read_parquet(path).sort(["user_id", "rank"])
    wanted = set(users)
    grouped = (
        frame.filter(pl.col("user_id").is_in(list(wanted)))
        .group_by("user_id")
        .agg(pl.col("item_id").alias("items"))
    )
    return {row["user_id"]: list(row["items"])[:k_max] for row in grouped.iter_rows(named=True)}


def reranked_lists(
    recommender: Recommender,
    users: Sequence[str],
    lam_for_user: dict[str, float],
    k: int,
    beta: float,
    anchor_min: int,
) -> dict[str, list[str]]:
    """Re-rank each user's ALS candidates at their own lambda."""
    out: dict[str, list[str]] = {}
    for user_id in users:
        recs = recommender.recommend(
            user_id, lam=lam_for_user[user_id], k=k, beta=beta, anchor_min=anchor_min
        )
        out[user_id] = [r.item_id for r in recs]
    return out


def lambda_sweep(
    recommender: Recommender,
    truths: Sequence[UserTruth],
    context: Context,
    grid: Sequence[float],
    k: int,
    beta: float,
    anchor_min: int,
) -> list[dict[str, Any]]:
    """Fixed lambda across the grid: the accuracy-versus-novelty frontier."""
    users = [t.user_id for t in truths]
    points: list[dict[str, Any]] = []
    for lam in grid:
        lists = reranked_lists(
            recommender, users, dict.fromkeys(users, lam), k, beta, anchor_floor(anchor_min, k)
        )
        scored = score_lists(lists, truths, context, k)
        points.append(
            {
                "lambda": lam,
                f"ndcg@{k}": scored[f"ndcg@{k}"],
                f"novelty@{k}": scored[f"novelty@{k}"],
                f"discovery_recall@{k}": scored[f"discovery_recall@{k}"],
            }
        )
    return points


def best_fixed_lambda(points: Sequence[dict[str, Any]], k: int) -> float:
    """The grid lambda with the highest NDCG@k, used for the fixed-global-lambda baseline."""
    scored = [p for p in points if p[f"ndcg@{k}"] is not None]
    if not scored:
        return 0.0
    return float(max(scored, key=lambda p: p[f"ndcg@{k}"])["lambda"])


def run(config_path: str, split: str) -> dict[str, Any]:
    """Evaluate every model on `split` and return the full results payload."""
    config = load_config(config_path)
    k_values = config["eval"]["k_values"]
    beta = config["rerank"]["beta"]
    anchor_min = config["rerank"]["anchor_min"]
    grid = config["rerank"]["lambda_grid"]
    processed = Path(config["data"]["processed_dir"])

    train = pl.read_parquet(processed / "train.parquet")
    evaluation = pl.read_parquet(processed / f"{split}.parquet")

    recommender = Recommender()
    truths = build_truth(train, evaluation, recommender.explorer)
    users = [t.user_id for t in truths]
    context = build_context(recommender, train)

    headline_k = max(k_values)
    sweep = lambda_sweep(recommender, truths, context, grid, headline_k, beta, anchor_min)
    fixed_lambda = best_fixed_lambda(sweep, headline_k)

    personalised = {user_id: recommender.personalised_lambda(user_id, config) for user_id in users}

    results: dict[str, Any] = {"models": {}, "lambda_sweep": sweep, "fixed_lambda": fixed_lambda}
    for k in k_values:
        floor = anchor_floor(anchor_min, k)
        variants: dict[str, dict[str, list[str]]] = {
            name: baseline_lists(Path("artifacts") / filename, users, k)
            for name, filename in BASELINE_FILES.items()
        }
        variants["als_mmr_fixed"] = reranked_lists(
            recommender, users, dict.fromkeys(users, fixed_lambda), k, beta, floor
        )
        variants["ours"] = reranked_lists(recommender, users, personalised, k, beta, floor)

        for name, lists in variants.items():
            scored = score_lists(lists, truths, context, k)
            scored.pop("_per_user", None)
            entry = results["models"].setdefault(name, {})
            entry.update(scored)
            entry[f"segments@{k}"] = segment_breakdown(lists, truths, context, k)

    results["users_evaluated"] = len(truths)
    results["cold_users"] = sum(1 for t in truths if t.is_cold)
    results["segment_sizes"] = {
        name: sum(1 for t in truths if t.segment == name)
        for name in ("loyalists", "middle", "explorers")
    }
    results["explorer_score_mean"] = float(np.mean([t.explorer_score for t in truths]))
    results["split"] = split
    return results


def main(config_path: str = "evals/configs/smoke.yaml", split: str = "val") -> None:
    """Evaluate each model and hand the results to report.write."""
    results = run(config_path, split)
    path = report_writer.write(results, load_config(config_path), config_path)
    print(f"\nreport: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the eval harness.")
    parser.add_argument("--config", default="evals/configs/smoke.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()
    main(args.config, args.split)
