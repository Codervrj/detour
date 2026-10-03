"""ALS on implicit feedback: the main candidate generator.

Confidence is `1 + alpha * log(1 + play_count)` (see detour.models.matrix). The model is
fitted on train only and writes the top-N candidates per user, along with the item factors
the re-ranker uses for its similarity term.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares

from detour.config import load_config, write_sidecar
from detour.models.matrix import Interactions, build


def fit(interactions: Interactions, config: dict[str, Any], seed: int) -> AlternatingLeastSquares:
    """Fit ALS on the confidence matrix."""
    model = AlternatingLeastSquares(
        factors=config["factors"],
        regularization=config["regularization"],
        iterations=config["iterations"],
        random_state=seed,
    )
    model.fit(interactions.matrix, show_progress=False)
    return model


def candidates(model: AlternatingLeastSquares, interactions: Interactions, n: int) -> pl.DataFrame:
    """Top-n unplayed items per user, as user_id / item_id / score / rank."""
    frames: list[pl.DataFrame] = []
    for user_id in interactions.users:
        index = interactions.user_index[user_id]
        ids, scores = model.recommend(
            index,
            interactions.matrix[index],
            N=n,
            filter_already_liked_items=True,
        )
        frames.append(
            pl.DataFrame(
                {
                    "user_id": [user_id] * len(ids),
                    "item_id": [interactions.items[i] for i in ids],
                    "score": np.asarray(scores, dtype=np.float64),
                    "rank": np.arange(1, len(ids) + 1, dtype=np.int32),
                }
            )
        )
    return pl.concat(frames)


def item_factor_frame(model: AlternatingLeastSquares, interactions: Interactions) -> pl.DataFrame:
    """Item factors, used for the re-ranker's similarity term and for ILD."""
    factors = np.asarray(model.item_factors)
    return pl.DataFrame(
        {
            "item_id": interactions.items,
            "factors": [
                factors[i].astype(np.float64).tolist() for i in range(len(interactions.items))
            ],
        }
    )


def main(
    config_path: str = "evals/configs/smoke.yaml",
    source: str = "data/processed/train.parquet",
    out_dir: str = "artifacts",
) -> None:
    """Fit ALS on train and write candidates plus item factors to artifacts/."""
    config = load_config(config_path)
    als_cfg = config["als"]
    seed = config["eval"]["seeds"][0]

    train = pl.read_parquet(source)
    interactions = build(train, als_cfg["alpha"])
    model = fit(interactions, als_cfg, seed)

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    table = candidates(model, interactions, als_cfg["candidates"])
    candidate_path = target_dir / "als_candidates.parquet"
    table.write_parquet(candidate_path)

    item_factor_frame(model, interactions).write_parquet(target_dir / "item_factors.parquet")

    write_sidecar(
        candidate_path,
        {
            "stage": "als",
            "computed_from": source,
            "seed": seed,
            "factors": als_cfg["factors"],
            "alpha": als_cfg["alpha"],
            "regularization": als_cfg["regularization"],
            "iterations": als_cfg["iterations"],
            "candidates_per_user": als_cfg["candidates"],
            "users": len(interactions.users),
            "items": len(interactions.items),
            "rows": table.height,
        },
    )
    print(
        f"als: {len(interactions.users)} users x {len(interactions.items)} items, "
        f"{table.height} candidates"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit ALS on implicit feedback.")
    parser.add_argument("--config", default="evals/configs/smoke.yaml")
    parser.add_argument("--source", default="data/processed/train.parquet")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()
    main(args.config, args.source, args.out_dir)
