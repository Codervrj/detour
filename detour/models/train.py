"""Train every model named in a config and write artefacts to artifacts/.

This is the single place that knows the order things have to happen in: features before
models, because the novelty term needs the popularity table.
"""

from __future__ import annotations

import argparse

from detour.features import explorer_score
from detour.features import popularity as popularity_feature
from detour.models import (
    als,
    itemknn,
)
from detour.models import (
    popularity as popularity_model,
)
from detour.models import (
    random as random_model,
)


def main(
    config_path: str = "evals/configs/smoke.yaml",
    source: str = "data/processed/train.parquet",
    out_dir: str = "artifacts",
) -> None:
    """Run each stage in turn, printing what it produced."""
    popularity_feature.main(source, f"{out_dir}/popularity.parquet")
    explorer_score.main(config_path, source, f"{out_dir}/explorer_scores.parquet")
    als.main(config_path, source, out_dir)
    popularity_model.main(source, f"{out_dir}/popularity_candidates.parquet")
    random_model.main(source, f"{out_dir}/random_candidates.parquet")
    itemknn.main(source, f"{out_dir}/itemknn_candidates.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all models for a config.")
    parser.add_argument("--config", default="evals/configs/smoke.yaml")
    parser.add_argument("--source", default="data/processed/train.parquet")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()
    main(args.config, args.source, args.out_dir)
