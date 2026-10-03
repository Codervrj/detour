"""Train the artist map.

PPMI + SVD is the map, chosen on evidence rather than fashion. On the held-out adoption
question it scored 0.1160 against item2vec's 0.2891 and got seven times the hit@10, so the
neural model was demoted to a baseline. The finding is reported rather than buried: for
predicting which artist somebody adopts next, "these artists share listeners over nine
months" beats "these artists were played in the same sitting".

What it does. Count how many listeners each pair of artists share, convert those counts to
positive pointwise mutual information so that a pair is only interesting when it co-occurs
more than chance would predict, then take a truncated SVD to compress the result into dense
vectors. Popularity drops out in the PMI step, which is why the map is not simply a chart.

    uv run python -m detour.models.artist_map
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from detour.config import load_config, write_sidecar
from detour.models.baselines import artist_plays, ppmi_svd_vectors


def vectors_frame(vectors: dict[str, np.ndarray], plays: dict[str, int]) -> pl.DataFrame:
    """Artist vectors as a frame, most played first."""
    artists = sorted(vectors, key=lambda a: -plays.get(a, 0))
    return pl.DataFrame(
        {
            "artist_mbid": artists,
            "count": [int(plays.get(a, 0)) for a in artists],
            "vector": [np.asarray(vectors[a], dtype=np.float64).tolist() for a in artists],
        }
    )


def main(
    config_path: str = "evals/configs/main.yaml",
    source: str = "data/processed/train.parquet",
    out_dir: str = "artifacts",
) -> None:
    """Build the map from train only and write it to artifacts/."""
    config = load_config(config_path)
    dimensions = config["map"]["dimensions"]

    train = pl.read_parquet(source)
    plays = artist_plays(train)
    print(f"building the map over {len(plays):,} artists")

    vectors = ppmi_svd_vectors(train, dimensions)
    frame = vectors_frame(vectors, plays)

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    vector_path = target_dir / "artist_vectors.parquet"
    frame.write_parquet(vector_path)

    write_sidecar(
        vector_path,
        {
            "stage": "artist_map",
            "model": "ppmi_svd",
            "computed_from": source,
            "dimensions": dimensions,
            "artists": frame.height,
            "chosen_because": (
                "best adoption rank percentile on val: 0.1160 against item2vec 0.2891"
            ),
        },
    )
    print(f"map: {frame.height:,} artists in {dimensions}d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the artist map.")
    parser.add_argument("--config", default="evals/configs/main.yaml")
    parser.add_argument("--source", default="data/processed/train.parquet")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()
    main(args.config, args.source, args.out_dir)
