"""Flatten the map to two dimensions, for drawing only.

The map lives in 128 dimensions. A screen has two, so a projection is needed to draw it.
UMAP is used because it preserves local neighbourhoods better than PCA, which matters when
the whole point is "who is near whom".

**Nothing is ever measured on these coordinates.** Every metric, every distance and every
cluster is computed in the full 128-dimensional space. A 2D projection distorts distance by
construction, and treating it as the truth is the standard way to produce a chart that
lies. These numbers exist so a person can look at the map, nothing more.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from detour.config import load_config, write_sidecar


def project(
    vectors: np.ndarray, seed: int, neighbours: int = 15, min_dist: float = 0.1
) -> np.ndarray:
    """Reduce to two dimensions with UMAP, on cosine distance."""
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbours,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return np.asarray(reducer.fit_transform(vectors), dtype=np.float64)


def to_unit_square(coordinates: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1] on both axes so the frontend needs no knowledge of the scale."""
    low = coordinates.min(axis=0)
    span = coordinates.max(axis=0) - low
    span[span < 1e-12] = 1.0
    return np.asarray((coordinates - low) / span, dtype=np.float64)


def main(
    config_path: str = "evals/configs/main.yaml",
    source: str = "artifacts/artist_vectors.parquet",
    out_path: str = "artifacts/map_2d.parquet",
) -> None:
    """Write 2D coordinates for every artist in the map."""
    config = load_config(config_path)
    seed = config["eval"]["seeds"][0]

    frame = pl.read_parquet(source)
    vectors = np.asarray(frame["vector"].to_list(), dtype=np.float64)
    print(f"projecting {len(vectors):,} artists from {vectors.shape[1]}d to 2d")

    coordinates = to_unit_square(project(vectors, seed))

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "artist_mbid": frame["artist_mbid"],
            "count": frame["count"],
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        }
    ).write_parquet(target)

    write_sidecar(
        target,
        {
            "stage": "project",
            "method": "umap",
            "metric": "cosine",
            "seed": seed,
            "artists": len(vectors),
            "note": "for drawing only; all metrics use the full-dimensional space",
        },
    )
    print(f"projected {len(vectors):,} artists")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project the map to 2d for display.")
    parser.add_argument("--config", default="evals/configs/main.yaml")
    parser.add_argument("--source", default="artifacts/artist_vectors.parquet")
    parser.add_argument("--out", default="artifacts/map_2d.parquet")
    args = parser.parse_args()
    main(args.config, args.source, args.out)
