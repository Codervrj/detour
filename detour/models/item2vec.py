"""item2vec: the learned map of music.

Word2vec applied to listening sessions. Artists are words, sessions are sentences, and an
artist's vector is learned from the company it keeps. Nothing is labelled with a genre;
whatever structure appears was discovered from listening behaviour alone.

Skip-gram with negative sampling, because the signal of interest is "which artists share
a context", and skip-gram handles infrequent artists better than CBOW, which matters when
the long tail is exactly where discovery lives.

The window is deliberately wide. Order inside a listening session is close to arbitrary,
so a narrow window would read meaning into sequencing that is not there.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from gensim.models import Word2Vec

from detour.config import load_config, write_sidecar


def train(
    sessions: list[list[str]],
    dimensions: int,
    window: int,
    min_count: int,
    epochs: int,
    seed: int,
    workers: int = 1,
) -> Word2Vec:
    """Fit skip-gram with negative sampling on the session sequences.

    `workers=1` is not an oversight: gensim is only reproducible single-threaded, because
    parallel workers interleave updates non-deterministically. A report that cannot be
    reproduced is not evidence, so determinism wins over speed here.
    """
    return Word2Vec(
        sentences=sessions,
        vector_size=dimensions,
        window=window,
        min_count=min_count,
        sg=1,
        negative=10,
        sample=1e-4,
        epochs=epochs,
        seed=seed,
        workers=workers,
    )


def embedding_frame(model: Word2Vec) -> pl.DataFrame:
    """Artist vectors as a frame, ordered by how often the artist was seen."""
    keys = list(model.wv.index_to_key)
    vectors = np.asarray([model.wv[key] for key in keys], dtype=np.float32)
    return pl.DataFrame(
        {
            "artist_mbid": keys,
            "count": [int(model.wv.get_vecattr(key, "count")) for key in keys],
            "vector": [row.astype(np.float64).tolist() for row in vectors],
        }
    )


def neighbours(model: Word2Vec, artist_mbid: str, k: int = 10) -> list[tuple[str, float]]:
    """The k nearest artists to one artist, by cosine similarity."""
    if artist_mbid not in model.wv:
        return []
    return [(str(key), float(score)) for key, score in model.wv.most_similar(artist_mbid, topn=k)]


def main(
    config_path: str = "evals/configs/main.yaml",
    source: str = "data/processed/sessions.parquet",
    out_dir: str = "artifacts",
) -> None:
    """Train the artist embedding and write it to artifacts/."""
    config = load_config(config_path)
    settings: dict[str, Any] = config["item2vec"]
    seed = config["eval"]["seeds"][0]

    frame = pl.read_parquet(source)
    sessions = frame["artists"].to_list()
    print(f"training on {len(sessions):,} sessions")

    model = train(
        sessions,
        dimensions=settings["dimensions"],
        window=settings["window"],
        min_count=settings["min_count"],
        epochs=settings["epochs"],
        seed=seed,
    )

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(target_dir / "item2vec.kv"))

    vectors = embedding_frame(model)
    vector_path = target_dir / "artist_vectors.parquet"
    vectors.write_parquet(vector_path)

    write_sidecar(
        vector_path,
        {
            "stage": "item2vec",
            "computed_from": source,
            "seed": seed,
            "dimensions": settings["dimensions"],
            "window": settings["window"],
            "min_count": settings["min_count"],
            "epochs": settings["epochs"],
            "sessions": len(sessions),
            "artists_in_vocabulary": vectors.height,
        },
    )
    print(f"item2vec: {vectors.height:,} artists in vocabulary, {settings['dimensions']}d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the artist embedding.")
    parser.add_argument("--config", default="evals/configs/main.yaml")
    parser.add_argument("--source", default="data/processed/sessions.parquet")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()
    main(args.config, args.source, args.out_dir)
