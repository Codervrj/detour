"""Other ways of building the same map, so item2vec has something to beat.

A single model with a plausible-looking score proves nothing. Each of these produces artist
vectors from the same train split, and the eval ranks them on the same held-out question.

| model        | what it is                                              |
|--------------|---------------------------------------------------------|
| random       | random vectors, the floor                                |
| popularity   | ranks by global play count and ignores the listener      |
| ppmi_svd     | count-based embedding, no neural training                |
| als          | matrix factorisation over listener-artist plays          |
| item2vec     | ours, trained on session sequences                       |

`popularity` is the one that matters. It is the confound: popular artists get adopted more
often, so any model that quietly learned popularity would look good. If item2vec cannot
beat it, the map adds nothing.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds


def artist_plays(train: pl.DataFrame) -> dict[str, int]:
    """Global play count per artist, used for the popularity baseline and the control."""
    counted = train.group_by("artist_mbid").agg(pl.len().alias("plays"))
    return {str(r["artist_mbid"]): int(r["plays"]) for r in counted.iter_rows(named=True)}


def random_vectors(artists: list[str], dimensions: int, seed: int) -> dict[str, np.ndarray]:
    """Random unit vectors. Any model that cannot beat this has learned nothing."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(len(artists), dimensions))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return dict(zip(artists, matrix, strict=True))


def listener_artist_matrix(train: pl.DataFrame) -> tuple[coo_matrix, list[str], list[str]]:
    """Sparse listener-by-artist play counts."""
    counted = train.group_by(["user_id", "artist_mbid"]).agg(pl.len().alias("plays"))
    users = sorted(counted["user_id"].unique().to_list())
    artists = sorted(counted["artist_mbid"].unique().to_list())
    user_index = {u: i for i, u in enumerate(users)}
    artist_index = {a: i for i, a in enumerate(artists)}

    rows = np.array([user_index[u] for u in counted["user_id"].to_list()], dtype=np.int32)
    cols = np.array([artist_index[a] for a in counted["artist_mbid"].to_list()], dtype=np.int32)
    values = np.asarray(counted["plays"].to_list(), dtype=np.float64)
    shape = (len(users), len(artists))
    return coo_matrix((values, (rows, cols)), shape=shape), users, artists


def ppmi_svd_vectors(
    train: pl.DataFrame, dimensions: int, shift: float = 1.0
) -> dict[str, np.ndarray]:
    """Classic count-based embedding: positive pointwise mutual information, then SVD.

    This is the honest "did you need a neural model at all" baseline. It captures
    co-listening structure with nothing but counts and linear algebra.
    """
    matrix, _, artists = listener_artist_matrix(train)
    binary = (matrix > 0).astype(np.float64).tocsr()

    # Co-listening counts between artists.
    co = (binary.T @ binary).tocoo()
    total = co.data.sum()
    artist_totals = np.asarray(binary.sum(axis=0)).ravel()
    artist_totals[artist_totals == 0] = 1.0

    expected = (artist_totals[co.row] * artist_totals[co.col]) / artist_totals.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co.data / total) / (expected / artist_totals.sum() + 1e-12) + 1e-12)
    ppmi = np.maximum(pmi - np.log(shift), 0.0)

    sparse = coo_matrix((ppmi, (co.row, co.col)), shape=co.shape).tocsr()
    k = min(dimensions, min(sparse.shape) - 1)
    left, singular, _ = svds(sparse, k=k)
    embedded = left * np.sqrt(singular)
    return dict(zip(artists, embedded, strict=True))


def als_vectors(
    train: pl.DataFrame, dimensions: int, seed: int, alpha: float = 40.0, iterations: int = 15
) -> dict[str, np.ndarray]:
    """Artist factors from implicit matrix factorisation."""
    from implicit.als import AlternatingLeastSquares

    matrix, _, artists = listener_artist_matrix(train)
    confidence = matrix.tocsr()
    confidence.data = 1.0 + alpha * np.log1p(confidence.data)

    model = AlternatingLeastSquares(
        factors=dimensions, regularization=0.05, iterations=iterations, random_state=seed
    )
    model.fit(confidence, show_progress=False)
    factors = np.asarray(model.item_factors, dtype=np.float64)
    return dict(zip(artists, factors, strict=True))
