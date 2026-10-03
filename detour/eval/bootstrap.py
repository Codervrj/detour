"""95% confidence intervals by bootstrapping over users, 1000 resamples."""

from __future__ import annotations


def confidence_interval(values: list[float], resamples: int, seed: int) -> tuple[float, float]:
    """Return the 2.5th and 97.5th percentile of the resampled mean."""
    raise NotImplementedError
