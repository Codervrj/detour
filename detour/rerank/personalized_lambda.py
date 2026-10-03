"""Map an explorer score to the per-user lambda used by the re-ranker.

This is the project's actual claim: instead of one global dial position for everyone,
each listener gets their own. Loyalists get a small lambda so discovery arrives as a
gentle step outward; explorers get a large one and are pushed further.

    lam(u) = lam_min + (lam_max - lam_min) * explorer_score(u) ** gamma

The mapping is monotonic in the explorer score by construction, so a more adventurous
listener never gets a more conservative list. `gamma` bends the curve: below 1 it lifts
the middle, above 1 it holds the middle back. All three parameters are fitted on val and
stored in the config under `personalized_lambda`; the test split never informs them.
"""

from __future__ import annotations

from typing import Any


def lambda_for(explorer_score: float, config: dict[str, Any]) -> float:
    """Return a lambda in [lam_min, lam_max] for one user."""
    low = float(config["lam_min"])
    high = float(config["lam_max"])
    gamma = float(config["gamma"])

    clamped = min(max(explorer_score, 0.0), 1.0)
    return float(low + (high - low) * clamped**gamma)


def lambdas_for(scores: list[float], config: dict[str, Any]) -> list[float]:
    """Vectorised convenience wrapper over `lambda_for`."""
    return [lambda_for(score, config) for score in scores]
