"""GET /recommendations/{user_id} - re-ranked recommendations for one listener.

A lambda outside [0, 1] is a 422 with a clear message. An unknown user is a 404 with an
actionable message.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query

from api import state
from api.schemas import RecommendationsResponse, TrackRecommendation
from detour.eval.harness import anchor_floor

router = APIRouter(tags=["recommendations"])


@router.get("/recommendations/{user_id}", response_model=RecommendationsResponse)
def recommendations(
    user_id: str = Path(description="A listener id from GET /listeners"),
    lam: float | None = Query(
        default=None,
        alias="lambda",
        ge=0.0,
        le=1.0,
        description="Dial position from 0 (familiar) to 1 (adventurous). "
        "Omit it to use this listener's own personalised lambda.",
    ),
    k: int = Query(default=20, ge=1, le=100, description="How many tracks to return"),
) -> RecommendationsResponse:
    """Re-rank this listener's candidates at the requested dial position."""
    rec = state.recommender()
    config = state.config()

    if user_id not in set(rec.users()):
        raise HTTPException(
            status_code=404,
            detail=(
                f"No listener '{user_id}'. Call GET /listeners for the available sample listeners."
            ),
        )

    personalised = rec.personalised_lambda(user_id, config)
    lambda_used = personalised if lam is None else lam

    items = rec.recommend(
        user_id,
        lam=lambda_used,
        k=k,
        beta=config["rerank"]["beta"],
        anchor_min=anchor_floor(config["rerank"]["anchor_min"], k),
    )
    return RecommendationsResponse(
        user_id=user_id,
        explorer_score=round(rec.explorer_score(user_id), 4),
        personalised_lambda=round(personalised, 4),
        lambda_used=round(lambda_used, 4),
        k=k,
        anchor_count=sum(1 for item in items if item.is_anchor),
        items=[TrackRecommendation(**vars(item)) for item in items],
    )
