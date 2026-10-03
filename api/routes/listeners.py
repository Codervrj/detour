"""GET /listeners - sample listeners with their explorer score and top tags."""

from __future__ import annotations

from fastapi import APIRouter

from api import state
from api.schemas import ListenerSummary

router = APIRouter(tags=["listeners"])


@router.get("/listeners", response_model=list[ListenerSummary])
def list_listeners() -> list[ListenerSummary]:
    """Every sample listener, ordered by how adventurous they are."""
    rec = state.recommender()
    config = state.config()

    summaries = [
        ListenerSummary(
            user_id=user_id,
            explorer_score=round(rec.explorer_score(user_id), 4),
            personalised_lambda=round(rec.personalised_lambda(user_id, config), 4),
            train_listens=int(rec.train_listens.get(user_id, 0)),
            top_artists=rec.top_artists.get(user_id, []),
        )
        for user_id in rec.users()
    ]
    return sorted(summaries, key=lambda s: s.explorer_score)
