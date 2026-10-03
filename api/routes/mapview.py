"""The artist map, and one listener's place on it."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api import state
from api.schemas import (
    ArtistNeighbours,
    ArtistPoint,
    Coverage,
    IslandOut,
    ListenerMap,
    MapResponse,
    NeighbourOut,
)

router = APIRouter(tags=["map"])

# Drawing every artist at once is unreadable and slow, so the background layer is the most
# played ones; a listener's own artists are always drawn on top regardless of popularity.
DEFAULT_MAP_ARTISTS = 3000


@router.get("/map", response_model=MapResponse)
def artist_map(limit: int = Query(default=DEFAULT_MAP_ARTISTS, ge=100, le=20000)) -> MapResponse:
    """The background layer: the most played artists, with their coordinates."""
    svc = state.service()
    top = sorted(svc.artists, key=lambda a: -svc.plays.get(a, 0))[:limit]
    return MapResponse(
        artists=[ArtistPoint(**vars(svc.to_artist(a, svc.plays.get(a, 0)))) for a in top],
        total_artists=len(svc.artists),
        showing=len(top),
    )


@router.get("/listeners", response_model=list[str])
def listeners() -> list[str]:
    """Held-out listeners available to explore when you have not imported your own."""
    return state.demo_listeners()


@router.get("/me", response_model=ListenerMap)
def me() -> ListenerMap:
    """Your own place on the map, from your imported Last.fm history."""
    listens = state.personal_listens()
    if listens is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No personal history imported. Run: "
                "uv run python -m detour.ingest.lastfm --user YOURNAME"
            ),
        )
    return _listener_map("you", listens)


@router.get("/listener/{listener}", response_model=ListenerMap)
def listener_map(listener: str) -> ListenerMap:
    """One held-out listener's place on the map."""
    listens = state.demo_listens(listener)
    if listens is None:
        raise HTTPException(
            status_code=404,
            detail=f"No listener '{listener}'. Call GET /listeners for the available ones.",
        )
    return _listener_map(listener, listens)


def _listener_map(label: str, listens) -> ListenerMap:  # type: ignore[no-untyped-def]
    """Shared body: resolve a history, place it, and describe what is there."""
    svc = state.service()
    resolution = svc.resolve_history(listens)
    if not resolution.counts:
        raise HTTPException(
            status_code=422,
            detail=(
                "None of these artists are in the map, so there is nothing to place. "
                "The map covers artists with at least five listeners in the training year."
            ),
        )

    counts = resolution.counts
    placement = svc.place_counts(counts)
    summary = resolution.summary()

    from detour.foldin import edge_artists

    edges = [svc.to_artist(a, counts.get(a, 0)) for a, _ in edge_artists(placement, k=12)]
    frontier = [
        NeighbourOut(
            artist_mbid=a,
            name=svc.name_of(a),
            similarity=round(1.0 - distance, 4),
            x=svc.position_of(a)[0],
            y=svc.position_of(a)[1],
        )
        for a, distance in svc.frontier(placement, counts, k=20)
    ]
    islands = [
        IslandOut(
            label=island.label,
            size=island.size,
            plays=island.plays,
            isolation=island.isolation,
            artists=[
                ArtistPoint(**vars(svc.to_artist(a, counts.get(a, 0)))) for a in island.artists[:25]
            ],
        )
        for island in svc.islands(counts)
    ]

    return ListenerMap(
        listener=label,
        coverage=Coverage(**summary),  # type: ignore[arg-type]
        total_plays=sum(counts.values()),
        artists=[
            ArtistPoint(**vars(svc.to_artist(a, plays)))
            for a, plays in sorted(counts.items(), key=lambda pair: -pair[1])[:400]
        ],
        islands=islands,
        edges=[ArtistPoint(**vars(a)) for a in edges],
        frontier=frontier,
    )


@router.get("/artist/{artist_mbid}", response_model=ArtistNeighbours)
def artist_neighbours(
    artist_mbid: str, k: int = Query(default=12, ge=1, le=50)
) -> ArtistNeighbours:
    """Who sits next to one artist on the map."""
    svc = state.service()
    if artist_mbid not in svc.index:
        raise HTTPException(status_code=404, detail=f"Artist '{artist_mbid}' is not in the map.")
    return ArtistNeighbours(
        artist_mbid=artist_mbid,
        name=svc.name_of(artist_mbid),
        neighbours=[
            NeighbourOut(
                artist_mbid=other,
                name=svc.name_of(other),
                similarity=round(similarity, 4),
                x=svc.position_of(other)[0],
                y=svc.position_of(other)[1],
            )
            for other, similarity in svc.neighbours(artist_mbid, k)
        ],
    )
