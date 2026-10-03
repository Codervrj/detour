"""FastAPI app. Loads the pre-computed artist map from artifacts/ at startup.

No training happens at request time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import state
from api.routes import mapview, reports
from api.schemas import HealthResponse
from detour.eval import report as report_reader

DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the map once. A missing pipeline is reported per request, not at boot."""
    with suppress(FileNotFoundError):
        state.load()
    yield


app = FastAPI(
    title="Detour",
    description="A map of music learned from how people listen, and where you sit on it.",
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["GET"], allow_headers=["*"]
)

app.include_router(mapview.router)
app.include_router(reports.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Whether the map loaded, for the frontend's error states."""
    try:
        svc = state.service()
    except Exception:
        return HealthResponse(status="no map", artists=0, has_report=False, has_coordinates=False)
    return HealthResponse(
        status="ok",
        artists=len(svc.artists),
        has_report=report_reader.latest() is not None,
        has_coordinates=bool(svc.coords),
    )
