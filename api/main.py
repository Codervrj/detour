"""FastAPI app. Loads pre-computed artefacts from artifacts/ at startup.

No training happens at request time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import state
from api.routes import listeners, recommendations, reports
from api.schemas import HealthResponse
from detour.eval import report as report_reader

# The Vite dev server, so the frontend can call the API while developing.
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load artefacts once. A missing pipeline is reported per request, not at boot."""
    with suppress(FileNotFoundError):
        state.load()
    yield


app = FastAPI(
    title="Detour",
    description="Music discovery that balances familiarity with exploration.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(listeners.router)
app.include_router(recommendations.router)
app.include_router(reports.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Whether the artefacts loaded, for the frontend's error states."""
    try:
        count = len(state.recommender().users())
    except Exception:
        return HealthResponse(status="no artefacts", listeners=0, has_report=False)
    return HealthResponse(
        status="ok", listeners=count, has_report=report_reader.latest() is not None
    )
