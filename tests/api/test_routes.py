"""FastAPI contract tests for every route.

These need the fixture pipeline to have run, so the module skips cleanly when the
artefacts are missing rather than failing for the wrong reason.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

REQUIRED = [
    Path("artifacts/als_candidates.parquet"),
    Path("artifacts/popularity.parquet"),
    Path("artifacts/explorer_scores.parquet"),
    Path("data/processed/train.parquet"),
]

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED),
    reason="pipeline artefacts missing; run: uv run python run.py pipeline",
)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as started:
        yield started


@pytest.fixture(scope="module")
def sample_user(client: TestClient) -> str:
    return str(client.get("/listeners").json()[0]["user_id"])


def test_health_reports_loaded_artefacts(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["listeners"] > 0


def test_listeners_returns_summaries_sorted_by_explorer_score(client: TestClient) -> None:
    response = client.get("/listeners")
    assert response.status_code == 200
    body = response.json()
    assert len(body) > 0

    scores = [row["explorer_score"] for row in body]
    assert scores == sorted(scores)
    for row in body:
        assert 0.0 <= row["explorer_score"] <= 1.0
        assert 0.0 <= row["personalised_lambda"] <= 1.0
        assert row["train_listens"] > 0


def test_recommendations_use_the_personalised_lambda_by_default(
    client: TestClient, sample_user: str
) -> None:
    body = client.get(f"/recommendations/{sample_user}").json()
    assert body["lambda_used"] == body["personalised_lambda"]
    assert body["k"] == 20
    assert len(body["items"]) == 20


def test_recommendations_respect_an_explicit_lambda(client: TestClient, sample_user: str) -> None:
    body = client.get(f"/recommendations/{sample_user}?lambda=0.75").json()
    assert body["lambda_used"] == 0.75


def test_recommendations_have_no_duplicates_and_carry_explanations(
    client: TestClient, sample_user: str
) -> None:
    items = client.get(f"/recommendations/{sample_user}?k=10").json()["items"]
    assert len(items) == 10
    assert len({item["item_id"] for item in items}) == 10
    for item in items:
        assert item["why"]
        assert 0.0 <= item["relevance"] <= 1.0
        assert 0.0 <= item["novelty"] <= 1.0
    assert [item["rank"] for item in items] == list(range(1, 11))


def test_turning_the_dial_up_raises_mean_novelty(client: TestClient, sample_user: str) -> None:
    def mean_novelty(lam: float) -> float:
        items = client.get(f"/recommendations/{sample_user}?lambda={lam}&k=20").json()["items"]
        return float(sum(item["novelty"] for item in items) / len(items))

    assert mean_novelty(0.9) >= mean_novelty(0.1)


def test_lambda_below_zero_is_rejected(client: TestClient, sample_user: str) -> None:
    response = client.get(f"/recommendations/{sample_user}?lambda=-0.1")
    assert response.status_code == 422
    assert "lambda" in response.text


def test_lambda_above_one_is_rejected(client: TestClient, sample_user: str) -> None:
    response = client.get(f"/recommendations/{sample_user}?lambda=1.5")
    assert response.status_code == 422
    assert "lambda" in response.text


def test_unknown_user_returns_an_actionable_404(client: TestClient) -> None:
    response = client.get("/recommendations/not-a-listener")
    assert response.status_code == 404
    assert "/listeners" in response.json()["detail"]


def test_latest_report_is_served_or_explains_how_to_make_one(client: TestClient) -> None:
    response = client.get("/report/latest")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        report = response.json()["report"]
        assert report["schema_version"] == 1
        assert "models" in report
        assert "lambda_sweep" in report
    else:
        assert "run.py eval" in response.json()["detail"]
