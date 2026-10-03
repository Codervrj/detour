"""Scaffold the Detour repository skeleton.

Idempotent: a file is written only when it does not already exist, so running this
twice never overwrites your edits. Stdlib only, so it runs before `uv sync`.

    uv run python scripts/create_project.py --with-web --git
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories that become Python packages (each gets an __init__.py).
PACKAGES = [
    "detour",
    "detour/ingest",
    "detour/clean",
    "detour/features",
    "detour/models",
    "detour/rerank",
    "detour/eval",
    "api",
    "api/routes",
    "tests",
    "tests/fixtures",
    "tests/unit",
    "tests/integration",
    "tests/api",
]

# Plain directories. data/ and artifacts/ are gitignored, so they carry a .gitkeep.
PLAIN_DIRS = [
    "data/raw",
    "data/interim",
    "data/processed",
    "artifacts",
    "evals/configs",
    "evals/reports",
    "notebooks",
    "scripts",
]

PYPROJECT = """\
[project]
name = "detour"
version = "0.1.0"
description = "Music discovery through listening patterns"
requires-python = ">=3.11,<3.12"
dependencies = [
    "duckdb>=1.1",
    "polars>=1.17",
    "pandas>=2.2",
    "numpy>=2.1",
    "scipy>=1.14",
    "pyarrow>=18.0",
    "implicit>=0.7.2",
    "gensim>=4.3.3",
    "pyyaml>=6.0",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.10",
]

[dependency-groups]
dev = [
    "ruff>=0.8",
    "mypy>=1.13",
    "pytest>=8.3",
    "hypothesis>=6.122",
    "httpx>=0.28",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["detour"]

[tool.uv]
# This machine sits behind a TLS-intercepting proxy. Its root CA is in the Windows
# certificate store but not in uv's bundled bundle, so use the platform store.
native-tls = true

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
"""

GITIGNORE = """\
data/
artifacts/
.env
__pycache__/
*.py[cod]
.venv/
.mypy_cache/
.pytest_cache/
.ruff_cache/
node_modules/
web/dist/
evals/reports/*.json
evals/reports/*.md
"""

ENV_EXAMPLE = """\
# Last.fm API key, for importing your own listening history.
# Get a free one in under a minute: https://www.last.fm/api/account/create
# Only the API key is needed; reading public scrobbles requires no secret or OAuth.
LASTFM_API_KEY=
LASTFM_USER=

# Optional: BigQuery access to the ListenBrainz public dataset.
# Not needed. The pipeline uses the public monthly dumps, which require no account.
GOOGLE_APPLICATION_CREDENTIALS=
GCP_PROJECT_ID=
"""

MAKEFILE = "\n".join(
    [
        "# make is not installed everywhere. run.py is the real entry point; these forward to it.",
        ".PHONY: setup pipeline eval eval-smoke eval-final api web demo test lint",
        "",
        "setup:",
        "\tuv run python run.py setup",
        "",
        "pipeline:",
        "\tuv run python run.py pipeline",
        "",
        "eval:",
        "\tuv run python run.py eval --config evals/configs/main.yaml",
        "",
        "eval-smoke:",
        "\tuv run python run.py eval --config evals/configs/smoke.yaml",
        "",
        "eval-final:",
        "\tuv run python run.py eval --config evals/configs/main.yaml --split test",
        "",
        "api:",
        "\tuv run python run.py api",
        "",
        "web:",
        "\tuv run python run.py web",
        "",
        "demo:",
        "\tuv run python run.py demo",
        "",
        "test:",
        "\tuv run pytest",
        "",
        "lint:",
        "\tuv run ruff check . && uv run ruff format --check . && uv run mypy detour",
        "",
    ]
)

README = """\
# Detour

Music discovery that balances familiarity with exploration. Detour learns taste from real listening
history, scores how adventurous each listener is, and re-ranks recommendations so explorers get
pushed further out while loyalists take gentler steps.

## Run it

```bash
uv run python scripts/create_project.py   # scaffold (idempotent)
uv run python run.py setup                # install Python and web deps
uv run python run.py demo                 # pipeline, then API + web together
```

The demo runs on synthetic fixture data, so it needs no credentials and no network.

## Commands

| Command | What it does |
|---|---|
| `run.py setup` | `uv sync`, enable pnpm, install web deps |
| `run.py pipeline` | fixtures to clean to split to train to artifacts |
| `run.py eval` | run the eval harness, write a report |
| `run.py api` | serve the FastAPI app |
| `run.py web` | serve the Vite dev server |
| `run.py demo` | pipeline, then API and web together |

See `plan.md` for the build order and `CLAUDE.md` for the full spec.

## Attribution

- Listening data: [ListenBrainz](https://listenbrainz.org/) / MetaBrainz Foundation (CC0)
- Tags and metadata: [MusicBrainz](https://musicbrainz.org/)
- Optional audio-similarity module: FMA (Defferrard et al., ISMIR 2017; metadata CC BY 4.0)
"""

SMOKE_YAML = """\
# Fast eval on fixture data. Must finish in under a minute.
name: smoke
data:
  source: fixtures
  processed_dir: data/processed
split:
  train_months: 9
  val_months: 1
  test_months: 2
filters:
  min_user_listens: 5      # relaxed for the tiny fixture set
  min_item_listeners: 2
eval:
  split: val
  k_values: [10, 20]
  seeds: [13]
  bootstrap_resamples: 0   # CIs are a later increment
explorer_score:
  smoothing_months: 3
  clip: [0.0, 1.0]
als:
  factors: 16
  alpha: 40.0
  regularization: 0.05
  iterations: 10
  candidates: 200
rerank:
  beta: 0.3
  anchor_min: 4
  lambda_grid: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
personalized_lambda:
  lam_min: 0.1
  lam_max: 0.8
  gamma: 1.0
"""

MAIN_YAML = """\
# Full eval on the val split. Never point this at test without an explicit request.
name: main
data:
  source: processed
  processed_dir: data/processed
split:
  train_months: 9
  val_months: 1
  test_months: 2
filters:
  min_user_listens: 50
  min_item_listeners: 5
eval:
  split: val
  k_values: [10, 20]
  seeds: [13, 17, 23]
  bootstrap_resamples: 1000
explorer_score:
  smoothing_months: 3
  clip: [0.0, 1.0]
als:
  factors: 128
  alpha: 40.0
  regularization: 0.05
  iterations: 20
  candidates: 200
rerank:
  beta: 0.3
  anchor_min: 4
  lambda_grid: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
personalized_lambda:
  lam_min: 0.1
  lam_max: 0.8
  gamma: 1.0
"""

THRESHOLDS = """\
{
  "note": "Regression gates for CI. Never edit these to make a failing run pass.",
  "tolerance": {
    "ndcg@20": 0.01,
    "discovery_recall@20": 0.01
  },
  "last_accepted": null,
  "must_beat_popularity_on": ["ndcg@20"]
}
"""

# path -> (module docstring, [(signature, body docstring)])
STUBS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "detour/ingest/bigquery_sample.py": (
        "Pull a listens sample from the ListenBrainz public BigQuery dataset.",
        [
            (
                "def main(months: int, users: int) -> None:",
                "Write a raw listens sample to data/raw/. TODO: increment beyond the demo.",
            )
        ],
    ),
    "detour/ingest/dumps_loader.py": (
        "Load listens from the monthly ListenBrainz dumps.",
        [
            (
                "def main(dump_dir: str) -> None:",
                "Read a dump directory into data/raw/. TODO: increment beyond the demo.",
            )
        ],
    ),
    "detour/clean/run.py": (
        "Dedupe scrobbles, apply filters and normalise identity.\n\n"
        "Dedupe rule: same user and same recording within 30 seconds is one listen.\n"
        "Identity: recording MBID, falling back to a normalised (artist, track) string.",
        [
            (
                "def main(config_path: str) -> None:",
                "Clean data/raw/ into data/processed/listens.parquet with a JSON sidecar.",
            )
        ],
    ),
    "detour/split.py": (
        "Time-based train/val/test split. Never random.\n\n"
        "Train is months 1-9, val is month 10, test is months 11-12.",
        [
            (
                "def main(train_months: int, val_months: int, test_months: int) -> None:",
                "Split the cleaned listens by time and write one Parquet file per split.",
            )
        ],
    ),
    "detour/features/explorer_score.py": (
        "Per-user explorer score from train history.\n\n"
        "Share of listens going to artists first heard that month, smoothed over months\n"
        "and scaled to [0, 1]. Computed from train only.",
        [
            (
                "def main(config_path: str) -> None:",
                "Write artifacts/explorer_scores.parquet.",
            )
        ],
    ),
    "detour/features/popularity.py": (
        "Item popularity from train only. Feeds Novelty@K and the popularity baseline.",
        [
            (
                "def main(config_path: str) -> None:",
                "Write artifacts/popularity.parquet: item id, listener count, play count.",
            )
        ],
    ),
    "detour/features/tag_vectors.py": (
        "MusicBrainz tag vectors joined on MBIDs, used for the shared-tag explanations.",
        [
            (
                "def main(config_path: str) -> None:",
                "Write artifacts/tag_vectors.parquet. TODO: later increment.",
            )
        ],
    ),
    "detour/models/popularity.py": (
        "Popularity baseline: recommend the most played items the user has not heard.",
        [
            (
                "def main(config_path: str) -> None:",
                "Write top-N candidates per user.",
            )
        ],
    ),
    "detour/models/random.py": (
        "Random baseline. Seeded, so runs are reproducible.",
        [
            (
                "def main(config_path: str) -> None:",
                "Write top-N random candidates per user.",
            )
        ],
    ),
    "detour/models/itemknn.py": (
        "Item-kNN baseline on co-listening counts.",
        [
            (
                "def main(config_path: str) -> None:",
                "Write top-N candidates per user. TODO: later increment.",
            )
        ],
    ),
    "detour/models/als.py": (
        "ALS on implicit feedback.\n\nConfidence is 1 + alpha * log(1 + play_count).",
        [
            (
                "def main(config_path: str) -> None:",
                "Fit ALS on train and write user/item factors plus top-200 candidates per user.",
            )
        ],
    ),
    "detour/models/item2vec.py": (
        "item2vec over listening sessions. A session breaks on a 30 minute gap.",
        [
            (
                "def main(config_path: str) -> None:",
                "Write artifacts/item2vec.kv. TODO: later increment.",
            )
        ],
    ),
    "detour/models/train.py": (
        "Train every model named in a config and write artefacts to artifacts/.",
        [
            (
                "def main(config_path: str) -> None:",
                "Run each configured model in turn.",
            )
        ],
    ),
    "detour/rerank/mmr.py": (
        "MMR-style re-ranking.\n\n"
        "score(i) = (1 - lam) * relevance(i) + lam * novelty(i) - beta * max_sim(i, selected)\n\n"
        "Every list of K keeps at least anchor_min tracks from artists the user already knows,\n"
        "so discovery never feels random.",
        [
            (
                "def rerank(\n"
                "    candidates: list[dict[str, object]],\n"
                "    lam: float,\n"
                "    beta: float,\n"
                "    k: int,\n"
                "    anchor_min: int,\n"
                ") -> list[dict[str, object]]:",
                "Return exactly k items, no duplicates, respecting the familiarity anchor floor.",
            )
        ],
    ),
    "detour/rerank/personalized_lambda.py": (
        "Map an explorer score to the per-user lambda used by the re-ranker.\n\n"
        "The mapping is monotonic and its parameters are fitted on val only.",
        [
            (
                "def lambda_for(explorer_score: float, config: dict[str, object]) -> float:",
                "Return a lambda in [0, 1] for one user.",
            )
        ],
    ),
    "detour/eval/metrics.py": (
        "Metric implementations. Every metric is computed @K and averaged over users.\n\n"
        "A list containing duplicate items is an error, not something to silently dedupe.",
        [
            (
                "def recall_at_k(recs: list[str], relevant: set[str], k: int) -> float:",
                "Hits in the relevant set divided by min(k, len(relevant)).",
            ),
            (
                "def ndcg_at_k(recs: list[str], relevant: set[str], k: int) -> float:",
                "Binary relevance with a log2 discount.",
            ),
            (
                "def discovery_recall_at_k(recs: list[str], discovery: set[str], k: int) -> float:",
                "Artist-level hits among artists absent from the user's train history.",
            ),
            (
                "def novelty_at_k(recs: list[str], popularity: dict[str, float], k: int) -> float:",
                "Mean of -log2(p_i), where p_i is the share of train users who played item i.",
            ),
        ],
    ),
    "detour/eval/harness.py": (
        "Run every baseline and our model over the eval split and collect metrics.\n\n"
        "Baselines, all required in every report: random, popularity, item-kNN, ALS,\n"
        "ALS + MMR with a fixed global lambda, and ALS + personalised lambda (ours).",
        [
            (
                'def main(config_path: str, split: str = "val") -> None:',
                "Evaluate each model and hand the results to report.write.",
            )
        ],
    ),
    "detour/eval/bootstrap.py": (
        "95% confidence intervals by bootstrapping over users, 1000 resamples.",
        [
            (
                "def confidence_interval(values: list[float], resamples: int, "
                "seed: int) -> tuple[float, float]:",
                "Return the 2.5th and 97.5th percentile of the resampled mean.",
            )
        ],
    ),
    "detour/eval/report.py": (
        "Report schema and writer.\n\n"
        "Writes evals/reports/<date>_<run_id>.json with all metrics, CIs, the config hash,\n"
        "the data sidecar stats and the git commit, plus a .md summary. The frontend's\n"
        "results page reads the latest JSON.",
        [
            (
                "def write(results: dict[str, object], config: dict[str, object]) -> str:",
                "Write the JSON and Markdown report and return the JSON path.",
            )
        ],
    ),
    "api/schemas.py": (
        "Pydantic v2 response models for the API.",
        [],
    ),
    "api/main.py": (
        "FastAPI app. Loads pre-computed artefacts from artifacts/ at startup.\n\n"
        "No training happens at request time.",
        [],
    ),
    "api/routes/listeners.py": (
        "GET /listeners - sample listeners with their explorer score and top tags.",
        [],
    ),
    "api/routes/recommendations.py": (
        "GET /recommendations/{user_id} - re-ranked recommendations for one listener.\n\n"
        "A lambda outside [0, 1] is a 422 with a clear message. An unknown user is a 404\n"
        "with an actionable message.",
        [],
    ),
    "api/routes/reports.py": (
        "GET /report/latest - the newest eval report JSON.",
        [],
    ),
    "tests/fixtures/make_fixtures.py": (
        "Generate tiny synthetic listens so the project runs with no network or keys.\n\n"
        "Around 50 users across 12 months, spanning the explorer spectrum: some replay\n"
        "a single artist (score must be 0), some adopt new artists monthly.",
        [
            (
                'def main(out_dir: str = "tests/fixtures", seed: int = 13) -> None:',
                "Write a deterministic listens Parquet file; print rows and date range.",
            )
        ],
    ),
}

Q = '"""'


def stub_source(module_doc: str, functions: list[tuple[str, str]]) -> str:
    """Render a stub module: a module docstring plus typed NotImplementedError functions."""
    # ruff format keeps a one-line docstring on one line; only wrap multi-line ones.
    opening = f"{Q}{module_doc}\n{Q}\n" if "\n" in module_doc else f"{Q}{module_doc}{Q}\n"
    parts = [opening]
    if functions:
        parts.append("\nfrom __future__ import annotations\n")
        for signature, doc in functions:
            parts.append(f"\n\n{signature}\n    {Q}{doc}{Q}\n    raise NotImplementedError\n")
    return "".join(parts)


def write(rel_path: str, content: str) -> bool:
    """Write content to rel_path unless it already exists. Return True if created."""
    path = ROOT / rel_path
    if path.exists():
        print(f"skip   {rel_path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"create {rel_path}")
    return True


def scaffold_web() -> None:
    """Create the Vite React+TS app and add the frontend dependencies."""
    if (ROOT / "web" / "package.json").exists():
        print("skip   web/ (package.json exists)")
        return
    subprocess.run(["corepack", "enable", "pnpm"], cwd=ROOT, check=False, shell=True)
    subprocess.run(
        ["pnpm", "create", "vite", "web", "--template", "react-ts"],
        cwd=ROOT,
        check=True,
        shell=True,
    )
    subprocess.run(
        ["pnpm", "--dir", "web", "add", "@tanstack/react-query", "@visx/scale", "@visx/shape"],
        cwd=ROOT,
        check=True,
        shell=True,
    )
    subprocess.run(
        [
            "pnpm",
            "--dir",
            "web",
            "add",
            "-D",
            "vitest",
            "@testing-library/react",
            "@testing-library/jest-dom",
            "@playwright/test",
            "@axe-core/playwright",
        ],
        cwd=ROOT,
        check=True,
        shell=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold the Detour repository.")
    parser.add_argument("--with-web", action="store_true", help="scaffold the Vite frontend")
    parser.add_argument("--git", action="store_true", help="git init if not already a repo")
    args = parser.parse_args()

    created = 0

    for package in PACKAGES:
        created += write(f"{package}/__init__.py", "")

    for directory in PLAIN_DIRS:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    created += write("data/.gitkeep", "")
    created += write("artifacts/.gitkeep", "")
    created += write("evals/reports/.gitkeep", "")

    for rel_path, content in {
        "pyproject.toml": PYPROJECT,
        ".gitignore": GITIGNORE,
        ".env.example": ENV_EXAMPLE,
        "Makefile": MAKEFILE,
        "README.md": README,
        "evals/configs/smoke.yaml": SMOKE_YAML,
        "evals/configs/main.yaml": MAIN_YAML,
        "evals/thresholds.json": THRESHOLDS,
    }.items():
        created += write(rel_path, content)

    for rel_path, (module_doc, functions) in STUBS.items():
        created += write(rel_path, stub_source(module_doc, functions))

    if args.git and not (ROOT / ".git").exists():
        subprocess.run(["git", "init"], cwd=ROOT, check=True)

    if args.with_web:
        scaffold_web()

    print(f"\n{created} file(s) created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
