# CLAUDE.md — Detour: music discovery through listening patterns

Detour recommends music that balances familiarity with discovery. It learns each listener's taste from real listening history, scores how adventurous they are, and re-ranks recommendations so explorers get pushed further and loyalists get gentler steps outward. The project is judged on three things: honest offline evaluation, a clean data pipeline, and an impressive UI that makes the discovery trade-off tangible.

Read this whole file before starting any task. If something here conflicts with a user instruction in the session, follow the user and point out the conflict.

---

## 1. Data sources (and what NOT to use)

**Primary: ListenBrainz listens (CC0).** Per-user listening events with timestamp, pseudonymous user id, track metadata, and optional MusicBrainz ids (recording, release, artist MBIDs).
- Get a sample via the ListenBrainz public BigQuery dataset (check https://listenbrainz.org/data for the current table name) or from the monthly dumps.
- Working sample: ~5,000–10,000 active users, 12 consecutive months. Keep raw pulls under `data/raw/`.

**Content features: MusicBrainz tags/genres** joined on MBIDs.

**Optional module: FMA (Free Music Archive)** — metadata CC BY 4.0, audio under each artist's licence, research use only. Use only for a separate "sounds like this, but lesser known" audio-similarity demo. Overlap with ListenBrainz is small; the core pipeline must never depend on joining the two.

**Do NOT use:**
- Spotify Web API `audio-features`, `audio-analysis`, `recommendations`, `related-artists` — deprecated for new apps since 27 Nov 2024 (they return 403). Do not copy tutorials that rely on them.
- Spotify Million Playlist Dataset — no longer publicly downloadable.
- Any attempt to re-identify users from listening data.

`data/` is gitignored. Never commit data files, API keys, or BigQuery credentials. Credentials go in `.env` (see `.env.example`).

---

## 2. Tech stack

**Pipeline + models (Python 3.11, managed with `uv`)**
- DuckDB + Parquet for storage and transforms; pandas/polars for analysis
- `implicit` (ALS on implicit feedback), `gensim` (item2vec), numpy/scipy
- ruff (lint + format), mypy (type check on `detour/`), pytest + hypothesis

**API:** FastAPI + pydantic v2, served with uvicorn. Loads pre-computed model artefacts from `artifacts/`; no training at request time.

**Frontend: React 19 + Vite + TypeScript** (in `web/`)
- TanStack Query for data fetching
- D3 or visx for custom charts (the frontier chart must be custom, not a default chart template)
- Vitest + Testing Library, Playwright + @axe-core/playwright
- Package manager: pnpm

---

## 3. Repository layout

```
detour/
├── CLAUDE.md
├── pyproject.toml
├── Makefile
├── .env.example
├── data/                    # gitignored: raw/ interim/ processed/
├── artifacts/               # gitignored: trained models, embeddings, popularity tables
├── detour/                  # Python package — all logic lives here
│   ├── ingest/              # bigquery_sample.py, dumps_loader.py
│   ├── clean/               # dedupe, filters, MBID normalisation
│   ├── split.py             # time-based train/val/test split
│   ├── features/            # explorer_score.py, popularity.py, tag_vectors.py
│   ├── models/              # popularity.py, random.py, itemknn.py, als.py, item2vec.py
│   ├── rerank/              # mmr.py, personalized_lambda.py
│   └── eval/                # metrics.py, harness.py, bootstrap.py, report.py
├── api/                     # FastAPI app: main.py, routes/, schemas.py
├── web/                     # React + Vite + TS frontend
├── evals/
│   ├── configs/             # main.yaml, smoke.yaml
│   ├── thresholds.json      # regression gates (see §7)
│   └── reports/             # generated: <date>_<run_id>.json + .md
├── tests/
│   ├── fixtures/            # tiny synthetic listens (≈50 users), golden metric cases
│   ├── unit/
│   ├── integration/
│   └── api/
└── notebooks/               # EDA only — no logic that other code imports
```

---

## 4. Commands

```bash
# setup
uv sync
pnpm --dir web install

# pipeline
uv run python -m detour.ingest.bigquery_sample --months 12 --users 8000
uv run python -m detour.clean.run
uv run python -m detour.split --train-months 9 --val-months 1 --test-months 2
uv run python -m detour.models.train --config evals/configs/main.yaml

# quality
uv run ruff check . && uv run ruff format --check .
uv run mypy detour
uv run pytest -q                       # unit + integration + api, uses fixtures only
make eval-smoke                        # fast eval on fixture data (< 1 min)
make eval                              # full eval on val split
make eval-final                        # test split — only when explicitly asked (see §6)

# app
uv run uvicorn api.main:app --reload
pnpm --dir web dev
pnpm --dir web test                    # vitest
pnpm --dir web e2e                     # playwright + axe
```

After any code change: run the relevant tests, plus ruff and mypy for Python or `pnpm --dir web typecheck && pnpm --dir web test` for the frontend, before saying the task is done.

---

## 5. Data pipeline rules

- **Dedupe scrobbles:** same user + same recording within 30 seconds counts as one listen.
- **Filters (on train period only):** users with ≥ 50 listens; items with ≥ 5 distinct listeners. Never compute filters using val/test data.
- **Identity:** prefer recording MBID → fall back to normalised `(artist, track)` string. Artist-level analysis uses artist MBID.
- **Split is by time, never random.** Train = months 1–9, val = month 10, test = months 11–12. Users must appear in train to be evaluated (cold-start users are reported separately, not mixed in).
- **Popularity, explorer scores, tag vectors and embeddings are all computed from train only.**
- Every stage writes Parquet to `data/processed/` with a small JSON sidecar: row counts, date range, filter stats.

---

## 6. Modelling approach

1. **Candidate generation:** ALS on implicit feedback (confidence = 1 + α·log(1 + play_count)) and item2vec on listening sessions (session break = 30 min gap). Generate top-200 candidates per user.
2. **Explorer score (per user, from train history):** share of listens going to artists first heard that month, smoothed over months, scaled to [0, 1]. Store the formula's parameters in config.
3. **Re-ranking:** MMR-style score
   `score(i) = (1 − λ_u) · relevance(i) + λ_u · novelty(i) − β · max_sim(i, already_selected)`
   where `λ_u` comes from the explorer score through a monotonic mapping fitted on val.
4. **Familiarity anchors:** every list of 20 keeps at least a configurable number of tracks from known artists, so discovery never feels random.
5. **Explanations:** each recommendation carries a short "why": nearest known track/artist and shared tags.

**Tuning discipline:** all hyper-parameters (ALS factors, α, λ mapping, β, anchor count) are tuned on val. The test split is touched only by `make eval-final`, run once at the end when the user explicitly asks. Never tune, re-train or pick models after looking at test numbers.

---

## 7. Evals

The project's claim is "better discovery without wrecking relevance". The eval harness exists to prove or disprove that honestly. Report whatever the numbers are.

### Ground truth
- **Relevant set:** items (and separately artists) the user listened to in the eval period.
- **Discovery set:** artists the user listened to in the eval period that never appear in their train history. These are real discoveries.

### Metrics (all @K, K ∈ {10, 20}, averaged over users)
| Metric | Definition |
|---|---|
| Recall@K | hits in relevant set / min(K, |relevant|) |
| NDCG@K | binary relevance, log2 discount |
| Discovery Recall@K | hits among the discovery set (artist level) / min(K, |discovery|) |
| Novelty@K | mean of −log2(p_i), p_i = share of train users who played item i |
| ILD@K | mean pairwise cosine distance of recommended items (item2vec embeddings) |
| Serendipity@K | share of recs that are relevant AND not in the popularity baseline's top-50 for that user |
| Familiarity anchor rate | share of recs from artists in the user's train history |
| Catalogue coverage | distinct recommended items across all users / catalogue size |
| Exposure Gini | Gini coefficient of recommendation counts across items |

### Baselines (all must be in every report)
random · popularity · item-kNN · ALS (no re-rank) · ALS + MMR with fixed global λ · **ALS + personalised λ (ours)**

### Protocol
- Fixed seeds; seed list lives in the eval config.
- 95% confidence intervals by bootstrapping over users (1,000 resamples).
- Segment every metric by explorer-score tercile (loyalists / middle / explorers) and report cold-ish users (< 100 train listens) separately.
- **λ sweep:** run fixed λ ∈ {0, 0.1, …, 1.0} and plot the accuracy (NDCG@20) vs. novelty frontier, with our personalised model drawn as a point on the same axes. This chart is the headline result for the report and the UI.

### Reports
`make eval` writes `evals/reports/<date>_<run_id>.json` (all metrics, CIs, config hash, data sidecar stats, git commit) and a `.md` summary with tables and the frontier plot. The frontend's results page reads the latest JSON report.

### Regression gates (`evals/thresholds.json`)
CI runs `make eval-smoke` and fails if:
- any metric computation errors or returns NaN;
- NDCG@20 or Discovery Recall@20 of "ours" drops by more than the stored tolerance versus the last accepted run;
- "ours" scores below the popularity baseline on NDCG@20.

Never edit `thresholds.json` to make a failing run pass. If a change legitimately moves the numbers, explain why in the PR description and update the file in a separate commit the user approves.

---

## 8. Tests

Tests run on `tests/fixtures/` only — no network, no BigQuery, no real data. The full suite must finish in under 60 seconds.

**Unit — metrics (`tests/unit/test_metrics.py`)**
- Golden cases with hand-computed values for every metric (put the arithmetic in a comment next to each case).
- Edge cases: empty relevant set, K larger than list, duplicate items in a list (must raise), all-relevant list (NDCG = 1).

**Unit — data**
- Dedupe: two plays of the same recording 10 s apart → one listen; 40 s apart → two.
- Filters computed on train only (build a fixture where a test-period listen would change the filter outcome and assert it doesn't).
- Split leakage: `max(train.ts) < min(val.ts) <= max(val.ts) < min(test.ts)`; popularity table contains no test-period counts.

**Unit — re-ranking (use hypothesis for property tests)**
- λ = 0 → order equals relevance order.
- Higher λ never decreases mean novelty of the list (monotonicity on fixtures).
- Output has no duplicates, has exactly K items, and respects the familiarity anchor minimum.
- Explorer score always within [0, 1]; a user who only replays one artist scores 0.

**Integration**
- Full pipeline on fixtures: ingest → clean → split → train → eval produces a report JSON matching the schema in `detour/eval/report.py`.
- Eval determinism: two runs with the same seed produce identical metrics.

**API (`tests/api/`)**
- FastAPI TestClient contract tests for every route; responses validated against pydantic schemas.
- `GET /recommendations/{user_id}?lambda=…` rejects λ outside [0, 1] with 422 and a clear message.
- Unknown user → 404 with an actionable message.

**Frontend**
- Vitest + Testing Library: discovery dial updates the list; empty and error states render with clear next steps.
- Playwright e2e: pick a sample listener → move the dial → list re-ranks → open a "why this" explanation → open the results page and see the frontier chart.
- axe accessibility scan on every page with zero serious/critical violations.
- Keyboard-only run through the main flow; `prefers-reduced-motion` disables non-essential animation.

When fixing a bug, first add a failing test that reproduces it.

---

## 9. UI — must be impressive

### Required tooling
Use the official **frontend-design** plugin for every UI task. Install once per machine:

```
/plugin install frontend-design@claude-plugins-official
```

If that marketplace isn't available: `/plugin marketplace add anthropics/claude-code` then `/plugin install frontend-design@claude-code-plugins`.

Before writing any frontend code for a new screen, follow the plugin's process: propose a compact design plan (palette as named hex values, typefaces and roles, layout with an ASCII wireframe, principles), review it against the brief below, revise anything that looks like a default, then build. Save the accepted plan to `web/DESIGN.md` and keep tokens as CSS variables in `web/src/styles/tokens.css`. Every later UI change must stay consistent with `web/DESIGN.md`.

### Design brief
- **Subject:** music discovery — the feeling of finding an artist you didn't know you'd love.
- **Audience:** examiners, recruiters and curious listeners seeing the project for the first time.
- **Primary job:** make the familiarity-vs-discovery trade-off something people can feel by moving one control.
- **Hero = the discovery dial, live.** The first screen is a working demo: choose a sample listener, move the dial, watch the list re-rank with each track's familiarity/novelty visible. No marketing hero, no stat-card row.
- **Spend boldness in one place** (the dial and how the list responds). Keep everything else quiet and disciplined.
- **Do not build a Spotify clone.** No dark-UI-with-green-accent music-app look, no generic SaaS card grid, no gradient washes.

### Screens
1. **Listen** — sample-listener picker (shows their explorer score and top tags), the discovery dial, the re-ranked list with per-track familiarity/novelty and a "why this" reveal.
2. **Listener profile** — listening timeline, artist adoption over time, where they sit on the explorer scale.
3. **Results** — the accuracy-vs-novelty frontier (custom D3/visx), baselines vs. ours with confidence intervals, segment breakdown by explorer tercile. Reads the latest eval report JSON.
4. **Method** — short plain-language explanation of data, split, metrics and limitations.

### Quality floor (non-negotiable)
Responsive down to 360 px, visible keyboard focus, `prefers-reduced-motion` respected, WCAG AA contrast, loading/empty/error states for every data view. Copy is plain, active voice, sentence case; buttons say exactly what happens. Take Playwright screenshots of each screen after building and critique them against `web/DESIGN.md` before calling a UI task done.

---

## 10. Working rules for Claude

- Plan first for any change touching more than two files; list the files and the approach, then implement.
- Logic goes in `detour/`; notebooks only import from it. Never copy notebook code into the package without tests.
- Keep functions small and typed; no hidden global state; configs in YAML under `evals/configs/`.
- Don't add new dependencies without saying why.
- Don't run `make eval-final`, delete data, or change `thresholds.json` unless the user explicitly asks.
- Commit messages: imperative mood, scoped (`eval: add bootstrap CIs`, `web: discovery dial keyboard support`).
- When results disappoint, report them plainly with likely causes — never cherry-pick seeds, segments or K values.

---

## 11. Attribution

README and the Method page must credit: ListenBrainz / MetaBrainz Foundation (CC0 listens), MusicBrainz (tags/metadata), and — if the optional module is used — FMA (Defferrard et al., ISMIR 2017; metadata CC BY 4.0).
