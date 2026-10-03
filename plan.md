# Plan — Detour v2: a learned map of music, and where you sit on it

## Context

Version 1 is built and runs: scaffold, pipeline, ALS, re-ranking, eval harness, API, React UI,
82 Python tests and 19 frontend tests, all green. It is described at the bottom of this file.

It is also **not useful to anyone**. Two reasons, and only one was a mistake:

1. **The data is invented.** `tests/fixtures/make_fixtures.py` generates 50 fake listeners. Every
   number in the app describes people who do not exist. That was a deliberate shortcut to get the
   plumbing running without credentials, and it worked, but it proves nothing about music.
2. **The output is predictions about strangers.** You select user-049 and see what a fictional
   person might play next. There is no way to put yourself in. Collaborative filtering needs a
   crowd, so the whole thing points at a population and never at the person using it.

This plan fixes both by **changing who the output is for**, not by abandoning the modelling:

> The crowd **trains** the model. One person — you — **uses** it.

Train on real ListenBrainz listening history. The model learns a map of music: every artist gets
a position derived purely from who listens to whom, with no genre labels supplied. Then anyone
drops in their own history, gets placed on that map, and sees their own taste as a shape.

The output is a mirror, not a recommendation list.

### What this gives each goal

- **Useful.** It works for one person, with their own data, about themselves. No strangers.
- **A real DS project.** A model is genuinely trained, tuned on validation, and measured against
  baselines on a held-out question it could fail.

---

## Conflict with CLAUDE.md — read this first

`CLAUDE.md` specifies a recommender. This plan drops that. The spec file must be updated in the
same change, or the repo will contradict itself. Sections affected:

| Section | Status under this plan |
|---|---|
| §1 Data sources | **Unchanged and now actually honoured.** ListenBrainz is finally used for real. |
| §5 Pipeline rules | **Unchanged.** Dedupe, time split, train-only features all still apply. |
| §6 Modelling | **Rewritten.** ALS candidate generation, MMR re-ranking, personalised λ and familiarity anchors all go. Replaced by embedding training, fold-in and clustering. |
| §7 Evals | **Rewritten.** Recall/NDCG/Discovery Recall/ILD/Serendipity/anchor rate replaced by adoption-proximity metrics (below). Coverage and Gini are dropped; they measure a recommender's catalogue behaviour. The *discipline* — fixed seeds, bootstrap CIs, segment breakdowns, test split touched once — is kept exactly. |
| §8 Tests | **Mostly kept.** Dedupe, split-leakage and train-only-filter tests are unchanged. Re-ranking property tests go. New tests for fold-in and projection. |
| §9 UI | **Rewritten.** The discovery dial is no longer the hero; the map is. The quality floor, design tokens and `web/DESIGN.md` carry over unchanged. |
| §11 Attribution | **Unchanged and more important**, since real ListenBrainz data is now used. |

I will not edit `CLAUDE.md` without you saying so. Flagging it, per its own instruction to point
out conflicts rather than silently resolve them.

---

## Open decision: where the personal history comes from

I asked and do not have an answer, so the plan covers all three. This changes increment 7 only;
everything before it is unaffected, so work can start without deciding.

| Source | Effort | Notes |
|---|---|---|
| **Last.fm** | low | `user.getRecentTracks`, paginated, full history with timestamps and sometimes MBIDs. Needs a free API key. Best option by far. |
| **Spotify export** | low, but slow | Privacy settings → download your data. Arrives in a few days as JSON. No API, so nothing deprecated applies. |
| **CSV** | low | `artist, track, timestamp`. Universal fallback. |
| **None** | — | The tool still works: it demos on a *real* held-out ListenBrainz listener. Far better than the invented ones. |

I will build the CSV path plus the "real demo listener" path first, since they need nothing from
you, then add Last.fm or Spotify once you say which.

---

## Risk gate: RESOLVED — real data is in hand

Increment 1 ran. Findings, with sizes measured rather than assumed:

| Route | Size | Verdict |
|---|---|---|
| ListenBrainz full listens dump | **229 GB** | rejected, only 84 GB free |
| ListenBrainz incremental | 352 MB **per day** | rejected, a year is ~128 GB |
| ListenBrainz `sample/` dump | 235 MB | **metadata only, contains no listens** |
| **MLHD+ shard** | 15 GB per shard, 16 shards | **chosen** |
| MusicBrainz canonical dump | 2 GB | **chosen**, for artist names |

**Chosen source: MLHD+** (Music Listening Histories Dataset), published by MetaBrainz:
27 billion plays by 583,000 **Last.fm** listeners, mapped to MusicBrainz ids. One listener per
zstd file inside a sharded tar, lines of
`timestamp <TAB> artist_mbid <TAB> release_mbid <TAB> recording_mbid`.

It aligns with the Last.fm route chosen for personal history, so the training data and your own
data are the same shape.

**The 15 GB is not downloaded.** `detour/ingest/mlhd_loader.py` streams the tar over HTTP and
abandons the connection once enough listeners are collected, transferring a few GB instead.

**Proven on a 30-listener probe:** 351,875 real plays across 9,801 artists, full 2011 coverage,
median 6,856 plays per listener. Artist names resolve correctly (Rihanna, tagged
"pop, r&b, dance-pop").

TLS interception did not block any of this; `curl` and `urllib` both work against MetaBrainz.
`truststore` is therefore **not** needed and has been dropped from the dependency list.

**Window: calendar 2011**, twelve consecutive months, so the time split has a clean year to cut.

---

## What the model learns

**Input:** real listening events, split by time.

**Sessions.** A listener's plays are cut into sessions on a 30-minute gap (already specified in
`CLAUDE.md` §6). Each session is a sequence of artists — the equivalent of a sentence.

**Embedding.** `item2vec` (gensim `Word2Vec`) over those sequences. Artists that keep turning up
in the same sessions end up near each other. Nothing is labelled; genre structure emerges or it
does not, and whether it does is itself a finding worth reporting.

**Baselines, each a different way of building the same map:**

| Model | What it is |
|---|---|
| random | random positions — the floor |
| popularity | one dimension, play count only |
| co-occurrence PPMI + SVD | classic count-based embedding, no neural training |
| ALS item factors | matrix factorisation, reuses `detour/models/als.py` |
| **item2vec** | ours |

**Placing a person (fold-in).** A new listener was never in training. Their position is the
weighted mean of their artists' vectors, weighted by play count. Standard, cheap, and it means
the model generalises to people it has never seen — which is the entire point.

**Islands.** Cluster each person's artists in the full embedding space to find their distinct
taste groups. KMeans with a silhouette-selected k, which is explainable; HDBSCAN if density
clusters turn out to fit better.

**2D projection.** UMAP for display only.

> **Methodological rule, enforced in code:** every metric is computed in the **full embedding
> space**, never on the 2D projection. The projection is a drawing; distances in it are distorted
> by construction. Mixing these up is the classic way to produce a chart that lies.

---

## Evaluation: the question the model could fail

**The claim:** the map is meaningful — artists near your territory are ones you would actually go
on to play.

**The test — adoption proximity.** Train on months 1–9. For each held-out listener, take the
artists they first played in months 10–12. Ask: how close are those to the territory they already
occupied at the end of month 9?

**Primary metric — Adoption Rank Percentile.** For each adopted artist, rank every candidate
artist by distance from the listener's train position. Report where the real adoption landed, as
a percentile. 0.5 is chance. Lower is better. Averaged over adoptions, then over listeners.

**Secondary:**
- **Hit@K** — share of adopted artists inside the K nearest neighbours of the listener's artists.
- **Median rank** of adopted artists.
- **Cluster hit rate** — did adoption land in one of the listener's own islands, or outside?

### The confound that must be controlled

**Popular artists are adopted more often *and* sit centrally in embedding space.** A model that
learned nothing except popularity would score well on the naive test. This is the single biggest
threat to the result's validity.

Control: score every real adoption against **popularity-matched negatives** — artists with
similar global play counts that the listener did *not* adopt. If item2vec only beats random but
not the popularity baseline on matched negatives, the map adds nothing, and the report says so.

This control is the difference between a credible project and a flattering one.

### Protocol (carried over from CLAUDE.md §7, unchanged)

- Fixed seeds from config; 95% CIs by bootstrapping over listeners, 1,000 resamples.
- Segment by listener activity and by explorer score tercile; cold-ish listeners reported apart.
- **Test split touched once, at the end, only when you explicitly ask.**
- `evals/thresholds.json` is never edited to make a run pass.
- Results reported as measured, including if item2vec loses.

---

## What you see

Four screens. `web/DESIGN.md`, the tokens and the whole design system carry over; the palette's
familiar↔unknown logic still holds, since the map has a centre and edges.

1. **Your map** *(hero)* — the artist map, your territory lit up. Pan and zoom. Your artists sized
   by play count. The honest version of "here is your taste".
2. **Your islands** — discovered clusters, each with its characteristic artists, and how isolated
   each one is. Answers "am I actually eclectic, or do I just think I am".
3. **Your drift** — monthly position replayed as a path across the map, with distance travelled
   per month and the points where taste turned. This is where `explorer_score.py` survives, as the
   drift measure rather than a knob.
4. **Results** — the offline eval. Adoption rank percentile against every baseline, with CIs, and
   the popularity-matched control shown as a separate panel. Reads the latest report JSON, exactly
   as now.
5. **Method** — plain language, with the limitations stated as prominently as on the current page.

**Privacy:** uploaded history is processed locally and never leaves the machine. Stated in the UI,
not just in a file.

---

## Increments

Each ends with a validation you can see. Do not start the next until the current passes.

### 1 — Data access spike *(decision gate)*
Establish that real ListenBrainz data can actually be fetched here. Try the monthly dumps first
(no account), BigQuery second. Resolve the TLS interception properly rather than disabling
verification. Pull one small slice and read it.

**Validate:** a real Parquet file in `data/raw/` with genuine MBIDs and timestamps, plus a written
note on route, size and time taken. **If this fails, stop and report — do not continue on fake
data.**

### 2 — Ingest and sample
`detour/ingest/dumps_loader.py` for real. Sample to roughly 5,000–10,000 active listeners over 12
consecutive months per `CLAUDE.md` §1, written to Parquet with the JSON sidecar.

**Validate:** sidecar shows real row counts and date range; listener and artist counts are
plausible; re-running the sample with the same seed gives the same set.

### 3 — Clean and split at real scale
Reuse `detour/clean/run.py` and `detour/split.py` **unchanged** — they are already correct and
tested. Confirm they hold at a hundred times the data.

**Validate:** existing dedupe, split-leakage and train-only-filter tests still pass; stage
completes in reasonable time and memory; sidecars look sane.

### 4 — Sessions and embeddings
`detour/features/sessions.py` (30-minute gap) and `detour/models/item2vec.py` for real, plus the
PPMI+SVD baseline. Reuse `als.py` for its factors.

**Validate:** nearest neighbours of a few well-known artists are inspected by hand. If the
neighbours of a famous metal band are not other metal bands, the embedding is wrong and no metric
will save it. This eyeball check comes before any number.

### 5 — Fold-in and clustering
Place an unseen listener from their history. Cluster their artists into islands.

**Validate:** property tests — fold-in of a listener who plays exactly one artist lands on that
artist; adding plays of an artist moves the position toward it; islands are stable across seeds.

### 6 — Evaluation
`detour/eval/` re-pointed: adoption rank percentile, Hit@K, median rank, cluster hit rate, all
baselines, bootstrap CIs, **and the popularity-matched negative control**. Report JSON and
Markdown as now.

**Validate:** golden tests with hand-computed arithmetic for each metric; two runs at one seed are
identical; a deliberately shuffled embedding scores at chance (0.5) — if it does not, the metric
is broken.

### 7 — Bring your own history
CSV import plus a real demo listener, then Last.fm or Spotify once chosen. Artist name resolution
to the trained vocabulary, with coverage reported honestly ("recognised 847 of your 1,021
artists") rather than silently dropping the rest.

**Validate:** API contract tests; a known CSV resolves to the expected artists; unmatched names
are surfaced, never hidden.

### 8 — The map UI
`web/` rebuilt on the existing design system. Canvas or WebGL for the map, since thousands of
points will not survive as SVG DOM nodes. Keyboard access and a non-visual route to the same
information — a map must not be the only way to read the data.

**Validate:** `pnpm typecheck && pnpm test`; Playwright screenshots at 360px and desktop critiqued
against `web/DESIGN.md`; axe scan clean; the whole flow usable by keyboard.

### 9 — Honest write-up
Results and Method pages reflecting what was actually found, including the popularity control's
verdict and anything that failed.

**Validate:** every claim on the Results page traceable to a number in the report JSON.

---

## What survives from v1

**Kept as is** — roughly half the Python and all the frontend infrastructure:

| Path | Role |
|---|---|
| `detour/clean/run.py` | dedupe, artist identity — unchanged |
| `detour/split.py` | time split, train-only filters — unchanged |
| `detour/features/popularity.py` | drives the popularity baseline and the confound control |
| `detour/features/explorer_score.py` | becomes the taste-drift measure |
| `detour/models/als.py`, `matrix.py` | ALS factors as a baseline embedding |
| `detour/config.py`, sidecars | unchanged |
| `api/`, `run.py`, `scripts/create_project.py`, `Makefile` | structure unchanged |
| `web/DESIGN.md`, `tokens.css`, `base.css`, states, tables | design system carries over wholesale |
| `tests/unit/test_clean.py`, `test_split.py`, `test_report.py` | still valid |

**Replaced:**

| Path | Why |
|---|---|
| `tests/fixtures/make_fixtures.py` | **demoted to test fixtures only.** Keeps the suite fast and offline. It stops being the product's data. |
| `detour/rerank/` | deleted — no list to re-rank |
| `detour/models/popularity.py`, `random.py`, `itemknn.py` | deleted as recommenders; popularity returns as an embedding baseline |
| `detour/eval/metrics.py`, `harness.py` | rewritten for adoption proximity |
| `web/src/screens/Listen.tsx`, `DiscoveryDial`, `TrackList` | replaced by the map |

---

## Risks, honestly

| Risk | Severity | Response |
|---|---|---|
| Real data unobtainable here (TLS, credentials, dump size) | **high** | Increment 1 is a gate. Stop and report rather than fake it. |
| Popularity confound makes a null result look positive | **high** | Matched-negative control built into the metric from the start, not bolted on. |
| Artist name resolution from personal exports is messy | medium | MBIDs where available, normalised names otherwise; report coverage openly. |
| Embedding quality poor on a small sample | medium | Increment 4's eyeball check catches this before metrics are trusted. |
| Map unreadable with thousands of points | medium | Canvas rendering, density-aware labelling, zoom. |
| Scope is larger than v1 | medium | Increments are independently valuable; stop at any boundary with something that works. |

**New dependencies, each justified:** `gensim` (already present, now actually used), `scikit-learn`
(clustering, nearest neighbours, silhouette), `umap-learn` (2D projection), `truststore` (the TLS
issue above). Nothing else without asking.

---

## Verification

```bash
uv run python run.py setup
uv run python -m detour.ingest.dumps_loader --months 12 --users 8000   # increment 1-2
uv run python run.py pipeline
uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run mypy detour api
uv run python run.py eval
uv run python run.py demo
pnpm --dir web typecheck && pnpm --dir web test && pnpm --dir web e2e
```

**Done means:** a model trained on real listening data; a map whose nearest neighbours make sense
to a human; an evaluation that reports the adoption question against every baseline *and* survives
the popularity control; and a person able to drop in their own history and see their own taste.

---

## Appendix: v1, as built

Complete and passing, kept in history and partly reused.

- Idempotent scaffold, `run.py` as the single entry point (`make` and `pnpm` were absent).
- Pipeline: synthetic fixtures → dedupe → 9/1/2 time split → train-only features → ALS, item-kNN,
  popularity, random → MMR re-rank with personalised λ.
- Eval: six baselines, λ sweep, segment breakdown, JSON and Markdown reports.
- FastAPI serving pre-computed artefacts; React 19 + Vite + visx frontend, four screens.
- 82 Python tests, 19 frontend tests, ruff, mypy and typecheck all clean.

**Findings worth carrying forward:**

1. Fixtures without latent taste made collaborative filtering impossible — popularity beat ALS
   until users were given taste vectors. *Synthetic data cannot validate a modelling claim.*
2. Even after that fix, the personalised λ **reduced** discovery recall (−0.0394). Rewarding
   globally rare items is not the same as finding what someone will actually adopt. **This lesson
   is the direct reason the v2 metric measures proximity to real adoptions instead of rarity.**
3. `latest()` sorted reports by filename, and the random run id made that non-chronological — the
   UI served a stale report. Fixed, with a regression test.
