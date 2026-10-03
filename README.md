# Detour

A map of music learned from how people actually listen, and a way to see where your own
taste sits on it.

There are no genre labels anywhere in this project. Every relationship on the map was
derived from one fact: which artists share listeners.

## What it does

Trained on **14.8 million real plays by 2,000 Last.fm listeners** across 2011, the model
places 21,951 artists in a 128-dimensional space. Artists that share listeners end up near
each other, so Miles Davis lands beside Coltrane, Monk and Mingus without anyone having
told it what jazz is.

You were never in that training data. Drop in your own listening history and you get folded
onto the map, and can see:

- **where your taste sits** against the whole catalogue
- **your islands** — the distinct corners of your taste, discovered by clustering
- **your edges** — your own artists furthest from your centre
- **what is closest to you that you have never played**

## Run it

```bash
uv run python run.py setup       # install Python and web dependencies
uv run python run.py pipeline    # fetch real data, build the map  (~15 min first time)
uv run python run.py demo        # serve the API and the web app
```

Then open http://127.0.0.1:5173.

The pipeline needs no credentials. It streams a slice of MLHD+ over HTTP and stops once it
has enough listeners, so it transfers a few GB rather than the full 16 GB shard.

## See yourself on the map

```bash
uv run python run.py lastfm --user YOUR_LASTFM_NAME
```

You need a free Last.fm API key in `.env` (see `.env.example`). Only the key is needed;
reading public scrobbles requires no login. Your history is processed locally and never
leaves the machine.

## Does it work?

Yes, and it is measured. Each listener is placed from their first nine months only, every
artist is ranked by distance from that position, and we check where the artists they really
went on to play landed. Chance is 0.5.

| model | adoption percentile | popularity-matched control |
|---|---|---|
| random | 0.4965 | 0.4962 |
| popularity only | 0.5115 | not meaningful |
| ALS factors | 0.3935 | 0.3793 |
| item2vec | 0.2891 | 0.2282 |
| **PPMI + SVD (the map)** | **0.1158** | **0.1880** |

The control matters more than the headline: popular artists get adopted more often *and*
sit centrally on any map, so each real adoption is also compared only against artists of
similar global popularity. The map survives that.

**The neural model lost.** item2vec was the first choice and scored 0.2891; counting
co-listening and taking an SVD scored 0.1158 with seven times the hit@10, so the simple
method became the map. Reported rather than buried.

## Commands

| Command | What it does |
|---|---|
| `run.py setup` | install dependencies |
| `run.py fetch` | download real listening history from MLHD+ |
| `run.py pipeline` | clean, split, build the map, project to 2D |
| `run.py lastfm --user X` | import your own history |
| `run.py eval` | score the map against every baseline |
| `run.py api` / `run.py web` | serve either half |
| `run.py demo` | serve both |
| `run.py test` / `run.py lint` | pytest; ruff and mypy |

`make` is not required; `run.py` is the entry point and the `Makefile` forwards to it.

## Layout

- `detour/ingest/` — MLHD+ streaming, Last.fm import, artist names, name resolution
- `detour/models/` — the map (PPMI+SVD), baselines, 2D projection
- `detour/eval/` — the adoption metric and its popularity control
- `detour/foldin.py`, `detour/mapservice.py` — placing a person, serving the map
- `api/` — FastAPI over pre-computed artefacts
- `web/` — React frontend; design decisions in `web/DESIGN.md`

See `plan.md` for the build order and `CLAUDE.md` for the original specification.

## Limitations

- Listening data is from 2011; nothing released since is on the map.
- Confidence intervals are not computed yet.
- Artists with fewer than five listeners in the training year are excluded.
- It knows nothing about how music sounds.

## Attribution

- Listening histories: **MLHD+**, MetaBrainz Foundation, research use, derived from Last.fm
- Artist names and identifiers: **MusicBrainz**
- Data model: **ListenBrainz** (CC0)
