# Detour — 5 minute presentation script

**Speaker A: 2 min · Speaker B: 2 min · Live demo: 1 min**

---

## SPEAKER A — Part 1: What the project does (0:00 – 2:00)

**(0:00 – 0:30) The idea**

> "Our project is called Detour. It builds a map of music.
>
> Imagine a huge hall with 22,000 musicians standing in it. Nobody told them where to
> stand. They arranged themselves by who their fans are. Artists whose listeners overlap
> drifted close together. So all the metal bands ended up in one corner, all the jazz
> players in another.
>
> And here is the key point — **we never gave the model a single genre label.** Not one.
> The only fact it ever saw was: which artists share listeners."

**(0:30 – 1:10) What you get out of it**

> "Then you walk in. You drop in your own listening history, and we switch on a light
> above every artist you play.
>
> Now you can see three things in one glance:
>
> - **How wide your taste is** — all your lights in one corner, or scattered across the hall.
> - **Your islands** — most people have two or three separate tastes and never noticed.
>   Someone might have a Hindi clump, a Punjabi clump and a tabla clump. Three tastes,
>   one person.
> - **And the useful one** — the artists standing right next to you that you have never
>   played. Those are the recommendations."

**(1:10 – 2:00) Why this is different**

> "Normally a recommender predicts what some stranger will click next. Ours is a mirror,
> not a playlist. The crowd *trains* the model — one person *uses* it.
>
> One honest warning about how it thinks. The map does not know what 'Punjabi' means. If a
> Kashmiri singer stands next to your Punjabi artists, it is for exactly one reason:
> **a lot of people listen to both.** If nobody in the data listened to both, they stand
> far apart — even if a human would say they are obviously related.
>
> Artists are neighbours when they share fans, not when they share a genre. Over to
> [Speaker B] for the data and the models."

---

## SPEAKER B — Part 2: Data, stack, models, results (2:00 – 4:00)

**(2:00 – 2:30) The dataset**

> "Our data is real. **14.8 million actual plays by 2,000 real Last.fm listeners**, across
> the whole of 2011. It comes from MLHD+, a research dataset published by MetaBrainz, with
> artist names from MusicBrainz.
>
> That's about 22,000 artists on the final map. Nobody is invented — every play actually
> happened.
>
> The full dataset is a 229 GB download, which we could not store. So we stream it over
> HTTP and cut the connection once we have enough listeners. A few GB instead of 229."

**(2:30 – 3:00) The tech stack**

> "Python 3.11 with uv. **DuckDB and Parquet** for the data pipeline — 14.8 million rows
> clean and split in under 40 seconds. **scikit-learn and numpy/scipy** for the model,
> **gensim** for one of the baselines, **UMAP** to flatten the map to 2D for drawing.
>
> Serving is **FastAPI**, which only loads pre-computed files — no training when you click.
> The frontend is **React 19, TypeScript and Vite**, with the map drawn on a canvas so it
> stays smooth with 22,000 dots."

**(3:00 – 3:30) How we tested it — the part that could have failed**

> "Now the important bit: the test. We hid the **last three months** of everyone's
> listening. We placed each person on the map using only their first nine months. Then we
> asked — where do the artists they really went on to discover actually sit?
>
> We report it as a percentile. **Random guessing is 0.5. Lower is better.**"

**(3:30 – 4:00) The comparison table**

> "We tried five ways of building the same map."

| Model | Score | Popularity-matched control |
|---|---|---|
| Random | 0.4965 | 0.4962 |
| Popularity only | 0.5115 | not meaningful |
| ALS (matrix factorisation) | 0.3935 | 0.3793 |
| item2vec (neural) | 0.2891 | 0.2282 |
| **PPMI + SVD ← the map** | **0.1158** | **0.1880** |

> "Three things to notice.
>
> **One — random scored 0.4965 against a theoretical 0.5.** That sanity check passing is
> what makes every other number here worth reading.
>
> **Two — the neural model lost.** item2vec was our first choice and scored 0.289. A much
> older counting method — count who listens to what, then do linear algebra — scored 0.116,
> more than twice as good. So we threw away the neural network and kept the simple one. We
> reported that instead of burying it.
>
> **Three — the trap we checked for.** Popular artists get discovered more anyway, *and*
> they sit centrally on any map. So a model that secretly only learned 'what is popular'
> would look brilliant without being clever. We tested again, comparing every real
> discovery **only against artists of similar popularity**. It still won: 0.188 against a
> chance of 0.5. **The map learned taste, not the charts.**"

---

## DEMO (4:00 – 5:00) — either speaker drives

> **(4:00)** "This is the map. Every grey dot is one of 22,000 artists. Nobody placed
> them — they sorted themselves by shared listeners."

> **(4:15)** *(load a listener)* "Teal dots are the artists this person plays — bigger dot
> means more plays. You can see the shape of their taste immediately."

> **(4:30)** *(islands screen)* "The model found their separate islands by clustering.
> Those are distinct tastes inside one person."

> **(4:40)** *(zoom to a cluster)* "Red rings are artists standing right beside them that
> they have never played. Those are the recommendations."

> **(4:50)** "And a quick proof it understood music with zero genre labels: Miles Davis
> landed beside Coltrane, Monk, Mingus and Charlie Parker. Metallica beside Iron Maiden,
> Megadeth and Pantera. It worked that out purely from who listens to whom."

> **(4:58)** "Limits, briefly: the data is 2011, it is mostly Western music, and it has
> never heard a single song. Thank you."

---

## Two notes for honesty if you're asked

- **"Did you tune on the test set?"** — No. Everything was tuned on the validation month.
  The test split has not been touched.
- **"What's missing?"** — Confidence intervals aren't computed yet, and segment breakdowns
  by listener activity are still outstanding. Say so if asked; it's in `plan.md`.
