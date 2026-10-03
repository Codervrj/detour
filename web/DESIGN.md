# Detour — design

The accepted design plan. Every later UI change must stay consistent with this document.
Tokens live in `web/src/styles/tokens.css`; this file explains why they are what they are.

## Brief

- **Subject:** music discovery — finding an artist you didn't know you'd love.
- **Audience:** examiners, recruiters and curious listeners, seeing the project cold.
- **Primary job:** make the familiarity-versus-discovery trade-off something you can feel by
  moving one control.

## The one idea

**The dial and the list share a single coordinate system.**

Every track carries a marker on the same familiar→unknown axis the dial sits on. Turning the
dial makes the whole constellation visibly drift outward. The list *is* the chart.

This is where the boldness is spent. Everything else stays quiet.

The alternative we rejected was album-art tiles, which this project cannot honestly show
(the data is synthetic and has no artwork) and which would have produced a Spotify pastiche.
The characteristic thing in this subject's world is not cover art, it is **distance travelled
from what you already know**.

## Colour

The palette is diverging because the *data* is diverging. Familiar sits at one end, unknown at
the other, and every colour decision descends from that.

| Token | Value | Role |
|---|---|---|
| `--paper` | `#eef1f3` | page ground, a cool grey-blue |
| `--surface` | `#f7f9fa` | raised surfaces, masthead |
| `--surface-sunk` | `#e4e9ec` | inset chips, code |
| `--ink` | `#16222b` | body text — the darkest value of the teal family |
| `--ink-soft` | `#3d4f5a` | secondary text |
| `--mid` | `#6b7d87` | axes, captions |
| `--rule` | `#cdd6db` | hairlines |
| `--familiar` | `#0f6e72` | deep teal — artists they know |
| `--unknown` | `#c2365b` | deep rose — artists they don't |
| `--focus` | `#0b5cc4` | focus ring, deliberately outside the data hues |

Dark mode re-declares the same tokens under both `prefers-color-scheme` and
`[data-theme="dark"]`, brightening the two data hues so they hold contrast on a dark ground.

`--ink` is a blue-black rather than a neutral near-black on purpose: it is the bottom of the
teal ramp, so text and the "familiar" data colour belong to one family.

## Type

- **Display — Bricolage Grotesque.** Variable, slightly condensed, idiosyncratic. Carries the
  personality in headings and the wordmark.
- **Body — Instrument Sans.** Clean grotesque with tabular figures, so metric columns do not
  jitter as values change.

Scale, base 16px at ratio 1.25: `0.8 / 0.9 / 1 / 1.25 / 1.5625 / 1.953 / 2.441rem`.
Body measure capped at 68ch.

The wordmark sets "De" in ink and "tour" in the unknown rose — the second half steps away from
the first, which is the product in two syllables.

## Layout

Left-aligned throughout. Content in a 68rem column with a 16px gutter.

Recommendations are **continuous hairline-separated rows over a shared axis**, never cards.
A card grid would have chopped the list into identical boxes and destroyed the one thing that
matters: that the markers line up with each other and with the dial.

```
Pick a listener   [user-000][user-005][user-013]...[user-049]
user-049 has 288 listens and roams, scoring 0.40.

How far out should we go?                        0.38
familiar |---------●-------------------| unknown
              ▲ their own setting

What we'd play next            7 of 20 by artists they already play
                                      familiar          unknown
1  Track 1 by Artist 0013    ───●──────────   you know them   why this
2  Track 0 by Artist 0240    ──────────●───   new to you      why this
```

## Motion

One orchestrated moment, and it answers a user action: when the dial moves, markers transition
to their new positions (`240ms`) while the outgoing list dims rather than blanking, so the
markers appear to *travel*. No scroll-triggered reveals, no hover animation on rows.
`prefers-reduced-motion` collapses the duration to `1ms` at the token level.

## Quality floor

- Responsive to 360px with no horizontal page scroll. Below 52rem the axis column is dropped
  and the word tags ("you know them" / "new to you") carry the same information.
- Nothing is conveyed by colour or position alone: every marker has a visually-hidden reading
  of its novelty, and every row carries a word tag.
- Visible focus ring in a hue outside the data palette.
- Loading, empty and error states on every data view; each error says the exact command to run.
- WCAG AA contrast in both themes.

## Defaults deliberately avoided

Checked against the usual generated-design tells:

- Not cream `#F4F1EA` with a serif display and a terracotta accent.
- Not near-black with a single acid accent, and not the dark-UI-plus-green music-app look.
- Not a broadsheet of hairline rules and zero radius.
- Not a SaaS card kit — no identical rounded cards, no uniform drop shadow, no gradient washes
  as decoration.
- No tracked-out all-caps eyebrow labels, no meta strings joined with middle dots, no `→`
  appended to buttons, no monospace face for small data labels.

## Screens

1. **Listen** — the hero and the default route. Listener picker spread evenly across the
   explorer range (showing all fifty in order would put eight identical 0.00 entries on
   screen), the dial, and the re-ranked list with per-track familiarity and novelty plus a
   "why this" disclosure.
2. **Listeners** — the explorer scale as a single axis with every listener a tick on it, then
   the selected listener in detail.
3. **Results** — the accuracy-versus-novelty frontier drawn with visx scales rather than a
   chart component, baselines against ours, and the segment breakdown. Reports the numbers as
   measured, including the run where our discovery recall fell below plain ALS.
4. **Method** — plain-language data, split, metrics and limitations, plus attribution.
