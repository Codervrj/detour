# Detour, explained simply

## Imagine a huge party

Picture an enormous hall with **21,951 musicians** standing in it.

Nobody told them where to stand. They arranged themselves by **who their fans are**.

Musicians whose fans overlap drifted together. So all the metal bands ended up in one
corner. All the jazz players in another. The pop singers somewhere else.

Nobody wrote the labels "metal" or "jazz". The crowd sorted itself.

**That hall is the map. Every dot is one musician standing in it.**

---

## Now you walk in

We switch on a light above every musician **you** listen to.

Look across the hall:

- **All your lights in one corner?** Your taste is narrow.
- **Lights scattered everywhere?** Your taste is broad.

You can see it in one glance. A playlist can never show you this.

---

## What the colours mean

| On screen | What it is |
|---|---|
| **Grey dots** | Every musician in the hall. All 21,951. |
| **Teal dots** | The ones **you** listen to. Bigger dot = you play them more. |
| **Red rings** | Musicians standing **right next to you** that you have **never heard**. |

**The red rings are the recommendations.**

---

## A real example

Say you listen to 10 artists:

- some **Hindi** singers
- some **Punjabi** singers
- some **tabla** players

The map will show you **three islands** — three separate clumps of light. Three different
tastes, inside one person.

Now suppose a **Kashmiri** singer is standing right beside your Punjabi clump, and you have
never played them.

**That is your recommendation.**

### But here is the important bit

The map does **not** know what "Punjabi" or "Kashmiri" means. It has never heard of
languages, regions or genres.

That Kashmiri singer is standing next to your Punjabi group for one reason only:

> **lots of people listen to both.**

If nobody in the data listened to both, they would stand far apart — even though a human
would say they are obviously related.

**Artists are neighbours when they share fans, not when they share a genre.**

---

## So what does it actually tell you?

**1. How wide is my taste?**
One corner, or the whole hall.

**2. How many different tastes do I have?**
Most people have two or three and never noticed.

**3. What are my strange corners?**
The few artists you play that stand far from everything else you play.

**4. What should I listen to next?** ← the useful one
The nearest musicians you have never heard. Not because a company paid for it. Because
they stand where your people stand.

---

## Does it actually work?

We tested it in a way it could have failed.

We hid the **last three months** of everyone's listening. We placed each person on the map
using only their earlier listening. Then we asked: *where do the artists they really went
on to discover sit?*

| | Score |
|---|---|
| Random guessing | 0.50 |
| **Our map** | **0.12** |

Lower is better. Real discoveries landed **far closer** to people than chance allows.

### The trap we checked for

Popular artists get discovered more anyway. So a map that secretly just learned "what is
popular" would look good without being clever.

So we tested again, comparing each real discovery **only against artists of similar
popularity**. It still won.

**The map learned taste, not the charts.**

### The honest bit

We first used a neural network model called item2vec. It scored **0.289**.

Then we tried a much simpler method from the 1970s — count who listens to what, do some
linear algebra. It scored **0.116**, more than twice as good.

So we **threw away the neural network** and kept the simple one.

---

## Proof it understood music

We never told it a single genre. Yet:

- **Miles Davis** ended up beside Coltrane, Monk, Mingus and Charlie Parker — the whole
  jazz canon
- **Metallica** beside Iron Maiden, Megadeth and Pantera
- **Daft Punk** beside Justice and The Chemical Brothers
- **Beyoncé** beside Rihanna, Usher and Lady Gaga

It worked all of that out purely from who listens to whom.

---

## Who is this for?

**A curious listener.** Someone who wants to see the shape of their own taste, not just get
handed another playlist.

**Anyone studying how recommendations work.** It shows the whole chain: real data in, a
model that genuinely learns, and an honest test at the end.

---

## What it cannot do

- **The music is from 2011.** Nothing released since exists on this map.
- **It is mostly Western music.** There is almost no Indian music in the data, so an Indian
  listener's lights would barely show up. The method would work fine on Indian listening
  data — that data just is not publicly available.
- **It has never heard a single song.** It knows nothing about melody, rhythm or language.
  Every connection comes from who listened to what.
- **It needs your listening history.** Without a few hundred plays, there is nothing to place.

---

## Where the data came from

**14.8 million real plays by 2,000 real Last.fm listeners**, across the whole of 2011, from
a research dataset published by MetaBrainz called MLHD+.

Nobody is invented. Every artist is real and every play actually happened.

---

## In one sentence

> **It shows you the shape of your music taste, and the nearest good thing you have not
> heard yet.**
