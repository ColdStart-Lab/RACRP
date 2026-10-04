# The window was never derived

## The observation

Every keep-alive policy this project has measured — including the deployed
ATC'20 one, and including our own conditional method that beat it by 26% —
decides *how long to hold an idle container* by reading a **percentile** off an
idle-gap histogram. ATC'20 uses the 5th percentile for when to bring the
container back and the 99th for when to let it go. Our conditional method swept
the second percentile as its cost knob.

Nobody ever asked where those numbers come from. They are a heuristic. There is
no argument that the 99th percentile is right, and — worse — a percentile is not
a quantity anyone operating a platform can price. An operator knows what a cold
start costs them relative to a GB-minute of memory. They do not know what
percentile that implies.

## The decision is an optimal-stopping problem

A container warm over the window `[a, a+L)` after an application goes quiet pays
memory for however much of that window the idle gap actually covers, and pays a
cold start unless the next invocation lands inside it. Writing `g` for the gap
and `m`, `K` for the price of a warm minute and of a cold start:

```
C(a, L)  =  m · E[ max(0, min(a+L, g) − a) ]  +  K · P(g < a  or  g ≥ a+L)
```

This is minimised directly, over a small grid of offsets, each in `O(range)`
by telescoping the memory term. The percentile knob disappears. What is swept
instead is `K/m` — how many GB-minutes of memory one cold start is worth — which
is an economic quantity, not a distributional artefact.

## Why the elegant version of this is wrong

Setting `∂C/∂L = 0` for the `a = 0` case gives a genuinely beautiful identity:

```
hazard(T)  =  m / K
```

*Drop the container at the first moment its hazard rate falls below the price
ratio.* It is the kind of result one wants to put on a slide.

**We implemented it and it was catastrophic**, and the reason is worth recording.
The first-order condition locates a *stationary point*, and `C(T)` is not
unimodal for real idle-gap distributions. A perfectly periodic application has
zero hazard everywhere except at its period — so the first-crossing rule stops at
minute 1 and cold-starts on *every single invocation*. Measured; it did exactly
that.

The fix is unglamorous and exact: take the global argmin over the grid. It costs
`O(range)`, which is the same order as evaluating the identity. The identity
remains the right way to *understand* the rule; it is not the algorithm. This is
now pinned by a test against brute-force search.

## Three further things that were wrong before this worked

1. **`head_margin=0.0` did not disable the pre-warm.** The 0th percentile is the
   shortest gap ever observed, which is routinely non-zero. Every earlier
   "derived window, no pre-warm" arm was therefore a different policy than the
   one described.

2. **The derived length does not compose with a pre-warm offset.** `L*` is the
   optimum for holding *continuously from the start of the gap*. Shifting that
   window by an offset it was not optimised for destroys it: −47% with the
   offset against +65% without, on the same applications.

3. **Optimising the length alone concedes the comparison.** Releasing the
   container and bringing it back is a strictly richer action than holding from
   the start, and it is the action the deployed policy actually takes. So the
   pair has to be optimised jointly. On a periodic application the joint
   optimiser recovers exactly the right answer — release, return at the period,
   hold for one minute — which the length-only version cannot express.

## What it says about the project's earlier result

The conditional histogram found gain by **using more of the application's
state**. The derived window finds gain by **using more of the distribution's
shape**: a percentile pair sees two points of a curve and throws away everything
between them, and it is exactly that discarded shape the cost model integrates
over.

Both are the same complaint — the deployed policy discards information it has —
and it turns out the second discard is the larger one.

They also do not simply add up, and the reason is the interesting part. See
`EXPERIMENT_LOG.md` E18.

## A fifth thing, found by drawing the figure

The figure comparing the two windows on a single application was annotated with
each window's expected cost under the very cost model the optimiser minimises.
The annotation said the derived window cost **more** — 0.311 against the
baseline's 0.238. An optimiser that loses to the thing it is optimising against,
on its own objective, is broken.

It was the prior. `smooth=1.0` added one pseudo-count *per bin*, and the support
is 241 minutes, so it added 241 pseudo-observations against that application's
992 real gaps. A 24% uniform prior asserts that a quarter of an application's
idle gaps are spread evenly out to four hours, which is false for every
application in this trace. It dragged mass into a tail that is not there and
made the optimiser give up early: a 9-minute window where the true optimum was
21, at 38% higher true cost.

`smooth` is now total pseudo-mass spread over the support.

### The part that was tempting to leave out

The broken setting **scores better on the test period**: +74.3% against +69.3%
for a weak prior, on the same 250 applications. It would have been the easiest
thing in the world to keep it, call the per-bin form a design choice, and report
the larger number.

So the prior strength was swept on the **validation** days instead:

| total pseudo-mass | validation median gain |
|---|---|
| 0 | +61.7% |
| 1 | +63.6% |
| 5 | +69.0% |
| **25** | **+70.3%** |
| 60 | +65.8% |
| 120 | +61.5% |
| 241 *(the old per-bin behaviour)* | +53.5% |

Validation picks 25, and ranks the setting that flatters the test score **last**.
Choosing on the test period would have selected the worst-generalising member of
the grid and attached the biggest number to it. `DEFAULT_SMOOTH = 25.0` is
carried to the test days unchanged, and a test pins the constant to the recorded
validation choice so it cannot drift back.
