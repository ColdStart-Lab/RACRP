# Method

Both algorithms, from first principles. Implementation lives in
`racrp/baselines/conditional.py` and `racrp/baselines/hazard.py`.

---

## 0. The decision

An application goes quiet at minute 0. The platform must choose a **warm
window** `[a, a+L)`, measured in minutes since the application went quiet:

- release the container immediately,
- bring it back at minute `a` (the **pre-warm offset**),
- hold it until minute `a+L` (the **keep-alive length**), then release it.

A request arriving inside the window is served warm. A request arriving outside
it pays a **cold start**. Memory is paid for every minute the container is held
while the application is still idle.

Writing `g` for the idle gap, `m` for the cost of one warm minute and `K` for the
cost of one cold start:

```
memory paid       =  max(0, min(a+L, g) − a)
cold start paid   =  1  if  g < a  or  g ≥ a+L
```

Every policy in this repository — including the deployed baseline — is a rule for
choosing `(a, L)` from an estimate of the distribution of `g`.

## 1. The baseline: ATC'20 (`racrp/baselines/histogram.py`)

Shahrad et al. estimate one histogram of `g` per application, pooling every gap
it has ever shown, and read the pair off two fixed percentiles:

```
a  =  P5  of the histogram
L  =  P99 − P5
```

with an ARIMA-style fallback for applications whose coefficient of variation is
too high for the histogram to be trusted. This is reimplemented faithfully and is
the thing both methods below are measured against.

Two observations motivate the rest of the work, and they are independent:

- **the histogram is a marginal** — it forgets the state each gap occurred in;
- **the percentiles are a heuristic** — nothing argues for the 99th, and a
  percentile is not a quantity any operator can price.

These are separate defects in separate places. Method A fixes the first, Method B
the second, and §4 explains why fixing both at once is worse than fixing the
second alone.

---

## 2. Method A — state-conditioned histogram

`racrp/baselines/conditional.py` · `conditional_policy(...)`

### The estimator

Instead of one histogram per application, keep one per **state**, where the state
is a coarse bucket of the gap that just closed. An application with bursty
traffic has short gaps inside a burst and long gaps between bursts; one pooled
histogram describes neither.

Splitting data into cells costs evidence, so each conditional histogram is
**partially pooled** toward the application's own marginal:

```
ĥ(g | b)  =  ( n_b · h_b  +  κ · h_marginal ) / ( n_b + κ )
```

- `n_b` — gaps observed in state `b`
- `κ` — the pooling weight, read as "the marginal is worth κ observed gaps"

Two properties follow directly and both are tested:

- **`κ → ∞` recovers ATC'20 exactly.** Method A is a strict generalisation along
  the conditioning axis, not a competing heuristic.
- **A state with no evidence falls back to the marginal**, so a thin cell is
  never trusted before it is earned.

### Pooling happens in quantile space

Blending two histograms as densities produces a **mixture**, which inflates the
tail: pooling a "5-minute" application with a "200-minute" one yields a
distribution with mass at both, and the 92nd percentile lands at 200. Blending
in **quantile** space — a 1-D Wasserstein barycenter — gives 102 instead, which
is what "halfway between" should mean. The implementation blends quantiles.

### Choice of conditioning variable

Not assumed — measured. See §5.

### Result

**+26.3%** median cold-start reduction at matched cost, with a random-key control
at **−1.4%** (`artifacts/tables/conditional_matched_cost.json`).

---

## 3. Method B — the derived window  ← the headline

`racrp/baselines/hazard.py` · `optimal_window(...)`, `hazard_policy(...)`

### The objective

Take the cost from §0 in expectation over the estimated gap distribution:

```
C(a, L)  =  m · E[ max(0, min(a+L, g) − a) ]  +  K · P(g < a or g ≥ a+L)
```

and minimise it over the pair. Dividing through by `K`, the only free quantity is
the ratio `m/K`; its reciprocal, **`K/m`, is how many GB-minutes of idle memory
one cold start is worth** — an economic quantity a platform can quote, replacing
a percentile that nobody can.

### Why the elegant version is wrong

Setting `∂C/∂L = 0` for the `a = 0` case gives

```
hazard(T)  =  m / K
```

*release the container the moment its hazard rate falls below the price ratio.*
It is a genuinely beautiful identity and it is **not the algorithm**.

The first-order condition locates a *stationary point*, and `C` is not unimodal
for real idle-gap distributions. A perfectly periodic application has zero hazard
everywhere before its period, so a first-crossing rule stops at minute 1 and
cold-starts on **every single invocation**. This was implemented, measured, and
did exactly that.

The global minimum over the grid is the same order of work and is what the cost
model actually asks for. `tests/test_hazard.py` pins the optimiser against a
brute-force search of its own objective.

### Why the pair, and not just the length

Releasing the container and bringing it back is a strictly richer action than
holding from the start, and it is the action the deployed policy already takes.
Optimising `L` alone concedes the comparison.

It also cannot be bolted on afterwards: `L*` is optimal for holding
*continuously from the start of the gap*, and shifting that window by an offset
it was not optimised for destroys it — measured at −47% with the offset against
+65% without. So the pair is optimised **jointly**.

### Complexity

For each candidate offset `a`, the memory term telescopes:

```
mem(L)  =  mem(L−1)  +  P(g ≥ a+L)
```

so a full sweep over `L` is `O(range)` per offset rather than `O(range²)`. With a
handful of candidate offsets drawn from the distribution's own quantiles, the
whole window is recomputed in ~0.13 ms — cheap enough to run online, which it
does: the policy recomputes as new gaps close.

### The prior

The derived window integrates over the whole distribution, so it is sensitive to
how the sparse tail is smoothed in a way a percentile rule is not. `smooth` is
**total** pseudo-count mass spread over the support, not a per-bin count — the
distinction matters, because the support is 241 minutes and a per-bin count would
add 241 pseudo-observations against an application's few hundred real gaps.

Its value was chosen on the **validation** days and carried to test unchanged.
See `docs/RESULTS.md` for why that choice mattered more than such choices usually
do.

### Result

**+78.8%** against the deployed policy, **+67.6%** against a baseline allowed to
tune both of its percentiles over 36 configurations, and **−427.5%** for the same
optimiser given another application's history.

---

## 4. Why the two do not compose

| | median at matched cost |
| --- | --- |
| Method A alone | +26.3% |
| Method B alone | **+78.8%** |
| Both | +68.6% |

Together they are worse than B alone, and far more variable (range dips to
−20.7% where B alone never falls below +15.6%).

The reason is mechanical. A percentile is a **robust statistic**: it survives
sparse counts, because it only needs to know where a mass boundary falls. A
cost-optimal window is not robust in that sense — it integrates over the entire
shape of the distribution, and is most sensitive precisely in the rare long tail.
Method A fragments the counts that estimate that tail.

Each idea buys information from a different place, and one of them spends the
resource the other depends on. This was not predicted; it is reported because it
is what the measurement says.

---

## 5. Choosing the mechanism by measurement

`scripts/run_information_audit.py`

Before building any policy, we measured how predictable the problem is *at all*,
model-free, via mutual information between each observable context and the next
idle gap. Over 264 applications:

| context | share of the gap's entropy explained |
| --- | --- |
| previous gap length | **11.0%** |
| recent 5-minute activity | 8.6% |
| time of day | 7.6% |
| recent 60-minute activity | 5.5% |
| day of week | 3.7% |
| burst level | 0.0% |
| **irreducible** | **88.95%** |

Three consequences:

1. **Serverless idle gaps are close to memoryless.** No model of any kind can
   predict most of the next gap. This is the single most useful number in the
   project and it bounds all future work on the problem.
2. **It explains four earlier failures at once** — see `NEGATIVE_RESULTS.md`.
3. **It selects the estimator.** When the informative key is one coarse discrete
   variable, the right retrieval is an exact-match partition with thousands of
   samples per cell, not a learned embedding with eight neighbours per query.

### The mechanism check

A result at matched cost shows a method works; it does not show *why*. Before
running Method A we committed to a falsification condition: if the gain comes
from the previous gap carrying information, it must vanish where it carries none.

Grouping 313 applications by the measured dependence between consecutive gaps,
on training data only:

| stratum | gain at matched cost |
| --- | --- |
| Q1 — gaps carry no information | **−0.6%** |
| Q2 | −11.6% |
| Q3 | +8.6% |
| Q4 — gaps carry the most | **+62.4%** |

Correlation between information present and gain obtained: **+0.99**.

The method helps exactly where the theory says it must and vanishes where it must.
A method that helped everywhere equally would be the suspicious result.
