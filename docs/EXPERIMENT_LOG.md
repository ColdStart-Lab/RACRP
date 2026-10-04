# Experiment log

Every run that informed a decision, including the ones that failed. Numbers are
copied from `artifacts/tables/*.json`; nothing here is typed by hand.

---

## E0 — Trace parsing validated against ATC'20  (`characterisation.json`)

24,274 applications x 20,160 minutes, 12.50 billion invocations, aggregated from
function level to application level (Azure allocates per app).

| statistic | ours | Shahrad et al. ATC'20 |
|---|---|---|
| apps invoked at most once per minute on average | **82.3%** | ~81% |
| dominant trigger | http (38.5% of apps) | http |
| apps producing 99% of invocations | 9.4% | ~18.6%* |
| median app duty cycle | 1.7% of minutes | — |
| apps with average idle time > 10 min | **55.6%** | — |

*The released trace is a subset of the data used in the paper, so the skew figure
is not expected to match exactly; the invocation-rate and trigger figures, which
the paper reports for the released subset, do match.

**Verdict:** parsing is sound. Days 6, 7, 13, 14 show ~15.5k active apps vs ~18k
on other days — these are the two weekends, an independent confirmation that the
day ordering is intact.

**Why this matters for the project:** 55.6% of applications have an average idle
time longer than the industry-standard 10-minute keep-alive window. A single
fixed window is structurally wrong for this workload — which is the opening the
project exists to exploit.

---

## E1 — The beat-bar  (`baselines_test.json`)

Cohort: 1,999 apps (stratified by activity decile x trigger, seed 42).
Test slice: days 11-14 (minutes 14400-20160). 146,435,719 invocations.

| policy | cold starts | idle GB-minutes |
|---|---|---|
| oracle (upper bound) | 0 | 0 |
| fixed keep-alive 120 min | 4,147 | 664,485 |
| fixed keep-alive 60 min | 6,160 | 616,612 |
| fixed keep-alive 30 min | 18,813 | 542,130 |
| **hybrid histogram (ATC'20)** | **22,359** | **336,198** |
| fixed keep-alive 20 min | 26,443 | 505,408 |
| fixed keep-alive 10 min | 60,235 | 443,769 |
| fixed keep-alive 5 min | 120,886 | 371,953 |
| scale-to-zero | 2,596,322 | 0 |

**The standard to beat:** the ATC'20 hybrid histogram Pareto-dominates the entire
fixed keep-alive family — it achieves *fewer* cold starts than a 20-minute window
while spending **33% less** memory-time. This is the strongest published
non-learned policy and it is a hard target, exactly as it should be.

RACRP counts as an improvement only if it lands below-and-left of the point
(336,198 GB-min, 22,359 cold starts).

**Caveat recorded up front:** aggregate cold-start *rate* is tiny (0.015%)
because invocations are dominated by a handful of very dense apps. Sparse apps —
the ones this project targets — are invisible in the aggregate. Stratified and
macro-averaged results are therefore reported alongside, and any claim that holds
only in the aggregate will be labelled as such.

---

## E2 — Smoke run: the ablation looks good, the control says otherwise

Small run (30k train situations, 2 epochs, 15k bank) to validate the pipeline.

| model | test AP |
|---|---|
| LSTM | 0.9229 |
| Transformer, no retrieval | 0.9477 |
| **RACRP** | **0.9647** |
| RACRP fed random memories | 0.9625 |
| Retrieval-only vote | 0.9215 |

Paired bootstrap: RACRP − no-retrieval = **+0.0169** (p < 0.0001), but
RACRP − random-memory control = **+0.0023**, CI [−0.0002, +0.0049].

**Reading:** ~86% of the apparent retrieval gain is the fusion block's extra
capacity, not the retrieved content. Reporting the +0.0169 as "the value of
retrieval" would have been wrong. This is exactly why the control arm exists.

*(Two flaws found and fixed afterwards: the control paired random memories with
the real neighbours' similarity scores; and the encoder had no pressure to
organise its space by anything but risk level.)*

---

## E3 — Retrieval ceiling diagnostic: the aggregate metric was hiding everything

No training. For 8,000 test situations, take a similarity-weighted vote of the
retrieved neighbours' realised labels under four key spaces, stratified by how
long the application had been idle at the decision point.

`ORACLE` ranks neighbours by similarity of their **post-horizon** future to the
query's true post-horizon future. Ranking on the *full* profile would be
tautological — its first 5 minutes are the label — so those minutes are excluded.

| regime | n | base rate | random | **raw history cosine** | **learned embedding** | ORACLE |
|---|---|---|---|---|---|---|
| all | 8,000 | 0.282 | 0.2794 | **0.9246** | 0.9226 | 0.9121 |
| idle < 15 min | 2,852 | 0.745 | 0.7494 | **0.9746** | 0.9490 | 0.9670 |
| idle 15–60 min | 827 | 0.129 | 0.1256 | **0.6383** | 0.3140 | 0.2408 |
| idle > 60 min | 4,321 | 0.006 | 0.0058 | 0.0171 | 0.0180 | 0.0595 |

Neighbour label agreement: random 0.637, **learned 0.911**, oracle 0.740.

### Three findings, all load-bearing

**1. The headline metric is measuring the wrong thing.** 36% of decision points
are idle < 15 min with a base rate of 0.745 — nearly trivial. 54% are idle > 60
min with a base rate of 0.006. An aggregate AP of 0.92 is almost entirely the
easy third. Every comparison made on the aggregate was dominated by cases where
no scheduler needs help. **The evaluation must be stratified by idle age.**

**2. The learned retrieval key space is worse than a trivial baseline.** Plain
cosine similarity on the raw 120-minute history retrieves better neighbours than
our trained encoder — overall (0.9246 vs 0.9226) and decisively in the 15–60
minute band (**0.638 vs 0.314**, more than double). The encoder is trained to
emit a risk, so its embedding collapses towards a risk scalar: it retrieves
situations that *agree with the prediction the model was already going to make*.
The label-agreement row confirms this directly — the learned space retrieves
same-label neighbours 91% of the time, far more often than the oracle (74%). It
is optimising for agreement, which carries no information, instead of for
evidence.

**3. Matching the later future is not the right key either.** The oracle, which
sees the true post-horizon trajectory, is *worse* than raw history at 15–60 min
(0.241 vs 0.638). Knowing what an application does an hour from now says little
about whether it fires in the next five minutes. So the target for metric
learning is not "same long-run trajectory".

### Consequence for the project

The retrieval mechanism is not failing because episodic memory is useless here —
it is failing because the key space was trained for the wrong objective, and
because the aggregate metric made the failure invisible. Both are fixable, and
the fix is the actual research contribution rather than a patch.

---

## E4 — Reframing: evaluate on held-out applications with limited own history

E2/E3 showed retrieval was redundant *by construction*: every test application's
own 8-day history was already supplied as features. New protocol (docs/REFRAME.md):
70/30 seen/unseen application split, own-history budget `H` swept, fleet bank
built from seen applications only, everything scored at matched memory cost.

First pass, **not** matched on cost, looked excellent — cold-start reduction vs the
own-histogram of +38.4% (just deployed), +17.3% (1 h), +12.7% (4 h), +6.5% (1 d),
+3.7% (3 d), +2.8% (8 d): the exact monotone decay the mechanism predicts.

**Then we matched the cost, and the result reversed.** The reduction had been
bought entirely with memory: at the histogram's own spend, shrinkage was *worse*
by 206% (just deployed) through 16% (8 days). Recorded because the unmatched
version is the number we would have reported had we not swept the cost knob.

---

## E5 — Three real defects, each found by measuring rather than reasoning

**E5.1 Density pooling inflates the tail.** Averaging analogues' densities makes a
*mixture*, whose spread is the union of theirs. Two analogues idling at 5 and 200
minutes give a 92nd percentile of 200 — the policy holds a container warm for the
worst analogue. Averaging *quantile functions* (1-D Wasserstein barycenter) gives
102, the typical value. Symptom in the data: the density-pooled prior barely
responded to its own cost knob (204k → 219k GB-min across tail percentiles 60→99),
because every percentile landed in the same inflated tail. Fixed; unit-tested.

**E5.2 Pre-warming off a borrowed prior is reckless.** Pre-warm releases the
container for the first `p5` minutes. That is a sound bet on an application's own
distribution and a bad one on analogues', which agree about typical *scale*, not
about this application's short tail. Symptom: the fleet arms spent *more* memory
**and** incurred *more* cold starts simultaneously — the signature of mis-placed,
not mis-sized, windows. Fixed: the offset is now estimated from own evidence only
and scaled by confidence `n/(n+κ)`.

**E5.3 The fleet bank was built over the target's window.** At `H = 0` the *seen*
applications had therefore also been observed for zero minutes, so the prior was
empty and silently fell back to the fixed default — "fleet-prior-only" was
measuring a 10-minute keep-alive. Fixed: retrieval keys use the target's budget
(so keys and queries are comparable), the borrowed distributions use the seen
applications' full history.

---

## E6 — Is the prior any good? Measured directly

For 219 held-out applications, the pooled prior's predicted 92nd-percentile
keep-alive against the value the application actually needed:

| | before calibration | after calibration |
|---|---|---|
| predicted median | 64 min | 18 min |
| actual median | 6 min | 6 min |
| median bias | **+1.75 octaves (3.4× too long)** | **+0.06 octaves** |
| Spearman ρ (predicted vs actual) | **0.370** | 0.370 |

So retrieval *does* carry real information — it ranks applications with ρ = 0.37 —
but the pooled level was 3.4× too long. A single global scale factor, fitted
leave-one-out on the **seen** applications only, removes the bias while preserving
the ranking. Calibration is now part of the estimator.

---

## E7 — The result, and it is negative

After all four fixes (quantile pooling, confidence-scaled pre-warm, correct bank,
calibration), on 600 held-out applications at `H = 0`, at matched memory cost:

| memory spent | own histogram (ATC'20) | fleet retrieval |
|---|---|---|
| ~91.6k idle GB-min | **22,811 cold starts** | 32,058 cold starts |

**Borrowing strength loses.** And this is not a failure of our retrieval rule: an
**oracle** retriever, which selects the seen applications whose *true* idle-gap
distribution is closest to the target's, also loses.

Stratified by how much data an application actually reveals (test period, `H = 0`):

| gaps revealed | n apps | own cold starts | fleet cold starts | own GB-min | fleet GB-min |
|---|---|---|---|---|---|
| ≤ 5 | 102 | 90 | 87 | 322 | 459 |
| 6–30 | 84 | 543 | 631 | 8,092 | 5,677 |
| 31–200 | 87 | 1,038 | 4,127 | 16,787 | 16,161 |
| > 200 | 327 | 21,140 | 27,212 | 66,521 | 69,260 |

(tail percentile 85; own-histogram wins or ties in every band at comparable spend.)

### Why, precisely

1. **The signal is weak.** ρ = 0.37 between a profile-retrieved prior and the
   application's real gap distribution. Trigger type, memory and declared duration
   do not determine how an application idles.
2. **Even a perfect retriever loses**, so this is not fixable by a better key —
   another application's gap distribution is simply not a good stand-in.
3. **The baseline's "fallback" is far stronger than its framing suggests.** A
   10-minute default plus online histogram updating adapts within a handful of
   gaps. Applications reveal their own gaps faster than borrowed information stays
   relevant.
4. **The regime where borrowing could help is economically negligible.** The 102
   applications that reveal ≤ 5 gaps over four days account for **90 of ~22,800
   cold starts — under 0.4%**. Even a perfect fix there cannot move the aggregate.

Point 4 is the one that settles it, and it could not have been known without
building the measurement apparatus first.

### What is being claimed, then

Not "retrieval helps". The defensible contributions are:

* a **validated measurement apparatus** — a minute-tick simulator checked against
  a reference implementation by randomised equivalence testing (27 tests), and a
  faithful ATC'20 hybrid-histogram reimplementation that reproduces the paper's
  central claim (it Pareto-dominates the entire fixed keep-alive family: 22,359
  cold starts at 336k GB-min versus 26,443 at 505k for a 20-minute window);
* a **negative result with a positive characterisation**, bounded by an oracle and
  quantified economically, for a mechanism that the literature has not tested;
* a **methodology**: the random-memory control and the matched-cost sweep each
  independently reversed a conclusion we would otherwise have reported. Neither is
  standard practice in retrieval-augmented systems work, and this project is a
  worked example of why they should be.

## E8 — The information audit: why every retrieval scheme failed, in one number

Training-free. For 264 applications with non-degenerate gap distributions, the
mutual information between each available context and the next idle gap
(both discretised; gaps into 10 log2 buckets), measured on the training period.

Median entropy of the gap itself: **1.242 bits**.

| context | median MI (bits) | p90 | share of gap entropy |
|---|---|---|---|
| burst level (24 h activity) | 0.0003 | 0.034 | **0.0%** |
| day of week | 0.0456 | 0.338 | **3.7%** |
| recent 60-min activity | 0.0679 | 0.244 | **5.5%** |
| time of day | 0.0946 | 0.408 | **7.6%** |
| recent 5-min activity | 0.1068 | 0.360 | **8.6%** |
| **previous gap** | **0.1372** | **0.682** | **11.0%** |

### This is the unified explanation

**About 89% of the uncertainty in an application's next idle gap is irreducible
from anything the scheduler can observe.** Serverless idle gaps on this trace are
close to memoryless. Every retrieval scheme we tried — cross-application transfer,
situation retrieval, and context-conditional retrieval over a rich key — was
paying a large variance cost to chase signal that is mostly absent. That is one
mechanism behind three separate negative results, and it also explains why the
ATC'20 authors' choice of a plain marginal histogram is close to the right call.

It also says precisely where the exploitable signal is: **one discrete,
low-dimensional variable — the gap that just closed.**

---

## E9 — Conditioning on the previous gap: a real improvement

If the informative key is one coarse discrete variable, the right non-parametric
conditional estimator is a **conditional histogram**, not a learned embedding with
k-NN lookup. Retrieval over a discrete key with many samples per cell *is* a
conditional histogram; the embedding machinery only pays when the key is
high-dimensional, and here it was pure variance.

    h_hat( g | b ) = ( n_b · h_b + κ · h_marginal ) / ( n_b + κ )

with `b` = log2 bucket (6 cells) of the gap that just closed, blended in quantile
space, partially pooled towards the marginal. `κ → ∞` recovers ATC'20 exactly.

Pilot, 400 applications, test days 11–14, identical code path with only the
conditioning key changed, both arms swept over the tail percentile and compared
at **matched memory cost**:

| memory budget (idle GB-min) | ATC'20 marginal | conditioned on previous gap | reduction |
|---|---|---|---|
| 52,906 | 24,466 | 15,411 | **−37.0%** |
| 61,384 | 13,735 | 9,446 | **−31.2%** |
| 67,601 | 8,909 | 5,925 | **−33.5%** |

An off-by-one was found and fixed along the way (conditioning on the gap *two*
steps back rather than the one that just closed); fixing it improved the result
slightly, in the expected direction.

Full-cohort run with the ablation set — time-of-day key, joint key, and a
**random-key control** that holds the number of cells fixed to separate real
conditioning from the variance-reduction artefact of merely splitting the data —
is the headline experiment (`scripts/run_conditional.py`).

Seven unit tests pin the estimator, including two that constrain what it is
allowed to do: it must reproduce the baseline exactly as `κ → ∞`, and it must
**not** produce a large gain on a synthetic memoryless application.

## E10 — The headline result, full cohort, with controls

1,999 applications, test days 11–14. Every arm is the *same* policy
implementation with only the conditioning key changed, swept over the same
tail-percentile cost knob, and compared by interpolating each curve at the
baseline's own operating points.

Cold starts at matched memory cost:

| memory budget (idle GB-min) | ATC'20 marginal | prev gap | tod | prev gap × tod | random key (6 cells) |
|---|---|---|---|---|---|
| 235,172 | 197,089 | 167,823 (+14.9%) | 182,658 (+7.3%) | **159,695 (+19.0%)** | 195,760 (+0.7%) |
| 274,305 | 136,411 | 107,173 (+21.4%) | 125,431 (+8.1%) | **100,569 (+26.3%)** | 136,758 (−0.3%) |
| 314,927 | 73,780 | 58,452 (+20.8%) | 68,294 (+7.4%) | **55,266 (+25.1%)** | 74,527 (−1.0%) |
| 350,695 | 44,026 | 32,755 (+25.6%) | 40,235 (+8.6%) | **30,518 (+30.7%)** | 43,451 (+1.3%) |
| 381,101 | 26,430 | 19,212 (+27.3%) | 24,414 (+7.6%) | **18,938 (+28.4%)** | 26,462 (−0.1%) |

Median reduction: **previous gap +21.4%**, time of day +7.6%, **joint key +26.3%**,
**random key −0.2%**.

### The internal validation

The information audit (E8) was computed *before* any of these policies were run,
and neither quantity was tuned to the other. The ranking of scheduling
improvement reproduces the ranking of mutual information:

| context | share of gap entropy (E8) | median cold-start reduction (E10) |
|---|---|---|
| random key | 0% (by construction) | −0.2% |
| time of day | 7.6% | +7.6% |
| previous gap | 11.0% | +21.4% |
| previous gap × time of day | (both) | +26.3% |

An information-theoretic measurement predicting the ordering of a scheduling
outcome it never saw is the strongest internal check available here.

### A rigor gap found and closed

The joint key uses 24 cells while the original random-key control used 6. A
control must match the arm it controls for in cell count, or it does not control
for the variance of splitting the data more finely. A 24-cell random control was
added (`key="shuffled_joint"`) and re-run before the joint-key result is claimed.

### Why the mechanism is defensible

* It is a **strict generalisation of the deployed policy**: `κ → ∞`, or a cell
  with no observations, reproduces the ATC'20 histogram exactly — pinned by a unit
  test, not merely asserted.
* The **estimator follows the measurement**. The audit said the informative key is
  one coarse discrete variable; when that is true, the right non-parametric
  conditional estimator is a conditional histogram, and a learned retrieval
  embedding is pure variance. That is why this works where three richer schemes
  failed.
* It is **cheap**: a handful of small histograms per application, updated online
  from gaps that have already closed. No embedding, no index, no GPU.

## E11 — A design bug in the risk-to-policy conversion, found by running it

`scripts/run_policy.py` had been written but never executed. Dry-running it
produced an identical result for every value of the cost knob — 866,001 cold
starts at every λ, exactly matching a 1-minute fixed keep-alive.

**Cause.** Prediction quality is measured on a *sample* of test situations, which
is correct for average precision. A policy is different: the scheduler must
decide at **every** idle minute. Scattering a sample of 8,000 situations onto the
(application × minute) grid leaves virtually every minute at risk zero, so the
threshold rule never fires and the policy collapses onto its unconditional
one-minute floor.

**Fix.** `scripts/run_policy_eval.py` evaluates a smaller cohort of applications
over **all** of their decision minutes rather than a large cohort over a sample of
theirs. Complete coverage per application is the unit that a policy evaluation
needs. It also scores the ATC'20 histogram and the conditional estimator on the
*same* applications so the three are comparable at matched cost.

Worth recording because the failure was silent: the script ran, produced a table,
and the table was meaningless. Nothing but running it would have surfaced that.

## E12 — Mechanism check: the gain appears exactly where the theory says

Falsification condition stated before the method was run: if conditioning helps
because an application's gaps genuinely depend on each other, the improvement must
concentrate where that dependence exists and be small where it does not.

Applications split into quartiles by the mutual information between consecutive
idle gaps, measured on the **training** period only. Within each quartile both
arms are swept over the cost knob and compared at matched budget.

| stratum | apps | median MI (bits) | reduction at matched cost |
|---|---|---|---|
| Q1 lowest | 80 | 0.000 | −0.6% |
| Q2 | 77 | 0.001 | −11.6% |
| Q3 | 78 | 0.092 | +8.6% |
| **Q4 highest** | 78 | 0.490 | **+62.4%** |

**Correlation across strata between median MI and matched-cost reduction: +0.99.**

Reported honestly: Q2 is *negative* (−11.6%), and Q4 rests on only 2 matched
budget points because the two arms' cost curves overlap over a narrow range in
that stratum. The monotone trend from Q2 upward and the near-perfect correlation
are the substance; the individual quartile figures are noisy.

### The first version of this check was wrong, and the way it was wrong matters

It compared the two arms at a **fixed tail percentile** within each quartile, and
appeared to show the method failing catastrophically in the most-autocorrelated
quartile: −149% cold starts. That arm was spending **20% less memory**. At a fixed
percentile the two policies simply sit at different points on their own cost
curves, so the comparison is meaningless.

This is the third time in this project that an unmatched-cost comparison produced
a confident and wrong answer — first a spurious +38%, then a spurious −149%, in
opposite directions. It is the single most reliable way to be wrong about a
keep-alive policy, and it is why every comparison here is a curve.

## E13 — The retrieval ablation at proper scale: the control overtakes the model

The earlier ablation (E2) came from a 2-epoch smoke run and was the reason the
project was redirected. It has now been repeated properly: 80,000 training
situations, 4 epochs, five arms, identical loop, seed and splits, on the full
1,999-application cohort.

Validation average precision by epoch:

| arm | ep1 | ep2 | ep3 | ep4 |
|---|---|---|---|---|
| LSTM (AWU-style) | 0.9218 | 0.9319 | 0.9339 | 0.9338 |
| Transformer, no retrieval | 0.9403 | 0.9673 | 0.9716 | 0.9720 |
| **RACRP (retrieval on)** | 0.9726 | 0.9747 | 0.9762 | 0.9762 |
| **Transformer, step-matched control** | 0.9743 | 0.9763 | **0.9782** | 0.9776 |
| RACRP fed random memories | 0.9710 | ... | | |

### The finding, stated plainly

RACRP is warm-started from the no-retrieval transformer and then trained for
further epochs, so comparing it against that transformer confounds retrieval with
extra optimisation. Giving the no-retrieval model the **same** extra epochs — the
`b3_continued` arm — does not merely close the gap. **It overtakes RACRP**
(0.9782 against 0.9762).

At smoke scale this control closed most of the gap; at proper scale it closes all
of it and more. Retrieval contributes nothing here once training steps are
equalised, and the honest statement is that the apparent +0.004 advantage over the
plain transformer was optimisation, not memory.

This is the cleanest form the ablation could have taken, and it is why the
project's contribution rests on the conditional estimator rather than on the
retrieval architecture. It also retrospectively vindicates the decision to
redirect after E2 rather than scaling the retrieval model up.

### E13 (continued) — the test-set ablation, and why the naive version would have lied

80,000 training situations, 4 epochs, five arms, identical loop / seed / splits,
full 1,999-application cohort. Held-out test days.

| model | average precision | ECE |
|---|---|---|
| LSTM (AWU-style) | 0.9433 | 0.0224 |
| Transformer, no retrieval | 0.9815 | 0.0050 |
| **Transformer, step-matched control** | **0.9862** | 0.0039 |
| RACRP (retrieval on) | 0.9851 | 0.0040 |
| RACRP fed RANDOM memories | 0.9854 | 0.0037 |
| Retrieval-only vote | 0.9672 | 0.0129 |

Paired bootstrap on the shared test set:

| comparison | ΔAP | 95% CI | p(A not better) |
|---|---|---|---|
| RACRP − Transformer (no retrieval) | **+0.0036** | [+0.0032, +0.0041] | 0.0000 |
| RACRP − Transformer (**step-matched**) | **−0.0011** | [−0.0014, −0.0008] | **1.0000** |
| RACRP − RACRP with random memories | −0.0003 | [−0.0005, −0.0000] | 0.9890 |
| RACRP − retrieval-only vote | +0.0179 | [+0.0164, +0.0194] | 0.0000 |

### Read this table in order — it is the whole methodological point

Against the obvious baseline, retrieval appears to win by +0.0036 with p < 0.0001.
That is the comparison most papers would report, and it is publishable-looking.

Against the **step-matched** control — the same model, no retrieval, given the
same number of extra training steps, because RACRP is warm-started from the plain
transformer and then trained further — retrieval is **significantly worse**
(−0.0011, p = 1.0000).

And **random memories score as well as real ones** (0.9854 vs 0.9851).

So the +0.0036 was entirely the extra optimisation that the architecture comparison
silently included. One additional arm, costing one training run, converts a
significant positive result into a significant negative one. This is the fourth
occasion in this project where a control reversed a conclusion, and the reason the
contribution rests on the conditional estimator rather than on retrieval.

The retrieval-only vote (0.9672) is well below the reader model, which is the
mechanical explanation: a rational fusion block learns to close the gate on a
memory that predicts worse than the model already does.

## E14 — Phase 8: giving the predictor the variable it was blind to

The predictor's only view of recent history is a 240-minute invocation window, and
the information audit says the most informative context is the length of the gap
that just closed. The window cannot represent a gap longer than itself, so the
model was blind to that variable wherever the window had saturated.

Two arms, identical in architecture, optimiser, seed, splits and step budget,
differing only in ten explicit renewal-state scalars (uncapped idle age, previous
gap length and bucket, time since a comparable gap).

| | average precision | ECE |
|---|---|---|
| window only (current predictor) | 0.9823 | 0.0068 |
| **+ explicit renewal state** | **0.9841** | **0.0042** |

Overall ΔAP **+0.0018** [+0.0015, +0.0021], p < 0.0001. **Calibration error falls
by 38%** (0.0068 → 0.0042), which matters more than the AP for a policy that
thresholds the probability rather than ranking it.

### By idle age — and the prediction was only half right

| idle age at the decision | n | base rate | window only | + state | gain |
|---|---|---|---|---|---|
| < 15 min | 28,272 | 0.740 | 0.9917 | 0.9928 | +0.0011 |
| 15–60 min | 8,026 | 0.131 | 0.7882 | 0.7854 | −0.0028 |
| **60–240 min** | **6,568** | **0.019** | **0.0669** | **0.1257** | **+0.0589** |
| 4–24 h | 17,099 | 0.0061 | 0.0129 | 0.0097 | −0.0032 |
| > 24 h | 19,078 | 0.0006 | 0.0025 | 0.0010 | −0.0014 |

**Stated before the run:** the gain must concentrate at *long* idle age, where the
window has saturated. **Measured:** the gain concentrates at *moderate* idle age
(60–240 minutes), where it nearly doubles average precision, and does **not**
appear beyond four hours — which is precisely the region the prediction named.

So the prediction fails as stated, and the corrected explanation is narrower. The
features help where the window is sparse but the event still sometimes happens.
Beyond four hours the base rate collapses to 0.6% and then 0.06%: there is so
little to predict that eleven per cent of a small entropy cannot be converted into
anything a metric can see. The binding constraint at long idle age is not the
model's inputs but the near-absence of events.

Reported as a partial confirmation, not a success. The band where it works
(60–240 minutes) is real and large — average precision 0.067 → 0.126 — but it is
not the band we said it would be.

## E15 — Objective O4: the calibrated risk, converted into a scheduling policy

The last unclosed Review-1 objective. 300 applications evaluated over **every** one
of their decision minutes (not a sample — see E11), the calibrated probability
turned into keep-warm decisions by the cost-aware rule, swept over lambda, and
compared against the ATC'20 histogram and the conditional estimator **on the same
applications at matched memory cost**.

| memory budget (idle GB-min) | ATC'20 | conditional (ours) | risk policy, LSTM | risk policy, Transformer |
|---|---|---|---|---|
| 40,242 | 28,206 | **26,025 (+7.7%)** | 41,694 (−47.8%) | 45,986 (−63.0%) |
| 44,613 | 22,966 | **19,914 (+13.3%)** | 30,803 (−34.1%) | 36,671 (−59.7%) |
| 50,384 | 12,040 | **10,519 (+12.6%)** | 19,794 (−64.4%) | 24,374 (−102.4%) |
| 55,133 | 7,304 | **6,134 (+16.0%)** | 16,942 (−132.0%) | 14,253 (−95.1%) |
| 59,363 | 4,688 | **3,901 (+16.8%)** | 14,402 (−207.2%) | 8,033 (−71.4%) |

### The finding: prediction accuracy does not transfer to scheduling

The transformer reaches **0.98 average precision** and an expected calibration
error of 0.005 on this task. Converted into a policy it is **63% to 118% worse
than a histogram** at matched cost. The conditional estimator, which has no
learned component at all, wins by 8–17% on the same applications.

Three reasons, and they compound:

1. **The metric and the decision are misaligned.** Average precision is dominated
   by the 36% of decision points that are less than fifteen minutes idle and have
   a base rate of 0.74. A scheduler's cold starts come from elsewhere.
2. **The policy shape is crude.** Thresholding `P(event within 5 minutes)` warms a
   fixed five-minute window. The histogram sets a *pre-warm offset* and a
   *keep-alive length* per application from its own gap distribution — two degrees
   of freedom fitted to the quantity the decision is actually a function of.
3. **The horizon is fixed.** The decision the scheduler makes is "how long to
   hold", which is a quantile of the gap distribution. A binary at one horizon
   cannot express it, however well calibrated that binary is.

This closes O4 with a negative result for the learned-risk route and a positive
one for the estimator that replaced it. It is also the clearest statement of why
this project ended up where it did: the winning method predicts the *right
quantity* with a simple estimator, rather than the wrong quantity with a
sophisticated one.

*(The conditional figures here, +7.7% to +16.8%, are on a 300-application subsample
chosen for inference cost; the headline +26.3% is the full 1,999-application
cohort. Different cohorts, both reported.)*

## E16 — Were the two hyper-parameters tuned on the test set?

The conditional estimator has exactly two knobs: `kappa` (the pull towards the
marginal, in equivalent observed gaps) and the number of buckets the previous gap
is discretised into. Both were set by judgement — kappa = 8 so a handful of gaps
cannot yet outvote an application's whole history, six log2 buckets because that
spans a minute to a day at a resolution the data supports. Neither was searched.

A reviewer is right to ask whether the result depends on that choice. Median
cold-start reduction against ATC'20 at matched memory cost, over the full grid:

| buckets \ kappa | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 4 | +29.9% | +28.1% | +31.4% | +29.9% | +25.2% |
| **6** | +32.5% | +30.1% | **+32.7%** | +30.7% | +25.8% |
| 8 | +32.5% | +30.0% | +32.7% | +30.7% | +25.8% |

**Positive at all 15 settings, spanning +25.2% to +32.7%.** The chosen setting
(6 buckets, kappa = 8) is not the best in the grid on this subsample, which is
what one would expect of a value picked before looking. The result is a property
of the method rather than of the two numbers.

Two things the grid also shows. Increasing kappa past 16 monotonically erodes the
gain — as it must, since kappa → ∞ is the ATC'20 baseline by construction, and
watching the method decay into its own baseline is a further check that the
implementation does what the algebra says. And 6 and 8 buckets are essentially
identical, so the discretisation is already fine enough that the data, not the
resolution, is the binding constraint.

*(350-application subsample, chosen for runtime; the headline +26.3% is the full
1,999-application cohort. Different cohorts, both reported.)*

## E17 — Deriving the keep-alive window instead of tuning it

Every policy so far, including the deployed one, reads the keep-alive off a **tail
percentile** of a gap distribution and sweeps that percentile as a cost knob. The
percentile is a heuristic and is not a quantity a platform can price.

The decision is an optimal-stopping problem. Holding an idle container for `T`
minutes costs memory while the gap runs and risks a cold start if the gap outlives
`T`:

    C(T) / K  =  (m/K) · Σ_{u≤T} P(gap ≥ u)   +   P(gap > T)

Setting the derivative to zero gives the identity **hazard(T) = m/K** — drop the
container when its hazard rate falls below the price ratio. What is swept is then
`K/m`, how many GB-minutes a cold start is worth, which a platform can quote.

### Two bugs, both caught by running it

**1. The first-order condition is not the algorithm.** `C(T)` is not unimodal for
real gap distributions. A perfectly periodic application has zero hazard
everywhere except at its period, so a "first crossing" rule stops at minute 1 and
cold-starts on *every* invocation — measured, exactly that. The identity is the
right way to *understand* the rule; the implementation must take the global argmin
of `C(T)`, which is exact and costs O(range). A unit test now pins the
implementation against a brute-force search of the same objective.

**2. The derived window cannot be composed with a pre-warm offset.** `T*`
minimises the cost of holding *continuously from the gap's start*. Applying it as
"hold from `pw` to `pw+T`" shifts the window it was optimised for. First measured
with the offset in place, all three refinements lost badly:

| arm | median vs ATC'20 |
|---|---|
| conditional, percentile window | **+33.0%** |
| A: derived window (composed with pre-warm) | −47.1% |
| B: + per-application selection | −64.1% |
| C: + regime weighting | −158.1% |

Isolating the length rule — percentile vs derived, **pre-warm disabled on both
arms**, marginal distribution, same code path — reverses it completely:

| memory budget | percentile window | derived window | reduction |
|---|---|---|---|
| 89,017 | 9,685 | 2,679 | **+72.3%** |
| 92,960 | 7,577 | 2,272 | **+70.0%** |
| 97,834 | 5,308 | 1,770 | **+66.6%** |
| 101,666 | 3,785 | 1,376 | **+63.7%** |
| 105,403 | 2,577 | 991 | **+61.5%** |
| 110,209 | 1,200 | 668 | **+44.4%** |

**Median +65.1%.** The derived rule is far better than the percentile once it is
applied to the policy shape it was derived for.

This is the seventh time in this project that a confident negative or positive
turned out to be an artefact of how two things were combined rather than of either
thing itself, and the second time the fix was to make two arms share a policy
*shape* so the ablation isolates one rule.

---

## E18 — The window was never derived, only tuned

**Question.** Every policy measured in this project so far, ours included, reads
the keep-alive window off a *percentile* of an idle-gap histogram. Where does
that percentile come from? Nowhere. What happens if the window is derived from
the cost model instead?

**Method.** The warm window `[a, a+L)` is chosen by minimising

```
C(a, L) = m · E[max(0, min(a+L, g) − a)] + K · P(g < a or g ≥ a+L)
```

jointly over offset and length, online and causally, from gaps that closed
strictly before the decision minute. The percentile knob is replaced by `K/m`,
the price of a cold start in GB-minutes. Full cohort, 1,999 applications, test
days 11–14, same simulator and same matched-cost accounting as everywhere else.

**Result (full cohort, 1,999 applications, vs the deployed ATC'20 policy).**

| arm | median reduction at matched cost | range | points |
|---|---|---|---|
| DERIVED window, marginal | **+78.8%** | +15.6% to +89.0% | 6 |
| DERIVED window + conditioning | **+68.6%** | -20.7% to +89.3% | 6 |
| conditional, percentile (E10, the previous headline) | **+26.3%** | +19.0% to +30.7% | 5 |
| **CONTROL** -- same optimisation, a donor application's histogram | **-427.5%** | -1819.8% to -110.4% | 6 |

The control is the number to read first. Handing the identical optimiser another
application's gap histogram does not merely erase the gain, it makes the policy
roughly **five times worse than the baseline it was beating**. The price ratio,
the optimiser, the simulator and the matched-cost accounting are unchanged;
only whose history it is changed. Whatever the derived window is doing, it is
not producing window sizes that happen to be good on average -- it is reading
something specific out of each application's own past.

**Result (vs a deliberately strengthened baseline).** The derived rule optimises
two degrees of freedom while the published ATC'20 sweep varied one, which is a
real asymmetry and the first thing a reviewer should attack. So the baseline was
swept over **both** percentiles, 6 × 6 = 36 configurations, and its lower
envelope taken — the best cost/latency frontier reachable by any fixed percentile
pair, chosen with the test period in view. 600 applications.

**The derived window still wins: +67.6% median
(range +26.8% to +76.5%, 13 matched points,
600 applications), positive at every point.** An earlier version of
this measurement, made before the estimator's prior was fixed, gave +62.6% and
went *negative* at the two highest-memory budgets. Fixing the prior removed both
negatives and raised the median.

The boundary of the claim is still visible, just no longer below zero: the gain
falls from around +76% at tight budgets to +26.8% at the loosest one. When memory
is nearly free the right policy is to hold everything warm for a long time, there
is little left for an optimiser to recover, and a long fixed keep-alive becomes
hard to beat. The gain lives where memory is scarce enough to be worth allocating
carefully, which is the regime a platform operator is actually in.

**The finding that was not expected.** Adding the state conditioning *on top of*
the derived window makes it **worse** (+68.6%
against +78.8%), and far
more variable -- its range dips to
-20.7% where the marginal version
never goes below +15.6%. The two ideas do
not compose. The reason is mechanical: a percentile is a robust statistic that
survives sparse counts, whereas the cost-optimal window integrates over the whole
shape of the distribution and therefore needs the tail estimated well.
Conditioning fragments the counts that estimate that tail. Each idea is buying
information from a different place, and one of them spends the resource the other
depends on.

**The prior strength was chosen on validation.** The derived window integrates
over the whole distribution, so it is sensitive to tail smoothing in a way a
percentile is not. The setting that scores best on the *test* period is the worst
of seven on validation (+53.5% against +70.3%); `DEFAULT_SMOOTH =
25` was chosen on the validation days and carried
across unchanged. See `WHAT_THE_CONTROLS_CAUGHT.md` #9.

**Five bugs found on the way here**, each recorded in
`WHAT_THE_CONTROLS_CAUGHT.md` and in `THE_WINDOW_WAS_NEVER_DERIVED.md`:
the first-order condition `hazard(T) = m/K` is a stationary point and not the
minimum (a periodic application cold-starts on every invocation under it);
`head_margin=0.0` did not disable the pre-warm; the derived length does not
compose with an offset it was not optimised for; and optimising the length alone
concedes an action the deployed policy actually uses.
