# Results

Every number this project reports, the control that accompanies it, and the file
it is read from. Nothing here is typed by hand into a document: the report, the
deck and this repository's figures all read these same JSON files, and
`scripts/verify_deliverables.py` fails the build if a deliverable disagrees with
them.

All comparisons are at **matched memory cost** on held-out test days 11–14 of the
Azure Functions 2019 trace.

---

## 1. Headline — the derived window

`artifacts/tables/derived_summary.json` · `scripts/run_derived.py`
1,999 applications.

| arm | median | range | points |
| --- | --- | --- | --- |
| **Derived window, marginal histogram** | **+78.8%** | +15.6% … +89.0% | 6 |
| Derived window + conditioning | +68.6% | −20.7% … +89.3% | 6 |
| Conditioning alone, percentile window | +26.3% | +19.0% … +30.7% | 5 |
| **CONTROL — same optimiser, a donor application's history** | **−427.5%** | −1819.8% … −110.4% | 6 |

Per-budget detail: `artifacts/tables/derived_matched_cost.json`.
Raw sweep: `artifacts/tables/derived_sweep.json`.

### The control is the number to read first

A 78.8% improvement deserves suspicion. The obvious sceptical reading is that the
rule merely produces sensible window sizes, and that anybody's history would do.

So the identical optimiser — same cost model, same price ratio, same simulator,
same matched-cost accounting — was given **another application's** gap histogram.
Only the ownership of the evidence changed. It lands at **−427.5%**: several
times worse than the baseline it had been beating.

A cost-optimal window computed from the wrong distribution is confidently,
precisely wrong about when to hold and when to release. A vague heuristic would
degrade gently; an optimiser does not. That asymmetry is the evidence.

---

## 2. Against a deliberately strengthened baseline

`artifacts/tables/strong_baseline_summary.json` · `scripts/run_strong_baseline.py`
600 applications, 36 baseline configurations.

**+67.6% median**, range **+26.8% … +76.5%**, positive at **all 13** matched
budgets.

The derived rule tunes two degrees of freedom where our sweep of the baseline
varied one. That is a real asymmetry and the first thing a reader should attack,
so it was removed rather than argued away: ATC'20 was swept over **both** of its
percentiles — six values of each — and the lower envelope of the resulting 36
configurations taken. That envelope is a stronger baseline than the published
policy, which does not tune the head at all, and it was allowed to be chosen with
the test period in view.

**Quote 67.6%, not 78.8%, if challenged on fairness.** It is the number
constructed against ourselves.

Cloud and envelope: `artifacts/tables/strong_baseline_cloud.json`,
`artifacts/tables/strong_baseline_matched.json`.

---

## 3. Conditioning, with its controls

`artifacts/tables/conditional_matched_cost.json` · `scripts/run_conditional.py`
1,999 applications.

| conditioning key | median | cell-count-matched control |
| --- | --- | --- |
| previous gap | +21.4% | −0.2% |
| previous gap × time of day | **+26.3%** | **−1.4%** |

The control replaces the real conditioning variable with a **random key having
the same number of cells**. If the gain came from the statistical side-effect of
splitting data more finely rather than from the information in the key, the
control would reproduce it. It does not.

A control that is not cell-count-matched is not a control for this claim, which
is why there are two of them — one sized for each arm.

### Sensitivity

`artifacts/tables/sensitivity.json` — positive at **all 15** settings of the two
hyper-parameters, +25.2% to +32.7% over 350 applications. The values used are not
the grid's best, which is what one expects of values chosen before looking.

---

## 4. Mechanism, not just outcome

`artifacts/tables/mechanism_check_summary.json` · `scripts/run_mechanism_check.py`
313 applications.

Before running the method we committed to a falsification condition: if
conditioning helps *because* consecutive gaps carry information about each other,
the gain must vanish where they do not.

| stratum (dependence measured on training data only) | gain |
| --- | --- |
| Q1 — lowest | **−0.6%** |
| Q2 | −11.6% |
| Q3 | +8.6% |
| Q4 — highest | **+62.4%** |

**Correlation between information present and gain obtained: +0.99.**

Q2 is negative and the top stratum rests on fewer points. Both are reported.

---

## 5. What is predictable at all

`artifacts/tables/context_information.json` · `scripts/run_information_audit.py`
264 applications.

**88.95% of the next idle gap's entropy is irreducible** from any observable
context. The remainder, by context:

| context | share |
| --- | --- |
| previous gap | 11.0% |
| recent 5-minute activity | 8.6% |
| time of day | 7.6% |
| recent 60-minute activity | 5.5% |
| day of week | 3.7% |
| burst level | 0.0% |

This is a model-free bound. It says what no amount of modelling can achieve on
this problem, it explains the four negative results, and it chose the estimator.

---

## 6. The hyper-parameter, and why its choice mattered

`artifacts/tables/smoothing_tuning.json` · `scripts/tune_smoothing.py`

The derived window has one hyper-parameter: the strength of the prior on the gap
distribution. It was swept on the **validation** days (9–10) and carried to the
test days unchanged.

| prior strength (total pseudo-mass) | median gain on VALIDATION |
| --- | --- |
| 0 | +61.7% |
| 1 | +63.6% |
| 5 | +69.0% |
| **25** ← chosen | **+70.3%** |
| 60 | +65.8% |
| 120 | +61.5% |
| 241 | +53.5% |

The setting at 241 is the accidental behaviour of an earlier implementation that
added one pseudo-count *per bin* over a 241-minute support. **It scores better on
the test period** than the corrected one — +74.3% against +69.3% on 250
applications — and **worst of the seven on validation.**

Selecting on the test period would have picked the worst-generalising point in
the grid and attached the largest number in this project to it. A unit test pins
`DEFAULT_SMOOTH` to the recorded validation choice so it cannot drift back.

---

## 7. Where the gain stops

The claim has a boundary and it is visible in the data rather than argued around.
Against the strengthened baseline the reduction falls from roughly **+76%** at
tight memory budgets to **+26.8%** at the loosest tested.

When memory is nearly free, the right policy is to hold everything warm for a
long time; there is little left for an optimiser to recover, and a long fixed
keep-alive becomes hard to beat. The method matters where memory is scarce enough
to be worth allocating carefully, which is the regime a platform operator is in.

### Per-application versus fleet-level

These are different statistics and both are true:

- The headline matches **total** memory across applications, which allows memory
  to be reallocated between them.
- The **median application** sees roughly no change, because most applications
  barely cold-start at all.
- The top 10% of applications carry **68%** of all cold starts, and that is where
  the reduction happens.

The fleet figure is a total, not a per-application average. The honest answer to
"does every application improve?" is no — the method removes cold starts where
cold starts actually are.

---

## 8. Figures

| figure | shows |
| --- | --- |
| `derived_frontier.png` | the derived window against all 36 tuned baseline pairs and their envelope |
| `derived_why.png` | both rules drawn on one application's distribution, with each window's expected cost |
| `derived_composition.png` | the three methods and the donor control, on a split axis |
| `information_audit.png` | what each context explains, and the irreducible share |
| `mechanism_check.png` | gain against measured dependence, by stratum |
| `conditional_result.png` | cost/benefit curves for the conditioning arms |
| `finding_matched_cost.png` | the fleet-transfer negative result, with its oracle bound |

All regenerated by `scripts/make_*_figures.py` from the tables above.

---

## Reproducing any of this

See [`REPRODUCING.md`](REPRODUCING.md). Each script writes its JSON next to its
figures, so documents regenerate offline without re-running the experiments.
