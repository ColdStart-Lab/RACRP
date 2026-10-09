# RACRP — the serverless keep-alive window, derived rather than guessed

[![tests](https://github.com/ColdStart-Lab/RACRP/actions/workflows/ci.yml/badge.svg)](https://github.com/ColdStart-Lab/RACRP/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **The keep-alive policy deployed in Azure Functions reads its warm window off
> two fixed percentiles of an idle-gap histogram. Nobody argues for those
> percentiles, and no operator can price one. We derive the window instead, by
> minimising what it actually costs — which cuts cold starts by a median 78.8%
> at matched memory spend, and by 67.6% against a baseline strengthened to tune
> both of its percentiles.**

**Parth Goyal** ([@parthgoyal974](https://github.com/parthgoyal974)) ·
**Saanvi Sharma** ([@saanvi3605](https://github.com/saanvi3605))
VIT BCSE497J Project-I · Team 15785UG03 · Guide: Dr. Narayanan Prashanth

---

## The problem

Serverless platforms scale to zero, so an idle application must be rebuilt
before it can serve — a **cold start**, paid for by a waiting user. Holding the
container warm avoids it, but idle memory costs money. The entire problem is one
decision, repeated forever: *after an application goes quiet, how long do you
keep it alive?*

The policy deployed in Azure Functions (Shahrad et al., USENIX ATC'20) answers it
with a histogram of each application's own idle gaps and two percentiles off it —
the 5th for when to bring a container back, the 99th for when to release it.

This project asks what that policy discards, and finds **two separate answers in
two separate places.**

## The two methods

Any policy of this kind does two independent things: it decides **which
histogram to look at**, and it decides **how to turn a histogram into a window**.
Those axes had never been separated. Once they are, there are four policies:

|                          | percentile window          | **derived window**        |
| ------------------------ | -------------------------- | ------------------------- |
| **marginal histogram**   | ATC'20 — the deployed baseline | **+78.8%** ← headline |
| **conditional histogram**| **+26.3%**                 | +68.6%                    |

### Method A — condition the histogram on the application's state
`racrp/baselines/conditional.py`

The deployed histogram pools every gap an application ever showed and forgets
the state each occurred in. An application that idles two minutes mid-burst and
three hours between bursts gets one blended distribution that is wrong in both.
We keep one histogram per bucket of *the gap that just closed*, partially pooled
towards the marginal so a thin cell is never trusted before it is earned.

Let κ → ∞ and the deployed policy is recovered **exactly**, so this is a strict
generalisation along one axis rather than a competing heuristic.

### Method B — derive the window from the cost model
`racrp/baselines/hazard.py`

A container warm over `[a, a+L)` pays memory for however much of the idle gap
that window covers, and pays a cold start if the next invocation lands outside
it:

```
C(a, L) = m · E[ max(0, min(a+L, g) − a) ]  +  K · P(g < a  or  g ≥ a+L)
```

Minimise it jointly over offset and length. The percentile disappears; what you
sweep instead is **K/m — how many GB-minutes a cold start is worth** — which is a
number a platform operator can actually quote.

> The elegant form of this, `hazard(T) = m/K`, is the right way to *understand*
> the rule but is **not** the algorithm. It locates a stationary point, and the
> cost is not unimodal: a perfectly periodic application has zero hazard before
> its period, so a first-crossing rule stops at minute 1 and cold-starts on every
> invocation. We measured exactly that. The global minimum is the same order of
> work and is what the cost model asks for.

## Results

All at **matched memory cost** on held-out test days (11–14) of the Azure
Functions 2019 trace. Every number below is read from `artifacts/tables/`.

| claim | median | range | n |
| --- | --- | --- | --- |
| **Derived window vs deployed ATC'20** | **+78.8%** | +15.6 … +89.0 | 1,999 apps |
| **Derived window vs best of 36 tuned ATC'20 pairs** | **+67.6%** | +26.8 … +76.5 | 600 apps |
| Derived window + conditioning | +68.6% | −20.7 … +89.3 | 1,999 apps |
| Conditioning alone (percentile window) | +26.3% | +19.0 … +30.7 | 1,999 apps |
| **CONTROL — same optimiser, a donor application's history** | **−427.5%** | −1819.8 … −110.4 | 1,999 apps |

**Read the control first.** Handing the identical optimiser — same cost model,
same price, same simulator, same accounting — *another application's* gap
histogram does not merely erase the gain. It lands at −427%, several times worse
than the baseline it was beating. Only the ownership of the evidence changed.
Whatever the derived window is doing, it is not producing window sizes that are
good on average; it is reading something specific out of each application's past.

Three further things this repository reports because they are true, not because
they help:

- **The two methods do not compose.** Together they give +68.6%, worse than
  deriving alone. A percentile survives sparse counts; a cost-optimal window
  integrates the whole distribution and needs its tail estimated well; and
  conditioning fragments exactly the counts that estimate that tail.
- **The gain has a boundary.** It falls from ~76% at tight memory budgets to
  +26.8% at the loosest tested. When memory is nearly free there is little left
  for an optimiser to recover.
- **Four earlier approaches failed**, including a 0.98-AP neural predictor that
  was 60–119% *worse* than a plain histogram once converted into a policy. See
  [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md).

## Why the method was chosen, not assumed

Before building any policy we measured how predictable the problem is at all,
model-free, using mutual information:

**88.95% of the next idle gap's entropy is irreducible** from any context a
scheduler can observe. The remainder is localised almost entirely in one coarse
discrete variable — the length of the gap that just closed (11.0%), ahead of
recent activity (8.6%), time of day (7.6%) and day of week (3.7%).

That single measurement explains all four earlier failures at once and selects
the estimator: when the informative key is one coarse discrete variable, an
exact-match partition has thousands of samples per cell and a learned embedding
with eight neighbours has eight.

**And the mechanism is validated, not just the outcome.** Grouping applications
by how much their consecutive gaps actually depend on each other — measured on
training data only — the gain is **−0.6%** where they carry no information and
**+62.4%** where they carry the most, correlation **+0.99** across strata
(313 applications). The method helps exactly where the theory says it must.

## How this was measured

The evaluation rules matter more than the result, and they are enforced in code:

- **Matched cost, always.** Every policy is swept over its own cost knob and
  arms are compared by interpolating each curve at the baseline's own operating
  points. An earlier version of this work showed a 38% improvement that vanished
  entirely once the budget was matched.
- **Temporal splits, never shuffled.** Train days 1–8, validate 9–10, test 11–14.
  Plus application-level splits for the transfer experiments.
- **Causal by construction.** A decision at minute *t* may only use gaps that
  closed strictly before *t*. There is a test that mutates the trace after *t*
  and asserts no earlier decision moves.
- **One hyper-parameter, chosen on validation.** The estimator's prior strength
  was swept on days 9–10 and carried to the test days unchanged, at 25. The
  setting that scores *best on test* ranks **last** on validation — selecting on
  the test period would have reported the largest number in this project for its
  worst-generalising estimator.
- **Every claim carries a control**, and the controls are in the main tables,
  not an appendix. [`docs/WHAT_THE_CONTROLS_CAUGHT.md`](docs/WHAT_THE_CONTROLS_CAUGHT.md)
  catalogues ten occasions on which one reversed a conclusion we would otherwise
  have published.

## Quickstart

```bash
git clone https://github.com/ColdStart-Lab/RACRP.git && cd RACRP
make setup          # pip install -e ".[demo,docs,dev]"
make test           # 53 tests, no dataset required
```

The test suite builds its own synthetic traces, so it runs anywhere. To
reproduce the measured results you need the trace — see
[`data/README.md`](data/README.md) — then:

```bash
python scripts/run_characterise.py    # parse the trace once, ~10 min
make headline                         # the two headline experiments
make verify                           # check the deliverables against the tables
```

Full pipeline, ordering and runtimes: [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

### The interactive demonstration

```bash
make demo           # http://127.0.0.1:8000
```

Three acts, each a single claim you watch happen rather than read:

1. **Decide** — the gap histogram with the cost curve drawn over it and its
   minimum marked. Drag the price of a cold start: the curve reshapes, the
   minimum slides, our window follows it, and ATC'20's window does not move at
   all.
2. **Race** — both policies run the same four days with a moving playhead, cold
   starts struck through in red as they land, counters climbing live.
3. **Prove** — one button hands the application someone else's history and
   re-derives everything else unchanged.

Acts 1 and 2 run from the committed cache, so the demonstration works
immediately after cloning. The live price knob and the donor swap re-simulate on
the real trace per request, so those two need the dataset; without it the page
says so and falls back to the cache rather than breaking.

## Repository layout

```
racrp/
├── baselines/
│   ├── histogram.py      ATC'20 hybrid histogram — the baseline being beaten
│   ├── conditional.py    METHOD A — state-conditioned, partially pooled
│   ├── hazard.py         METHOD B — the derived window (headline)
│   ├── selective.py      per-application selection and regime weighting
│   ├── fleet.py          cross-application transfer   [negative result]
│   └── episodic.py       per-application episodic memory [negative result]
├── data/                 trace parsing, splits, windowing, features
├── sim/                  minute-tick simulator and the policy family
├── eval/                 metrics, matched-cost interpolation, strata
├── models/               retrieval stack                 [negative result]
└── web/                  FastAPI server for the demonstration

scripts/                  one entry point per experiment; each writes JSON
tests/                    53 tests, no dataset required (4 need optional torch)
artifacts/tables/         every reported number, as JSON — committed
artifacts/figures/        every figure — committed
docs/                     method, results, reproduction, experiment log
overview/                 a from-scratch explainer (markdown, docx, pdf)
paper/                    the project report and the review deck
```

The code for the failed approaches is kept on purpose. The report cites those
results, and deleting the code would make four reported negative results
unreproducible — a worse outcome than the tidiness it would buy.

## Documentation

| | |
| --- | --- |
| [`docs/METHOD.md`](docs/METHOD.md) | both algorithms, derived from first principles |
| [`docs/RESULTS.md`](docs/RESULTS.md) | every number, its control, and the file it comes from |
| [`docs/REPRODUCING.md`](docs/REPRODUCING.md) | exact pipeline order and runtimes |
| [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) | the four approaches that failed, and why |
| [`docs/WHAT_THE_CONTROLS_CAUGHT.md`](docs/WHAT_THE_CONTROLS_CAUGHT.md) | ten times a control reversed a conclusion |
| [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) | the full record, E0–E18, in order |
| [`overview/PRESENTATION_GUIDE.md`](overview/PRESENTATION_GUIDE.md) | the whole project explained assuming no background |

## Limitations

Stated here rather than buried, because they bound what the numbers mean.

- **Simulation, not deployment.** The trace records execution time but not
  cold-start latency, so cold starts are counted in a minute-tick simulator using
  a published latency model. Nothing was deployed to a real platform.
- **One warm instance per application**, modelling aggregate concurrency as the
  ATC'20 and FaasCache evaluations do. Real platforms may hold several.
- **One trace.** Azure Functions 2019. Transfer to another provider is untested;
  carrying the rule to the Huawei trace (Joosen et al., EuroSys 2025) is the
  obvious next step.
- **Minute granularity**, which is all the data has. Sub-minute effects are
  invisible here.
- **The per-application and fleet-level pictures differ.** The headline matches
  *total* memory across applications, which lets memory be reallocated between
  them. Most individual applications barely cold-start at all; the top 10% of
  applications carry 68% of all cold starts, and that is where the reduction
  happens.

## Authors

Joint work by **Parth Goyal** (23BCE0411) and **Saanvi Sharma** (23BDS0064),
VIT BCSE497J Project-I, Team 15785UG03, supervised by Dr. Narayanan Prashanth.

The repository is hosted under a shared organisation rather than either personal
account, so neither author's name sits in the URL ahead of the other's.

## Citation

```bibtex
@software{goyal_sharma_racrp_2026,
  author  = {Goyal, Parth and Sharma, Saanvi},
  title   = {{RACRP}: deriving the serverless keep-alive window from a cost model},
  year    = {2026},
  url     = {https://github.com/ColdStart-Lab/RACRP},
  version = {1.0.0}
}
```

The baseline this work is measured against:

> M. Shahrad et al., "Serverless in the wild: characterizing and optimizing the
> serverless workload at a large cloud provider," in *Proc. USENIX ATC*, 2020,
> pp. 205–218.

## Licence

Code and documentation: [MIT](LICENSE).
The Azure Functions Trace 2019 is not redistributed here; it is published by
Microsoft under CC-BY 4.0. See [`data/README.md`](data/README.md).
