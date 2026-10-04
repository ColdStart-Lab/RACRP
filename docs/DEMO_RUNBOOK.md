# Review-2 demonstration runbook

Five minutes, offline, no GPU, no network. Read this once the night before and
once an hour before. The numbers below are read live from the artefacts, so if a
figure on screen disagrees with this page, **trust the screen** and say so.

---

## Before you leave

```bash
cd C:\Users\91896\Desktop\os-scheduling
python -m pytest tests/ -q                 # expect: 53 passed
python scripts/verify_deliverables.py      # expect: all checks passed
python -m racrp.web.server                 # leave running
```

Open `http://127.0.0.1:8000` and click one application to warm the page. Leave
the browser open on that tab. **Do not close the terminal** — the server runs in
it.

If the server will not start, the whole demo still works from
`artifacts/figures/*.png`; the ones you need are named in the fallback section.

Have open, in this order, as browser tabs or windows:

1. the demo at `127.0.0.1:8000`
2. `artifacts/figures/information_audit.png`
3. `artifacts/figures/derived_why.png`
4. `artifacts/figures/derived_frontier.png`
5. `artifacts/figures/derived_composition.png`
6. `artifacts/figures/mechanism_check.png` *(only if asked about the 26% result)*

---

## The shape of the story

There is one idea and it has two halves. **The deployed policy holds information
it then throws away.** It throws it away twice, in two different places, and we
measured what each discard costs.

| what is discarded | how we recover it | worth |
|---|---|---|
| the *state* each idle gap occurred in — all gaps pooled into one histogram | condition the histogram on the gap that just closed | **+26.3%** |
| the *shape* of that histogram — summarised into two fixed percentiles | derive the window by minimising its expected cost | **+78.8%** |

Both figures are median cold-start reductions at matched memory cost against the
deployed policy, 1,999 applications, held-out days. Lead with the second. Mention
the first. Never quote either without its control.

---

## The five minutes

### 0:00 — The problem, in one sentence (20 s)

> "Serverless platforms scale to zero, so an idle application has to be rebuilt
> before it can serve. Azure's deployed policy decides how long to hold a
> container warm from a histogram of that application's own idle times: the 5th
> percentile says when to bring it back, the 99th says when to let it go."

> "We asked one question. What does that policy already know and then throw
> away?"

### 0:20 — What we measured before building anything (60 s)

Switch to **information_audit.png**.

> "First we measured what is even predictable, model-free. Roughly **89% of the
> next idle gap's entropy is irreducible** from any context a scheduler can see.
> Serverless idle gaps are close to memoryless."

> "That is why the first three things we built failed — retrieval over a learned
> embedding, transfer across applications, a 0.98-AP risk predictor. We report
> all three as negative results. The audit also told us where the remaining
> signal is: one variable, the length of the gap that just closed, worth 11%."

This is the slide that makes everything after it credible. Do not rush it.

### 1:20 — The window was never derived (70 s)

Switch to **derived_why.png**.

> "Here is one application's idle-gap distribution. The blue span is what ATC'20
> holds — 5th to 99th percentile, sixty minutes, to catch the last one percent of
> gaps. Almost all of that span is empty."

> "Nothing argues that the 99th percentile is right. And a percentile is not
> something an operator can price. So we stopped reading the window off the
> distribution and started **deriving** it."

> "A container warm over a window pays memory for however much of the gap it
> covers, and pays a cold start if the next invocation lands outside it. Minimise
> that. The percentile knob disappears and what you sweep instead is **K/m — how
> many GB-minutes a cold start is worth** — which is a number a platform knows."

Point at the annotation box:

> "That box reports each window's expected cost under the model being minimised.
> It is in the deliverable on purpose: it is the check that caught a real bug in
> our own estimator, and it stays there so anyone can see the optimiser is
> actually beating the heuristic on its own objective."

### 2:30 — The result, and the baseline made as strong as we could (70 s)

Switch to **derived_frontier.png**.

> "Our rule tunes two things where our sweep of the baseline tuned one. That is a
> real asymmetry, so we removed it rather than argue about it. We swept ATC'20
> over **both** its percentiles — 36 configurations, the light blue cloud — and
> took the best frontier any of them reach. That is a stronger baseline than the
> published policy, and we let it be chosen with the test period in view."

> "**+67.6% fewer cold starts at matched memory cost against that**, positive at
> all thirteen budgets. Against the policy as actually deployed, on the full
> 1,999-application cohort, **+78.8%**."

Then, before anyone asks:

> "The right-hand panel shows where it runs out. The gain falls from about 76% at
> tight budgets to 27% at the loosest. When memory is nearly free, holding
> everything warm for a long time is hard to beat — there is nothing left to
> optimise. The gain lives where memory is scarce, which is where operators are."

### 3:40 — Why we believe it (60 s)

Switch to **derived_composition.png**.

> "Two things on this figure. The right-hand bar is the control, and it is the
> number to read first. We ran the **identical** optimiser — same cost model,
> same price ratio, same simulator, same accounting — on **another application's**
> gap histogram. It does not just lose the gain; it lands at **−427%**, about five
> times worse than the baseline it was beating."

> "So this is not a rule that happens to produce good window sizes on average. It
> reads something specific out of each application's own history. That is the
> whole claim of the project: a system's own past, used to prepare it for its
> future."

> "The left-hand panel is the part we did not expect. Conditioning gives +26%.
> Deriving gives +79%. Doing both gives +69% — **worse than either idea alone.**
> A percentile survives sparse counts; a cost-optimal window integrates the whole
> distribution and needs the tail estimated well. Conditioning fragments exactly
> the counts that estimate that tail. Each idea spends what the other depends on."

### 4:20 — Drag the knob (30 s)

Back to the demo, on any application. Take hold of **"What is one cold start
worth?"** and drag it.

> "This is not a replay. Every time I move this, the server re-derives the window
> and re-runs the simulation on the real trace. Watch the window: at 2 GB-minutes
> a cold start is cheap, so it holds for two minutes and takes the misses. At 200
> it holds for hours."

> "Now the baseline's knob, next to it. It moves too — but it is a *percentile*.
> That is the whole argument in one gesture: an operator can tell you what a cold
> start costs them. Nobody can tell you which percentile they want."

Then hit **Play** and let the four days run while you say the closing line.

### 4:50 — Close (20 s)

> "One hyper-parameter had to be chosen. We chose it on the validation days. The
> setting that scores **best** on the test period ranks **last** on validation —
> tuning on the test set would have handed us the biggest number in this project
> for our worst estimator. It is in the report."

---

## The five questions you will be asked

**"Is +78.8% at matched cost, or did you just keep more containers warm?"**
> "Matched cost, and it is the first thing we check. Every policy is swept over
> its own cost knob and arms are compared by interpolating at a common memory
> budget. We learned that the hard way — an earlier version of this work showed a
> 38% improvement that vanished entirely once we matched the budget."

**"Your policy has two free parameters and the baseline has one."**
> "It did, so we removed the asymmetry. We swept the baseline over both of its
> percentiles, 36 configurations, took the lower envelope, and let that envelope
> be picked with the test period in view. We still win by 67.6% median, positive
> at every budget."

**"How do you know it is not overfitting to the test days?"**
> "Temporal splits, never shuffled: train 1–8, validate 9–10, test 11–14. The
> statistics are built online from gaps that have already closed, and there is a
> unit test asserting that changing the trace after minute *t* cannot change any
> decision up to *t*. The one hyper-parameter was chosen on validation, and the
> report shows the sweep — including that the test-flattering setting ranks last
> there."

**"Why not deep learning?"**
> "Because we measured where the signal is before choosing a mechanism. It is one
> coarse discrete variable. When that is true, an exact-match partition has
> thousands of samples per cell and a learned embedding with eight neighbours has
> eight. We built the learned version first; it lost to its own control — a model
> fed *random* memories scored the same as one fed real ones."

**"What did you get wrong?"**
> "Enough that we keep a catalogue of it. Nine times a control or a figure
> reversed something we would otherwise have reported. The one I would point at:
> our optimiser was quietly handicapped by a smoothing prior for weeks, and the
> unit tests never caught it because they pinned the optimiser against a brute
> force search of the *same* flawed objective — both sides shared the bug and
> agreed perfectly. What caught it was plotting the result against a quantity that
> had no reason to agree."

---

## If something breaks

| what fails | do this |
|---|---|
| server will not start | present from the PNGs; say the demo is precomputed and offline, which it is |
| a page is blank | reload once; then fall back to the PNGs |
| a number differs from this page | read the screen aloud and say the runbook is stale — the artefacts are the source of truth |
| the demo's cost box shows a red warning | say so out loud; it means the optimiser is losing on its own objective on that application, which is exactly what the box is there to reveal |
| asked for something not cached | say it is a rerun of `scripts/run_derived.py` and offer to show the code |
| the slider says "live re-simulation unavailable" | the page has fallen back to the cached run and everything else still works; say so and carry on |

**Never** improvise a number. Everything quoted here exists in
`artifacts/tables/*.json`, and every figure is regenerated from those files by
`scripts/finalize.py`.

---

## One-line summary, if you only get one sentence

> Azure's deployed keep-alive policy reads its warm window off two fixed
> percentiles of an idle-time histogram; we derive it instead by minimising what
> the window actually costs, which cuts cold starts by 78.8% at matched memory
> spend — and the same optimiser given another application's history is five times
> *worse* than the baseline, so the gain is genuinely each system's own past
> preparing it for its own future.
