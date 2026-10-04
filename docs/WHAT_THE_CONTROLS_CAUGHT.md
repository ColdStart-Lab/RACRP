# What the controls caught

Six times in this project a confident-looking result turned out to be an artefact.
Each was caught by a control, a matched-cost comparison, or simply by running the
thing end to end. They are collected here because the pattern is more useful than
any individual number, and because five of the six would have been publishable as
they first appeared.

| # | what it looked like | what it actually was | what caught it |
|---|---|---|---|
| 1 | Retrieval improves prediction: **+0.0169 AP, p < 0.0001** | the fusion block's extra capacity | a model fed **random memories** scored the same |
| 2 | Borrowing across applications cuts cold starts **38%** | bought entirely with memory | **matched-cost** sweep flipped the sign |
| 3 | The method fails on its own target stratum: **−149%** | it was spending 20% *less* memory there | matched cost **within** each stratum |
| 4 | RACRP beats the no-retrieval transformer | RACRP is warm-started from it and trained longer | a **step-matched** control, which then *beat* RACRP |
| 5 | Risk-to-policy produced a full results table | the risk grid was a sparse sample; the policy was inert | reading a table that was identical in every row |
| 6 | Risk policy: identical cold starts at all 14 λ | it warmed idle minutes; cold starts happen at **active** minutes | a test asserting the policy *responds to its own knob* |

## The two that generalise

**Unmatched cost is the single most reliable way to be confidently wrong about a
keep-alive policy.** It appeared three times (#2, #3, and implicitly #6), and
twice in *opposite* directions — once inventing a +38% gain, once inventing a
−149% loss. A policy that holds more containers warm always shows fewer cold
starts, so any comparison between two policies at different memory spend is
uninterpretable. Every comparison in this project is therefore a curve, and arms
are compared by interpolating at a common budget.

**An architecture comparison silently includes optimisation.** #4 is the sharpest
case: RACRP is initialised from the no-retrieval transformer and then trained
further, so "RACRP vs. transformer" measures retrieval *plus* extra training. One
additional arm — the same model, no retrieval, same extra steps — converted a
significant positive (+0.0036, p < 0.0001) into a significant negative (−0.0011,
p = 1.0000). Neither the random-memory control nor the step-matched control is
standard practice in retrieval-augmented systems work.

## Why #5 and #6 deserve their own mention

Both produced *output*. Neither raised an exception. #5 filled a results table
with numbers that meant nothing because the risk grid had been built from a
sample rather than from every decision minute. #6 returned an identical
cold-start count for all fourteen threshold values while the memory cost varied —
a physically impossible combination that is easy to skim past in a log.

The lesson is narrow and practical: **a metric that does not move when its own
knob moves is a bug, not a finding.** The test that now guards this asserts
exactly that, and would have caught #6 on the first run.

## What this cost, and what it bought

Roughly half the project's experimental effort went into controls and diagnostics
rather than into the method. The return was that the surviving result — a 26.3%
cold-start reduction at matched memory cost, with its mechanism validated at
r = +0.99 against a quantity measured before the policies were run — is one we
can defend under questioning, and that three separate negative results are stated
with the evidence that establishes them rather than quietly dropped.

---

## 8. The optimiser that lost to its own objective

**Caught by:** annotating a figure with the expected cost of each window under
the cost model being minimised.

The derived window's annotated cost was 0.311 against the baseline's 0.238 on the
same application. That is impossible if the optimiser works. It did not: the
Laplace prior added one pseudo-count per bin over a 241-minute support, a 24%
uniform prior, which pulled probability mass into a tail that does not exist and
made the rule stop 12 minutes too early at 38% higher true cost.

Every derived-window number measured before this point was produced by a
handicapped optimiser.

**The lesson is about the figure, not the prior.** The bug had been through unit
tests that pinned the optimiser against a brute-force search of the *same
smoothed objective* — so both sides shared the flaw and agreed perfectly. What
caught it was plotting the result against an external quantity that had no reason
to agree.

## 9. The hyper-parameter that was best on test and worst on validation

**Caught by:** refusing to select on the test period.

Having found the prior was wrong, the obvious next step was to check whether
fixing it helped. It did not: the broken setting scored **+74.3%** on the test
days against **+69.3%** for the corrected one.

Swept on the validation days instead, the ranking inverts completely — the
test-flattering setting comes **last** of seven. Selecting on test would have
picked the worst-generalising point in the grid and reported the largest number
in the project for it.

This is the clearest single argument in this project for holding a validation
split you never look at until you have to choose something.


---

## 10. The control that mattered most

**Caught by:** running the method on the wrong application's history.

The derived window beats the deployed policy by 78.8% at matched memory cost, and
beats a baseline tuned over 36 percentile pairs by 67.6%. Both numbers are large
enough to deserve suspicion, and the obvious sceptical reading is that the rule
simply produces better window sizes on average -- that any application would
benefit from windows of roughly that shape, and nothing is being learned from
anyone's history.

So the identical optimiser, at the identical price ratio, through the identical
simulator, was handed **another application's** gap histogram. Only the ownership
of the evidence changed.

It lands at **-427.5%**: roughly five times worse than the baseline it had been
beating. Not neutral, not merely smaller -- catastrophically worse, because a
cost-optimal window computed for the wrong distribution is confidently wrong
about exactly when to hold and when to release.

That is the number to quote first. It is the difference between "this rule picks
good windows" and "this rule reads each application's own past", and only the
second of those is what the project set out to demonstrate.
