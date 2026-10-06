# Negative results

Four approaches were built, measured, and found not to work. They are documented
here, their code is kept in the repository, and they appear in the project report
and on a slide rather than in an appendix.

They are not an embarrassment. Three of them are the reason the eventual result
is believable, and one of them — the information audit that explains all four —
is arguably the most useful measurement in the project.

---

## 1. Cross-application situation retrieval → null

**Idea.** Let each application learn from the fleet. Embed every application's
recent activity, index the whole fleet, retrieve the most similar past situations
from *other* applications, and fuse them into a transformer that predicts
cold-start risk.

**Code.** `racrp/models/` (encoder, memory, fusion, retrieval cache),
`scripts/run_train.py`, `scripts/diag_retrieval_ceiling.py`.

**Result.** Nothing. Average precision:

| arm | AP |
| --- | --- |
| RACRP with real retrieved memories | 0.9851 |
| the same model fed **random** memories | 0.9854 |
| no retrieval at all, same extra training steps | **0.9862** |

RACRP − step-matched control = **−0.0011**, p = 1.0000.

**What killed it.** The random-memory control. If a model fed random memories
scores the same as one fed real ones, the retrieved information contributed
nothing, and the apparent "retrieval gain" was extra capacity and extra training
steps. The step-matched no-retrieval model then beat it outright.

**Why, in hindsight.** Every test application already had eight days of its own
history compressed into its static features — idle-time quantiles, duty cycle,
rate. That is a per-application memory handed over for free, so a *shared* memory
had nothing left to add. Retrieval was redundant **by construction**. Any future
attempt must limit the application's own history explicitly, or the experiment
cannot answer the question it is asking.

---

## 2. Borrowing gap distributions across applications → loses, and the oracle loses too

**Idea.** An application with little history of its own could borrow the idle-gap
distribution of similar applications — shrinkage toward a fleet prior, with the
deployed policy recovered as "retrieve only from yourself".

**Code.** `racrp/baselines/fleet.py`, `scripts/diag_fleet_ceiling.py`.

**Result.** Worse than the baseline at matched cost, at every history budget
tested.

**What killed it.** We built an **oracle** retriever allowed to peek at the
answer and select the applications whose true gap distributions were genuinely
closest. **The oracle lost too.**

That distinction is what makes this a finding rather than a failure. If the
oracle had won, the problem would have been the retrieval metric and worth more
work. It did not, so the ceiling itself is below the baseline: applications are
individuals, and another application's distribution is not useful evidence about
this one — however well chosen.

**This is the result the headline's donor control later confirmed from the other
direction**: giving the derived window another application's history scores
−427%. Two independent experiments, the same conclusion.

---

## 3. A 0.98-AP risk predictor → 60–119% *worse* as a policy

**Idea.** Predict, per application per minute, the probability of a cold start in
the next *h* minutes. Threshold it to decide what to keep warm.

**Code.** `racrp/models/`, `scripts/run_policy_eval.py`.

**Result.** The predictor is genuinely good: **AP 0.98**, expected calibration
error 0.0042. Converted into a keep-alive policy and compared at matched cost, it
was **60–119% worse than a plain histogram**, on applications where the
conditional estimator was 8–17% *better*.

**The lesson, and it is the most transferable one here.** A high prediction score
is not a good decision. The model was accurate about the 99% of minutes that are
easy and irrelevant — most minutes are quiet, and saying so earns most of the
score. The decision problem lives in the thin tail the metric barely notices.

**A bug worth recording.** The first version of this comparison showed the risk
policy to be *completely inert* — identical cold starts at all 14 thresholds. It
warmed minute *s* when the risk at *s* was high, but risk is only populated at
**idle** minutes, while cold starts occur at **active** ones. The policy was
being scored on a grid that was zero by construction. A high risk at minute *s*
licenses warming *s+1 … s+h*, not *s* itself. Three tests now assert that cold
starts respond to the threshold at all.

---

## 4. Conditioning on a rich key → loses

**Idea.** If conditioning on one coarse variable helps, condition on more: time
of day, day of week, recent activity shape, a k-nearest-neighbour lookup over all
of them.

**Result.** Loses to conditioning on the previous gap alone.

**Why.** The information audit explains it quantitatively. Those contexts explain
**4–9%** of the gap's entropy each — too little to pay for the variance of
estimating a whole distribution from a few dozen neighbours. Splitting data into
more cells costs evidence, and the evidence buys less than it costs.

---

## The measurement that explains all four

`scripts/run_information_audit.py`

Rather than guessing what to try next, we measured how predictable the problem is
at all, model-free, using mutual information. Over 264 applications:

**88.95% of the next idle gap's entropy is irreducible** from any context a
scheduler can observe. Serverless idle gaps are close to memoryless.

The remaining signal is concentrated in one coarse discrete variable — the length
of the gap that just closed, at 11.0%, ahead of recent activity (8.6%), time of
day (7.6%) and day of week (3.7%).

This one number explains every failure above simultaneously:

- **#1 and #2** tried to find signal across applications. There is barely any
  signal to find anywhere, and none of it is shared.
- **#3** optimised a metric dominated by the predictable majority of minutes,
  while the decision depends on the unpredictable minority.
- **#4** spent evidence on contexts that do not carry enough to pay for it.

And it **selected the method that works**: when the informative key is one coarse
discrete variable, the right retrieval is an exact-match partition with thousands
of samples per cell, not a learned embedding with eight neighbours per query. A
conditional histogram *is* retrieval — the metric is a partition and *k* is "all
matches".

---

## Why the code is still here

Deleting it would make four reported results unreproducible, which is a worse
outcome than the tidiness it would buy. The report cites these numbers; a reader
should be able to check them.

`racrp/models/`, `racrp/baselines/fleet.py` and `racrp/baselines/episodic.py` are
therefore kept, clearly labelled, and excluded from the core dependency set —
`torch` is an optional extra (`pip install -e ".[negative-results]"`), because
nothing in the headline result depends on it.
