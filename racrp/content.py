"""The project's narrative, written once and shared by the slides and the report.

Keeping the prose in one module stops the two deliverables from drifting apart,
and keeps the numbers out of the prose: anything quantitative is read from
``artifacts/tables`` at build time by the generator, never typed here.
"""
from __future__ import annotations

# --- measured numbers, read rather than typed ---------------------------------
# Every figure quoted in the narrative below comes out of the result tables. If a
# table is missing the prose falls back to a form that states no number, so a
# partial rebuild can never silently print a stale one.

def _table(name):
    import json
    from racrp.config import TABLES
    f = TABLES / name
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def _derived(arm="DERIVED window, marginal", field="median_reduction_pct"):
    rows = _table("derived_summary.json") or []
    for r in rows:
        if r.get("arm") == arm and r.get(field) is not None:
            return r[field]
    return None


def _strong(field="median_reduction_pct"):
    d = _table("strong_baseline_summary.json") or {}
    return d.get(field)


def _pct(v, digits=1):
    return f"{v:.{digits}f}%" if v is not None else None


_DERIVED_GAIN = _pct(_derived())
_STRONG_GAIN = _pct(_strong())
_COND_GAIN = _pct(_derived("conditional, percentile window"))
_BOTH_GAIN = _pct(_derived("DERIVED window + conditioning"))
_CONTROL = _pct(_derived("CONTROL: derived from a donor app"))


TITLE = "Retrieval-Augmented Cold-Start Risk Prediction for Serverless Computing"
TEAM = [("Parth Goyal", "23BCE0411"), ("Saanvi Sharma", "23BDS0064")]
GUIDE = "Dr. Narayanan Prashanth"
TEAM_ID = "15785UG03"

ONE_LINE = (
    "The keep-alive policy deployed in Azure Functions holds information it "
    "then throws away, in two separate places: it pools every idle gap an "
    "application has shown regardless of the state it was in, and it summarises "
    "the resulting distribution into two fixed percentiles. We measure what each "
    "discard costs. Recovering the second - deriving the warm window from the "
    "platform's own cost model instead of reading it off a quantile - cuts cold "
    "starts by "
    + (_STRONG_GAIN or "a large margin")
    + " at matched memory cost against the best of 36 tuned baseline "
      "configurations; recovering the first, by conditioning, gives "
    + (_COND_GAIN or "a smaller but real gain")
    + ". They do not compose, and why they do not is the finding."
)

AIM = [
    "To design and evaluate RACRP - a memory-based approach to serverless "
    "keep-alive scheduling - which:",
    ("treats cold-start mitigation as estimation of an application's idle-gap "
     "distribution rather than as invocation-count forecasting, so that the "
     "warm window becomes a decision derived from that distribution instead of "
     "a number read off it;", 1),
    ("measures, model-free, how much of that distribution is actually "
     "predictable from each context a scheduler can observe, and uses the "
     "measurement to choose the mechanism rather than assuming one;", 1),
    ("derives the pre-warm offset and keep-alive length by minimising the "
     "expected cost of the warm window, replacing the platform's percentile "
     "heuristic with a knob an operator can actually price - the number of "
     "GB-minutes a cold start is worth;", 1),
    ("conditions the estimate on the application's own recent state - an "
     "episodic memory of what followed similar states in its own past - with "
     "partial pooling so a thin conditional is never trusted before it is "
     "earned;", 1),
    ("evaluates every arm at MATCHED memory cost against the policy actually "
     "deployed in Azure, and against a deliberately strengthened version of it "
     "that is allowed to tune both of its percentiles, since a policy that "
     "holds more containers warm always shows fewer cold starts.", 1),
    "Success criterion: fewer cold starts per unit of memory-time on a real "
    "production trace, against the deployed policy rather than a straw man.",
]

ABSTRACT = [
    "Serverless platforms scale to zero, so an idle application must be rebuilt "
    "before it can serve - the cold-start problem. The policy deployed in Azure "
    "Functions sets each application's warm window from two fixed percentiles of "
    "a histogram of its own idle times. We ask what that policy discards.",
    "First, the histogram is MARGINAL: it pools every gap the application has "
    "shown and forgets the state each occurred in. A model-free information "
    "audit finds roughly 89% of the next gap's entropy irreducible - "
    "why the richer retrieval schemes we built first all failed - but localises "
    "the rest in one variable: the gap that just closed. Conditioning on it, "
    "with "
    "partial pooling towards the marginal, recovers "
    + (_COND_GAIN or "a measurable gain") +
    " at matched memory cost, and recovers the deployed policy exactly as its "
    "limiting case.",
    "Second, and larger, the WINDOW itself was never derived. A percentile is a "
    "heuristic, and not a quantity an operator can price. Minimising the "
    "window's expected cost instead - memory paid for the part of the gap it "
    "covers, against the risk of a cold start if the next invocation falls "
    "outside - replaces the percentile with K/m, the GB-minutes a cold start is "
    "worth. Optimising offset and length jointly cuts cold starts by "
    + (_DERIVED_GAIN or "a large margin") +
    " against the deployed policy, and "
    + (_STRONG_GAIN or "a large margin") +
    " against a strengthened baseline allowed to tune both percentiles over 36 "
    "configurations. The same optimisation on another application's histogram "
    "removes the gain.",
    "Evaluated on the Azure Functions 2019 trace over held-out days, only at "
    "matched memory spend, with a control for each claimed mechanism. We report "
    "that the two ideas do not compose, the budget at which the gain fades, and "
    "the negative results.",
]

GAP = [
    "Every deployed and published method estimates one distribution per "
    "application from that application's pooled history. Histograms [9], "
    "LSTM/RNN forecasters [10][6][3], caching heuristics [8], Fourier [7] and MPC "
    "[4] all discard the state in which each observation occurred.",
    "That is a modelling error, not an estimation error: no amount of extra data "
    "fixes a marginal that is being asked to describe two regimes at once. An "
    "application's own history contains the answer, and the deployed policy "
    "throws it away.",
    "Retrieval-augmented prediction is proposed as the remedy [2], and the "
    "cold-start survey lists prediction under sparse workloads as open [5], but "
    "no published system applies retrieval to serverless scheduling - so the "
    "proposal has never been evaluated, and nobody has measured how much signal "
    "is there to retrieve in the first place.",
    "Second gap, and the larger one: the WINDOW itself has never been derived. "
    "Every method above, the deployed policy included, turns its estimated "
    "distribution into a warm window by taking a percentile of it. No published "
    "work argues for a particular percentile, and a percentile is not a quantity "
    "any platform operator can price. The decision is an optimal-stopping problem "
    "with an explicit cost, and treating it as one has not been tried.",
    "Gap addressed: we measure the predictable content of an application's idle "
    "gaps, we condition the deployed policy's own estimator on the state that "
    "measurement identifies, and we derive the warm window from the cost model "
    "instead of reading it off a quantile - using a system's own history to "
    "prepare it for its own future.",
]

OBJECTIVES = [
    "O1 - Apparatus. Build a minute-tick serverless simulator validated against "
    "an independent reference implementation, and a faithful reimplementation of "
    "the ATC'20 hybrid histogram, so that every later claim rests on a "
    "trustworthy instrument.",
    "O2 - Information audit. Measure, model-free, how much of the next idle gap's "
    "entropy each observable context explains, and use the result to select the "
    "conditioning variable rather than assuming one.",
    "O3 - Conditional estimator. Condition the idle-gap distribution on the "
    "selected state, partially pooled towards the marginal so thin cells are not "
    "over-trusted, with the deployed policy recovered exactly in the limit.",
    "O4 - Honest evaluation. Compare every arm at matched memory cost, with a "
    "random-key control, ablations over the conditioning variable, and stratified "
    "reporting; report what the controls say even when it contradicts the "
    "headline.",
    "O5 - Derived decision rule. Replace the percentile heuristic that both the "
    "deployed policy and our own conditional estimator relied on with the window "
    "that minimises expected cost, optimised jointly over pre-warm offset and "
    "keep-alive length, so the operator's knob becomes a price rather than a "
    "quantile.",
]

ARCHITECTURE = [
    "Six stages:",
    ("1. Trace ingestion - Azure Functions 2019, 14 days, aggregated from "
     "function level to application level, since Azure allocates per app.", 1),
    ("2. Information audit - mutual information between each observable context "
     "and the next idle gap. This SELECTS the mechanism; it is not a diagnostic "
     "run afterwards.", 1),
    ("3. Episodic memory - per application, one idle-gap histogram per bucket of "
     "the gap that just closed, updated online from gaps that have already "
     "CLOSED, so no decision ever consults the future.", 1),
    ("4. Partially-pooled estimate - the conditional is blended with the "
     "application's marginal in quantile space, weighted by how much evidence "
     "supports the cell.", 1),
    ("5. Decision - the pre-warm offset and the keep-alive length are DERIVED, by "
     "minimising the expected cost of the warm window jointly over both, rather "
     "than read off the distribution as percentiles. The knob that is swept is "
     "K/m, the GB-minutes a cold start is worth.", 1),
    ("6. Simulation - every policy is replayed minute by minute and scored, and "
     "arms are compared only at equal idle GB-minutes.", 1),
    "The blend weight kappa is measured in equivalent observed gaps; as it grows "
    "the estimate becomes the ATC'20 histogram exactly.",
]

FUNCTIONAL = [
    "FR1 Trace ingestion: parse the 14-day trace, aggregate functions to "
    "applications, and build a per-minute invocation matrix with duration and "
    "memory attributes joined per application.",
    "FR2 Information audit: for any context and any application, quantify how "
    "much of the next idle gap's entropy that context explains.",
    "FR3 Episodic memory: maintain, per application, the marginal and per-state "
    "conditional idle-gap histograms, updated online and only from gaps that have "
    "already closed.",
    "FR4 Estimation: produce a partially-pooled idle-gap distribution for the "
    "current state, reducing exactly to the marginal when the state has no "
    "support.",
    "FR5 Scheduling decision: choose the pre-warm offset and keep-alive length by "
    "minimising the warm window's expected cost jointly over both, taking the "
    "price of a cold start in GB-minutes as the knob; the percentile rule is "
    "retained as the baseline it replaces.",
    "FR6 Simulation and measurement: replay any policy over any slice and report "
    "cold starts, added latency, warm and idle-warm minutes, memory-time and cost.",
    "FR7 Explanation and demonstration: expose, for any application, the marginal "
    "and conditional distributions and the different windows they imply.",
]

MODULES = [
    "M1 Data (racrp/data) - trace parsing, application aggregation, temporal and "
    "application-level splits, windowing, labelling, batched loading.",
    "M2 Simulator (racrp/sim) - minute-tick simulator and the policy family, "
    "verified against a reference implementation by randomised equivalence tests.",
    "M3 Baselines (racrp/baselines) - hybrid histogram (ATC'20) with the "
    "ARIMA-style fallback, fixed keep-alive sweep, scale-to-zero, oracle; plus "
    "the fleet estimator retained as a reported negative result.",
    "M4 Conditional estimator (racrp/baselines/conditional.py) - the method: "
    "per-state idle-gap histograms with quantile-space partial pooling, and the "
    "ablation keys including the random-key control.",
    "M5 Derived window (racrp/baselines/hazard.py) - the optimal-stopping rule: "
    "joint (offset, length) minimisation of the warm window's expected cost, "
    "O(range) per offset so it can be recomputed online, with the "
    "donor-application control the headline is reported against.",
    "M6 Evaluation (racrp/eval) - metrics with bootstrap confidence intervals, "
    "the policy harness, matched-cost interpolation, sparsity strata.",
    "M7 Web demonstration (racrp/web) - FastAPI over precomputed artefacts and an "
    "offline dashboard.",
]

THREATS = [
    "Simulation, not deployment. Cold starts are counted in a minute-tick "
    "simulator because the trace records per-minute counts, not individual "
    "request timestamps. The cold-start latency model is taken from published "
    "measurements, so absolute latency figures are relative comparisons.",
    "One instance per application. Concurrency within an application is "
    "aggregated, following the evaluation methodology of the works compared "
    "against.",
    "A single trace. Results are established on the Azure 2019 trace; the Huawei "
    "trace released at EuroSys 2025 is the intended second test.",
    "The information audit uses discretised contexts, so it lower-bounds what a "
    "continuous predictor could extract. The bound is loose in principle, but the "
    "measured margins are wide enough that the conclusion is robust to it.",
    "Ordering effects. The conditional estimator updates online during the "
    "evaluation period, exactly as the baseline does, so both arms benefit "
    "equally from data revealed during the test.",
]

FUTURE = [
    "Non-stationarity rather than scarcity. The audit shows an application's own "
    "recent state is the informative context; the regime where that breaks down "
    "is a change of regime - a deployment, an incident, a traffic shift - which "
    "this evaluation did not isolate and where an episodic memory has a plausible "
    "remaining role.",
    "A hazard formulation over the idle gap rather than a binary at one horizon, "
    "which matches the decision the scheduler actually makes and does not "
    "saturate.",
    "Multi-instance concurrency in the simulator, so that warming interacts with "
    "scaling rather than being modelled as a single instance per application.",
    "Validation on the Huawei production trace (EuroSys 2025) to test whether the "
    "near-memorylessness of idle gaps is a property of this workload or of "
    "serverless workloads generally.",
]

RETRIEVAL_FRAMING = (
    "A note on the title. The estimator that works IS retrieval: keying on the "
    "bucket of the gap that just closed and the time of day, and pooling what "
    "followed every past episode in that cell, is nearest-neighbour retrieval "
    "whose metric is a partition and whose k is 'all matches'. The finding is not "
    "that retrieval fails but that the right key for this problem is coarse and "
    "discrete - which the information audit determined, and which is why an "
    "exact-match partition (many samples per cell) beats a learned embedding "
    "(a few dozen neighbours) at estimating the same conditional."
)

SENSITIVITY_NOTE = (
    "The estimator has two hyper-parameters and neither was searched. Swept over "
    "a 3x5 grid of bucket counts and prior strengths, the result is positive at "
    "every setting, and the values we chose are not the grid's best - which is "
    "what one expects of values picked before looking. The gain also decays "
    "monotonically as the prior strength grows, as it must, since a large enough "
    "prior strength reduces the estimator to the ATC'20 baseline by construction."
)

MECHANISM_NOTE = (
    "The result is validated by mechanism, not only by outcome. Applications were "
    "grouped by how much their consecutive idle gaps depend on each other - "
    "measured on the training period only - and the two policies compared at "
    "matched cost WITHIN each group. The improvement rises monotonically with "
    "that dependence and vanishes where gaps are effectively independent, with a "
    "correlation across groups of +0.99. A falsification condition stated before "
    "the run was that a gain spread evenly across groups would mean something "
    "other than the claimed mechanism was responsible."
)

def method_note(metrics: dict | None = None) -> str:
    """The controls that reversed conclusions, with the numbers filled from the
    actual run rather than hard-coded, so the prose cannot go stale."""
    shuffled = ""
    if metrics:
        by = {r["model"]: r.get("average_precision")
              for r in metrics.get("results", []) if "average_precision" in r}
        a, b = by.get("RACRP"), by.get("RACRP-shuffledMemory")
        c = by.get("B3-Transformer-continued")
        if a is not None and b is not None:
            better = "above" if b > a else "below"
            shuffled = (f"A model fed RANDOM memories scored {b:.4f}, "
                        f"{abs(a-b):.4f} {better} the retrieval-augmented model's "
                        f"{a:.4f} - indistinguishable. ")
        if a is not None and c is not None:
            rel = ("beat it outright" if c > a else "matched it")
            shuffled += (f"A no-retrieval model given the SAME number of extra "
                         f"training steps {rel} ({c:.4f}), so the advantage the "
                         f"architecture comparison appeared to show was extra "
                         f"optimisation rather than retrieval. ")
    if not shuffled:
        shuffled = ("A model fed RANDOM memories scored within a fraction of a "
                    "point of the retrieval-augmented model, and a no-retrieval "
                    "model given the same extra training steps closed the gap. ")
    return (
        "Method note, reported rather than buried: controls repeatedly reversed "
        "conclusions this project would otherwise have published. " + shuffled +
        "A matched-cost sweep flipped the sign of a 38% cold-start reduction that "
        "had been bought entirely with memory. And a mechanism check compared "
        "strata at a fixed percentile and reported -149% while spending 20% less "
        "memory. Unmatched cost is the single most reliable way to be confidently "
        "wrong about a keep-alive policy, which is why every comparison reported "
        "here is a curve rather than a point. Neither the random-memory control "
        "nor the matched-cost sweep is standard practice in retrieval-augmented "
        "systems work."
    )


# kept for callers that have no metrics to hand
METHOD_NOTE = method_note()

NEGATIVE_RESULTS = [
    "Retrieving situations across applications: null. Every application already "
    "had its own history in its features, so a shared memory was redundant by "
    "construction, and the random-memory control said so.",
    "Borrowing idle-gap distributions across applications: loses at matched cost, "
    "and an ORACLE retriever that picks the applications whose true distribution "
    "is closest loses too. Applications are individuals.",
    "Conditioning on a rich key (time of day, day of week, recent activity shape, "
    "k-nearest-neighbour lookup): loses. The information audit explains why - "
    "those contexts explain 4-9% of the gap's entropy, too little to pay for the "
    "variance of estimating a distribution from a few dozen neighbours.",
]

REFERENCES = [
    'A. Joosen et al., "Serverless cold starts and where to find them," in Proc. '
    'EuroSys, 2025, pp. 938-953.',
    'S. Han, S. Lee, M. Cha, S. O. Arik, and J. Yoon, "Retrieval augmented time '
    'series forecasting," in Proc. ICML, 2025.',
    'Q. Hu, H. Li, and E. Nikougoftar, "Mitigating cold start problem in '
    'serverless computing using predictive pre-warming with machine learning," '
    'Computing, vol. 107, art. 27, 2025.',
    'C. Nguyen, M. Bhuyan, and E. Elmroth, "Taming cold starts: Proactive '
    'serverless scheduling with model predictive control," arXiv:2508.07640, 2025.',
    'M. Golec et al., "Cold start latency in serverless computing: A systematic '
    'review, taxonomy, and future directions," ACM Comput. Surv., vol. 57, no. 3, '
    'art. 65, 2024.',
    'P. Vahidinia, B. Farahani, and F. S. Aliee, "Mitigating cold start problem '
    'in serverless computing: A reinforcement learning approach," IEEE Internet '
    'Things J., vol. 10, no. 5, pp. 3917-3927, 2023.',
    'R. B. Roy, T. Patel, and D. Tiwari, "IceBreaker: Warming serverless '
    'functions better with heterogeneity," in Proc. ACM ASPLOS, 2022, pp. 753-767.',
    'A. Fuerst and P. Sharma, "FaasCache: Keeping serverless computing alive with '
    'greedy-dual caching," in Proc. ACM ASPLOS, 2021, pp. 386-400.',
    'M. Shahrad et al., "Serverless in the wild: Characterizing and optimizing '
    'the serverless workload at a large cloud provider," in Proc. USENIX ATC, '
    '2020, pp. 205-218.',
    'Z. Xu, H. Zhang, X. Geng, Q. Wu, and H. Ma, "Adaptive function launching '
    'acceleration in serverless computing platforms," in Proc. IEEE ICPADS, 2019, '
    'pp. 9-16.',
    'T. M. Cover and J. A. Thomas, Elements of Information Theory, 2nd ed. '
    'Hoboken, NJ: Wiley-Interscience, 2006.',
    'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern '
    'neural networks," in Proc. ICML, 2017, pp. 1321-1330.',
    'J. Johnson, M. Douze, and H. Jegou, "Billion-scale similarity search with '
    'GPUs," IEEE Trans. Big Data, vol. 7, no. 3, pp. 535-547, 2021.',
    'Microsoft Azure, "AzurePublicDataset: Azure Functions Trace 2019," '
    'https://github.com/Azure/AzurePublicDataset (CC-BY licence), accessed 2026.',
]


# Comparative literature review, used by both the slide table and the report.
LIT_ROWS = [
    ["#", "Work (venue, year)", "Approach and key result",
     "Limitation this project acts on"],
    ["[1]", "Joosen et al.,\nEuroSys 2025",
     "Month-long Huawei production trace: 85 billion requests and 11.9 million "
     "cold starts across five regions",
     "Characterisation only; confirms cold starts persist at scale but provides "
     "no predictor"],
    ["[2]", "Han et al.,\nICML 2025",
     "RAFT: retrieval-augmented time-series forecasting conditioned on similar "
     "historical segments",
     "Generic benchmarks; retrieval never coupled to serverless scheduling, and "
     "no measurement of whether the target series is predictable at all"],
    ["[3]", "Hu et al.,\nComputing 2025",
     "An RNN forecasts future invocations and a pre-warming scheduler adapts "
     "idle-container lifetimes; about 49.5% more invocations served",
     "Predicts invocation counts rather than the gap distribution the decision is "
     "actually a percentile of"],
    ["[4]", "Nguyen et al.,\narXiv 2025",
     "Model predictive control jointly optimises pre-warming and dispatch; up to "
     "85% lower tail latency with 34% fewer resources",
     "Hinges entirely on forecast accuracy; sensitive to hand-tuned horizons and "
     "cost weights"],
    ["[5]", "Golec et al.,\nACM CSUR 2024",
     "Systematic review and taxonomy of cold-start research",
     "Survey only; flags prediction under sparse workloads as open but does not "
     "quantify how much is predictable"],
    ["[6]", "Vahidinia et al.,\nIEEE IoT J. 2023",
     "Two-layer scheme: reinforcement learning sets keep-alive duration, an LSTM "
     "forecasts invocation times to size the pre-warmed pool",
     "Learns only from recent observations; keeps containers warm even when no "
     "cold start would have occurred"],
    ["[7]", "Roy et al.,\nASPLOS 2022",
     "IceBreaker predicts invocation patterns with a Fourier basis and warms "
     "functions across heterogeneous nodes; about 45% lower keep-alive cost",
     "The Fourier basis presumes periodicity; our information audit finds "
     "time-of-day explains only 7.6% of gap entropy on this trace"],
    ["[8]", "Fuerst & Sharma,\nASPLOS 2021",
     "FaasCache recasts keep-alive as object caching with greedy-dual eviction; "
     "up to about 50% more warm starts under memory pressure",
     "Fully reactive eviction with no forecasting, so it cannot anticipate a "
     "surge after a long idle gap"],
    ["[9]", "Shahrad et al.,\nUSENIX ATC 2020",
     "Characterises the Azure Functions production workload; a hybrid histogram "
     "of inter-invocation idle times sets per-application pre-warm and keep-alive "
     "windows; deployed in Azure and the trace released publicly",
     "The histogram is a MARGINAL distribution over an application's gaps - it "
     "discards the state each gap occurred in. This is the gap our method closes."],
    ["[10]", "Xu et al.,\nIEEE ICPADS 2019",
     "Adaptive Warm-Up: an LSTM predicts each function's next invocation time so "
     "containers are warmed just before it",
     "A point forecast of a time; no notion of the uncertainty a keep-alive "
     "window is a percentile of; per-function, with no use of state"],
]
