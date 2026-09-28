"""Figures for the derived-window result.

Three panels, each answering the objection a reader raises next:

1. **The frontier.** Cost against cold starts, derived window versus the *best*
   of 36 tuned ATC'20 percentile pairs. Drawn as curves, and drawn far enough to
   the right that the region where the baseline wins is visible rather than
   cropped out.
2. **Why a percentile cannot do this.** One application's idle-gap distribution
   with the percentile pair and the derived window marked on it. The percentile
   sees two points of the curve; the cost model integrates the shape between them.
3. **The two ideas do not compose.** Conditioning and deriving each beat the
   baseline, and their combination beats neither, because they compete for the
   same finite counts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from racrp.config import FIGURES, TABLES, TEST_SLICE, PROCESSED, SEED

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 165, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

BASE, DERIVED, COND, CONTROL = "#4C72B0", "#C44E52", "#DD8452", "#8C8C8C"


def load(name):
    p = TABLES / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_frontier(out="derived_frontier.png"):
    """Derived window against the strongest reachable percentile baseline."""
    cloud = load("strong_baseline_cloud.json")
    matched = load("strong_baseline_matched.json")
    if not cloud or not matched:
        print("  (skipped frontier: run scripts/run_strong_baseline.py)")
        return
    dfc = pd.DataFrame(cloud)
    env = pd.DataFrame(matched).sort_values("budget_idle_gb_min")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))

    ax = axes[0]
    ax.scatter(dfc["idle_gb_minutes"] / 1e3, dfc["cold_starts"] / 1e3,
               s=16, c=BASE, alpha=0.35,
               label="ATC'20, each of 36 percentile pairs")
    ax.plot(env["budget_idle_gb_min"] / 1e3,
            env["best_baseline_cold_starts"] / 1e3,
            "-o", color=BASE, ms=4, lw=2, label="ATC'20, best envelope of the 36")
    ax.plot(env["budget_idle_gb_min"] / 1e3, env["derived_cold_starts"] / 1e3,
            "-o", color=DERIVED, ms=4, lw=2, label="derived window (this work)")
    ax.set_xlabel("idle memory cost (thousand GB-minutes)")
    ax.set_ylabel("cold starts (thousands)")
    ax.set_title("Down and to the left is better")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    colours = [DERIVED if v > 0 else "#B0B0B0" for v in env["reduction_pct"]]
    ax.bar(range(len(env)), env["reduction_pct"], color=colours)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(env)))
    ax.set_xticklabels([f"{v/1e3:.0f}" for v in env["budget_idle_gb_min"]],
                       rotation=45, fontsize=7.5)
    ax.set_xlabel("memory budget (thousand GB-minutes)")
    ax.set_ylabel("cold-start reduction (%)")
    med = float(np.median(env["reduction_pct"]))
    ax.axhline(med, color=DERIVED, ls="--", lw=1, label=f"median {med:.1f}%")
    ax.set_title("Gain at each budget, and where it runs out")
    ax.legend(frameon=False, fontsize=8.5)

    fig.suptitle("The window, derived rather than tuned, against the best "
                 "percentile pair any operator could have picked", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_why(out="derived_why.png"):
    """What a percentile pair sees, and what the cost model sees."""
    from racrp.data.trace import build_app_matrix
    from racrp.baselines.hazard import optimal_window
    from racrp.baselines.histogram import DEFAULT_RANGE

    counts, _ = build_app_matrix()
    rows = np.load(PROCESSED / "cohort.npy")
    rng = np.random.default_rng(SEED)

    # An application with visible structure: most gaps short, but a real
    # secondary mode further out, which is the case where the two rules diverge.
    pick, best_score = None, -1.0
    for r in rng.choice(rows, 600, replace=False):
        inv = counts[r].astype(np.int64)
        a = np.flatnonzero(inv[:TEST_SLICE[0]] > 0)
        if len(a) < 300:
            continue
        g = np.minimum(np.diff(a), DEFAULT_RANGE)
        med = np.percentile(g, 50)
        if not (2 <= med <= 25):
            continue
        # how much mass sits well beyond the median: the tail the percentile
        # rule pays for and the cost model declines to
        score = float((g > 4 * med).mean())
        if 0.02 < score < 0.20 and score > best_score:
            pick, best_score = (int(r), g), score
    if pick is None:
        print("  (skipped why: no suitable application)")
        return
    app, g = pick

    hist = np.zeros(DEFAULT_RANGE + 1)
    np.add.at(hist, g, 1.0)
    pmf = hist / hist.sum()
    p5, p99 = np.percentile(g, 5), np.percentile(g, 99)
    off, ln = optimal_window(hist, 0.05)

    grid = np.arange(len(hist))

    def cost(a, L, ratio=0.05):
        ov = np.clip(np.minimum(a + L, grid) - a, 0, None)
        mem = float((pmf * ov).sum())
        miss = float(pmf[(grid < a) | (grid >= a + L)].sum())
        return mem, miss, ratio * mem + miss

    m_b, k_b, c_b = cost(p5, max(p99 - p5, 1))
    m_d, k_d, c_d = cost(off, ln)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    ax.bar(grid, pmf, width=1.0, color="#B4C7E7", label="observed idle gaps")
    ax.axvspan(p5, p99, color=BASE, alpha=0.18,
               label=f"ATC'20 window: p{5:.0f}={p5:.0f} to p99={p99:.0f} min")
    ax.axvspan(off, off + ln, color=DERIVED, alpha=0.45,
               label=f"derived window: {off} to {off + ln} min")
    ax.set_xlim(0, DEFAULT_RANGE)
    ax.set_yscale("log")
    ax.set_ylim(max(pmf[pmf > 0].min() * 0.7, 1e-5), 1.5)
    ax.set_xlabel("idle gap length (minutes)")
    ax.set_ylabel("probability (log scale)")
    ax.set_title("Full range, log scale: the percentile pair covers\n"
                 f"{p99 - p5:.0f} minutes to reach the last 1% of gaps")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax = axes[1]
    hi = int(min(DEFAULT_RANGE, max(off + ln, np.percentile(g, 90)) * 2.5 + 5))
    ax.bar(grid[:hi], pmf[:hi], width=1.0, color="#B4C7E7")
    ax.axvspan(p5, min(p99, hi), color=BASE, alpha=0.18)
    ax.axvspan(off, off + ln, color=DERIVED, alpha=0.45)
    ax.set_xlim(0, hi)
    ax.set_xlabel("idle gap length (minutes)")
    ax.set_ylabel("probability")
    ax.set_title("Where the mass actually is")
    ax.text(0.97, 0.72,
            f"at K/m = 20 GB-min per cold start\n\n"
            f"ATC'20    {m_b:6.1f} warm min   {100*k_b:4.1f}% miss\n"
            f"derived   {m_d:6.1f} warm min   {100*k_d:4.1f}% miss\n\n"
            f"expected cost  {c_b:.3f}  vs  {c_d:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#BBBBBB"))

    fig.suptitle(f"Application {app}: a percentile sees two points of the curve; "
                 "the cost model integrates the shape between them", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


LABELS = {
    "DERIVED window, marginal": "derive the window\n(marginal)",
    "DERIVED window + conditioning": "derive the window\n+ condition",
    "conditional, percentile window": "condition only\n(percentile window)",
    "CONTROL: derived from a donor app": "CONTROL\ndonor application",
}
COLS = {
    "DERIVED window, marginal": DERIVED,
    "DERIVED window + conditioning": "#9B59B6",
    "conditional, percentile window": COND,
    "CONTROL: derived from a donor app": CONTROL,
}


def fig_composition(out="derived_composition.png"):
    """Two good ideas that compete for the same counts."""
    s = load("derived_summary.json")
    if not s:
        print("  (skipped composition: run scripts/run_derived.py)")
        return
    df = pd.DataFrame(s)
    have = set(df["arm"])
    df = (df.set_index("arm")
            .reindex([o for o in LABELS if o in have])
            .dropna(subset=["median_reduction_pct"])
            .reset_index())

    ctl_name = "CONTROL: derived from a donor app"
    methods = df[df["arm"] != ctl_name]
    control = df[df["arm"] == ctl_name]

    # The control fails by hundreds of percent, so it cannot share an axis with
    # the methods without flattening them into a line. Two panels, each on its
    # own scale, with the break stated rather than disguised.
    fig, axes = plt.subplots(
        1, 2, figsize=(10.6, 4.6),
        gridspec_kw={"width_ratios": [2.5, 1] if len(control) else [1, 0.001]})

    ax = axes[0]
    x = np.arange(len(methods))
    ax.bar(x, methods["median_reduction_pct"],
           color=[COLS[a] for a in methods["arm"]],
           yerr=[methods["median_reduction_pct"] - methods["min"],
                 methods["max"] - methods["median_reduction_pct"]],
           capsize=4, ecolor="#555555", error_kw={"lw": 1})
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in methods["arm"]], fontsize=9)
    ax.set_ylabel("median cold-start reduction vs ATC'20 (%)")
    ax.set_title("Each idea helps; together they help less", fontsize=10.5)
    top = float(methods["max"].max())
    ax.set_ylim(min(-8, float(methods["min"].min()) * 1.15), top * 1.18)
    for xi, v in zip(x, methods["median_reduction_pct"]):
        ax.text(xi, v + top * 0.03, f"{v:+.1f}%",
                ha="center", fontsize=10, fontweight="bold")

    ax = axes[1]
    if len(control):
        v = float(control["median_reduction_pct"].iloc[0])
        lo = float(control["min"].iloc[0])
        ax.bar([0], [v], color=CONTROL, width=0.55)
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks([0])
        ax.set_xticklabels([LABELS[ctl_name]], fontsize=9)
        ax.set_ylim(lo * 1.12, max(60.0, -lo * 0.08))
        ax.text(0, v * 0.55, f"{v:+.0f}%", ha="center", fontsize=11,
                fontweight="bold", color="white")
        ax.set_title("The same optimisation,\nsomeone else's history",
                     fontsize=10.5)
        ax.set_ylabel("(note the different scale)", fontsize=8.5)
    else:
        ax.axis("off")

    fig.suptitle("The derived window needs the tail estimated well; conditioning "
                 "splits the counts that estimate it.\nAnd it needs the "
                 "application's OWN tail: a donor's history is far worse than "
                 "doing nothing clever at all.", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("[figures] derived window")
    fig_frontier()
    fig_why()
    fig_composition()


if __name__ == "__main__":
    main()
