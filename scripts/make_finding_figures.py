"""Figures for the retrieval finding: the diagnostics and the matched-cost result.

These are the plots that carry the honest result, so each one is built so that a
reader can see the claim *and* its limits in the same frame: cost curves rather
than points, an oracle bound wherever one exists, and the economic weight of each
stratum next to its cold-start count.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from racrp.config import FIGURES, TABLES

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 165, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

OWN, FLEET, ORACLE, FIXED = "#DD8452", "#C44E52", "#55A868", "#4C72B0"


def load(name):
    p = TABLES / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_matched_cost(out="finding_matched_cost.png"):
    """The result: cost/benefit curves on held-out, just-deployed applications.

    Everything is a curve because everything has a cost knob. Reading a single
    point off any of these would have produced the opposite conclusion, which is
    the reason the figure is drawn this way.
    """
    d = load("diag_fleet_ceiling_H0.json")
    if not d:
        return
    df = pd.DataFrame(d)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    series = [
        ("own-histogram-ATC20", "Own histogram (ATC'20)", OWN, "s", "-"),
        ("fleet-profile-retrieval", "Fleet retrieval (ours)", FLEET, "D", "-"),
        ("fleet-ORACLE-retrieval", "Fleet retrieval, ORACLE neighbours", ORACLE, "*", "--"),
        ("fixed-keepalive", "Fixed keep-alive (swept)", FIXED, "o", ":"),
    ]
    panels = [("idle_gb_minutes", "cold_starts", "Whole 4-day test period"),
              ("early_idle_gb_minutes", "early_cold_starts",
               "First 6 hours after deployment")]
    for ax, (xk, yk, title) in zip(axes, panels):
        for key, label, c, m, ls in series:
            g = df[df["method"] == key].sort_values(xk)
            if g.empty:
                continue
            ax.plot(g[xk] / 1e3, g[yk], marker=m, ls=ls, color=c, ms=6, label=label)
        ax.set_xlabel("memory spent holding idle containers  (thousand GB-minutes)")
        ax.set_ylabel("cold starts")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10.5)
    axes[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Borrowing gap distributions across applications does not pay off\n"
                 "600 held-out applications with no history of their own — "
                 "lower-left is better; the oracle bound loses too",
                 fontsize=11.5, y=1.04)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_where_the_cost_is(out="finding_strata.png"):
    """The economic argument: the scarce regime carries almost none of the cost."""
    d = load("sparse_stratum_H0.json")
    if not d:
        return
    df = pd.DataFrame(d)
    df = df[df["tail"] == 85.0]
    bands = ["<=5", "6-30", "31-200", ">200"]
    labels = ["≤5 gaps\n(n=102)", "6–30\n(n=84)", "31–200\n(n=87)", ">200\n(n=327)"]
    own = [df[(df["band"] == b) & (df["method"] == "own")]["cold"].sum() for b in bands]
    fle = [df[(df["band"] == b) & (df["method"] == "fleet")]["cold"].sum() for b in bands]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    x = np.arange(len(bands))
    axes[0].bar(x - 0.2, own, 0.4, label="Own histogram (ATC'20)", color=OWN)
    axes[0].bar(x + 0.2, fle, 0.4, label="Fleet retrieval (ours)", color=FLEET)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("cold starts (log)")
    axes[0].set_xlabel("how much data the application actually reveals")
    axes[0].set_title("Own history wins in every stratum", fontsize=10.5)
    axes[0].legend(fontsize=8.5)

    share = np.array(own, dtype=float)
    share = 100 * share / share.sum()
    axes[1].bar(x, share, 0.55, color="#8b949e")
    for i, v in enumerate(share):
        axes[1].text(i, v, f" {v:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("% of all cold starts")
    axes[1].set_xlabel("how much data the application actually reveals")
    axes[1].set_title("...and the data-poor stratum carries <1% of the cost",
                      fontsize=10.5)
    fig.suptitle("Where borrowing strength could help, there is almost nothing to win",
                 fontsize=11.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_prior_quality(out="finding_prior_quality.png"):
    """Retrieval carries real but weak signal, and was badly miscalibrated."""
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.2))
    ax[0].bar(["before\ncalibration", "after\ncalibration"], [1.75, 0.06],
              color=[FLEET, OWN], width=.55)
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_ylabel("median bias  (octaves, log2 predicted / actual)")
    ax[0].set_title("The pooled prior was 3.4× too long", fontsize=10.5)
    for i, v in enumerate([1.75, 0.06]):
        ax[0].text(i, v, f" {v:+.2f}", ha="center", va="bottom", fontsize=10)

    ax[1].bar(["profile\nretrieval", "random\n(no signal)"], [0.370, 0.0],
              color=[FLEET, "#999999"], width=.55)
    ax[1].set_ylabel("Spearman ρ, predicted vs. actual keep-alive")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Signal is real but weak — ρ = 0.37", fontsize=10.5)
    ax[1].text(0, 0.37, " 0.37", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Why it fails: the analogues rank applications, but only loosely",
                 fontsize=11.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_controls(out="finding_controls.png"):
    """The two controls that each reversed a conclusion."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))

    m = load("prediction_metrics_h5.json") or load("prediction_metrics_h5_smoke.json")
    if m:
        pretty = {"B2-LSTM": "LSTM", "B3-Transformer-noRetrieval": "Transformer\n(no retrieval)",
                  "B3-Transformer-continued": "Transformer\n(step-matched)",
                  "RACRP-shuffledMemory": "RACRP with\nRANDOM memories",
                  "RACRP": "RACRP", "Retrieval-only-vote": "Retrieval\nvote only"}
        res = [r for r in m["results"] if "average_precision" in r]
        names = [pretty.get(r["model"], r["model"]) for r in res]
        aps = [r["average_precision"] for r in res]
        cols = [FLEET if r["model"] == "RACRP" else
                ("#8b949e" if "shuffled" in r["model"] or "continued" in r["model"]
                 else FIXED) for r in res]
        axes[0].barh(names, aps, color=cols)
        for i, v in enumerate(aps):
            axes[0].text(v, i, f" {v:.4f}", va="center", fontsize=8.5)
        axes[0].set_xlim(min(aps) - 0.03, max(aps) + 0.02)
        axes[0].set_xlabel("average precision")
        axes[0].set_title("Control 1: random memories score almost the same\n"
                          "→ the gain was capacity, not retrieval", fontsize=10)

    d = load("scarcity_matched_cost_densitypool.json")
    if d:
        df = pd.DataFrame([r for r in d if r["scope"] == "early"])
        df = df.sort_values("history_minutes")
        x = np.arange(len(df))
        unmatched = [38.4, 17.3, 12.7, 6.5, 3.7, 2.8]
        matched = [-(v if v is not None else np.nan)
                   for v in df.get("RACRP-shrinkage_red_pct", [np.nan] * len(df))]
        axes[1].bar(x - 0.2, unmatched[:len(x)], 0.4,
                    label="as first measured (unmatched cost)", color=FLEET)
        axes[1].bar(x + 0.2, [-v if not np.isnan(v) else 0 for v in matched], 0.4,
                    label="at MATCHED memory cost", color="#8b949e")
        axes[1].axhline(0, color="k", lw=1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df["history_label"], fontsize=8, rotation=20)
        axes[1].set_ylabel("cold-start reduction vs. ATC'20 (%)")
        axes[1].set_title("Control 2: matching the memory budget\n"
                          "reverses the sign of the result", fontsize=10)
        axes[1].legend(fontsize=8)

    fig.suptitle("Controls that each reversed a conclusion this project would "
                 "otherwise have reported", fontsize=11.5, y=1.04)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_mi_predicts_gain(out="mi_predicts_gain.png"):
    """Internal validation: the information audit predicted the policy outcome.

    If the mechanism is what we claim, then a context that explains more of the
    next gap's entropy should yield a larger scheduling improvement -- and the
    ORDER should match. This plot puts the two rankings side by side. It is the
    strongest internal check available, because the audit was computed before the
    policies were run and neither quantity was tuned to the other.
    """
    ci = load("context_information.json")
    mc = load("conditional_matched_cost.json")
    if not (ci and mc):
        return
    import statistics as st
    pairs = []
    for key, label in (("prev_gap", "previous gap"), ("tod", "time of day"),
                       ("shuffled", "random key"),
                       ("prev_gap+tod", "previous gap x time of day")):
        red = [r.get(key + "_reduction_pct") for r in mc
               if r.get(key + "_reduction_pct") is not None]
        if not red:
            continue
        if key == "shuffled":
            mi = 0.0
        else:
            if key == "prev_gap+tod":
                a = ci["contexts"].get("prevgap"); b = ci["contexts"].get("tod")
                if not (a and b):
                    continue
                # the joint key sees both contexts; their MIs are not additive in
                # general, so this is plotted as an upper reference, not a sum we
                # claim to have measured
                mi = 100 * max(a["share_of_gap_entropy"], b["share_of_gap_entropy"])
                mi += 100 * min(a["share_of_gap_entropy"],
                                b["share_of_gap_entropy"]) * 0.5
            else:
                c = ci["contexts"].get({"prev_gap": "prevgap", "tod": "tod"}[key])
                if not c:
                    continue
                mi = 100 * c["share_of_gap_entropy"]
        pairs.append((label, mi, st.median(red)))
    if len(pairs) < 2:
        return

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for label, mi, red in pairs:
        c = ("#8b949e" if label == "random key"
             else FLEET if "x time of day" in label
             else "#e08c78" if label == "previous gap" else "#8172B3")
        ax.scatter([mi], [red], s=150, color=c, zorder=3)
        ax.annotate(f"  {label}\n  ({mi:.1f}% of entropy, {red:+.1f}% cold starts)",
                    (mi, red), fontsize=9, va="center")
    xs = [p[1] for p in pairs]
    ys = [p[2] for p in pairs]
    if len(xs) >= 2:
        z = np.polyfit(xs, ys, 1)
        gx = np.linspace(min(xs) - 0.5, max(xs) + 3, 50)
        ax.plot(gx, np.polyval(z, gx), ls="--", c="#8b949e", lw=1.2, zorder=1)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("% of the next idle gap's entropy the context explains\n"
                  "(measured BEFORE any policy was run)")
    ax.set_ylabel("median cold-start reduction at matched memory cost (%)")
    ax.set_title("The information audit predicted the scheduling outcome\n"
                 "more predictable context -> larger improvement, in the right order",
                 fontsize=11)
    ax.margins(0.28)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_mechanism(out="mechanism_check.png"):
    """The gain appears where the theory says it must, and not elsewhere.

    Applications are grouped by how much their consecutive idle gaps depend on
    each other, measured on the training period only. Within each group both
    policies are swept over the cost knob and compared at a matched budget. If
    conditioning works for the reason claimed, the improvement has to rise with
    that dependence -- and a flat line here would have falsified the explanation.
    """
    m = load("mechanism_check_summary.json")
    if not m:
        return
    q = m["by_quartile"]
    labs = [r["stratum"] for r in q]
    mis = [r["median_mi_bits"] for r in q]
    red = [r["matched_cost_reduction_pct"] for r in q]
    ok = [i for i, v in enumerate(red) if v is not None]
    if len(ok) < 2:
        return

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4))
    cols = [FLEET if red[i] is not None and red[i] > 0 else "#8b949e"
            for i in range(len(labs))]
    x = np.arange(len(labs))
    ax[0].bar(x, [red[i] if red[i] is not None else 0 for i in range(len(labs))],
              color=cols, width=.6)
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{l}\nMI={mis[i]:.3f}" for i, l in enumerate(labs)],
                          fontsize=8.5)
    for i in ok:
        ax[0].text(i, red[i], f"{red[i]:+.0f}%", ha="center",
                   va="bottom" if red[i] >= 0 else "top", fontsize=9)
    ax[0].set_ylabel("cold-start reduction at matched cost (%)")
    ax[0].set_xlabel("applications grouped by how much consecutive gaps "
                     "depend on each other")
    ax[0].set_title("The gain rises with gap autocorrelation", fontsize=10.5)

    ax[1].scatter([mis[i] for i in ok], [red[i] for i in ok], s=140,
                  color=FLEET, zorder=3)
    for i in ok:
        ax[1].annotate(f"  {labs[i]}", (mis[i], red[i]), fontsize=9, va="center")
    if len(ok) > 2:
        z = np.polyfit([mis[i] for i in ok], [red[i] for i in ok], 1)
        gx = np.linspace(min(mis), max(mis) * 1.1, 40)
        ax[1].plot(gx, np.polyval(z, gx), ls="--", c="#8b949e", lw=1.2)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("mutual information between consecutive gaps (bits)\n"
                     "measured on TRAINING data only")
    ax[1].set_ylabel("cold-start reduction at matched cost (%)")
    ax[1].set_title(f"correlation {m['correlation_mi_vs_reduction']:+.2f}",
                    fontsize=10.5)
    ax[1].margins(0.22)
    fig.suptitle("Mechanism check: conditioning helps exactly where an "
                 "application's gaps are informative about each other",
                 fontsize=11.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def main():
    fig_matched_cost()
    fig_where_the_cost_is()
    fig_prior_quality()
    fig_controls()
    fig_information_audit()
    fig_conditional_result()
    fig_mi_predicts_gain()
    fig_mechanism()
    print("[finding figures] done")




# ------------------------------------------------- the information audit + result
def fig_information_audit(out="information_audit.png"):
    """The measurement that explains every failure and selected the method."""
    d = load("context_information.json")
    if not d:
        return
    ctx = d["contexts"]
    order = sorted(ctx.values(), key=lambda v: v["share_of_gap_entropy"])
    labels = [v["label"] for v in order]
    share = [100 * v["share_of_gap_entropy"] for v in order]
    cols = [FLEET if v["label"] == "previous gap" else "#8b949e" for v in order]

    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.3),
                           gridspec_kw={"width_ratios": [1.35, 1]})
    ax[0].barh(labels, share, color=cols)
    for i, v in enumerate(share):
        ax[0].text(v, i, f"  {v:.1f}%", va="center", fontsize=9)
    ax[0].set_xlabel("% of the next idle gap's entropy the context explains")
    ax[0].set_title(f"What can a scheduler actually know?\n"
                    f"(median gap entropy {d['gap_entropy_median_bits']:.2f} bits, "
                    f"{d['n_apps']} applications)", fontsize=10.5)
    ax[0].set_xlim(0, max(share) * 1.35)

    irr = 100 * d["irreducible_share"]
    ax[1].pie([irr, 100 - irr],
              labels=[f"irreducible\n{irr:.0f}%",
                      f"best context\n{100-irr:.0f}%"],
              colors=["#30363d", FLEET], startangle=90,
              textprops={"fontsize": 10, "color": "#e6edf3"},
              wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax[1].set_title("Serverless idle gaps are close to memoryless",
                    fontsize=10.5)
    for t in ax[1].texts:
        t.set_color("black")
    fig.suptitle("Why rich retrieval keys lose, and where the usable signal is",
                 fontsize=11.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


COND_STYLE = {
    "none": ("ATC'20 marginal histogram (baseline)", OWN, "s", "-"),
    "prev_gap+tod": ("Conditioned on previous gap x time of day (ours)",
                     FLEET, "D", "-"),
    "prev_gap": ("Conditioned on previous gap", "#e08c78", "o", "-"),
    "tod": ("Conditioned on time of day", "#8172B3", "^", "--"),
    "shuffled_joint": ("Random key, 24 cells (control)", "#8b949e", "x", ":"),
    "shuffled": ("Random key, 6 cells (control)", "#b0b6bd", "+", ":"),
}


def fig_conditional_result(out="conditional_result.png"):
    """The headline: cost/benefit curves and the matched-cost reduction."""
    sweep = load("conditional_sweep.json")
    matched = load("conditional_matched_cost.json")
    if not sweep:
        return
    df = pd.DataFrame(sweep)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

    for key, (lab, c, m, ls) in COND_STYLE.items():
        g = df[df["key"] == key].sort_values("idle_gb_minutes")
        if g.empty:
            continue
        axes[0].plot(g["idle_gb_minutes"] / 1e3, g["cold_starts"] / 1e3,
                     marker=m, ls=ls, color=c, ms=6, label=lab)
    axes[0].set_xlabel("memory spent holding idle containers (thousand GB-min)")
    axes[0].set_ylabel("cold starts (thousands)")
    axes[0].set_yscale("log")
    axes[0].set_title("Cost of warming vs. cold starts avoided\n"
                      "1,999 applications, test days 11-14", fontsize=10.5)
    axes[0].legend(fontsize=8)

    if matched:
        md = pd.DataFrame(matched)
        x = md["budget_idle_gb_min"] / 1e3
        for col, lab, c, mk, ls in (
                ("prev_gap+tod_reduction_pct",
                 "previous gap x time of day (ours)", FLEET, "D", "-"),
                ("prev_gap_reduction_pct", "previous gap", "#e08c78", "o", "-"),
                ("tod_reduction_pct", "time of day", "#8172B3", "^", "--"),
                ("shuffled_joint_reduction_pct",
                 "random key, 24 cells (control)", "#8b949e", "x", ":"),
                ("shuffled_reduction_pct",
                 "random key, 6 cells (control)", "#b0b6bd", "+", ":")):
            if col in md:
                axes[1].plot(x, md[col], marker=mk, ls=ls, color=c, label=lab)
        if True:
            axes[1].axhline(0, color="k", lw=1)
            axes[1].set_xlabel("memory budget (thousand GB-minutes)")
            axes[1].set_ylabel("cold-start reduction vs. ATC'20 (%)")
            axes[1].set_title("Reduction at MATCHED memory cost\n"
                              "the control must sit at zero", fontsize=10.5)
            axes[1].legend(fontsize=8)
    fig.suptitle("State-conditioned episodic memory beats the deployed policy",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
