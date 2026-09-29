"""Generate every figure and table used in the report, PPT and demo.

Reads only from ``artifacts/tables/*.json`` -- nothing here recomputes an
experiment, so the documents can always be rebuilt offline from whatever results
currently exist, and a figure can never disagree with the numbers it came from.
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
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False,
})

STYLE = {
    "scale-to-zero":          dict(c="#999999", marker="x", ls=":"),
    "fixed-keepalive":        dict(c="#4C72B0", marker="o", ls="-"),
    "hybrid-histogram-ATC20": dict(c="#DD8452", marker="s", ls="-"),
    "oracle":                 dict(c="#55A868", marker="*", ls="--"),
    "B2_LSTM":                dict(c="#8172B3", marker="^", ls="-"),
    "B3_Transformer_noRetrieval": dict(c="#937860", marker="v", ls="-"),
    "RACRP":                  dict(c="#C44E52", marker="D", ls="-"),
    "Retrieval_only_vote":    dict(c="#64B5CD", marker="P", ls="--"),
}
PRETTY = {
    "B2_LSTM": "LSTM (AWU-style) [2]",
    "B3_Transformer_noRetrieval": "Transformer, no retrieval (ablation)",
    "RACRP": "RACRP (ours)",
    "Retrieval_only_vote": "Retrieval-only vote (no learned fusion)",
    "hybrid-histogram-ATC20": "Hybrid histogram [1] (ATC'20)",
    "fixed-keepalive": "Fixed keep-alive (swept)",
    "scale-to-zero": "Scale-to-zero",
    "oracle": "Oracle (upper bound)",
}


def load(name):
    p = TABLES / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_pareto(baselines, sweep, out="pareto.png"):
    """The headline figure: cold starts vs. memory-time wasted keeping apps warm.

    Down-and-left is better. A method is only an improvement if its curve lies
    below the baselines at equal cost, so this plot -- not any single number --
    is what the contribution rests on.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ka = sorted([r for r in baselines if r["policy"].startswith("fixed-keepalive")],
                key=lambda r: r["idle_gb_minutes"])
    if ka:
        ax.plot([r["idle_gb_minutes"] / 1e3 for r in ka],
                [r["cold_starts"] / 1e3 for r in ka],
                label=PRETTY["fixed-keepalive"], **STYLE["fixed-keepalive"], ms=5)
        for r in ka:
            if r.get("keepalive") in (1, 5, 10, 30, 120):
                ax.annotate(f"{r['keepalive']}m",
                            (r["idle_gb_minutes"] / 1e3, r["cold_starts"] / 1e3),
                            textcoords="offset points", xytext=(5, 4), fontsize=7,
                            color="#4C72B0")

    for key in ("hybrid-histogram-ATC20", "scale-to-zero", "oracle"):
        r = next((x for x in baselines if x["policy"] == key), None)
        if r:
            ax.scatter([r["idle_gb_minutes"] / 1e3], [r["cold_starts"] / 1e3],
                       label=PRETTY[key], s=110, zorder=5,
                       c=STYLE[key]["c"], marker=STYLE[key]["marker"])

    if sweep:
        df = pd.DataFrame(sweep)
        for model, g in df.groupby("model"):
            g = g.sort_values("idle_gb_minutes")
            st = STYLE.get(model, dict(c="black", marker=".", ls="-"))
            ax.plot(g["idle_gb_minutes"] / 1e3, g["cold_starts"] / 1e3,
                    label=PRETTY.get(model, model), ms=5, **st)

    ax.set_xlabel("wasted memory-time  (thousand GB-minutes, idle but warm)")
    ax.set_ylabel("cold starts  (thousands)")
    ax.set_yscale("log")
    ax.set_title("Cost of warming vs. cold starts avoided\n"
                 "Azure Functions 2019, test days 11-14 — lower-left is better")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / out)
    plt.close(fig)
    print(f"[fig] {FIGURES / out}")


def fig_prediction(metrics, out="prediction_quality.png"):
    """Average precision with the imbalance baseline drawn in, plus calibration."""
    res = [r for r in metrics["results"] if "average_precision" in r]
    if not res:
        return
    names = [PRETTY.get(r["model"].replace("-", "_"), r["model"]) for r in res]
    ap = [r["average_precision"] for r in res]
    ece = [r["ece"] for r in res]
    prior = res[0]["positive_rate"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = ["#C44E52" if "RACRP" in n else "#4C72B0" for n in names]
    ax[0].barh(names, ap, color=colors)
    ax[0].axvline(prior, ls="--", c="grey",
                  label=f"always-predict-prior ({prior:.3f})")
    ax[0].set_xlabel("average precision (area under PR curve)")
    ax[0].set_title("Cold-start risk prediction quality")
    ax[0].legend(fontsize=8)
    for i, v in enumerate(ap):
        ax[0].text(v, i, f" {v:.3f}", va="center", fontsize=8)

    ax[1].barh(names, ece, color=colors)
    ax[1].set_xlabel("expected calibration error (lower is better)")
    ax[1].set_title("Is the risk a real probability?")
    for i, v in enumerate(ece):
        ax[1].text(v, i, f" {v:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / out)
    plt.close(fig)
    print(f"[fig] {FIGURES / out}")


def fig_strata(strata_file="strata.json", out="strata.png"):
    """Where the gains actually come from, by how sparse the application is."""
    data = load(strata_file)
    if not data:
        return
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bands = df["band"].unique().tolist()
    models = df["policy"].unique().tolist()
    w = 0.8 / max(len(models), 1)
    x = np.arange(len(bands))
    for i, m in enumerate(models):
        g = df[df["policy"] == m].set_index("band").reindex(bands)
        ax.bar(x + i * w, g["cold_start_rate_macro"] * 100, width=w,
               label=PRETTY.get(m.replace("-", "_"), m))
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(bands, fontsize=8)
    ax.set_xlabel("application sparsity (median idle time between invocations)")
    ax.set_ylabel("cold-start rate, averaged per app (%)")
    ax.set_title("Cold starts by workload sparsity, at matched memory cost")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / out)
    plt.close(fig)
    print(f"[fig] {FIGURES / out}")


def fig_characterisation_summary(out="motivation.png"):
    """Why a single fixed keep-alive window cannot work for this workload."""
    st = load("characterisation.json")
    if not st:
        return
    frac = st["frac_apps_avg_idle_time_gt"]
    ks = sorted(int(k) for k in frac)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ks, [frac[str(k)] * 100 for k in ks], marker="o", c="#C44E52")
    ax.axvline(10, ls="--", c="grey")
    ax.annotate("typical platform\nkeep-alive (10 min)", (10, 50),
                textcoords="offset points", xytext=(12, 10), fontsize=8, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("keep-alive window (minutes)")
    ax.set_ylabel("% of applications whose average idle time exceeds it")
    ax.set_title("A fixed keep-alive window misses most applications")
    fig.tight_layout()
    fig.savefig(FIGURES / out)
    plt.close(fig)
    print(f"[fig] {FIGURES / out}")


def table_headline(baselines, sweep, metrics, out="headline.json"):
    """The comparison at matched cost: for each learned policy, find the fixed
    keep-alive and histogram settings that spend the same memory-time, and report
    the cold starts each of them incurs."""
    if not (baselines and sweep):
        return
    ka = [r for r in baselines if r["policy"].startswith("fixed-keepalive")]
    hist = next((r for r in baselines if r["policy"] == "hybrid-histogram-ATC20"), None)
    df = pd.DataFrame(sweep)
    rows = []
    for model, g in df.groupby("model"):
        if hist is None:
            continue
        budget = hist["idle_gb_minutes"]
        # the model's operating point closest to the histogram's spend
        cand = g.iloc[(g["idle_gb_minutes"] - budget).abs().argsort()[:1]]
        c = cand.iloc[0]
        best_ka = min(ka, key=lambda r: abs(r["idle_gb_minutes"] - budget))
        rows.append({
            "model": model,
            "matched_budget_idle_gb_min": budget,
            "model_idle_gb_min": float(c["idle_gb_minutes"]),
            "model_cold_starts": int(c["cold_starts"]),
            "histogram_cold_starts": int(hist["cold_starts"]),
            "best_keepalive_cold_starts": int(best_ka["cold_starts"]),
            "keepalive_minutes": best_ka.get("keepalive"),
            "reduction_vs_histogram_pct": 100 * (1 - c["cold_starts"] / max(hist["cold_starts"], 1)),
            "reduction_vs_keepalive_pct": 100 * (1 - c["cold_starts"] / max(best_ka["cold_starts"], 1)),
        })
    (TABLES / out).write_text(json.dumps(rows, indent=2))
    print(f"[table] {TABLES / out}")
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))




# --------------------------------------------------------------- scarcity story
SCARCITY_STYLE = {
    "own-histogram-ATC20": dict(c="#DD8452", marker="s", ls="-",
                                label="Own histogram (ATC'20)"),
    "fleet-prior-only": dict(c="#64B5CD", marker="^", ls="--",
                             label="Fleet prior only (no own history)"),
    "RACRP-shrinkage": dict(c="#C44E52", marker="D", ls="-",
                            label="RACRP shrinkage (ours)"),
}


def fig_scarcity_pareto(sweep, out="scarcity_pareto.png"):
    """Cost/benefit curves on unseen applications, one panel per history budget.

    This is the figure the contribution rests on. Each panel fixes how much of its
    own history a newly seen application has, and plots cold starts against the
    memory-time spent avoiding them. The claim is that borrowing strength wins
    when own history is short, and that the curves converge as it lengthens --
    a method that won in every panel would be a sign of a measurement error, not
    a better method.
    """
    if not sweep:
        return
    df = pd.DataFrame([r for r in sweep if r.get("history_minutes", -1) >= 0])
    if df.empty:
        return
    hs = sorted(df["history_minutes"].unique())
    n = len(hs)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, H in zip(axes, hs):
        g = df[df["history_minutes"] == H]
        for method, st in SCARCITY_STYLE.items():
            gg = g[g["method"] == method].sort_values("idle_gb_minutes")
            if gg.empty:
                continue
            ax.plot(gg["idle_gb_minutes"] / 1e3, gg["cold_starts"] / 1e3,
                    ms=4.5, lw=1.6, **st)
        lab = gg["history_label"].iloc[0] if not gg.empty else str(H)
        ax.set_title(f"own history: {lab}", fontsize=10)
        ax.set_xlabel("idle GB-minutes (thousands)")
        ax.set_yscale("log")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("cold starts (thousands, log)")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Borrowing strength across applications helps exactly when an "
                 "application has little history of its own\n"
                 "600 held-out applications never seen when the fleet bank was built; "
                 "lower-left is better",
                 fontsize=11, y=1.06)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {FIGURES / out}")


def fig_equivalence(matched, out="equivalence_curve.png"):
    """The headline number: how much own history a shared memory is worth."""
    if not matched:
        return
    df = pd.DataFrame(matched)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, scope, title in zip(
            axes, ("early", "full"),
            ("First 6 hours after deployment", "Whole 4-day test period")):
        g = df[df["scope"] == scope].sort_values("history_minutes")
        if g.empty:
            continue
        x = np.arange(len(g))
        ax.plot(x, g["own_histogram"], marker="s", c="#DD8452",
                label="Own histogram (ATC'20)")
        if "RACRP-shrinkage" in g:
            ax.plot(x, g["RACRP-shrinkage"], marker="D", c="#C44E52",
                    label="RACRP shrinkage (ours)")
        if "fleet-prior-only" in g:
            ax.plot(x, g["fleet-prior-only"], marker="^", c="#64B5CD", ls="--",
                    label="Fleet prior only")
        ax.set_xticks(x)
        ax.set_xticklabels(g["history_label"], fontsize=8, rotation=20)
        ax.set_xlabel("how much of its own history the application has")
        ax.set_ylabel("cold starts at matched memory budget")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle("Convergence: the advantage of a shared memory shrinks as an "
                 "application accumulates its own history", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {FIGURES / out}")


def fig_diagnostic(diag, out="retrieval_diagnostic.png"):
    """Why the first framing failed: the aggregate metric hid the whole problem."""
    if not diag:
        return
    reg = diag.get("regimes", {})
    if not reg:
        return
    names = list(reg.keys())
    keys = [("random", "random memories", "#999999"),
            ("raw_history_cosine", "raw history cosine", "#4C72B0"),
            ("learned", "learned embedding", "#C44E52"),
            ("oracle_outcome_space", "oracle (post-horizon)", "#55A868")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(names))
    w = 0.8 / len(keys)
    for i, (k, lab, c) in enumerate(keys):
        vals = [reg[n].get(k, np.nan) for n in names]
        ax.bar(x + i * w, vals, width=w, label=lab, color=c)
    base = [reg[n]["base_rate"] for n in names]
    ax.plot(x + 0.4 - w / 2, base, "k--o", ms=4, lw=1, label="base rate")
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([f"{n}\n(n={reg[n]['n']:,})" for n in names], fontsize=8.5)
    ax.set_ylabel("average precision of a neighbour vote")
    ax.set_title("The learned retrieval key space is worse than raw-history cosine,\n"
                 "and the aggregate metric is dominated by the easy regime",
                 fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / out)
    plt.close(fig)
    print(f"[fig] {FIGURES / out}")


def main() -> None:
    baselines = load("baselines_test.json")
    metrics = load("prediction_metrics_h5.json")
    sweep = load("policy_sweep_h5.json")

    fig_characterisation_summary()
    if baselines:
        fig_pareto(baselines, sweep)
    if metrics:
        fig_prediction(metrics)
    fig_strata()
    if baselines and sweep and metrics:
        table_headline(baselines, sweep, metrics)

    fig_scarcity_pareto(load("scarcity_sweep.json"))
    fig_equivalence(load("scarcity_matched_cost.json"))
    fig_diagnostic(load("diag_retrieval_ceiling_smoke.json")
                   or load("diag_retrieval_ceiling.json"))
    print("[figures] done")


if __name__ == "__main__":
    main()
