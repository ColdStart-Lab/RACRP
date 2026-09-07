"""Phase 0.4 - reproduce the ATC'20 characterisation as a parsing correctness check.

If our app-level matrix is built correctly, these statistics must land close to
the numbers published in Shahrad et al. (USENIX ATC 2020) for the same trace:

  * invocation skew: a small minority of applications issue almost all invocations
  * most applications are invoked very infrequently (~ once per minute or less)
  * idle times are long and heavy-tailed -- around half of applications have an
    average idle time of tens of minutes, which is exactly the regime where a
    fixed keep-alive window is either wasteful or useless
  * HTTP is the dominant trigger

Anything wildly off here means the parser is wrong and every later number would
be built on sand, so this runs before any modelling.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from racrp.config import FIGURES, TABLES, N_MINUTES, MIN_PER_DAY
from racrp.data.trace import build_app_matrix, idle_times, TRIGGERS


def main() -> None:
    counts, meta = build_app_matrix()
    n_apps, T = counts.shape
    total = meta["total_invocations"].to_numpy().astype(np.float64)
    active = meta["active_minutes"].to_numpy().astype(np.float64)

    order = np.argsort(-total)
    cum = np.cumsum(total[order]) / total.sum()
    share_for_99 = float(np.searchsorted(cum, 0.99) + 1) / n_apps
    share_for_95 = float(np.searchsorted(cum, 0.95) + 1) / n_apps
    top1_share = float(cum[max(int(0.01 * n_apps) - 1, 0)])

    rate = total / N_MINUTES
    frac_le_1_per_min = float((rate <= 1.0).mean())

    # average idle time per app (minutes between consecutive active minutes)
    rng = np.random.default_rng(0)
    sample = rng.choice(n_apps, size=min(6000, n_apps), replace=False)
    avg_it, med_it = [], []
    for r in sample:
        its = idle_times(counts[r])
        if len(its):
            avg_it.append(its.mean()); med_it.append(np.median(its))
    avg_it = np.array(avg_it); med_it = np.array(med_it)

    trig_counts = meta["trigger"].value_counts(normalize=True).to_dict()
    trig_inv = (meta.groupby("trigger")["total_invocations"].sum()
                / meta["total_invocations"].sum()).to_dict()

    # what a fixed keep-alive would leave on the table, per app
    frac_it_gt = {k: float((avg_it > k).mean()) for k in (5, 10, 20, 30, 60, 120)}

    stats = {
        "n_apps": int(n_apps),
        "n_minutes": int(T),
        "total_invocations": float(total.sum()),
        "app_share_producing_99pct_invocations": share_for_99,
        "app_share_producing_95pct_invocations": share_for_95,
        "invocation_share_of_top_1pct_apps": top1_share,
        "frac_apps_at_most_1_invocation_per_minute": frac_le_1_per_min,
        "median_app_duty_cycle": float(np.median(active / T)),
        "median_avg_idle_time_min": float(np.median(avg_it)),
        "p75_avg_idle_time_min": float(np.percentile(avg_it, 75)),
        "p90_avg_idle_time_min": float(np.percentile(avg_it, 90)),
        "frac_apps_avg_idle_time_gt": frac_it_gt,
        "trigger_share_of_apps": trig_counts,
        "trigger_share_of_invocations": trig_inv,
        "memory_mb_median": float(meta["memory_mb"].median()),
        "avg_duration_ms_median": float(meta["avg_duration_ms"].median()),
    }
    out = TABLES / "characterisation.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(np.arange(1, n_apps + 1) / n_apps * 100, cum * 100, lw=2)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("% of applications (sorted by activity)")
    ax[0].set_ylabel("% of all invocations")
    ax[0].set_title("Invocation skew")
    ax[0].grid(alpha=.3)

    x = np.sort(avg_it)
    ax[1].plot(x, np.arange(1, len(x) + 1) / len(x) * 100, lw=2)
    ax[1].set_xscale("log")
    ax[1].axvline(10, ls="--", c="crimson", label="10-min keep-alive")
    ax[1].set_xlabel("average idle time between invocations (min)")
    ax[1].set_ylabel("CDF of applications (%)")
    ax[1].set_title("Idle times are long and heavy-tailed")
    ax[1].legend(); ax[1].grid(alpha=.3)

    ts = [trig_counts.get(t, 0) * 100 for t in TRIGGERS]
    ax[2].barh(TRIGGERS, ts, color="steelblue")
    ax[2].set_xlabel("% of applications")
    ax[2].set_title("Trigger mix")
    ax[2].grid(alpha=.3, axis="x")

    fig.tight_layout()
    fig.savefig(FIGURES / "characterisation.png", dpi=150)
    print(f"\nwrote {out} and {FIGURES / 'characterisation.png'}")


if __name__ == "__main__":
    main()
