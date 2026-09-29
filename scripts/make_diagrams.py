"""Architecture, DFD, use-case, sequence and Gantt diagrams for the report and PPT.

Drawn with matplotlib so they regenerate with the rest of the artefacts and need
no external tool. The report template marks the data-flow and use-case diagrams
as mandatory.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.dates as mdates
import datetime as dt

from racrp.config import FIGURES

BLUE, ORANGE, GREEN, RED, GREY = "#2f6fb0", "#d98a2b", "#3d8f5a", "#c0392b", "#6b7280"


def box(ax, x, y, w, h, text, fc="#eef4fb", ec=BLUE, fs=8.5, bold=False, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q, text=None, color=GREY, style="-|>", rad=0.0, fs=7.5, dy=0.012):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12,
                                 color=color, lw=1.2, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center",
                va="bottom", fontsize=fs, color=color, zorder=3)


def blank(w=12, h=6.4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


# ------------------------------------------------------------------ architecture
def architecture():
    fig, ax = blank(12.4, 6.8)

    box(ax, .02, .78, .19, .15,
        "Azure Functions\nTrace 2019\n24,274 apps x 20,160 min",
        fc="#f3f4f6", ec=GREY, bold=True)
    box(ax, .02, .55, .19, .16,
        "Trace ingestion\nfunctions -> applications\nper-minute matrix\n+ duration, memory, trigger")
    arrow(ax, (.115, .78), (.115, .715))

    box(ax, .26, .55, .21, .30,
        "SITUATION ENCODER\n\n240-min window,\nmulti-scale (1/5/15/60/240)\n"
        "+ cyclical time-of-day, day-of-week\n+ static app descriptors\n\n"
        "strided conv -> transformer\n-> 128-d embedding",
        fc="#eef4fb", ec=BLUE, bold=False)
    arrow(ax, (.21, .63), (.26, .68), "situation (app a, minute t)")

    box(ax, .52, .70, .22, .18,
        "EPISODIC MEMORY BANK\n(training days only)\n\n"
        "per past situation:\nembedding | 60-min outcome | label",
        fc="#fdf3e7", ec=ORANGE, bold=False)
    box(ax, .52, .55, .22, .11,
        "Retrieval\nexact cosine top-k (FAISS)\nself-duplicates excluded",
        fc="#fdf3e7", ec=ORANGE)
    arrow(ax, (.63, .70), (.63, .665))
    ax.text(.645, .681, "keys", fontsize=7, color=GREY, ha="left")
    arrow(ax, (.47, .655), (.52, .625))
    ax.text(.487, .667, "query", fontsize=7, color=GREY, ha="center")

    box(ax, .79, .55, .19, .30,
        "RETRIEVAL-AUGMENTED\nFUSION\n\ncross-attention\nquery = now\nkeys = neighbours\n"
        "VALUES = their OUTCOMES\n\n+ learned memory gate",
        fc="#eaf5ee", ec=GREEN, bold=False)
    arrow(ax, (.74, .595), (.79, .625))
    ax.text(.762, .55, "k episodes +" + chr(10) + "what followed", fontsize=7,
            color=GREY, ha="center", va="top")
    arrow(ax, (.47, .78), (.79, .83), rad=-0.14)
    ax.text(.62, .862, "current representation", fontsize=7,
            color=GREY, ha="center")

    box(ax, .52, .30, .22, .12,
        "Calibrated risk\nP(cold start within dt)\ntemperature scaled on val",
        fc="#eaf5ee", ec=GREEN, bold=True)
    arrow(ax, (.885, .55), (.74, .40), rad=0.2)

    box(ax, .26, .30, .21, .12,
        "Risk-aware policy\nkeep warm iff\nrisk x latency saved > memory cost",
        fc="#fdeaea", ec=RED, bold=True)
    arrow(ax, (.52, .36), (.47, .36))

    box(ax, .02, .30, .19, .12,
        "Minute-tick simulator\ncold starts | added latency\nidle GB-minutes | cost",
        fc="#fdeaea", ec=RED)
    arrow(ax, (.26, .36), (.21, .36))

    box(ax, .02, .07, .45, .14,
        "EVALUATION  —  vs. hybrid histogram (ATC'20), LSTM, transformer without\n"
        "retrieval, fixed keep-alive sweep, oracle bound.  Ablations: random-memory\n"
        "control, retrieval-only vote.  Reported as a cost/benefit Pareto curve.",
        fc="#f3f4f6", ec=GREY, fs=8)
    arrow(ax, (.115, .30), (.115, .21))

    box(ax, .52, .07, .46, .14,
        "WEB DEMONSTRATION  —  serves precomputed artefacts: timeline, per-policy\n"
        "decisions, risk trace, and for any minute the retrieved episodes with what\n"
        "actually followed them.  Runs offline, no GPU.",
        fc="#f3f4f6", ec=GREY, fs=8)
    arrow(ax, (.63, .30), (.70, .21))

    ax.text(.5, .97, "RACRP — system architecture", ha="center", fontsize=13,
            fontweight="bold")
    ax.text(.5, .935,
            "the contribution is the orange + green path: past situations are stored "
            "with their outcomes, and the scheduler attends over those outcomes",
            ha="center", fontsize=8.5, color=GREY, style="italic")
    fig.tight_layout()
    fig.savefig(FIGURES / "architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[diagram] architecture.png")


# -------------------------------------------------------------------------- DFD
def dfd():
    fig, ax = blank(12, 6.2)

    def store(x, y, w, h, label):
        ax.add_patch(Rectangle((x, y), w, h, fc="#fdf3e7", ec=ORANGE, lw=1.3))
        ax.plot([x, x + w], [y + h, y + h], color=ORANGE, lw=1.3)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5)

    def proc(x, y, r, label, n):
        ax.add_patch(Circle((x, y), r, fc="#eef4fb", ec=BLUE, lw=1.3))
        ax.text(x, y + .012, label, ha="center", va="center", fontsize=8, linespacing=1.3)
        ax.text(x, y - r + .022, n, ha="center", fontsize=7, color=BLUE, fontweight="bold")

    box(ax, .01, .60, .13, .10, "Azure trace\n(external)", fc="#f3f4f6", ec=GREY, bold=True)
    box(ax, .01, .18, .13, .10, "Operator /\nreviewer", fc="#f3f4f6", ec=GREY, bold=True)

    proc(.26, .72, .075, "Parse &\naggregate", "1.0")
    proc(.26, .42, .075, "Encode\nsituation", "2.0")
    proc(.50, .72, .075, "Build\nmemory", "3.0")
    proc(.50, .42, .075, "Retrieve\nanalogues", "4.0")
    proc(.72, .42, .075, "Predict\nrisk", "5.0")
    proc(.72, .13, .075, "Decide &\nsimulate", "6.0")
    proc(.92, .13, .062, "Serve\ndemo", "7.0")

    store(.19, .90, .28, .07, "D1  application x minute invocation matrix")
    store(.42, .90, .001, .001, "")
    store(.60, .78, .26, .07, "D2  episodic memory bank (train split)")
    store(.60, .60, .26, .07, "D3  cached neighbour sets + embeddings")
    store(.60, .27, .26, .07, "D4  results: metrics, sweeps, artefacts")

    arrow(ax, (.14, .65), (.19, .72), "daily CSV")
    arrow(ax, (.26, .795), (.30, .90), "matrix")
    arrow(ax, (.29, .90), (.28, .495), "windows + labels", rad=-.25)
    arrow(ax, (.335, .72), (.425, .72), "train situations")
    arrow(ax, (.50, .795), (.62, .85), "embedding, outcome")
    arrow(ax, (.335, .42), (.425, .42), "query embedding")
    arrow(ax, (.60, .81), (.575, .49), "keys", rad=.2)
    arrow(ax, (.55, .47), (.66, .63), "top-k ids", rad=.15)
    arrow(ax, (.575, .42), (.645, .42), "episodes + outcomes")
    arrow(ax, (.72, .345), (.72, .205), "calibrated risk")
    arrow(ax, (.72, .06), (.66, .30), "cold starts, GB-min", rad=.25)
    arrow(ax, (.795, .13), (.858, .13), "artefacts")
    arrow(ax, (.86, .30), (.93, .195), "results", rad=-.2)
    arrow(ax, (.92, .068), (.14, .20), "timeline, decisions,\nretrieved episodes",
          rad=-.12)

    ax.text(.5, .99, "RACRP — data flow diagram (level 1)", ha="center",
            fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "dfd.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[diagram] dfd.png")


# --------------------------------------------------------------------- use case
def usecase():
    fig, ax = blank(10.5, 6.6)

    def actor(x, y, label):
        ax.plot([x], [y + .045], marker="o", ms=9, mfc="white", mec="black")
        ax.plot([x, x], [y + .038, y - .005], color="black", lw=1.2)
        ax.plot([x - .022, x + .022], [y + .022, y + .022], color="black", lw=1.2)
        ax.plot([x, x - .018], [y - .005, y - .045], color="black", lw=1.2)
        ax.plot([x, x + .018], [y - .005, y - .045], color="black", lw=1.2)
        ax.text(x, y - .075, label, ha="center", fontsize=8.5, fontweight="bold")

    def uc(x, y, label, fc="#eef4fb", ec=BLUE):
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((x, y), .30, .085, fc=fc, ec=ec, lw=1.2, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=3,
                linespacing=1.3)

    ax.add_patch(Rectangle((.24, .06), .52, .86, fc="none", ec=GREY, lw=1.2, ls="--"))
    ax.text(.50, .935, "RACRP system", ha="center", fontsize=9.5,
            color=GREY, fontweight="bold")

    actor(.10, .72, "Platform\noperator")
    actor(.10, .30, "Researcher")
    actor(.90, .50, "Serverless\nscheduler")

    cases = [(.50, .855, "Ingest and validate trace"),
             (.50, .745, "Train risk predictor"),
             (.50, .635, "Build episodic memory"),
             (.50, .525, "Predict cold-start risk"),
             (.50, .415, "Decide keep warm / release"),
             (.50, .305, "Simulate and score a policy"),
             (.50, .195, "Compare against baselines"),
             (.50, .095, "Explain a decision (retrieved episodes)")]
    for x, y, l in cases:
        uc(x, y, l, fc="#eaf5ee" if "Explain" in l or "Decide" in l else "#eef4fb",
           ec=GREEN if "Explain" in l or "Decide" in l else BLUE)

    for y in (.855, .525, .415, .095):
        arrow(ax, (.135, .72), (.35, y), color=GREY)
    for y in (.745, .635, .305, .195):
        arrow(ax, (.135, .30), (.35, y), color=GREY)
    for y in (.525, .415):
        arrow(ax, (.865, .50), (.65, y), color=GREY)

    arrow(ax, (.50, .48), (.50, .458), color=BLUE, style="-|>")
    ax.text(.53, .468, "<<include>>", fontsize=6.5, color=BLUE)
    arrow(ax, (.50, .59), (.50, .568), color=BLUE, style="-|>")
    ax.text(.53, .578, "<<include>>", fontsize=6.5, color=BLUE)

    ax.text(.5, .99, "RACRP — use case diagram", ha="center", fontsize=13,
            fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "usecase.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[diagram] usecase.png")


# --------------------------------------------------------------------- sequence
def sequence():
    fig, ax = blank(11.5, 5.8)
    actors = [("Scheduler", .08), ("Encoder", .27), ("Memory bank", .46),
              ("Fusion + head", .65), ("Policy", .84)]
    for name, x in actors:
        box(ax, x - .075, .87, .15, .075, name, fc="#eef4fb", ec=BLUE, fs=8.5, bold=True)
        ax.plot([x, x], [.87, .08], color=GREY, lw=.9, ls="--")

    steps = [(.08, .27, .80, "situation (app a, minute t)"),
             (.27, .46, .71, "query embedding q"),
             (.46, .27, .62, "top-k episodes: embedding, 60-min outcome, label"),
             (.27, .65, .53, "q + retrieved outcomes"),
             (.65, .65, .44, "cross-attention; memory gate g"),
             (.65, .84, .35, "calibrated risk r"),
             (.84, .84, .26, "keep warm iff r x latency saved > memory cost"),
             (.84, .08, .17, "decision + the episodes that justified it")]
    for x0, x1, y, label in steps:
        if x0 == x1:
            ax.add_patch(FancyArrowPatch((x0, y + .03), (x0 + .05, y + .03),
                                         arrowstyle="-", color=BLUE, lw=1.1))
            ax.add_patch(FancyArrowPatch((x0 + .05, y + .03), (x0 + .05, y),
                                         arrowstyle="-", color=BLUE, lw=1.1))
            ax.add_patch(FancyArrowPatch((x0 + .05, y), (x0, y), arrowstyle="-|>",
                                         mutation_scale=11, color=BLUE, lw=1.1))
            ax.text(x0 + .06, y + .012, label, fontsize=7.5, color="#222")
        else:
            ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                         mutation_scale=11, color=BLUE, lw=1.1))
            ax.text((x0 + x1) / 2, y + .014, label, ha="center", fontsize=7.5,
                    color="#222")

    ax.text(.5, .97, "RACRP — sequence of a single keep-warm decision",
            ha="center", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "sequence.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[diagram] sequence.png")


# ------------------------------------------------------------------------ Gantt
def gantt():
    tasks = [
        ("Trace acquisition & parsing", "2026-07-01", "2026-07-14", GREEN),
        ("Characterisation & validation", "2026-07-10", "2026-07-20", GREEN),
        ("Review-1 (literature, gap, objectives)", "2026-07-15", "2026-07-19", GREY),
        ("Simulator + unit tests", "2026-07-20", "2026-08-02", GREEN),
        ("Baseline policies & beat-bar", "2026-07-28", "2026-08-10", GREEN),
        ("Risk task, features, LSTM/Transformer", "2026-08-05", "2026-08-16", GREEN),
        ("Episodic memory & retrieval", "2026-08-12", "2026-08-22", GREEN),
        ("Fusion model, calibration, policy", "2026-08-18", "2026-08-28", GREEN),
        ("Ablations & failure analysis", "2026-08-22", "2026-08-30", ORANGE),
        ("Web demonstration", "2026-08-25", "2026-08-31", ORANGE),
        ("Draft report & Review-2 deck", "2026-08-27", "2026-09-01", ORANGE),
        ("Review-2", "2026-09-02", "2026-09-03", RED),
        ("Project-II: concurrency, drift, 2nd trace", "2026-09-05", "2026-10-25", "#c9ccd1"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (name, s, e, c) in enumerate(reversed(tasks)):
        s = dt.datetime.strptime(s, "%Y-%m-%d"); e = dt.datetime.strptime(e, "%Y-%m-%d")
        ax.barh(i, (e - s).days, left=s, height=.6, color=c, edgecolor="white")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in reversed(tasks)], fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.axvline(dt.datetime(2026, 9, 2), color=RED, ls="--", lw=1.2)
    ax.text(dt.datetime(2026, 9, 2), len(tasks) - .2, " Review-2", color=RED, fontsize=8.5)
    ax.grid(axis="x", alpha=.3)
    ax.set_title("RACRP — project schedule", fontsize=12, fontweight="bold")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "gantt.png", dpi=170)
    plt.close(fig)
    print("[diagram] gantt.png")


def main():
    architecture(); dfd(); usecase(); sequence(); gantt()
    print("[diagrams] done")


if __name__ == "__main__":
    main()
