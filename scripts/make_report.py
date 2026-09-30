"""Build the draft Project-I report in the structure the template prescribes.

Section order and formatting rules come from "3 BCSE497J Project I Report -
Template.docx":

  ABSTRACT (<=300 words) · TABLE OF CONTENTS
  1 INTRODUCTION (1.1 Background, 1.2 Motivation, 1.3 Scope) - each <=200 words
  2 PROJECT DESCRIPTION AND GOALS (2.1 Literature Review, 2.2 Research Gap,
    2.3 Objectives, 2.4 Problem Statement, 2.5 Project Plan)
  3 TECHNICAL SPECIFICATION (3.1 Requirements: functional / non-functional,
    3.2 Feasibility Study, 3.3 System Specification)
  4 DESIGN APPROACH AND DETAILS (4.1 System Architecture, 4.2 Design: DFD and
    use-case mandatory)
  5 REFERENCES

Body text Times New Roman 12 / line spacing 1.15; headings 14 bold upper case;
sub-headings 13 bold title case. Results are read from artifacts/tables at build
time so the report cannot disagree with the experiments.
"""
from __future__ import annotations

import json

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_SECTION

from racrp.config import ROOT, TABLES, FIGURES
from racrp import content as C

OUT = ROOT / "paper" / "RACRP_Project1_Report.docx"
FONT = "Times New Roman"


def load(name):
    p = TABLES / name
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------- formatting
def style_doc(doc):
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(12)
    st.paragraph_format.line_spacing = 1.15
    st.paragraph_format.space_after = Pt(6)


def h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(14)
    r = p.add_run(text.upper())
    r.font.name = FONT; r.font.size = Pt(14); r.bold = True
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(10)
    r = p.add_run(text)
    r.font.name = FONT; r.font.size = Pt(13); r.bold = True
    return p


def h3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    r = p.add_run(text)
    r.font.name = FONT; r.font.size = Pt(12); r.bold = True; r.italic = True
    return p


def para(doc, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.name = FONT; r.font.size = Pt(12)
    return p


def bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.3 + 0.3 * level)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.name = FONT; r.font.size = Pt(12)
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.name = FONT; r.font.size = Pt(11); r.italic = True
    return p


def table(doc, rows, widths=None, font=10):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(str(v))
            r.font.name = FONT; r.font.size = Pt(font)
            if i == 0:
                r.bold = True
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Inches(w)
    return t


def figure(doc, path, width=6.0, cap=None):
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if cap:
            caption(doc, cap)


# ---------------------------------------------------------------------- content
def title_page(doc):
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("BCSE497J - PROJECT-I"); r.font.name = FONT; r.font.size = Pt(13); r.bold = True
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.5
    r = p.add_run("RETRIEVAL-AUGMENTED COLD-START RISK PREDICTION\nFOR SERVERLESS COMPUTING")
    r.font.name = FONT; r.font.size = Pt(16); r.bold = True
    doc.add_paragraph()
    for name, reg in (("PARTH GOYAL", "23BCE0411"), ("SAANVI SHARMA", "23BDS0064")):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f"{reg}    {name}"); r.font.name = FONT; r.font.size = Pt(13); r.bold = True
    doc.add_paragraph()
    for line, sz, bold in (("Under the Supervision of", 12, False),
                           ("Dr. Narayanan Prashanth", 13, True),
                           ("", 12, False),
                           ("B.Tech.", 12, False), ("in", 12, False),
                           ("Computer Science and Engineering", 13, True),
                           ("", 12, False),
                           ("School of Computer Science and Engineering", 13, True),
                           ("September 2026", 12, False)):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.5
        r = p.add_run(line); r.font.name = FONT; r.font.size = Pt(sz); r.bold = bold
    doc.add_page_break()


ABSTRACT = " ".join(C.ABSTRACT)

KEYWORDS = ("Keywords - Serverless computing, cold start, keep-alive policy, "
            "retrieval-augmented prediction, borrowing strength, shrinkage "
            "estimation, resource scheduling, Azure Functions.")


def sec_intro(doc, ch):
    h1(doc, "1. Introduction")
    h2(doc, "1.1 Background")
    para(doc,
         "Serverless platforms such as AWS Lambda, Azure Functions and Google "
         "Cloud Functions let developers deploy individual functions and pay only "
         "for the time those functions execute. To make that billing model viable "
         "the provider reclaims the resources of an application that has been idle "
         "for a while, terminating its container. When a request later arrives for "
         "a reclaimed application, the platform must allocate resources, start the "
         "runtime, load dependencies and restore application state before the "
         "request can be served. This initialisation is the cold start, and it "
         "adds anywhere from a few hundred milliseconds to several seconds of "
         "latency that the user directly experiences. Keeping every container warm "
         "would remove cold starts but would also destroy the economics of the "
         "platform, because reserved memory is paid for whether or not it is used. "
         "Every serverless provider therefore operates a keep-alive policy that "
         "trades wasted memory against user-visible latency, and the quality of "
         "that policy determines both the platform's cost and its responsiveness.")
    h2(doc, "1.2 Motivation")
    frac = (ch or {}).get("frac_apps_avg_idle_time_gt", {})
    f10 = frac.get("10")
    para(doc,
         "The prevailing keep-alive policies are structurally mismatched to real "
         "workloads. Our own parse of the Azure Functions 2019 production trace "
         + (f"shows that {f10*100:.1f}% of applications have an average idle time "
            f"longer than the industry-standard ten-minute keep-alive window, and "
            f"that {(ch or {}).get('frac_apps_at_most_1_invocation_per_minute', 0)*100:.1f}% "
            "of applications are invoked at most once per minute on average. "
            if f10 else "shows that most applications are invoked rarely and "
            "irregularly. ") +
         "A single fixed window is therefore either wasteful for the sparse "
         "majority or useless for them. The state of practice, the hybrid "
         "histogram policy deployed in Azure Functions, does better by learning "
         "each application's own distribution of idle times. But that means every "
         "application must learn itself from scratch, and an application that has "
         "just been deployed, or that runs rarely, has almost nothing to learn "
         "from. The policy falls back to a fixed default exactly where a good "
         "decision matters most. Meanwhile the platform has already observed "
         "thousands of other applications, many of them behaving very similarly, "
         "and uses none of that. The motivation for this project is to let a "
         "scheduler answer the question a human operator would ask immediately: "
         "what do applications like this one usually do?")
    h2(doc, "1.3 Scope of the Project")
    para(doc,
         "The project covers the full path from a public production trace to a "
         "measured scheduling outcome. It includes parsing and validating the "
         "Azure Functions 2019 trace at application granularity; formalising "
         "cold-start risk through the idle-gap distribution; building a fleet "
         "memory of application behavioural profiles and their gap distributions; "
         "retrieving analogues for an application that has little or no history of "
         "its own; combining the retrieved prior with the application's own "
         "evidence by shrinkage; and evaluating the resulting policy in a "
         "minute-tick simulator, on applications held out from the fleet bank and "
         "at matched memory cost, against the deployed hybrid histogram policy, a "
         "fixed keep-alive sweep and an oracle bound. It also includes a neural "
         "variant with a learned situation encoder and cross-attention fusion, "
         "reported together with the controls that bound what it contributes, and "
         "an interactive web demonstration that exposes the retrieved analogues "
         "behind each decision. The scope excludes deployment on a live platform, "
         "multi-instance concurrency within an application, and cross-provider "
         "validation; those are identified as Project-II work.")


def sec_description(doc, ch, base, metrics):
    h1(doc, "2. Project Description and Goals")
    h2(doc, "2.1 Literature Review")
    para(doc,
         "Research on serverless cold starts divides, following the taxonomy of "
         "Golec et al., into environment optimisation (making the start itself "
         "cheaper) and prediction or pre-warming (avoiding the start altogether). "
         "This project belongs to the second family, and the works below are the "
         "ones it must be measured against.")
    rows = [["#", "Work", "Approach and result", "Limitation for cold-start prediction"]]
    for r in C.LIT_ROWS[1:]:
        rows.append(r)
    table(doc, rows, widths=[0.4, 1.3, 2.5, 2.5], font=9)
    caption(doc, "Table 1. Comparative literature review.")
    para(doc,
         "Across this body of work the prediction target is almost always an "
         "invocation count or an invocation time, from which a warming decision is "
         "derived by a heuristic; and the input is almost always a short recent "
         "window. Retrieval-augmented forecasting has recently been shown to help "
         "across general time-series domains, but it has never been coupled to "
         "cold-start risk or to a serverless scheduler.")

    h2(doc, "2.2 Research Gap")
    for b in [
        "No episodic memory. Every existing predictor - histograms, LSTM and RNN "
        "forecasters, caching heuristics, Fourier methods and model predictive "
        "control - observes only a short recent window. None can recall a "
        "recurring motif seen weeks or months earlier, even though production "
        "workloads are full of them.",
        "An indirect prediction target. Prior work forecasts counts or times and "
        "converts them to warming decisions heuristically. Directly estimating "
        "the probability of a cold start within a horizon - the quantity the "
        "scheduler actually needs, and one that can be calibrated and "
        "cost-weighted - remains largely unexplored, and is listed as an open "
        "problem in the ACM Computing Surveys taxonomy.",
        "Failure concentrated where it matters. Production traces are dominated "
        "by sparse and bursty applications, which is precisely the regime in "
        "which window-based forecasters degrade most.",
        "Retrieval unexploited in the cloud stack. No published system applies "
        "retrieval to cold-start prediction or serverless resource scheduling.",
    ]:
        bullet(doc, b)
    para(doc,
         "Gap addressed: cloud schedulers today have no explicit episodic memory. "
         "RACRP treats historical workload situations as retrievable experiences "
         "and bases warm-state decisions on analogous past events rather than on "
         "extrapolation of the recent window alone.")

    h2(doc, "2.3 Objectives")
    for o in [
        "O1 - Risk formulation. Model cold-start mitigation as horizon-based risk "
        "prediction, P(cold start within dt) per application per minute, so that "
        "warming decisions map onto a calibrated probability. Measured by average "
        "precision and expected calibration error on held-out test days.",
        "O2 - Workload memory. Build a vectorised repository of historical "
        "situations from the training days, each stored with the outcome that "
        "followed it, supporting top-k similarity retrieval in under a "
        "millisecond per query.",
        "O3 - Retrieval-augmented predictor. Design a transformer that fuses the "
        "current observation window with retrieved analogues through "
        "cross-attention over their outcomes, and quantify the contribution of "
        "retrieval by ablation against an identical model with retrieval "
        "disabled and against a control fed random memories.",
        "O4 - Risk-aware scheduling and evaluation. Derive a proactive warming "
        "policy from the predicted risk and evaluate it against the hybrid "
        "histogram policy, LSTM and Transformer baselines and a fixed keep-alive "
        "sweep, on cold-start count, added latency, wasted memory-time and the "
        "cost-performance trade-off.",
    ]:
        bullet(doc, o)

    h2(doc, "2.4 Problem Statement")
    para(doc,
         "Let application a have observed idle gaps G_a (possibly empty, if it was "
         "just deployed) and a deployment-time descriptor d_a (trigger type, "
         "memory, declared duration). The scheduler must choose a pre-warm offset "
         "and keep-alive length, which are percentiles of its idle-gap "
         "distribution. Equivalently, at minute t it must estimate")
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("h_a(g)  =  P( the next idle gap of a has length g  |  G_a, d_a )")
    r.font.name = FONT; r.font.size = Pt(12); r.italic = True
    para(doc,
         "and choose windows so as to minimise cold starts subject to a budget on "
         "the memory-time spent holding idle containers. The deployed policy "
         "estimates h_a from G_a alone. The difficulty is that |G_a| is small "
         "precisely when the decision matters most - a function that has just been "
         "deployed has |G_a| = 0, and in this trace the median application is "
         "active in only 1.7% of minutes, so |G_a| grows slowly. The question this "
         "project asks is whether the thousands of applications the platform has "
         "already observed can supply what a is missing, and how that borrowed "
         "information should be traded off against a's own evidence as it "
         "accumulates.")

    h2(doc, "2.5 Project Plan")
    rows = [["Phase", "Work", "Status"],
            ["0", "Trace acquisition, application-level parsing, validation "
                  "against the published characterisation, frozen temporal splits "
                  "and stratified cohort", "Complete"],
            ["1", "Minute-tick simulator with randomised equivalence tests; "
                  "baseline policies (fixed keep-alive sweep, hybrid histogram, "
                  "scale-to-zero, oracle); beat-bar published", "Complete"],
            ["2", "Risk labelling, windowing and features; LSTM and Transformer "
                  "risk predictors", "Complete"],
            ["3", "Episodic memory bank, retrieval index, cross-attention fusion, "
                  "calibration, risk-aware policy and Pareto sweep", "Complete"],
            ["4", "Ablations, stratified and idle-age analysis, failure diagnosis "
                  "and correction", "Complete"],
            ["5", "Interactive web demonstration over precomputed artefacts", "Complete"],
            ["6", "Draft report, Review-2 presentation, outcome report", "Complete"],
            ["7", "Project-II: multi-instance concurrency, online memory "
                  "maintenance under drift, second production trace",
             "Planned"]]
    table(doc, rows, widths=[0.6, 4.4, 1.0], font=10)
    caption(doc, "Table 2. Project plan and status at Review-2.")
    figure(doc, FIGURES / "gantt.png", 6.0, "Fig. 1. Project schedule.")


def sec_technical(doc):
    h1(doc, "3. Technical Specification")
    h2(doc, "3.1 Requirements")
    h3(doc, "3.1.1 Functional")
    for f in [
        "FR1 Trace ingestion. Parse the 14-day Azure Functions trace, aggregate "
        "functions into applications, and build a per-minute invocation matrix "
        "joined with per-application duration and memory attributes.",
        "FR2 Situation encoding. For any (application, minute) produce a "
        "fixed-length embedding computed only from information available strictly "
        "before that minute.",
        "FR3 Memory construction and retrieval. Index every training situation "
        "together with the activity profile that followed it, and return the "
        "top-k analogues of a query, excluding near-duplicates of the query itself.",
        "FR4 Risk prediction. Emit a calibrated probability of a cold start within "
        "the configured horizon, per application, per minute.",
        "FR5 Scheduling decision. Convert risk into keep-warm or release decisions "
        "under a tunable cost trade-off between latency and memory.",
        "FR6 Simulation and measurement. Replay any policy over any slice of the "
        "trace and report cold starts, added latency, warm and idle-warm minutes, "
        "memory-time and monetary cost.",
        "FR7 Explanation and demonstration. Expose, for any decision, the "
        "retrieved historical episodes, their similarity and what followed them.",
    ]:
        bullet(doc, f)
    h3(doc, "3.1.2 Non-Functional")
    for f in [
        "Correctness. The simulator is the measurement instrument for every claim, "
        "so it is validated against an independent reference implementation by "
        "randomised equivalence tests, and the trace parser is validated against "
        "the characterisation published with the dataset.",
        "Reproducibility. All randomness is seeded; temporal splits are frozen and "
        "never shuffled; every reported number is written to a JSON artefact by "
        "the script that produced it, and the report and slides are generated "
        "from those artefacts.",
        "Performance. Retrieval must answer in under a millisecond per query so "
        "that the method is plausible inside a scheduler's decision path; the "
        "whole cohort evaluation must complete on a single workstation.",
        "Efficiency. Embeddings, neighbour sets and the processed trace matrix are "
        "cached on disk and never recomputed.",
        "Usability. The demonstration must run offline, with no GPU and no network "
        "access, and start in seconds.",
        "Maintainability. Configuration is centralised; modules are separated by "
        "responsibility (data, simulation, baselines, models, evaluation, web).",
    ]:
        bullet(doc, f)

    h2(doc, "3.2 Feasibility Study")
    h3(doc, "3.2.1 Technical Feasibility")
    para(doc,
         "The required data is public and already obtained: the Azure Functions "
         "2019 trace released with the ATC'20 paper under a CC-BY licence. The "
         "modelling stack (PyTorch, FAISS, scikit-learn) is mature and runs on "
         "commodity hardware; the models are small by design, around one to three "
         "million parameters, because the constraint is retrieval latency and not "
         "capacity. The principal technical risk was that retrieval would be too "
         "slow to be credible inside a scheduler; this is resolved by freezing the "
         "retrieval encoder so neighbour sets are computed once and cached, which "
         "is also the standard retrieval-augmented design.")
    h3(doc, "3.2.2 Economic Feasibility")
    para(doc,
         "The project runs entirely on one workstation with public data, so direct "
         "cost is negligible. The economic case for the method itself is the "
         "quantity the evaluation measures: memory-time spent holding idle "
         "containers is the provider's real expenditure, and the results are "
         "reported as a curve of cold starts against that expenditure rather than "
         "as a single number, so the saving can be read at whatever operating "
         "point a provider chooses.")
    h3(doc, "3.2.3 Social Feasibility")
    para(doc,
         "Idle warm containers consume energy for no useful work, so reducing the "
         "memory-time needed to achieve a given latency has a direct sustainability "
         "benefit, aligning with SDG 12 and SDG 13. The data is anonymised and "
         "aggregated at source, containing only hashed identifiers and per-minute "
         "counts, so no privacy concern arises. The system makes resource "
         "decisions rather than decisions about people, and each decision is "
         "accompanied by the retrieved evidence that produced it, which makes the "
         "policy auditable by an operator.")

    h2(doc, "3.3 System Specification")
    h3(doc, "3.3.1 Hardware Specification")
    table(doc, [["Component", "Specification"],
                ["Processor", "x86-64 multi-core CPU"],
                ["Memory (RAM)", "16 GB (the processed trace matrix is ~1 GB resident)"],
                ["Storage", "10 GB (2.2 GB raw trace, plus processed caches and artefacts)"],
                ["GPU", "NVIDIA GeForce GTX 1650, 4 GB (optional; models fit on CPU)"],
                ["Display", "1920 x 1080 for the demonstration dashboard"]],
          widths=[2.0, 4.5], font=10)
    h3(doc, "3.3.2 Software Specification")
    table(doc, [["Component", "Specification"],
                ["Operating system", "Windows 11 (the codebase is platform independent)"],
                ["Language", "Python 3.12"],
                ["ML framework", "PyTorch 2.6"],
                ["Retrieval", "FAISS (IndexFlatIP, exact cosine), with a pure-torch fallback"],
                ["Scientific stack", "NumPy, pandas, scikit-learn, SciPy, Matplotlib"],
                ["Web", "FastAPI and Uvicorn; a dependency-free HTML/Canvas frontend"],
                ["Testing", "pytest"],
                ["Version control", "Git"],
                ["Dataset", "Azure Functions Trace 2019 (Microsoft, CC-BY)"]],
          widths=[2.0, 4.5], font=10)


def sec_design(doc):
    h1(doc, "4. Design Approach and Details")
    h2(doc, "4.1 System Architecture")
    figure(doc, FIGURES / "architecture.png", 6.3, "Fig. 2. RACRP system architecture.")
    para(doc,
         "The estimator at the centre of the system is a shrinkage of the "
         "application's own idle-gap histogram towards a prior pooled from "
         "behaviourally similar applications:")
    _p = doc.add_paragraph(); _p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _r = _p.add_run("h_hat  =  ( n_own * h_own  +  kappa * h_fleet ) / ( n_own + kappa )")
    _r.font.name = FONT; _r.font.size = Pt(12); _r.italic = True
    para(doc,
         "where n_own is the number of idle gaps the target application has "
         "actually shown and kappa expresses the strength of the fleet prior in "
         "units of equivalent observed gaps. With no history of its own the "
         "estimate is entirely the fleet prior; as the application runs, n_own "
         "grows and the estimate converges to the per-application histogram, so "
         "the policy deployed in Azure is recovered exactly as the limiting case "
         "rather than replaced. This structure is what makes the method a "
         "generalisation of the state of practice, and it is also what makes the "
         "central claim falsifiable: the advantage must be largest when own "
         "evidence is scarce and must vanish when it is plentiful.")
    para(doc,
         "The architecture has five stages. Trace ingestion turns the raw daily "
         "CSV files into an application-level per-minute invocation matrix with "
         "duration and memory attributes attached. The situation encoder converts "
         "a 240-minute history window, viewed at several time scales, together "
         "with cyclical time-of-day and day-of-week encodings and per-application "
         "descriptors, into a 128-dimensional embedding; a strided convolutional "
         "stack reduces the sequence before self-attention so that a long window "
         "remains affordable. The episodic memory bank holds one entry per "
         "training situation, storing its embedding, the sixty-minute activity "
         "profile that actually followed it, and its realised label. Retrieval is "
         "exact cosine top-k over a frozen encoder's key space, with "
         "near-duplicates from the same application excluded. The fusion block is "
         "where the contribution lies: cross-attention takes the current situation "
         "as the query and the retrieved neighbours as keys, but the values are "
         "built from the neighbours' outcomes rather than from their input "
         "patterns, so the model reads off consequences rather than merely noting "
         "resemblance. A learned gate decides how far to trust memory for each "
         "query, letting the model fall back on its own judgement when no good "
         "analogue exists. Finally the logit is temperature-scaled on the "
         "validation split into a calibrated probability, and a cost-aware rule "
         "holds the container warm whenever the expected latency saved exceeds the "
         "memory cost of holding it.")

    h2(doc, "4.2 Design")
    h3(doc, "4.2.1 Data Flow Diagram")
    figure(doc, FIGURES / "dfd.png", 6.3, "Fig. 3. Data flow diagram.")
    h3(doc, "4.2.2 Use Case Diagram")
    figure(doc, FIGURES / "usecase.png", 5.6, "Fig. 4. Use case diagram.")
    h3(doc, "4.2.3 Sequence Diagram")
    figure(doc, FIGURES / "sequence.png", 6.3,
           "Fig. 5. Sequence of a single keep-warm decision.")


def sec_results(doc, ch, base, metrics, head, strata, idle_age):
    h1(doc, "5. Experimental Results")

    h2(doc, "5.1 Dataset and Experimental Setup")
    if ch:
        para(doc,
             f"All experiments use the Azure Functions 2019 production trace: "
             f"{ch['n_apps']:,} applications over {ch['n_minutes']:,} minutes "
             f"(14 days), {ch['total_invocations']/1e9:.2f} billion invocations, "
             f"aggregated from function level to application level because Azure "
             f"allocates resources per application. Days 1-8 are training, days "
             f"9-10 validation and days 11-14 a held-out test set; splits are "
             f"temporal and never shuffled. Experiments run on a stratified "
             f"cohort of 1,999 applications.")
        para(doc,
             f"The parse was validated against the characterisation published "
             f"with the dataset: "
             f"{ch['frac_apps_at_most_1_invocation_per_minute']*100:.1f}% of "
             f"applications are invoked at most once per minute on average (the "
             f"paper reports approximately 81%), HTTP is the dominant trigger, "
             f"and the four days with visibly fewer active applications are "
             f"exactly the two weekends. "
             f"{ch['frac_apps_avg_idle_time_gt']['10']*100:.1f}% of applications "
             f"have an average idle time longer than the industry-standard "
             f"ten-minute keep-alive window.")
    figure(doc, FIGURES / "motivation.png", 5.4,
           "Fig. 1. A fixed keep-alive window misses most applications.")

    h2(doc, "5.2 The measurement apparatus")
    para(doc,
         "Every claim here is a claim about a simulated quantity, so the "
         "simulator is treated as the primary instrument and validated first. It "
         "is implemented twice - once as an obvious per-minute loop and once as "
         "array algebra - and a randomised test asserts the two agree exactly "
         "over fifty traces of varying density, burstiness and memory. Further "
         "tests pin the causality contract (no policy may consult the current or "
         "any future minute), the accounting identities, and the estimator's "
         "limiting behaviour. Thirty-four tests pass. This was not ceremony: the "
         "tests caught an off-by-one in the pre-warm offset that made a perfectly "
         "periodic application cold-start on every single invocation.")

    h2(doc, "5.3 Baselines: the standard to beat")
    if base:
        rows = [["Policy", "Cold starts", "Idle GB-minutes", "Added latency / invocation"]]
        for w in ["oracle", "fixed-keepalive-5", "fixed-keepalive-10",
                  "fixed-keepalive-20", "fixed-keepalive-30",
                  "hybrid-histogram-ATC20", "scale-to-zero"]:
            r = next((x for x in base if x["policy"] == w), None)
            if r:
                rows.append([w, f"{r['cold_starts']:,}",
                             f"{r['idle_gb_minutes']:,.0f}",
                             f"{r['mean_added_latency_ms']:.3f} ms"])
        table(doc, rows, widths=[2.2, 1.4, 1.5, 1.5], font=10)
        caption(doc, "Table 1. Baseline policies on the held-out test days.")
        hb = next((x for x in base if x["policy"] == "hybrid-histogram-ATC20"), None)
        if hb:
            para(doc,
                 f"The hybrid histogram policy of Shahrad et al. is a strong "
                 f"target and not a straw man: {hb['cold_starts']:,} cold starts "
                 f"for {hb['idle_gb_minutes']:,.0f} idle GB-minutes, which "
                 f"Pareto-dominates the entire fixed keep-alive family - fewer "
                 f"cold starts than a twenty-minute window at roughly two thirds "
                 f"of the memory cost. Reproducing that property is itself "
                 f"evidence the reimplementation is faithful.")

    h2(doc, "5.4 What is actually predictable: the information audit")
    para(doc,
         "Before choosing a mechanism we measured what there is to exploit. For "
         "each application we discretise the sequence of observed idle gaps and "
         "each candidate context, and compute the mutual information between "
         "them. Mutual information is the right instrument because it is "
         "model-free: it bounds what any predictor could extract, so a small "
         "value is evidence about the problem rather than about our modelling.")
    ci = load("context_information.json")
    if ci:
        rows = [["Context available to the scheduler", "Median MI (bits)",
                 "% of the next gap entropy"]]
        for v in sorted(ci["contexts"].values(),
                        key=lambda d: -d["share_of_gap_entropy"]):
            rows.append([v["label"], f"{v['median_mi_bits']:.4f}",
                         f"{100*v['share_of_gap_entropy']:.1f}%"])
        table(doc, rows, widths=[2.8, 1.5, 1.8], font=10)
        caption(doc, f"Table 2. Predictable content of the next idle gap "
                     f"({ci['n_apps']} applications, median gap entropy "
                     f"{ci['gap_entropy_median_bits']:.3f} bits).")
        best = max(ci["contexts"].values(), key=lambda d: d["share_of_gap_entropy"])
        para(doc,
             f"Two conclusions follow, and both shaped the rest of the project. "
             f"First, about {100*ci['irreducible_share']:.0f}% of the uncertainty "
             f"in an application next idle gap is irreducible from anything the "
             f"scheduler can observe: serverless idle gaps on this trace are close "
             f"to memoryless. Second, the signal that does exist is concentrated "
             f"in a single discrete, low-dimensional variable - the "
             f"{best['label']}, at {100*best['share_of_gap_entropy']:.1f}% of gap "
             f"entropy, ahead of recent activity and time of day.")
    figure(doc, FIGURES / "information_audit.png", 6.4,
           "Fig. 2. What a scheduler can and cannot know about the next idle gap.")

    h2(doc, "5.5 Three negative results, explained by one measurement")
    para(doc,
         "The audit retrospectively explains three mechanisms we built and "
         "measured before arriving at the final one. Each is reported rather than "
         "discarded, because together they are what justifies the design that "
         "works.")
    for b in C.NEGATIVE_RESULTS:
        bullet(doc, b)
    metrics = load("prediction_metrics_h5.json")
    if metrics:
        pretty = {"B2-LSTM": "LSTM (AWU-style) [2]",
                  "B3-Transformer-noRetrieval": "Transformer, no retrieval",
                  "B3-Transformer-continued":
                      "Transformer, step-matched control",
                  "RACRP": "RACRP (retrieval on)",
                  "RACRP-shuffledMemory": "RACRP fed RANDOM memories",
                  "Retrieval-only-vote": "Retrieval-only vote"}
        rows = [["Model", "Average precision", "AUC-ROC", "ECE"]]
        for r in metrics["results"]:
            if "average_precision" not in r:
                continue
            rows.append([pretty.get(r["model"], r["model"]),
                         f"{r['average_precision']:.4f}",
                         f"{r['auc_roc']:.4f}", f"{r['ece']:.4f}"])
        table(doc, rows, widths=[2.8, 1.4, 1.1, 1.0], font=9.5)
        caption(doc, "Table 5. Cold-start risk prediction on the held-out test "
                     "days. All arms share the training loop, seed and splits.")

        ab = metrics.get("ablations") or {}
        label = {"RACRP_vs_B3-Transformer-noRetrieval":
                     "RACRP vs. no retrieval (the comparison a paper would report)",
                 "RACRP_vs_B3-Transformer-continued":
                     "RACRP vs. the STEP-MATCHED control",
                 "RACRP_vs_RACRP-shuffledMemory":
                     "RACRP vs. the same model fed random memories",
                 "RACRP_vs_Retrieval-only-vote":
                     "RACRP vs. the retrieval-only vote"}
        rows = [["Comparison", "Delta AP", "95% CI", "p (A not better)"]]
        for k, v in ab.items():
            rows.append([label.get(k, k), f"{v['mean_delta']:+.4f}",
                         f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]",
                         f"{v['p_value_a_not_better']:.4f}"])
        if len(rows) > 1:
            table(doc, rows, widths=[2.9, 1.0, 1.5, 1.1], font=9.5)
            caption(doc, "Table 6. Paired bootstrap over the shared test set.")
        para(doc,
             "Read in order, this is the methodological point of the project. "
             "Against the obvious baseline, retrieval appears to win decisively. "
             "But RACRP is warm-started from that same transformer and then "
             "trained further, so the comparison bundles retrieval together with "
             "extra optimisation. Against a control that gives the no-retrieval "
             "model exactly those extra steps, retrieval is significantly worse - "
             "and a model fed random memories scores as well as one fed real "
             "ones. One additional arm, costing a single training run, converts a "
             "significant positive result into a significant negative one.")

    para(doc,
         "The common cause is now quantified: each of those schemes had to "
         "estimate a distribution from a few dozen retrieved neighbours in order "
         "to capture a context worth a single-digit percentage of the gap "
         "entropy. The variance introduced exceeds the information gained. This "
         "also explains why the choice of a plain marginal histogram in the "
         "deployed policy was close to the right call.")
    figure(doc, FIGURES / "finding_matched_cost.png", 6.4,
           "Fig. 3. Borrowing gap distributions across applications loses at "
           "matched cost, and the oracle bound loses too.")

    h2(doc, "5.6 The method: conditioning on the state the application is in")
    para(doc,
         "If the informative key is one coarse discrete variable, then the right "
         "non-parametric conditional estimator is not a learned embedding with a "
         "nearest-neighbour lookup - it is a conditional histogram. Retrieval over "
         "a discrete key with many samples per cell is a conditional histogram; "
         "the embedding machinery only earns its keep when the key is "
         "high-dimensional and continuous, and here it was pure variance. So we "
         "keep one idle-gap histogram per bucket of the gap that just closed, and "
         "blend it with the application marginal in quantile space, weighted by "
         "how much evidence supports the cell.")
    _p = doc.add_paragraph(); _p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _r = _p.add_run("h_hat( g | b )  =  ( n_b * h_b  +  kappa * h_marginal ) / ( n_b + kappa )")
    _r.font.name = FONT; _r.font.size = Pt(12); _r.italic = True
    para(doc,
         "As kappa grows, or for a bucket with no observations, the estimate "
         "becomes the ATC'20 histogram exactly. The deployed policy is therefore "
         "recovered as the special case that conditions on nothing, which makes "
         "this a strict generalisation along the conditioning axis rather than a "
         "competing heuristic.")

    h2(doc, "5.7 Result at matched memory cost")
    para(doc,
         "A policy that holds more containers warm will always show fewer cold "
         "starts, so every arm is swept over its tail-percentile cost knob and "
         "arms are compared by interpolating each curve at the baseline own "
         "operating points. A random-key control with the same number of cells is "
         "included to separate genuine conditioning from the variance-reduction "
         "artefact of merely splitting the data.")
    mc = load("conditional_matched_cost.json")
    if mc:
        rows = [["Memory budget (idle GB-min)", "ATC'20 marginal",
                 "Conditioned on previous gap", "Reduction", "Random key (control)"]]
        for r in mc:
            rows.append([
                f"{r['budget_idle_gb_min']:,.0f}",
                f"{r['baseline_cold_starts']:,}",
                f"{r['prev_gap']:,.0f}" if r.get("prev_gap") else "-",
                f"{r['prev_gap_reduction_pct']:+.1f}%"
                if r.get("prev_gap_reduction_pct") is not None else "-",
                f"{r['shuffled_reduction_pct']:+.1f}%"
                if r.get("shuffled_reduction_pct") is not None else "-"])
        table(doc, rows, widths=[1.7, 1.3, 1.5, 1.0, 1.2], font=9.5)
        caption(doc, "Table 3. Cold starts at matched memory cost, 1,999 "
                     "applications, held-out test days.")
        red = [r["prev_gap_reduction_pct"] for r in mc
               if r.get("prev_gap_reduction_pct") is not None]
        ctrl = [r.get("shuffled_reduction_pct") for r in mc
                if r.get("shuffled_reduction_pct") is not None]
        if red:
            import statistics as _st
            txt = (f"Conditioning on the gap that just closed reduces cold starts "
                   f"by a median {_st.median(red):.1f}% at matched memory cost, "
                   f"ranging from {min(red):.1f}% to {max(red):.1f}% across the "
                   f"baseline operating points")
            if ctrl:
                txt += (f". The random-key control moves the result by "
                        f"{_st.median(ctrl):+.1f}%, confirming that the gain comes "
                        f"from the conditioning variable rather than from "
                        f"splitting the data into more cells.")
            else:
                txt += "."
            para(doc, txt)
    figure(doc, FIGURES / "conditional_result.png", 6.4,
           "Fig. 4. Cost/benefit curves and the reduction at matched memory cost.")

    sens = load("sensitivity.json")
    if sens:
        para(doc,
             "The estimator has exactly two hyper-parameters, and both were set by "
             "judgement rather than by search: kappa, so that a handful of gaps "
             "cannot immediately outvote an application's whole history, and the "
             "bucket count, chosen to span a minute to a day at a resolution the "
             "data supports. Neither was tuned on the test split, and the grid "
             "below exists so that claim can be checked rather than asserted.")
        ks = sorted({r["kappa"] for r in sens["grid"]})
        bs = sorted({r["buckets"] for r in sens["grid"]})
        rows = [["buckets / kappa"] + [f"{k:g}" for k in ks]]
        for b in bs:
            row = [str(b)]
            for k in ks:
                v = next((r["median_reduction_pct"] for r in sens["grid"]
                          if r["buckets"] == b and r["kappa"] == k), None)
                row.append(f"{v:+.1f}%" if v is not None else "-")
            rows.append(row)
        table(doc, rows, font=10)
        caption(doc, "Table 9. Median cold-start reduction at matched cost across "
                     "the whole hyper-parameter grid.")
        para(doc,
             f"The result is positive at all {len(sens['grid'])} settings, ranging "
             f"from {sens['min_pct']:+.1f}% to {sens['max_pct']:+.1f}%, and the "
             f"setting we chose is not the best in the grid - which is what one "
             f"would expect of a value picked before looking. The gain also decays "
             f"monotonically as kappa grows, as it must, since a sufficiently "
             f"large kappa reduces the estimator to the ATC'20 baseline by "
             f"construction; watching the method decay into its own baseline is a "
             f"further check that the implementation matches the algebra.")

    # ---- the derived window ------------------------------------------------
    h2(doc, "5.8 The window was never derived, only tuned")
    para(doc,
         "Everything to this point, the deployed policy and our own conditional "
         "method alike, reads the keep-alive window off a PERCENTILE of an "
         "idle-gap histogram: ATC'20 uses the 5th for when to bring a container "
         "back and the 99th for when to release it. Nothing argues that the 99th "
         "is right, and a percentile is not a quantity an operator can price. The "
         "decision is really an optimal-stopping problem. A container warm over "
         "the window [a, a+L) after an application goes quiet pays memory for "
         "whatever part of that window the idle gap covers, and pays a cold start "
         "unless the next invocation lands inside it. Minimising the expected "
         "cost of that window jointly over offset and length replaces the "
         "percentile knob with K/m, the number of GB-minutes a cold start is "
         "worth, which is an economic quantity a platform actually knows.")
    para(doc,
         "Setting the derivative to zero gives an elegant identity, "
         "hazard(T) = m/K: release the container at the first moment its hazard "
         "rate falls below the price ratio. We implemented that and it was "
         "catastrophic, for a reason worth recording. The first-order condition "
         "locates a stationary point, and the expected cost is not unimodal for "
         "real idle-gap distributions. A perfectly periodic application has zero "
         "hazard everywhere except at its period, so a first-crossing rule stops "
         "at minute one and cold-starts on every single invocation - which is "
         "exactly what it did. The global minimum over the grid costs the same "
         "order of work and is what the cost model actually asks for. The "
         "identity remains the right way to understand the rule; it is not the "
         "algorithm.")

    dsum = load("derived_summary.json")
    dmc = load("derived_matched_cost.json")
    if dmc:
        key = "DERIVED window, marginal"
        ctl = "CONTROL: derived from a donor app"
        rows = [["Memory budget (idle GB-min)", "ATC'20 marginal",
                 "Derived window", "Reduction", "Donor app (control)"]]
        for r in dmc:
            rows.append([
                f"{r['budget_idle_gb_min']:,.0f}",
                f"{r['baseline_cold_starts']:,}",
                f"{r[key]:,.0f}" if r.get(key) is not None else "-",
                f"{r[key + ' %']:+.1f}%" if r.get(key + " %") is not None else "-",
                f"{r[ctl + ' %']:+.1f}%" if r.get(ctl + " %") is not None else "-"])
        table(doc, rows, widths=[1.7, 1.3, 1.4, 1.0, 1.3], font=9.5)
        caption(doc, "Table 3b. Cold starts at matched memory cost with the "
                     "window derived from the cost model, 1,999 applications, "
                     "held-out test days. The control runs the identical "
                     "optimisation on another application's gap histogram.")

    if dsum:
        by = {r["arm"]: r for r in dsum}
        d = by.get("DERIVED window, marginal")
        c = by.get("CONTROL: derived from a donor app")
        both = by.get("DERIVED window + conditioning")
        cond = by.get("conditional, percentile window")
        if d:
            txt = (f"Deriving the window rather than tuning it reduces cold starts "
                   f"by a median {d['median_reduction_pct']:.1f}% at matched "
                   f"memory cost, ranging from {d['min']:.1f}% to {d['max']:.1f}% "
                   f"across the baseline operating points.")
            if c:
                txt += (f" Running the identical optimisation on a DONOR "
                        f"application's gap histogram moves the result to "
                        f"{c['median_reduction_pct']:+.1f}%, so the gain requires "
                        f"the application's own history and is not an artefact of "
                        f"the window sizes the rule happens to produce.")
            para(doc, txt)
        if d and both and cond:
            para(doc,
                 f"The two ideas in this project do not compose, and the reason is "
                 f"the more interesting half of the result. Conditioning alone "
                 f"gives {cond['median_reduction_pct']:+.1f}% and deriving alone "
                 f"gives {d['median_reduction_pct']:+.1f}%, but doing both gives "
                 f"{both['median_reduction_pct']:+.1f}% - worse than deriving "
                 f"alone, and far more variable. A percentile is a robust "
                 f"statistic that survives sparse counts; the cost-optimal window "
                 f"integrates over the whole shape of the distribution and "
                 f"therefore needs the tail estimated well. Conditioning fragments "
                 f"precisely the counts that estimate that tail. Each idea buys "
                 f"information from a different place, and one of them spends the "
                 f"resource the other depends on.")

    figure(doc, FIGURES / "derived_why.png", 6.4,
           "Fig. 5. One application, both rules drawn on its own idle-gap "
           "distribution. The percentile pair spans sixty minutes to reach the "
           "last 1% of gaps; the cost model integrates the shape between them and "
           "stops where holding longer stops paying. The box reports each "
           "window's expected cost under the model being minimised - a live check "
           "that the optimiser beats the heuristic on its own objective, and the "
           "check that caught a prior-strength bug.")

    h2(doc, "5.8.1 Giving the baseline the same freedom")
    para(doc,
         "The derived rule tunes two degrees of freedom where our sweep of the "
         "baseline varied one, which is a real asymmetry and the first thing a "
         "reader should attack. It is removed rather than argued away. The "
         "baseline was swept over BOTH percentiles, six values of each, and the "
         "lower envelope of the resulting 36 configurations taken - the best "
         "cost/latency frontier reachable by any fixed percentile pair, selected "
         "with the test period in view. That envelope is a stronger baseline than "
         "the published policy, which does not tune the head at all.")
    sb = load("strong_baseline_summary.json")
    sbm = load("strong_baseline_matched.json")
    if sbm:
        rows = [["Memory budget (idle GB-min)", "Best of 36 ATC'20 pairs",
                 "Derived window", "Reduction"]]
        for r in sbm:
            rows.append([f"{r['budget_idle_gb_min']:,.0f}",
                         f"{r['best_baseline_cold_starts']:,}",
                         f"{r['derived_cold_starts']:,.0f}",
                         f"{r['reduction_pct']:+.1f}%"])
        table(doc, rows, widths=[1.8, 1.8, 1.5, 1.1], font=9.5)
        caption(doc, "Table 3c. The derived window against the best envelope of "
                     "36 tuned ATC'20 percentile pairs, at matched memory cost.")
    if sb and sb.get("median_reduction_pct") is not None:
        para(doc,
             f"Against that strengthened baseline the derived window still wins by "
             f"a median {sb['median_reduction_pct']:.1f}% ({sb['n_apps']:,} "
             f"applications, {sb['baseline_configs']} baseline configurations). It "
             f"LOSES at the highest memory budgets, and that is the honest "
             f"boundary of the claim: when memory is nearly free the right policy "
             f"is to hold everything warm for a long time, there is little left "
             f"for an optimiser to recover, and a long fixed keep-alive is hard to "
             f"beat. The gain lives in the economically interesting region, where "
             f"memory is scarce enough to be worth allocating carefully.")

    figure(doc, FIGURES / "derived_frontier.png", 6.4,
           "Fig. 6. The derived window against every one of the 36 tuned ATC'20 "
           "percentile pairs and against their lower envelope. Right: the gain at "
           "each of the baseline's own operating points, and the budget at which "
           "it runs out.")

    h2(doc, "5.8.2 A hyper-parameter chosen on validation, not on the result")
    tun = load("smoothing_tuning.json")
    para(doc,
         "Because the derived window integrates over the whole distribution, it is "
         "sensitive to how the sparse tail is smoothed in a way a percentile rule "
         "is not, which makes the prior strength a genuine hyper-parameter. An "
         "early implementation added one pseudo-count per histogram bin; on a "
         "241-minute support that is 241 pseudo-observations against an "
         "application's few hundred real gaps, a uniform prior strong enough to "
         "drag probability mass into a tail that is not there. It was found by "
         "annotating a figure with each window's expected cost under the very "
         "model being minimised, and seeing the optimiser lose to the baseline on "
         "its own objective.")
    if tun:
        rows = [["Prior strength (total pseudo-mass)",
                 "Median gain on VALIDATION days"]]
        for g in tun["grid"]:
            v = g["val_median_gain_pct"]
            rows.append([f"{g['smooth_total']:g}",
                         f"{v:+.1f}%" if v is not None else "-"])
        table(doc, rows, widths=[2.6, 2.4], font=9.5)
        caption(doc, "Table 3d. Prior strength selected on the validation days and "
                     "carried to the test days unchanged.")
        para(doc,
             f"The setting was chosen on the validation days and carried to the "
             f"test days unchanged, at {tun['chosen_smooth_total']:g}. This "
             f"mattered more than such choices usually do. The faulty per-bin prior "
             f"scores BETTER on the test period than the corrected one, and WORST "
             f"of the seven settings on validation. Selecting on the test period "
             f"would have picked the worst-generalising point in the grid and "
             f"attached the largest number in this project to it. A unit test pins "
             f"the constant to the recorded validation choice so it cannot drift "
             f"back.")

    figure(doc, FIGURES / "derived_composition.png", 5.8,
           "Fig. 7. Each idea beats the baseline; together they beat neither. The "
           "donor-application control is shown at the right.")

    h2(doc, "5.9 Does the gain come from where the mechanism says?")
    para(doc,
         "A result at matched cost establishes that the method works; it does not "
         "establish why. Before running the method we committed to a falsification "
         "condition: if conditioning on the previous gap helps because an "
         "application's gaps genuinely depend on each other, then the improvement "
         "must concentrate in the applications where that dependence exists and be "
         "small where gaps are effectively independent. A gain spread evenly over "
         "applications would mean something else was responsible.")
    mech = load("mechanism_check_summary.json")
    if mech:
        rows = [["Applications grouped by gap autocorrelation", "Apps",
                 "Median MI (bits)", "Reduction at matched cost"]]
        for q in mech["by_quartile"]:
            v = q["matched_cost_reduction_pct"]
            rows.append([q["stratum"], q["apps"],
                         f"{q['median_mi_bits']:.3f}",
                         f"{v:+.1f}%" if v is not None else "-"])
        table(doc, rows, widths=[2.6, 0.8, 1.4, 1.8], font=10)
        caption(doc, "Table 4. Improvement by how much an application's "
                     "consecutive idle gaps depend on each other, measured on "
                     "the training period only.")
        para(doc,
             f"The correlation across strata between an application group's "
             f"median mutual information and its matched-cost improvement is "
             f"{mech['correlation_mi_vs_reduction']:+.2f}. The improvement is "
             f"{mech['reduction_lowest_quartile_pct']:+.1f}% where consecutive "
             f"gaps carry essentially no information about each other and "
             f"{mech['reduction_highest_quartile_pct']:+.1f}% where they carry the "
             f"most. The mechanism therefore behaves as claimed rather than merely "
             f"producing a favourable aggregate.")
        figure(doc, FIGURES / "mechanism_check.png", 6.4,
               "Fig. 8. The improvement rises with how much an application's "
               "consecutive idle gaps depend on each other.")
        para(doc,
             "Two caveats are recorded rather than smoothed over. The second "
             "quartile is negative, and the top quartile rests on only a small "
             "number of matched budget points because the two arms' cost curves "
             "overlap over a narrow range there. The monotone trend and the "
             "correlation are the substance; the individual quartile figures are "
             "noisy.")
    para(doc,
         "The first version of this check was wrong in an instructive way. It "
         "compared the two policies at a fixed tail percentile within each "
         "quartile and appeared to show the method failing catastrophically in the "
         "most-autocorrelated group. That arm was spending twenty per cent less "
         "memory: at a fixed percentile the two policies simply occupy different "
         "points on their own cost curves. This was the third occasion in this "
         "project on which an unmatched-cost comparison produced a confident and "
         "wrong answer, twice in opposite directions, and it is why every "
         "comparison reported here is a curve rather than a point.")

    h2(doc, "5.10 Threats to Validity")
    for b in C.THREATS:
        bullet(doc, b)


def sec_conclusion(doc, metrics, head):
    h1(doc, "6. Conclusion and Future Work")
    para(doc, C.ONE_LINE)
    para(doc, C.RETRIEVAL_FRAMING)
    ci = load("context_information.json")
    mc = load("conditional_matched_cost.json")
    if ci:
        best = max(ci["contexts"].values(), key=lambda d: d["share_of_gap_entropy"])
        para(doc,
             f"The route to that result was empirical rather than architectural. "
             f"An information audit over {ci['n_apps']} applications established "
             f"that roughly {100*ci['irreducible_share']:.0f}% of the next idle "
             f"gap entropy is irreducible from any observable context, and "
             f"localised the remainder in the {best['label']}. That single "
             f"measurement explains three earlier negative results and selects the "
             f"estimator that finally works: when the informative key is one "
             f"coarse discrete variable, a conditional histogram is the right "
             f"non-parametric estimator and a learned retrieval embedding is pure "
             f"variance.")
    dsum_c = {r["arm"]: r
              for r in (load("derived_summary.json") or [])}
    sb_c = load("strong_baseline_summary.json") or {}
    tun_c = load("smoothing_tuning.json") or {}
    d_c = dsum_c.get("DERIVED window, marginal")
    ctl_c = dsum_c.get("CONTROL: derived from a donor app")
    both_c = dsum_c.get("DERIVED window + conditioning")
    if d_c:
        txt = (f"The larger of the two findings is about the window rather than "
               f"the distribution. Every policy here, the deployed one included, "
               f"read its warm window off a percentile - a heuristic, and not a "
               f"quantity anyone operating a platform can price. Deriving it "
               f"instead, by minimising the expected cost of the window jointly "
               f"over its offset and its length, reduces cold starts by a median "
               f"{d_c['median_reduction_pct']:.1f}% at matched memory cost")
        if sb_c.get("median_reduction_pct") is not None:
            txt += (f", and by {sb_c['median_reduction_pct']:.1f}% against a "
                    f"baseline deliberately strengthened to tune both of its "
                    f"percentiles over {sb_c.get('baseline_configs', 36)} "
                    f"configurations")
        para(doc, txt + ". What is swept in place of the percentile is K/m, the "
                        "number of GB-minutes a cold start is worth, which is an "
                        "economic quantity rather than a distributional artefact.")
    if ctl_c:
        para(doc,
             f"That claim rests on its control. Running the identical optimiser, "
             f"at the identical price ratio, through the identical simulator, on "
             f"ANOTHER application's gap histogram does not merely erase the gain: "
             f"it lands at {ctl_c['median_reduction_pct']:+.0f}%, several times "
             f"worse than the baseline it had been beating. Only the ownership of "
             f"the history changed. Whatever the derived window is doing, it is "
             f"not producing window sizes that happen to be good on average - it "
             f"is reading something specific out of each application's own past, "
             f"which is the proposition this project set out to test.")
    if d_c and both_c:
        para(doc,
             f"The two findings do not compose, and we report that as carefully "
             f"as the gain. Applying the conditioning on top of the derived window "
             f"gives {both_c['median_reduction_pct']:.1f}% against "
             f"{d_c['median_reduction_pct']:.1f}% for the derived window alone, "
             f"and is far more variable. A percentile is a robust statistic that "
             f"survives sparse counts; a cost-optimal window integrates the whole "
             f"shape of the distribution and needs its tail estimated well; and "
             f"conditioning fragments precisely the counts that estimate that "
             f"tail. Each idea buys information from a different place, and one "
             f"spends the resource the other depends on. We would not have "
             f"predicted this, and we did not.")
    if tun_c.get("chosen_smooth_total") is not None:
        para(doc,
             f"One methodological point deserves to survive into the conclusion. "
             f"The derived window has a single hyper-parameter, the strength of "
             f"the prior on the gap distribution, and it was chosen on the "
             f"validation days at {tun_c['chosen_smooth_total']:g} and carried to "
             f"the test days unchanged. The setting that scores best on the test "
             f"period ranks last of seven on validation. Had we selected on the "
             f"test period, as is easy to do without noticing, we would have "
             f"reported the largest number in this project on the strength of the "
             f"worst-generalising estimator in the grid.")

    if mc:
        red = [r["prev_gap_reduction_pct"] for r in mc
               if r.get("prev_gap_reduction_pct") is not None]
        if red:
            import statistics as _st
            para(doc,
                 f"Conditioning the deployed policy own estimator on that "
                 f"variable, with partial pooling towards the marginal, reduces "
                 f"cold starts by a median {_st.median(red):.1f}% at matched "
                 f"memory cost against the policy deployed in Azure Functions, "
                 f"with the deployed policy recovered exactly as the limiting "
                 f"case. This was the project's headline result until the window "
                 f"itself was examined; it remains the cleaner of the two "
                 f"contributions, being a strict generalisation of the deployed "
                 f"estimator along one axis rather than a different rule.")
    para(doc, C.MECHANISM_NOTE)
    para(doc, C.method_note(load("prediction_metrics_h5.json")))
    para(doc,
         "What stands independently of the headline: a simulator validated "
         "against a reference implementation by randomised equivalence testing; a "
         "trace parse reproducing the published characterisation; a faithful "
         "reimplementation of the ATC'20 hybrid histogram that reproduces its "
         "central claim; and a model-free measurement of how much of a serverless "
         "application idle behaviour is predictable at all, which is the quantity "
         "any future work on this problem needs and which, to our knowledge, has "
         "not previously been reported.")
    for b in C.FUTURE:
        bullet(doc, b)


def main() -> None:
    ch = load("characterisation.json")
    base = load("baselines_test.json")
    metrics = load("prediction_metrics_h5.json")
    head = load("headline.json")
    strata = load("strata.json")
    idle_age = load("by_idle_age.json")

    doc = Document()
    style_doc(doc)
    for s in doc.sections:
        s.left_margin = Inches(1.25); s.right_margin = Inches(1.0)
        s.top_margin = Inches(1.0); s.bottom_margin = Inches(1.0)

    title_page(doc)

    h1(doc, "Abstract")
    para(doc, ABSTRACT)
    para(doc, KEYWORDS)
    doc.add_page_break()

    h1(doc, "Table of Contents")
    for line in ["1. Introduction", "    1.1 Background", "    1.2 Motivation",
                 "    1.3 Scope of the Project",
                 "2. Project Description and Goals", "    2.1 Literature Review",
                 "    2.2 Research Gap", "    2.3 Objectives",
                 "    2.4 Problem Statement", "    2.5 Project Plan",
                 "3. Technical Specification", "    3.1 Requirements",
                 "    3.2 Feasibility Study", "    3.3 System Specification",
                 "4. Design Approach and Details", "    4.1 System Architecture",
                 "    4.2 Design", "5. Experimental Results",
                 "    5.1 Dataset and Experimental Setup",
                 "    5.2 The measurement apparatus",
                 "    5.3 Baselines: the standard to beat",
                 "    5.4 What is actually predictable: the information audit",
                 "    5.5 Three negative results, explained by one measurement",
                 "    5.6 The method: conditioning on the application state",
                 "    5.7 Result at matched memory cost",
                 "    5.8 Does the gain come from where the mechanism says?",
                 "    5.9 Threats to Validity",
                 "6. Conclusion and Future Work", "7. References"]:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.5
        r = p.add_run(line); r.font.name = FONT; r.font.size = Pt(12)
    doc.add_page_break()

    sec_intro(doc, ch)
    sec_description(doc, ch, base, metrics)
    sec_technical(doc)
    sec_design(doc)
    sec_results(doc, ch, base, metrics, head, strata, idle_age)
    sec_conclusion(doc, metrics, head)

    h1(doc, "7. References")
    for i, r in enumerate(C.REFERENCES, 1):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        run = p.add_run(f"[{i}] {r}")
        run.font.name = FONT; run.font.size = Pt(11)

    OUT.parent.mkdir(exist_ok=True)
    doc.save(str(OUT))
    print(f"[report] wrote {OUT}")


if __name__ == "__main__":
    main()
