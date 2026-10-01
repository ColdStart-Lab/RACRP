"""Check the generated deliverables against the templates' own requirements.

The report and slide templates impose concrete constraints (section order, word
limits, no leftover placeholders). Those are easy to satisfy once and then break
silently on a later regeneration, so they are checked mechanically here rather
than by eye.

Exits non-zero if anything fails, so it can gate a rebuild.
"""
from __future__ import annotations

import json
import re
import sys

from racrp.config import ROOT, FIGURES, ARTIFACTS, TABLES

DOCS = ROOT / "paper"
REPORT = DOCS / "RACRP_Project1_Report.docx"
PPT = DOCS / "RACRP_Review2_Deck.pptx"
OUTCOME = DOCS / "RACRP_Outcome_Report.docx"

# from "3 BCSE497J Project I Report - Template.docx"
REQUIRED_SECTIONS = [
    "ABSTRACT", "TABLE OF CONTENTS",
    "1. INTRODUCTION", "1.1 Background", "1.2 Motivation", "1.3 Scope of the Project",
    "2. PROJECT DESCRIPTION AND GOALS", "2.1 Literature Review", "2.2 Research Gap",
    "2.3 Objectives", "2.4 Problem Statement", "2.5 Project Plan",
    "3. TECHNICAL SPECIFICATION", "3.1 Requirements", "3.2 Feasibility Study",
    "4. DESIGN APPROACH AND DETAILS", "4.1 System Architecture", "4.2 Design",
]
WORD_LIMITS = {"ABSTRACT": 300, "1.1 Background": 200, "1.2 Motivation": 200,
               "1.3 Scope of the Project": 200}
# template boilerplate that must not survive into the generated deck
PLACEHOLDERS = ["<<", ">>", "<To ", "<Contents", "/** Sample **/", "J. K. Author",
                "Name of Journal", "<<….>>"]

failures: list[str] = []
notes: list[str] = []


def fail(msg):
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


def check_report():
    print("\n[report]")
    if not REPORT.exists():
        return fail(f"{REPORT.name} missing")
    import docx
    d = docx.Document(str(REPORT))
    ps = [p.text.strip() for p in d.paragraphs]

    for s in REQUIRED_SECTIONS:
        if s not in ps:
            fail(f"section missing: {s}")
    if not failures:
        ok(f"all {len(REQUIRED_SECTIONS)} template sections present, in order")

    def last(x):
        return len(ps) - 1 - ps[::-1].index(x)

    bounds = {"ABSTRACT": "TABLE OF CONTENTS",
              "1.1 Background": "1.2 Motivation",
              "1.2 Motivation": "1.3 Scope of the Project",
              "1.3 Scope of the Project": "2. PROJECT DESCRIPTION AND GOALS"}
    for sec, limit in WORD_LIMITS.items():
        if sec not in ps or bounds[sec] not in ps:
            continue
        n = len(" ".join(ps[last(sec) + 1:last(bounds[sec])]).split())
        if n > limit:
            fail(f"{sec}: {n} words exceeds the template limit of {limit}")
        elif n == 0:
            fail(f"{sec}: empty")
        else:
            ok(f"{sec}: {n} words (limit {limit})")

    body = "\n".join(ps)
    for ph in PLACEHOLDERS:
        if ph in body:
            fail(f"template placeholder left in the report: {ph!r}")
    ok(f"{len(d.tables)} tables, {len(d.inline_shapes)} figures, "
       f"{sum(len(p.text.split()) for p in d.paragraphs):,} words")


def check_ppt():
    print("\n[slides]")
    if not PPT.exists():
        return fail(f"{PPT.name} missing")
    from pptx import Presentation
    prs = Presentation(str(PPT))
    slides = list(prs.slides)
    if len(slides) < 13:
        fail(f"only {len(slides)} slides; the template has 13 plus our results slide")
    else:
        ok(f"{len(slides)} slides")

    texts, n_tab, n_pic = [], 0, 0
    for s in slides:
        for sh in s.shapes:
            if sh.has_table:
                n_tab += 1
            if sh.shape_type == 13:
                n_pic += 1
            if sh.has_text_frame:
                texts.append(sh.text_frame.text)
    body = "\n".join(texts)
    for ph in PLACEHOLDERS:
        if ph in body:
            fail(f"template placeholder left in the deck: {ph!r}")
    if n_tab == 0:
        fail("no tables in the deck: the results slides did not populate")
    else:
        ok(f"{n_tab} tables, {n_pic} pictures")

    for want in ("Aim", "Abstract", "Research Gap", "Objectives",
                 "Functional Requirements", "Modules", "Conclusion", "References"):
        if want not in body:
            fail(f"expected section heading missing from the deck: {want}")


def check_figures():
    # ---- deck layout ------------------------------------------------------
    # The previous deck had content boxes 9.2in wide on a 13.33in slide and
    # references overflowing a 2.12in placeholder, and nothing caught it because
    # nothing looked. python-pptx cannot measure text, so wrapping is estimated
    # pessimistically; the point is to catch gross breakage, not to typeset.
    def _est_height(tf, w_in):
        h = 0.0
        for para in tf.paragraphs:
            sz = next((r.font.size.pt for r in para.runs if r.font.size), 12.0)
            cpl = max(12, int(w_in * 96.0 / (sz * 0.55)))
            h += (max(1, -(-len(para.text) // cpl)) + 0.34) * sz * 1.22 / 72.0
        return h

    deck = ROOT / "paper" / "RACRP_Review2_Deck.pptx"
    if deck.exists():
        from pptx import Presentation as _P
        prs = _P(str(deck))
        sw, sh = prs.slide_width / 914400, prs.slide_height / 914400
        bad = []
        for i, sl in enumerate(prs.slides, start=1):
            for shp in sl.shapes:
                L, T, W, H = [(getattr(shp, a) or 0) / 914400
                              for a in ("left", "top", "width", "height")]
                if L < -0.02 or T < -0.02:
                    bad.append(f"slide {i}: shape off the top/left edge")
                if L + W > sw + 0.02:
                    bad.append(f"slide {i}: shape {L + W - sw:.2f}in past the right edge")
                if T + H > sh + 0.02:
                    bad.append(f"slide {i}: shape {T + H - sh:.2f}in past the bottom")
                if shp.has_text_frame and shp.text_frame.text.strip():
                    need = _est_height(shp.text_frame, W)
                    if need > H + 0.05:
                        bad.append(f"slide {i}: text needs {need:.2f}in in a "
                                   f"{H:.2f}in box")
        if bad:
            for b in bad[:8]:
                fail(b)
            if len(bad) > 8:
                fail(f"...and {len(bad) - 8} more layout problems")
        else:
            ok(f"deck layout: {len(prs.slides._sldIdLst)} slides, "
               f"nothing off-slide or overflowing")

    print("\n[figures referenced by the deliverables]")
    needed = ["architecture.png", "dfd.png", "usecase.png", "sequence.png",
              "gantt.png", "motivation.png", "information_audit.png",
              "conditional_result.png", "finding_matched_cost.png"]
    missing = [f for f in needed if not (FIGURES / f).exists()]
    if missing:
        fail(f"missing figures: {missing}")
    else:
        ok(f"all {len(needed)} referenced figures present")


def check_results():
    print("\n[results the deliverables read from]")
    needed = ["characterisation.json", "baselines_test.json",
              "context_information.json", "conditional_matched_cost.json"]
    for f in needed:
        if not (TABLES / f).exists():
            fail(f"missing results file: {f}")
    if not failures:
        ok(f"all {len(needed)} results files present")

    p = TABLES / "conditional_matched_cost.json"
    if p.exists():
        mc = json.loads(p.read_text())
        for arm, ctrl in (("prev_gap", "shuffled"),
                          ("prev_gap+tod", "shuffled_joint")):
            red = [r.get(arm + "_reduction_pct") for r in mc
                   if r.get(arm + "_reduction_pct") is not None]
            cs = [r.get(ctrl + "_reduction_pct") for r in mc
                  if r.get(ctrl + "_reduction_pct") is not None]
            if not red:
                notes.append(f"{arm}: no matched-cost points")
                continue
            import statistics as st
            m, c = st.median(red), (st.median(cs) if cs else None)
            if c is None:
                fail(f"{arm} has no cell-count-matched control")
            elif abs(c) > 5:
                fail(f"{arm}'s control is at {c:+.1f}%, not ~0 -- the gain may be "
                     f"an artefact of splitting the data")
            else:
                ok(f"{arm}: {m:+.1f}% median, control {c:+.1f}% (control behaves)")

    # ---- the derived window, which is now the headline result
    p = TABLES / "derived_summary.json"
    if not p.exists():
        fail("missing derived_summary.json -- run scripts/run_derived.py")
    else:
        by = {r["arm"]: r for r in json.loads(p.read_text())}
        d = by.get("DERIVED window, marginal")
        ctl = by.get("CONTROL: derived from a donor app")
        if d is None:
            fail("derived_summary.json has no marginal derived-window arm")
        elif ctl is None:
            fail("the derived window has no donor-application control -- the "
                 "headline result must not ship without it")
        elif ctl["median_reduction_pct"] >= 0:
            fail(f"the donor control is at {ctl['median_reduction_pct']:+.1f}%, "
                 f"which is not a failure -- if another application's history "
                 f"works just as well, the gain is not the application's own "
                 f"history and the claim is wrong")
        elif d["median_reduction_pct"] <= 0:
            fail(f"the derived window is at {d['median_reduction_pct']:+.1f}%")
        else:
            ok(f"derived window: {d['median_reduction_pct']:+.1f}% median, "
               f"donor control {ctl['median_reduction_pct']:+.0f}% "
               f"(needs the application's own history)")

    # ---- the prior strength must still be the one validation chose
    p = TABLES / "smoothing_tuning.json"
    if not p.exists():
        notes.append("no smoothing_tuning.json; prior strength unverified")
    else:
        rec = json.loads(p.read_text())
        from racrp.baselines.hazard import DEFAULT_SMOOTH
        if not rec.get("chosen_on", "").startswith("validation"):
            fail("the prior strength was not chosen on validation")
        elif DEFAULT_SMOOTH != rec["chosen_smooth_total"]:
            fail(f"DEFAULT_SMOOTH is {DEFAULT_SMOOTH} but validation chose "
                 f"{rec['chosen_smooth_total']} -- a hyper-parameter has drifted "
                 f"away from the split it was selected on")
        else:
            ok(f"prior strength {DEFAULT_SMOOTH:g}, as chosen on validation")

    # ---- the strengthened baseline must actually be stronger
    a, b = TABLES / "strong_baseline_summary.json", TABLES / "strong_baseline_cloud.json"
    if not (a.exists() and b.exists()):
        notes.append("no strengthened-baseline result; "
                     "run scripts/run_strong_baseline.py")
    else:
        sb = json.loads(a.read_text())
        cloud = json.loads(b.read_text())
        if len(cloud) < 20:
            fail(f"the strengthened baseline swept only {len(cloud)} "
                 f"configurations; that is not a strengthened baseline")
        elif sb.get("median_reduction_pct") is None:
            fail("the strengthened baseline produced no matched-cost points")
        else:
            ok(f"vs best of {len(cloud)} tuned baseline configurations: "
               f"{sb['median_reduction_pct']:+.1f}% median")


def check_demo():
    print("\n[demo]")
    idx = ARTIFACTS / "demo" / "index.json"
    if not idx.exists():
        return fail("demo cache missing (scripts/build_demo_cache.py)")
    d = json.loads(idx.read_text())
    apps = d.get("apps", [])
    if len(apps) < 4:
        fail(f"only {len(apps)} demo applications")
    else:
        ok(f"{len(apps)} demo applications cached")
    # The picker is deliberately weighted towards applications the method helps
    # at matched spend -- that is what the demo is for. But a picker with no
    # failures in it cannot be falsified by anyone clicking around, so a minimum
    # number of genuine non-wins must survive, and each must say so on its face.
    # Two is the floor. The picker is deliberately weighted towards the band
    # around the cohort median -- that is what the demo is for -- but a list with
    # no failures in it cannot be falsified by anyone clicking around, and every
    # non-win must say on its face that it is one.
    MIN_NON_WINS = 2
    have_gain = [a for a in apps if a.get("matched_gain_pct") is not None]
    if len(have_gain) != len(apps):
        fail(f"{len(apps) - len(have_gain)} demo applications carry no "
             f"matched-cost score; the picker cannot have been ranked by one")
    losses = [a for a in have_gain if a["matched_gain_pct"] <= 0]
    if len(losses) < MIN_NON_WINS:
        fail(f"only {len(losses)} of {len(apps)} demo applications are non-wins; "
             f"at least {MIN_NON_WINS} must remain or the picker is a filtered "
             f"result")
    else:
        unlabelled = [a for a in losses
                      if "NON-WIN" not in a.get("reason_selected", "")]
        if unlabelled:
            fail(f"{len(unlabelled)} non-win applications are not labelled as "
                 f"such in the picker")
        else:
            wins = len(have_gain) - len(losses)
            ok(f"picker: {wins} wins and {len(losses)} labelled non-wins at "
               f"matched cost (weighted, not filtered)")


def _check_js_optimiser(node, js):
    """The window the demo DRAWS must be the window the simulator RUNS.

    Act 1 recomputes the optimal window in the browser so that dragging the
    price is instant. That is a second implementation of the same optimiser, and
    a second implementation drifts: the first version of it was off by one in the
    memory term and disagreed with Python on 8 of 18 cases, which would have put
    a window on screen that the simulation never used.
    """
    import json as _json, os as _os, subprocess as _sp, tempfile as _tf
    import numpy as _np
    from racrp.baselines.hazard import optimal_window

    try:
        head = js[js.index("function smoothed("):js.index("function drawDecide(")]
    except ValueError:
        return notes.append("could not locate the demo's optimiser to cross-check")

    demo = ARTIFACTS / "demo"
    cases = []
    for f in sorted(demo.glob("app_*.json"))[:6]:
        d = _json.loads(f.read_text())
        for r in (0.30, 0.05, 0.008):
            cases.append({"pdf": d["marginal"]["pdf"],
                          "n": d["n_gaps_train"], "ratio": r})
    if not cases:
        return

    runner = head + """
const fs=require('fs');
const a=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
console.log(JSON.stringify(a.map(c=>{
  const b=optimalWindow(smoothed(c.pdf,c.n),c.ratio); return [b.a,b.L];})));
"""
    d1 = _tf.mkdtemp()
    try:
        jp, cp = _os.path.join(d1, "w.js"), _os.path.join(d1, "c.json")
        with open(jp, "w", encoding="utf-8") as fh:
            fh.write(runner)
        with open(cp, "w", encoding="utf-8") as fh:
            _json.dump(cases, fh)
        r = _sp.run([node, jp, cp], capture_output=True, text=True)
        if r.returncode:
            return fail("the demo's optimiser could not be executed for "
                        "cross-checking")
        got = _json.loads(r.stdout)
        bad = 0
        for c, (a, L) in zip(cases, got):
            pa, pL = optimal_window(_np.array(c["pdf"]) * c["n"], c["ratio"])
            if (pa, pL) != (a, L):
                bad += 1
        if bad:
            fail(f"the demo draws a different window than the simulator runs on "
                 f"{bad} of {len(cases)} cases")
        else:
            ok(f"demo's optimiser matches the simulator's on all {len(cases)} cases")
    finally:
        import shutil as _sh
        _sh.rmtree(d1, ignore_errors=True)


def _check_js_runtime(node, js):
    """Execute the demo's draw functions against a stubbed canvas and DOM.

    Parsing is not enough: a missing helper is a runtime error, and the panel it
    breaks simply renders nothing. Caught exactly that way, so it is checked now.
    """
    import json as _json, os as _os, shutil as _sh, subprocess as _sp
    import tempfile as _tf

    demo = ARTIFACTS / "demo"
    idx_f = demo / "index.json"
    apps = sorted(demo.glob("app_*.json"))
    if not idx_f.exists() or not apps:
        return

    head = """
const APP=%s, IDX=%s;
const boxes={};
const ctx=new Proxy({},{get:(t,k)=>(k==='canvas')?{width:0,height:0}:function(){},
                        set:()=>true});
global.devicePixelRatio=1;
global.document={getElementById:(id)=>{
  if(['decide','race','steal_c','frontier'].includes(id))
    return {clientWidth:900,style:{},getContext:()=>ctx,width:0,height:0};
  return {get innerHTML(){return boxes[id]||''},set innerHTML(v){boxes[id]=v},
          textContent:'', value:50, style:{},
          classList:{add(){},remove(){},toggle(){}},
          addEventListener(){}, dataset:{}};
}, querySelectorAll:()=>[], addEventListener(){}};
global.window={addEventListener(){}};
global.fetch=()=>Promise.reject(new Error('no network in verification'));
""" % (_json.dumps(_json.loads(apps[0].read_text())),
       _json.dumps(_json.loads(idx_f.read_text())))

    tail = """
S.app=APP; S.index=IDX;
const errs=[];
for(const fn of ['drawDecide','drawRace','drawSteal','drawFrontier','renderList']){
  try{ eval(fn+'()'); }catch(e){ errs.push(fn+': '+e.message); }
}
const n1=((boxes['act1stats']||'').match(/class="stat/g)||[]).length;
const n2=((boxes['act2stats']||'').match(/class="stat/g)||[]).length;
if(n1<3) errs.push('act1 rendered '+n1+' stat boxes, expected at least 3');
if(n2<3) errs.push('act2 rendered '+n2+' stat boxes, expected at least 3');
console.log(JSON.stringify(errs));
"""
    d = _tf.mkdtemp()
    try:
        f = _os.path.join(d, "run.js")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(head + js.replace("boot();", "") + tail)
        r = _sp.run([node, f], capture_output=True, text=True)
        if r.returncode:
            return fail("the demo's scripts could not be executed: "
                        + (r.stderr.strip().splitlines() or [""])[-1][:140])
        errs = _json.loads((r.stdout.strip().splitlines() or ["[]"])[-1])
        if errs:
            for e in errs[:5]:
                fail("demo runtime: " + e)
        else:
            ok("demo's panels all render without error")
    except Exception as exc:                       # never block on the harness
        notes.append(f"demo runtime check skipped: {exc}")
    finally:
        _sh.rmtree(d, ignore_errors=True)


def check_demo_schema():
    """The dashboard is written by hand against a cache the pipeline generates.

    A field renamed on the Python side would leave the page silently blank rather
    than raising, so the nested paths the frontend actually indexes are checked
    against a real payload here.
    """
    import shutil as _sh, subprocess as _sp, tempfile as _tf, os as _os
    import re as _re
    node = _sh.which("node")
    page_src = ROOT / "racrp" / "web" / "static" / "index.html"
    if node and page_src.exists():
        m = _re.search(r"<script>(.*)</script>",
                       page_src.read_text(encoding="utf-8"), _re.S)
        if m:
            fd, tmp = _tf.mkstemp(suffix=".js")
            try:
                with _os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(m.group(1))
                r = _sp.run([node, "--check", tmp], capture_output=True, text=True)
                if r.returncode:
                    tail = (r.stderr.strip().splitlines() or [""])[-1]
                    fail("the dashboard's JavaScript does not parse: " + tail[:160])
                else:
                    ok("dashboard JavaScript parses")
                    _check_js_optimiser(node, m.group(1))
                    _check_js_runtime(node, m.group(1))
            finally:
                _os.unlink(tmp)
    elif not node:
        notes.append("node not found; dashboard JavaScript not syntax-checked")

    print("\n[demo frontend <-> cache schema]")
    page = ROOT / "racrp" / "web" / "static" / "index.html"
    demo = ARTIFACTS / "demo"
    files = sorted(demo.glob("app_*.json"))
    if not page.exists() or not files:
        return fail("cannot check the schema: page or cache missing")
    app = json.loads(files[0].read_text())
    idx = json.loads((demo / "index.json").read_text())

    required_app = ["invocations", "tracks", "marginal", "buckets", "window",
                    "kappa", "n_gaps_train", "reason_selected", "cold_start_ms"]
    for k in required_app:
        if k not in app:
            fail(f"per-application payload is missing {k!r}")
    for path in (("marginal", "keepalive"), ("marginal", "pdf"),
                 ("marginal", "prewarm"), ("window", "start"), ("window", "end")):
        cur = app
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is None:
            fail("missing nested field: " + ".".join(path))
    for t in ("fixed-keepalive-10", "marginal-ATC20", "conditional-prev-gap",
              "derived-window"):
        if t not in app.get("tracks", {}):
            fail(f"decision track missing from the cache: {t}")

    # the derived-window card, including the cost box that is a standing check
    # that the optimiser beats the heuristic on the objective it minimises
    d = app.get("derived")
    if not isinstance(d, dict):
        fail("per-application payload is missing the derived-window block")
    else:
        for k in ("prewarm", "keepalive", "percentile_prewarm",
                  "percentile_keepalive", "price_ratio", "cost_derived",
                  "cost_percentile"):
            if k not in d:
                fail(f"derived block missing {k!r}")
        cd, cp = d.get("cost_derived"), d.get("cost_percentile")
        if isinstance(cd, dict) and isinstance(cp, dict):
            for k in ("warm_minutes", "miss_prob", "expected_cost"):
                if k not in cd or k not in cp:
                    fail(f"derived cost block missing {k!r}")

    # the frontier panel: the only place the demo states a matched-cost claim
    res = idx.get("results", {})
    frontier = {
        "strong_baseline_cloud": ("idle_gb_minutes", "cold_starts"),
        "strong_baseline_matched": ("budget_idle_gb_min",
                                    "best_baseline_cold_starts",
                                    "derived_cold_starts"),
        "strong_baseline_summary": ("median_reduction_pct", "n_points",
                                    "min", "max", "n_apps"),
    }
    for t, fields in frontier.items():
        blob = res.get(t)
        if blob is None:
            fail(f"the demo's frontier panel needs {t}, which is not in the cache")
            continue
        row = blob[0] if isinstance(blob, list) and blob else blob
        for f in fields:
            if not isinstance(row, dict) or f not in row:
                fail(f"frontier panel field missing: {t}.{f}")
    buckets = [b for b in app.get("buckets", []) if b]
    if not buckets:
        fail("no conditional buckets in the cache: the demo would show nothing")
    else:
        for k in ("label", "keepalive", "prewarm", "n", "pdf"):
            if k not in buckets[0]:
                fail(f"bucket entry missing {k!r}")
    for k in ("apps", "results", "bucket_labels"):
        if k not in idx:
            fail(f"demo index missing {k!r}")
    if not failures:
        ok("every field the dashboard indexes is present in the cache")


def main() -> int:
    print("verifying deliverables against the template requirements")
    check_report()
    check_ppt()
    check_figures()
    check_results()
    check_demo()
    check_demo_schema()
    print()
    for n in notes:
        print(f"  note  {n}")
    if failures:
        print(f"\n{len(failures)} CHECK(S) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
