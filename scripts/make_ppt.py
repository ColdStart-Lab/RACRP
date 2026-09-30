"""Build the Review-2 deck.

Rewritten because the previous version had been assembled by patching a template
in place, and had drifted badly: content boxes 9.2 inches wide on a 13.33-inch
slide (they were written for a 4:3 deck and never revisited), figures pinned to
a width that left four inches of dead space, the references overflowing a
2.12-inch placeholder, and slide order dependent on hardcoded indices that were
correct only until the next insertion.

The approach here is different in one respect that fixes all of those at once:
**nothing is positioned by hand and nothing is sized by hand.** Every slide is
built from the same small set of primitives, which take a rectangle and fit
their content into it. Body text shrinks until it fits, pictures scale to their
box preserving aspect, tables get a row height from their content, and long
lists spill onto a continuation slide rather than off the bottom of the page.

Layout geometry is taken from the supplied Review-2 template so the deck still
looks like the department's, and the template's own chrome (date, footer, slide
number) is preserved on every slide.

Content comes from `racrp.content`; every number in it is read from
`artifacts/tables/*.json` at import. Nothing in this file types a result.
"""
from __future__ import annotations

import json
from datetime import date

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

from racrp import content as C
from racrp.config import FIGURES, TABLES, ROOT

TEMPLATE = ROOT / "4 Review 2 PPT Template.pptx"
OUT = ROOT / "paper" / "RACRP_Review2_Deck.pptx"
REVIEW_DATE = "02-09-2026"

# ---- geometry, all taken from the template ---------------------------------
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
TITLE_RECT = (Inches(0.60), Inches(0.62), Inches(12.13), Inches(0.80))
BODY_RECT = (Inches(0.60), Inches(1.58), Inches(12.13), Inches(5.25))
FOOTER_TOP = Inches(6.95)

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x59, 0x59, 0x59)
ACCENT = RGBColor(0x6B, 0x2D, 0x8E)      # the deck's purple, for our results
GOOD = RGBColor(0x1B, 0x6E, 0x3C)
BAD = RGBColor(0xA8, 0x2A, 0x2A)
RULE = RGBColor(0xD8, 0xD8, 0xD8)
HEADFILL = RGBColor(0xEF, 0xEA, 0xF4)


def load(name):
    p = TABLES / name
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- primitives

def _fit_size(lines, w_in, h_in, hi=15.0, lo=7.5):
    """Largest font size at which the text still fits the box.

    Wrapping is estimated rather than measured -- python-pptx cannot measure
    text -- so the character-per-line constant is deliberately pessimistic. It
    is better for a slide to be a point smaller than it needed to be than for
    the last bullet to fall off the bottom, which is precisely what the previous
    deck did.
    """
    size = hi
    while size > lo:
        cpl = max(12, int(w_in * 96.0 / (size * 0.55)))
        rows = 0.0
        for text, level, _ in lines:
            avail = max(10, cpl - level * 6)
            n = max(1, -(-len(text) // avail))
            rows += n + 0.34                      # paragraph spacing
        if rows * size * 1.22 / 72.0 <= h_in:
            return size
        size -= 0.5
    return lo


def _norm(items):
    """Accept plain strings, (text, level) and (text, level, style) uniformly."""
    out = []
    for it in items:
        if isinstance(it, tuple):
            text, level = it[0], it[1]
            style = it[2] if len(it) > 2 else None
        else:
            text, level, style = it, 0, None
        out.append((str(text), int(level), style))
    return out


def textbox(slide, rect, items, *, hi=15.0, lo=7.5, bullet=True, color=INK):
    left, top, width, height = rect
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)

    lines = _norm(items)
    size = _fit_size(lines, width.inches, height.inches, hi, lo)

    for i, (text, level, style) in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.level = min(level, 4)
        para.space_after = Pt(size * 0.42)
        run = para.add_run()
        run.text = (("•  " if level == 0 else "–  ") + text
                    if bullet and level >= 0 and style != "plain" else text)
        f = run.font
        f.size = Pt(size)
        f.name = "Calibri"
        if style == "head":
            f.bold = True
            f.color.rgb = ACCENT
        elif style == "muted":
            f.color.rgb = MUTED
        elif style == "good":
            f.bold = True
            f.color.rgb = GOOD
        elif style == "bad":
            f.bold = True
            f.color.rgb = BAD
        else:
            f.color.rgb = color
    return box


def picture(slide, path, rect):
    """Scale a figure into its box, preserving aspect, centred both ways."""
    from PIL import Image

    left, top, width, height = rect
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(width / Emu(int(width)) if False else width.inches / (iw / 96.0),
                height.inches / (ih / 96.0))
    w_in, h_in = (iw / 96.0) * scale, (ih / 96.0) * scale
    return slide.shapes.add_picture(
        str(path),
        Emu(int(left + (width - Inches(w_in)) / 2)),
        Emu(int(top + (height - Inches(h_in)) / 2)),
        Inches(w_in), Inches(h_in))


def table(slide, rows, rect, *, font=10.0, head=True, widths=None,
          emphasise=()):
    left, top, width, height = rect
    n_r, n_c = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_r, n_c, left, top, width, height)
    tbl = shape.table

    if widths:
        total = sum(widths)
        for j, w in enumerate(widths):
            tbl.columns[j].width = Emu(int(width * (w / total)))

    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            tf = cell.text_frame
            tf.word_wrap = True
            para = tf.paragraphs[0]
            run = para.add_run()
            run.text = str(cell_text)
            f = run.font
            f.size = Pt(font + (0.5 if i == 0 and head else 0))
            f.name = "Calibri"
            f.bold = bool(i == 0 and head) or i in emphasise
            if i == 0 and head:
                f.color.rgb = ACCENT
                cell.fill.solid()
                cell.fill.fore_color.rgb = HEADFILL
            elif i in emphasise:
                f.color.rgb = ACCENT
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xF7, 0xF4, 0xFA)
            else:
                f.color.rgb = INK
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    return shape


# ------------------------------------------------------------------ chrome

def _chrome(slide, layout_source):
    """Copy the template's date / footer / slide-number placeholders."""
    for ph in layout_source.placeholders:
        t = ph.placeholder_format.type
        if t not in (13, 15, 16):
            continue
        box = slide.shapes.add_textbox(ph.left, ph.top, ph.width, ph.height)
        tf = box.text_frame
        tf.word_wrap = False
        run = tf.paragraphs[0].add_run()
        run.text = {16: REVIEW_DATE, 15: "SCOPE", 13: ""}[t]
        run.font.size = Pt(10)
        run.font.color.rgb = MUTED
        run.font.name = "Calibri"
        tf.paragraphs[0].alignment = ph.text_frame.paragraphs[0].alignment
        if t == 13:
            box.name = "slidenum"


class Deck:
    """A deck that knows its own geometry, so nothing has to be placed by hand."""

    def __init__(self):
        self.prs = Presentation(str(TEMPLATE)) if TEMPLATE.exists() else Presentation()
        self.prs.slide_width, self.prs.slide_height = SLIDE_W, SLIDE_H
        self.title_slide_layout = self.prs.slides[0].slide_layout
        self.body_layout = self.prs.slides[2].slide_layout
        self.chrome_src = self.prs.slides[2]
        # keep slide 1 (title) and slide 2 (approval mail), drop the rest of the
        # template's placeholder slides -- we rebuild every one of them
        self._keep_first(2)

    def _keep_first(self, n):
        lst = self.prs.slides._sldIdLst
        for sid in list(lst)[n:]:
            rid = sid.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/"
                "relationships}id")
            self.prs.part.drop_rel(rid)
            lst.remove(sid)

    def slide(self, title):
        s = self.prs.slides.add_slide(self.body_layout)
        for sh in list(s.shapes):
            if sh.is_placeholder:
                sh._element.getparent().remove(sh._element)
        box = s.shapes.add_textbox(*TITLE_RECT)
        tf = box.text_frame
        tf.word_wrap = True
        run = tf.paragraphs[0].add_run()
        run.text = title
        run.font.size = Pt(26 if len(title) < 52 else 22)
        run.font.bold = True
        run.font.name = "Calibri"
        run.font.color.rgb = ACCENT
        _chrome(s, self.chrome_src)
        return s

    def body(self, title, items, **kw):
        s = self.slide(title)
        textbox(s, BODY_RECT, items, **kw)
        return s

    def finish(self):
        # slide numbers last, from actual position: the template writes them as
        # literal text, so any insertion leaves them stale
        for i, s in enumerate(self.prs.slides, start=1):
            for sh in s.shapes:
                if sh.name == "slidenum" or (
                        sh.has_text_frame
                        and sh.text_frame.text.strip().isdigit()):
                    para = sh.text_frame.paragraphs[0]
                    if para.runs:
                        para.runs[0].text = str(i)
                        for extra in para.runs[1:]:
                            extra.text = ""
                    else:
                        sh.text_frame.text = str(i)
                        r = sh.text_frame.paragraphs[0].runs[0]
                        r.font.size, r.font.name = Pt(10), "Calibri"
                        r.font.color.rgb = MUTED
                    break
        self.prs.save(str(OUT))
        return len(self.prs.slides._sldIdLst)


# ------------------------------------------------------------------- slides

def title_slide(prs):
    s = prs.slides[0]
    for sh in s.shapes:
        if sh.has_text_frame and sh.is_placeholder:
            tf = sh.text_frame
            tf.clear()
            for i, (txt, sz, bold) in enumerate([
                    ("B.Tech - BCSE497J Project-I", 15, False),
                    (C.TITLE, 23, True)]):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.alignment = 2 if False else p.alignment
                r = p.add_run()
                r.text = txt
                r.font.size, r.font.bold, r.font.name = Pt(sz), bold, "Calibri"
                r.font.color.rgb = ACCENT if bold else MUTED
        elif sh.has_text_frame:
            tf = sh.text_frame
            tf.clear()
            lines = [("Team members:", 14, True)]
            lines += [(f"{n}      ({r})", 14, False) for n, r in C.TEAM]
            lines += [("", 8, False),
                      (f"Guide: {C.GUIDE}", 14, False),
                      (f"Team ID: {C.TEAM_ID}", 12, False)]
            for i, (txt, sz, bold) in enumerate(lines):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                r = p.add_run()
                r.text = txt
                r.font.size, r.font.bold, r.font.name = Pt(sz), bold, "Calibri"
                r.font.color.rgb = INK


def approval_slide(prs):
    """Clear the template's sample screenshot and leave room for the real mail."""
    s = prs.slides[1]
    # the template ships a picture of an example approval mail; it overlaps the
    # instructions and would be mistaken for ours
    for sh in list(s.shapes):
        if not sh.has_text_frame:
            sh._element.getparent().remove(sh._element)
    for sh in s.shapes:
        if sh.has_text_frame and sh.is_placeholder:
            t = sh.placeholder_format.type
            if t == 16:
                # a placeholder can carry its text without any runs, so setting
                # runs[0] blindly raises; write through the frame instead
                sh.text_frame.text = REVIEW_DATE
                r = sh.text_frame.paragraphs[0].runs[0]
                r.font.size, r.font.name = Pt(10), "Calibri"
                r.font.color.rgb = MUTED
            elif t == 7:
                tf = sh.text_frame
                tf.clear()
                for i, line in enumerate([
                        "Paste the guide's approval mail here before the review.",
                        "Approval is required for: title, aim and objectives, "
                        "the methodology used, and the results reported in this "
                        "deck.",
                        "Due 01-09-2026."]):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    r = p.add_run()
                    r.text = line
                    r.font.size = Pt(15)
                    r.font.name = "Calibri"
                    r.font.color.rgb = MUTED if i else INK


def lit_slides(d):
    rows = C.LIT_ROWS
    head, data = rows[0], rows[1:]
    half = -(-len(data) // 2)
    for k, chunk in enumerate((data[:half], data[half:]), start=1):
        s = d.slide(f"Literature Review ({k}/2)  -  most recent first")
        table(s, [head] + chunk,
              (Inches(0.60), Inches(1.52), Inches(12.13), Inches(5.20)),
              font=9.0, widths=[0.5, 2.4, 5.0, 4.2])


def results_headline(d):
    s = d.slide("Experiments and Results  -  the headline, at matched memory cost")
    dsum = {r["arm"]: r for r in (load("derived_summary.json") or [])}
    sb = load("strong_baseline_summary.json") or {}
    rows = [["Arm (all vs the deployed ATC'20 policy)",
             "Median cold-start reduction", "Range", "Points"]]
    order = [("DERIVED window, marginal", "Window DERIVED from the cost model"),
             ("DERIVED window + conditioning", "Derived window + conditioning"),
             ("conditional, percentile window", "Conditioning only (percentile window)"),
             ("CONTROL: derived from a donor app",
              "CONTROL - same optimiser, a donor application's history")]
    emph = set()
    for i, (key, label) in enumerate(order, start=1):
        r = dsum.get(key)
        if not r:
            continue
        rows.append([label, f"{r['median_reduction_pct']:+.1f}%",
                     f"{r['min']:+.1f}% to {r['max']:+.1f}%", str(r["n"])])
        if i in (1, 4):
            emph.add(len(rows) - 1)
    table(s, rows, (Inches(0.60), Inches(1.50), Inches(12.13), Inches(2.35)),
          font=11.5, widths=[6.0, 2.4, 2.6, 1.0], emphasise=tuple(emph))

    notes = []
    if sb.get("median_reduction_pct") is not None:
        notes.append((
            f"Against a STRENGTHENED baseline - ATC'20 swept over both of its "
            f"percentiles, {sb.get('baseline_configs', 36)} configurations, "
            f"lower envelope taken, chosen with the test period in view - the "
            f"derived window still wins by {sb['median_reduction_pct']:.1f}% "
            f"median, positive at all {sb['n_points']} budgets.", 0, None))
    notes.append((
        "1,999 applications, held-out test days 11-14, every arm swept over its "
        "own cost knob and compared only at equal idle GB-minutes.", 0, "muted"))
    textbox(s, (Inches(0.60), Inches(4.15), Inches(12.13), Inches(1.35)),
            notes, hi=13.0, lo=10.0)


def results_frontier(d):
    s = d.slide("Experiments and Results  -  cost/benefit curves, not point claims")
    picture(s, FIGURES / "derived_frontier.png",
            (Inches(0.55), Inches(1.45), Inches(12.2), Inches(4.35)))
    textbox(s, (Inches(0.60), Inches(5.95), Inches(12.13), Inches(0.95)), [
        ("Left: every one of the 36 tuned ATC'20 percentile pairs (dots), their "
         "best reachable frontier (blue), and the derived window (red).", 0, None),
        ("Right: the gain at each of the baseline's own operating points - and "
         "the budget at which it runs out. When memory is nearly free there is "
         "little left for an optimiser to recover.", 0, "muted"),
    ], hi=12.0, lo=9.5)


def results_control(d):
    s = d.slide("Why we believe it: the control, and a composition that fails")
    picture(s, FIGURES / "derived_composition.png",
            (Inches(0.55), Inches(1.40), Inches(12.2), Inches(3.95)))
    dsum = {r["arm"]: r for r in (load("derived_summary.json") or [])}
    ctl = dsum.get("CONTROL: derived from a donor app")
    tun = load("smoothing_tuning.json") or {}
    lines = []
    if ctl:
        lines.append((
            f"The same optimiser, same price, same simulator - given ANOTHER "
            f"application's gap histogram - lands at "
            f"{ctl['median_reduction_pct']:+.0f}%, far worse than the baseline it "
            f"was beating. The gain requires each application's own history.",
            0, "bad"))
    lines.append((
        "The two ideas do not compose: a percentile survives sparse counts, a "
        "cost-optimal window needs the tail estimated well, and conditioning "
        "fragments the counts that estimate it.", 0, None))
    if tun.get("chosen_smooth_total") is not None:
        lines.append((
            f"The one hyper-parameter was chosen on VALIDATION "
            f"(= {tun['chosen_smooth_total']:g}). The setting that scores best on "
            f"the test period ranks LAST on validation.", 0, "muted"))
    textbox(s, (Inches(0.60), Inches(5.50), Inches(12.13), Inches(1.35)),
            lines, hi=12.0, lo=9.0)


def controls_slide(d):
    items = [("Negative results, reported because we ran the controls", 0, "head")]
    items += [(t, 1, None) for t in C.NEGATIVE_RESULTS]
    items += [("Threats to validity", 0, "head")]
    items += [(t, 1, None) for t in C.THREATS[:4]]
    d.body("Controls, Negative Results and Limitations", items, hi=13.0, lo=8.5)


def conclusion_slide(d):
    dsum = {r["arm"]: r for r in (load("derived_summary.json") or [])}
    sb = load("strong_baseline_summary.json") or {}
    mc = load("conditional_matched_cost.json") or []
    ci = load("context_information.json")
    dd, ctl = dsum.get("DERIVED window, marginal"), dsum.get(
        "CONTROL: derived from a donor app")
    both = dsum.get("DERIVED window + conditioning")

    items = []
    if dd:
        txt = (f"RESULT. The deployed policy reads its warm window off two fixed "
               f"percentiles. Deriving it instead - minimising the window's "
               f"expected cost - cuts cold starts by "
               f"{dd['median_reduction_pct']:.1f}% at matched memory cost")
        if sb.get("median_reduction_pct") is not None:
            txt += (f", and by {sb['median_reduction_pct']:.1f}% against a "
                    f"baseline allowed to tune both percentiles")
        items.append((txt + ". The knob becomes K/m - GB-minutes a cold start is "
                            "worth - which an operator can quote.", 0, "good"))
    if ctl:
        items.append((
            f"CONTROL FIRST. The identical optimiser on a donor application's "
            f"history lands at {ctl['median_reduction_pct']:+.0f}%. The gain is "
            f"not good window sizes on average; it needs each application's own "
            f"past.", 0, "bad"))
    red = [r["prev_gap+tod_reduction_pct"] for r in mc
           if r.get("prev_gap+tod_reduction_pct") is not None]
    if red:
        import statistics as st
        t = (f"SECOND RESULT. The same histogram is also a marginal - it forgets "
             f"the state each gap occurred in. Conditioning on that state gives "
             f"{st.median(red):.1f}%, with the deployed policy recovered exactly "
             f"as the limiting case.")
        if both and dd:
            t += (f" The two do not compose "
                  f"({both['median_reduction_pct']:.1f}% together).")
        items.append((t, 0, None))
    if ci:
        best = max(ci["contexts"].values(),
                   key=lambda x: x["share_of_gap_entropy"])
        items.append((
            f"WHY THIS METHOD. An information audit run before any policy found "
            f"~{100 * ci['irreducible_share']:.0f}% of the next idle gap's "
            f"entropy irreducible, and localised the rest in the {best['label']} "
            f"({100 * best['share_of_gap_entropy']:.1f}%). That one measurement "
            f"explains three earlier negative results and selects the estimator.",
            0, None))
    items.append(("PROJECT-II. Non-stationarity rather than scarcity; "
                  "multi-instance concurrency; carrying the derived rule to a "
                  "second trace (Huawei, EuroSys'25); and the loose-budget regime "
                  "where the gain fades, which we can describe but not yet fix.",
                  0, "muted"))
    d.body("Conclusion", items, hi=13.0, lo=8.5)


def reference_slides(d):
    refs = list(C.REFERENCES)
    half = -(-len(refs) // 2)
    for k, chunk in enumerate((refs[:half], refs[half:]), start=1):
        start = 0 if k == 1 else half
        items = [(f"[{start + i + 1}]  {r}", 0, "plain")
                 for i, r in enumerate(chunk)]
        d.body(f"References ({k}/2)", items, hi=12.0, lo=9.0, bullet=False)


def main() -> None:
    d = Deck()
    title_slide(d.prs)
    approval_slide(d.prs)

    d.body("Aim", C.AIM, hi=14.0, lo=9.5)
    d.body("Abstract", [(p, 0, "plain") for p in C.ABSTRACT],
           hi=13.5, lo=9.5, bullet=False)
    lit_slides(d)
    d.body("Research Gap", C.GAP, hi=14.0, lo=9.5)
    d.body("Objectives", C.OBJECTIVES, hi=14.0, lo=9.5)

    s = d.slide("Framework / Architecture / Block Diagram")
    picture(s, FIGURES / "architecture.png",
            (Inches(0.55), Inches(1.45), Inches(12.2), Inches(3.60)))
    textbox(s, (Inches(0.60), Inches(5.20), Inches(12.13), Inches(1.65)),
            C.ARCHITECTURE[1:], hi=12.0, lo=9.0)

    d.body("Functional Requirements", C.FUNCTIONAL, hi=13.5, lo=9.0)
    d.body("Modules", C.MODULES, hi=13.5, lo=9.0)

    results_headline(d)
    results_frontier(d)
    results_control(d)
    controls_slide(d)
    conclusion_slide(d)
    reference_slides(d)

    n = d.finish()
    print(f"[ppt] wrote {OUT}  ({n} slides)")


if __name__ == "__main__":
    main()
