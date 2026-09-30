"""Render the presentation guide to .docx and then to PDF.

There is no pandoc, wkhtmltopdf or LibreOffice on this machine, but Word is
installed, so the route is markdown -> .docx (python-docx, which this project
already uses for the report) -> .pdf (Word via COM). That leaves an editable
document beside the PDF rather than only a flat file.

The markdown subset here is exactly what the guide uses -- headings, bullets to
two levels, numbered lists, horizontal rules, and inline bold / italic / code.
It is not a general markdown implementation and does not pretend to be; anything
it does not recognise is emitted as plain text rather than silently dropped.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor, Inches

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "overview" / "PRESENTATION_GUIDE.md"
DOCX = ROOT / "overview" / "RACRP_Overview.docx"
PDF = ROOT / "overview" / "RACRP_Overview.pdf"

INK = RGBColor(0x1A, 0x1A, 0x1A)
ACCENT = RGBColor(0x6B, 0x2D, 0x8E)
MUTED = RGBColor(0x55, 0x55, 0x55)

INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+?`|\*[^*\n]+?\*)")


def emit_inline(para, text):
    """Split a line into runs, honouring **bold**, *italic* and `code`."""
    for piece in INLINE.split(text):
        if not piece:
            continue
        run = para.add_run()
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            run.text = piece[2:-2]
            run.bold = True
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            run.text = piece[1:-1]
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
            run.font.color.rgb = ACCENT
        elif (piece.startswith("*") and piece.endswith("*")
              and not piece.startswith("**") and len(piece) > 2):
            run.text = piece[1:-1]
            run.italic = True
        else:
            run.text = piece
        if run.font.color.rgb is None:
            run.font.color.rgb = INK


def rule(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("─" * 62)
    r.font.size = Pt(7)
    r.font.color.rgb = RGBColor(0xC8, 0xC8, 0xC8)


def build_docx() -> Path:
    md = SRC.read_text(encoding="utf-8").split("\n")
    doc = Document()

    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(5)
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Inches(0.85)
        sec.top_margin = sec.bottom_margin = Inches(0.75)

    # a paragraph's continuation lines are wrapped in the source; join them so
    # the document rewraps to its own measure instead of keeping hard breaks
    blocks, buf, kind = [], [], None

    def flush():
        nonlocal buf, kind
        if buf:
            blocks.append((kind, " ".join(x.strip() for x in buf)))
            buf, kind = [], None

    for raw in md:
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        if line.startswith("#"):
            flush()
            n = len(line) - len(line.lstrip("#"))
            blocks.append((f"h{n}", line[n:].strip()))
        elif line.strip() == "---":
            flush()
            blocks.append(("hr", ""))
        elif re.match(r"^\s*-\s", line):
            flush()
            indent = len(line) - len(line.lstrip())
            kind = "b1" if indent >= 2 else "b0"
            buf = [re.sub(r"^\s*-\s", "", line)]
        elif re.match(r"^\s*\d+\.\s", line):
            flush()
            kind = "num"
            buf = [line.strip()]
        elif kind in ("b0", "b1", "num") and line.startswith(" "):
            buf.append(line)                      # indented bullet continuation
        elif kind == "p":
            # A markdown paragraph continues until a blank line, and its
            # continuation lines are flush left. Treating them as new paragraphs
            # split every bold span that happened to wrap, leaving literal
            # asterisks in the output -- which is exactly what it did.
            buf.append(line)
        else:
            flush()
            kind = "p"
            buf = [line]
    flush()

    for k, text in blocks:
        if k == "hr":
            rule(doc)
            continue
        if k == "h1":
            p = doc.add_paragraph()
            r = p.add_run(text)
            r.font.size, r.bold, r.font.color.rgb = Pt(21), True, ACCENT
            p.paragraph_format.space_after = Pt(4)
            continue
        if k in ("h2", "h3"):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(13 if k == "h2" else 9)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            r = p.add_run(text)
            r.bold = True
            r.font.size = Pt(14 if k == "h2" else 11.5)
            r.font.color.rgb = ACCENT if k == "h2" else INK
            continue

        p = doc.add_paragraph()
        pf = p.paragraph_format
        if k == "b0":
            pf.left_indent, pf.first_line_indent = Inches(0.26), Inches(-0.16)
            emit_inline(p, "•   " + text)
        elif k == "b1":
            pf.left_indent, pf.first_line_indent = Inches(0.56), Inches(-0.16)
            emit_inline(p, "–   " + text)
        elif k == "num":
            pf.left_indent, pf.first_line_indent = Inches(0.30), Inches(-0.20)
            emit_inline(p, text)
        else:
            emit_inline(p, text)
        pf.space_after = Pt(4.5)

    DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(DOCX))
    return DOCX


def to_pdf(docx_path: Path) -> Path | None:
    """Convert through Word. Returns None if Word is not usable."""
    try:
        import win32com.client as win32
    except ImportError:
        try:
            import comtypes.client as ct
        except ImportError:
            return None
        word = ct.CreateObject("Word.Application")
        word.Visible = False
        d = word.Documents.Open(str(docx_path))
        d.SaveAs(str(PDF), FileFormat=17)
        d.Close()
        word.Quit()
        return PDF

    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    try:
        d = word.Documents.Open(str(docx_path))
        d.SaveAs(str(PDF), FileFormat=17)      # wdFormatPDF
        d.Close(0)
    finally:
        word.Quit()
    return PDF


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    d = build_docx()
    print(f"[overview] wrote {d.name}  ({d.stat().st_size/1024:.0f} KB)")
    p = to_pdf(d)
    if p and p.exists():
        print(f"[overview] wrote {p.name}  ({p.stat().st_size/1024:.0f} KB)")
    else:
        print("[overview] Word not reachable from Python; the .docx is there and "
              "'Save as PDF' from Word will finish the job")


if __name__ == "__main__":
    main()
