"""Copy the two graded deliverables into `FINAL DOCS/`.

The `docs/` directory holds generators, working notes and three documents; only
two of them are handed in. This puts those two somewhere unambiguous, and runs
last in `finalize.py` so the copies can never be older than what was just
verified.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "FINAL DOCS"
FINAL = ("RACRP_Review2_Deck.pptx", "RACRP_Project1_Report.docx")


def main() -> None:
    DEST.mkdir(exist_ok=True)
    for name in FINAL:
        src = ROOT / "paper" / name
        if not src.exists():
            raise SystemExit(f"missing deliverable: {src}")
        shutil.copy2(src, DEST / name)
        kb = (DEST / name).stat().st_size / 1024
        print(f"  {name:42s} {kb:7.0f} KB")

    # anything else in here is a stale hand-copy and will be mistaken for final
    for stray in DEST.iterdir():
        if stray.is_file() and stray.name not in FINAL:
            print(f"  !! not a final deliverable, left in place: {stray.name}")
    print(f"[final] {len(FINAL)} deliverables in {DEST}")


if __name__ == "__main__":
    main()
