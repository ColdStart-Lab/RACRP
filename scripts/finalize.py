"""Regenerate every artefact and deliverable, then gate on the verifier.

Run after any experiment changes. Order matters: diagrams and figures are read by
the documents, the demo cache is read by the dashboard, and the verifier checks
the lot against the templates' own requirements and this project's integrity
conditions.

Exits non-zero if verification fails, so a broken rebuild cannot pass silently.
"""
from __future__ import annotations

import subprocess
import sys
import time

from racrp.config import ROOT

STEPS = [
    ("tests", [sys.executable, "-m", "pytest", "tests/", "-q"]),
    ("diagrams", [sys.executable, "scripts/make_diagrams.py"]),
    ("figures", [sys.executable, "scripts/make_figures.py"]),
    ("finding figures", [sys.executable, "scripts/make_finding_figures.py"]),
    ("derived-window figures", [sys.executable,
                                "scripts/make_derived_figures.py"]),
    ("demo cache", [sys.executable, "scripts/build_demo_cache.py"]),
    ("slides", [sys.executable, "scripts/make_ppt.py"]),
    ("report", [sys.executable, "scripts/make_report.py"]),
    ("outcome report", [sys.executable, "scripts/make_outcome_report.py"]),
    ("verify", [sys.executable, "scripts/verify_deliverables.py"]),
    ("final docs", [sys.executable, "scripts/publish_final.py"]),
]


def main() -> int:
    failed = []
    for name, cmd in STEPS:
        t0 = time.time()
        print(f"\n=== {name} " + "=" * (60 - len(name)))
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        tail = (r.stdout or "").strip().splitlines()[-12:]
        for line in tail:
            print("   ", line)
        if r.returncode != 0:
            failed.append(name)
            err = (r.stderr or "").strip().splitlines()[-8:]
            for line in err:
                print("  !!", line)
        print(f"    [{name}: {'ok' if r.returncode == 0 else 'FAILED'} "
              f"in {time.time()-t0:.0f}s]")

    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("everything regenerated and verified")
    print("\ndeliverables:")
    for f in sorted((ROOT / "paper").glob("RACRP_*")):
        print(f"  {f.name:42s} {f.stat().st_size/1024:8.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
