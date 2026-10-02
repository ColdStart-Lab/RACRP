"""FastAPI backend for the RACRP demonstration.

Serves only precomputed artefacts from ``artifacts/demo``. No model is loaded and
no GPU is touched, so the demo starts instantly and cannot fail in the room
because of a missing checkpoint or a slow forward pass.

Run:
    python -m racrp.web.server          # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from racrp.config import ARTIFACTS

DEMO = ARTIFACTS / "demo"
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="RACRP — Retrieval-Augmented Cold-Start Risk Prediction")


TRACE_HELP = (
    "The live re-simulation needs the Azure Functions trace, which this "
    "repository does not redistribute. See data/README.md, then run "
    "scripts/run_characterise.py. The cached acts of the demonstration work "
    "without it."
)


@lru_cache(maxsize=1)
def _trace():
    """The parsed trace, loaded once per process.

    Every live request needs it and it is far too large to reload per call, so
    this is memoised at module level rather than inside the endpoint.
    """
    from racrp.data.trace import build_app_matrix
    return build_app_matrix()


@lru_cache(maxsize=64)
def _row_for(hash_prefix: str) -> int:
    _, table = _trace()
    hits = table.index[table["HashApp"].astype(str).str.startswith(hash_prefix)]
    if len(hits) != 1:
        raise HTTPException(404, f"{len(hits)} applications match {hash_prefix!r}")
    return int(hits[0])


def _read(p: Path):
    if not p.exists():
        raise HTTPException(
            404, f"{p.name} not found. Run scripts/build_demo_cache.py first.")
    return json.loads(p.read_text())


@app.on_event("startup")
def _warm() -> None:
    """Load the trace before the first request rather than during it.

    The first live re-simulation would otherwise pay ~3 s to parse the trace,
    and it would pay it in front of an audience mid-drag.
    """
    if not (ARTIFACTS.parent / "data" / "processed" / "app_matrix.npz").exists():
        print("[web] no trace cache; the live acts are disabled but the "
              "precomputed demonstration works. See data/README.md.")
        return
    try:
        _trace()
    except Exception as exc:                     # a demo must still start
        print(f"[web] trace not preloaded: {exc}")


@app.get("/", response_class=HTMLResponse)
def index_page() -> str:
    page = STATIC / "index.html"
    if not page.exists():
        return "<h1>RACRP demo</h1><p>Frontend not built.</p>"
    return page.read_text(encoding="utf-8")


@app.get("/api/index")
def api_index() -> JSONResponse:
    """Application picker + all headline results."""
    return JSONResponse(_read(DEMO / "index.json"))


@app.get("/api/app/{app_id}")
def api_app(app_id: int) -> JSONResponse:
    """Timeline, policy decision tracks, risk trace and retrieved episodes."""
    return JSONResponse(_read(DEMO / f"app_{app_id}.json"))


@app.get("/api/simulate/{app_id}")
def api_simulate(
    app_id: int,
    price_ratio: float = Query(0.05, gt=0.0, le=5.0),
    tail: float = Query(92.0, ge=50.0, le=99.9),
    donor: int | None = Query(None),
) -> JSONResponse:
    """Re-derive the window and re-simulate one application, live.

    ``price_ratio`` is m/K -- the cost of holding a warm minute divided by the
    cost of a cold start. Its reciprocal is the number the demo shows: how many
    GB-minutes of idle memory one cold start is worth. ``tail`` is the percentile
    the ATC'20 baseline reads its keep-alive off, exposed so the two knobs can be
    moved side by side and seen to be different KINDS of quantity.

    Nothing here is cached. The policies, the simulator and the accounting are
    the same objects the experiments call, so what appears on screen is a real
    run of the method and not a replay of one.
    """
    import numpy as np

    from racrp.config import TEST_SLICE, PROCESSED
    from racrp.baselines.conditional import conditional_policy
    from racrp.baselines.hazard import hazard_policy, optimal_window
    from racrp.baselines.fleet import hist_to_quantiles, quantiles_to_windows
    from racrp.baselines.histogram import DEFAULT_RANGE
    from racrp.sim.simulator import simulate_app

    meta_path = DEMO / f"app_{app_id}.json"
    cached = _read(meta_path)
    lo, hi = cached["window"]["start"], cached["window"]["end"]
    mem = float(cached["memory_mb"])

    # Without the trace the cached acts still work, so this is a 503 with an
    # explanation rather than a 500 with a stack trace.
    if not (PROCESSED / "app_matrix.npz").exists():
        raise HTTPException(503, TRACE_HELP)
    try:
        counts, _ = _trace()
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise HTTPException(503, TRACE_HELP) from exc
    inv = counts[_row_for(cached["hash"])].astype(np.int64)

    # the control, on demand: derive the window from ANOTHER application's
    # history and change nothing else
    donor_hist = None
    if donor is not None and donor != app_id:
        dc = _read(DEMO / f"app_{donor}.json")
        dinv = counts[_row_for(dc["hash"])].astype(np.int64)
        da = np.flatnonzero(dinv[:lo] > 0)
        donor_hist = np.zeros(DEFAULT_RANGE + 1)
        if len(da) > 1:
            np.add.at(donor_hist, np.minimum(np.diff(da), DEFAULT_RANGE), 1.0)

    out = {"app": app_id, "price_ratio": price_ratio, "tail": tail,
           "donor": donor,
           "gb_minutes_per_cold_start": round(1.0 / price_ratio, 1),
           "tracks": {}}

    for name, warm in (
        ("marginal-ATC20", conditional_policy(inv, eval_slice=TEST_SLICE,
                                              key="none", tail_margin=tail)),
        ("conditional-prev-gap", conditional_policy(inv, eval_slice=TEST_SLICE,
                                                    key="prev_gap",
                                                    tail_margin=tail)),
        ("derived-window", hazard_policy(inv, eval_slice=TEST_SLICE, key="none",
                                         price_ratio=price_ratio, joint=True,
                                         donor_hist=donor_hist)),
    ):
        w = np.asarray(warm)
        res, states = simulate_app(inv[lo:hi], w[lo:hi], mem,
                                   trace_decisions=True)
        out["tracks"][name] = {"states": states.astype(int).tolist(),
                               "summary": res.as_dict()}

    # what each rule picks from the history observable at the start of the
    # evaluation window, so the numbers on screen can be tied to a window
    act = np.flatnonzero(inv[:lo] > 0)
    hist = np.zeros(DEFAULT_RANGE + 1)
    if len(act) > 1:
        np.add.at(hist, np.minimum(np.diff(act), DEFAULT_RANGE), 1.0)
    off, ln = optimal_window(donor_hist if donor_hist is not None else hist,
                             price_ratio)
    pw, ka = quantiles_to_windows(hist_to_quantiles(hist), 5.0, tail, 10)
    out["hist"] = [round(float(v), 5) for v in (hist / max(hist.sum(), 1))]
    out["windows"] = {"derived_prewarm": int(off), "derived_keepalive": int(ln),
                      "percentile_prewarm": int(pw),
                      "percentile_keepalive": int(ka)}
    return JSONResponse(out)


@app.get("/api/health")
def health() -> dict:
    apps = sorted(p.stem.replace("app_", "") for p in DEMO.glob("app_*.json"))
    return {"status": "ok" if (DEMO / "index.json").exists() else "cache missing",
            "apps_available": len(apps),
            "demo_dir": str(DEMO)}


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
