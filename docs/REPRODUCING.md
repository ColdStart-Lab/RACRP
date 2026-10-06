# Reproducing

Every number in `artifacts/tables/` is produced by a script in `scripts/`. Each
script writes its JSON beside its figures, so the report, deck and demo can be
rebuilt offline without re-running any experiment.

---

## Without the dataset

The test suite builds its own synthetic traces and needs nothing else. This is
what CI runs.

```bash
make setup
make test          # 53 tests, under 30 seconds
```

The documents and the interactive demonstration also work from the committed
tables and demo cache:

```bash
make demo          # http://127.0.0.1:8000
make verify        # checks every deliverable against artifacts/tables/
```

## With the dataset

Obtain the Azure Functions Trace 2019 into `data/` — see
[`../data/README.md`](../data/README.md) — then parse it once:

```bash
python scripts/run_characterise.py        # ~10 min, ~6 GB peak RAM
```

This writes `data/processed/app_matrix.npz`, a 24,274 × 20,160 matrix of
per-minute invocation counts plus per-application metadata. Everything
downstream reads that cache; the CSVs are never touched again.

---

## Pipeline

In dependency order. Runtimes are for an 8-core laptop CPU; nothing here needs a
GPU.

| # | command | produces | ~time |
| --- | --- | --- | --- |
| 1 | `scripts/run_characterise.py` | `characterisation.json`, the app matrix | 10 min |
| 2 | `scripts/run_baselines.py` | `baselines_test.json` — fixed keep-alive sweep, ATC'20, oracle | 15 min |
| 3 | `scripts/run_information_audit.py` | `context_information.json` — the audit that chose the method | 20 min |
| 4 | `scripts/run_conditional.py` | `conditional_*.json` — Method A and its controls | 60 min |
| 5 | `scripts/tune_smoothing.py` | `smoothing_tuning.json` — prior strength, on validation | 25 min |
| 6 | **`scripts/run_derived.py`** | **`derived_*.json` — the headline and the donor control** | 90 min |
| 7 | **`scripts/run_strong_baseline.py`** | **`strong_baseline_*.json` — the 6×6 baseline envelope** | 50 min |
| 8 | `scripts/run_mechanism_check.py` | `mechanism_check_summary.json` — gain against dependence | 30 min |
| 9 | `scripts/run_sensitivity.py` | `sensitivity.json` — the 15-setting grid | 40 min |

```bash
make headline      # steps 6 and 7 only — the two headline experiments
make results       # the whole pipeline, in order
```

**Order matters in one place only.** `tune_smoothing.py` (5) must run before
`run_derived.py` (6), because step 6 reads `DEFAULT_SMOOTH`, and a test asserts
that constant still matches the value recorded on validation. Everything else is
independent.

### The negative results

Reported in the project, so reproducible here. These are the only parts that want
`torch`, installed via `pip install -e ".[negative-results]"`.

| command | produces | ~time |
| --- | --- | --- |
| `scripts/run_train.py` | the retrieval stack; `prediction_metrics_h5.json` | 5 h CPU |
| `scripts/diag_retrieval_ceiling.py` | `diag_retrieval_ceiling.json` | 40 min |
| `scripts/diag_fleet_ceiling.py` | `diag_fleet_ceiling_H0.json` — includes the oracle bound | 35 min |
| `scripts/run_policy_eval.py` | `policy_eval_*.json` — the 0.98-AP predictor as a policy | 25 min |
| `scripts/run_predictor_ablation.py` | `predictor_ablation.json` | 50 min |

See [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) for what they show.

---

## Rebuilding the deliverables

```bash
make docs
```

Runs `scripts/finalize.py`, which regenerates in order: diagrams, figures,
derived-window figures, the demo cache, the review deck, the project report and
the outcome report — then runs `verify_deliverables.py` and refuses to pass if
any of them disagrees with the tables. `scripts/make_overview_pdf.py` then
renders `overview/PRESENTATION_GUIDE.md` to .docx and PDF.

### What verification actually checks

`scripts/verify_deliverables.py` is the gate, and it is deliberately
uncomfortable:

- the report's section word limits, and that no placeholder text survives;
- that every figure a document references exists;
- that the conditioning arms' **cell-count-matched controls behave** — a control
  more than 5% from zero fails the build;
- that the derived-window result is present **with its donor control**, and that
  the control actually fails — if another application's history worked just as
  well, the central claim would be wrong, and the check says so in those words;
- that `DEFAULT_SMOOTH` still matches the validation record, so a
  hyper-parameter cannot drift toward whatever flatters the headline;
- that the strengthened baseline really swept 36 configurations rather than
  quietly shrinking to something easier to beat;
- that the deck has nothing off-slide or overflowing its box;
- that the demo's JavaScript parses, that its panels render without throwing,
  and that **the window the demo draws equals the window the simulator runs** —
  the browser reimplements the optimiser for responsiveness, and a second
  implementation drifts.

---

## Determinism

Seeded throughout (`--seed 42`, `racrp/config.py`). The simulator is
deterministic; a randomised equivalence test pins the vectorised implementation
against an independent reference loop over 50 trials.

Re-running the pipeline from a clean `artifacts/` should reproduce every table
bit-for-bit. If it does not, that is a bug worth reporting.

## Caching

Expensive things are cached and reused:

| path | holds | regenerate |
| --- | --- | --- |
| `data/processed/app_matrix.npz` | the parsed trace | `run_characterise.py` |
| `artifacts/demo/_scored.json` | per-application matched-cost scores (~20 min) | `build_demo_cache.py --rescore` |
| `artifacts/embeddings/` | retrieval embeddings | `run_train.py` |
| `artifacts/checkpoints/` | model weights | `run_train.py` |

The last three are `.gitignore`d: large, regenerable, and not needed for any
reported number — those all live in `artifacts/tables/`, which is committed.
