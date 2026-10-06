# Data

This repository does not redistribute the trace. It is 2.1 GB and it is not ours
to republish.

## What you need

**Azure Functions Trace 2019**, released by Microsoft alongside Shahrad et al.,
*"Serverless in the wild"*, USENIX ATC 2020. Licensed CC-BY 4.0.

- https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md

Download and unpack it so that this directory contains:

```
data/
├── invocations_per_function_md.anon.d01.csv  …  d14.csv
├── function_durations_percentiles.anon.d01.csv  …  d14.csv
└── app_memory_percentiles.anon.d01.csv  …  d12.csv
```

Three facts about the data that shape everything in this repository:

- **Invocations are per minute.** Each `invocations_*` row is a function, and
  the last 1440 columns are that function's invocation count for each minute of
  the day. Minute granularity is all the data has, which is why the simulator is
  minute-ticked — the ATC'20 policy it is compared against operates on minute
  idle-times too.
- **Durations exclude cold-start time.** The trace records execution time only,
  so cold-start latency is modelled, not measured. See `racrp/config.py`.
- **Memory is allocated per application, not per function.** So warming
  decisions are made at `HashApp` granularity, and functions are aggregated up to
  applications before anything else happens. `app_memory_percentiles` covers days
  1-12 only; days 13-14 fall back to the per-application median.

## Building the cache

```bash
python scripts/run_characterise.py
```

This parses the fourteen daily files once and writes
`data/processed/app_matrix.npz` — a 24,274 x 20,160 matrix of per-minute
invocation counts, plus a per-application metadata table. Everything downstream
reads that cache, so the CSVs are touched exactly once.

Expect ~10 minutes and ~6 GB of peak memory for this step. It is the only part
of the pipeline that is slow for reasons other than simulation.
