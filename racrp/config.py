"""Central configuration. No magic numbers anywhere else."""
from __future__ import annotations
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
TABLES = ARTIFACTS / "tables"
CKPT = ARTIFACTS / "checkpoints"
EMBED = ARTIFACTS / "embeddings"
for _p in (PROCESSED, CACHE, FIGURES, TABLES, CKPT, EMBED):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------- trace layout
N_DAYS = 14
MIN_PER_DAY = 1440
N_MINUTES = N_DAYS * MIN_PER_DAY          # 20160

# Temporal splits (day indices, 1-based inclusive). Never shuffle across these.
TRAIN_DAYS = (1, 8)
VAL_DAYS = (9, 10)
TEST_DAYS = (11, 14)

def day_range_to_minutes(rng: tuple[int, int]) -> tuple[int, int]:
    d0, d1 = rng
    return (d0 - 1) * MIN_PER_DAY, d1 * MIN_PER_DAY

TRAIN_SLICE = day_range_to_minutes(TRAIN_DAYS)
VAL_SLICE = day_range_to_minutes(VAL_DAYS)
TEST_SLICE = day_range_to_minutes(TEST_DAYS)

# ------------------------------------------------------------- app cohort
# Full trace has ~50k apps. We sample a stratified cohort for tractability;
# the sample is stratified by activity decile and dominant trigger, seeded.
COHORT_SIZE = 2000
MIN_INVOCATIONS_14D = 20      # drop apps that essentially never run

# ------------------------------------------------------------- simulator
# Cold-start latency model. The trace does not record cold-start time, so we use
# the values measured for Azure/AWS in the literature (Shahrad ATC'20 Fig. 12,
# Wang et al. ATC'18): a fixed platform component plus a memory-dependent term.
COLD_START_BASE_MS = 500.0        # container + runtime bring-up
COLD_START_PER_GB_MS = 1000.0     # dependency/state load, scales with app size
DEFAULT_APP_MEMORY_MB = 256.0     # when the memory file has no row for the app

# Cost model. Memory-time held warm while idle is the resource we pay for.
# Reported as GB-minutes; monetary conversion uses AWS Lambda GB-second pricing.
USD_PER_GB_SECOND = 0.0000166667

# ------------------------------------------------------------- prediction task
HORIZONS = (1, 5, 10)             # Delta t, minutes
PRIMARY_HORIZON = 5
WINDOW = 240                      # minutes of invocation history fed to the model
STRIDE_TRAIN = 5                  # windows every N minutes when building train set
STRIDE_EVAL = 1

# ------------------------------------------------------------- model
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
DROPOUT = 0.1
EMB_DIM = 128                     # situation embedding dimension
TOP_K = 8                         # retrieved neighbours
OUTCOME_LEN = 60                  # minutes of future profile stored per memory entry

BATCH_SIZE = 512
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 12
PATIENCE = 3

DEVICE = os.environ.get("RACRP_DEVICE", "cuda")

# ------------------------------------------------------------- policy
# Risk-aware policy keeps an app warm iff risk * cold_penalty > lam * mem_cost.
LAMBDA_SWEEP = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 8.0, 16.0, 40.0)
KEEPALIVE_SWEEP = (1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120)
