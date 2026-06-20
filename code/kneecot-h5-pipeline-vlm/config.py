"""
config.py  –  Shared paths, model toggles, and constants for the VLM pipeline.
Edit this file to point to your local data and choose which models to run.
"""
import os
from pathlib import Path

# ── Root paths ───────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent.parent.resolve()  # auto-detect project root
DATA_ROOT = ROOT / "data"

TRAIN_NII   = DATA_ROOT / "train"
TEST_NII    = DATA_ROOT / "test"
TRAIN_ANN   = DATA_ROOT / "annotations" / "train"
TEST_ANN    = DATA_ROOT / "annotations" / "test"
TRAIN_SLC   = DATA_ROOT / "slices"      / "train"
TEST_SLC    = DATA_ROOT / "slices"      / "test"
EVAL_PATH   = DATA_ROOT / "eval_set.json"
RESULTS_DIR = DATA_ROOT / "vlm_results"

for d in [TRAIN_SLC, TEST_SLC, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Slice dirs (used by evaluate.py encode_images) ────────────────────────────────
SLICE_DIRS = {
    "train": str(TRAIN_SLC),
    "test":  str(TEST_SLC),
}

# ── Model toggle ────────────────────────────────────────────────────────────────────
MODELS = {
    "qwen2.5vl": {"enabled": True,  "note": "PRIMARY"},
    "minicpm-v": {"enabled": False, "note": "Secondary"},
}
ACTIVE_MODELS = [m for m, cfg in MODELS.items() if cfg["enabled"]]

# ── Evaluation size ──────────────────────────────────────────────────────────────────
N_EVAL = 50   # set to None for all available cases

# ── Prompt conditions ─────────────────────────────────────────────────────────────────
# Plain strings — must match keys in VLM_PROMPTS (prompts.py)
CONDITIONS = ["DA", "CoT", "DA_findings", "CoT_findings"]
