"""
Qualitative Grounding Analysis — Sampling Script
=================================================
Generates a 40-row annotation template (20 yes/no + 20 inference) for manual
A / B / C labeling of VLM CoT reasoning traces.

Label definitions
-----------------
A  True visual grounding   : trace explicitly describes an image finding that is
                             consistent with the ground truth (e.g. correctly
                             identifies a structural signal abnormality).
B  Text repetition / vague : trace only restates the MR findings text verbatim,
                             or contains generic boilerplate with no concrete
                             image observation.
C  Hallucination           : trace describes something contradicted by the ground
                             truth (claims a finding that does not exist, or
                             gives the wrong location / severity).

How to annotate
---------------
Open  results/grounding_annotation_template.csv  in Excel or Google Sheets.
For each row, read the [cot_reasoning_trace] column and fill in:
  - [label_A_B_C]      → one of  A / B / C
  - [annotation_notes] → optional short note (e.g. "cites ACL tear but GT says intact")

Running in Colab
----------------
  !git clone https://github.com/YifangLi922/kneecot-h5-pipeline.git
  %cd kneecot-h5-pipeline
  !pip install pandas -q
  !python analysis/sample_grounding_traces.py

The script auto-detects Colab and sets paths accordingly.
"""

import json
import os
import sys
import pandas as pd
from pathlib import Path

# ── Path configuration ────────────────────────────────────────────────────────
IN_COLAB = os.path.exists("/content")
if IN_COLAB:
    REPO_ROOT = Path("/content/kneecot-h5-pipeline")
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent  # one level up from analysis/

BASE = REPO_ROOT / "code" / "kneecot-h5-pipeline-vlm" / "vlm_results"
OUT  = REPO_ROOT / "results" / "grounding_annotation_template.csv"

SEED       = 42
N_PER_TYPE = 20   # 20 yes/no + 20 inference = 40 total

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))

def build_row(item: dict, condition_label: str) -> dict:
    return {
        "case_id":             item["case_id"],
        "qtype":               item.get("qtype", ""),
        "condition":           condition_label,
        "model":               item.get("model", ""),
        "question":            item.get("question", ""),
        "ground_truth":        item.get("ground_truth", ""),
        "model_prediction":    item.get("prediction", ""),
        "mr_findings_text":    item.get("findings", ""),
        "cot_reasoning_trace": item.get("raw_response", ""),
        "label_A_B_C":         "",   # ← fill in: A / B / C
        "annotation_notes":    "",   # ← optional
    }

# ── YES / NO pool ─────────────────────────────────────────────────────────────
# Priority: CoT_image_only (model sees image only, no text findings) preferred
# over CoT_findings (model sees image + MR text). Both filtered to correct=True.
YN_FILES = [
    (BASE / "minicpm-v_CoT_yn.json",         "CoT_image_only"),
    (BASE / "qwen2.5vl_CoT_yn.json",          "CoT_image_only"),
    (BASE / "minicpm-v_CoT_findings_yn.json", "CoT_findings_image+text"),
    (BASE / "qwen2.5vl_CoT_findings_yn.json", "CoT_findings_image+text"),
]

yn_rows = []
for fpath, cond in YN_FILES:
    for item in load_json(fpath):
        if item.get("correct", False):
            yn_rows.append(build_row(item, cond))

yn_df = pd.DataFrame(yn_rows)
# Keep image-only version when the same case+question appears in both conditions
yn_df["_pri"] = yn_df["condition"].map({"CoT_image_only": 0, "CoT_findings_image+text": 1})
yn_df = (yn_df
         .sort_values("_pri")
         .drop_duplicates(subset=["case_id", "question"])
         .drop(columns="_pri")
         .reset_index(drop=True))

print(f"Yes/No pool  : {len(yn_df)} unique correct items")

yn_sample = yn_df.sample(n=N_PER_TYPE, random_state=SEED).sort_values("case_id")

# ── INFERENCE pool ────────────────────────────────────────────────────────────
# Binary correct/incorrect is unavailable for VLM eval inference items
# (LLM-judge scores exist only for the full 200-item corpus whose raw text is
# corrupted in the combined file). We sample from CoT_findings inference items
# (model has image + MR text), which is where text-shortcut risk is highest.
INF_FILES = [
    (BASE / "qwen2.5vl_CoT_findings_inference.json",  "CoT_findings_image+text"),
    (BASE / "minicpm-v_CoT_findings_inference.json",   "CoT_findings_image+text"),
]

inf_rows = []
for fpath, cond in INF_FILES:
    for item in load_json(fpath):
        inf_rows.append(build_row(item, cond))

inf_df = (pd.DataFrame(inf_rows)
          .drop_duplicates(subset=["case_id", "question"])
          .reset_index(drop=True))

print(f"Inference pool: {len(inf_df)} unique items")

inf_sample = inf_df.sample(n=N_PER_TYPE, random_state=SEED).sort_values("case_id")

# ── Combine & export ──────────────────────────────────────────────────────────
final = pd.concat([yn_sample, inf_sample], ignore_index=True)
final.insert(0, "row_id", range(1, len(final) + 1))   # 1-based row number

OUT.parent.mkdir(parents=True, exist_ok=True)
final.to_csv(OUT, index=False, encoding="utf-8-sig")   # utf-8-sig: Excel opens Chinese correctly

print(f"\nExported → {OUT}")
print(f"  yes_no rows    : {(final['qtype'] == 'yes_no').sum()}")
print(f"  inference rows : {(final['qtype'] == 'inference').sum()}")
print(f"\nCondition breakdown:")
print(final.groupby(["qtype", "condition"]).size().rename("count").to_string())
print(f"\nModel breakdown:")
print(final.groupby(["qtype", "model"]).size().rename("count").to_string())
print(f"\nColumns in template:")
for col in final.columns:
    note = " ← fill this in" if col in ("label_A_B_C", "annotation_notes") else ""
    print(f"  {col}{note}")
