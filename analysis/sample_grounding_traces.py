"""
Qualitative Grounding Analysis — Sampling Script
=================================================
Generates a 40-row annotation template (20 yes/no + 20 inference) for manual
A / B / C labeling of Qwen2.5-VL CoT reasoning traces.

Sampling design
---------------
Yes/No  (20 rows): correct=True items from Qwen2.5-VL CoT yes/no results.
                   CoT_image_only (model sees images only, no MR text) is
                   preferred; CoT_findings_image+text fills remaining slots.

Inference (20 rows): correct=True items from the LLM-judge–scored inference
                     results (judged_inference_vlm.json, condition=CoT_findings).
                     All 118 correctly-answered items are from Qwen2.5-VL with
                     image + MR findings text provided — the condition where
                     text-shortcut risk is highest.

Label definitions
-----------------
A  True visual grounding   : trace explicitly describes an image finding that is
                             consistent with the ground truth.
B  Text repetition / vague : trace restates the MR findings text verbatim, or
                             contains generic boilerplate with no concrete image
                             observation.
C  Hallucination           : trace describes something contradicted by the ground
                             truth (wrong finding, wrong location/severity).

How to annotate
---------------
Open  results/grounding_annotation_template.csv  in Excel or Google Sheets.
For each row read [cot_reasoning_trace] (the model's full step-by-step reasoning)
and set:
  [label_A_B_C]      → A / B / C
  [annotation_notes] → optional note, e.g. "cites ACL tear but GT says intact"

For inference rows [ground_truth] is the full reference answer; use it to judge
whether the model's reasoning is consistent with the gold standard.

Running in Colab
----------------
  !git clone https://github.com/YifangLi922/kneecot-h5-pipeline.git
  %cd kneecot-h5-pipeline
  !pip install pandas -q
  !python analysis/sample_grounding_traces.py

Output appears at  results/grounding_annotation_template.csv
(download from the Colab left-panel file browser, or run the next cell):
  from google.colab import files
  files.download('results/grounding_annotation_template.csv')
"""

import json
import os
import pandas as pd
from pathlib import Path

# ── Path configuration (auto-detects Colab) ───────────────────────────────────
IN_COLAB  = os.path.exists("/content")
REPO_ROOT = Path("/content/kneecot-h5-pipeline") if IN_COLAB else Path(__file__).resolve().parent.parent
BASE      = REPO_ROOT / "code" / "kneecot-h5-pipeline-vlm" / "vlm_results"
OUT       = REPO_ROOT / "results" / "grounding_annotation_template.csv"

SEED       = 42
N_PER_TYPE = 20   # 20 yes/no + 20 inference = 40 total

# ── YES / NO pool ─────────────────────────────────────────────────────────────
# Qwen2.5-VL only; correct=True; prefer CoT_image_only over CoT_findings.
YN_FILES = [
    (BASE / "qwen2.5vl_CoT_yn.json",          "CoT_image_only"),
    (BASE / "qwen2.5vl_CoT_findings_yn.json",  "CoT_findings_image+text"),
]

yn_rows = []
for fpath, cond in YN_FILES:
    items = json.loads(fpath.read_text(encoding="utf-8"))
    for item in items:
        if not item.get("correct", False):
            continue
        yn_rows.append({
            "case_id":             item["case_id"],
            "qtype":               "yes_no",
            "condition":           cond,
            "model":               item["model"],
            "question":            item["question"],
            "ground_truth":        item["ground_truth"],
            "model_prediction":    item.get("prediction", ""),
            "mr_findings_text":    item.get("findings", ""),
            "cot_reasoning_trace": item["raw_response"],
            "label_A_B_C":         "",
            "annotation_notes":    "",
        })

yn_df = pd.DataFrame(yn_rows)
# Keep one row per (case_id, question): prefer image-only over image+text
yn_df["_pri"] = yn_df["condition"].map({"CoT_image_only": 0, "CoT_findings_image+text": 1})
yn_df = (yn_df
         .sort_values("_pri")
         .drop_duplicates(subset=["case_id", "question"])
         .drop(columns="_pri")
         .reset_index(drop=True))

print(f"Yes/No pool  : {len(yn_df)} unique correct items (Qwen2.5-VL)")
assert len(yn_df) >= N_PER_TYPE, (
    f"Not enough yes/no items: have {len(yn_df)}, need {N_PER_TYPE}"
)

yn_sample = yn_df.sample(n=N_PER_TYPE, random_state=SEED).sort_values("case_id")

# ── INFERENCE pool ────────────────────────────────────────────────────────────
# Source: LLM-judge–scored results (judged_inference_vlm.json).
# Filter: condition=CoT_findings, correct=True → 118 Qwen2.5-VL items.
# Fields:
#   candidate_model_answer  = full raw CoT trace (same as raw_response in source file)
#   extracted_model_conclusion = extracted final answer (clean, no trace text)
#   expected_answer         = full reference answer used by the judge
#   mr_findings             = MR report text available to the model

JUDGED_PATH = REPO_ROOT / "results" / "judged_inference_vlm.json"
judged_all  = json.loads(JUDGED_PATH.read_text(encoding="utf-8"))

inf_rows = []
for j in judged_all:
    if j.get("condition") != "CoT_findings":
        continue
    if not j.get("correct", False):
        continue
    inf_rows.append({
        "case_id":             j["case_id"],
        "qtype":               "inference",
        "condition":           "CoT_findings_image+text",
        "model":               "qwen2.5vl",
        "question":            j["question"],
        "ground_truth":        j["expected_answer"],       # full reference answer
        "model_prediction":    j["extracted_model_conclusion"],  # extracted conclusion only
        "mr_findings_text":    j["mr_findings"],
        "cot_reasoning_trace": j["candidate_model_answer"],      # full CoT trace
        "label_A_B_C":         "",
        "annotation_notes":    "",
    })

inf_df = (pd.DataFrame(inf_rows)
          .drop_duplicates(subset=["case_id", "question"])
          .reset_index(drop=True))

print(f"Inference pool: {len(inf_df)} correct items (Qwen2.5-VL, CoT_findings, judge-verified)")
assert len(inf_df) >= N_PER_TYPE, (
    f"Not enough inference items: have {len(inf_df)}, need {N_PER_TYPE}"
)

inf_sample = inf_df.sample(n=N_PER_TYPE, random_state=SEED).sort_values("case_id")

# ── Combine & export ──────────────────────────────────────────────────────────
final = pd.concat([yn_sample, inf_sample], ignore_index=True)
final.insert(0, "row_id", range(1, len(final) + 1))

OUT.parent.mkdir(parents=True, exist_ok=True)
final.to_csv(OUT, index=False, encoding="utf-8-sig")  # utf-8-sig: Excel opens Chinese correctly

print(f"\nExported {len(final)} rows  →  {OUT}")
print(f"  yes_no rows    : {(final['qtype']=='yes_no').sum()}")
print(f"  inference rows : {(final['qtype']=='inference').sum()}")
print()
print("Condition breakdown:")
print(final.groupby(["qtype","condition"]).size().rename("count").to_string())
print()
print("Columns in annotation template:")
for col in final.columns:
    tag = "  ← FILL IN" if col in ("label_A_B_C", "annotation_notes") else ""
    print(f"  {col}{tag}")
