# Running Guidelines — KneeCoT H5 Pipeline on a Rented GPU Pod

This document is the step-by-step runbook for executing the full H5 pipeline
(data download → generation → scoring → comparison). It assumes the code in
`code/` is already in the state described in the main `README.md`.

In practice, this pipeline was run on a single GPU pod rented on
[RunPod](https://www.runpod.io/) (an **NVIDIA RTX PRO 6000**), not a
university HPC/SLURM cluster due to the limitation of computing power— `setup_runpod.sh` automates exactly this
setup end-to-end. The manual steps below still apply to any single-GPU
Linux box with enough VRAM (≥24GB) to serve both the VLM and the 32B judge
model through Ollama; substitute your own GPU provider's setup steps for "0.2
Install dependencies" if you are not using RunPod.

The pipeline has four phases, matching the plan in `notes/coding problems
and solutions.md`:

0. Setup
1. Generate raw outputs (LLM line + VLM line) — the GPU-heavy phase
2. Score the raw outputs (yes/no parsing + LLM-as-judge for inference)
3. Compare the two lines and produce the result tables
4. (Writing phase — not covered here)

All paths below are relative to the repository root unless stated otherwise.

---

## 0. Setup

### 0.1 Get HuggingFace dataset access

KneeCoT requires HuggingFace account approval from the dataset authors.
Request access at:
https://huggingface.co/datasets/YiHui0124/KneeCoT

Then create a **read** token at https://huggingface.co/settings/tokens.
Do **not** commit this token anywhere.

### 0.2 Install dependencies

```bash
pip install huggingface_hub          # for data/prepare_data.py
pip install -r code/kneecot-h5-pipeline-llm/requirements.txt
pip install -r code/kneecot-h5-pipeline-vlm/requirements.txt
```

The LLM line needs a GPU (Qwen2.5-7B-Instruct, 4-bit) — the actual run used
a single rented RunPod GPU pod with an NVIDIA RTX PRO 6000. The VLM line
talks to a local **Ollama** server instead of loading weights directly in
Python, so make sure Ollama is installed and reachable on the node you run
on:

```bash
ollama serve &                 # start the local Ollama server (background)
ollama pull qwen2.5vl          # VLM model used by evaluate.py
ollama pull qwen2.5:32b        # judge model used by judge.py — must be a
                                # different, stronger model than the one
                                # being evaluated (no self-grading)
```

### 0.3 Check this repo is clean

`.gitignore` is already set up to exclude the downloaded dataset, generated
results, and tokens. Do not `git add -f` anything under `data/`, `results/`,
`vlm_results/`, or `compare_out/`.

---

## 1. Download the dataset

```bash
cd data
python prepare_data.py --token hf_xxxxxxxxxxxx
cd ..
```

This downloads `.nii` MRI volumes and `.json` annotations from
`YiHui0124/KneeCoT` and lays them out as:

```
data/
├── train/                  # .nii files
├── test/                   # .nii files
└── annotations/
    ├── train/               # .json files
    └── test/                # .json files
```

This is exactly the layout `code/kneecot-h5-pipeline-vlm/config.py`
expects (`TRAIN_NII`, `TEST_NII`, `TRAIN_ANN`, `TEST_ANN`), so no path
changes are needed for the next step.

If disk space is limited you can cap the download with `--max-gb` (see
`data/README.md`), but a full download is simpler and avoids ending up with
annotation files that have no matching `.nii` volume. Re-running the script
is safe — it skips files that already exist.

---

## 2. Build the shared evaluation set (`eval_set.json`)

Both the LLM line and the VLM line must be scored on **exactly the same
questions**. That shared question list is built once, by the VLM side's
`build_eval_set.py` (it is the one that also resolves and attaches each
case's matching `.nii` path):

```bash
cd code/kneecot-h5-pipeline-vlm
python build_eval_set.py                 # use ALL available data
# or, for a smaller pilot run:
python build_eval_set.py --n-eval 50      # 50 yes/no + 50 inference items
cd ../..
```

This writes the frozen question list to `data/eval_set.json`. **Do not
build a second, separate eval set from the LLM side's own
`build_eval_set.py`/`--data_dir` flow** — that was the old, pre-alignment
workflow and will produce a question list that does not match the VLM
line's, breaking the matched comparison in step 4.

---

## 3. Phase 1 — Generate raw outputs

This is the GPU/compute-heavy phase. Both lines only produce raw,
**unscored** per-question records here.

### 3.1 VLM line

```bash
cd code/kneecot-h5-pipeline-vlm
python run.py --eval-set ../../data/eval_set.json
cd ../..
```

`run.py` runs, in order: slice extraction (`preprocessing.py`) → eval-set
load (`build_eval_set.py`, skipped since we pass `--eval-set`) → Ollama
inference (`evaluate.py`) → a descriptive metrics summary (`metrics.py`,
optional/informational only — not the official scoring step).

Useful flags:
- `--skip-preprocess` — skip slice extraction if PNGs already exist
- `--eval-only` — skip straight to inference + metrics

Raw output lands in `data/vlm_results/`, one file per model × condition ×
question type, e.g.:
```
data/vlm_results/qwen2.5vl_DA_findings_yn.json
data/vlm_results/qwen2.5vl_DA_findings_inference.json
data/vlm_results/qwen2.5vl_CoT_findings_yn.json
data/vlm_results/qwen2.5vl_CoT_findings_inference.json
```
(plus the image-only ablation files `..._DA_...`/`..._CoT_...` without
`_findings`, used only for the RQ3 ablation analysis later.)

The matched comparison with the LLM line uses the **`_findings`** condition
files (image + MR findings text) — that is the condition where the VLM
sees the same text evidence as the text-only LLM, plus the image.

### 3.2 LLM line

```bash
cd code/kneecot-h5-pipeline-llm
python run.py --eval_set ../../data/eval_set.json --model_name Qwen/Qwen2.5-7B-Instruct --out_dir results
cd ../..
```

This writes `code/kneecot-h5-pipeline-llm/results/raw_results.json`,
containing both the `DA` and `CoT` conditions in one file (split later by
the `prompt_key` field). To smoke-test the pipeline without a GPU first,
add `--mock`.

**Do not pass `--data_dir` here** — passing `--eval_set` makes the script
load the shared question list directly and skip its own (now-deprecated)
case-loading path.

---

## 4. Phase 2 — Score the raw outputs

### 4.1 Combine the VLM line's result files into one

`compare.py` (the shared scoring/comparison script) expects either four
single-condition files, or one combined file per line. The LLM line already
writes one combined file (`raw_results.json`); the VLM line writes one file
per condition × qtype, so combine the matched-condition files first:

```bash
python -c "
import json, glob
files = sorted(glob.glob('data/vlm_results/qwen2.5vl_*_findings_*.json'))
combined = []
for fp in files:
    combined.extend(json.load(open(fp, encoding='utf-8')))
json.dump(combined, open('data/vlm_results/combined_findings_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'Combined {len(files)} files -> {len(combined)} records')
"
```

### 4.2 Score yes/no questions

Yes/no scoring happens automatically inside `compare.py` (step 5) — there is
no separate script to run for this.

### 4.3 Score inference questions (LLM-as-judge + manual review)

Run the local judge model (`qwen2.5:32b`, pulled in step 0.2) separately
over each line's inference records:

```bash
# LLM line
python code/analysis/judge.py \
  --input code/kneecot-h5-pipeline-llm/results/raw_results.json \
  --rubric code/analysis/inference_rubric_for_LLM_judge.json \
  --model qwen2.5:32b \
  --output judged_inference_llm.jsonl \
  --json-output judged_inference_llm.json \
  --review-output manual_review_llm.jsonl

# VLM line (matched DA_findings / CoT_findings conditions only)
python code/analysis/judge.py \
  --input data/vlm_results/qwen2.5vl_DA_findings_inference.json data/vlm_results/qwen2.5vl_CoT_findings_inference.json \
  --rubric code/analysis/inference_rubric_for_LLM_judge.json \
  --model qwen2.5:32b \
  --output judged_inference_vlm.jsonl \
  --json-output judged_inference_vlm.json \
  --review-output manual_review_vlm.jsonl
```

`judge.py` only calls the local Ollama API — no data leaves the machine.
On a SLURM-managed HPC cluster, submit each of these as its own batch job
with GPU access; on a rented single-GPU pod (e.g. the RunPod RTX PRO 6000
setup actually used here), just run them sequentially on the same pod —
Ollama serves the 32B judge model the same way either setup.

**Manual review (required, do not skip):**
1. Open `manual_review_llm.jsonl` and `manual_review_vlm.jsonl`. These
   already contain every `unclear` / `incorrect` / flagged record, plus a
   random ~20% sample of records the judge marked `correct`.
2. Have two people independently re-label a shared subset (e.g. 15–20
   items) and compute agreement (human-human and human-judge).
3. For any record the judge got wrong, edit the corresponding entry directly
   in `judged_inference_llm.json` / `judged_inference_vlm.json` (the files
   `compare.py` actually reads in step 5) — editing only the
   `*_review.jsonl` sample file has no effect on the final scoring.
4. Record what fraction was reviewed and the agreement rates — this goes in
   the paper's evaluation section.

---

## 5. Phase 3 — Compare and produce result tables

```bash
python code/analysis/compare.py \
  --eval_set data/eval_set.json \
  --llm_results code/kneecot-h5-pipeline-llm/results/raw_results.json \
  --vlm_results data/vlm_results/combined_findings_results.json \
  --judged_llm judged_inference_llm.json \
  --judged_vlm judged_inference_vlm.json \
  --out_dir compare_out \
  --missing_policy wrong
```

This is the **only** scoring/comparison script for the whole project — both
lines are judged by the same yes/no parser and the same judge-then-review
process. Outputs in `compare_out/`:

- `summary_2x2_comparison.json` — accuracy per condition (LLM direct/CoT, VLM
  direct/CoT) for yes/no and inference, plus McNemar's test for RQ1 (LLM CoT
  vs direct) and RQ2 (VLM CoT vs direct)
- `per_item.csv` — every question's correct/incorrect outcome under all
  four conditions
- `rq3_yesno.csv` / `rq3_inference.csv` — per-item LLM-vs-VLM comparison
  under the CoT condition, labelled `vision_necessary` /
  `text_better` / `both_correct_text_sufficient` / `both_wrong` — this is
  the RQ3 (visual necessity) breakdown

For the RQ3 **image-only ablation** (VLM with no MR findings text), rerun
`compare.py` with `--vlm_prompt_direct DA --vlm_prompt_cot CoT` (no
`_findings` suffix) pointed at a combined file built from the plain
`DA`/`CoT` result files instead of the `_findings` ones.

---

## 6. Quick recap of the full command order

```bash
# 0. setup: install deps, start Ollama, pull qwen2.5vl + qwen2.5:32b
# 1. download data
cd data && python prepare_data.py --token hf_xxx && cd ..
# 2. build shared eval set
cd code/kneecot-h5-pipeline-vlm && python build_eval_set.py && cd ../..
# 3. generate raw outputs
cd code/kneecot-h5-pipeline-vlm && python run.py --eval-set ../../data/eval_set.json --eval-only && cd ../..
cd code/kneecot-h5-pipeline-llm && python run.py --eval_set ../../data/eval_set.json --out_dir results && cd ../..
# 4. score: combine VLM files, then run the judge for both lines
python -c "..."   # see 4.1
python code/analysis/judge.py --input .../raw_results.json ... --json-output judged_inference_llm.json
python code/analysis/judge.py --input .../*_findings_inference.json ... --json-output judged_inference_vlm.json
# (manual review step — see 4.3)
# 5. compare
python code/analysis/compare.py --eval_set data/eval_set.json --llm_results ... --vlm_results ... --judged_llm ... --judged_vlm ... --out_dir compare_out
```

---

## 7. Troubleshooting

- **`0 QA pairs loaded` / `0 cases` from `build_eval_set.py`** — check that
  step 1 actually populated `data/annotations/train`/`test` and
  `data/train`/`test`; the script needs both an annotation JSON and a
  matching `.nii` file per case.
- **`compare.py` reports many `missing` records** — usually means the
  `question_id`s in `eval_set.json` don't match the ones in the raw results
  file. This should not happen if you always pass `--eval_set` (rather than
  letting either line build its own eval set independently).
- **Ollama connection errors in `evaluate.py`/`judge.py`** — confirm
  `ollama serve` is running and the right model has been `ollama pull`-ed
  on the same node/job.
- **Inference questions show as `not_judged` in `summary_2x2_comparison.json`** — means
  `judge.py` hasn't been run yet (or wasn't passed via `--judged_llm`/
  `--judged_vlm`); yes/no scoring works independently of this.
