#!/usr/bin/env bash
# setup_runpod.sh — one-shot setup + run for the KneeCoT H5 pipeline on a
# rented GPU pod (RunPod or similar single-GPU Linux box).
#
# Assumes this script is run from the repo root, inside a tmux/screen
# session (so it survives SSH disconnects), on a pod with >=24GB VRAM.
#
# What this does, in order:
#   0. sanity checks + Python deps
#   1. install Ollama, start it, pull qwen2.5vl + qwen2.5:32b
#   2. download the dataset (capped by --max-gb so it doesn't fill the disk)
#   3. build the shared eval set
#   4. run the VLM line, then the LLM line
#   5. combine VLM result files, run the judge on both lines
#   6. print the manual-review + compare.py instructions (NOT automated —
#      see Running_Guidelines.md section 4.3, manual review is required
#      before the final numbers can be trusted)
#
# Usage:
#   export HF_TOKEN=hf_xxxxxxxxxxxx
#   ./setup_runpod.sh                # full run
#   N_EVAL=50 ./setup_runpod.sh       # pilot run (50 yn + 50 inference items)
#   MAX_GB=25 ./setup_runpod.sh       # cap dataset download size

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: set HF_TOKEN first, e.g.:"
  echo "  export HF_TOKEN=hf_xxxxxxxxxxxx"
  exit 1
fi

MAX_GB="${MAX_GB:-25}"
N_EVAL="${N_EVAL:-}"          # empty = use all data
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== [0/6] Python dependencies ==="
pip install -q huggingface_hub
pip install -q -r code/kneecot-h5-pipeline-llm/requirements.txt
pip install -q -r code/kneecot-h5-pipeline-vlm/requirements.txt

echo "=== [1/6] Ollama: install, serve, pull models ==="
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! pgrep -x ollama >/dev/null 2>&1; then
  ollama serve > /tmp/ollama_serve.log 2>&1 &
fi
for i in $(seq 1 30); do
  if curl -s http://localhost:11434 >/dev/null 2>&1; then break; fi
  sleep 1
done
ollama pull qwen2.5vl
ollama pull qwen2.5:32b   # judge model — must stay different from qwen2.5vl

echo "=== [2/6] Download dataset (capped at ${MAX_GB} GB) ==="
cd data
python prepare_data.py --token "$HF_TOKEN" --max-gb "$MAX_GB"
cd "$REPO_ROOT"

echo "=== [3/6] Build shared eval set ==="
cd code/kneecot-h5-pipeline-vlm
if [ -n "$N_EVAL" ]; then
  python build_eval_set.py --n-eval "$N_EVAL"
else
  python build_eval_set.py
fi
cd "$REPO_ROOT"

echo "=== [4/6] Generate raw outputs: VLM line, then LLM line ==="
cd code/kneecot-h5-pipeline-vlm
python run.py --eval-set ../../data/eval_set.json
cd "$REPO_ROOT"

cd code/kneecot-h5-pipeline-llm
python run.py --eval_set ../../data/eval_set.json \
  --model_name Qwen/Qwen2.5-7B-Instruct --out_dir results
cd "$REPO_ROOT"

echo "=== [5/6] Combine VLM result files, run the judge on both lines ==="
python -c "
import json, glob
files = sorted(glob.glob('data/vlm_results/qwen2.5vl_*_findings_*.json'))
combined = []
for fp in files:
    combined.extend(json.load(open(fp, encoding='utf-8')))
json.dump(combined, open('data/vlm_results/combined_findings_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'Combined {len(files)} files -> {len(combined)} records')
"

python code/analysis/judge.py \
  --input code/kneecot-h5-pipeline-llm/results/raw_results.json \
  --rubric code/analysis/inference_rubric_for_LLM_judge.json \
  --model qwen2.5:32b \
  --output judged_inference_llm.jsonl \
  --json-output judged_inference_llm.json \
  --review-output manual_review_llm.jsonl

python code/analysis/judge.py \
  --input data/vlm_results/qwen2.5vl_DA_findings_inference.json data/vlm_results/qwen2.5vl_CoT_findings_inference.json \
  --rubric code/analysis/inference_rubric_for_LLM_judge.json \
  --model qwen2.5:32b \
  --output judged_inference_vlm.jsonl \
  --json-output judged_inference_vlm.json \
  --review-output manual_review_vlm.jsonl

cat <<'EOF'

=== [6/6] Generation + judging done. Manual steps before final scoring: ===

1. Open manual_review_llm.jsonl and manual_review_vlm.jsonl, have two people
   independently re-label a shared subset and compute agreement.
2. Fix any wrong judge calls directly in judged_inference_llm.json /
   judged_inference_vlm.json (compare.py reads these, not the .jsonl files).
3. Then run the final comparison:

   python code/analysis/compare.py \
     --eval_set data/eval_set.json \
     --llm_results code/kneecot-h5-pipeline-llm/results/raw_results.json \
     --vlm_results data/vlm_results/combined_findings_results.json \
     --judged_llm judged_inference_llm.json \
     --judged_vlm judged_inference_vlm.json \
     --out_dir compare_out \
     --missing_policy wrong

See Running_Guidelines.md section 4.3 for details on the manual review step.
EOF
