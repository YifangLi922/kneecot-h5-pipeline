"""End-to-end runner for the H5 text-only LLM line.

Pipeline:  build_eval_set -> inference -> save raw results

  --eval_set   : load a prebuilt eval_set.json (shared with VLM line)
  --data_dir   : build the eval set on the fly from raw case JSONs
                 (uses ALL data unless --sample_size is also given)
  --sample_size: limit yes/no AND inference to this many items each
  --mock       : run without a model to test the pipeline end-to-end

This runner used to also compute yes/no accuracy, McNemar, and an inference
accuracy heuristic right here. That has been removed on purpose: scoring is
now a separate, shared step (see code_new/analysis/compare.py and
judge.py) so that the LLM and VLM lines are always judged by the same
code instead of two independently-evolving copies. This script's only job is
to produce results/raw_results.json with the fields case_id, question_id,
qtype, ground_truth, prompt_key, raw_output -- nothing here is scored.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from preprocessing import build_eval_set, describe_eval_set
from evaluation import save_json

# Prompt keys must match VLM prompt keys (DA / CoT)
PROMPT_MODES = ("DA", "CoT")


def mock_run_inference(items, prompt_modes=PROMPT_MODES):
    results = []
    for item in items:
        for mode in prompt_modes:
            if item["qtype"] == "yesno":
                h = hashlib.md5("{}{}" .format(item["question"], mode).encode()).hexdigest()
                guess = "Yes" if int(h, 16) % 2 == 0 else "No"
                raw = "(mock {} output)\n【答案】{}".format(mode, guess)
            else:
                raw = "(mock {} output)\n【答案】这是一个用于测试管道的占位推理结论。".format(mode)
            rec = dict(item)
            rec["prompt_key"] = mode
            rec["raw_output"] = raw
            results.append(rec)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/cases")
    p.add_argument(
        "--sample_size", type=int, default=None,
        help="Max items per qtype (yes/no and inference) to sample. "
             "Omit (or set to 0) to use ALL data in --data_dir.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--eval_set", default=None,
                   help="path to a prebuilt eval_set.json (shared with VLM, overrides --data_dir)")
    p.add_argument("--output_eval", default=None,
                   help="save the built eval set to this path")
    p.add_argument("--mock", action="store_true",
                   help="run without a model to test the pipeline")
    args = p.parse_args()

    # None or 0 both mean "use all data"
    sample_size = args.sample_size if (args.sample_size and args.sample_size > 0) else None

    if sample_size is None:
        print("sample_size not set → using ALL data from: {}".format(args.data_dir))
    else:
        print("sample_size={} → sampling from: {}".format(sample_size, args.data_dir))

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load or build eval set ────────────────────────────────────────────────
    if args.eval_set and os.path.exists(args.eval_set):
        with open(args.eval_set, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Support both flat list (VLM / new build_eval_set.py)
        # and wrapped dict {"items": [...]} (legacy format)
        items = raw if isinstance(raw, list) else raw.get("items", raw)
        source_desc = os.path.basename(args.eval_set)
    else:
        output_eval = args.output_eval or os.path.join(args.out_dir, "eval_set.json")
        items = build_eval_set(args.data_dir, sample_size=sample_size,
                               seed=args.seed, output_path=output_eval)
        source_desc = "built from " + args.data_dir

    comp = describe_eval_set(items)
    print("Eval set ({}): yesno={} (Yes={}, No={}), inference={}".format(
        source_desc, comp["yes_no"], comp["yes"], comp["no"], comp["inference"]))
    if not items:
        sys.exit("No items found. Check --data_dir and that JSON files exist.")

    # ── Inference (generation only -- no scoring) ───────────────────────────
    out_path = os.path.join(args.out_dir, "raw_results.json")

    if args.mock:
        results = mock_run_inference(items)
        save_json(results, out_path)
    else:
        from inference import load_model, run_inference, save_results, checkpoint_path
        model, tokenizer = load_model(args.model_name)
        results = run_inference(items, model, tokenizer,
                                 checkpoint_file=checkpoint_path(out_path))
        save_results(results, out_path)

    print("Saved {} raw records -> {}".format(len(results), out_path))

    print(
        "\nGeneration done. This script does not score results anymore.\n"
        "Run the shared scoring step next:\n"
        "  yes/no   -> python code_new/analysis/compare.py ...\n"
        "  inference -> python judge.py --input {} ...\n".format(out_path)
    )


if __name__ == "__main__":
    main()
