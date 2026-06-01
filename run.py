"""End-to-end runner for the H5 text-only LLM line.

Pipeline:  preprocessing -> inference -> evaluation

Usage (Colab or HPC, from the repo root):
    python run.py --data_dir data/cases --sample_size 50
    python run.py --data_dir data/cases --sample_size 5 --mock   # no GPU needed

--mock skips loading the model and produces fake outputs, so you can verify the
data + parsing + scoring pipeline on a laptop before spending GPU time.
"""
import argparse
import hashlib
import os
import sys

# Make the modules in src/ importable when running `python run.py` from root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from preprocessing import build_eval_set, describe_eval_set  # noqa: E402
from evaluation import (  # noqa: E402
    collect_inference_outputs,
    mcnemar_direct_vs_cot,
    save_json,
    yes_no_accuracy,
)


def mock_run_inference(items, prompt_modes=("direct", "cot")):
    """Deterministic fake generator for --mock (no model, no GPU)."""
    results = []
    for item in items:
        for mode in prompt_modes:
            if item["qtype"] == "yes_no":
                # deterministic pseudo-answer so accuracy isn't degenerate
                h = hashlib.md5(f"{item['question']}{mode}".encode()).hexdigest()
                guess = "Yes" if int(h, 16) % 2 == 0 else "No"
                raw = f"(mock {mode} output)\n【答案】{guess}"
            else:
                raw = f"(mock {mode} output)\n【答案】这是一个用于测试管道的占位推理结论。"
            rec = dict(item)
            rec["prompt_mode"] = mode
            rec["raw_output"] = raw
            results.append(rec)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/cases",
                   help="folder of KneeCoT *.json annotation files")
    p.add_argument("--sample_size", type=int, default=50,
                   help="number of knee-only cases (None-like 0 = use all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--mock", action="store_true",
                   help="run without a model to test the pipeline")
    args = p.parse_args()

    sample_size = args.sample_size if args.sample_size > 0 else None
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Preprocessing -------------------------------------------------------
    items = build_eval_set(args.data_dir, sample_size=sample_size, seed=args.seed)
    comp = describe_eval_set(items)
    print(f"Eval set: {comp}")
    if not items:
        sys.exit("No items found. Check --data_dir and that JSON files exist.")

    # 2. Inference -----------------------------------------------------------
    if args.mock:
        results = mock_run_inference(items)
    else:
        from inference import load_model, run_inference, save_results
        model, tokenizer = load_model(args.model_name)
        results = run_inference(items, model, tokenizer)
        save_results(results, os.path.join(args.out_dir, "raw_results.json"))

    # 3. Evaluation ----------------------------------------------------------
    acc = yes_no_accuracy(results)
    mcnemar = mcnemar_direct_vs_cot(results)
    inf_outputs = collect_inference_outputs(results)

    save_json(acc, os.path.join(args.out_dir, "yes_no_accuracy.json"))
    save_json(mcnemar, os.path.join(args.out_dir, "mcnemar.json"))
    save_json(inf_outputs, os.path.join(args.out_dir, "inference_outputs.json"))

    print("\n=== Yes/No accuracy by prompt mode ===")
    for mode, s in acc.items():
        print(f"  {mode:7s}  acc={s['accuracy']:.3f}  "
              f"({s['correct']}/{s['scored']}, unparsed={s['unparsed']})")
    print("\n=== McNemar (direct vs CoT) ===")
    print(f"  {mcnemar}")
    print(f"\nInference outputs collected for later judging: {len(inf_outputs)}")
    print(f"All outputs saved under: {args.out_dir}/")


if __name__ == "__main__":
    main()
