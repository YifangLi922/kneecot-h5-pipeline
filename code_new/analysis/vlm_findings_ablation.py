"""
vlm_findings_ablation.py
-------------------------
VLM-only ablation: does adding MR findings text help, within each prompt
style? Compares DA_findings vs DA, and CoT_findings vs CoT — both VLM,
no LLM line involved. Reuses compare.py's scoring/McNemar code so the
yes/no parsing and inference judging stay identical to the main analysis.

Usage:
    python vlm_findings_ablation.py \
        --eval_set data/eval_set.json \
        --findings_results data/vlm_results/combined_findings_results.json \
        --ablation_results data/vlm_results/combined_ablation_results.json \
        --judged_findings judged_inference_vlm.json \
        --judged_ablation judged_inference_vlm_ablation.json \
        --out_dir compare_out_vlm_findings_ablation \
        --missing_policy wrong
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import load_eval_set, load_run, load_judged, score_run, accuracy, mcnemar


def block(eval_set, scores, qids, label_a, label_b):
    return {
        "n": len(qids),
        "acc": {
            label_a: accuracy(scores["a"], qids),
            label_b: accuracy(scores["b"], qids),
        },
        f"mcnemar_{label_b}_vs_{label_a}": mcnemar(scores["a"], scores["b"], qids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--findings_results", required=True, help="combined_findings_results.json")
    ap.add_argument("--ablation_results", required=True, help="combined_ablation_results.json")
    ap.add_argument("--judged_findings", default=None, help="judged_inference_vlm.json")
    ap.add_argument("--judged_ablation", default=None, help="judged_inference_vlm_ablation.json")
    ap.add_argument("--out_dir", default="compare_out_vlm_findings_ablation")
    ap.add_argument("--missing_policy", choices=["wrong", "drop"], default="wrong")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    eval_set = load_eval_set(args.eval_set)
    judged_findings = load_judged(args.judged_findings)
    judged_ablation = load_judged(args.judged_ablation)

    runs = {
        "DA_findings": load_run(args.findings_results, filter_prompt_key="DA_findings"),
        "CoT_findings": load_run(args.findings_results, filter_prompt_key="CoT_findings"),
        "DA": load_run(args.ablation_results, filter_prompt_key="DA"),
        "CoT": load_run(args.ablation_results, filter_prompt_key="CoT"),
    }
    judged_for = {
        "DA_findings": (judged_findings, "DA_findings"),
        "CoT_findings": (judged_findings, "CoT_findings"),
        "DA": (judged_ablation, "DA"),
        "CoT": (judged_ablation, "CoT"),
    }

    scores = {
        name: score_run(eval_set, run, *judged_for[name], args.missing_policy)
        for name, run in runs.items()
    }

    common_da = sorted(set(scores["DA_findings"]) & set(scores["DA"]), key=str)
    common_cot = sorted(set(scores["CoT_findings"]) & set(scores["CoT"]), key=str)

    yesno_da = [q for q in common_da if eval_set[q]["qtype"] == "yesno"]
    infer_da = [q for q in common_da if eval_set[q]["qtype"] == "inference"]
    yesno_cot = [q for q in common_cot if eval_set[q]["qtype"] == "yesno"]
    infer_cot = [q for q in common_cot if eval_set[q]["qtype"] == "inference"]

    summary = {
        "direct_style_yes_no": block(eval_set, {"a": scores["DA"], "b": scores["DA_findings"]},
                                      yesno_da, "DA_image_only", "DA_findings_image_plus_text"),
        "cot_style_yes_no": block(eval_set, {"a": scores["CoT"], "b": scores["CoT_findings"]},
                                   yesno_cot, "CoT_image_only", "CoT_findings_image_plus_text"),
        "direct_style_inference": block(eval_set, {"a": scores["DA"], "b": scores["DA_findings"]},
                                         infer_da, "DA_image_only", "DA_findings_image_plus_text"),
        "cot_style_inference": block(eval_set, {"a": scores["CoT"], "b": scores["CoT_findings"]},
                                      infer_cot, "CoT_image_only", "CoT_findings_image_plus_text"),
    }

    out_path = os.path.join(args.out_dir, "summary.json")
    json.dump(summary, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n写入: {out_path}")


if __name__ == "__main__":
    main()
