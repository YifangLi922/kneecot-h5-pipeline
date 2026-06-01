"""Evaluation for the H5 text-only LLM line.

- yes_no questions: parse the final Yes/No, compute accuracy per prompt mode,
  and run McNemar's test comparing direct vs CoT.
- inference questions: NOT auto-scored here. Their ground truth is a free-text
  "conclusion + reasoning", so they are collected for later scoring
  (LLM-as-judge / manual review in Round 3).
"""
import json
import re
from collections import defaultdict

ANSWER_MARKER = "【答案】"


def parse_yes_no(raw_output):
    """Extract a Yes/No label from a model output, or None if not parseable."""
    text = raw_output
    if ANSWER_MARKER in text:
        text = text.split(ANSWER_MARKER, 1)[1]
    m = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    # fallback: Chinese 是 / 否 right after the marker
    head = text.strip()[:8]
    if "是" in head:
        return "Yes"
    if "否" in head:
        return "No"
    return None


def yes_no_accuracy(results):
    """Accuracy on yes_no questions, broken down by prompt mode."""
    by_mode = defaultdict(lambda: {"correct": 0, "total": 0, "unparsed": 0})
    for r in results:
        if r["qtype"] != "yes_no":
            continue
        s = by_mode[r["prompt_mode"]]
        s["total"] += 1
        pred = parse_yes_no(r["raw_output"])
        if pred is None:
            s["unparsed"] += 1
            continue
        if pred == r.get("gt_label"):
            s["correct"] += 1
    summary = {}
    for mode, s in by_mode.items():
        scored = s["total"] - s["unparsed"]
        summary[mode] = {
            "accuracy": round(s["correct"] / scored, 4) if scored else 0.0,
            "correct": s["correct"],
            "scored": scored,
            "unparsed": s["unparsed"],
            "total": s["total"],
        }
    return summary


def mcnemar_direct_vs_cot(results):
    """McNemar's test on paired yes_no items (same case+question, both modes)."""
    keyed = defaultdict(dict)
    for r in results:
        if r["qtype"] != "yes_no":
            continue
        pred = parse_yes_no(r["raw_output"])
        correct = (pred == r.get("gt_label")) if pred is not None else None
        keyed[(r["case_id"], r["question"])][r["prompt_mode"]] = correct

    b = c = n = 0  # b: direct correct & cot wrong; c: direct wrong & cot correct
    for modes in keyed.values():
        d, t = modes.get("direct"), modes.get("cot")
        if d is None or t is None:
            continue
        n += 1
        if d and not t:
            b += 1
        elif not d and t:
            c += 1

    out = {"n_pairs": n, "direct_only_correct": b, "cot_only_correct": c}
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        res = mcnemar([[0, b], [c, 0]], exact=(b + c < 25))
        out["statistic"] = round(float(res.statistic), 4)
        out["p_value"] = round(float(res.pvalue), 4)
    except Exception as e:  # statsmodels not installed
        out["note"] = f"install statsmodels for the p-value ({e})"
    return out


def collect_inference_outputs(results):
    """Gather inference-type outputs (model answer + GT) for later judging."""
    return [
        {
            "case_id": r["case_id"],
            "question": r["question"],
            "prompt_mode": r["prompt_mode"],
            "gt_answer": r.get("gt_answer", ""),
            "model_output": r["raw_output"],
        }
        for r in results
        if r["qtype"] == "inference"
    ]


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
