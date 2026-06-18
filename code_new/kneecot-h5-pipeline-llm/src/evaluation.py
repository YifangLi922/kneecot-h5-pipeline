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
    text = raw_output
    if ANSWER_MARKER in text:
        text = text.split(ANSWER_MARKER)[-1].strip()

    m = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()

    last_yes = text.rfind("是")
    last_no = text.rfind("否")
    if last_yes == -1 and last_no == -1:
        return None
    if last_yes > last_no:
        return "Yes"
    else:
        return "No"


def yes_no_accuracy(results):
    """Accuracy on yesno questions, broken down by prompt_key."""
    by_mode = defaultdict(lambda: {"correct": 0, "total": 0, "unparsed": 0})
    for r in results:
        if r["qtype"] != "yesno":
            continue
        s = by_mode[r["prompt_key"]]
        s["total"] += 1
        pred = parse_yes_no(r["raw_output"])
        if pred is None:
            s["unparsed"] += 1
            continue
        if pred == r.get("ground_truth"):
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
    """McNemar's test on paired yesno items (same case+question, both modes)."""
    keyed = defaultdict(dict)
    for r in results:
        if r["qtype"] != "yesno":
            continue
        pred = parse_yes_no(r["raw_output"])
        correct = (pred == r.get("ground_truth")) if pred is not None else None
        keyed[(r["case_id"], r["question"])][r["prompt_key"]] = correct

    b = c = n = 0  # b: DA correct & CoT wrong; c: DA wrong & CoT correct
    for modes in keyed.values():
        d, t = modes.get("DA"), modes.get("CoT")
        if d is None or t is None:
            continue
        n += 1
        if d and not t:
            b += 1
        elif not d and t:
            c += 1

    out = {"n_pairs": n, "DA_only_correct": b, "CoT_only_correct": c}

    if b == 0 and c == 0:
        out["statistic"] = 0.0
        out["p_value"] = 1.0
    else:
        try:
            from statsmodels.stats.contingency_tables import mcnemar
            res = mcnemar([[0, b], [c, 0]], exact=(b + c < 25))
            out["statistic"] = round(float(res.statistic), 4)
            out["p_value"] = round(float(res.pvalue), 4)
        except Exception as e:
            out["note"] = f"install statsmodels for the p-value ({e})"
    return out


def collect_inference_outputs(results):
    """Gather inference-type outputs (model answer + GT) for later judging."""
    return [
        {
            "case_id": r["case_id"],
            "question": r["question"],
            "prompt_key": r["prompt_key"],
            "ground_truth": r.get("ground_truth", ""),
            "model_output": r["raw_output"],
        }
        for r in results
        if r["qtype"] == "inference"
    ]


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── Inference question scoring ────────────────────────────────────────────────

def extract_inference_answer(raw_output):
    """Extract the text after the answer marker as the final conclusion."""
    if ANSWER_MARKER in raw_output:
        return raw_output.split(ANSWER_MARKER)[-1].strip()
    return raw_output.strip()


def _chinese_bigrams(text):
    """Extract all 2-character Chinese bigrams from a string."""
    bigrams = set()
    for i in range(len(text) - 1):
        pair = text[i:i+2]
        if re.match(r"[\u4e00-\u9fff]{2}", pair):
            bigrams.add(pair)
    return bigrams


def score_inference_one(pred_answer, gt_answer):
    """Score one inference answer using Chinese bigram overlap (>= 30%)."""
    if not gt_answer or not gt_answer.strip():
        return None
    gt_bigrams = _chinese_bigrams(gt_answer)
    if not gt_bigrams:
        return None
    hits = sum(1 for b in gt_bigrams if b in pred_answer)
    ratio = hits / len(gt_bigrams)
    return ratio >= 0.30


def inference_accuracy(results):
    """Compute inference question accuracy per prompt_key."""
    by_mode = defaultdict(lambda: {"correct": 0, "total": 0, "unscored": 0})
    for r in results:
        if r["qtype"] != "inference":
            continue
        s = by_mode[r["prompt_key"]]
        s["total"] += 1
        pred_ans = extract_inference_answer(r["raw_output"])
        is_correct = score_inference_one(pred_ans, r.get("ground_truth", ""))
        if is_correct is None:
            s["unscored"] += 1
        elif is_correct:
            s["correct"] += 1

    summary = {}
    for mode, s in by_mode.items():
        scored = s["total"] - s["unscored"]
        summary[mode] = {
            "accuracy": round(s["correct"] / scored, 4) if scored else 0.0,
            "correct": s["correct"],
            "scored": scored,
            "unscored": s["unscored"],
            "total": s["total"],
        }
    return summary
