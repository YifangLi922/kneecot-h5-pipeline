"""Data loading and preprocessing for the H5 text-only LLM line.

Reads KneeCoT JSON annotation files and builds the evaluation set used by
inference.py.

Design decisions (paper Section 3.3.1 Input):
- The model is fed ONLY the free-text MR findings ("MR表现").
- The diagnostic impression ("诊断意见") and the structured labels ("标签")
  are deliberately EXCLUDED: they contain the answers to the VQA questions and
  would leak ground truth, turning a reasoning task into copying.
- For H5 we keep only yes_no and inference questions
  (descriptive/localization are discussed as limitations).
- Cases are filtered to knee-only studies for a clean knee MRI study.
"""
import glob
import json
import os
import random

FINDINGS_FIELD = "MR表现"
METHOD_FIELD = "检查方法"
ID_FIELD = "顺序编号"
QA_FIELD = "问答数据"

H5_QUESTION_TYPES = ("yes_no", "inference")


def load_case(path):
    """Load a single KneeCoT JSON annotation file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_knee_only_case(case):
    """Keep studies that include the knee and do NOT also include the shoulder.

    Some reports cover both knee and shoulder; mixing in shoulder findings would
    pollute the knee-only inputs, so we drop those for the main experiment.
    """
    method = case.get(METHOD_FIELD, "")
    return ("膝关节" in method) and ("肩关节" not in method)


def extract_yes_no_gt(answer):
    """Ground-truth label for a yes/no question = the leading Yes/No token.

    KneeCoT yes/no answers start with 'Yes。...' or 'No。...'.
    """
    a = answer.strip().lower()
    if a.startswith("yes"):
        return "Yes"
    if a.startswith("no"):
        return "No"
    return None


def build_eval_items(case):
    """Flatten one case into a list of per-question evaluation items."""
    findings = case.get(FINDINGS_FIELD, "").strip()
    case_id = case.get(ID_FIELD, "")
    qa_pairs = case.get(QA_FIELD, {}).get("qa_pairs", [])
    items = []
    for qa in qa_pairs:
        qtype = qa.get("type")
        if qtype not in H5_QUESTION_TYPES:
            continue
        item = {
            "case_id": case_id,
            "findings": findings,
            "question": qa["question"],
            "qtype": qtype,
            "gt_answer": qa["answer"].strip(),
        }
        if qtype == "yes_no":
            item["gt_label"] = extract_yes_no_gt(qa["answer"])
        items.append(item)
    return items


def build_eval_set(data_dir, sample_size=None, seed=42, pattern="*.json"):
    """Build the full evaluation set.

    Samples `sample_size` knee-only CASES (reproducibly, via `seed`) for the
    prototype stage, then flattens them into per-question items. Set
    sample_size=None to use every knee-only case (scale-up stage).
    """
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    cases = [load_case(p) for p in paths]
    knee_cases = [c for c in cases if is_knee_only_case(c)]

    if sample_size is not None and sample_size < len(knee_cases):
        rng = random.Random(seed)
        knee_cases = rng.sample(knee_cases, sample_size)
        # sort for a stable, reproducible ordering
        knee_cases.sort(key=lambda c: c.get(ID_FIELD, ""))

    items = []
    for c in knee_cases:
        items.extend(build_eval_items(c))
    return items


def describe_eval_set(items):
    """Return a small dict summarising the composition of the eval set."""
    n_cases = len({it["case_id"] for it in items})
    n_yes_no = sum(1 for it in items if it["qtype"] == "yes_no")
    n_inference = sum(1 for it in items if it["qtype"] == "inference")
    return {
        "cases": n_cases,
        "questions": len(items),
        "yes_no": n_yes_no,
        "inference": n_inference,
    }
