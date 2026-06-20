"""Data loading and preprocessing for the H5 text-only LLM line.

This module provides lightweight helpers to load case JSON files and
describe an already-built eval set. The eval set itself is built by
build_eval_set.py (which is the single source of truth for the schema).
"""
import glob
import json
import os
import random

FINDINGS_FIELD = "MR表现"
METHOD_FIELD = "检查方法"
ID_FIELD = "顺序编号"
QA_FIELD = "问答数据"
H5_QUESTION_TYPES = ("yesno", "inference")


def load_case(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_knee_only_case(case):
    """Consistent with VLM filter: must contain 膝关节, must not contain 肩关节."""
    method = case.get(METHOD_FIELD, "")
    return ("膝关节" in method) and ("肩关节" not in method)


def extract_yes_no_gt(answer):
    a = answer.strip()
    if a.startswith("Yes"):
        return "Yes"
    if a.startswith("No"):
        return "No"
    return None


def build_eval_items(case):
    """Parse one case JSON into eval item dicts using the shared schema."""
    findings = case.get(FINDINGS_FIELD, "").strip()
    case_id = case.get(ID_FIELD, "")
    qa_pairs = case.get(QA_FIELD, {}).get("qa_pairs", [])
    items = []
    for qa in qa_pairs:
        raw_qtype = qa.get("type")
        # Normalise yes_no -> yesno to match shared schema
        qtype = "yesno" if raw_qtype == "yes_no" else raw_qtype
        if qtype not in H5_QUESTION_TYPES:
            continue
        item = {
            "case_id": case_id,
            "findings": findings,
            "question": qa["question"],
            "qtype": qtype,
            "full_answer": qa["answer"].strip(),
        }
        if qtype == "yesno":
            item["ground_truth"] = extract_yes_no_gt(qa["answer"])
        else:
            item["ground_truth"] = qa["answer"].strip()
        items.append(item)
    return items


def build_eval_set(data_dir, sample_size=None, seed=42,
                   pattern="*.json", output_path=None):
    """Build the full evaluation set with stratified yes/no sampling.

    Prefer using build_eval_set.py directly for the canonical shared eval set.
    This function is kept for backwards compatibility with run.py --data_dir.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    cases = [load_case(p) for p in paths]
    knee_cases = [c for c in cases if is_knee_only_case(c)]

    rng = random.Random(seed)
    items = []

    if sample_size is not None and sample_size <= 200:
        yes_pool = {}
        no_pool = {}
        inference_items = []
        for c in knee_cases:
            for it in build_eval_items(c):
                if it["qtype"] == "yesno":
                    label = it["ground_truth"]
                    if label == "Yes" and it["case_id"] not in yes_pool:
                        yes_pool[it["case_id"]] = it
                    elif label == "No" and it["case_id"] not in no_pool:
                        no_pool[it["case_id"]] = it
                else:
                    inference_items.append(it)

        yes_list = list(yes_pool.values())
        no_list = list(no_pool.values())

        n_yes_no = sample_size if sample_size % 2 == 0 else sample_size - 1
        half = n_yes_no // 2
        sampled_yes = rng.sample(yes_list, min(half, len(yes_list)))
        sampled_no = rng.sample(no_list, min(half, len(no_list)))
        items = sampled_yes + sampled_no
        rng.shuffle(items)
        items.extend(inference_items)

    else:
        if sample_size is not None and sample_size > 200:
            knee_cases = rng.sample(knee_cases, sample_size)
        knee_cases.sort(key=lambda c: c.get(ID_FIELD, ""))
        for c in knee_cases:
            items.extend(build_eval_items(c))

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print("Eval set saved to " + output_path + "  (" + str(len(items)) + " items)")

    return items


def describe_eval_set(items):
    n_cases = len({it["case_id"] for it in items})
    n_yes_no = sum(1 for it in items if it["qtype"] == "yesno")
    n_yes = sum(1 for it in items if it["qtype"] == "yesno" and it.get("ground_truth") == "Yes")
    n_no = sum(1 for it in items if it["qtype"] == "yesno" and it.get("ground_truth") == "No")
    n_inference = sum(1 for it in items if it["qtype"] == "inference")
    return {"cases": n_cases, "questions": len(items), "yes_no": n_yes_no, "yes": n_yes, "no": n_no, "inference": n_inference}
