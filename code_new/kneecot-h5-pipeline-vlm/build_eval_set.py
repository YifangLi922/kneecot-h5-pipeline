"""
build_eval_set.py  –  Step 2: build the balanced evaluation JSON.
Usage:
    python build_eval_set.py              # all data
    python build_eval_set.py --n-eval 50  # 50 yes/no + 50 inference
"""
import os, json, glob, random, argparse
from config import TRAIN_ANN, TEST_ANN, TRAIN_NII, TEST_NII, EVAL_PATH

YESNO_MARKERS  = ["是否", "有无", "能否", "可否", "是不是"]
EXCLUDE_JOINTS = ["髋", "肩", "肘", "踝", "腕", "脊"]

ANN_DIRS = {"test": str(TEST_ANN), "train": str(TRAIN_ANN)}
NII_DIRS = {"test": str(TEST_NII), "train": str(TRAIN_NII)}


def _clean(name: str) -> str:
    return str(name).strip().replace(".json", "")


def _find_nii(case_name, case_id, nii_dir):
    for candidate in dict.fromkeys([case_name, case_id, f"{case_name}_01", f"{case_id}_01"]):
        p = os.path.join(nii_dir, f"{candidate}.nii")
        if os.path.exists(p):
            return p
    for pat in [os.path.join(nii_dir, f"{case_name}*.nii"),
                os.path.join(nii_dir, f"{case_id}*.nii")]:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def _extract_qa_pairs(data: dict) -> list:
    candidates = [
        data.get("qa_pairs"),
        data.get("QA_pairs"),
        data.get("questions"),
        data.get("问答数据"),
        data.get("annotations", {}).get("qa_pairs") if isinstance(data.get("annotations"), dict) else None,
        data.get("data", {}).get("qa_pairs") if isinstance(data.get("data"), dict) else None,
    ]
    for c in candidates:
        if isinstance(c, list) and len(c) > 0:
            return c
    for key, val in data.items():
        if isinstance(val, dict) and "qa_pairs" in val:
            return val["qa_pairs"]
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            if "question" in val[0] or "answer" in val[0] or "Q" in val[0]:
                return val
    return []


def _extract_findings(data: dict) -> str:
    for key in ["MR_findings", "findings", "MR表现", "mr_findings", "impression"]:
        if key in data:
            return data[key]
    return ""


def _extract_case_id(data: dict, fallback: str) -> str:
    for key in ["case_id", "id", "case_name", "caseID"]:
        if key in data:
            return _clean(data[key])
    return fallback


def _get_answer(qa: dict) -> str:
    return qa.get("answer", qa.get("ground_truth", qa.get("A", ""))).strip()


def _get_question(qa: dict) -> str:
    return qa.get("question", qa.get("Q", ""))


def _is_yesno(qa: dict) -> bool:
    qtype    = qa.get("type", qa.get("question_type", "")).lower()
    answer   = _get_answer(qa)
    question = _get_question(qa)
    type_ok   = "yesno" in qtype or "yes_no" in qtype or "yes/no" in qtype
    answer_ok = (answer.startswith("Yes") or answer.startswith("No") or
                 answer.startswith("是") or answer.startswith("否"))
    if not (type_ok or answer_ok):
        return False
    if any(j in question for j in EXCLUDE_JOINTS):
        return False
    return True


def _is_inference(qa: dict) -> bool:
    qtype = qa.get("type", qa.get("question_type", "")).lower()
    if "inference" not in qtype and "open" not in qtype and "reason" not in qtype:
        return False
    if any(j in _get_question(qa) for j in EXCLUDE_JOINTS):
        return False
    return True


def build_eval_set(ann_dirs, nii_dirs, eval_out_path, n_eval=None, seed=42):
    all_cases, miss_nii = [], {"train": 0, "test": 0}
    first_file_printed = False

    for split, ann_dir in ann_dirs.items():
        if not os.path.exists(ann_dir):
            print(f"  [WARN] Missing annotation dir: {ann_dir}")
            continue
        nii_dir    = nii_dirs[split]
        json_files = sorted(glob.glob(os.path.join(ann_dir, "**", "*.json"), recursive=True))
        print(f"{split}: found {len(json_files)} annotation json files")

        for jp in json_files:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not first_file_printed:
                print(f"  [DEBUG] First file: {os.path.basename(jp)}")
                print(f"  [DEBUG] Top-level keys: {list(data.keys())}")
                qa_sample = _extract_qa_pairs(data)
                if qa_sample:
                    print(f"  [DEBUG] QA pair count: {len(qa_sample)}")
                    print(f"  [DEBUG] First QA keys: {list(qa_sample[0].keys())}")
                    print(f"  [DEBUG] First QA sample: {qa_sample[0]}")
                else:
                    print(f"  [DEBUG] No QA pairs found. Full data preview:")
                    print(f"  {json.dumps(data, ensure_ascii=False)[:600]}")
                first_file_printed = True

            case_name = _clean(os.path.splitext(os.path.basename(jp))[0])
            case_id   = _extract_case_id(data, case_name)
            findings  = _extract_findings(data)
            qa_pairs  = _extract_qa_pairs(data)

            nii_path = _find_nii(case_name, case_id, nii_dir)
            if nii_path is None:
                miss_nii[split] += 1
                continue

            # Knee-only filter: check jian-cha-fang-fa field (consistent with LLM pipeline)
            method = data.get("检查方法", "")
            if "膝关节" not in method or "肩关节" in method:
                continue

            for qa in qa_pairs:
                answer   = _get_answer(qa)
                question = _get_question(qa)
                if _is_yesno(qa):
                    all_cases.append({
                        "case_id": case_id, "case_name": case_name,
                        "split": split, "nii_path": nii_path,
                        "findings": findings, "question": question,
                        "ground_truth": "Yes" if (answer.startswith("Yes") or answer.startswith("是")) else "No",
                        "full_answer": answer, "qtype": "yesno",
                    })
                elif _is_inference(qa):
                    all_cases.append({
                        "case_id": case_id, "case_name": case_name,
                        "split": split, "nii_path": nii_path,
                        "findings": findings, "question": question,
                        "ground_truth": answer, "full_answer": answer, "qtype": "inference",
                    })

    yesno_pool = [c for c in all_cases if c["qtype"] == "yesno"]
    infer_pool = [c for c in all_cases if c["qtype"] == "inference"]
    print(f"Missing NII-match cases: train={miss_nii['train']} test={miss_nii['test']}")
    print(f"Total QA pairs loaded: {len(all_cases)}  YesNo={len(yesno_pool)}  Inference={len(infer_pool)}")

    if len(all_cases) == 0:
        print("\n[ERROR] 0 QA pairs loaded. Check the [DEBUG] output above.")
        raise ValueError("No valid cases loaded.")

    if n_eval is None:
        # Use ALL data — no sampling
        sampled = all_cases
        print(f"N_EVAL not set → using all {len(sampled)} cases")
    else:
        # Sample n_eval from yes/no AND n_eval from inference equally
        random.seed(seed)

        # Yes/No: balanced Yes + No, one per case_id
        yes_by, no_by = {}, {}
        for c in yesno_pool:
            if c["ground_truth"] == "Yes" and c["case_id"] not in yes_by:
                yes_by[c["case_id"]] = c
            elif c["ground_truth"] == "No" and c["case_id"] not in no_by:
                no_by[c["case_id"]] = c
        half = n_eval // 2
        sy = random.sample(list(yes_by.values()), min(half, len(yes_by)))
        sn = random.sample(list(no_by.values()),  min(half, len(no_by)))
        sampled_yn = sy + sn

        # Inference: also cap at n_eval
        sampled_inf = random.sample(infer_pool, min(n_eval, len(infer_pool)))

        sampled = sampled_yn + sampled_inf
        random.shuffle(sampled)
        print(f"N_EVAL={n_eval} → sampled {len(sampled_yn)} yes/no + {len(sampled_inf)} inference")

    yn_count  = sum(1 for c in sampled if c["qtype"] == "yesno")
    inf_count = sum(1 for c in sampled if c["qtype"] == "inference")
    print(f"Evaluation set size: {len(sampled)}  (YesNo={yn_count}  Inference={inf_count})")

    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"Evaluation set saved to {eval_out_path}")
    return sampled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VLM evaluation set")
    parser.add_argument(
        "--n-eval", type=int, default=None,
    help="Number of cases to sample per type (yes/no and inference). "
         "Omit to use ALL available data."
    )
    parser.add_argument("--eval-set", type=str, default=None,
        help="Path to existing shared eval_set.json (skip building)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.eval_set and os.path.exists(args.eval_set):
        print(f"Loading existing shared eval set from: {args.eval_set}")
        import shutil
        shutil.copy(args.eval_set, str(EVAL_PATH))
        print(f"Copied to {EVAL_PATH}")
    elif args.n_eval:
        print(f"N_EVAL={args.n_eval} (sampling {args.n_eval} inference)")
        build_eval_set(ANN_DIRS, NII_DIRS, str(EVAL_PATH), n_eval=args.n_eval, seed=args.seed)
    else:
        print("N_EVAL not provided using all data")
        build_eval_set(ANN_DIRS, NII_DIRS, str(EVAL_PATH), n_eval=args.n_eval, seed=args.seed)
