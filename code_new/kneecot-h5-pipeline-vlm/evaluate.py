"""
evaluate.py  –  Step 3: run VLM inference via Ollama.
Usage:
    python evaluate.py
"""
import os, json, base64, time, glob
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from config import RESULTS_DIR, MODELS, SLICE_DIRS
from prompts import VLM_PROMPTS, parse_yes_no


def load_eval_set(eval_path):
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_images(nii_path: str, slice_dirs: dict) -> list:
    """
    Load the pre-stitched grid PNG for a given nii_path and encode as base64.

    preprocessing.py saves exactly one file per case:
        <slice_dir>/<case_id>_grid.png

    We send that single grid image to the VLM, which already contains
    10 sagittal slices arranged in a 2×5 mosaic.
    """
    case_id = os.path.splitext(os.path.basename(nii_path))[0]
    split   = "train" if "train" in nii_path.lower() else "test"
    slc_dir = slice_dirs.get(split, "")

    grid_path = os.path.join(slc_dir, f"{case_id}_grid.png")

    # Graceful fallback: if grid not found, try legacy slice patterns
    if not os.path.exists(grid_path):
        for pat in [
            os.path.join(slc_dir, f"{case_id}_slice*.png"),
            os.path.join(slc_dir, f"{case_id}*.png"),
        ]:
            paths = sorted(glob.glob(pat))
            if paths:
                images = []
                for p in paths:
                    with open(p, "rb") as f:
                        images.append(base64.b64encode(f.read()).decode("utf-8"))
                return images
        return []  # nothing found

    with open(grid_path, "rb") as f:
        return [base64.b64encode(f.read()).decode("utf-8")]


def calculate_metrics(results: list) -> dict:
    evaluated = [r for r in results if r.get("prediction") in ("Yes", "No")]
    unclear   = len(results) - len(evaluated)
    if not evaluated:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "correct": 0, "total": 0, "unclear": unclear,
                "unclear_pct": round(unclear / len(results) * 100, 1) if results else 0}
    y_true = [r["ground_truth"] for r in evaluated]
    y_pred = [r["prediction"]   for r in evaluated]
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return {
        "accuracy":    round(accuracy_score(y_true, y_pred), 3),
        "f1":          round(f1_score(y_true, y_pred, pos_label="Yes", zero_division=0), 3),
        "precision":   round(precision_score(y_true, y_pred, pos_label="Yes", zero_division=0), 3),
        "recall":      round(recall_score(y_true, y_pred, pos_label="Yes", zero_division=0), 3),
        "correct":     correct,
        "total":       len(evaluated),
        "unclear":     unclear,
        "unclear_pct": round(unclear / len(results) * 100, 1) if results else 0,
    }


def _is_complete(existing: list, expected_n: int) -> bool:
    """Return True only when the cached file has >=1 valid prediction
    AND covers at least 90 % of the expected case count.
    This prevents a stale all-null file from permanently blocking a rerun."""
    if not existing:
        return False
    has_valid = any(r.get("prediction") in ("Yes", "No") for r in existing)
    covers_enough = len(existing) >= max(1, int(expected_n * 0.9))
    return has_valid and covers_enough


def run_yn_eval(model_name: str, prompt_key: str, cases: list,
                results_dir: str, slice_dirs: dict, skip_existing: bool = True):
    import ollama
    out_path = os.path.join(results_dir, f"{model_name}_{prompt_key}_yn.json")
    yn_cases = [c for c in cases if c["qtype"] == "yesno"]

    if skip_existing and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if _is_complete(existing, len(yn_cases)):
            m = calculate_metrics(existing)
            print(f"  [SKIP] {model_name} / {prompt_key} yn — already done "
                  f"({len(existing)} cases, Acc={m['accuracy']:.3f})")
            return existing
        else:
            n_valid = sum(1 for r in existing if r.get("prediction") in ("Yes", "No"))
            print(f"  [REDO] {model_name} / {prompt_key} yn — stale file "
                  f"({len(existing)} records, only {n_valid} valid predictions). Rerunning.")

    print(f"{'='*55}")
    print(f"  Model: {model_name}  Prompt: {prompt_key}")
    print(f"{'='*55}")

    results = []
    for i, case in enumerate(yn_cases, 1):
        prompt_text = VLM_PROMPTS[prompt_key].format(
            question=case["question"],
            findings=case.get("findings", ""),
        )
        images = encode_images(case["nii_path"], slice_dirs)
        try:
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text, "images": images}],
            )
            raw     = resp["message"]["content"]
            pred    = parse_yes_no(raw)
            correct = pred == case["ground_truth"]
            print(f"  {i:4}/{len(yn_cases)} {'\u2713' if correct else '\u2717'}  "
                  f"{case['case_id']:<16} GT={case['ground_truth']}  Pred={pred or 'UNCLEAR'}")
            results.append({**case, "raw_output": raw, "prediction": pred})
        except Exception as e:
            print(f"  {i:4}/{len(yn_cases)} ERROR: {e}")
            results.append({**case, "raw_output": None, "prediction": None, "error": str(e)})

    m = calculate_metrics(results)
    print(f"  Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  "
          f"UNCLEAR={m['unclear']}/{len(yn_cases)}  Errors={sum(1 for r in results if r.get('error'))}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved \u2192 {out_path}")
    return results


def run_inference_eval(model_name: str, prompt_key: str, cases: list,
                       results_dir: str, slice_dirs: dict, skip_existing: bool = True):
    import ollama
    out_path = os.path.join(results_dir, f"{model_name}_{prompt_key}_inference.json")
    inf_cases = [c for c in cases if c["qtype"] == "inference"]

    if skip_existing and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if _is_complete(existing, len(inf_cases)):
            print(f"  [SKIP] {model_name} / {prompt_key} inference — already done "
                  f"({len(existing)} cases)")
            return existing
        else:
            n_valid = sum(1 for r in existing if r.get("raw_output"))
            print(f"  [REDO] {model_name} / {prompt_key} inference — stale file "
                  f"({len(existing)} records, only {n_valid} have raw output). Rerunning.")

    done_ids = set()
    results  = []
    partial_path = out_path + ".partial"
    if os.path.exists(partial_path):
        with open(partial_path, "r", encoding="utf-8") as f:
            results  = json.load(f)
        done_ids = {(r["case_id"], r["question"]) for r in results}
        print(f"  [RESUME] Found {len(results)} already processed inference cases")

    remaining = [c for c in inf_cases
                 if (c["case_id"], c["question"]) not in done_ids]

    print(f"{'='*55}")
    print(f"  Model: {model_name}  Prompt: {prompt_key}  Type: Inference")
    print(f"{'='*55}")

    errors = 0
    for i, case in enumerate(remaining, len(results) + 1):
        prompt_text = VLM_PROMPTS[prompt_key].format(
            question=case["question"],
            findings=case.get("findings", ""),
        )
        images = encode_images(case["nii_path"], slice_dirs)
        try:
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text, "images": images}],
            )
            raw = resp["message"]["content"]
            print(f"  {i:4}/{len(inf_cases)} \u2713  {case['case_id']} processed.")
            results.append({**case, "raw_output": raw, "prediction": raw})
        except Exception as e:
            print(f"  {i:4}/{len(inf_cases)} ERROR: {e}")
            results.append({**case, "raw_output": None, "prediction": None, "error": str(e)})
            errors += 1

        if i % 10 == 0:
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Completed {len(results)}/{len(inf_cases)} inference cases. Errors: {errors}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved \u2192 {out_path}")

    if os.path.exists(partial_path):
        os.remove(partial_path)

    return results


if __name__ == "__main__":
    from config import EVAL_PATH, RESULTS_DIR, MODELS, SLICE_DIRS, CONDITIONS

    cases = load_eval_set(str(EVAL_PATH))
    os.makedirs(str(RESULTS_DIR), exist_ok=True)

    active_models = [m for m, cfg in MODELS.items() if cfg.get("enabled")]

    for model_name in active_models:
        for prompt_key in CONDITIONS:
            run_yn_eval(model_name, prompt_key, cases,
                        str(RESULTS_DIR), SLICE_DIRS)
            run_inference_eval(model_name, prompt_key, cases,
                               str(RESULTS_DIR), SLICE_DIRS)

    print("\nEvaluation complete.")
