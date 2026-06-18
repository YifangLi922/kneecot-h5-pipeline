"""
evaluate.py  –  Step 3: run VLM inference via Ollama.
Usage:
    python evaluate.py

This used to also score results inline with sklearn (accuracy/f1/precision/
recall) while the LLM line scored with a completely different metric set
(accuracy + McNemar) and the LLM line's own yes/no parser. That made the two
lines fundamentally non-comparable. Scoring has been removed from here on
purpose and moved to one shared place:

    code_new/analysis/compare.py   -- yes/no accuracy + McNemar
    judge.py                       -- inference verdicts (LLM-as-judge)

This script's only job is to produce raw per-question records. Two field
gaps relative to the LLM line have also been fixed here:
  - every saved record now carries "prompt_key" (the LLM line already wrote
    this on every record; this VLM line only encoded it in the output
    filename before, so a record on its own had no way to say which
    condition it came from).
  - every saved record now carries "model" (needed once minicpm-v is
    enabled, so results from two VLMs in the same results dir can be told
    apart record-by-record, not just by filename).
"""
import os, json, base64, glob
from config import RESULTS_DIR, MODELS, SLICE_DIRS
from prompts import VLM_PROMPTS


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


def _is_complete(existing: list, expected_n: int) -> bool:
    """Return True only when the cached file has >=1 record with a raw_output
    AND covers at least 90 % of the expected case count.
    This prevents a stale all-null file from permanently blocking a rerun."""
    if not existing:
        return False
    has_valid = any(r.get("raw_output") for r in existing)
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
            print(f"  [SKIP] {model_name} / {prompt_key} yn — already done "
                  f"({len(existing)} cases)")
            return existing
        else:
            n_valid = sum(1 for r in existing if r.get("raw_output"))
            print(f"  [REDO] {model_name} / {prompt_key} yn — stale file "
                  f"({len(existing)} records, only {n_valid} have raw output). Rerunning.")

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
            raw = resp["message"]["content"]
            print(f"  {i:4}/{len(yn_cases)}  {case['case_id']:<16} generated ({len(raw)} chars)")
            results.append({**case, "model": model_name, "prompt_key": prompt_key, "raw_output": raw})
        except Exception as e:
            print(f"  {i:4}/{len(yn_cases)} ERROR: {e}")
            results.append({**case, "model": model_name, "prompt_key": prompt_key,
                            "raw_output": None, "error": str(e)})

    n_errors = sum(1 for r in results if r.get("error"))
    print(f"  Generated {len(results) - n_errors}/{len(yn_cases)}  Errors={n_errors}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {out_path}")
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
            print(f"  {i:4}/{len(inf_cases)} ✓  {case['case_id']} processed.")
            results.append({**case, "model": model_name, "prompt_key": prompt_key, "raw_output": raw})
        except Exception as e:
            print(f"  {i:4}/{len(inf_cases)} ERROR: {e}")
            results.append({**case, "model": model_name, "prompt_key": prompt_key,
                            "raw_output": None, "error": str(e)})
            errors += 1

        if i % 10 == 0:
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Completed {len(results)}/{len(inf_cases)} inference cases. Errors: {errors}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {out_path}")

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

    print("\nGeneration complete. This script does not score results anymore -- "
          "run code_new/analysis/compare.py (yes/no) and judge.py (inference) next.")
