"""
run.py  –  Master runner: executes the full VLM pipeline in order.

Steps:
    1. preprocessing.py   – extract PNG slices from .nii files
    2. build_eval_set.py  – build the balanced evaluation JSON
    3. evaluate.py        – run VLM inference with Ollama (raw outputs only)

Scoring is not part of this runner — run these next:
    code_new/analysis/compare.py   – yes/no accuracy + McNemar
    judge.py                       – inference verdicts (LLM-as-judge)

Usage:
    python run.py                         # all data, all steps
    python run.py --n-eval 50             # sample 50 yes/no + 50 inference
    python run.py --skip-preprocess       # skip slice extraction
    python run.py --skip-build            # skip eval set building
    python run.py --eval-only             # run only step 3
    python run.py --skip-preprocess --n-eval 100
"""
import argparse
import subprocess
import sys
import os

SCRIPTS = {
    "preprocess": "preprocessing.py",
    "build":      "build_eval_set.py",
    "evaluate":   "evaluate.py",
    "metrics":    "metrics.py",
}


def run_step(name: str, script: str, extra_args: list = None) -> bool:
    script_path = os.path.join(os.path.dirname(__file__), script)
    cmd = [sys.executable, script_path] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  STEP: {name.upper()}  ({script})")
    if extra_args:
        print(f"  ARGS: {extra_args}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{name}' failed with code {result.returncode}.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="VLM Pipeline Runner")
    parser.add_argument("--n-eval", type=int, default=None,
                        help="Sample N cases per type (yes/no and inference). "
                             "Omit to run on ALL data.")
    parser.add_argument("--eval-set", type=str, default=None,
                        help="Path to existing shared eval_set.json (skips step 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip slice extraction (step 1)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip eval set building (step 2)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run only evaluate + metrics (steps 3–4)")
    args = parser.parse_args()

    # Build extra args to pass to build_eval_set.py
    build_args = []
    if args.n_eval:
        build_args += ["--n-eval", str(args.n_eval)]
    if args.seed != 42:
        build_args += ["--seed", str(args.seed)]

    steps = []
    if not args.eval_only and not args.skip_preprocess:
        steps.append(("preprocess", SCRIPTS["preprocess"], []))
    if not args.eval_only and not args.skip_build:
        if args.eval_set:
            build_args += ["--eval-set", args.eval_set]
        steps.append(("build", SCRIPTS["build"], build_args))
    steps.append(("evaluate", SCRIPTS["evaluate"], []))
    # metrics.py is stale: it imports calculate_metrics from evaluate.py, which
    # was removed when scoring moved to code_new/analysis/compare.py + judge.py
    # (see evaluate.py's module docstring). Run those two scripts instead.

    if args.n_eval:
        print(f"\nMode: SAMPLED — {args.n_eval} yes/no + {args.n_eval} inference cases")
    else:
        print(f"\nMode: FULL — all available data")

    print(f"Running {len(steps)} step(s): {[s[0] for s in steps]}")

    for name, script, extra in steps:
        ok = run_step(name, script, extra)
        if not ok:
            print(f"\nPipeline aborted at step: {name}")
            sys.exit(1)

    print("\n" + "="*60)
    print("  ALL STEPS COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()
