#!/usr/bin/env python3
"""run_pipeline.py - Run the entire H5 KneeCoT project in one shot."""

import argparse, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
   for c in [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]:
    llm = os.path.join(c, "code_new", "kneecot-h5-pipeline-llm")
    vlm = os.path.join(c, "code_new", "kneecot-h5-pipeline-vlm")
    ana = os.path.join(c, "code_new", "analysis")
    if os.path.isdir(llm) and os.path.isdir(vlm):
        ROOT, LLM_DIR, VLM_DIR, ANA_DIR = c, llm, vlm, ana
        break
else:
    print("Cannot find project structure. Run this script from the project root.")
    sys.exit(1)


def banner(msg):
    print("\n" + "#" * 60)
    print(f"  Step: {msg}")
    print("#" * 60)


def sh(cmd, cwd=None):
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd or ROOT, capture_output=True, text=True)
    for line in (r.stdout or "").strip().splitlines():
        if line.strip():
            print(f"    {line}")
    if r.returncode != 0:
        print(f"  [FAILED] exit={r.returncode}")
        if r.stderr:
            print(r.stderr[-300:])
        sys.exit(r.returncode)
    return r


def main():
    ap = argparse.ArgumentParser(description="KneeCoT H5: one script to run them all")
    ap.add_argument("--data_dir", help="Path to KneeCoT JSON cases (required for real run)")
    ap.add_argument("--mock", action="store_true", help="Mock mode: no GPU/data needed")
    ap.add_argument("--n_yesno", type=int, default=50)
    ap.add_argument("--n_inference", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vlm_results", help="Path to existing VLM results JSON")
    ap.add_argument("--out", default="pipeline_output")
    args = ap.parse_args()

    if not args.mock and not args.data_dir:
        print("Need --data_dir for real run, or --mock for verification")
        sys.exit(1)

    OUT = os.path.join(ROOT, args.out)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    # Step 1
    banner("1/4: Build shared eval set")
    bes = os.path.join(ANA_DIR, "build_eval_set.py")
    if not os.path.isfile(bes):
        bes = os.path.join(LLM_DIR, "src", "build_eval_set.py")
    dat = args.data_dir or os.path.join(LLM_DIR, "data", "sample")
    nyn = 1 if args.mock else args.n_yesno
    nif = 1 if args.mock else args.n_inference
    eo = os.path.join(OUT, "eval_set.json")
    sh(f'python "{bes}" --data_dir "{dat}" --n_yesno {nyn} --n_inference {nif} --seed {args.seed} --out "{eo}" --flat')

    # Step 2
    banner("2/4: Run LLM pipeline")
    lo = os.path.join(OUT, "llm_results")
    cmd = f'python "{LLM_DIR}/run.py" --data_dir "{dat}" --eval_set "{eo}" --out_dir "{lo}"'
    if args.mock:
        cmd += " --mock --sample_size 0"
    sh(cmd, cwd=LLM_DIR)

    # Step 3
    banner("3/4: Prepare VLM results")
    vp = args.vlm_results
    if vp and os.path.isfile(vp):
        print(f"  Using provided: {vp}")
    elif args.mock:
        vp = os.path.join(OUT, "vlm_mock.json")
        if not os.path.exists(vp):
            with open(eo, "r", encoding="utf-8") as f:
                raw = json.load(f)
            items = raw if isinstance(raw, list) else raw.get("items", [])
            mock = []
            for it in items:
                mock.append({
                    "question_id": it.get("question_id", ""),
                    "case_id": it.get("case_id", ""),
                    "question": it.get("question", ""),
                    "raw_response": "(mock VLM) simulated answer",
                    "prompt_type": "da",
                    "model": "qwen2.5vl",
                })
            json.dump(mock, open(vp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"  Generated mock: {vp}")
    else:
        vg = os.path.join(VLM_DIR, "vlm_results", "qwen2.5vl_DA_findings_yn.json")
        if os.path.isfile(vg):
            vp = vg
            print(f"  Found existing: {vp}")
        else:
            print("  No VLM results found. Run VLM.ipynb in Colab first, then use --vlm_results.")
            print(f"  Notebook: {VLM_DIR}/VLM.ipynb")
            vp = os.path.join(OUT, "vlm_empty.json")
            if not os.path.exists(vp):
                json.dump([], open(vp, "w", encoding="utf-8"))

    # Step 4
    banner("4/4: Compare analysis")
    cp = os.path.join(ANA_DIR, "compare.py")
    co = os.path.join(OUT, "compare_out")
    if os.path.isfile(cp):
        sh(f'python "{cp}" --eval_set "{eo}" --llm_results "{lo}/raw_results.json" --vlm_results "{vp}" --model qwen2.5vl --out_dir "{co}" --missing_policy wrong')
    else:
        print(f"  compare.py not found at {cp}, skipping")

    # Report
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {OUT}")
    for f in sorted(glob.glob(os.path.join(OUT, "**", "*"), recursive=True)):
        if os.path.isfile(f):
            print(f"  {os.path.relpath(f, OUT):40s} {os.path.getsize(f):>8} bytes")


if __name__ == "__main__":
    main()

