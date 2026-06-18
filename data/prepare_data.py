"""
prepare_data.py
===============
Place this script at:  ds2026-TBD-IDTBD-group4\\data\\prepare_data.py

What it does
------------
1. Downloads all .nii MRI files from HuggingFace dataset YiHui0124/KneeCoT
2. Downloads all .json annotation files from the same dataset
3. Flattens .nii files into  data/train/  and  data/test/
4. Flattens .json annotations into  data/annotations/train/  and  data/annotations/test/
5. Cleans up the temporary download cache

Usage
-----
    python prepare_data.py
    python prepare_data.py --nii-only        # skip annotations
    python prepare_data.py --annotations-only  # skip MRI files

Requirements
------------
    pip install huggingface_hub
"""

import os
import shutil
import argparse
from pathlib import Path
from huggingface_hub import list_repo_tree, hf_hub_download

# ─────────────────────────────────────────────
# CONFIGURATION  –  edit these values
# ─────────────────────────────────────────────

REPO_ID     = "YiHui0124/KneeCoT"
REPO_TYPE   = "dataset"

# Root of the project data folder (the folder containing this script)
DATA_ROOT   = Path(__file__).parent.resolve()   # ds2026-TBD-IDTBD-group4/data

# Top-level splits in the repo
SPLITS      = ["train_data", "test_data"]
# ─────────────────────────────────────────────

SPLIT_MAP_NII = {
    "train_data": DATA_ROOT / "train",
    "test_data":  DATA_ROOT / "test",
}

SPLIT_MAP_JSON = {
    "train_data": DATA_ROOT / "annotations" / "train",
    "test_data":  DATA_ROOT / "annotations" / "test",
}


def scan_repo_files(repo_id: str, repo_type: str, token: str,
                    extensions: tuple[str, ...]) -> list[tuple[str, int]]:
    """Scan the repo and return (repo_path, size) for files matching given extensions."""
    print(f"Scanning repository for {extensions} files …")
    files = []
    for item in list_repo_tree(repo_id, repo_type=repo_type, token=token, recursive=True):
        if hasattr(item, "size") and item.path.endswith(extensions):
            files.append((item.path, item.size))
    print(f"  Found {len(files)} file(s) matching {extensions}.")
    return files

def apply_max_gb(files: list, max_gb: float, train_ratio: float = 0.7) -> list:
    """
    Keep files up to max_gb total, respecting a train/test ratio.
    train_ratio=0.7 means 70% of budget goes to train_data, 30% to test_data.
    """
    train_budget = max_gb * train_ratio * 1e9
    test_budget  = max_gb * (1 - train_ratio) * 1e9

    train_files = [(p, s) for p, s in files if p.replace("\\", "/").startswith("train_data/")]
    test_files  = [(p, s) for p, s in files if p.replace("\\", "/").startswith("test_data/")]

    def take_until(file_list, budget):
        selected, cumulative = [], 0
        for path, size in file_list:
            if cumulative + size > budget:
                break
            selected.append((path, size))
            cumulative += size
        return selected, cumulative

    selected_train, train_used = take_until(train_files, train_budget)
    selected_test,  test_used  = take_until(test_files,  test_budget)

    total_used = (train_used + test_used) / 1e9
    print(f"  Train : {len(selected_train)} files  ({train_used / 1e9:.2f} GB / {max_gb * train_ratio:.2f} GB)")
    print(f"  Test  : {len(selected_test)}  files  ({test_used  / 1e9:.2f} GB / {max_gb * (1 - train_ratio):.2f} GB)")
    print(f"  Total : {total_used:.2f} GB of {max_gb} GB budget")

    return selected_train + selected_test

def download_and_flatten(files: list[tuple[str, int]],
                         split_map: dict[str, Path],
                         tmp_dir: Path,
                         repo_id: str,
                         repo_type: str,
                         token: str,
                         label: str = "file") -> None:
    """
    Download each file into tmp_dir, then move it flat into the matching split folder.

    Repo path pattern:   train_data/<PATIENT_ID>/<filename>.<ext>
    Target flat layout:  split_map["train_data"]/<filename>.<ext>
    """
    for dest in split_map.values():
        dest.mkdir(parents=True, exist_ok=True)

    total = len(files)
    for idx, (repo_path, _size) in enumerate(files, start=1):
        parts = Path(repo_path).parts   # ('train_data', 'GJB0004201T', 'GJB0004201T01.nii')
        if len(parts) < 2:
            print(f"  [SKIP] Unexpected path: {repo_path}")
            continue

        split_key = parts[0]            # 'train_data' or 'test_data'
        filename   = parts[-1]          # e.g. 'GJB0004201T01.nii' or 'GJB0004201TT.json'

        if split_key not in split_map:
            print(f"  [SKIP] Unknown split '{split_key}' in {repo_path}")
            continue

        dest_file = split_map[split_key] / filename

        if dest_file.exists():
            print(f"  [{idx}/{total}] Already exists, skipping: {filename}")
            continue

        print(f"  [{idx}/{total}] Downloading {label}: {repo_path} …")
        try:
            cached_path = hf_hub_download(
                repo_id   = repo_id,
                repo_type = repo_type,
                filename  = repo_path,
                local_dir = str(tmp_dir),
                token     = token,
            )
            shutil.move(cached_path, dest_file)
            print(f"           → {dest_file.relative_to(DATA_ROOT)}")
        except Exception as exc:
            print(f"  [ERROR] {repo_path}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Download KneeCoT dataset files.")
    parser.add_argument("--nii-only",         action="store_true", help="Download MRI files only")
    parser.add_argument("--annotations-only", action="store_true", help="Download annotation JSON files only")
    parser.add_argument("--max-gb", type=float, default=None,help="Max total GB to download (applies per file type)")
    parser.add_argument("--train-ratio", type=float, default=0.7,help="Fraction of --max-gb budget for train split (default: 0.7)")
    parser.add_argument("--token", type=str, required=True,help="Your HuggingFace read token (hf_xxxxxxxxxxxx)")
    args = parser.parse_args()

    do_nii  = not args.annotations_only
    do_json = not args.nii_only

    print(f"Data root : {DATA_ROOT}")
    print(f"Repository: {REPO_ID}")
    print(f"Download  : {'MRI (.nii) ' if do_nii else ''}{'Annotations (.json)' if do_json else ''}\\n")

    tmp_dir = DATA_ROOT / "tmp_hf_cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── MRI files ──────────────────────────────────────────────────────────
    if do_nii:
        nii_files = scan_repo_files(REPO_ID, REPO_TYPE, args.token, (".nii",))
        nii_files = [(p, s) for p, s in nii_files
                     if any(p.startswith(sp + "/") for sp in SPLITS)]
        if args.max_gb:
            nii_files = apply_max_gb(nii_files, args.max_gb, args.train_ratio)
        print(f"  .nii files in target splits: {len(nii_files)}\\n")
        download_and_flatten(
            nii_files, SPLIT_MAP_NII, tmp_dir,
            REPO_ID, REPO_TYPE, args.token, label=".nii"
        )
        nii_train = len(list(SPLIT_MAP_NII["train_data"].glob("*.nii")))
        nii_test  = len(list(SPLIT_MAP_NII["test_data"].glob("*.nii")))
        print(f"\\n  ✓ MRI files  – train: {nii_train}  |  test: {nii_test}\\n")

    # ── Annotation JSON files ───────────────────────────────────────────────
    if do_json:
        json_files = scan_repo_files(REPO_ID, REPO_TYPE, args.token, (".json",))
        json_files = [(p, s) for p, s in json_files
                      if any(p.startswith(sp + "/") for sp in SPLITS)]
        if args.max_gb:
            json_files = apply_max_gb(json_files, args.max_gb, args.train_ratio)
        print(f"  .json files in target splits: {len(json_files)}\\n")
        download_and_flatten(
            json_files, SPLIT_MAP_JSON, tmp_dir,
            REPO_ID, REPO_TYPE, args.token, label=".json"
        )
        ann_train = len(list(SPLIT_MAP_JSON["train_data"].glob("*.json")))
        ann_test  = len(list(SPLIT_MAP_JSON["test_data"].glob("*.json")))
        print(f"\\n  ✓ Annotations – train: {ann_train}  |  test: {ann_test}\\n")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    print("Cleaning up temporary download cache …")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("  tmp_hf_cache removed.")

    print("\\n✓ All done!")


if __name__ == "__main__":
    main()
