"""
preprocessing.py  –  Step 1: extract PNG slices from .nii files.

Strategy: extract 10 evenly-spaced sagittal slices from each volume,
then stitch them into a single 2×5 grid PNG.  Sending one stitched image
per case keeps the VLM token budget low while giving the model coverage
across the full depth of the volume.

Usage:
    python preprocessing.py
"""
import os
import math
import numpy as np
import nibabel as nib
from PIL import Image
from config import TRAIN_NII, TEST_NII, TRAIN_SLC, TEST_SLC

# ── Slice extraction settings ─────────────────────────────────────────────────
N_SLICES   = 10          # number of slices sampled per volume
GRID_COLS  = 5           # columns in the stitched grid  (rows = N_SLICES / GRID_COLS)
SLICE_SIZE = (224, 224)  # each individual tile is resized to this before stitching
# ─────────────────────────────────────────────────────────────────────────────


def _normalise_slice(sl: np.ndarray) -> np.ndarray:
    """Percentile-clip and scale a 2D array to uint8."""
    p1, p99 = np.percentile(sl, 1), np.percentile(sl, 99)
    sl_norm = np.clip((sl - p1) / (p99 - p1 + 1e-8) * 255, 0, 255).astype(np.uint8)
    return sl_norm


def extract_slices(nii_path: str, output_dir: str, case_id: str,
                   n_slices: int = N_SLICES,
                   grid_cols: int = GRID_COLS,
                   tile_size: tuple = SLICE_SIZE) -> str:
    """
    Extract `n_slices` evenly-spaced sagittal slices from a .nii volume,
    stitch them into a (grid_rows × grid_cols) grid, and save as one PNG.

    Returns the path to the saved grid image.

    The grid is always saved as:  <output_dir>/<case_id>_grid.png
    Individual slice PNGs are NOT saved — only the grid is kept so that
    evaluate.py can load a single image per case.
    """
    img  = nib.load(nii_path)
    data = img.get_fdata()

    # Auto-detect the sagittal axis as the largest dimension
    sag_axis = int(np.argmax(data.shape))
    D = data.shape[sag_axis]

    # Pick n_slices evenly across [10%, 90%] of the depth to skip blank edges
    lo = max(0, int(D * 0.10))
    hi = min(D - 1, int(D * 0.90))
    indices = [int(round(lo + (hi - lo) * i / (n_slices - 1)))
               for i in range(n_slices)]

    tiles = []
    for idx in indices:
        sl = np.take(data, idx, axis=sag_axis)
        sl_norm = _normalise_slice(sl)
        tile = Image.fromarray(sl_norm).convert("RGB").resize(
            tile_size, Image.LANCZOS)
        tiles.append(tile)

    # Build the stitched grid
    grid_rows = math.ceil(n_slices / grid_cols)
    grid_w = tile_size[0] * grid_cols
    grid_h = tile_size[1] * grid_rows
    grid = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))
    for i, tile in enumerate(tiles):
        row = i // grid_cols
        col = i % grid_cols
        grid.paste(tile, (col * tile_size[0], row * tile_size[1]))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{case_id}_grid.png")
    grid.save(out_path)
    return out_path


def run_slice_extraction(src_nii_dir, out_slc_dir, split_name: str) -> None:
    nii_files = [f for f in os.listdir(str(src_nii_dir)) if f.endswith(".nii")]
    print(f"\n[{split_name}] Found {len(nii_files)} .nii files — extracting {N_SLICES}-slice grids...")
    errors = []
    for fname in nii_files:
        case_id = fname.replace(".nii", "")
        try:
            extract_slices(os.path.join(str(src_nii_dir), fname),
                           str(out_slc_dir), case_id)
            print(f"  {case_id}", end=" ")
        except Exception as e:
            print(f"\n  [ERROR] {case_id}: {e}")
            errors.append(case_id)
    total = len([f for f in os.listdir(str(out_slc_dir)) if f.endswith(".png")])
    print(f"\n[{split_name}] Done! Grid PNGs: {total}  Errors: {len(errors)}")
    if errors:
        print(f"  Failed: {errors}")


if __name__ == "__main__":
    run_slice_extraction(TRAIN_NII, TRAIN_SLC, "TRAIN")
    run_slice_extraction(TEST_NII,  TEST_SLC,  "TEST")
    print("\nSlice extraction complete.")
