# prepare_data.py — KneeCoT Dataset Downloader

Downloads MRI (`.nii`) and annotation (`.json`) files from the
[YiHui0124/KneeCoT](https://huggingface.co/datasets/YiHui0124/KneeCoT)
HuggingFace dataset and organizes them into a flat folder structure.

---

## Requirements

```bash
pip install huggingface_hub
```

---

## Folder Structure

Place `prepare_data.py` inside `ds2026-TBD-IDTBD-group4/data/`.
After running, the following structure will be created:

```
ds2026-TBD-IDTBD-group4/
└── data/
    ├── prepare_data.py
    ├── train/               ← MRI files (.nii)
    ├── test/                ← MRI files (.nii)
    └── annotations/
        ├── train/           ← annotation files (.json)
        └── test/            ← annotation files (.json)
```

---

## Usage

Navigate to the `data/` folder first:

```bash
cd ds2026-TBD-IDTBD-group4/data
```

Then run with your HuggingFace read token:

```bash
# Download everything (MRI + annotations, no size limit)
python prepare_data.py --token hf_xxxxxxxxxxxx

# Download with a total size limit (default split: 70% train / 30% test)
python prepare_data.py --token hf_xxxxxxxxxxxx --max-gb 20

# Custom train/test ratio — 80% train, 20% test
python prepare_data.py --token hf_xxxxxxxxxxxx --max-gb 20 --train-ratio 0.8

# 50/50 split
python prepare_data.py --token hf_xxxxxxxxxxxx --max-gb 20 --train-ratio 0.5

# Download MRI files only
python prepare_data.py --token hf_xxxxxxxxxxxx --nii-only

# Download annotations only
python prepare_data.py --token hf_xxxxxxxxxxxx --annotations-only

# MRI only with size limit and custom ratio
python prepare_data.py --token hf_xxxxxxxxxxxx --nii-only --max-gb 10 --train-ratio 0.8
```

---

## Arguments

| Argument             | Type    | Required | Default | Description                                        |
|----------------------|---------|----------|---------|----------------------------------------------------|
| `--token`            | string  | ✅ Yes   | —       | HuggingFace read token (`hf_xxx...`)               |
| `--max-gb`           | float   | No       | None    | Max total GB to download (applies per file type)   |
| `--train-ratio`      | float   | No       | `0.7`   | Fraction of `--max-gb` budget for train split      |
| `--nii-only`         | flag    | No       | False   | Download MRI `.nii` files only                     |
| `--annotations-only` | flag    | No       | False   | Download annotation `.json` files only             |

---

## How `--max-gb` works

The budget is split between `train_data` and `test_data` according to `--train-ratio`:

```
--max-gb 20 --train-ratio 0.7
  → train budget : 14.0 GB
  → test  budget :  6.0 GB
```

Files are selected in repo order within each split until the budget is reached.
The limit applies independently to `.nii` and `.json` file types.

---

## Get your HuggingFace Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → select **Read** role
3. Copy the token and pass it via `--token`

---

## Notes

- Re-running the script is **resume-safe** — already downloaded files are skipped automatically.
- Do **not** commit your token to Git.
- Add the following to your `.gitignore` to avoid pushing large data files:

```gitignore
data/train/
data/test/
data/annotations/
data/tmp_hf_cache/
.env
**/.env
```