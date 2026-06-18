# Does Vision Help Reason? — H5 Text-Only LLM Line

Code for the **H5** experiment of our knee MRI Chain-of-Thought study: comparing
**direct** vs **Chain-of-Thought (CoT)** prompting of a text-only LLM on Chinese
knee MRI VQA, using the [KneeCoT](https://huggingface.co/datasets/YiHui0124/KneeCoT)
dataset.

> This repository covers the **text-only LLM line**. The multimodal VLM line and
> the H6 fine-tuning experiment live in separate modules (see *H6* below).

## What this code does

For each case it feeds the model **only the free-text MR findings** (`MR表现`),
asks the dataset's **yes/no** and **inference** questions, and runs each question
under two prompting conditions — `direct` and `cot` — with **greedy
(deterministic) decoding**. It then scores yes/no accuracy and runs McNemar's
test comparing the two conditions.

**Round 2 note:** this stage demonstrates a working end-to-end pipeline on a
50 case prototype. For the matched LLM–VLM comparison, both pipelines use the same fixed evaluation split generated with seed 42. This ensures that all four conditions are evaluated on identical QA pairs. The accuracy numbers themselves are analysed in Round 3.

## Repository structure

```
.
├── run.py                  # end-to-end runner: preprocess -> infer -> evaluate
├── requirements.txt
├── src/
│   ├── build_eval_set.py   
│   ├── prompts.py          # direct + 4-step CoT templates
│   ├── preprocessing.py    # load JSON, filter knee-only, build eval set
│   ├── inference.py        # load Qwen2.5-7B (4-bit), greedy decoding
│   └── evaluation.py       # parse Yes/No, accuracy, McNemar
├── tests/
│   └── test_pipeline.py    # data + parsing sanity tests (no GPU needed)
└── data/
    ├── sample/             # 1–2 de-identified example cases ONLY
    └── cases/              # <- put downloaded KneeCoT JSONs here (gitignored)
```

## Data access (required)

KneeCoT requires a Hugging Face account and **dataset access approval from the
authors** — request it on the
[dataset page](https://huggingface.co/datasets/YiHui0124/KneeCoT).
For H5 only the **JSON annotations** are needed (not the ~1 TB of MRI volumes).

**Do not commit the dataset to this repo** — it is access-restricted. Download
the JSON files locally into `data/cases/`. The repo ships only a couple of
de-identified example cases under `data/sample/`.

## Setup

### Option A — Google Colab (recommended for the prototype)

In a Colab notebook with a GPU runtime (`Runtime > Change runtime type > T4 GPU`):

```python
!git clone https://github.com/<your-org>/<your-repo>.git
%cd <your-repo>
!pip install -q -r requirements.txt

from huggingface_hub import login
login()  # paste your HF token (needed for dataset access)

# put your downloaded KneeCoT JSONs in data/cases/, then:
!python run.py --data_dir data/cases --sample_size 50
```

4-bit loading (the default) lets the 7B model fit on a free Colab T4 (~16 GB).

### Option B — HPC (for scale-up and H6)

```bash
pip install -r requirements.txt
python run.py --data_dir /path/to/cases --sample_size 0   # 0 = use all cases
```

## How to run

```bash
# prototype: 50 knee-only cases
python run.py --data_dir data/cases --sample_size 50

# test the pipeline WITHOUT a GPU/model (fake outputs)
python run.py --data_dir data/sample --sample_size 0 --mock

# unit tests
python tests/test_pipeline.py
```

Outputs are written to `results/`:
`raw_results.json`, `yes_no_accuracy.json`, `mcnemar.json`,
`inference_outputs.json` (inference answers kept for LLM-as-judge in Round 3).

## Method summary

- **Models:** text-only `Qwen/Qwen2.5-7B-Instruct` (strong Chinese, open,
  runs locally — no external API for medical data).
- **Input:** free-text findings (`MR表现`) only. The diagnostic impression
  (`诊断意见`) and structured labels (`标签`) are **excluded** to avoid leaking
  the answers.
- **Conditions:** `direct` (answer immediately) vs `cot` (four-step structured
  reasoning grounded in the dataset's expert annotations).
- **Tasks:** yes/no questions form the main quantitative accuracy backbone. 
             Inference questions are included as a pilot diagnostic-inference analysis in                Round 2, using final-conclusion extraction and rule-based matching. Yes/no                  accuracy and inference accuracy are reported separately because the two task
             types differ in difficulty.
- **Decoding:** greedy (`do_sample=False`) for determinism and reproducibility.
- **Metrics:** yes/no accuracy per condition; McNemar's test for significance.

## H6 (planned)

Fine-tuning (QLoRA on the CoT annotations) vs few-shot prompting of a larger
model is an optional extension. The data-formatting and training scripts will
be added under `src/` as `finetune.py`; this is described in the paper's
methodology and is not part of the Round 2 deliverable.

## Reproducibility

Fixed sampling seed (`--seed`, default 42) + greedy decoding ⇒ the same cases
and the same outputs on every run.
