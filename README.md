# Does Vision Help Reason? — H5 LLM and VLM Evaluation Pipeline

This repository contains the **H5 experiment** for a knee MRI Chain-of-Thought study on the KneeCoT dataset. The project compares **Direct Answer (DA)** prompting and **Chain-of-Thought (CoT)** prompting under two matched settings:

1. **Text-only LLM pipeline**: uses only the free-text MR findings (`MR表现`).
2. **Vision-language VLM pipeline**: uses the same MR findings plus knee MRI slice images extracted from `.nii` volumes, and also supports an image-only condition (no MR findings) as an ablation.

The goal is to evaluate whether structured CoT prompting improves diagnostic question answering, and whether visual input provides additional value beyond text-only MR findings.

---

## What this repository includes

This repository contains two parallel pipelines for the H5 experiment — a text-only LLM pipeline and a multimodal VLM pipeline — plus a single shared scoring/comparison layer that both lines are judged by. The two generation pipelines are kept in separate folders because they use different input modalities, runtime environments, and output formats, but they read the same frozen evaluation set and are scored by the same code, so the comparison between them is apples-to-apples.

## Pipeline Overview

| Component | Text-only LLM pipeline | Multimodal VLM pipeline |
|---|---|---|
| Folder | `code/kneecot-h5-pipeline-llm/` | `code/kneecot-h5-pipeline-vlm/` |
| Main entry point | `run.py` | `run.py` (`VLM.ipynb` was the original Colab prototype) |
| Input modality | MR findings text only (`MR表现`) | MRI slices, alone or combined with MR findings text |
| Main output folder | `data/llm_results/` (LLM line) | `data/vlm_results/` |
| Prompting conditions | Direct Answer and CoT | DA / CoT (image-only) and DA_findings / CoT_findings (image + text) |
| Purpose | Measures text-only reasoning performance | Measures whether visual input improves reasoning, and whether MR findings text adds on top of the image |

Both generation pipelines only produce **raw, unscored** per-question records (`case_id`, `question_id`, `qtype`, `prompt_key`, `raw_output`, ...). Scoring happens afterward in one shared place: `code/analysis/compare.py` (yes/no accuracy + McNemar) and `code/analysis/judge.py` (LLM-as-judge for inference questions). 

## Repository Structure

```text
README.md
Running_Guidelines.md             # step-by-step runbook (data -> generation -> scoring -> compare)
setup_runpod.sh                   # one-shot setup + run script for a rented GPU pod
JOURNAL.md                        # weekly project journal
notes/
└── coding problems and solutions.md   # internal dev notes / architecture decisions (not for final submission)

data/
├── prepare_data.py                 # downloads KneeCoT from HuggingFace
├── eval_set.json                   # the single frozen, shared evaluation set (seed=42)
├── llm_results/                    # combined LLM raw outputs used in the final comparison
└── vlm_results/                    # combined VLM raw outputs used in the final comparison

code/
├── analysis/                       # the ONLY scoring/comparison code — shared by both lines
│   ├── compare.py                  # yes/no accuracy + McNemar; the main RQ1/RQ2/RQ3 comparison script
│   ├── judge.py                    # local LLM-as-judge for inference questions (Ollama, qwen2.5:32b)
│   ├── inference_rubric_for_LLM_judge.json   # question-aware rubric used by judge.py
│   └── vlm_findings_ablation.py    # VLM-only ablation: image-only vs image+findings, reuses compare.py
│
├── kneecot-h5-pipeline-llm/
│   ├── data/sample/                # small de-identified sample case for smoke-testing
│   ├── src/
│   │   ├── preprocessing.py        # load case JSON, filter knee-only cases, build eval items
│   │   ├── prompts.py              # Direct Answer and Chain-of-Thought prompt templates
│   │   ├── inference.py            # loads Qwen2.5-7B-Instruct (4-bit) and runs greedy generation
│   │   └── evaluation.py           # raw-output I/O only; no scoring (moved to code/analysis/)
│   ├── tests/                      # sanity tests for preprocessing and parsing
│   ├── run.py                      # end-to-end LLM runner (generation only)
│   └── requirements.txt
│
└── kneecot-h5-pipeline-vlm/
    ├── config.py                   # paths, model toggle, N_EVAL, CONDITIONS
    ├── preprocessing.py             # extract + stitch MRI slices into one grid PNG per case
    ├── build_eval_set.py           # builds the shared eval_set.json (also resolves .nii paths)
    ├── prompts.py                   # DA / CoT / DA_findings / CoT_findings prompt templates
    ├── evaluate.py                  # runs VLM inference via a local Ollama server (generation only)
    ├── run.py                       # master runner: preprocess -> build_eval_set -> evaluate
    ├── metrics.py                   # legacy/unused — see "Evaluation" section below
    ├── VLM.ipynb                    # original Colab prototype notebook (superseded by run.py)
    └── requirements.txt

results/
├── compare_out/                    # output of code/analysis/compare.py
│   ├── summary_2x2_comparison.json # 2x2 accuracy table + McNemar (RQ1/RQ2), per qtype
│   ├── summary_vlm_ablation.json   # output of vlm_findings_ablation.py (RQ3 image-only ablation)
│   ├── per_item.csv                # every question's correct/incorrect outcome under all 4 conditions
│   ├── rq3_yesno.csv               # per-item LLM-vs-VLM comparison under CoT (visual-necessity labels)
│   └── rq3_inference.csv
├── judged_inference_llm.json / judged_inference_vlm.json   # judge.py output, after manual correction
└── manual_review_llm.jsonl / manual_review_vlm.jsonl       # risk-based manual review samples
```

The LLM pipeline and the VLM pipeline are not identical in structure because they process different inputs (text-only JSON vs. `.nii` MRI volumes + text) and the VLM line talks to a local Ollama server instead of loading weights directly in Python. Both contain the same kind of pieces though: source code, dependency specification, and a generation-only `run.py` entry point. Everything under `notes/` is internal team reference and is not part of the final project submission.

## Data access

The project uses the [KneeCoT dataset](https://huggingface.co/datasets/YiHui0124/KneeCoT), which requires a Hugging Face account and access approval from the dataset authors.

Do **not** commit the full dataset to this repository. The full KneeCoT data are access-restricted and may include very large MRI volumes. This repository should only contain small de-identified samples under `code/kneecot-h5-pipeline-llm/data/sample/`.

For the LLM pipeline, only JSON annotations are required. For the VLM pipeline, both JSON annotations and corresponding `.nii` MRI volumes are required. See `Running_Guidelines.md` for the full data download + setup steps.

---

# 1. Text-only LLM Pipeline

## 1.1 Model Architecture

### Input

The text-only LLM receives only the free-text MR findings field (`MR表现`) from each KneeCoT JSON case. The diagnostic opinion (`诊断意见`) and structured labels (`标签`) are deliberately excluded because they state the answer directly and would cause answer leakage.

The preprocessing module loads the case identifier, MR findings, and QA pairs. Non-knee cases are removed (cases whose `检查方法` field does not contain `膝关节`, or that contain `肩关节`). Yes/no questions (`type: yes_no`, normalised to `yesno`) and inference questions (`type: inference`) are the two question types used for H5; other question types in the dataset are not part of this evaluation.

For the matched LLM–VLM comparison, both pipelines read the same frozen evaluation set, `data/eval_set.json`, built once by the VLM side's `build_eval_set.py --n-eval 200` with `seed=42` (see `Running_Guidelines.md` step 2 — the LLM line must **not** build its own separate eval set, or its question list will not match the VLM line's). The set used for this round contains **400 items across 278 distinct cases**: 200 yes/no questions (balanced 100 `Yes` / 100 `No`, at most one `Yes` and one `No` example per case) and 200 inference questions.

### Model

The LLM pipeline uses:

```text
Qwen/Qwen2.5-7B-Instruct
```

This model was selected because it is open, has strong Chinese-language capability, and can run locally without sending medical data to an external API. 4-bit loading (`bitsandbytes`, NF4) lets the 7B model fit on a single consumer/Colab-class GPU.

---

## 1.2 Processing

### Prompting conditions

The LLM is evaluated under two prompting conditions, defined in `code/kneecot-h5-pipeline-llm/src/prompts.py`:

1. **Direct Answer (DA)**: the model answers immediately without writing reasoning.
2. **Structured Chain-of-Thought (CoT)**: the model follows a four-step diagnostic reasoning template before giving the final answer.

All prompts are written in Chinese to match the dataset language. Using English prompts with Chinese MR findings can make the model mix languages during long generations, which may reduce the reliability of answer extraction.

Both templates share two instruction fragments that pin down the output format so it can be parsed reliably:

```text
YESNO_INSTRUCTION:
- 若为是非类问题：请在【答案】后单独另起一行，只写英文单词 Yes 或 No，
  不要写中文"是"/"否"，不要附加任何其他文字、标点或解释。

INFERENCE_INSTRUCTION:
- 若为推理类问题：禁止只回答一个词或一个短语（例如仅回答"骨关节炎"）。
  请在【答案】后用1-2句完整的话给出明确结论，并说明支撑该结论的具体依据
  （引用 MR 表现中的具体征象），不少于20个字。
```

### Text-only LLM — Direct Answer prompt

```text
你是一位资深骨骼肌肉放射科医生。下面给出一份膝关节 MR 表现和一个相关问题。
请直接给出最终结论作答，不需要展示逐步推理过程。

<YESNO_INSTRUCTION>
<INFERENCE_INSTRUCTION>

【MR 表现】
{findings}

【问题】
{question}

【答案】
```

### Text-only LLM — Structured 4-Step CoT prompt

```text
你是一位资深骨骼肌肉放射科医生。请根据下面的膝关节 MR 表现，按以下四个步骤进行系统推理，然后回答问题。

步骤一 系统性梳理（Systematic Observation）
按解剖部位（半月板、韧带、骨与软骨、关节腔与滑膜、其他结构如脂肪垫与软组织）逐项梳理 MR 表现中提到的关键征象，记录每个部位的信号、形态与连续性。

步骤二 解读与核对（Interpretation and Verification）
对每条征象判断属于正常还是异常，并说明该改变通常提示什么（如 T2WI 高信号提示水肿或损伤），核对前后是否一致。

步骤三 解剖结构分析（Anatomical Structure Analysis）
按系统逐一分析：
3.1 半月板：形态是否完整、高信号是否达关节面、损伤程度。
3.2 韧带：前/后交叉韧带、内/外侧副韧带的连续性与信号。
3.3 骨与软骨：骨髓信号（水肿/挫伤）、关节面软骨是否光整。
3.4 关节腔与滑膜：有无积液、滑膜情况。
3.5 其他结构：髌下脂肪垫、关节周围软组织。

步骤四 诊断推理与核对（Diagnostic Reasoning and Verification）
综合以上分析推导出针对问题的结论；如适用，简要排除主要鉴别诊断；自检结论是否由前述证据支持。

最后必须单独另起一行给出对问题的明确回答：
<YESNO_INSTRUCTION>
- 推理类问题：给出明确结论及推理依据。

【MR 表现】
{findings}

【问题】
{question}

请依次完成步骤一至步骤四，最后给出【答案】。
```

`max_new_tokens` is capped per condition (128 for DA, 1024 for CoT) since DA answers are short and CoT answers need room for all four steps.

### Answer parsing and scoring

Answer parsing and scoring are **not** done by this pipeline itself anymore — `run.py` only writes raw generations to `results/raw_results.json`. Scoring is one shared step used by both the LLM and VLM lines; see [Section 3 — Evaluation](#3-evaluation) for the actual scoring logic (`code/analysis/compare.py` for yes/no, `code/analysis/judge.py` for inference).

---

## 1.3 Optimization

The H5 LLM pipeline is **inference-only**. There is no training loop, no loss function, and no optimizer. The model weights remain frozen throughout the experiment.

Greedy decoding is used for deterministic and reproducible inference:

```text
do_sample = False
```

Because the same frozen evaluation split and deterministic decoding are used for both DA and CoT conditions, any accuracy difference between the two LLM settings reflects the effect of prompt structure rather than random sampling or model stochasticity.

---

## 1.4 Model Implementation

The LLM pipeline can be run from a Python script (`run.py`) on any machine with a GPU, or smoke-tested on CPU with `--mock`.

Core dependencies (`code/kneecot-h5-pipeline-llm/requirements.txt`):

| Library | Role |
|---|---|
| `transformers` | Loading and running Qwen2.5-7B-Instruct |
| `torch` | Model execution |
| `bitsandbytes` | 4-bit (NF4) loading |
| `accelerate` | Device mapping and efficient inference |
| `sentencepiece` | Tokenizer dependency for Qwen2.5 |
| `statsmodels` | Listed but not currently imported by this pipeline's own code |

(`scipy`, used for the McNemar test, is a dependency of `code/analysis/compare.py`, not of this pipeline directly.)

Example setup and run:

```bash
pip install -r requirements.txt

# load the shared eval set (built by the VLM side, see Running_Guidelines.md step 2)
python run.py --eval_set ../../data/eval_set.json --model_name Qwen/Qwen2.5-7B-Instruct --out_dir results
```

To test the pipeline without a GPU or model call:

```bash
python run.py --data_dir data/sample --sample_size 0 --mock
python tests/test_pipeline.py
```

The final full-scale run for this experiment did not use Colab. Both the
LLM and VLM lines were run together on a single GPU pod rented on
[RunPod](https://www.runpod.io/) (an **NVIDIA RTX PRO 6000**), driven by
`setup_runpod.sh`. See `Running_Guidelines.md` for the full step-by-step
runbook on that setup.

LLM outputs are written to `llm_results/raw_results.json`, one record per `(case_id, question, prompt_key)` with fields `case_id`, `question_id`/`question`, `qtype`, `ground_truth`/`full_answer`, `prompt_key` (`DA`/`CoT`), and `raw_output`.

---

# 2. Vision-Language VLM Pipeline

## 2.1 Model Architecture

### Input

The VLM receives one or two inputs depending on condition:

1. MRI slice images extracted from the `.nii` volume (always present).
2. The same free-text MR findings (`MR表现`) used by the text-only LLM (present only in the `_findings` conditions, see §2.2).

For the `_findings` conditions, the comparison with the LLM line is matched: the LLM and VLM receive identical text evidence, and the only additional input to the VLM is the image. The plain (no `_findings`) conditions instead test the VLM on the image alone, as an ablation.

As in the LLM pipeline, the diagnostic opinion (`诊断意见`) and structured labels (`标签`) are excluded to avoid answer leakage. The loader reads only the case identifier, MR findings, QA pairs, and fields required for knee-only filtering.

The VLM pipeline does not re-sample evaluation examples. It reads the same frozen `data/eval_set.json` as the text-only LLM pipeline (in fact, this is the file the VLM side's `build_eval_set.py` builds), so all conditions are evaluated on the same `(case_id, question)` entries.

### Image processing

The visual input comes from 3D `.nii` MRI volumes, processed once up front by `preprocessing.py` (step 1 of `run.py`), not on the fly during inference.

For each volume:

- Loaded with `nibabel`; the sagittal axis is auto-detected as the dimension with the largest extent, rather than assuming a fixed axis index, so slicing stays correct even when a scan's sagittal dimension isn't stored as axis 2.
- 10 slices are sampled evenly across the central 10%–90% of that axis's depth, skipping the blank edges of the volume.
- Each slice is percentile-clipped (1st/99th percentile) and scaled to 8-bit, then resized to 224×224.
- All 10 slices are stitched into a single 2×5 grid PNG (`<case_id>_grid.png`) and saved once per case.

`evaluate.py` then base64-encodes that single grid PNG and sends it as the only image attached to the Ollama chat call. Sending one stitched image (instead of 10 separate ones) keeps the per-case visual token cost low while still giving the model coverage across most of the volume's depth, which is a meaningfully broader view than an earlier two-slice (`[D//3, D//2]`) design that only looked at two fixed central slices.

This is still a coarse downsampling of a full 3D volume into 10 flattened 2D slices, so peripheral or off-axis structures can still be missed, and the visual contribution measured by this pipeline should still be read as a lower bound rather than the VLM's full potential — this remains an acknowledged paper limitation.

### Model

The primary VLM is:

```text
Qwen2.5-VL
```

served locally through Ollama (`ollama pull qwen2.5vl`). It was selected because it shares a model family with the text-only Qwen2.5-7B LLM, making the LLM–VLM comparison cleaner.

A secondary model, `MiniCPM-V`, can also be enabled in `config.py` (`MODELS["minicpm-v"]["enabled"] = True`) for multi-model comparison; it is disabled by default.

---

## 2.2 Processing

### Prompting conditions

The VLM is evaluated under **four** prompt conditions, defined in `code/kneecot-h5-pipeline-vlm/prompts.py` (`VLM_PROMPTS`) and listed in `config.py`'s `CONDITIONS`:

1. **DA** — Direct Answer, image only.
2. **CoT** — Structured 4-Step Chain-of-Thought, image only.
3. **DA_findings** — Direct Answer, image + MR findings text.
4. **CoT_findings** — Structured 4-Step CoT, image + MR findings text.

All four conditions are actually run and scored (not a future/planned ablation) — `evaluate.py` runs every condition in `CONDITIONS` over the full eval set, producing one result file per `(model, prompt_key, qtype)`. The `_findings` conditions are the ones used for the main matched LLM-vs-VLM comparison (RQ1/RQ2: both models see the same MR findings text, the VLM additionally sees the image). The plain `DA`/`CoT` (image-only) conditions are instead the RQ3 ablation — do images alone, without any MR findings text, support the same conclusions? See §3.3 for how this ablation is scored from the same raw outputs.

The prompts are written in Chinese and use the same final `【答案】` marker and `YESNO_INSTRUCTION`/`INFERENCE_INSTRUCTION` format rules as the LLM pipeline (see §1.2), so the shared parser in `code/analysis/compare.py` can read both lines' outputs the same way.

### Image-only VLM — Direct Answer (DA)

```text
你是一位资深骨骼肌肉放射科医生。下面给出一组膝关节 MR 图像和一个相关问题。
请直接给出最终结论作答，不需要展示逐步推理过程（不要写"步骤一/步骤二"这类分步过程）。
<YESNO_INSTRUCTION>
<INFERENCE_INSTRUCTION>
【问题】{question}
【答案】
```

### Image-only VLM — Structured 4-Step CoT

```text
你是一位资深骨骼肌肉放射科医生。请根据下面给出的膝关节 MR 图像，按以下四个步骤进行系统推理，然后回答问题。

步骤一 系统性观察（Systematic Observation）
系统性地观察所给的 MR 图像，按解剖部位逐项描述关键征象。

步骤二 解读与核对（Interpretation and Verification）
对每条征象判断正常还是异常，说明该改变通常提示什么。

步骤三 解剖结构分析（Anatomical Structure Analysis）
3.1 半月板  3.2 韧带  3.3 骨与软骨  3.4 关节腔与滑膜  3.5 其他结构

步骤四 诊断推理与核对（Diagnostic Reasoning and Verification）
综合分析推导结论；自检结论是否由证据支持。

完成以上推理步骤后，最后必须单独另起一行给出【答案】：
<YESNO_INSTRUCTION>
- 若为推理类问题：在【答案】后给出明确结论及依据。
【问题】{question}
请依次完成步骤一至步骤四，最后给出【答案】
```

### Image + MR findings VLM — Direct Answer (DA_findings)

```text
你是一位资深骨骼肌肉放射科医生。下面给出一组膝关节 MR 图像、对应的 MR 表现文字和一个相关问题。
请结合图像与 MR 表现直接给出最终结论作答，不需要展示逐步推理过程（不要写"步骤一/步骤二"这类分步过程）。
<YESNO_INSTRUCTION>
<INFERENCE_INSTRUCTION>
【MR 表现】{findings}
【问题】{question}
【答案】
```

### Image + MR findings VLM — Structured 4-Step CoT (CoT_findings)

```text
你是一位资深骨骼肌肉放射科医生。请根据下面给出的膝关节 MR 图像及其对应的 MR 表现文字，按以下四个步骤进行系统推理，然后回答问题。

步骤一 系统性观察（Systematic Observation）
系统性地观察所给的 MR 图像，并对照所提供的 MR 表现文字，按解剖部位逐项梳理关键征象。

步骤二 解读与核对（Interpretation and Verification）
对每条征象判断正常还是异常，说明该改变通常提示什么。

步骤三 解剖结构分析（Anatomical Structure Analysis）
3.1 半月板  3.2 韧带  3.3 骨与软骨  3.4 关节腔与滑膜  3.5 其他结构

步骤四 诊断推理与核对（Diagnostic Reasoning and Verification）
综合分析推导结论；如适用简要排除主要鉴别诊断；自检结论是否由证据支持。

完成以上推理步骤后，最后必须单独另起一行给出【答案】：
<YESNO_INSTRUCTION>
- 若为推理类问题：在【答案】后给出明确结论及依据。
【MR 表现】{findings}
【问题】{question}
请依次完成步骤一至步骤四，最后给出【答案】。
```

---

## 2.3 Optimization

The H5 VLM pipeline is also **inference-only**. There is no training, no loss function, no optimizer, no fine-tuning, no QLoRA, and no gradient update.

The model is evaluated using frozen pretrained weights served locally through Ollama. All inference calls use:

```text
temperature = 0.0
num_ctx = 4096
num_predict = 2400
```

This matches the LLM line's `do_sample=False` greedy decoding: any accuracy difference between DA and CoT (or between the image-only and `_findings` conditions) reflects the effect of prompt/input structure, not sampling randomness or model adaptation to the KneeCoT domain.

---

## 2.4 Model Implementation

The VLM pipeline's production entry point is the script-based `run.py` (steps: `preprocessing.py` → `build_eval_set.py` → `evaluate.py`), run on the same rented RunPod GPU pod (NVIDIA RTX PRO 6000) as the LLM line, so both lines plus the local judge model shared one machine and one Ollama server — see `Running_Guidelines.md` and `setup_runpod.sh`.

`VLM.ipynb` was the original Colab prototyping notebook the script-based pipeline was developed from; it is kept in the repository for reference but is not what produced the final results.

Core dependencies actually used by the active pipeline scripts:

| Library | Role |
|---|---|
| `ollama` | Multimodal inference through the Ollama REST API |
| `nibabel` | Loading `.nii` MRI volumes |
| `numpy` | Percentile normalization of slice arrays |
| `Pillow` | Resizing slices and stitching/saving the grid PNG |

`requirements.txt` also lists `pandas`, `matplotlib`, and `opencv-python`: `pandas`/`matplotlib` were used by `metrics.py`, which is **no longer part of the official scoring path** (see §3 — scoring moved to `code/analysis/compare.py` and `judge.py`); `opencv-python` was used by an earlier CLAHE-based slice-processing design and is not imported anywhere in the current `preprocessing.py`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Then point `config.py`'s `ROOT`-derived paths at your KneeCoT data folder (or just run from the repo root, since `ROOT` auto-detects the project root) and run:

```bash
python run.py --eval-set ../../data/eval_set.json
```

VLM outputs are written to `data/vlm_results/`, one file per `(model, prompt_key, qtype)`, e.g. `qwen2.5vl_DA_findings_yn.json`. Results are saved after every 10 cases and on completion, so a disconnect from the GPU pod does not lose completed work (`evaluate.py` resumes from the `.partial` checkpoint).

---

# 3. Evaluation

Scoring is intentionally **not** part of either generation pipeline. Both `code/kneecot-h5-pipeline-llm/run.py` and `code/kneecot-h5-pipeline-vlm/evaluate.py` only write raw per-question records. All scoring lives in `code/analysis/`, the single shared layer both lines are judged by — see `Running_Guidelines.md` for the full step-by-step command sequence.

## 3.1 Quantitative yes/no evaluation

Yes/no questions form the main quantitative evaluation backbone. `code/analysis/compare.py` contains the one shared `parse_yes_no()` parser used for both lines:

1. Look for the literal word `Yes`/`No` (case-insensitive) after the `【答案】` marker.
2. If not found there, look for it anywhere in the raw text.
3. If neither is found, return `None` — the answer is recorded as missing/unparsable rather than guessed.

An earlier version of this parser also fell back to the last Chinese `是`/`否` character in the text. That fallback was removed: `是` is an extremely common Chinese function word ("这是…", "不一定是…") unrelated to the model's actual answer, so on long CoT outputs it was disguising a genuinely unparsable response as a seemingly valid but essentially random Yes/No, instead of honestly reporting a parse failure.

For each model and prompt condition, `compare.py` reports:

```text
accuracy
McNemar's test vs. the paired condition (exact binomial if < 25 discordant pairs, chi-square with continuity correction otherwise)
```

The main comparisons (see `summary_2x2_comparison.json`) are:

```text
RQ1: LLM direct vs. LLM CoT
RQ2: VLM direct vs. VLM CoT
RQ3: LLM vs. VLM under the matched (_findings) condition, and image-only vs. image+findings (vlm_findings_ablation.py)
```

## 3.2 Inference-question evaluation

Inference questions are evaluated separately from yes/no questions because they require a diagnostic conclusion plus supporting rationale rather than a single binary label. This is **not** rule-based string/keyword matching — both lines' raw outputs are scored by a local LLM-as-judge:

1. `code/analysis/judge.py` sends each inference record, together with a question-aware rubric (`code/analysis/inference_rubric_for_LLM_judge.json`), to a locally-served judge model (`qwen2.5:32b` via Ollama). The judge must be a different, stronger model than any model being evaluated, so it is never grading its own output.
2. The judge returns a structured verdict per item: `correct` / `incorrect` / `unclear`, plus the matched rubric rule, the extracted ground-truth core and model conclusion, and a `needs_manual_review` flag.
3. All `unclear` and `incorrect` verdicts, plus a random ~15–20% sample of judge-marked `correct` cases, are written to a manual-review file. Two reviewers independently re-label a shared subset to measure human-human and human-judge agreement, and any judge errors found are corrected directly in `judged_inference_llm.json` / `judged_inference_vlm.json` before `compare.py` is run.
4. `compare.py` then reads those corrected judged files and scores inference questions exactly like yes/no questions (accuracy + McNemar), keeping both question types under one comparison script.

See `Running_Guidelines.md` §4.3 for the exact commands and the manual-review checklist.

## 3.3 RQ3 image-only ablation (does the VLM need the MR findings text?)

This ablation asks: if the VLM only sees the MRI image, with no MR findings text at all, does it still answer as well, and does CoT still help? It is implemented and has already been run — it is **not** a planned/future experiment.

**No separate generation run was needed for this.** `config.py`'s `CONDITIONS` list always includes all four prompt keys — `DA`, `CoT`, `DA_findings`, `CoT_findings` — so every time `evaluate.py` runs, it produces all four conditions' raw outputs in the same pass, not just the `_findings` pair used for the main LLM-vs-VLM comparison. The image-only files (`qwen2.5vl_DA_yn.json`, `qwen2.5vl_DA_inference.json`, `qwen2.5vl_CoT_yn.json`, `qwen2.5vl_CoT_inference.json`) were already sitting in `data/vlm_results/` as a byproduct of the main 2×2 run.

The image-only prompts are the plain `DA`/`CoT` templates shown in §2.2 above ("Image-only VLM — Direct Answer (DA)" / "Image-only VLM — Structured 4-Step CoT"). These are not a separate, older version of the prompts: when the `【答案】`-format fixes were made (forcing a bare `Yes`/`No` line, and forcing inference answers to give a ≥20-character conclusion with supporting evidence instead of a single word), they were applied to **both** the `_findings` templates and these plain image-only templates in the same edit to `prompts.py`. So the ablation results already reflect the same prompt-format fix as the main comparison — no extra prompt work was needed before running it.

What was actually left to do was combine, judge, and compare, reusing the existing shared scoring code:

```bash
# 1. combine the 4 image-only result files into one
python -c "
import json
files = ['data/vlm_results/qwen2.5vl_DA_yn.json', 'data/vlm_results/qwen2.5vl_DA_inference.json',
         'data/vlm_results/qwen2.5vl_CoT_yn.json', 'data/vlm_results/qwen2.5vl_CoT_inference.json']
combined = []
for fp in files:
    combined.extend(json.load(open(fp, encoding='utf-8')))
json.dump(combined, open('data/vlm_results/combined_ablation_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('combined:', len(combined), 'records')
"

# 2. judge the image-only inference outputs (separate judged file -- does not
#    touch/overwrite judged_inference_vlm.json from the main _findings run)
python code/analysis/judge.py \
  --input data/vlm_results/qwen2.5vl_DA_inference.json data/vlm_results/qwen2.5vl_CoT_inference.json \
  --rubric code/analysis/inference_rubric_for_LLM_judge.json \
  --model qwen2.5:32b \
  --output judged_inference_vlm_ablation.jsonl \
  --json-output judged_inference_vlm_ablation.json \
  --review-output manual_review_vlm_ablation.jsonl

# 3a. compare.py, pointed at the image-only files via --vlm_prompt_direct/--vlm_prompt_cot
#     (Running_Guidelines.md documents this exact override) -- this reruns the same
#     2x2-style accuracy + McNemar table, but with the VLM columns now image-only
python code/analysis/compare.py \
  --eval_set data/eval_set.json \
  --llm_results code/kneecot-h5-pipeline-llm/results/raw_results.json \
  --vlm_results data/vlm_results/combined_ablation_results.json \
  --vlm_prompt_direct DA --vlm_prompt_cot CoT \
  --judged_llm judged_inference_llm.json \
  --judged_vlm judged_inference_vlm_ablation.json \
  --out_dir results/compare_out_vlm_ablation

# 3b. vlm_findings_ablation.py -- a more direct, VLM-only version of the same question:
#     does adding MR findings text help, within each prompt style? (DA vs DA_findings,
#     CoT vs CoT_findings). Reuses compare.py's scoring/McNemar code, so the yes/no
#     parsing and inference judging are identical to the main analysis.
python code/analysis/vlm_findings_ablation.py \
  --eval_set data/eval_set.json \
  --findings_results data/vlm_results/combined_findings_results.json \
  --ablation_results data/vlm_results/combined_ablation_results.json \
  --judged_findings judged_inference_vlm.json \
  --judged_ablation judged_inference_vlm_ablation.json \
  --out_dir results/compare_out
```

Step 3a and 3b answer slightly different questions and both are kept: 3a re-runs the full LLM-vs-VLM comparison with the VLM swapped to image-only, so it can still be read against the LLM line; 3b is the narrower, VLM-internal comparison (image-only vs. image+text, holding the prompt style fixed) and is what actually produced `results/compare_out/summary_vlm_ablation.json`. Neither step touches or overwrites `summary_2x2_comparison.json` from the main `_findings` comparison — they write to separate output paths, so the matched-condition result and the ablation result are two independent analyses that coexist in `results/`.

## 3.4 Research questions

This repository supports the following H5 research questions:

| Research question | Pipeline evidence |
|---|---|
| Does CoT improve knee MRI VQA performance? | DA vs. CoT within LLM and VLM (RQ1, RQ2 in `summary_2x2_comparison.json`) |
| Does CoT yield a larger gain for VLM than LLM? | Matched LLM–VLM comparison on the same `eval_set.json`, `_findings` condition |
| When is visual input necessary? | VLM vs. LLM per-item breakdown (`rq3_yesno.csv` / `rq3_inference.csv`, labelled `vision_necessary` / `text_better` / `both_correct_text_sufficient` / `both_wrong`), plus the image-only ablation in §3.3 (`results/compare_out/summary_vlm_ablation.json`, `vlm_findings_ablation.py`) |

---

# 4. Reproducibility

The experiment is designed to be reproducible through:

1. A single fixed evaluation split (`data/eval_set.json`), generated once with seed 42 and shared by both lines.
2. Matched LLM and VLM evaluation entries — same `(case_id, question)` pairs in every condition.
3. Greedy, deterministic decoding for both lines (`do_sample=False` for the LLM, `temperature=0.0` for the VLM).
4. Saved raw outputs for every condition, generated independently of scoring.
5. A single shared scoring/comparison script (`code/analysis/compare.py`) instead of two independently-evolving copies.
6. Unit tests for data loading and answer parsing (`code/kneecot-h5-pipeline-llm/tests/`).

The expected command order (see `Running_Guidelines.md` for the full version with all flags):

```bash
# 1. build the shared evaluation set (VLM side is the single source of truth)
cd code/kneecot-h5-pipeline-vlm && python build_eval_set.py --n-eval 200 && cd ../..

# 2. generate raw outputs (VLM line, then LLM line)
cd code/kneecot-h5-pipeline-vlm && python run.py --eval-set ../../data/eval_set.json --eval-only && cd ../..
cd code/kneecot-h5-pipeline-llm && python run.py --eval_set ../../data/eval_set.json --out_dir results && cd ../..

# 3. score: shared yes/no parser + local LLM-as-judge for inference (+ manual review)
python code/analysis/judge.py --input code/kneecot-h5-pipeline-llm/results/raw_results.json \
  --rubric code/analysis/inference_rubric_for_LLM_judge.json --json-output judged_inference_llm.json
python code/analysis/judge.py --input data/vlm_results/qwen2.5vl_*_findings_inference.json \
  --rubric code/analysis/inference_rubric_for_LLM_judge.json --json-output judged_inference_vlm.json

# 4. compare
python code/analysis/compare.py \
  --eval_set data/eval_set.json \
  --llm_results code/kneecot-h5-pipeline-llm/results/raw_results.json \
  --vlm_results data/vlm_results/combined_findings_results.json \
  --judged_llm judged_inference_llm.json --judged_vlm judged_inference_vlm.json \
  --out_dir results/compare_out
```
