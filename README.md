# Does Vision Help Reason? — H5 LLM and VLM Evaluation Pipeline

This repository contains the **H5 experiment** for a knee MRI Chain-of-Thought study on the KneeCoT dataset. The project compares **Direct Answer (DA)** prompting and **Chain-of-Thought (CoT)** prompting under two matched settings:

1. **Text-only LLM pipeline**: uses only the free-text MR findings (`MR表现`).
2. **Vision-language VLM pipeline**: uses the same MR findings plus knee MRI slice images extracted from `.nii` volumes.

The goal is to evaluate whether structured CoT prompting improves diagnostic question answering, and whether visual input provides additional value beyond text-only MR findings.

---

## What this repository includes

This repository contains two parallel pipelines for the H5 experiment: a text-only LLM pipeline and a multimodal VLM pipeline. They are kept in separate folders because they use different input modalities, runtime environments, and output formats, but they are evaluated under the same experimental goal.

## Repository Structure

This repository contains two parallel pipelines for the H5 experiment: a text-only LLM pipeline and a multimodal VLM pipeline. They are kept in separate folders because they use different input modalities, runtime environments, and output formats, but they are evaluated under the same experimental goal.

## Pipeline Overview

| Component | Text-only LLM pipeline | Multimodal VLM pipeline |
|---|---|---|
| Folder | `kneecot-h5-pipeline-llm/` | `kneecot-h5-pipeline-vlm/` |
| Main entry point | `run.py` | `VLM.ipynb` |
| Input modality | MR findings text only (`MR表现`) | MRI slices + MR findings text |
| Main output folder | `results/` | `vlm_results/` |
| Prompting conditions | Direct Answer and CoT | Direct Answer and CoT |
| Purpose | Measures text-only reasoning performance | Measures whether visual input improves reasoning |

## Pipeline In Detail

```text
run_pipeline.py/                    # use it to run the whole project
code_new/
├── analysis/
│   ├── compare.py/
├── kneecot-h5-pipeline-llm/
│   ├── data/
│   │   ├── sample/                 # Small de-identified sample cases
│   │   └── cases/                  # Full KneeCoT JSON annotations; not committed
│   ├── results/                    # LLM output files and evaluation results
│   ├── src/
│   │   ├── build_eval_set.py       # Build eval sets
│   │   ├── preprocessing.py        # Load JSON, filter knee cases, build/evaluate QA items
│   │   ├── prompts.py              # Direct and Chain-of-Thought prompt templates
│   │   ├── inference.py            # Text-only LLM inference
│   │   └── evaluation.py           # Yes/No parsing, accuracy, and McNemar test
│   ├── tests/                      # Sanity tests for preprocessing and parsing
│   ├── run.py                      # End-to-end LLM runner
│   ├── requirements.txt            # Python dependencies for the LLM pipeline
│
└── kneecot-h5-pipeline-vlm/
    ├── vlm_results/                # VLM output files and evaluation results
    ├── VLM.ipynb                   # Google Colab notebook for the VLM pipeline
    ├── requirements.txt            # Python dependencies for the VLM pipeline
```
*** Everything else is just for reference within the group and won’t be included in the final project submission.

The LLM pipeline is implemented as a script-based Python project because it only processes text annotations and can be run through `run.py`. The VLM pipeline is implemented as a Google Colab notebook because it requires MRI slice loading, image preprocessing, multimodal model serving, and GPU/TPU runtime setup. Therefore, the two folders are not identical in structure, but both contain the required components: source code, dependency specification, usage instructions, and output directories.

## Data access

The project uses the [KneeCoT dataset](https://huggingface.co/datasets/YiHui0124/KneeCoT), which requires a Hugging Face account and access approval from the dataset authors.

Do **not** commit the full dataset to this repository. The full KneeCoT data are access-restricted and may include very large MRI volumes. This repository should only contain small de-identified samples or synthetic examples under `data/sample/`.

For the LLM pipeline, only JSON annotations are required. For the VLM pipeline, both JSON annotations and corresponding `.nii` MRI volumes are required.

---

# 1. Text-only LLM Pipeline

## 1.1 Model Architecture

### Input

The text-only LLM receives only the free-text MR findings field (`MR表现`) from each KneeCoT JSON case. The diagnostic opinion (`诊断意见`) and structured labels (`标签`) are deliberately excluded because they state the answer directly and would cause answer leakage.

The preprocessing module loads the case identifier, MR findings, and QA pairs. Non-knee cases are removed. Yes/no questions are retained only when their answer begins with a clean `Yes` or `No` and the question contains one of the five binary-question markers:

```text
是否, 有无, 是不是, 有没有, 能否
```

Inference questions (`推理`) are also retained, but they are scored separately because they require a conclusion plus rationale rather than a single binary label.

For the matched LLM–VLM comparison, both pipelines use the same frozen evaluation set, `eval_set.json`, generated with `random.seed(42)`. In the Round 2 prototype, the set contains 50 binary QA examples, stratified as 25 `Yes` and 25 `No`, with at most one `Yes` and one `No` example per case. Round 3 can expand the evaluation set to 500–1,000 examples by setting `N_EVAL = None`.

### Model

The LLM pipeline uses:

```text
Qwen/Qwen2.5-7B-Instruct
```

This model was selected because it is open, has strong Chinese-language capability, and can run locally without sending medical data to an external API. In the prototype setup, 4-bit loading allows the 7B model to fit on a free Google Colab T4 GPU.

---

## 1.2 Processing

### Prompting conditions

The LLM is evaluated under two prompting conditions:

1. **Direct Answer (DA)**: the model answers immediately without writing reasoning.
2. **Structured Chain-of-Thought (CoT)**: the model follows a four-step diagnostic reasoning template before giving the final answer.

All prompts are written in Chinese to match the dataset language. Using English prompts with Chinese MR findings can make the model mix languages during long generations, which may reduce the reliability of answer extraction.

### Text-only LLM — Direct Answer prompt

```text
你是一位资深骨骼肌肉放射科医生。下面给出一段膝关节 MR 表现文字和一个相关问题。
请根据 MR 表现直接回答问题，不要写出推理过程。
- 若为是非类问题：请在【答案】后只回答 Yes 或 No。
- 若为推理类问题：请在【答案】后给出明确结论及简要依据。
【MR 表现】{findings}
【问题】{question}
【答案】
```

### Text-only LLM — Structured 4-Step CoT prompt

```text
你是一位资深骨骼肌肉放射科医生。请根据下面给出的膝关节 MR 表现文字，按以下四个步骤进行系统推理，然后回答问题。

步骤一 系统性观察（Systematic Observation）
根据 MR 表现文字，按解剖部位（半月板、韧带、骨与软骨、关节腔与滑膜、其他结构如脂肪垫与软组织）逐项梳理关键征象。

步骤二 解读与核对（Interpretation and Verification）
对每条征象判断正常还是异常，说明该改变通常提示什么，并核对前后一致性。

步骤三 解剖结构分析（Anatomical Structure Analysis）
3.1 半月板：形态是否完整、高信号是否达关节面、损伤程度。
3.2 韧带：前/后交叉韧带、内/外侧副韧带的连续性与信号。
3.3 骨与软骨：骨髓信号、关节面软骨是否光整。
3.4 关节腔与滑膜：有无积液、滑膜情况。
3.5 其他结构：髌下脂肪垫、关节周围软组织。

步骤四 诊断推理与核对（Diagnostic Reasoning and Verification）
综合分析推导结论；如适用简要排除主要鉴别诊断；自检结论是否由证据支持。

最后在【答案】后给出明确回答：是非类答 Yes/No；推理类给结论及依据。
【MR 表现】{findings}
【问题】{question}
请依次完成步骤一至步骤四，最后给出【答案】。
```

### Answer parsing

For yes/no questions, the shared `parse_yes_no()` function uses a three-level cascade:

1. Search for `Yes` or `No` after the `【答案】` marker.
2. Search for the English words `yes` or `no` case-insensitively.
3. Fall back to the last occurrence of the Chinese characters `是` or `否`.

If all three steps fail, the answer is recorded as `None` rather than forcing a fabricated label.

For inference questions (`推理`), the final conclusion is extracted from the text after `【答案】` and compared with the ground-truth conclusion using rule-based matching of the key diagnostic category. Low-confidence matches are flagged for review.

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

The LLM pipeline can be run from either a Python script or a Google Colab notebook.

Core dependencies include:

| Library | Role |
|---|---|
| `transformers` | Loading and running Qwen2.5-7B-Instruct |
| `torch` | Model execution |
| `bitsandbytes` | 4-bit loading for Colab T4 GPU |
| `accelerate` | Device mapping and efficient inference |
| `pandas` | Result aggregation and CSV/JSON export |
| `scipy` | McNemar test and statistical analysis |

Example setup:

```bash
pip install -r requirements.txt
```

Example Colab run:

```python
!git clone https://github.com/<your-org>/<your-repo>.git
%cd <your-repo>
!pip install -q -r requirements.txt

from huggingface_hub import login
login()  # paste your Hugging Face token if needed for dataset access

!python run.py --data_dir data/cases --sample_size 50
```

Example local or HPC run:

```bash
python run.py --data_dir /path/to/cases --sample_size 0
```

To test the pipeline without a GPU or model call:

```bash
python run.py --data_dir data/sample --sample_size 0 --mock
python tests/test_pipeline.py
```

LLM outputs are written to `results/`, including:

```text
raw_results.json
yes_no_accuracy.json
mcnemar.json
inference_outputs.json
```

---

# 2. Vision-Language VLM Pipeline

## 2.1 Model Architecture

### Input

The VLM receives two inputs:

1. MRI slice images extracted from the `.nii` volume.
2. The same free-text MR findings (`MR表现`) used by the text-only LLM.

This makes the comparison matched: the LLM and VLM receive identical text evidence, and the only additional input to the VLM is the image.

As in the LLM pipeline, the diagnostic opinion (`诊断意见`) and structured labels (`标签`) are excluded to avoid answer leakage. The loader reads only the case identifier, MR findings, QA pairs, and fields required for knee-only filtering.

The VLM pipeline does not re-sample evaluation examples. It reads the same frozen `eval_set.json` as the text-only LLM pipeline, so all DA and CoT conditions are evaluated on the same `(case_id, question_id)` entries.

### Image processing

The visual input comes from 3D `.nii` MRI volumes.

Each volume is loaded with `nibabel`. The sagittal axis is auto-detected as the
dimension with the largest extent, rather than assuming a fixed axis. This avoids
incorrect slicing for scans whose sagittal dimension is not stored as axis 2.

Two center-biased slice indices are computed:

```text
[D//3, D//2]
```

These slices fall in the central region of the volume, where structures such as
the ACL and PCL are typically visible.

Each selected slice is processed as follows:

- contrast enhancement with CLAHE;
- `clipLimit=2.0`;
- `tileGridSize=8×8`;
- conversion to a base64-encoded PNG;
- input to the VLM together with the text prompt.

Restricting the input to two fixed central slices keeps the number of visual
tokens within the model's context budget.

However, peripheral structures may be missed by this two-slice strategy.
Therefore, the visual contribution measured in this pipeline should be
interpreted as a lower bound. This is acknowledged as a paper limitation.



### Model

The primary VLM is:

```text
Qwen2.5-VL
```

It was selected because it shares a model family with the text-only Qwen2.5-7B LLM, making the LLM–VLM comparison cleaner.

A secondary model, `MiniCPM-V`, can also be enabled for multi-model comparison.

---

## 2.2 Processing

### Prompting conditions

The VLM is evaluated under the same two prompting conditions:

1. **Direct Answer (DA)**
2. **Structured 4-Step Chain-of-Thought (CoT)**

The prompts are written in Chinese and use the same final `【答案】` marker as the LLM pipeline.

### Image + MR findings VLM — Direct Answer prompt

```text
你是一位资深骨骼肌肉放射科医生。下面给出一组膝关节 MR 图像、对应的 MR 表现文字和一个相关问题。
请结合图像与 MR 表现直接回答问题，不要写出推理过程。
- 若为是非类问题：请在【答案】后只回答 Yes 或 No。
- 若为推理类问题：请在【答案】后给出明确结论及简要依据。
【MR 表现】{findings}
【问题】{question}
【答案】
```

### Image + MR findings VLM — Structured 4-Step CoT prompt

```text
你是一位资深骨骼肌肉放射科医生。请根据下面给出的膝关节 MR 图像及其对应的 MR 表现文字，按以下四个步骤进行系统推理，然后回答问题。

步骤一 系统性观察（Systematic Observation）
系统性地观察所给的 MR 图像，并对照所提供的 MR 表现文字，按解剖部位（半月板、韧带、骨与软骨、关节腔与滑膜、其他结构如脂肪垫与软组织）逐项梳理关键征象，记录信号、形态与连续性。

步骤二 解读与核对（Interpretation and Verification）
对每条征象判断正常还是异常，说明该改变通常提示什么（如 T2WI 高信号提示水肿/损伤），核对前后一致性。

步骤三 解剖结构分析（Anatomical Structure Analysis）
3.1 半月板：形态是否完整、高信号是否达关节面、损伤程度。
3.2 韧带：前/后交叉韧带、内/外侧副韧带的连续性与信号。
3.3 骨与软骨：骨髓信号（水肿/挫伤）、关节面软骨是否光整。
3.4 关节腔与滑膜：有无积液、滑膜情况。
3.5 其他结构：髌下脂肪垫、关节周围软组织。

步骤四 诊断推理与核对（Diagnostic Reasoning and Verification）
综合分析推导结论；如适用简要排除主要鉴别诊断；自检结论是否由证据支持。

最后在【答案】后给出明确回答：是非类答 Yes/No；推理类给结论及依据。
【MR 表现】{findings}
【问题】{question}
请依次完成步骤一至步骤四，最后给出【答案】。
```

### Planned Round 3 input ablation

For Round 3, the VLM can also be tested under an image-only condition by removing `MR表现` from the prompt. This supports an ablation study comparing:

```text
image only
vs.
image + MR findings
```

This ablation is reserved for the VLM pipeline and should be evaluated separately from the matched Round 2 LLM–VLM comparison.

**Round 3 Ablation (Planned): Image-only VLM — Direct Answer (DA):**

```
你是一位资深骨骼肌肉放射科医生。下面给出一组膝关节 MR 图像和一个相关问题。
请根据图像直接回答问题，不要写出推理过程。
- 若为是非类问题：请在【答案】后只回答 Yes 或 No。
- 若为推理类问题：请在【答案】后给出明确结论及简要依据。
【问题】{question}
【答案】
```
**Round 3 Ablation (Planned): Image-only VLM — Structured 4-Step CoT:**

```
你是一位资深骨骼肌肉放射科医生。请根据下面给出的膝关节 MR 图像，按以下四个步骤进行系统推理，然后回答问题。

步骤一 系统性观察（Systematic Observation）
系统性地观察所给的 MR 图像，按解剖部位（半月板、韧带、骨与软骨、关节腔与滑膜、其他结构如脂肪垫与软组织）逐项描述你在图像中观察到的关键征象，记录信号、形态与连续性。

步骤二 解读与核对（Interpretation and Verification）
对每条征象判断正常还是异常，说明该改变通常提示什么（如 T2WI 高信号提示水肿/损伤），核对前后一致性。

步骤三 解剖结构分析（Anatomical Structure Analysis）
3.1 半月板：形态是否完整、高信号是否达关节面、损伤程度。
3.2 韧带：前/后交叉韧带、内/外侧副韧带的连续性与信号。
3.3 骨与软骨：骨髓信号（水肿/挫伤）、关节面软骨是否光整。
3.4 关节腔与滑膜：有无积液、滑膜情况。
3.5 其他结构：髌下脂肪垫、关节周围软组织。

步骤四 诊断推理与核对（Diagnostic Reasoning and Verification）
综合分析推导结论；如适用简要排除主要鉴别诊断；自检结论是否由证据支持。

最后在【答案】后给出明确回答：是非类答 Yes/No；推理类给结论及依据。
【问题】{question}
请依次完成步骤一至步骤四，最后给出【答案】。
```

---

## 2.3 Optimization

The H5 VLM pipeline is also **inference-only**. There is no training, no loss function, no optimizer, no fine-tuning, no QLoRA, and no gradient update.

The model is evaluated using frozen pretrained weights served locally through Ollama.

All inference calls use greedy decoding:

```text
temperature = 0.0
num_ctx = 4096
num_predict = 2400
```

This setup ensures deterministic outputs and gives enough generation headroom for the four-step CoT response without truncation.

Any improvement between DA and CoT in the VLM pipeline therefore reflects the effect of prompt structure alone, not model adaptation to the KneeCoT domain.

---

## 2.4 Model Implementation

The VLM pipeline is implemented as a Google Colab notebook:

```text
VLM.ipynb
```

The notebook runs top-to-bottom. The Colab runtime may provide a T4 GPU or v6e-1 TPU depending on session availability. Ollama is installed at runtime via `curl`, launched as a background subprocess, and then used to pull and serve model weights.

Core dependencies include:

| Library | Role |
|---|---|
| `ollama` | Multimodal inference through the Ollama REST API |
| `nibabel` | Loading `.nii` MRI volumes |
| `Pillow` | Image conversion and PNG encoding |
| `opencv-python-headless` | CLAHE contrast enhancement |
| `pandas` | Results aggregation and CSV/JSON export |
| `matplotlib` | RQ2 and RQ3 visualization |
| `numpy` | Array operations on volumetric MRI data |

Install dependencies with:

```bash
pip install ollama nibabel pillow pandas matplotlib opencv-python-headless numpy
```

Then set `ROOT` in the first notebook cell to the KneeCoT data folder on Google Drive and run all cells in order.

VLM outputs are written to `vlm_results/`. Results for each model and prompt type are saved immediately after each run so that completed evaluations are not lost if a Colab session disconnects.

---

# 3. Evaluation

## 3.1 Quantitative yes/no evaluation

Yes/no questions form the main quantitative evaluation backbone. For each model and prompt condition, the pipeline reports:

```text
accuracy
number of correct predictions
number of incorrect predictions
number of unclear / unparsable outputs
```

The main comparison is:

```text
LLM direct vs. LLM CoT
VLM direct vs. VLM CoT
LLM vs. VLM under matched prompt settings
```

McNemar's test is used to compare paired predictions between prompting conditions.

## 3.2 Inference-question evaluation

Inference questions are evaluated separately from yes/no questions because they require a diagnostic conclusion and supporting rationale. The pipeline extracts the final answer after `【答案】` and compares the predicted diagnostic category with the reference conclusion using rule-based matching. Ambiguous cases are flagged for manual review.

## 3.3 Research questions

This repository supports the following H5 research questions:

| Research question | Pipeline evidence |
|---|---|
| Does CoT improve knee MRI VQA performance? | DA vs. CoT within LLM and VLM |
| Does CoT yield a larger gain for VLM than LLM? | Matched LLM–VLM comparison on the same `eval_set.json` |
| When is visual input necessary? | VLM vs. LLM and planned image-only ablation |

---

# 4. Reproducibility

The experiment is designed to be reproducible through:

1. A fixed evaluation split generated with seed 42.
2. Matched LLM and VLM evaluation entries.
3. Greedy deterministic decoding.
4. Saved raw outputs for every condition.
5. Unit tests for data loading and answer parsing.

The expected command pattern is:

```bash
# build or load the shared evaluation set
python src/build_eval_set.py --data_dir data/cases --output eval_set.json --seed 42

# run LLM pipeline
python run.py --data_dir data/cases --eval_set eval_set.json --pipeline llm

# run VLM pipeline, or execute VLM.ipynb in Colab
python run.py --data_dir data/cases --eval_set eval_set.json --pipeline vlm

# evaluate outputs
python src/evaluation.py --results_dir results
python src/evaluation.py --results_dir vlm_results
```
