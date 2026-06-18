# VLM Evaluation Pipeline — KneeCoT (H5)

This pipeline produces the VLM half of the matched comparison; the text-only LLM pipeline supplies the other half. Together they address RQ2 — whether CoT yields a larger gain for the VLM than for the text-only LLM — and RQ3 — for which cases visual input is necessary and when text-based reasoning suffices.

---

## Model Architecture

### Input

In the Round 2 configuration the VLM receives two inputs: the MRI slice images (from the .nii volume) and the free-text MR findings (MR表现) from the JSON — the same text evidence given to the text-only LLM, so the only difference between the two pipelines is the image. From the JSON, only MR表现 is used as text; the diagnostic opinion (诊断意见) and the structured label set (标签) are deliberately excluded, because both state the answer in plain language and would constitute answer leakage. The loader reads 顺序编号, MR表现, 问答数据.qa_pairs (and 检查方法/MR表现 for the knee-only filter), and never reads the leakage-prone fields. 

After loading, the QA pairs go through a strict filter. Yes/no pairs are kept only if the answer begins with a clean Yes/No and the question carries one of the five markers (是否, 有无, 是不是, 有没有, 能否) — these are the quantitative backbone. Inference (推理) pairs are also retained; their open-ended "conclusion + rationale" answers are scored separately (see Evaluation). Non-knee cases are removed. This filtering produces a clean pool of 2,746 genuine binary QA pairs across 203 cases.

Both pipelines consume the same frozen evaluation set, eval_set.json, produced by build_eval_set.py. The VLM pipeline does not re-sample: it reads the shared item list and is evaluated on exactly the same (case_id, question_id) entries as the text-only LLM. This guarantees a matched comparison by construction — any LLM-vs-VLM difference comes from modality, not from a different question sample. The frozen set is built once: from the genuine binary QA pool, sampling is stratified by ground-truth label with random.seed(42), drawing 25 Yes and 25 No (N_EVAL = 50) with at most one Yes and one No per case_id. Inference (推理) items are carried in the same file. Round 3 expands the set to 500–1,000 examples via N_EVAL = None.

The visual input comes from 3D `.nii` MRI volumes. Each volume is loaded with `nibabel`, and the sagittal axis is auto-detected as the dimension with the largest extent (rather than assuming axis 2, which would be wrong for some scans). Two center-biased slice indices are computed as `[ D//3, D//2]`, concentrating on the central half of the volume where structures like the ACL and PCL are actually visible. Each slice is contrast-enhanced with CLAHE (`clipLimit=2.0`, `tileGridSize=8×8`) before being encoded as a base64 PNG and passed to the model. Peripheral structures may be missed by this two-slice strategy — this is acknowledged as a paper limitation, meaning the visual contribution measured here is a lower bound.

We experimented with larger slice counts; however, the token limitations imposed by the VLM’s context window restricted the visual input to two center-biased slices per volume, further reinforcing that the measured visual contribution represents a conservative lower bound.

---

### Processing

#### Text Section — Prompts

Two types of prompts(Direct prompting and Chain-of-Thought prompting) are used, both written entirely in Chinese to match the language of the dataset. Using English prompts with Chinese data causes models to mix languages during long CoT generations, making the `【答案】` answer tag unreachable and inflating the UNCLEAR rate artificially.

**Image + MR findings VLM — Direct Answer (DA):**

```
你是一位资深骨骼肌肉放射科医生。下面给出一组膝关节 MR 图像、对应的 MR 表现文字和一个相关问题。
请结合图像与 MR 表现直接回答问题，不要写出推理过程。
- 若为是非类问题：请在【答案】后只回答 Yes 或 No。
- 若为推理类问题：请在【答案】后给出明确结论及简要依据。
【MR 表现】{findings}
【问题】{question}
【答案】
```

**Image + MR findings VLM — Structured 4-Step CoT:**

```
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
The two templates below drop the MR findings and present the image alone. They are not used in the Round 2 quantitative run; they are reserved for the Round 3 input ablation (image-only vs. image + findings, VLM only) because they require different codes for comparison.

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

The CoT template makes the model first describe the relevant anatomical structures — reading the MRI slices and cross-checking them against the MR findings — then judge whether each is abnormal and give a reason, before committing to the final answer.

The answer is extracted by a shared `parse_yes_no()` function with a three-priority cascade: (1) look for Yes/No after `【答案】`, (2) search for the English words `yes` or `no` case-insensitively, and (3) fall back to finding the last occurrence of the Chinese characters 是 or 否. A result that fails all three is recorded as `None` rather than as a fabricated label. The parse_yes_no() cascade applies only to yes/no items. For inference (推理) questions, the model's final conclusion is taken from after the 【答案】 marker and compared with the ground-truth conclusion in compare.py (rule-based matching of the key diagnostic category, with low-confidence items flagged for review). This is why both templates carry the "若为推理类问题…结论及依据" branch.

#### VLM Section — Image Processing and Model

The visual input comes from the 3D .nii MRI volumes. Each volume is loaded with nibabel, and the sagittal axis is auto-detected as the dimension with the largest extent (rather than assuming axis 2, which would be incorrect for some scans). Two center-biased slice indices are computed as [D//3, D//2], which fall in the central region of the volume where structures such as the ACL and PCL are typically visible. Each slice is contrast-enhanced with CLAHE (clipLimit=2.0, tileGridSize=8×8), then encoded as a base64 PNG and passed to the model. Restricting the input to these two fixed central slices keeps the number of visual tokens within the model's context budget; peripheral structures may be missed by this two-slice strategy, so the visual contribution measured here should be interpreted as a lower bound — this is acknowledged as a paper limitation.

**Greedy decoding** (`temperature=0.0`) is used for all inference calls. This ensures that for the same input, the model always produces the same output — a prerequisite for reproducibility and for fair comparison between DA and CoT conditions. The Ollama default temperature of 0.8 would introduce stochasticity and make runs non-comparable, so it is explicitly overridden. Context length is set to `num_ctx=4096` tokens, and generation is capped at `num_predict=2400` tokens — enough headroom for a full four-step CoT response without truncation.

---

### Optimization

**H5 is inference-only. There is no training, no loss function, and no optimizer.** This is the most important architectural distinction between the H5 pipeline and the default model setup described in the course slides, which assumes a training loop. H5 takes pre-trained model weights as they are — specifically Qwen2.5-VL, served locally through Ollama — and evaluates their performance directly on the KneeCoT test set. No fine-tuning, no QLoRA, no gradient updates of any kind take place during a pipeline run. The weights are frozen at the values released by the model authors. Any improvement in accuracy between DA and CoT therefore reflects the effect of prompt structure alone, not any adaptation to the KneeCoT domain.

This design choice was deliberate: the research questions concern what pre-trained VLMs can already do when given structured prompts and medical imaging context, without any task-specific supervision.

---

## Model Implementation
The pipeline is implemented as a single Google Colab notebook (`VLM.ipynb`) and runs top-to-bottom without any manual steps between cells. The entire environment is managed through Colab's runtime, which provides either a **T4 GPU** or a **v6e-1 TPU** depending on the session assigned. The Ollama server is installed at runtime via `curl` and started as a background subprocess before model weights are pulled with `ollama pull`.

The core Python dependencies are:

| Library | Role |
|---|---|
| `ollama` | Multimodal inference via the Ollama REST API |
| `nibabel` | Loading `.nii` MRI volumes |
| `Pillow` + `opencv-python` | Slice extraction, CLAHE contrast enhancement, base64 encoding |
| `pandas` | Results aggregation and CSV export |
| `matplotlib` | RQ2 and RQ3 visualisation |
| `numpy` | Array operations on volumetric data |

The primary model is **Qwen2.5-VL**, chosen because it shares a base with the text-only Qwen2.5-7B used in the LLM pipeline — making the VLM-vs-LLM comparison clean..A secondary model, **MiniCPM-V**, is also enabled by default for multi-model comparison. Results for each model and prompt type are saved to a separate JSON file in `vlm_results/` immediately after each run, so a Colab session crash does not lose completed evaluations.

To reproduce a full run, install dependencies with:

```bash
pip install ollama nibabel pillow pandas matplotlib opencv-python-headless
```

Then set `ROOT` in Cell 1 to point to the KneeCoT data folder on Google Drive, and run all cells in order.
