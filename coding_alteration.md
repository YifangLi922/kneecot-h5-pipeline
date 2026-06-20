# Coding Alterations — Round 3 Summary

This document summarizes the main code-level changes made during the Round 3 finalization of the H5 pipeline, based on the git commit history and a full audit of the current code against the previous README. It is meant as a reference for journal writing and will not be part of the final repository submission.

---

## 1. Bug fixes that directly affected result validity

These were the highest-impact changes: each one was diagnosed from an accuracy number that looked wrong, traced to a root cause in the parsing/prompting code, and fixed.

### 1.1 Yes/No parsing was producing near-chance accuracy (VLM line)

- **Symptom:** the 200-case run showed `D_vlm_cot` (VLM, CoT + findings, yes/no questions) at **47% accuracy — essentially a coin flip.**
- **Root cause:** `qwen2.5vl` almost never output the literal English word `Yes`/`No` in this condition (0/200 matched via the `【答案】` marker or a whole-text regex). The scoring code's third fallback rule then guessed an answer from the **last Chinese "是"/"否" character anywhere in the long reasoning text** — but "是" is an ordinary copula that appears throughout normal Chinese prose ("这是…", "不一定是…"), unrelated to the model's actual answer. This fallback fired for 159/200 records, silently turning "the model didn't answer in the expected format" into "a plausible-looking but essentially random Yes/No."
- **Fix:**
  - `prompts.py` (VLM line): explicitly instruct the model to answer yes/no questions with **only the literal English word `Yes`/`No` on its own line**, no Chinese 是/否 substitute, no extra text.
  - `compare.py`: **dropped the 是/否 fallback entirely.** An unparseable answer now returns `None`, is honestly counted as wrong, and is flagged for manual review — instead of a guessed label that hid the problem.
  - A stale duplicate `parse_yes_no`/`parse_inference_answer` pair that nothing in the current pipeline actually called was also removed, to stop it from being a future source of drift.
- **Consequence:** required re-running `evaluate.py` for the affected yes/no conditions before the numbers in `compare_out/` could be trusted. Inference-question numbers were unaffected, since those are scored by `judge.py`, not this parser.

### 1.2 Same parsing bug, LLM line

- The same root cause was found on the text-only LLM line via the same rule-1/2/3 breakdown: **DA had 31/200** yes/no answers resolved only by the (now-removed) 是/否 guess rule; **CoT had 98/200** resolved that way, plus **21/200 fully unparseable.**
- **Fix:** the same literal-`Yes`/`No`-only instruction was applied to `code/kneecot-h5-pipeline-llm/src/prompts.py`, keeping both lines' answer format — and therefore both lines' scoring — consistent.

### 1.3 Inference answers were too short to be substantively judged (DA conditions)

- **Symptom:** `DA_findings` averaged **6 characters per answer** (min 3, max 104) on inference questions — the model was answering with a bare word (e.g. "Yes", "三级") and no supporting evidence, even though the prompt already asked for "结论及简要依据" (a conclusion plus brief justification). Many judge-marked-incorrect cases turned out to be "no justification given" rather than a genuinely wrong conclusion.
- **Root cause:** the prompt's opening line, "不要写出推理过程" ("don't write out the reasoning process"), was being read by the model as a blanket instruction for brevity, overriding the more specific inference-answer bullet beneath it.
- **Fix:** reworded the opening line to clarify it only means "skip the step-by-step reasoning chain," and replaced the inference-answer bullet with an explicit instruction banning single-word/phrase answers and requiring a cited-evidence sentence of **at least ~20 characters**. Applied to both the LLM line and VLM line prompts. `CoT`/`CoT_findings` conditions were already unaffected, since the four-step structure already produces detailed answers.
- **Consequence:** required regenerating the DA and DA_findings inference records (VLM) and the DA-mode inference records (LLM) before `judge.py`/`compare.py` numbers for those conditions could be trusted.

---

## 2. Architecture: separating generation from scoring

Before this round, the LLM line and VLM line each implemented their own scoring inline (different field names — `yes_no` vs `yesno`, `gt_label` vs `ground_truth`, `prompt_mode` vs `prompt_key` — different yes/no parsers, and different/no inference-scoring heuristics). This meant the "comparison" between the two lines was not actually apples-to-apples. The fix was architectural, not a one-line patch:

- `run.py` (LLM line) and `evaluate.py` (VLM line) now **only produce raw, unscored per-question records** with a unified field schema (`case_id`, `question_id`/`question`, `qtype`, `prompt_key`, `raw_output`, ...).
- All scoring now lives in exactly one shared place, `code/analysis/`:
  - `compare.py` — the single shared `parse_yes_no()` parser, used by both lines, plus McNemar's test and the RQ1/RQ2/RQ3 comparison tables.
  - `judge.py` — inference-question scoring via a local LLM-as-judge.
- Inference-question scoring was redesigned from a rule-based/bigram-overlap heuristic (different per line, and not implemented at all on the VLM line) to a shared **LLM-as-judge + rubric + manual review** pipeline:
  - `judge.py` sends each inference record, plus a question-aware rubric (`inference_rubric_for_LLM_judge.json`), to a locally-served judge model (`qwen2.5:32b` via Ollama — deliberately a different, stronger model than anything being evaluated, to avoid self-grading).
  - The judge returns a structured verdict (`correct`/`incorrect`/`unclear`) plus its reasoning and a `needs_manual_review` flag.
  - All `unclear`/`incorrect` verdicts, plus a random ~15–20% sample of judge-marked `correct` cases, go into a manual-review file; corrected labels are written back before `compare.py` is run.

## 3. New ablation script: VLM image-only vs. image+findings

`code/analysis/vlm_findings_ablation.py` was added to directly compare VLM(image+findings) vs. VLM(image-only), reusing `compare.py`'s scoring/McNemar code instead of duplicating the parsing/judging logic. It pairs `DA_findings` vs. `DA` and `CoT_findings` vs. `CoT` within the VLM line only (no LLM line involved).

This ablation needed **no separate generation run**: `config.py`'s `CONDITIONS` list always includes all four prompt keys (`DA`, `CoT`, `DA_findings`, `CoT_findings`), so every `evaluate.py` run already produces the image-only raw outputs as a byproduct of the main run. The §1.3 prompt-format fix above was also applied to the image-only templates in the same edit, so the ablation results already reflect that fix — no extra prompt work was needed before scoring it. What was left was purely a scoring/comparison step: combine the four image-only result files, run `judge.py` on the inference half, then run either `compare.py --vlm_prompt_direct DA --vlm_prompt_cot CoT` or `vlm_findings_ablation.py` directly — both write to separate output paths and do not overwrite the main `_findings` comparison results.

## 4. VLM decoding was not actually greedy/deterministic

- **Symptom found during this round's documentation review:** `evaluate.py`'s `ollama.chat()` calls passed no `options` at all, meaning the Round 3 generation run used Ollama's default sampling parameters, not greedy decoding — even though the LLM line uses strict `do_sample=False` and the original prototype notebook (`VLM.ipynb`) did pass `temperature=0.0`. This was an oversight carried over when the notebook prototype was rewritten as the production `evaluate.py` script.
- **Fix:** added `OLLAMA_OPTIONS = {"temperature": 0.0, "num_ctx": 4096, "num_predict": 2400}` and passed it to both `ollama.chat()` call sites in `evaluate.py`, matching the LLM line's greedy decoding. This keeps the pipeline correct for any rerun before the final paper; it does not retroactively change the Round 3 numbers already collected under default sampling.

## 5. Image preprocessing redesign (VLM line)

The VLM line's slice-extraction strategy changed from an early two-slice design to the current approach:

| | Earlier design | Current (`preprocessing.py`) |
|---|---|---|
| Slices used | 2 fixed center-biased slices (`[D//3, D//2]`) | 10 slices, evenly spaced across the central 10%–90% of the depth |
| Contrast handling | CLAHE (`clipLimit=2.0`, `tileGridSize=8×8`) | Percentile clip (1st/99th) + scale to 8-bit |
| Image sent to the VLM | Slices sent individually | All 10 slices stitched into one 2×5 grid PNG per case |

The current design trades a fixed two-slice view for broader depth coverage at a similar visual-token cost (one stitched image instead of many separate ones). `opencv-python` remains listed in `requirements.txt` from the earlier CLAHE-based design but is no longer imported anywhere in the current code.

## 6. Repository reorganization (non-functional, for final submission cleanliness)

- `code_new/` renamed to `code/`; all path references in docs and scripts updated accordingly.
- `judge.py` and `inference_rubric_for_LLM_judge.json` moved into `code/analysis/`, alongside `compare.py` — these three files are the complete shared scoring layer described in §2, and now live together.
- Internal dev notes (`coding problems and solutions.md`) moved into `notes/`.
- `results/compare_out/summary.json` renamed to `results/compare_out/summary_2x2_comparison.json` (and the writer in `compare.py` updated to match) so the filename reflects its content and is unambiguous next to `summary_vlm_ablation.json`.
- Documented that the final full-scale run used a rented RunPod GPU pod (NVIDIA RTX PRO 6000), not Google Colab or a university SLURM/HPC cluster as earlier drafts assumed.

## 7. Documentation catch-up (README)

The README had fallen behind the actual code across several rounds of fixes above. It was rewritten to match the current code rather than the original prototype design, covering: the actual DA/CoT prompt text (§1.2/§2.2, including the §1.3 fixes above), the redesigned image preprocessing (§5 above), the actual evaluation set size (400 items: 200 yes/no + 200 inference, seed 42, 278 cases), the LLM-as-judge scoring methodology (§2 above) replacing the old "rule-based matching" description, the now-completed RQ3 image-only ablation (§3 above, previously documented as "planned"), and per-pipeline dependency tables corrected against actual imports (e.g. `pandas`/`scipy`/`opencv-python` removed or re-attributed where they were never actually used by the script being described).
