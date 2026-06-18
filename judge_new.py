#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-judge script for KneeCoT inference-question evaluation.

What it does
------------
1. Reads raw inference outputs from JSON / JSONL / CSV files or directories.
2. Normalizes each record into a judge input:
   case_id, qa_index, question, expected_answer, candidate_model_answer,
   diagnosis_opinion, mr_findings, ground_truth_labels, condition.
3. Sends a question-aware rubric prompt to a local Ollama judge model.
4. Saves structured judgments incrementally as JSONL and optionally as JSON.
5. Creates a risk-based manual-review file:
   - all incorrect / unclear / needs_manual_review samples;
   - a random 15-20% style sample of correct cases, configurable.

Recommended HPC use
-------------------
    ollama pull qwen2.5:32b
    python judge_new.py \
      --input results/inference_outputs.json vlm_results/inference_outputs.json \
      --rubric inference_rubric_for_LLM_judge.json \
      --model qwen2.5:32b \
      --output judged_inference.jsonl \
      --json-output judged_inference.json \
      --review-output manual_review_sample.jsonl

This script uses only the Python standard library. It calls the local Ollama REST API
(default: http://localhost:11434) and does not send medical data to external APIs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import json
import math
import os
import random
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


# -----------------------------
# Field-name normalization
# -----------------------------
#
# These lists were originally written for a generic/assumed schema and did not
# match this project's actual eval_set / raw_results field names. The project's
# own field names (qtype, ground_truth, prompt_key) are listed first in each
# group below so they take priority, while the generic aliases are kept for
# compatibility with other input shapes.

CASE_ID_KEYS = [
    "case_id", "case", "case_uid", "study_id", "exam_id", "patient_id", "file_id",
    "filename", "file_name", "json_file", "病例ID", "病例编号",
]
QUESTION_ID_KEYS = [
    "question_id", "qa_id", "qid", "id", "question_uid", "问题ID",
]
QA_INDEX_KEYS = [
    "qa_index", "qa_idx", "index", "idx", "question_index", "q_index", "问题序号",
]
QUESTION_KEYS = [
    "question", "query", "prompt_question", "问题", "题目",
]
# "qtype" is the actual field name produced by build_eval_set.py / preprocessing.py
# for both the LLM and VLM lines. It was missing before, which made
# looks_like_inference() unable to find the question type at all.
QUESTION_TYPE_KEYS = [
    "qtype", "question_type", "qa_type", "type", "category", "问题类型", "task_type",
]
# "ground_truth" and "full_answer" are the actual ground-truth fields written by
# build_eval_set.py for both lines. Neither was in this list before, which meant
# expected_answer was always empty for every real record from this pipeline.
EXPECTED_KEYS = [
    "ground_truth", "full_answer", "expected_answer", "reference_answer", "reference",
    "gt_answer", "ground_truth_answer", "gold_answer", "target_answer", "standard_answer",
    "answer", "expected", "标准答案", "参考答案",
]
CANDIDATE_KEYS_STRICT = [
    "candidate_model_answer", "candidate_answer", "prediction", "predicted_answer",
    "parsed_answer", "generated_answer", "generation_answer", "model_prediction",
    "llm_answer", "vlm_answer", "final_answer", "模型答案", "候选答案",
]
CANDIDATE_TEXT_KEYS = [
    "model_output", "raw_output", "raw_response", "response", "output", "generated_text",
    "completion", "generation", "text", "full_output", "模型输出",
]
# Only use model_answer as candidate when an explicit expected/reference key is also present.
AMBIGUOUS_CANDIDATE_KEYS = ["model_answer"]
DIAGNOSIS_KEYS = [
    "diagnosis_opinion", "diagnostic_opinion", "impression", "diagnosis", "诊断意见", "诊断结论",
]
FINDINGS_KEYS = [
    "mr_findings", "findings", "MR表现", "mr表现", "report_findings", "radiology_findings",
    "report", "检查所见", "影像表现",
]
LABEL_KEYS = [
    "ground_truth_labels", "labels", "structured_labels", "label", "标签", "结构化标签",
]
# "prompt_key" is the actual field name written by inference.py / evaluate.py for both
# lines (DA / CoT). It was missing before, which made condition always resolve to
# "unknown" -- losing the DA-vs-CoT breakdown and, worse, collapsing the dedup key
# used by stable_record_key() so that --resume could silently skip one condition.
CONDITION_KEYS = [
    "prompt_key", "condition", "prompt_type", "prompt_mode", "model_condition",
    "setting", "pipeline", "run_name", "实验条件",
]

ANSWER_MARKERS = [
    "【答案】", "【Answer】", "[Answer]", "[答案]", "Answer:", "Answer：",
    "答案：", "答案:", "最终答案：", "最终答案:", "结论：", "结论:",
]

ALLOWED_VERDICTS = {"correct", "incorrect", "unclear"}


# -----------------------------
# Small utilities
# -----------------------------

def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def truncate_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except TypeError:
            value = str(value)
    value = value.strip()
    if max_chars and len(value) > max_chars:
        return value[:max_chars] + f"\n...[TRUNCATED to {max_chars} chars]"
    return value


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def get_by_path(obj: Dict[str, Any], key: str) -> Any:
    """Get value by flat key or dotted path."""
    if key in obj:
        return obj[key]
    cur: Any = obj
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def first_present(obj: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        val = get_by_path(obj, key)
        if val is not None and val != "":
            return val
    return None


def first_present_key(obj: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        val = get_by_path(obj, key)
        if val is not None and val != "":
            return key
    return None


def stable_record_key(record: Dict[str, Any]) -> str:
    parts = [
        str(record.get("condition", "")),
        str(record.get("case_id", "")),
        str(record.get("qa_index", "")),
        str(record.get("question_id", "")),
        str(record.get("question", ""))[:80],
    ]
    return "||".join(parts)


def sanitize_for_json(obj: Any) -> Any:
    """Ensure an object can be JSON-serialized."""
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [sanitize_for_json(v) for v in obj]
        return str(obj)


# -----------------------------
# Input loading and flattening
# -----------------------------

def load_json_file(path: Path) -> Any:
    text = read_text(path).strip()
    if not text:
        return []
    return json.loads(text)


def load_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                obj.setdefault("_source_line", line_no)
            out.append(obj)
    return out


def load_csv_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def flatten_loaded_obj(obj: Any, source_file: str, inherited_condition: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Flatten common result layouts into item dictionaries.

    Supports:
    - a list of result records;
    - {"results": [...]}, {"outputs": [...]}, {"data": [...]};
    - {"llm_direct": [...], "llm_cot": [...], ...};
    - nested dicts containing those structures.
    """
    if obj is None:
        return

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                rec = dict(item)
                rec.setdefault("_source_file", source_file)
                rec.setdefault("_source_index", i)
                if inherited_condition and not first_present(rec, CONDITION_KEYS):
                    rec["condition"] = inherited_condition
                yield rec
            else:
                yield {
                    "_source_file": source_file,
                    "_source_index": i,
                    "raw_item": item,
                    "condition": inherited_condition or "",
                }
        return

    if not isinstance(obj, dict):
        yield {"_source_file": source_file, "raw_item": obj, "condition": inherited_condition or ""}
        return

    known_container_keys = ["results", "outputs", "data", "items", "records", "examples", "inference_outputs"]
    for key in known_container_keys:
        if isinstance(obj.get(key), list):
            yield from flatten_loaded_obj(obj[key], source_file, inherited_condition)
            return

    # Common pattern: {"llm_direct": [..], "llm_cot": [..], ...}
    yielded = False
    for key, val in obj.items():
        if key.startswith("_"):
            continue
        if isinstance(val, list):
            yielded = True
            yield from flatten_loaded_obj(val, source_file, inherited_condition=str(key))
        elif isinstance(val, dict):
            nested_has_list = any(isinstance(v, list) for v in val.values())
            if nested_has_list:
                yielded = True
                yield from flatten_loaded_obj(val, source_file, inherited_condition=str(key))

    if yielded:
        return

    rec = dict(obj)
    rec.setdefault("_source_file", source_file)
    if inherited_condition and not first_present(rec, CONDITION_KEYS):
        rec["condition"] = inherited_condition
    yield rec


def expand_input_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        matched = glob.glob(item)
        if matched:
            paths.extend(Path(p) for p in matched)
        else:
            paths.append(Path(item))
    expanded: List[Path] = []
    for path in paths:
        if path.is_dir():
            for pattern in ("**/*.jsonl", "**/*.json", "**/*.csv"):
                expanded.extend(path.glob(pattern))
        else:
            expanded.append(path)
    # Stable de-duplication
    seen = set()
    unique: List[Path] = []
    for p in expanded:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def load_records(inputs: Sequence[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in expand_input_paths(inputs):
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            obj = load_jsonl_file(path)
        elif suffix == ".json":
            obj = load_json_file(path)
        elif suffix == ".csv":
            obj = load_csv_file(path)
        else:
            eprint(f"[WARN] Skipping unsupported input file: {path}")
            continue
        records.extend(flatten_loaded_obj(obj, source_file=str(path)))
    return records


# -----------------------------
# Record normalization
# -----------------------------

def extract_final_answer(text: Any, max_chars: int = 6000) -> str:
    """Extract the final answer after the last explicit answer marker.

    If no marker is found, returns the original text trimmed. This is intentional:
    some pipelines already store only the final answer.
    """
    text = truncate_text(text, max_chars=max_chars)
    if not text:
        return ""

    last_pos = -1
    last_marker = ""
    lower_text = text.lower()
    for marker in ANSWER_MARKERS:
        pos = lower_text.rfind(marker.lower())
        if pos > last_pos:
            last_pos = pos
            last_marker = marker

    if last_pos >= 0:
        answer = text[last_pos + len(last_marker):].strip()
    else:
        answer = text.strip()

    # Remove markdown fences or common heading leftovers.
    answer = re.sub(r"^```(?:json|text)?\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"\s*```$", "", answer).strip()
    return answer


def looks_like_inference(record: Dict[str, Any]) -> bool:
    qtype = truncate_text(first_present(record, QUESTION_TYPE_KEYS), 200).lower()
    if any(tok in qtype for tok in ["inference", "推理", "diagnostic inference"]):
        return True
    if qtype:
        # A recognized, non-inference qtype (e.g. "yesno") must NOT be treated as
        # inference. Only fall through to the "keep if unknown" rule when the
        # qtype field genuinely could not be found at all.
        return False
    # If no type is available, keep the record. Some exported inference files omit qtype.
    return True


def normalize_record(raw: Dict[str, Any], idx: int, max_field_chars: int) -> Dict[str, Any]:
    expected_key = first_present_key(raw, EXPECTED_KEYS)
    expected_answer = first_present(raw, EXPECTED_KEYS)

    candidate = first_present(raw, CANDIDATE_KEYS_STRICT)
    candidate_source = first_present_key(raw, CANDIDATE_KEYS_STRICT)

    if candidate is None:
        # Use full generation fields and extract the final [Answer] segment.
        raw_candidate_text = first_present(raw, CANDIDATE_TEXT_KEYS)
        if raw_candidate_text is not None:
            candidate = extract_final_answer(raw_candidate_text, max_chars=max_field_chars)
            candidate_source = first_present_key(raw, CANDIDATE_TEXT_KEYS)

    if candidate is None and expected_key is not None:
        # Some scripts use model_answer for the prediction while answer/reference_answer
        # stores the GT. Only enable this when expected/reference is explicitly present.
        ambiguous = first_present(raw, AMBIGUOUS_CANDIDATE_KEYS)
        if ambiguous is not None:
            candidate = extract_final_answer(ambiguous, max_chars=max_field_chars)
            candidate_source = first_present_key(raw, AMBIGUOUS_CANDIDATE_KEYS)

    condition = first_present(raw, CONDITION_KEYS)
    if not condition:
        condition = raw.get("_condition") or raw.get("_source_condition") or "unknown"

    case_id = first_present(raw, CASE_ID_KEYS)
    if not case_id:
        # Fall back to the source filename stem to keep keys usable.
        source_file = raw.get("_source_file", "")
        case_id = Path(str(source_file)).stem if source_file else "unknown_case"

    qa_index = first_present(raw, QA_INDEX_KEYS)
    if qa_index is None:
        qa_index = raw.get("_source_index", idx)

    question = first_present(raw, QUESTION_KEYS)
    qid = first_present(raw, QUESTION_ID_KEYS)
    if not qid:
        # Avoid keying on the raw list position: VLM eval_set items have no
        # question_id field, and a position-based id would silently change
        # across re-exports / partial reruns, breaking dedup and cross-condition
        # matching. Hash case_id+question instead so the id is stable regardless
        # of file ordering.
        import hashlib
        digest = hashlib.md5(f"{case_id}||{question or ''}".encode("utf-8")).hexdigest()[:10]
        qid = f"{case_id}_{digest}"

    labels = first_present(raw, LABEL_KEYS)

    norm = {
        "question_id": str(qid),
        "case_id": str(case_id),
        "qa_index": int(qa_index) if str(qa_index).isdigit() else qa_index,
        "condition": str(condition),
        "question_type": truncate_text(first_present(raw, QUESTION_TYPE_KEYS), 300),
        "question": truncate_text(question, max_field_chars),
        "expected_answer": truncate_text(expected_answer, max_field_chars),
        "candidate_model_answer": truncate_text(candidate, max_field_chars),
        "diagnosis_opinion": truncate_text(first_present(raw, DIAGNOSIS_KEYS), max_field_chars),
        "mr_findings": truncate_text(first_present(raw, FINDINGS_KEYS), max_field_chars),
        "ground_truth_labels": sanitize_for_json(labels) if labels is not None else {},
        "_source_file": raw.get("_source_file", ""),
        "_source_index": raw.get("_source_index", idx),
        "_candidate_source_field": candidate_source or "",
        "_expected_source_field": expected_key or "",
    }
    return norm


# -----------------------------
# Rubric and prompts
# -----------------------------

def default_rubric_path() -> Path:
    here = Path(__file__).resolve().parent
    # Try this project's actual rubric filename first, then fall back to the
    # generic name the script originally assumed.
    for name in ("inference_rubric_for_LLM_judge.json", "inference_rubric_v2_question_aware.json"):
        p = here / name
        if p.exists():
            return p
    return here / "inference_rubric_for_LLM_judge.json"


def load_rubric(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Rubric not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "category_id", "display_name", "priority", "applies_when", "ground_truth_sources",
        "related_terms", "important_note", "subrules", "manual_review_if", "common_pitfalls",
    ]
    return {k: rule.get(k) for k in keep_keys if k in rule}


def make_rubric_payload(rubric: Dict[str, Any], mode: str, max_chars: int) -> str:
    if mode == "full":
        payload = rubric
    else:
        payload = {
            "rubric_name": rubric.get("rubric_name"),
            "language": rubric.get("language"),
            "version_notes": rubric.get("version_notes"),
            "judge_output_schema": rubric.get("judge_output_schema"),
            "global_decision_policy": rubric.get("global_decision_policy"),
            "normalization": rubric.get("normalization"),
            "rules": [compact_rule(r) for r in rubric.get("rules", [])],
            "coverage_checklist_for_current_dataset": rubric.get("coverage_checklist_for_current_dataset"),
        }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n...[RUBRIC TRUNCATED to {max_chars} chars]"
    return text


def judge_json_schema() -> Dict[str, Any]:
    """JSON schema passed to Ollama's `format` field."""
    return {
        "type": "object",
        "properties": {
            "question_id": {"type": "string"},
            "case_id": {"type": "string"},
            "qa_index": {"type": ["integer", "string"]},
            "matched_rule_ids": {"type": "array", "items": {"type": "string"}},
            "question_intent": {"type": "string"},
            "ground_truth_core": {"type": "string"},
            "extracted_model_conclusion": {"type": "string"},
            "required_elements": {"type": "array", "items": {"type": "string"}},
            "matched_elements": {"type": "array", "items": {"type": "string"}},
            "contradictions": {"type": "array", "items": {"type": "string"}},
            "verdict": {"type": "string", "enum": ["correct", "incorrect", "unclear"]},
            "correct": {"type": "boolean"},
            "needs_manual_review": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": [
            "question_id", "case_id", "qa_index", "matched_rule_ids", "question_intent",
            "ground_truth_core", "extracted_model_conclusion", "required_elements",
            "matched_elements", "contradictions", "verdict", "correct",
            "needs_manual_review", "reason",
        ],
        "additionalProperties": True,
    }


def build_messages(record: Dict[str, Any], rubric: Dict[str, Any], rubric_text: str) -> List[Dict[str, str]]:
    prompt_template = rubric.get("llm_judge_prompt_template", {})
    system = prompt_template.get("system") or (
        "你是一个医学 QA 评估判官。你的任务不是重新诊断病例，而是判断待评模型答案是否正确回答了 inference question 的核心结论。"
        "必须遵守 question-aware 规则；不确定则输出 unclear。"
    )

    labels = record.get("ground_truth_labels", {})
    if not isinstance(labels, str):
        labels_text = json.dumps(labels, ensure_ascii=False)
    else:
        labels_text = labels

    user = f"""
请根据下面的 rubric_v2 评估一个 Knee MRI inference QA。你必须只依据 rubric 和给定参考信息评分，不要重新诊断病例，不要把 related_terms / positive_terms 当作自动判对清单。

<RUBRIC_V2_JSON>
{rubric_text}
</RUBRIC_V2_JSON>

<CASE_TO_JUDGE>
case_id: {record.get('case_id', '')}
qa_index: {record.get('qa_index', '')}
question_id: {record.get('question_id', '')}
condition: {record.get('condition', '')}
question: {record.get('question', '')}
expected_answer_or_reference_answer: {record.get('expected_answer', '')}
candidate_model_answer: {record.get('candidate_model_answer', '')}
diagnosis_opinion: {record.get('diagnosis_opinion', '')}
mr_findings: {record.get('mr_findings', '')}
ground_truth_labels: {labels_text}
</CASE_TO_JUDGE>

输出要求：
1. 只输出严格 JSON，不要 markdown，不要解释性段落。
2. 字段必须包含：question_id, case_id, qa_index, matched_rule_ids, question_intent, ground_truth_core, extracted_model_conclusion, required_elements, matched_elements, contradictions, verdict, correct, needs_manual_review, reason。
3. verdict 只能是 correct / incorrect / unclear。
4. reason 必须是一句话，说明你按哪类 question_intent 判定，以及模型答案与 ground_truth_core 的关系。
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# -----------------------------
# Ollama call and parsing
# -----------------------------

def post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def call_ollama_chat(
    *,
    host: str,
    model: str,
    messages: List[Dict[str, str]],
    schema: Dict[str, Any],
    temperature: float,
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> str:
    host = host.rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": schema,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "keep_alive": keep_alive,
    }
    url = f"{host}/api/chat"
    last_exc: Optional[BaseException] = None
    for attempt in range(1, retries + 2):
        try:
            resp = post_json(url, payload, timeout=timeout)
            msg = resp.get("message", {})
            content = msg.get("content", "")
            if not content:
                raise RuntimeError(f"Ollama returned no message.content: {resp}")
            return content
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_exc = exc
            if attempt <= retries:
                eprint(f"[WARN] Ollama call failed on attempt {attempt}/{retries + 1}: {exc}; retrying in {retry_sleep}s")
                time.sleep(retry_sleep)
            else:
                break
    raise RuntimeError(f"Ollama call failed after {retries + 1} attempts: {last_exc}")


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
        raise ValueError(f"Judge JSON is not an object: {type(obj).__name__}")
    except json.JSONDecodeError:
        pass

    # Fallback: extract the largest likely JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(cleaned[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Could not parse judge response as JSON: {cleaned[:500]}")


def normalize_judgment(judgment: Dict[str, Any], record: Dict[str, Any], model: str) -> Dict[str, Any]:
    out = dict(judgment)

    out["question_id"] = str(out.get("question_id") or record.get("question_id") or "")
    out["case_id"] = str(out.get("case_id") or record.get("case_id") or "")
    out["qa_index"] = out.get("qa_index", record.get("qa_index"))

    for key in ["matched_rule_ids", "required_elements", "matched_elements", "contradictions"]:
        val = out.get(key)
        if val is None:
            out[key] = []
        elif not isinstance(val, list):
            out[key] = [str(val)]
        else:
            out[key] = [str(v) for v in val]

    verdict = str(out.get("verdict", "unclear")).strip().lower()
    if verdict not in ALLOWED_VERDICTS:
        verdict = "unclear"
        out.setdefault("contradictions", []).append("Judge returned invalid verdict; coerced to unclear.")
    out["verdict"] = verdict
    out["correct"] = verdict == "correct"

    needs_review = bool(out.get("needs_manual_review", False))
    if verdict in {"incorrect", "unclear"}:
        needs_review = True
    if out.get("contradictions"):
        needs_review = True
    out["needs_manual_review"] = needs_review

    out.setdefault("question_intent", "")
    out.setdefault("ground_truth_core", "")
    out.setdefault("extracted_model_conclusion", "")
    out.setdefault("reason", "")

    out["condition"] = record.get("condition", "unknown")
    out["question"] = record.get("question", "")
    out["expected_answer"] = record.get("expected_answer", "")
    out["candidate_model_answer"] = record.get("candidate_model_answer", "")
    out["diagnosis_opinion"] = record.get("diagnosis_opinion", "")
    out["mr_findings"] = record.get("mr_findings", "")
    out["ground_truth_labels"] = record.get("ground_truth_labels", {})
    out["judge_model"] = model
    out["judged_at"] = now_iso()
    out["_source_file"] = record.get("_source_file", "")
    out["_source_index"] = record.get("_source_index", "")
    out["_record_key"] = stable_record_key(record)
    return out


def make_error_judgment(record: Dict[str, Any], model: str, exc: BaseException) -> Dict[str, Any]:
    reason = f"Judge failed: {type(exc).__name__}: {exc}"
    return normalize_judgment(
        {
            "question_id": record.get("question_id", ""),
            "case_id": record.get("case_id", ""),
            "qa_index": record.get("qa_index", ""),
            "matched_rule_ids": [],
            "question_intent": "judge_error",
            "ground_truth_core": "",
            "extracted_model_conclusion": record.get("candidate_model_answer", ""),
            "required_elements": [],
            "matched_elements": [],
            "contradictions": [reason],
            "verdict": "unclear",
            "correct": False,
            "needs_manual_review": True,
            "reason": reason,
        },
        record,
        model,
    )


# -----------------------------
# Output, review sampling, summary
# -----------------------------

def read_existing_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except json.JSONDecodeError:
                eprint(f"[WARN] Could not parse existing output line in {path}; ignoring.")
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json_array(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not path or str(path).lower() in {"none", "null", ""}:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=2)


def write_json_object(path: Path, obj: Dict[str, Any]) -> None:
    if not path or str(path).lower() in {"none", "null", ""}:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_manual_review_sample(
    judgments: Sequence[Dict[str, Any]],
    correct_sample_rate: float,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    boundary: List[Dict[str, Any]] = []
    correct: List[Dict[str, Any]] = []

    for j in judgments:
        if j.get("verdict") == "correct" and not j.get("needs_manual_review"):
            correct.append(j)
        else:
            boundary.append(j)

    sample_n = int(math.ceil(max(0.0, correct_sample_rate) * len(correct)))
    sampled_correct = rng.sample(correct, sample_n) if sample_n and len(correct) >= sample_n else list(correct)

    review_rows: List[Dict[str, Any]] = []
    for row in boundary:
        r = dict(row)
        r["review_selected"] = True
        r["review_reason"] = "boundary_case_all_incorrect_unclear_or_flagged"
        review_rows.append(r)
    for row in sampled_correct:
        r = dict(row)
        r["review_selected"] = True
        r["review_reason"] = f"random_correct_spotcheck_{correct_sample_rate:.0%}"
        review_rows.append(r)

    # Stable order: boundary first in original order, then correct sample sorted by key.
    return review_rows


def summarize(judgments: Sequence[Dict[str, Any]], review_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_verdict = Counter(j.get("verdict", "unknown") for j in judgments)
    by_condition: Dict[str, Dict[str, int]] = {}
    temp: Dict[str, Counter] = defaultdict(Counter)
    for j in judgments:
        temp[str(j.get("condition", "unknown"))][str(j.get("verdict", "unknown"))] += 1
    for cond, ctr in temp.items():
        by_condition[cond] = dict(ctr)

    total = len(judgments)
    correct = by_verdict.get("correct", 0)
    return {
        "created_at": now_iso(),
        "total_judged": total,
        "verdict_counts": dict(by_verdict),
        "accuracy_if_unclear_counted_incorrect": (correct / total) if total else None,
        "counts_by_condition": by_condition,
        "manual_review_count": len(review_rows),
        "manual_review_policy": {
            "included_all_unclear": True,
            "included_all_incorrect": True,
            "included_all_needs_manual_review": True,
            "included_random_correct_spotcheck": True,
        },
    }


# -----------------------------
# Main
# -----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Ollama LLM-as-judge for KneeCoT inference-question answers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", nargs="+", required=True, help="Input JSON/JSONL/CSV files, globs, or directories.")
    parser.add_argument("--rubric", default=str(default_rubric_path()), help="Path to inference_rubric_for_LLM_judge.json.")
    parser.add_argument("--model", default="qwen2.5:32b", help="Local Ollama judge model. Must not be the evaluated 7B model.")
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"), help="Local Ollama host URL.")
    parser.add_argument("--output", default="judged_inference.jsonl", help="Streaming JSONL output path.")
    parser.add_argument("--json-output", default="judged_inference.json", help="Optional final JSON array output path; set to 'none' to skip.")
    parser.add_argument("--review-output", default="manual_review_sample.jsonl", help="Manual review JSONL output path; set to 'none' to skip.")
    parser.add_argument("--summary-output", default="judged_inference_summary.json", help="Summary JSON path; set to 'none' to skip.")
    parser.add_argument("--rubric-mode", choices=["compact", "full"], default="compact", help="Whether to send compact or full rubric JSON in each prompt.")
    parser.add_argument("--max-rubric-chars", type=int, default=70000, help="Maximum rubric characters inserted into prompt; 0 means no truncation.")
    parser.add_argument("--max-field-chars", type=int, default=8000, help="Maximum characters for each long record field.")
    parser.add_argument("--num-ctx", type=int, default=32768, help="Ollama num_ctx option.")
    parser.add_argument("--num-predict", type=int, default=1400, help="Ollama num_predict option for judgment JSON.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge temperature; keep at 0 for reproducibility.")
    parser.add_argument("--keep-alive", default="30m", help="Ollama keep_alive value.")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout in seconds per judge call.")
    parser.add_argument("--retries", type=int, default=2, help="Number of retries after the first failed Ollama call.")
    parser.add_argument("--retry-sleep", type=float, default=5.0, help="Seconds to sleep between retries.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum records to judge after filtering; 0 means all.")
    parser.add_argument("--start", type=int, default=0, help="Start offset after filtering.")
    parser.add_argument("--no-filter-inference", action="store_true", help="Do not filter to inference/推理 records.")
    parser.add_argument("--resume", action="store_true", help="Append to existing JSONL and skip already judged record keys.")
    parser.add_argument("--correct-sample-rate", type=float, default=0.20, help="Random sample rate from correct judgments for manual spot-check.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for correct-case spot-check sampling.")
    parser.add_argument("--dry-run", action="store_true", help="Build and print the first prompt, but do not call Ollama.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately if one record fails; otherwise save unclear judge_error.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    rubric_path = Path(args.rubric)
    rubric = load_rubric(rubric_path)
    rubric_text = make_rubric_payload(rubric, args.rubric_mode, args.max_rubric_chars)
    schema = judge_json_schema()

    raw_records = load_records(args.input)
    normalized = [normalize_record(r, i, args.max_field_chars) for i, r in enumerate(raw_records)]

    if not args.no_filter_inference:
        before = len(normalized)
        normalized = [r for r in normalized if looks_like_inference(r)]
        eprint(f"[INFO] Inference filter: kept {len(normalized)}/{before} records.")

    missing_required = []
    usable_records = []
    for r in normalized:
        if not r.get("question") or not r.get("candidate_model_answer"):
            missing_required.append(r)
        else:
            usable_records.append(r)
    if missing_required:
        eprint(f"[WARN] {len(missing_required)} records missing question or candidate_model_answer; they will be saved as unclear judge_error.")

    selected = usable_records[args.start :]
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]

    output_path = Path(args.output)
    existing: List[Dict[str, Any]] = []
    existing_keys = set()
    if args.resume:
        existing = read_existing_jsonl(output_path)
        existing_keys = {str(j.get("_record_key", "")) for j in existing if j.get("_record_key")}
        eprint(f"[INFO] Resume enabled: loaded {len(existing)} existing judgments; {len(existing_keys)} keys to skip.")
    elif output_path.exists():
        output_path.unlink()

    if args.dry_run:
        if not selected:
            eprint("[ERROR] No usable records selected for dry run.")
            return 2
        messages = build_messages(selected[0], rubric, rubric_text)
        print(json.dumps({"model": args.model, "messages": messages, "format": schema}, ensure_ascii=False, indent=2))
        return 0

    all_new: List[Dict[str, Any]] = []

    # Save missing-required records as review-needed errors first.
    if missing_required and not args.resume:
        error_rows = [
            make_error_judgment(r, args.model, ValueError("Missing question or candidate_model_answer after normalization"))
            for r in missing_required
        ]
        write_jsonl(output_path, error_rows, append=True)
        all_new.extend(error_rows)

    total = len(selected)
    eprint(f"[INFO] Judging {total} records with local Ollama model {args.model} at {args.ollama_host}")
    for n, record in enumerate(selected, 1):
        key = stable_record_key(record)
        if args.resume and key in existing_keys:
            continue
        try:
            messages = build_messages(record, rubric, rubric_text)
            raw_response = call_ollama_chat(
                host=args.ollama_host,
                model=args.model,
                messages=messages,
                schema=schema,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
                keep_alive=args.keep_alive,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            parsed = parse_json_object(raw_response)
            judgment = normalize_judgment(parsed, record, args.model)
        except BaseException as exc:  # noqa: BLE001 - controlled by fail_fast
            if args.fail_fast:
                traceback.print_exc()
                return 1
            eprint(f"[ERROR] Record {n}/{total} failed: {exc}")
            judgment = make_error_judgment(record, args.model, exc)

        write_jsonl(output_path, [judgment], append=True)
        all_new.append(judgment)
        if n % 10 == 0 or n == total:
            eprint(f"[INFO] Progress: {n}/{total} processed.")

    judgments = existing + all_new if args.resume else read_existing_jsonl(output_path)

    # Final JSON array, manual review sample, and summary.
    if str(args.json_output).lower() not in {"none", "null", ""}:
        write_json_array(Path(args.json_output), judgments)
        eprint(f"[INFO] Wrote JSON array: {args.json_output}")

    review_rows: List[Dict[str, Any]] = []
    if str(args.review_output).lower() not in {"none", "null", ""}:
        review_rows = make_manual_review_sample(judgments, args.correct_sample_rate, args.seed)
        write_jsonl(Path(args.review_output), review_rows, append=False)
        eprint(f"[INFO] Wrote manual review sample: {args.review_output} ({len(review_rows)} rows)")

    if str(args.summary_output).lower() not in {"none", "null", ""}:
        summary = summarize(judgments, review_rows)
        write_json_object(Path(args.summary_output), summary)
        eprint(f"[INFO] Wrote summary: {args.summary_output}")
        eprint(json.dumps(summary, ensure_ascii=False, indent=2))

    eprint(f"[INFO] Done. Main streaming output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
