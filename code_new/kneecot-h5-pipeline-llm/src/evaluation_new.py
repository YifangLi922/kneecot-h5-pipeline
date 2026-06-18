"""I/O helpers for the H5 text-only LLM line.

This module used to also score results (yes/no accuracy, McNemar, and a
Chinese-bigram-overlap heuristic for inference questions). All of that has
been removed: the LLM line and the VLM line each ran their own scoring with
different field names, different yes/no parsers, and different inference
heuristics, so the two lines were never actually comparable on the same
ruler. Scoring now lives in one place, shared by both lines:

    code_new/analysis/compare_new.py   -- yes/no accuracy + McNemar
    judge_new.py                       -- inference verdicts (LLM-as-judge)

This file only saves the raw per-question generation records
(case_id, question_id, qtype, ground_truth, prompt_key, raw_output, ...)
produced by inference.py. It does not parse or score anything.
"""
import json


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
