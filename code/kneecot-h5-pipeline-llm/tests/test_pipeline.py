"""Minimal sanity tests for the H5 data pipeline (no GPU/model needed).

Run from the repo root:  python -m pytest tests/  (or: python tests/test_pipeline.py)
"""
import os
import sys

_LLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_NEW_DIR = os.path.dirname(_LLM_DIR)
sys.path.insert(0, os.path.join(_LLM_DIR, "src"))
# parse_yes_no now lives in the shared scoring script (analysis/compare.py)
# instead of this line's own evaluation.py -- see that file's module
# docstring for why generation and scoring were split apart.
sys.path.insert(0, os.path.join(_CODE_NEW_DIR, "analysis"))

from preprocessing import (  # noqa: E402
    build_eval_items,
    extract_yes_no_gt,
    is_knee_only_case,
)
from compare import parse_yes_no  # noqa: E402


def test_extract_yes_no_gt():
    assert extract_yes_no_gt("Yes。前交叉韧带增粗。") == "Yes"
    assert extract_yes_no_gt("No。内侧副韧带未见异常。") == "No"
    assert extract_yes_no_gt("属于 Stoller I-II 级。") is None


def test_knee_only_filter():
    assert is_knee_only_case({"检查方法": "单侧膝关节（右膝关节）磁共振平扫"}) is True
    assert is_knee_only_case({"检查方法": "膝关节磁共振平扫；肩关节磁共振平扫"}) is False
    assert is_knee_only_case({"检查方法": "肩关节磁共振平扫"}) is False


def test_parse_yes_no():
    assert parse_yes_no("步骤一...\n【答案】Yes") == "Yes"
    assert parse_yes_no("一些推理\n【答案】No，理由是...") == "No"
    assert parse_yes_no("【答案】无法判断") is None


def test_build_eval_items_keeps_only_h5_types():
    case = {
        "MR表现": "findings text",
        "顺序编号": "X001",
        "问答数据": {"qa_pairs": [
            {"question": "q1", "answer": "Yes。a", "type": "yes_no"},
            {"question": "q2", "answer": "desc", "type": "descriptive"},
            {"question": "q3", "answer": "推理", "type": "inference"},
            {"question": "q4", "answer": "loc", "type": "localization"},
        ]},
    }
    items = build_eval_items(case)
    types = sorted(it["qtype"] for it in items)
    # build_eval_items() normalizes the raw "yes_no" type to "yesno" to match
    # the shared eval_set schema used by both the LLM and VLM lines.
    assert types == ["inference", "yesno"]
    assert items[0]["findings"] == "findings text"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All tests passed.")
