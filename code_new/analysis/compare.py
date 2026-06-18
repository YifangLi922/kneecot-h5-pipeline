"""
compare.py
-----------
把 LLM 线和 VLM 线、direct 和 cot 四次运行的结果放在一起对比。

输入文件（共 5 个）：
  1) eval_set.json          —— build_eval_set.py 产出的冻结题表（标准答案的唯一来源）
  2) llm_direct.json        ┐
  3) llm_cot.json           │ 四条线各自的「成绩单」，每个文件是一个 list：
  4) vlm_direct.json        │   [{"question_id": "...", "raw_output": "模型完整输出"}, ...]
  5) vlm_cot.json           ┘ （case_id / type / 标准答案都不用放，compare 从 eval_set 取）

用法：
    python compare.py --eval_set eval_set.json \
        --llm_direct llm_direct.json --llm_cot llm_cot.json \
        --vlm_direct vlm_direct.json --vlm_cot vlm_cot.json \
        --out_dir compare_out --missing_policy wrong

  --missing_policy: 某条线缺某题时怎么办
        wrong = 记为答错（默认，最保守，四条线题集仍对齐）
        drop  = 四条线统一剔除这道题

产出（写到 out_dir/）：
    summary.json     —— 2×2 准确率表 + McNemar 结果
    per_item.csv     —— 每道题在四条线下的对错明细
    rq3_yesno.csv    —— yes/no 逐题对比（VLM 对、LLM 错 = 视觉必要）
    rq3_inference.csv—— inference 逐题对比
"""

import os
import re
import csv
import json
import argparse

try:
    from scipy.stats import binomtest, chi2
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------- 解析模型输出 ----------

def parse_yes_no(raw):
    """从模型输出里抽 yes/no，三级兜底，抽不到返回 None。"""
    text = raw or ""
    # 1) 【答案】之后找 Yes/No
    if "【答案】" in text:
        tail = text.split("【答案】")[-1]
        m = re.search(r"\b(Yes|No)\b", tail, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    # 2) 全文找英文 yes/no
    m = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # 3) 最后出现的 是 / 否
    last_shi, last_fou = text.rfind("是"), text.rfind("否")
    if last_shi == -1 and last_fou == -1:
        return None
    return "yes" if last_shi > last_fou else "no"


REASON_SEPARATORS = ["推理依据", "依据", "理由"]

def extract_pred_conclusion(raw):
    """从模型输出里抽预测结论：取【答案】之后、第一个分隔词之前的部分。"""
    text = raw or ""
    seg = text.split("【答案】")[-1] if "【答案】" in text else text
    for sep in REASON_SEPARATORS:
        if sep in seg:
            seg = seg.split(sep, 1)[0]
            break
    return seg.strip().strip("。.，,：: \n")


# ---------- inference 结论匹配（启发式 + 标记需复核） ----------

PUNCT = "。.，,、；;：:！!？?（）()【】[]「」“”\"' \t\n"
NEGATION = ["不", "未", "无", "否", "非", "排除", "没有"]

def _norm(s):
    return "".join(ch for ch in (s or "") if ch not in PUNCT)

def _polarity(s):
    """结论是否含否定意味。"""
    return any(neg in (s or "") for neg in NEGATION)

def _stoller_grades(s):
    """抽 Stoller 分级里的级别 token 集合，用于专门比较分级题。"""
    s = (s or "").upper().replace("Ⅰ","I").replace("Ⅱ","II").replace("Ⅲ","III")
    return set(re.findall(r"III|II|I", s)) if "级" in s else set()

def match_inference(gt_conclusion, pred_conclusion):
    """
    返回 (correct: bool, method: str, needs_review: bool)
    method 表明用什么规则判的，needs_review=True 表示自动判分不可靠，建议人工/LLM-judge 复核。
    """
    gt, pred = _norm(gt_conclusion), _norm(pred_conclusion)
    if not pred:
        return False, "empty", False

    # 规则 A：归一后完全一致 / 互相包含
    if gt and (gt == pred or gt in pred or pred in gt):
        return True, "exact_or_contain", False

    # 规则 B：Stoller 分级题，比级别集合
    g_grades, p_grades = _stoller_grades(gt_conclusion), _stoller_grades(pred_conclusion)
    if g_grades and p_grades:
        return (g_grades == p_grades), "stoller_grade", False

    # 规则 C：极性相反 → 判错（如 支持 vs 不支持、构成 vs 不构成）
    if _polarity(gt_conclusion) != _polarity(pred_conclusion):
        return False, "polarity_mismatch", False

    # 其余：自动判不准，保守记错并标记需复核
    return False, "needs_review", True


# ---------- 加载 ----------

def load_eval_set(path):
    """Load eval_set.json (supports {meta, items} and flat list)."""
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unrecognized eval_set format in {path}")
    result = {}
    for it in items:
        if "question_id" in it and it["question_id"]:
            key = it["question_id"]
        else:
            key = (it.get("case_id", ""), it.get("question", ""))
        result[key] = it
    return result

def load_run(path, filter_prompt_mode=None, filter_model=None):
    """Load results JSON -> {key: raw_output}. Supports native/LLM/VLM formats."""
    data = json.load(open(path, encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Result file {path} should contain a JSON list, got {type(data)}")
    result = {}
    for r in data:
        if filter_prompt_mode:
            mode = (r.get("prompt_mode") or r.get("prompt_type") or "").lower()
            if mode != filter_prompt_mode.lower():
                continue
        if filter_model:
            if r.get("model", "").lower() != filter_model.lower():
                continue
        if "question_id" in r and r["question_id"]:
            key = r["question_id"]
        else:
            key = (r.get("case_id", ""), r.get("question", ""))
        raw_out = r.get("raw_output") or r.get("raw_response") or ""
        result[key] = raw_out
    return result


# ---------- 判分 ----------

def score_run(eval_set, run, missing_policy):
    """
    给一条线判分，返回 {question_id: {"correct": bool, "pred": ..., "method": ..., "needs_review": bool}}
    某题在该线缺失时，按 missing_policy 处理。
    """
    out = {}
    for qid, item in eval_set.items():
        raw = run.get(qid, None)
        if raw is None:
            if missing_policy == "wrong":
                out[qid] = {"correct": False, "pred": None, "method": "missing",
                            "needs_review": False, "present": False}
            # drop 模式：直接不放进结果，后面对齐时统一剔除
            continue

        if item.get("qtype", item.get("type")) == "yes_no":
            pred = parse_yes_no(raw)
            correct = (pred is not None and pred == item["gt_label"])
            out[qid] = {"correct": correct, "pred": pred, "method": "yes_no",
                        "needs_review": False, "present": True}
        else:  # inference
            pred_c = extract_pred_conclusion(raw)
            correct, method, review = match_inference(item["gt_conclusion"], pred_c)
            out[qid] = {"correct": correct, "pred": pred_c, "method": method,
                        "needs_review": review, "present": True}
    return out


# ---------- McNemar ----------

def mcnemar(scores_a, scores_b, qids):
    """
    配对检验：在同一批 qids 上比较 a、b 两条线的对错。
    b = a错&b对 的数量；c = a对&b错 的数量。
    样本少用精确二项检验，多用带连续校正的卡方。
    """
    b = sum(1 for q in qids if not scores_a[q]["correct"] and scores_b[q]["correct"])
    c = sum(1 for q in qids if scores_a[q]["correct"] and not scores_b[q]["correct"])
    n = b + c
    if n == 0:
        return {"b_a_wrong_b_right": b, "c_a_right_b_wrong": c, "p_value": 1.0, "method": "no_discordant"}
    if not HAVE_SCIPY:
        return {"b_a_wrong_b_right": b, "c_a_right_b_wrong": c, "p_value": None,
                "method": "scipy_missing(pip install scipy)"}
    if n < 25:
        p = binomtest(min(b, c), n, 0.5).pvalue
        method = "exact_binomial"
    else:
        stat = (abs(b - c) - 1) ** 2 / n
        p = float(chi2.sf(stat, df=1))
        method = "chi2_continuity"
    return {"b_a_wrong_b_right": b, "c_a_right_b_wrong": c, "p_value": p, "method": method}


# ---------- 准确率 ----------

def accuracy(scores, qids):
    if not qids:
        return None
    return sum(scores[q]["correct"] for q in qids) / len(qids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--llm_results", default=None, help="LLM raw_results.json (auto-split by prompt_mode)")
    ap.add_argument("--vlm_results", default=None, help="VLM result file (auto-split by prompt_type, use --model)")
    ap.add_argument("--model", default="qwen2.5vl", help="VLM model name filter")
    ap.add_argument("--llm_direct", default=None)
    ap.add_argument("--llm_cot", default=None)
    ap.add_argument("--vlm_direct", default=None)
    ap.add_argument("--vlm_cot", default=None)
    ap.add_argument("--out_dir", default="compare_out")
    ap.add_argument("--missing_policy", choices=["wrong", "drop"], default="wrong")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    eval_set = load_eval_set(args.eval_set)
    # Determine input mode: individual or combined
    if args.llm_direct and args.llm_cot and args.vlm_direct and args.vlm_cot:
        runs_raw = {
            "llm_direct": load_run(args.llm_direct),
            "llm_cot":    load_run(args.llm_cot),
            "vlm_direct": load_run(args.vlm_direct),
            "vlm_cot":    load_run(args.vlm_cot),
        }
    elif args.llm_results and args.vlm_results:
        runs_raw = {
            "llm_direct": load_run(args.llm_results, filter_prompt_mode="direct"),
            "llm_cot":    load_run(args.llm_results, filter_prompt_mode="cot"),
            "vlm_direct": load_run(args.vlm_results, filter_prompt_mode="da", filter_model=args.model),
            "vlm_cot":    load_run(args.vlm_results, filter_prompt_mode="cot", filter_model=args.model),
        }
    else:
        raise ValueError("Need either 4 individual files or --llm_results + --vlm_results")

    # 判分
    scores = {name: score_run(eval_set, run, args.missing_policy)
              for name, run in runs_raw.items()}

    # 对齐：取四条线都有结果的 qid（drop 模式下天然剔除缺题；wrong 模式下四条线都齐）
    common = set(eval_set.keys())
    for s in scores.values():
        common &= set(s.keys())
    common = sorted(common)

    # 缺题报告
    missing_report = {name: sorted(set(eval_set) - set(runs_raw[name]))
                      for name in runs_raw}

    # 按题型切片
    yesno_q = [q for q in common if eval_set[q].get("qtype", eval_set[q].get("type")) == "yes_no"]
    infer_q = [q for q in common if eval_set[q].get("qtype", eval_set[q].get("type")) == "inference"]

    def block(qids):
        A, B, C, D = (scores["llm_direct"], scores["llm_cot"],
                      scores["vlm_direct"], scores["vlm_cot"])
        return {
            "n": len(qids),
            "acc": {
                "A_llm_direct": accuracy(A, qids), "B_llm_cot": accuracy(B, qids),
                "C_vlm_direct": accuracy(C, qids), "D_vlm_cot": accuracy(D, qids),
            },
            "RQ1_llm_cot_vs_direct_(B-A)": mcnemar(A, B, qids),
            "RQ2_vlm_cot_vs_direct_(D-C)": mcnemar(C, D, qids),
        }

    summary = {
        "n_total_aligned": len(common),
        "missing_policy": args.missing_policy,
        "missing_counts": {k: len(v) for k, v in missing_report.items()},
        "yes_no": block(yesno_q),
        "inference": block(infer_q),
        "inference_needs_review": sum(
            scores["llm_cot"][q]["needs_review"] or scores["vlm_cot"][q]["needs_review"]
            for q in infer_q),
    }
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # per_item 明细
    with open(os.path.join(args.out_dir, "per_item.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "case_id", "type", "gt",
                    "A_llm_direct", "B_llm_cot", "C_vlm_direct", "D_vlm_cot", "needs_review"])
        for q in common:
            it = eval_set[q]
            gt = it["gt_label"] if it.get("qtype", it.get("type")) == "yes_no" else it["gt_conclusion"]
            review = scores["llm_cot"][q]["needs_review"] or scores["vlm_cot"][q]["needs_review"]
            w.writerow([q, it["case_id"], it.get("qtype", it.get("type")), gt,
                        scores["llm_direct"][q]["correct"], scores["llm_cot"][q]["correct"],
                        scores["vlm_direct"][q]["correct"], scores["vlm_cot"][q]["correct"], review])

    # RQ3：用 cot 条件下 VLM vs LLM 逐题对比
    def write_rq3(qids, fname):
        with open(os.path.join(args.out_dir, fname), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["question_id", "case_id", "question",
                        "llm_cot_correct", "vlm_cot_correct", "verdict"])
            for q in qids:
                llm_ok = scores["llm_cot"][q]["correct"]
                vlm_ok = scores["vlm_cot"][q]["correct"]
                if vlm_ok and not llm_ok:
                    verdict = "vision_necessary"     # VLM 对、LLM 错
                elif llm_ok and not vlm_ok:
                    verdict = "text_better"
                elif llm_ok and vlm_ok:
                    verdict = "both_correct_text_sufficient"
                else:
                    verdict = "both_wrong"
                w.writerow([q, eval_set[q]["case_id"], eval_set[q]["question"],
                            llm_ok, vlm_ok, verdict])

    write_rq3(yesno_q, "rq3_yesno.csv")
    write_rq3(infer_q, "rq3_inference.csv")

    # 终端摘要
    print("==== 对比摘要 ====")
    print(f"对齐题数: {summary['n_total_aligned']}  缺题: {summary['missing_counts']}")
    for tname in ["yes_no", "inference"]:
        blk = summary[tname]
        print(f"\n[{tname}] n={blk['n']}")
        for k, v in blk["acc"].items():
            print(f"  {k:16s} = {v:.3f}" if v is not None else f"  {k:16s} = NA")
        print(f"  RQ1 (B-A) p={blk['RQ1_llm_cot_vs_direct_(B-A)']['p_value']}")
        print(f"  RQ2 (D-C) p={blk['RQ2_vlm_cot_vs_direct_(D-C)']['p_value']}")
    print(f"\ninference 需人工/LLM 复核的题数: {summary['inference_needs_review']}")
    print(f"\n详细结果写到: {args.out_dir}/")


if __name__ == "__main__":
    main()


