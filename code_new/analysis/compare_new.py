"""
compare_new.py
-----------
把 LLM 线和 VLM 线、direct 和 cot 四次运行的结果放在一起对比。

这是全组唯一一份评分代码。LLM 线 (run_new.py) 和 VLM 线 (evaluate_new.py)
现在都只产出原始逐题记录，不再各自评分——之前两条线各写一份 parse_yes_no、
各用一套字段名（yes_no vs yesno、gt_label vs ground_truth、prompt_mode vs
prompt_key）、inference 题各用一套规则（bigram 重叠 vs 完全不评分），导致
"对比"环节其实不是在同一把尺子下比较。本文件把这些全部收回到一处：

  - yes/no 题：用本文件里唯一一份 parse_yes_no 解析，跟 eval_set 里的
    ground_truth 比较。
  - inference 题：不再用规则/bigram 自动判分，统一读 judge_new.py 产出的
    judged_inference.json（LLM-as-judge 初筛 + 人工复核后的结果），按
    question_id + condition 对齐到本题。如果还没跑 judge，本文件会把
    inference 题标成「未评分」，不会编造一个分数出来。

输入文件：
  1) eval_set.json          —— build_eval_set.py 产出的冻结题表（标准答案的唯一来源）
                                字段：case_id, question_id, question, qtype
                                ("yesno" / "inference"), ground_truth
  2) llm_direct.json / llm_cot.json / vlm_direct.json / vlm_cot.json
       —— 四条线各自的原始逐题记录（run_new.py / evaluate_new.py 的输出），
          每条记录至少有 case_id / question_id / qtype / raw_output。
          也可以直接传 run_new.py 的 raw_results.json（同时含 DA 和 CoT），
          用 --llm_results / --vlm_results，按 prompt_key 自动拆开。
  3) --judged_llm / --judged_vlm（可选）—— judge_new.py 产出的
       judged_inference.json，按 condition 字段（即 prompt_key 的值，
       "DA"/"CoT"）区分四条线里的 inference 部分。不传就不评 inference。

用法：
    python compare_new.py --eval_set eval_set.json \
        --llm_direct llm_direct.json --llm_cot llm_cot.json \
        --vlm_direct vlm_direct.json --vlm_cot vlm_cot.json \
        --judged_llm judged_inference_llm.json \
        --judged_vlm judged_inference_vlm.json \
        --out_dir compare_out --missing_policy wrong

  --missing_policy: 某条线缺某题时怎么办
        wrong = 记为答错（默认，最保守，四条线题集仍对齐）
        drop  = 四条线统一剔除这道题

产出（写到 out_dir/）：
    summary.json     —— 2×2 准确率表 + McNemar 结果（yes/no 和 inference 都算）
    per_item.csv     —— 每道题在四条线下的对错明细
    rq3_yesno.csv    —— yes/no 逐题对比（VLM 对、LLM 错 = 视觉必要）
    rq3_inference.csv—— inference 逐题对比
"""

import os
import re
import csv
import json
import hashlib
import argparse

try:
    from scipy.stats import binomtest, chi2
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

PROMPT_VALUES = ("DA", "CoT")


# ---------- 解析模型输出（全组唯一一份 yes/no 解析器） ----------

def parse_yes_no(raw):
    """从模型输出里抽 yes/no，三级兜底，抽不到返回 None。"""
    text = raw or ""
    if not text.strip():
        return None
    # 1) 【答案】之后找 Yes/No
    if "【答案】" in text:
        tail = text.split("【答案】")[-1]
        m = re.search(r"\b(Yes|No)\b", tail, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
    # 2) 全文找英文 yes/no
    m = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    # 3) 最后出现的 是 / 否
    last_shi, last_fou = text.rfind("是"), text.rfind("否")
    if last_shi == -1 and last_fou == -1:
        return None
    return "Yes" if last_shi > last_fou else "No"


# ---------- 加载 ----------

def derive_qid(case_id, question):
    """Same fallback id scheme as judge_new.py's normalize_record(): when a
    record has no question_id (e.g. the VLM eval_set never had one), hash
    case_id+question instead of using a (case_id, question) tuple key. This
    keeps ids stable and, crucially, lets judged_inference.json (produced by
    judge_new.py, which uses this same scheme) join back to these records."""
    digest = hashlib.md5(f"{case_id}||{question or ''}".encode("utf-8")).hexdigest()[:10]
    return f"{case_id}_{digest}"


def load_eval_set(path):
    """Load eval_set.json. Schema: case_id, question_id, question, qtype, ground_truth."""
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unrecognized eval_set format in {path}")
    result = {}
    for it in items:
        key = it["question_id"] if it.get("question_id") else derive_qid(it.get("case_id", ""), it.get("question", ""))
        result[key] = it
    return result


def load_run(path, filter_prompt_key=None):
    """Load a raw results JSON (list of records) -> {question_id: raw_output}.

    Accepts either a file that already only contains one condition (e.g.
    llm_direct.json) or a combined file with both DA and CoT records (e.g.
    raw_results.json from run_new.py), in which case filter_prompt_key picks
    out just one condition.
    """
    data = json.load(open(path, encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Result file {path} should contain a JSON list, got {type(data)}")
    result = {}
    for r in data:
        if filter_prompt_key:
            key_val = (r.get("prompt_key") or "").strip()
            if key_val.lower() != filter_prompt_key.lower():
                continue
        key = r["question_id"] if r.get("question_id") else derive_qid(r.get("case_id", ""), r.get("question", ""))
        result[key] = r.get("raw_output")
    return result


def load_judged(path):
    """Load judge_new.py output -> {(question_id, condition): judgment dict}."""
    if not path:
        return {}
    data = json.load(open(path, encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Judged file {path} should contain a JSON list, got {type(data)}")
    out = {}
    for j in data:
        qid = j.get("question_id")
        cond = (j.get("condition") or "").strip()
        if qid:
            out[(qid, cond)] = j
    return out


# ---------- 判分 ----------

def score_run(eval_set, run, judged, condition, missing_policy):
    """
    给一条线判分，返回 {question_id: {"correct": bool, "method": str,
                                       "needs_review": bool, "present": bool}}
    yes/no 用 parse_yes_no；inference 用 judged（judge_new.py 输出），
    没有 judged 记录时标 "not_judged"，不参与 accuracy 统计。
    """
    out = {}
    for qid, item in eval_set.items():
        raw_present = qid in run
        raw = run.get(qid)

        if item["qtype"] == "yesno":
            if not raw_present:
                if missing_policy == "wrong":
                    out[qid] = {"correct": False, "method": "missing", "needs_review": False, "present": False}
                continue
            pred = parse_yes_no(raw)
            correct = (pred is not None and pred == item["ground_truth"])
            out[qid] = {"correct": correct, "method": "yes_no", "needs_review": pred is None, "present": True}

        else:  # inference
            j = judged.get((qid, condition))
            if j is None:
                if missing_policy == "wrong":
                    out[qid] = {"correct": False, "method": "not_judged", "needs_review": True, "present": raw_present}
                continue
            out[qid] = {
                "correct": bool(j.get("correct", j.get("verdict") == "correct")),
                "method": f"judge:{j.get('verdict', 'unknown')}",
                "needs_review": bool(j.get("needs_manual_review", False)),
                "present": True,
            }
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


def accuracy(scores, qids):
    if not qids:
        return None
    return sum(scores[q]["correct"] for q in qids) / len(qids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--llm_results", default=None, help="LLM raw_results.json (combined DA+CoT, auto-split by prompt_key)")
    ap.add_argument("--vlm_results", default=None, help="VLM raw results (combined conditions, auto-split by prompt_key)")
    ap.add_argument("--llm_direct", default=None)
    ap.add_argument("--llm_cot", default=None)
    ap.add_argument("--vlm_direct", default=None)
    ap.add_argument("--vlm_cot", default=None)
    # The VLM line has 4 prompt_key values (prompts.py): "DA"/"CoT" are
    # image-only (the Round 3 ablation), "DA_findings"/"CoT_findings" are
    # image+MR-findings. The main matched LLM-vs-VLM comparison (RQ1/RQ2)
    # must use DA_findings/CoT_findings, since that is the condition where
    # the VLM gets the *same* text evidence as the LLM plus the image. Using
    # plain DA/CoT here would silently compare against the image-only
    # ablation condition instead of the matched one. Override these two
    # flags (e.g. to "DA"/"CoT") when you specifically want to run this
    # script over the image-only ablation results for RQ3.
    ap.add_argument("--vlm_prompt_direct", default="DA_findings",
                     help="VLM prompt_key value to treat as the 'direct' condition when using --vlm_results.")
    ap.add_argument("--vlm_prompt_cot", default="CoT_findings",
                     help="VLM prompt_key value to treat as the 'cot' condition when using --vlm_results.")
    ap.add_argument("--judged_llm", default=None, help="judge_new.py output for the LLM line (covers both DA and CoT)")
    ap.add_argument("--judged_vlm", default=None, help="judge_new.py output for the VLM line (covers both prompt conditions)")
    ap.add_argument("--out_dir", default="compare_out")
    ap.add_argument("--missing_policy", choices=["wrong", "drop"], default="wrong")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    eval_set = load_eval_set(args.eval_set)

    if args.llm_direct and args.llm_cot and args.vlm_direct and args.vlm_cot:
        runs_raw = {
            "llm_direct": load_run(args.llm_direct),
            "llm_cot":    load_run(args.llm_cot),
            "vlm_direct": load_run(args.vlm_direct),
            "vlm_cot":    load_run(args.vlm_cot),
        }
    elif args.llm_results and args.vlm_results:
        runs_raw = {
            "llm_direct": load_run(args.llm_results, filter_prompt_key="DA"),
            "llm_cot":    load_run(args.llm_results, filter_prompt_key="CoT"),
            "vlm_direct": load_run(args.vlm_results, filter_prompt_key=args.vlm_prompt_direct),
            "vlm_cot":    load_run(args.vlm_results, filter_prompt_key=args.vlm_prompt_cot),
        }
    else:
        raise ValueError("Need either 4 individual files (--llm_direct/--llm_cot/--vlm_direct/--vlm_cot) "
                          "or --llm_results + --vlm_results")

    judged_llm = load_judged(args.judged_llm)
    judged_vlm = load_judged(args.judged_vlm)
    if not judged_llm and not judged_vlm:
        print("[WARN] No --judged_llm/--judged_vlm given. Inference questions will be reported as "
              "not_judged (run judge_new.py first to score them).")

    # The judged file's "condition" field is whatever prompt_key value was on
    # the raw record (see judge_new.py's CONDITION_KEYS), so it must match
    # the same prompt_key values used to build runs_raw above.
    line_judged = {
        "llm_direct": (judged_llm, "DA"), "llm_cot": (judged_llm, "CoT"),
        "vlm_direct": (judged_vlm, args.vlm_prompt_direct), "vlm_cot": (judged_vlm, args.vlm_prompt_cot),
    }

    # 判分
    scores = {name: score_run(eval_set, run, *line_judged[name], args.missing_policy)
              for name, run in runs_raw.items()}

    # 对齐：取四条线都有结果的 qid
    common = set(eval_set.keys())
    for s in scores.values():
        common &= set(s.keys())
    common = sorted(common, key=str)

    missing_report = {name: sorted((set(eval_set) - set(scores[name])), key=str) for name in runs_raw}

    yesno_q = [q for q in common if eval_set[q]["qtype"] == "yesno"]
    infer_q = [q for q in common if eval_set[q]["qtype"] == "inference"]
    infer_judged_q = [q for q in infer_q if all(scores[line][q]["method"] != "not_judged" for line in scores)]

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
        "inference_all": block(infer_q),
        "inference_judged_only": block(infer_judged_q),
        "inference_not_judged_count": len(infer_q) - len(infer_judged_q),
        "inference_needs_review": sum(
            scores["llm_cot"][q]["needs_review"] or scores["vlm_cot"][q]["needs_review"]
            for q in infer_judged_q),
    }
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # per_item 明细
    with open(os.path.join(args.out_dir, "per_item.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "case_id", "qtype", "ground_truth",
                    "A_llm_direct", "B_llm_cot", "C_vlm_direct", "D_vlm_cot",
                    "needs_review", "inference_judged"])
        for q in common:
            it = eval_set[q]
            review = scores["llm_cot"][q]["needs_review"] or scores["vlm_cot"][q]["needs_review"]
            judged_ok = it["qtype"] != "inference" or q in infer_judged_q
            w.writerow([q, it["case_id"], it["qtype"], it.get("ground_truth", ""),
                        scores["llm_direct"][q]["correct"], scores["llm_cot"][q]["correct"],
                        scores["vlm_direct"][q]["correct"], scores["vlm_cot"][q]["correct"],
                        review, judged_ok])

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
    write_rq3(infer_judged_q, "rq3_inference.csv")

    # 终端摘要
    print("==== 对比摘要 ====")
    print(f"对齐题数: {summary['n_total_aligned']}  缺题: {summary['missing_counts']}")
    for tname in ["yes_no", "inference_all", "inference_judged_only"]:
        blk = summary[tname]
        print(f"\n[{tname}] n={blk['n']}")
        for k, v in blk["acc"].items():
            print(f"  {k:16s} = {v:.3f}" if v is not None else f"  {k:16s} = NA")
        print(f"  RQ1 (B-A) p={blk['RQ1_llm_cot_vs_direct_(B-A)']['p_value']}")
        print(f"  RQ2 (D-C) p={blk['RQ2_vlm_cot_vs_direct_(D-C)']['p_value']}")
    print(f"\ninference 未被 judge 评分的题数: {summary['inference_not_judged_count']}")
    print(f"inference 需人工复核的题数: {summary['inference_needs_review']}")
    print(f"\n详细结果写到: {args.out_dir}/")


if __name__ == "__main__":
    main()
