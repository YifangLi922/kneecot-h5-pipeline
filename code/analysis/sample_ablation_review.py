"""
sample_ablation_review.py
--------------------------
把 judge.py 的 make_manual_review_sample() 为 VLM image-only ablation 选出的
复核队列（results/manual_review_vlm_ablation.jsonl，359 条 = 344 条 incorrect
全量 + 4 条 unclear 全量 + 11 条 correct 抽 20%）进一步压缩，降低人工复核量。

背景：judge.py 的 manual_review_policy 写死 included_all_incorrect=True，
即"incorrect 判定一律全审"。这条策略对主线 2x2 对比（62 条待审）是可行的，
但 VLM image-only ablation 准确率只有 13%，incorrect 桶膨胀到 344 条，两个人
全审不现实。本脚本只对 incorrect 桶按 condition（DA/CoT）分层随机抽样
（默认 20%，和 correct 桶已有的抽样比例一致），unclear（本来就少，且定义
上就是有歧义、最值得看）和已经抽出的 correct 抽检保持不变、全部保留。

这是相对于 judge.py 默认复核协议的主动收窄，请在论文方法/局限性部分注明：
"对 image-only ablation 的 incorrect 判定做了 {rate:.0%} 分层抽样复核，
未做全量复核（主线 2x2 对比的 incorrect 判定做了全量复核）"。

用法（在仓库根目录下）：
    python code/analysis/sample_ablation_review.py \
        --input results/manual_review_vlm_ablation.jsonl \
        --out_jsonl results/manual_review_vlm_ablation_sampled.jsonl \
        --out_csv results/vlm_ablation_review_sheet.csv \
        --incorrect_sample_rate 0.20 --seed 42

产出两份文件：
    *_sampled.jsonl —— 给 apply_ablation_review.py 用的机器可读版本
    *_sheet.csv      —— 给人复核用，在 Excel / Google Sheets 里打开，
                         逐行填 my_verdict / agree_with_judge /
                         disagreement_note 三列
"""
import json
import csv
import random
import math
import argparse
from collections import Counter, defaultdict


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_sample(rows, rate, seed, strata_key="condition"):
    """按 strata_key 分层，每层各自抽 rate 比例（向上取整），seed 固定保证可复现。"""
    by_stratum = defaultdict(list)
    for r in rows:
        by_stratum[r.get(strata_key)].append(r)

    sampled = []
    for stratum, items in sorted(by_stratum.items(), key=str):
        rng = random.Random(f"{seed}:{stratum}")
        n = int(math.ceil(max(0.0, rate) * len(items)))
        n = min(n, len(items))
        sampled.extend(rng.sample(items, n) if n else [])
    return sampled


CSV_FIELDS = [
    "review_batch", "condition", "case_id", "question_id", "question_intent",
    "review_reason", "verdict", "question", "expected_answer",
    "candidate_model_answer", "reason", "ground_truth_core",
    "required_elements", "matched_elements", "contradictions",
    "extracted_model_conclusion", "mr_findings", "judge_model", "judged_at",
    "my_verdict", "agree_with_judge", "disagreement_note",
]


def as_text(v):
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v if v is not None else ""


def to_csv_row(r, batch_label):
    return {
        "review_batch": batch_label,
        "condition": r.get("condition", ""),
        "case_id": r.get("case_id", ""),
        "question_id": r.get("question_id", ""),
        "question_intent": r.get("question_intent", ""),
        "review_reason": r.get("review_reason", ""),
        "verdict": r.get("verdict", ""),
        "question": as_text(r.get("question")),
        "expected_answer": as_text(r.get("expected_answer")),
        "candidate_model_answer": as_text(r.get("candidate_model_answer")),
        "reason": as_text(r.get("reason")),
        "ground_truth_core": as_text(r.get("ground_truth_core")),
        "required_elements": as_text(r.get("required_elements")),
        "matched_elements": as_text(r.get("matched_elements")),
        "contradictions": as_text(r.get("contradictions")),
        "extracted_model_conclusion": as_text(r.get("extracted_model_conclusion")),
        "mr_findings": as_text(r.get("mr_findings")),
        "judge_model": r.get("judge_model", ""),
        "judged_at": r.get("judged_at", ""),
        "my_verdict": "",        # 人工填: correct / incorrect / unclear
        "agree_with_judge": "",  # 人工填: Yes / No
        "disagreement_note": "", # 人工填: 不同意 judge 时简单写明原因
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="results/manual_review_vlm_ablation.jsonl")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--incorrect_sample_rate", type=float, default=0.20,
                     help="incorrect 桶的分层抽样比例；unclear 和已有的 correct 抽检始终全部保留")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    print(f"读入 {len(rows)} 条候选（来自 {args.input}）")

    by_verdict = defaultdict(list)
    for r in rows:
        by_verdict[r.get("verdict")].append(r)
    print("原始 verdict 分布:", {k: len(v) for k, v in by_verdict.items()})

    incorrect_rows = by_verdict.get("incorrect", [])
    unclear_rows = by_verdict.get("unclear", [])
    other_rows = [r for v, rs in by_verdict.items() if v not in ("incorrect", "unclear") for r in rs]

    sampled_incorrect = stratified_sample(incorrect_rows, args.incorrect_sample_rate, args.seed)
    final_rows = sampled_incorrect + unclear_rows + other_rows

    print(f"\nincorrect 桶: {len(incorrect_rows)} 条 -> 按 condition 分层抽样 "
          f"{args.incorrect_sample_rate:.0%} -> {len(sampled_incorrect)} 条")
    print(f"unclear 桶: {len(unclear_rows)} 条 -> 全部保留")
    print(f"其余桶（correct 抽检等）: {len(other_rows)} 条 -> 全部保留")
    reduction = 1 - len(final_rows) / len(rows) if rows else 0
    print(f"最终复核队列: {len(final_rows)} 条（原 {len(rows)} 条，压缩 {reduction:.0%}）")

    cond_counter = Counter((r.get("condition"), r.get("verdict")) for r in final_rows)
    print("最终队列按 condition x verdict 分布:", dict(cond_counter))

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n写入: {args.out_jsonl}")

    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in final_rows:
            is_incorrect_sample = r.get("verdict") == "incorrect"
            batch_label = (f"incorrect_sample_{args.incorrect_sample_rate:.0%}"
                           if is_incorrect_sample else r.get("review_reason", ""))
            w.writerow(to_csv_row(r, batch_label))
    print(f"写入: {args.out_csv}")
    print("  -> 给复核的同学在 Excel / Google Sheets 里打开这份 csv，"
          "逐行填 my_verdict / agree_with_judge / disagreement_note 三列")


if __name__ == "__main__":
    main()
