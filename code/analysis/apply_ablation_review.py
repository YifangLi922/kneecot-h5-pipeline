"""
apply_ablation_review.py
--------------------------
把两位同学人工复核的结果（sample_ablation_review.py 产出的 csv，人工填好
my_verdict / agree_with_judge / disagreement_note 三列之后）合并回 VLM
image-only ablation 的完整 judge 输出（results/judged_inference_vlm_ablation.jsonl，
400 条 inference 记录：DA 200 条 + CoT 200 条），产出
vlm_findings_ablation.py 需要的 judged_inference_vlm_ablation.json。

主线 2x2 对比当时（commit d289161）是手工改 json 应用复核结果的，这里写成
脚本，方便复现、留痕，以后要是再抽一批复核也能直接复用。

没被抽中复核的题目（分层抽样里 incorrect 桶没抽到的那部分）维持 judge 的
原始判断不变——这是有意为之：分层抽样只对抽中的子集做人工核验，没抽到的
题目仍按 judge 的结论计数。这也是为什么复核覆盖率要在论文里写清楚。

用法（在仓库根目录下）：
    python code/analysis/apply_ablation_review.py \
        --raw_judged results/judged_inference_vlm_ablation.jsonl \
        --review_csv results/vlm_ablation_review_sheet_completed.csv \
        --out_json results/judged_inference_vlm_ablation.json
"""
import json
import csv
import argparse
from collections import Counter

VALID_VERDICTS = {"correct", "incorrect", "unclear"}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_judged", required=True, help="judged_inference_vlm_ablation.jsonl（400条，未合并复核结果）")
    ap.add_argument("--review_csv", required=True, help="填好 my_verdict/agree_with_judge 的复核表")
    ap.add_argument("--out_json", required=True, help="合并后的 JSON 数组，供 vlm_findings_ablation.py 使用")
    args = ap.parse_args()

    raw = load_jsonl(args.raw_judged)
    by_key = {(r.get("question_id"), r.get("condition")): r for r in raw}
    print(f"读入 {len(raw)} 条原始 judge 结果（{args.raw_judged}）")
    if len(by_key) != len(raw):
        print(f"[警告] (question_id, condition) 有重复，去重后剩 {len(by_key)} 条，"
              "请检查 raw_judged 是否含重复记录。")

    with open(args.review_csv, encoding="utf-8-sig") as f:
        review_rows = list(csv.DictReader(f))
    print(f"读入 {len(review_rows)} 条复核表记录（{args.review_csv}）")

    applied = Counter()
    unresolved = []
    unfilled = 0
    flips = []

    for row in review_rows:
        qid, cond = row.get("question_id"), row.get("condition")
        target = by_key.get((qid, cond))
        if target is None:
            unresolved.append((qid, cond))
            continue

        my_verdict = (row.get("my_verdict") or "").strip().lower()
        agree = (row.get("agree_with_judge") or "").strip().lower()

        if not my_verdict and not agree:
            unfilled += 1
            continue  # 这一行还没填，跳过——不能假装已经复核过
        if my_verdict and my_verdict not in VALID_VERDICTS:
            raise ValueError(f"{qid}/{cond}: my_verdict 只能是 correct/incorrect/unclear，读到 {my_verdict!r}")

        judge_verdict = target.get("verdict")
        # agree_with_judge 优先；没填这一列就用 my_verdict 是否等于 judge 的 verdict 来判断
        agrees = (agree == "yes") if agree else (my_verdict == judge_verdict)
        applied["reviewed"] += 1

        if agrees:
            applied["agree"] += 1
            continue

        applied["disagree"] += 1
        final_verdict = my_verdict or judge_verdict
        new_correct = (final_verdict == "correct")
        if target.get("correct") != new_correct or target.get("verdict") != final_verdict:
            flips.append({
                "question_id": qid, "condition": cond,
                "judge_verdict": judge_verdict, "judge_correct": target.get("correct"),
                "human_verdict": final_verdict, "human_correct": new_correct,
            })
            target["verdict"] = final_verdict
            target["correct"] = new_correct
            target["human_reviewed"] = True
            target["human_review_note"] = row.get("disagreement_note", "")

    if unfilled:
        print(f"\n[提示] {unfilled} 行 my_verdict / agree_with_judge 都是空的，已跳过（不算已复核）。")
    if unresolved:
        print(f"\n[警告] {len(unresolved)} 条复核记录在原始 judge 文件里找不到对应 (question_id, condition)，已跳过：")
        for k in unresolved[:10]:
            print("   ", k)

    print(f"\n已复核: {applied['reviewed']} 条 | 同意 judge: {applied['agree']} 条 | 不同意: {applied['disagree']} 条")
    if applied["reviewed"]:
        print(f"人工-judge 一致率: {applied['agree'] / applied['reviewed']:.1%}")

    print(f"\n实际改动 verdict/correct 的记录: {len(flips)}")
    for f_ in flips:
        print(f"  {f_['question_id']} ({f_['condition']}): judge={f_['judge_verdict']} -> human={f_['human_verdict']}")

    out_rows = list(by_key.values())
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    print(f"\n写入 {len(out_rows)} 条（含未复核题目的原始 judge 判断 + 已应用的人工修正）: {args.out_json}")


if __name__ == "__main__":
    main()
