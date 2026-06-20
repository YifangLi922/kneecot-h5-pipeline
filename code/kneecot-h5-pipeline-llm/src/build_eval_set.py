"""
build_eval_set.py
-------------------
生成一份「冻结的」共享评测题表，LLM 线和 VLM 线都从这份表读题，
保证两条线测的是完全相同的 (case_id, question_id)。
本版本同时收录 yes/no 题和 inference 题。

用法（不需要 GPU，几秒跑完）：
    python build_eval_set.py --data_dir data/cases \
        --n_yesno 50 --n_inference 30 --seed 42 --out eval_set.json

    # n_yesno=0     表示用全部均衡的 yes/no 题池
    # n_inference=0 表示用全部 inference 题

输出 eval_set.json 里每道题一条记录（flat list，与 VLM 格式一致）：
  yes/no 题：
    {
      "case_id": "GJB0000001",
      "question_id": "GJB0000001_q5",
      "question": "……是否存在关节腔积液？",
      "full_answer": "Yes。关节腔内及关节囊内可见积液信号。",
      "qtype": "yesno",
      "ground_truth": "Yes"
    }
  inference 题：
    {
      "case_id": "GJB0000001",
      "question_id": "GJB0000001_q33",
      "question": "……该患者的半月板损伤属于几级？请说明推理依据。",
      "full_answer": "属于 Stoller I-II 级。推理依据：……",
      "qtype": "inference",
      "ground_truth": "属于 Stoller I-II 级。推理依据：……"
    }
"""

import os
import re
import json
import random
import argparse

# yes/no 题：题目必须含其中一个中文是非疑问词
YES_NO_MARKERS = ["是否", "有无", "是不是", "有没有", "能否"]

# 非膝关节部位 → 整个病例剔除
NON_KNEE_JOINTS = ["踝", "肩", "髋", "腕", "肘"]

# inference 答案里「结论」和「依据」的分隔词
REASON_SEPARATORS = ["推理依据", "依据", "理由"]

# 每个病例最多收几道 inference 题（避免单个病例刷屏）
MAX_INFERENCE_PER_CASE = 2


def load_cases(data_dir):
    """读取目录下所有 .json 病例文件。"""
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            try:
                yield json.load(f)
            except json.JSONDecodeError:
                print(f"[warn] 跳过无法解析的文件: {fname}")


def is_knee_only(case):
    """检查方法里必须含「膝关节」且不含「肩关节」（与 VLM 过滤逻辑一致）。"""
    method = case.get("检查方法", "") or ""
    return "膝关节" in method and "肩关节" not in method


def get_yesno_label(answer):
    """yes/no 答案必须以干净的 Yes / No 开头，返回 'Yes'/'No'/None。"""
    a = answer.strip()
    if a.startswith("Yes"):
        return "Yes"
    if a.startswith("No"):
        return "No"
    return None


def has_marker(question):
    return any(m in question for m in YES_NO_MARKERS)


def build_candidates(data_dir):
    """遍历所有病例和题目，分别收集 yes/no 候选和 inference 候选。"""
    yesno, inference = [], []

    for case in load_cases(data_dir):
        case_id = case.get("顺序编号")
        if not case_id or not is_knee_only(case):
            continue

        qa_pairs = case.get("问答数据", {}).get("qa_pairs", [])
        seen_yesno = {"Yes": False, "No": False}   # 每病例每极性最多 1 道
        inf_count = 0                               # 每病例 inference 计数

        for idx, qa in enumerate(qa_pairs):
            qtype = qa.get("type")
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            qid = f"{case_id}_q{idx}"

            if qtype == "yes_no":
                label = get_yesno_label(answer)
                if label is None or not has_marker(question):
                    continue
                if seen_yesno[label]:
                    continue
                seen_yesno[label] = True
                yesno.append({
                    "case_id": case_id, "question_id": qid,
                    "question": question, "full_answer": answer,
                    "qtype": "yesno", "ground_truth": label,
                })

            elif qtype == "inference":
                if inf_count >= MAX_INFERENCE_PER_CASE:
                    continue
                if not answer:
                    continue
                inf_count += 1
                inference.append({
                    "case_id": case_id, "question_id": qid,
                    "question": question, "full_answer": answer,
                    "qtype": "inference", "ground_truth": answer,
                })

    return yesno, inference


def sample_yesno(candidates, n, seed):
    """yes/no：分层均衡抽样（Yes/No 各一半）。n=0 表示全量均衡。"""
    yes = [c for c in candidates if c["ground_truth"] == "Yes"]
    no  = [c for c in candidates if c["ground_truth"] == "No"]
    rng = random.Random(seed)
    rng.shuffle(yes); rng.shuffle(no)
    if n and n > 0:
        half = n // 2
        if len(yes) < half or len(no) < half:
            print(f"[warn] yes/no 题不够：Yes={len(yes)} No={len(no)} 想要每极 {half}")
        per = min(half, len(yes), len(no))
    else:
        per = min(len(yes), len(no))
    chosen = yes[:per] + no[:per]
    rng.shuffle(chosen)
    return chosen


def sample_inference(candidates, n, seed):
    """inference：直接随机抽 n 道（无 Yes/No 之分）。n=0 表示全部。"""
    rng = random.Random(seed + 1)   # 用不同种子，避免和 yes/no 抽样耦合
    pool = list(candidates)
    rng.shuffle(pool)
    if n and n > 0:
        if len(pool) < n:
            print(f"[warn] inference 题不够：有 {len(pool)} 道，想要 {n} 道")
        return pool[:n]
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="存放 KneeCoT 病例 JSON 的目录")
    ap.add_argument("--n_yesno", type=int, default=50, help="yes/no 题数；0=全量均衡")
    ap.add_argument("--n_inference", type=int, default=30, help="inference 题数；0=全部")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval_set.json")
    args = ap.parse_args()

    yesno_cand, inf_cand = build_candidates(args.data_dir)
    yesno     = sample_yesno(yesno_cand, args.n_yesno, args.seed)
    inference = sample_inference(inf_cand, args.n_inference, args.seed)
    items     = yesno + inference

    # Output as flat list — identical schema to VLM build_eval_set.py
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    n_yes = sum(c["ground_truth"] == "Yes" for c in yesno)
    n_no  = sum(c["ground_truth"] == "No"  for c in yesno)
    print(f"[ok] 写出 {len(items)} 道题 -> {args.out}")
    print(f"     yes/no={len(yesno)} (Yes={n_yes} No={n_no})  "
          f"inference={len(inference)}  覆盖病例={len({c['case_id'] for c in items})}")


if __name__ == "__main__":
    main()
