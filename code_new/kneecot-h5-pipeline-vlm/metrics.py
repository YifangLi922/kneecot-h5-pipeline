"""
metrics.py  –  Step 4: aggregate results, print RQ2/RQ3 tables, plot charts,
               save vlm_results to vlm_results/ folder.
Usage:
    python metrics.py
"""
import os, json, glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate import calculate_metrics
from config import RESULTS_DIR


# ── helpers ───────────────────────────────────────────────────────────────────

def load_yn_results(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_yn.json"))):
        fname = os.path.basename(path).replace("_yn.json", "")
        prompt = "unknown"
        model  = fname
        for label in ["CoT_findings", "DA_findings", "CoT", "DA"]:
            if fname.endswith("_" + label):
                model  = fname[: -(len(label) + 1)]
                prompt = label
                break

        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)

        if not results:
            print(f"  [SKIP] {os.path.basename(path)} — empty results")
            continue

        m = calculate_metrics(results)
        if m.get("total", 0) == 0:
            print(f"  [SKIP] {os.path.basename(path)} — 0 evaluated cases")
            continue

        rows.append({
            "Model":     model,
            "Prompt":    prompt,
            "Accuracy":  m.get("accuracy"),
            "F1":        m.get("f1"),
            "Precision": m.get("precision"),
            "Recall":    m.get("recall"),
            "Correct":   m.get("correct"),
            "Total":     m.get("total"),
            "UNCLEAR":   m.get("unclear"),
            "UNCLEAR%":  m.get("unclear_pct"),
        })
    return pd.DataFrame(rows)


def load_case_level(results_dir):
    """
    Build a per-case comparison table for RQ3 case-type distribution.
    Compares DA_findings vs DA for each model.
    """
    case_rows = []
    for model_dir_path in sorted(glob.glob(os.path.join(results_dir, "*_DA_yn.json"))):
        fname  = os.path.basename(model_dir_path)
        model  = fname.replace("_DA_yn.json", "")
        da_path  = os.path.join(results_dir, f"{model}_DA_yn.json")
        daf_path = os.path.join(results_dir, f"{model}_DA_findings_yn.json")
        if not (os.path.exists(da_path) and os.path.exists(daf_path)):
            continue
        with open(da_path,  "r", encoding="utf-8") as f: da_res  = json.load(f)
        with open(daf_path, "r", encoding="utf-8") as f: daf_res = json.load(f)

        # index by case_id + question
        da_map  = {(r["case_id"], r["question"]): r for r in da_res}
        daf_map = {(r["case_id"], r["question"]): r for r in daf_res}

        for key in set(da_map) & set(daf_map):
            da_r  = da_map[key]
            daf_r = daf_map[key]
            da_ok  = (da_r["prediction"]  == da_r["ground_truth"])
            daf_ok = (daf_r["prediction"] == daf_r["ground_truth"])
            if da_ok and daf_ok:
                case_type = "easy"
            elif not da_ok and not daf_ok:
                case_type = "hard"
            elif not da_ok and daf_ok:
                case_type = "findings_helps"
            else:
                case_type = "findings_hurts"
            case_rows.append({
                "model": model, "case_id": key[0],
                "comparison": "DA_findings_vs_DA", "case_type": case_type,
            })
    return pd.DataFrame(case_rows)


def _get_acc(df, model, prompt):
    row = df[(df["Model"] == model) & (df["Prompt"] == prompt)]
    if len(row) == 0:
        return None
    return row["Accuracy"].values[0]


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df       = load_yn_results(str(RESULTS_DIR))
    df_cases = load_case_level(str(RESULTS_DIR))

    if df.empty:
        print("No results found — run evaluate.py first.")
    else:
        # ── Print table ───────────────────────────────────────────────────────
        print("\n=== VLM Performance (Yes/No) ===")
        print(df.to_string(index=False))
        csv_path = os.path.join(str(RESULTS_DIR), "vlm_yn_results_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved → {csv_path}")

        # ── RQ2: CoT vs DA (within each input modality) ──────────────────────
        print(f"\n{'─'*55}")
        print("RQ2: CoT vs DA — does reasoning help?")
        print(f"{'─'*55}")
        for model in df["Model"].unique():
            da_acc   = _get_acc(df, model, "DA")
            cot_acc  = _get_acc(df, model, "CoT")
            daf_acc  = _get_acc(df, model, "DA_findings")
            cotf_acc = _get_acc(df, model, "CoT_findings")
            print(f"  {model}")
            if da_acc is not None and cot_acc is not None:
                print(f"    CoT vs DA (images only):          {(cot_acc - da_acc)*100:+.1f} pp")
            if daf_acc is not None and cotf_acc is not None:
                print(f"    CoT_findings vs DA_findings (images+text): {(cotf_acc - daf_acc)*100:+.1f} pp")

        # ── RQ3: findings vs no-findings (within each prompt style) ──────────
        print(f"\n{'─'*55}")
        print("RQ3: findings vs no-findings — does text help?")
        print(f"{'─'*55}")
        for model in df["Model"].unique():
            da_acc   = _get_acc(df, model, "DA")
            cot_acc  = _get_acc(df, model, "CoT")
            daf_acc  = _get_acc(df, model, "DA_findings")
            cotf_acc = _get_acc(df, model, "CoT_findings")
            print(f"  {model}")
            if da_acc is not None and daf_acc is not None:
                print(f"    DA_findings vs DA (direct answer):   {(daf_acc - da_acc)*100:+.1f} pp")
            if cot_acc is not None and cotf_acc is not None:
                print(f"    CoT_findings vs CoT (CoT reasoning): {(cotf_acc - cot_acc)*100:+.1f} pp")

        # ── Plots ─────────────────────────────────────────────────────────────
        models_list = sorted(df["Model"].unique())
        x, w = range(len(models_list)), 0.35

        def get_acc_list(prompt):
            return [_get_acc(df, m, prompt) or 0 for m in models_list]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: RQ2 — CoT vs DA (images only)
        da_acc_l   = get_acc_list("DA")
        cot_acc_l  = get_acc_list("CoT")
        b1 = axes[0,0].bar([xi - w/2 for xi in x], da_acc_l,  w, label="DA",  color="#3498db", alpha=0.85)
        b2 = axes[0,0].bar([xi + w/2 for xi in x], cot_acc_l, w, label="CoT", color="#e74c3c", alpha=0.85)
        axes[0,0].set_xticks(list(x)); axes[0,0].set_xticklabels(models_list, rotation=15, ha="right")
        axes[0,0].set_ylabel("Accuracy"); axes[0,0].set_ylim(0, 1.15)
        axes[0,0].set_title("RQ2 — CoT vs DA (images only)")
        axes[0,0].legend(); axes[0,0].grid(axis="y", alpha=0.3)
        for bar in list(b1) + list(b2):
            axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                           f"{bar.get_height():.2f}", ha="center", fontsize=9)

        # Plot 2: RQ2 — CoT_findings vs DA_findings (images + text)
        daf_acc_l  = get_acc_list("DA_findings")
        cotf_acc_l = get_acc_list("CoT_findings")
        b3 = axes[0,1].bar([xi - w/2 for xi in x], daf_acc_l,  w, label="DA_findings",  color="#3498db", alpha=0.85)
        b4 = axes[0,1].bar([xi + w/2 for xi in x], cotf_acc_l, w, label="CoT_findings", color="#e74c3c", alpha=0.85)
        axes[0,1].set_xticks(list(x)); axes[0,1].set_xticklabels(models_list, rotation=15, ha="right")
        axes[0,1].set_ylabel("Accuracy"); axes[0,1].set_ylim(0, 1.15)
        axes[0,1].set_title("RQ2 — CoT vs DA (images + MR findings)")
        axes[0,1].legend(); axes[0,1].grid(axis="y", alpha=0.3)
        for bar in list(b3) + list(b4):
            axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                           f"{bar.get_height():.2f}", ha="center", fontsize=9)

        # Plot 3: RQ3 — accuracy delta (findings vs no-findings)
        delta_da  = [f - d for f, d in zip(daf_acc_l,  da_acc_l)]
        delta_cot = [f - c for f, c in zip(cotf_acc_l, cot_acc_l)]
        b5 = axes[1,0].bar([xi - w/2 for xi in x], delta_da,  w, label="DA_findings vs DA",   color="#9b59b6", alpha=0.85)
        b6 = axes[1,0].bar([xi + w/2 for xi in x], delta_cot, w, label="CoT_findings vs CoT", color="#f39c12", alpha=0.85)
        axes[1,0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1,0].set_xticks(list(x)); axes[1,0].set_xticklabels(models_list, rotation=15, ha="right")
        axes[1,0].set_ylabel("Accuracy Delta"); axes[1,0].set_title("RQ3 — Gain from Adding MR Findings")
        axes[1,0].legend(); axes[1,0].grid(axis="y", alpha=0.3)
        for bar in list(b5) + list(b6):
            axes[1,0].text(bar.get_x()+bar.get_width()/2,
                           bar.get_height() + (0.005 if bar.get_height() >= 0 else -0.015),
                           f"{bar.get_height():+.2f}", ha="center", fontsize=9)

        # Plot 4: RQ3 — case type distribution (DA_findings vs DA)
        if not df_cases.empty:
            rq3_df     = df_cases[df_cases["comparison"] == "DA_findings_vs_DA"]
            case_types = ["easy", "findings_helps", "findings_hurts", "hard"]
            colors_rq3 = {"easy": "#2ecc71", "findings_helps": "#3498db",
                          "findings_hurts": "#e74c3c", "hard": "#95a5a6"}
            dist = rq3_df.groupby(["model", "case_type"]).size().unstack(fill_value=0)
            dist = dist.reindex(columns=case_types, fill_value=0)
            bar_x = range(len(dist.index))
            for j, ct in enumerate(case_types):
                axes[1,1].bar([xi + j*w/2 for xi in bar_x], dist[ct].values, w/2,
                              label=ct, color=colors_rq3[ct], alpha=0.85)
            axes[1,1].set_xticks(list(bar_x))
            axes[1,1].set_xticklabels([m.split(":")[0] for m in dist.index], rotation=15, ha="right")
            axes[1,1].set_ylabel("Cases"); axes[1,1].set_title("RQ3 — Case Type: DA_findings vs DA")
            axes[1,1].legend(fontsize=8); axes[1,1].grid(axis="y", alpha=0.3)
        else:
            axes[1,1].set_visible(False)

        plt.tight_layout()
        fig_path = os.path.join(str(RESULTS_DIR), "vlm_plots.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {fig_path}")
        plt.close()
