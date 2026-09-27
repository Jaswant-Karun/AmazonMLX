#!/usr/bin/env python3
"""
Phase 3: Systematic Threshold Optimization for Macro F0.5.
Searches coarse and fine thresholds on validation set to maximize official Macro F0.5.
Outputs:
  - output/optimal_threshold_v2.json
  - experiments/threshold_analysis_v2.tsv
"""

import sys
import os
import json
import collections
import pandas as pd
import numpy as np
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(base_dir, "output")
    exp_dir = os.path.join(base_dir, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "entity_matching_model.joblib")
    val_path = os.path.join(out_dir, "val_features.parquet")
    gt_path = os.path.join(base_dir, "..", "dataset", "train", "train_ground_truth.tsv")

    print("=" * 80)
    print("PHASE 3: SYSTEMATIC THRESHOLD SEARCH FOR MACRO F0.5")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Val path  : {val_path}")

    model = joblib.load(model_path)
    val_df = pd.read_parquet(val_path)

    feature_cols = [c for c in val_df.columns if c not in {"source1_entity_id", "target_entity_id", "target_source", "label"}]
    X_val = val_df[feature_cols].values
    y_probs = model.predict_proba(X_val)[:, 1]

    all_val_s1 = sorted(val_df["source1_entity_id"].unique())
    print(f"Total validation S1 entities: {len(all_val_s1):,}")

    gt_df = pd.read_csv(gt_path, sep="\t", keep_default_na=False)
    gt_map = collections.defaultdict(set)
    val_s1_set = set(all_val_s1)
    for s1_id, m_str in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if s1_id in val_s1_set and m_str.strip():
            for m in m_str.split(","):
                gt_map[s1_id].add(m.strip())

    total_gt = sum(len(v) for v in gt_map.values())
    singletons = [s1 for s1 in all_val_s1 if len(gt_map[s1]) == 0]
    n_singletons = len(singletons)
    print(f"Total GT matches: {total_gt:,} | Singletons: {n_singletons:,} ({n_singletons/len(all_val_s1):.2%})")

    # Group pairs by s1_id for fast threshold evaluation
    s1_pairs = collections.defaultdict(list)
    for s1_id, t_id, prob in zip(val_df["source1_entity_id"], val_df["target_entity_id"], y_probs):
        s1_pairs[s1_id].append((t_id, float(prob)))

    def eval_th(th):
        f05_list = []
        exact_count = 0
        singleton_fp_count = 0
        total_tp = 0
        total_fp = 0
        total_preds = 0

        for s1_id in all_val_s1:
            gt_set = gt_map.get(s1_id, set())
            pairs = s1_pairs.get(s1_id, [])
            pred_set = {t_id for t_id, p in pairs if p >= th}

            len_gt = len(gt_set)
            len_p = len(pred_set)
            total_preds += len_p

            if len_gt == 0:
                if len_p == 0:
                    f05_list.append(1.0)
                    exact_count += 1
                else:
                    f05_list.append(0.0)
                    singleton_fp_count += 1
                    total_fp += len_p
            else:
                if len_p == 0:
                    f05_list.append(0.0)
                else:
                    tp = len(gt_set & pred_set)
                    fp = len_p - tp
                    total_tp += tp
                    total_fp += fp

                    prec = tp / len_p
                    rec = tp / len_gt
                    denom = 0.25 * prec + rec
                    f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0
                    f05_list.append(f05)
                    if gt_set == pred_set:
                        exact_count += 1

        macro_f05 = float(np.mean(f05_list))
        pair_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        pair_r = total_tp / total_gt if total_gt else 0.0
        singleton_fp_rate = singleton_fp_count / n_singletons if n_singletons else 0.0
        avg_preds = total_preds / len(all_val_s1)
        exact_rate = exact_count / len(all_val_s1)

        return {
            "threshold": round(th, 4),
            "macro_f05": round(macro_f05, 5),
            "pair_precision": round(pair_p, 5),
            "pair_recall": round(pair_r, 5),
            "singleton_fp_rate": round(singleton_fp_rate, 5),
            "avg_predictions": round(avg_preds, 4),
            "exact_match_rate": round(exact_rate, 5)
        }

    coarse_thresholds = [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.4365,
        0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    ]

    results = []
    print("\n[Step 1/2] Running coarse threshold search...")
    for th in coarse_thresholds:
        res = eval_th(th)
        results.append(res)
        print(
            f"  Th={res['threshold']:<6.4f} | Macro F0.5={res['macro_f05']:<7.4f} | "
            f"Prec={res['pair_precision']:<7.4f} | Rec={res['pair_recall']:<7.4f} | "
            f"S_FP={res['singleton_fp_rate']:<6.2%} | AvgPred={res['avg_predictions']:<5.2f} | "
            f"Exact={res['exact_match_rate']:<6.2%}"
        )

    # Find best coarse region
    best_coarse = max(results, key=lambda x: x["macro_f05"])
    best_th = best_coarse["threshold"]
    print(f"\nBest coarse threshold: {best_th} (Macro F0.5: {best_coarse['macro_f05']:.4f})")

    # Fine search around best region (+/- 0.08 in steps of 0.01)
    existing_th = {r["threshold"] for r in results}
    fine_thresholds = [round(best_th + d, 4) for d in np.arange(-0.08, 0.085, 0.01)]
    fine_thresholds = [t for t in fine_thresholds if 0.01 <= t <= 0.99 and t not in existing_th]

    print(f"\n[Step 2/2] Running fine search around {best_th} across {len(fine_thresholds)} points...")
    for th in sorted(fine_thresholds):
        res = eval_th(th)
        results.append(res)
        print(
            f"  Th={res['threshold']:<6.4f} | Macro F0.5={res['macro_f05']:<7.4f} | "
            f"Prec={res['pair_precision']:<7.4f} | Rec={res['pair_recall']:<7.4f} | "
            f"S_FP={res['singleton_fp_rate']:<6.2%} | AvgPred={res['avg_predictions']:<5.2f} | "
            f"Exact={res['exact_match_rate']:<6.2%}"
        )

    # Sort results by threshold
    results = sorted(results, key=lambda x: x["threshold"])
    df_res = pd.DataFrame(results)
    tsv_path = os.path.join(exp_dir, "threshold_analysis_v2.tsv")
    df_res.to_csv(tsv_path, sep="\t", index=False)
    print(f"\nSaved full threshold analysis to: {tsv_path}")

    # Best overall
    best_overall = max(results, key=lambda x: x["macro_f05"])
    print("\n" + "=" * 80)
    print(f"OPTIMAL THRESHOLD FOUND: {best_overall['threshold']}")
    print(f"Validation Macro F0.5  : {best_overall['macro_f05']:.5f} (vs. 0.9280 at 0.4365, +{best_overall['macro_f05'] - 0.9280:+.4f})")
    print(f"Pair Precision         : {best_overall['pair_precision']:.4f}")
    print(f"Pair Recall            : {best_overall['pair_recall']:.4f}")
    print(f"Singleton FP Rate      : {best_overall['singleton_fp_rate']:.2%}")
    print(f"Exact Match Rate       : {best_overall['exact_match_rate']:.2%}")
    print("=" * 80)

    out_cfg = {
        "optimal_threshold": best_overall["threshold"],
        "macro_f05": best_overall["macro_f05"],
        "pair_precision": best_overall["pair_precision"],
        "pair_recall": best_overall["pair_recall"],
        "singleton_fp_rate": best_overall["singleton_fp_rate"],
        "avg_predictions": best_overall["avg_predictions"],
        "exact_match_rate": best_overall["exact_match_rate"],
        "baseline_threshold": 0.4365,
        "baseline_macro_f05": 0.9280,
        "metric": "Macro F0.5 per Source-1 entity",
        "search_range": [0.05, 0.95],
        "sample_size": len(all_val_s1)
    }
    cfg_path = os.path.join(out_dir, "optimal_threshold_v2.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2)
    print(f"Saved optimal threshold config to: {cfg_path}")

if __name__ == "__main__":
    main()
