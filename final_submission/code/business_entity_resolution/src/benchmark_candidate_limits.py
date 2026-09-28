#!/usr/bin/env python3
"""
Benchmark Candidate Selection Strategies & Adaptive Limits on Validation Data
Evaluates candidate set size, pair recall, entity recall, and official Macro-F0.5.
"""

import sys
import os
import time
import json
import collections
import pandas as pd
import numpy as np
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

def compute_macro_f05(gt_matches_dict, pred_matches_dict, all_s1_ids):
    """
    Computes official competition Macro F_0.5 across all S1 entities:
      F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    Singletons:
      - true singleton with empty prediction -> 1.0
      - true singleton with non-empty prediction -> 0.0
    """
    scores = []
    for s1_id in all_s1_ids:
        gt_set = gt_matches_dict.get(s1_id, set())
        pred_set = pred_matches_dict.get(s1_id, set())

        len_gt = len(gt_set)
        len_pred = len(pred_set)

        if len_gt == 0:
            scores.append(1.0 if len_pred == 0 else 0.0)
        elif len_pred == 0:
            scores.append(0.0)
        else:
            tp = len(gt_set & pred_set)
            if tp == 0:
                scores.append(0.0)
            else:
                prec = tp / len_pred
                rec = tp / len_gt
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                scores.append(f05)

    return float(np.mean(scores))


def main():
    t_start = time.time()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(base_dir, "output")
    exp_dir = os.path.join(base_dir, "experiments")
    train_dir = os.path.join(base_dir, "..", "dataset", "train")

    val_pairs_path = os.path.join(out_dir, "val_pairs.tsv")
    val_features_path = os.path.join(out_dir, "val_features.parquet")
    model_path = os.path.join(out_dir, "entity_matching_model.joblib")
    threshold_path = os.path.join(out_dir, "optimal_threshold.json")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print("=" * 85)
    print("AMAZON ML CHALLENGE 2026 — CANDIDATE LIMIT & MACRO-F0.5 BENCHMARK")
    print("=" * 85)

    # 1. Load model and threshold
    print(f"Loading LightGBM model from {model_path}...")
    model = joblib.load(model_path)
    with open(threshold_path, "r", encoding="utf-8") as f:
        th_cfg = json.load(f)
    optimal_th = float(th_cfg.get("optimal_threshold", 0.4365))
    feature_names = th_cfg["feature_names"]
    print(f"  Model loaded. Operational decision threshold: {optimal_th:.4f}")

    # 2. Load validation features & pairs
    print(f"\nLoading validation features from {val_features_path}...")
    t0 = time.time()
    df_feat = pd.read_parquet(val_features_path)
    print(f"  Loaded {len(df_feat):,} validation pairs in {time.time() - t0:.2f}s.")

    # Model inference probabilities
    print("Computing validation model probabilities...")
    t0 = time.time()
    X_val = df_feat[feature_names].values
    val_probs = model.predict_proba(X_val)[:, 1]
    df_feat["model_prob"] = val_probs
    print(f"  Scored {len(val_probs):,} pairs in {time.time() - t0:.2f}s.")

    # 3. Load ground truth for validation S1 entities
    val_s1_ids = sorted(list(df_feat["source1_entity_id"].unique()))
    val_s1_set = set(val_s1_ids)
    print(f"\nLoading ground truth for {len(val_s1_ids):,} validation entities...")
    gt_dict = collections.defaultdict(set)
    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=200000, keep_default_na=False):
        matching = chunk[chunk["source1_entity_id"].isin(val_s1_set)]
        for s1_id, m_str in zip(matching["source1_entity_id"], matching["matched_entity_ids"]):
            m_str = m_str.strip()
            if m_str:
                for m in m_str.split(","):
                    if m.strip():
                        gt_dict[s1_id].add(m.strip())

    total_gt_pairs = sum(len(s) for s in gt_dict.values())
    gt_entities = set(k for k, v in gt_dict.items() if len(v) > 0)
    print(f"  Total validation ground truth matches: {total_gt_pairs:,} across {len(gt_entities):,} non-singleton entities.")

    # 4. Group pairs by S1 entity
    print("\nIndexing validation candidates per entity...")
    pairs_per_s1 = collections.defaultdict(list)
    for row in df_feat[["source1_entity_id", "target_entity_id", "num_blocks_matched", "label", "model_prob"]].itertuples(index=False):
        # row: (source1_entity_id, target_entity_id, num_blocks_matched, label, model_prob)
        pairs_per_s1[row.source1_entity_id].append(row)

    # Candidate ranking logic based on blocking evidence (BEFORE ML inference)
    # Primary: num_blocks_matched descending
    # Secondary: deterministic target_id ordering
    for s1_id in pairs_per_s1:
        pairs_per_s1[s1_id].sort(key=lambda r: (r.num_blocks_matched), reverse=True)

    # 5. Evaluate Candidate Selection Strategies
    strategies = [
        ("Full Blocker (Uncapped)", "all"),
        ("Limit Top 5", 5),
        ("Limit Top 10", 10),
        ("Limit Top 15", 15),
        ("Limit Top 20", 20),
        ("Adaptive A (All multi-block >=2 + Top 5 single)", "adaptive_5"),
        ("Adaptive B (All multi-block >=2 + Top 10 single)", "adaptive_10"),
        ("Adaptive C (All multi-block >=2 + Top 15 single)", "adaptive_15"),
    ]

    print("\n" + "=" * 128)
    print(f"{'Strategy':<30} | {'Candidates':>10} | {'Avg/S1':>6} | {'Med':>3} | {'Max':>4} | {'Zero Cands':>10} | {'Pair Rec %':>10} | {'Ent Rec %':>9} | {'Lost GT':>7} | {'Macro F0.5':>10}")
    print("-" * 128)

    results_table = []

    for strat_name, strat_param in strategies:
        cand_counts = []
        zero_cands = 0
        retained_gt_pairs = 0
        retained_gt_entities = set()
        total_cands = 0

        pred_matches = collections.defaultdict(set)

        for s1_id in val_s1_ids:
            all_cands = pairs_per_s1.get(s1_id, [])
            if not all_cands:
                cand_counts.append(0)
                zero_cands += 1
                continue

            if strat_param == "all":
                selected = all_cands
            elif isinstance(strat_param, int):
                selected = all_cands[:strat_param]
            elif strat_param == "adaptive_5":
                multi = [r for r in all_cands if r.num_blocks_matched >= 2]
                single = [r for r in all_cands if r.num_blocks_matched == 1][:5]
                selected = multi + single
            elif strat_param == "adaptive_10":
                multi = [r for r in all_cands if r.num_blocks_matched >= 2]
                single = [r for r in all_cands if r.num_blocks_matched == 1][:10]
                selected = multi + single
            elif strat_param == "adaptive_15":
                multi = [r for r in all_cands if r.num_blocks_matched >= 2]
                single = [r for r in all_cands if r.num_blocks_matched == 1][:15]
                selected = multi + single

            n_c = len(selected)
            total_cands += n_c
            cand_counts.append(n_c)
            if n_c == 0:
                zero_cands += 1

            # Check true match recall within candidate set
            gt_set = gt_dict.get(s1_id, set())
            for r in selected:
                if r.target_entity_id in gt_set:
                    retained_gt_pairs += 1
                    retained_gt_entities.add(s1_id)

                # ML prediction step on candidate set
                if r.model_prob >= optimal_th:
                    pred_matches[s1_id].add(r.target_entity_id)

        avg_c = total_cands / len(val_s1_ids)
        med_c = float(np.median(cand_counts))
        max_c = max(cand_counts) if cand_counts else 0
        pair_rec = (retained_gt_pairs / total_gt_pairs) * 100
        ent_rec = (len(retained_gt_entities) / len(gt_entities)) * 100
        lost_gt = total_gt_pairs - retained_gt_pairs

        # Compute Macro F_0.5
        macro_f05 = compute_macro_f05(gt_dict, pred_matches, val_s1_ids)

        row_res = {
            "strategy": strat_name,
            "candidates": total_cands,
            "avg_per_s1": avg_c,
            "median": med_c,
            "max": max_c,
            "zero_cands": zero_cands,
            "pair_recall": pair_rec,
            "entity_recall": ent_rec,
            "lost_gt": lost_gt,
            "macro_f05": macro_f05
        }
        results_table.append(row_res)

        print(
            f"{strat_name:<30} | {total_cands:>10,} | {avg_c:>6.1f} | {med_c:>3.0f} | {max_c:>4} | "
            f"{zero_cands:>10,} | {pair_rec:>9.2f}% | {ent_rec:>8.2f}% | {lost_gt:>7,} | {macro_f05:>10.4f}"
        )

    print("-" * 128)

    # Save benchmark report
    report_path = os.path.join(exp_dir, "candidate_limits_benchmark_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 128 + "\n")
        f.write("AMAZON ML CHALLENGE 2026 — CANDIDATE LIMITS BENCHMARK REPORT\n")
        f.write("=" * 128 + "\n")
        f.write(f"Validation Cohort: {len(val_s1_ids):,} Source 1 Entities\n")
        f.write(f"Total True Matches: {total_gt_pairs:,}\n")
        f.write(f"Decision Threshold: {optimal_th:.4f}\n\n")
        f.write(f"{'Strategy':<30} | {'Candidates':>10} | {'Avg/S1':>6} | {'Med':>3} | {'Max':>4} | {'Zero Cands':>10} | {'Pair Rec %':>10} | {'Ent Rec %':>9} | {'Lost GT':>7} | {'Macro F0.5':>10}\n")
        f.write("-" * 128 + "\n")
        for r in results_table:
            f.write(
                f"{r['strategy']:<30} | {r['candidates']:>10,} | {r['avg_per_s1']:>6.1f} | {r['median']:>3.0f} | {r['max']:>4} | "
                f"{r['zero_cands']:>10,} | {r['pair_recall']:>9.2f}% | {r['entity_recall']:>8.2f}% | {r['lost_gt']:>7,} | {r['macro_f05']:>10.4f}\n"
            )
        f.write("=" * 128 + "\n")

    print(f"\nReport written to: {report_path}")
    print(f"Benchmark finished in {time.time() - t_start:.2f}s.")

if __name__ == "__main__":
    main()
