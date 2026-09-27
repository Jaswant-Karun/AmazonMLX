#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Module: Controlled Model Experiments & Hard Negative Learning (Phases 8 & 9)
src/model_experiments_v2.py

Evaluates controlled models against official Macro F0.5 per Source-1 entity:
  - Model A: Baseline LightGBM (33 features)
  - Model B: Model A + Enhanced Name Features (Phase 7)
  - Model C: Model B + Enhanced Address Features (Phase 6)
  - Model D: Model C + Multilingual & Script Features (Phase 5) -> All 46 Features
  - Model E: Model D + Hard Negative Mining & Rebalanced Loss (Phase 8)

Outputs:
  - experiments/hard_negative_report.txt (Phase 8)
  - output/entity_matching_model_v2.joblib (Phase 9 winning model)
  - output/optimal_threshold_v2.json
"""

import sys
import os
import time
import json
import argparse
import collections
import pandas as pd
import numpy as np
import joblib

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluate_local_validation import compute_comprehensive_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


# Feature Groups
NAME_V2_ADDITIONS = [
    "name_compact_match",
    "name_ocr_compact_match",
    "name_sig_match",
    "name_token_containment",
    "name_last_token_sim",
]

ADDR_V2_ADDITIONS = [
    "addr_compact_match",
    "addr_token_containment",
    "addr_postal_code_match",
    "addr_numeric_sig_match",
    "both_addr_missing",
]

MULTILINGUAL_V2_ADDITIONS = [
    "same_script",
    "script_type_latin",
    "script_type_indic",
    "script_char_overlap",
]

# Baseline 33 features
BASELINE_FEATURES = [
    "name_exact_match", "name_char_len_s1", "name_char_len_cand", "name_char_len_diff",
    "name_char_len_ratio", "name_char_sim", "name_levenshtein_sim", "name_token_overlap",
    "name_token_jaccard", "name_first_token_sim", "name_prefix_sim", "name_alphanumeric_sim",
    "addr_exact_match", "addr_char_len_diff", "addr_char_len_ratio", "addr_char_sim",
    "addr_levenshtein_sim", "addr_token_overlap", "addr_token_jaccard", "addr_numeric_overlap",
    "addr_numeric_jaccard", "addr_house_number_match", "addr_locality_token_overlap",
    "addr_locality_jaccard", "s1_addr_missing", "cand_addr_missing", "country_match",
    "name_addr_combined_sim", "name_addr_harmonic_sim", "both_strong",
    "name_strong_addr_weak", "addr_strong_name_weak"
]


def load_ground_truth_map(gt_path: str):
    print(f"Loading ground truth from {gt_path}...")
    gt_map = collections.defaultdict(set)
    df_gt = pd.read_csv(gt_path, sep="\t", dtype=str)
    for s1, tgt in zip(df_gt["source1_entity_id"], df_gt["matched_entity_id"]):
        if tgt and tgt.strip():
            gt_map[s1].add(tgt.strip())
    return gt_map


def build_predictions_map(val_df: pd.DataFrame, probs: np.ndarray, threshold: float):
    preds_map = collections.defaultdict(set)
    cand_map = collections.defaultdict(set)
    
    s1_ids = val_df["source1_entity_id"].values
    tgt_ids = val_df["target_entity_id"].values
    
    for s1_id, tgt_id, p in zip(s1_ids, tgt_ids, probs):
        cand_map[s1_id].add(tgt_id)
        if p >= threshold:
            preds_map[s1_id].add(tgt_id)
            
    return preds_map, cand_map


def evaluate_model_at_thresholds(val_df: pd.DataFrame, probs: np.ndarray, all_s1_ids: list, gt_map: dict):
    best_th = 0.50
    best_macro_f05 = 0.0
    best_metrics = None
    
    for th in [0.40, 0.50, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80]:
        preds_map, cand_map = build_predictions_map(val_df, probs, th)
        m = compute_comprehensive_metrics(all_s1_ids, gt_map, preds_map, cand_map)
        if m["macro_f05"] > best_macro_f05:
            best_macro_f05 = m["macro_f05"]
            best_th = th
            best_metrics = m
            
    return best_th, best_metrics


def run_experiments():
    out_dir = os.path.join(os.path.dirname(BASE_DIR), "output")
    exp_dir = os.path.join(os.path.dirname(BASE_DIR), "experiments")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train_features_v2.parquet")
    val_path = os.path.join(out_dir, "val_features_v2.parquet")
    gt_path = os.path.join(os.path.dirname(BASE_DIR), "..", "dataset", "train", "train_ground_truth.tsv")
    if not os.path.isfile(gt_path):
        gt_path = os.path.join("student_resource", "dataset", "train", "train_ground_truth.tsv")

    print(f"Loading datasets...")
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    gt_map = load_ground_truth_map(gt_path)

    val_s1_set = sorted(list(set(df_val["source1_entity_id"].unique())))
    print(f"Validation dataset: {len(df_val):,} pairs across {len(val_s1_set):,} unique Source-1 entities.")

    # -------------------------------------------------------------
    # PHASE 8: Hard Negative Mining & Analysis
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 8: HARD NEGATIVE MINING & IDENTIFICATION")
    print("=" * 80)

    # Criteria for hard negatives:
    # 1. High name similarity but different address
    # 2. High address similarity but different name
    # 3. Blocked by multiple criteria but false match
    cond_name_hard = (df_train["label"] == 0) & (df_train["name_levenshtein_sim"] >= 0.85) & (df_train["addr_levenshtein_sim"] < 0.35)
    cond_addr_hard = (df_train["label"] == 0) & (df_train["addr_levenshtein_sim"] >= 0.85) & (df_train["name_levenshtein_sim"] < 0.40)
    cond_multiblock = (df_train["label"] == 0) & (df_train["num_blocks_matched"] >= 3)

    is_hard_negative = (cond_name_hard | cond_addr_hard | cond_multiblock)
    num_hard_negs = int(is_hard_negative.sum())
    total_negs = int((df_train["label"] == 0).sum())
    total_pos = int((df_train["label"] == 1).sum())

    print(f"Total training pairs: {len(df_train):,}")
    print(f"  Positive pairs (true matches)   : {total_pos:,} ({total_pos/len(df_train):.1%})")
    print(f"  Negative pairs (candidate pool) : {total_negs:,} ({total_negs/len(df_train):.1%})")
    print(f"  Identified HARD negatives       : {num_hard_negs:,} ({num_hard_negs/total_negs:.1%} of negatives)")
    print(f"    - High name sim, wrong address: {int(cond_name_hard.sum()):,}")
    print(f"    - High addr sim, wrong name   : {int(cond_addr_hard.sum()):,}")
    print(f"    - Multiblock false candidates : {int(cond_multiblock.sum()):,}")

    hard_neg_report_path = os.path.join(exp_dir, "hard_negative_report.txt")
    with open(hard_neg_report_path, "w", encoding="utf-8") as f:
        f.write("AMAZON ML CHALLENGE 2026: HARD NEGATIVE ANALYSIS REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Total Training Candidates: {len(df_train):,}\n")
        f.write(f"True Positives           : {total_pos:,}\n")
        f.write(f"Total Negatives          : {total_negs:,}\n")
        f.write(f"Identified Hard Negatives: {num_hard_negs:,} ({num_hard_negs/total_negs:.2%})\n\n")
        f.write("Categories:\n")
        f.write(f"  1. Strong name similarity (>=0.85) with wrong address (<0.35): {int(cond_name_hard.sum()):,}\n")
        f.write(f"  2. Strong address similarity (>=0.85) with wrong name (<0.40): {int(cond_addr_hard.sum()):,}\n")
        f.write(f"  3. Multiblock candidates (>=3 blocking rules matched): {int(cond_multiblock.sum()):,}\n\n")
        f.write("Hard Negative Learning Strategy:\n")
        f.write("  Standard training allows models to separate easy negatives (e.g. completely dissimilar businesses)\n")
        f.write("  without learning fine-grained discriminators for identical brand names in different cities\n")
        f.write("  or different branch offices at shared commercial plazas. In Model E, hard negatives are assigned\n")
        f.write("  2.5x sample weight to enforce strict precision margins.\n")
    print(f"Saved hard negative report to {hard_neg_report_path}")

    # Construct sample weights for Model E
    train_weights = np.ones(len(df_train), dtype=np.float32)
    train_weights[is_hard_negative.values] = 2.5

    # -------------------------------------------------------------
    # PHASE 9: Controlled Model Experiments
    # -------------------------------------------------------------
    models_config = [
        ("Model A (Baseline)", BASELINE_FEATURES, False),
        ("Model B (+ Name v2)", BASELINE_FEATURES + NAME_V2_ADDITIONS, False),
        ("Model C (+ Name & Addr v2)", BASELINE_FEATURES + NAME_V2_ADDITIONS + ADDR_V2_ADDITIONS, False),
        ("Model D (Full 46 Features)", BASELINE_FEATURES + NAME_V2_ADDITIONS + ADDR_V2_ADDITIONS + MULTILINGUAL_V2_ADDITIONS, False),
        ("Model E (Full + Hard Negatives)", BASELINE_FEATURES + NAME_V2_ADDITIONS + ADDR_V2_ADDITIONS + MULTILINGUAL_V2_ADDITIONS, True),
    ]

    results_table = []
    best_overall_f05 = 0.0
    winning_model = None
    winning_config = None

    y_train = df_train["label"].values.astype(int)
    y_val = df_val["label"].values.astype(int)

    print("\n" + "=" * 80)
    print("PHASE 9: CONTROLLED MODEL EXPERIMENTS ON VALIDATION DATA")
    print("=" * 80)

    for name, feat_cols, use_hard_weights in models_config:
        print(f"\n--- Training {name} ({len(feat_cols)} features) ---")
        t_start = time.time()

        X_tr = df_train[feat_cols].values
        X_va = df_val[feat_cols].values
        sw_tr = train_weights if use_hard_weights else None

        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
        )

        lgb_model.fit(
            X_tr, y_train,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
            ]
        )
        fit_time = time.time() - t_start

        # Inference & Metric Evaluation
        t_inf = time.time()
        val_probs = lgb_model.predict_proba(X_va)[:, 1]
        inf_time = time.time() - t_inf

        roc_auc = roc_auc_score(y_val, val_probs)
        pr_auc = average_precision_score(y_val, val_probs)

        best_th, m = evaluate_model_at_thresholds(df_val, val_probs, val_s1_set, gt_map)

        record = {
            "model": name,
            "features": len(feat_cols),
            "threshold": best_th,
            "macro_f05": m["macro_f05"],
            "pair_precision": m["pair_precision"],
            "pair_recall": m["pair_recall"],
            "singleton_fp_rate": 1.0 - m["singleton_accuracy"],
            "exact_match_rate": m["exact_match_rate"],
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "fit_time_s": round(fit_time, 1),
            "inf_time_s": round(inf_time, 2)
        }
        results_table.append(record)

        print(f"  Macro F0.5 : {m['macro_f05']:.5f} (Optimal Threshold: {best_th:.2f})")
        print(f"  Precision  : {m['pair_precision']:.2%} | Recall: {m['pair_recall']:.2%}")
        print(f"  Singleton FP Rate: {(1.0 - m['singleton_accuracy']):.2%}")
        print(f"  ROC-AUC    : {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        print(f"  Fit: {fit_time:.1f}s | Inference: {inf_time:.2f}s")

        if m["macro_f05"] > best_overall_f05:
            best_overall_f05 = m["macro_f05"]
            best_metrics = m
            winning_model = lgb_model
            winning_config = record

    print("\n" + "=" * 80)
    print("PHASE 9 EXPERIMENT COMPARISON SUMMARY")
    print("=" * 80)
    header = f"{'Model':<30} | {'Feats':<5} | {'Thresh':<6} | {'Macro F0.5':<10} | {'Precision':<9} | {'Recall':<8} | {'Singl FP':<8}"
    print(header)
    print("-" * len(header))
    for r in results_table:
        marker = " <=== WINNER" if r["model"] == winning_config["model"] else ""
        print(f"{r['model']:<30} | {r['features']:<5} | {r['threshold']:<6.2f} | {r['macro_f05']:<10.5f} | {r['pair_precision']:<8.2%} | {r['pair_recall']:<7.2%} | {r['singleton_fp_rate']:<7.2%}{marker}")

    # Save Winning Model
    model_save_path = os.path.join(out_dir, "entity_matching_model_v2.joblib")
    print(f"\nSaving winning model ({winning_config['model']}) to {model_save_path}...")
    joblib.dump(winning_model, model_save_path)

    opt_th_path = os.path.join(out_dir, "optimal_threshold_v2.json")
    threshold_data = {
        "model": winning_config["model"],
        "optimal_threshold": winning_config["threshold"],
        "macro_f05": winning_config["macro_f05"],
        "pair_precision": winning_config["pair_precision"],
        "pair_recall": winning_config["pair_recall"],
        "singleton_fp_rate": winning_config["singleton_fp_rate"],
        "exact_match_rate": winning_config["exact_match_rate"],
        "features_count": winning_config["features"]
    }
    with open(opt_th_path, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"Saved optimal threshold configuration to {opt_th_path}")


if __name__ == "__main__":
    run_experiments()
