#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Step 6: Supervised ML Model Training & Threshold Optimization (06_train_model.py)

Author: Amazon ML Challenge Team
Date: September 2026

Description:
  Trains a gradient-boosted decision tree (LightGBM) on the pairwise engineered
  features extracted from train_features.parquet. Evaluates out-of-fold performance
  on val_features.parquet, optimizes the decision threshold to maximize F1 score,
  and exports the trained model and inference threshold.

Outputs:
  - output/entity_matching_model.joblib
  - output/optimal_threshold.json
  - experiments/model_training_report.txt
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

from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def load_dataset(parquet_path: str):
    print(f"Loading features from {parquet_path}...")
    t0 = time.time()
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} rows with {df.shape[1]} columns in {time.time() - t0:.2f}s.")
    return df


def optimize_threshold_for_f1(y_true, y_probs):
    """
    Performs a fine-grained sweep across decision thresholds
    to find the global maximum F1 score on validation set.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * (precisions * recalls) / (precisions + recalls),
        0.0
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.50
    best_f1 = float(f1_scores[best_idx])
    best_prec = float(precisions[best_idx])
    best_rec = float(recalls[best_idx])

    return best_threshold, best_f1, best_prec, best_rec


def evaluate_at_thresholds(y_true, y_probs, target_thresholds=(0.30, 0.40, 0.50, 0.60, 0.70)):
    records = []
    for th in target_thresholds:
        preds = (y_probs >= th).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        records.append({
            "threshold": th,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Step 6: Train ML Entity Resolution Model.")
    parser.add_argument("--train-features", type=str, default=None, help="Path to train_features.parquet.")
    parser.add_argument("--val-features", type=str, default=None, help="Path to val_features.parquet.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for model artifacts.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--n-estimators", type=int, default=600, help="Max trees.")
    parser.add_argument("--num-leaves", type=int, default=63, help="Max leaves.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    t_start = time.time()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(base_dir, "output")
    exp_dir = os.path.join(base_dir, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    train_path = args.train_features or os.path.join(out_dir, "train_features.parquet")
    val_path = args.val_features or os.path.join(out_dir, "val_features.parquet")

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 — STEP 6: ML MODEL TRAINING (LIGHTGBM GBDT)")
    print("=" * 80)
    print(f"Train features: {train_path}")
    print(f"Val features  : {val_path}")
    print(f"Output dir    : {out_dir}")

    # Step 1: Load Datasets
    df_train = load_dataset(train_path)
    df_val = load_dataset(val_path)

    meta_cols = {"source1_entity_id", "target_entity_id", "target_source", "label"}
    feature_cols = [c for c in df_train.columns if c not in meta_cols]

    print(f"\nFeature count: {len(feature_cols)} features used for model training:")
    for i, fc in enumerate(feature_cols, 1):
        print(f"  {i:>2}. {fc}")

    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values.astype(int)

    X_val = df_val[feature_cols].values
    y_val = df_val["label"].values.astype(int)

    train_pos = int((y_train == 1).sum())
    train_neg = int((y_train == 0).sum())
    val_pos = int((y_val == 1).sum())
    val_neg = int((y_val == 0).sum())

    print(f"\nTraining set balance  : {train_pos:,} pos / {train_neg:,} neg (pos ratio: {train_pos/len(y_train):.2%})")
    print(f"Validation set balance: {val_pos:,} pos / {val_neg:,} neg (pos ratio: {val_pos/len(y_val):.2%})")

    # Step 2: Train Model
    print("\nTraining LightGBM Classifier...")
    t_train_start = time.time()

    if HAS_LIGHTGBM:
        model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=args.seed,
            n_jobs=-1,
            importance_type="gain",
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=True),
                lgb.log_evaluation(period=50),
            ]
        )
    else:
        print("Using scikit-learn HistGradientBoostingClassifier...")
        model = HistGradientBoostingClassifier(
            learning_rate=args.learning_rate,
            max_iter=args.n_estimators,
            max_leaf_nodes=args.num_leaves,
            max_depth=7,
            early_stopping=True,
            random_state=args.seed,
        )
        model.fit(X_train, y_train)

    train_duration = time.time() - t_train_start
    print(f"Model training finished in {train_duration:.2f}s.")

    # Step 3: Evaluate on Validation Set
    print("\nEvaluating model on out-of-sample validation pairs...")
    val_probs = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, val_probs)
    pr_auc = average_precision_score(y_val, val_probs)

    print(f"  Validation ROC-AUC : {roc_auc:.4f}")
    print(f"  Validation PR-AUC  : {pr_auc:.4f}")

    # Optimize threshold
    best_th, best_f1, best_p, best_r = optimize_threshold_for_f1(y_val, val_probs)
    print(f"\nOptimal Decision Threshold Search:")
    print(f"  Best Threshold : {best_th:.4f}")
    print(f"  Peak F1 Score  : {best_f1:.4f} (Precision: {best_p:.4f}, Recall: {best_r:.4f})")

    # Fixed threshold evaluations
    sweep_results = evaluate_at_thresholds(y_val, val_probs, [0.30, 0.40, 0.50, 0.60, best_th, 0.70])

    print("\nThreshold Sensitivity Analysis:")
    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1 Score':>10}")
    print("-" * 46)
    for sr in sweep_results:
        marker = " <--- (OPTIMAL)" if abs(sr["threshold"] - best_th) < 1e-4 else ""
        print(f"{sr['threshold']:>10.4f} | {sr['precision']:>10.4f} | {sr['recall']:>10.4f} | {sr['f1']:>10.4f}{marker}")

    # Feature Importance
    feat_importances = []
    if HAS_LIGHTGBM and hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
        imp_sum = sum(raw_imp) if sum(raw_imp) > 0 else 1.0
        for name, imp in sorted(zip(feature_cols, raw_imp), key=lambda x: x[1], reverse=True):
            feat_importances.append({"feature": name, "importance_gain": float(imp), "relative_pct": float(imp / imp_sum * 100)})

        print("\nTop 15 Most Discriminative Engineered Features:")
        print(f"{'Rank':>4} | {'Feature Name':<32} | {'Relative Gain %':>15}")
        print("-" * 55)
        for rk, item in enumerate(feat_importances[:15], 1):
            print(f"{rk:>4} | {item['feature']:<32} | {item['relative_pct']:>14.2f}%")

    # Step 4: Save Model & Optimal Threshold Artifacts
    model_save_path = os.path.join(out_dir, "entity_matching_model.joblib")
    print(f"\nSaving trained model to {model_save_path}...")
    joblib.dump(model, model_save_path)

    threshold_config = {
        "optimal_threshold": round(best_th, 4),
        "validation_f1": round(best_f1, 4),
        "validation_precision": round(best_p, 4),
        "validation_recall": round(best_r, 4),
        "validation_roc_auc": round(roc_auc, 4),
        "validation_pr_auc": round(pr_auc, 4),
        "feature_names": feature_cols,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    threshold_save_path = os.path.join(out_dir, "optimal_threshold.json")
    with open(threshold_save_path, "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=2)
    print(f"Saved threshold configuration to {threshold_save_path}")

    # Save detailed report
    report_save_path = os.path.join(exp_dir, "model_training_report.txt")
    report_lines = [
        "=" * 85,
        "AMAZON ML CHALLENGE 2026 — STEP 6 MODEL TRAINING & EVALUATION REPORT",
        "=" * 85,
        f"Generated At           : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model Architecture     : LightGBM GBDT (binary classification)",
        f"Training Sample Size   : {len(y_train):,} pairs ({train_pos:,} pos / {train_neg:,} neg)",
        f"Validation Sample Size : {len(y_val):,} pairs ({val_pos:,} pos / {val_neg:,} neg)",
        f"Training Duration      : {train_duration:.2f}s",
        f"Validation ROC-AUC     : {roc_auc:.4f}",
        f"Validation PR-AUC      : {pr_auc:.4f}",
        f"Optimal Decision Thresh: {best_th:.4f}",
        f"Validation F1 @ Thresh : {best_f1:.4f} (Precision: {best_p:.4f}, Recall: {best_r:.4f})",
        "=" * 85,
        "\nTHRESHOLD SENSITIVITY TABLE:",
        "-" * 55,
        f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1 Score':>10}",
        "-" * 55,
    ]
    for sr in sweep_results:
        report_lines.append(f"{sr['threshold']:>10.4f} | {sr['precision']:>10.4f} | {sr['recall']:>10.4f} | {sr['f1']:>10.4f}")

    if feat_importances:
        report_lines.extend([
            "\n" + "-" * 55,
            "FEATURE IMPORTANCES (Normalized Gain):",
            "-" * 55,
            f"{'Rank':>4} | {'Feature Name':<32} | {'Relative Gain %':>15}",
            "-" * 55,
        ])
        for rk, item in enumerate(feat_importances, 1):
            report_lines.append(f"{rk:>4} | {item['feature']:<32} | {item['relative_pct']:>14.2f}%")

    report_lines.extend([
        "=" * 85,
        f"Total Pipeline Execution Time: {time.time() - t_start:.2f}s",
        "=" * 85,
    ])

    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Model report written to: {report_save_path}")

    print("\n" + "=" * 80)
    print("STEP 6 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Ready for Step 7: Test Prediction & Step 8: Submission Generation.")


if __name__ == "__main__":
    main()
