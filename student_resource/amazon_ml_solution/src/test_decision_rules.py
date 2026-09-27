#!/usr/bin/env python3
"""
Test Dynamic Multi-Match Decision Rules on Validation Features (Phase 10)
"""
import sys
import os
import joblib
import collections
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_local_validation import compute_comprehensive_metrics
from model_experiments_v2 import load_ground_truth_map

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
val_features_path = os.path.join(out_dir, "val_features_v2.parquet")
model_path = os.path.join(out_dir, "entity_matching_model_v2.joblib")
gt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "dataset", "train", "train_ground_truth.tsv")

print("Loading validation features and model...")
df_val = pd.read_parquet(val_features_path)
model = joblib.load(model_path)
gt_map = load_ground_truth_map(gt_path)
all_s1_ids = sorted(list(set(df_val["source1_entity_id"].unique())))

meta_cols = {"source1_entity_id", "target_entity_id", "target_source", "matched_by_blocks", "num_blocks_matched", "label"}
feat_cols = [c for c in df_val.columns if c not in meta_cols]

print(f"Predicting probabilities across {len(df_val):,} pairs...")
probs = model.predict_proba(df_val[feat_cols].values)[:, 1]
df_val["prob"] = probs

# Candidate map
c_map = collections.defaultdict(set)
for s1, tgt in zip(df_val["source1_entity_id"], df_val["target_entity_id"]):
    c_map[s1].add(tgt)

# Rule 1: Flat threshold 0.60
p1_map = collections.defaultdict(set)
for s1, tgt, p in zip(df_val["source1_entity_id"], df_val["target_entity_id"], df_val["prob"]):
    if p >= 0.60:
        p1_map[s1].add(tgt)
m1 = compute_comprehensive_metrics(all_s1_ids, gt_map, p1_map, c_map)

# Rule 2: Flat threshold 0.65
p2_map = collections.defaultdict(set)
for s1, tgt, p in zip(df_val["source1_entity_id"], df_val["target_entity_id"], df_val["prob"]):
    if p >= 0.65:
        p2_map[s1].add(tgt)
m2 = compute_comprehensive_metrics(all_s1_ids, gt_map, p2_map, c_map)

# Rule 3: Dynamic Margin (p >= 0.60 and if p_max >= 0.85 then p >= max(0.60, p_max - 0.25))
p3_map = collections.defaultdict(set)
grouped = df_val.groupby("source1_entity_id")
for s1, group in grouped:
    p_max = group["prob"].max()
    th = max(0.60, p_max - 0.25) if p_max >= 0.85 else 0.60
    for tgt, p in zip(group["target_entity_id"], group["prob"]):
        if p >= th:
            p3_map[s1].add(tgt)
m3 = compute_comprehensive_metrics(all_s1_ids, gt_map, p3_map, c_map)

# Rule 4: Dynamic Agreement (p >= 0.75 or (p >= 0.58 and (name_exact == 1 or both_strong == 1 or addr_house_number_match == 1)))
p4_map = collections.defaultdict(set)
for s1, tgt, p, ne, bs, hn in zip(df_val["source1_entity_id"], df_val["target_entity_id"], df_val["prob"],
                                  df_val["name_exact_match"], df_val["both_strong"], df_val["addr_house_number_match"]):
    if p >= 0.75 or (p >= 0.58 and (ne == 1.0 or bs == 1.0 or hn == 1.0)):
        p4_map[s1].add(tgt)
m4 = compute_comprehensive_metrics(all_s1_ids, gt_map, p4_map, c_map)

print("\n" + "=" * 80)
print("PHASE 10: MULTI-MATCH DECISION RULE BENCHMARK ON VALIDATION DATA")
print("=" * 80)
print(f"Rule 1 (Flat 0.60)           : Macro F0.5 = {m1['macro_f05']:.5f} | Prec = {m1['pair_precision']:.2%} | Singl FP = {(1.0-m1['singleton_accuracy']):.2%}")
print(f"Rule 2 (Flat 0.65)           : Macro F0.5 = {m2['macro_f05']:.5f} | Prec = {m2['pair_precision']:.2%} | Singl FP = {(1.0-m2['singleton_accuracy']):.2%}")
print(f"Rule 3 (Dynamic Margin)      : Macro F0.5 = {m3['macro_f05']:.5f} | Prec = {m3['pair_precision']:.2%} | Singl FP = {(1.0-m3['singleton_accuracy']):.2%}")
print(f"Rule 4 (Dynamic Agreement)   : Macro F0.5 = {m4['macro_f05']:.5f} | Prec = {m4['pair_precision']:.2%} | Singl FP = {(1.0-m4['singleton_accuracy']):.2%}")
