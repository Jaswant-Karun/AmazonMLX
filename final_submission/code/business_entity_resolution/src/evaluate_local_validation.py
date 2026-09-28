#!/usr/bin/env python3
"""
Comprehensive Local Validation Framework for Amazon ML Challenge 2026.
Evaluates official competition-style Macro F0.5 per Source-1 entity,
singleton accuracy, exact-match rate, pair-level precision/recall, and candidate statistics.
"""

import sys
import os
import time
import json
import collections
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def compute_comprehensive_metrics(all_s1_ids, gt_map, pred_map, cand_map=None):
    """
    Computes all competition-style validation metrics:
      - Per-entity Precision, Recall, F0.5
      - Macro F0.5 (mean across all entities)
      - Singleton accuracy & singleton false positives
      - Exact-match entity rate
      - Pair precision & pair recall
      - Candidate pair recall & candidate entity recall
      - Candidate count statistics (avg, median, p95, p99, max, zero-cand)
      - Per-entity FP and FN averages
    """
    entity_f05_list = []
    entity_prec_list = []
    entity_rec_list = []
    entity_fp_list = []
    entity_fn_list = []

    exact_match_count = 0
    singleton_total = 0
    singleton_correct = 0
    singleton_fp_matches = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt_pairs = sum(len(v) for v in gt_map.values())

    cand_gt_recovered = 0
    cand_entity_recovered = 0
    non_singleton_total = 0

    cand_lengths = []

    for s1_id in all_s1_ids:
        gt_set = gt_map.get(s1_id, set())
        pred_set = pred_map.get(s1_id, set())
        cand_set = cand_map.get(s1_id, set()) if cand_map is not None else set()

        cand_lengths.append(len(cand_set))

        len_gt = len(gt_set)
        len_pred = len(pred_set)

        if len_gt == 0:
            singleton_total += 1
            if len_pred == 0:
                singleton_correct += 1
                exact_match_count += 1
                entity_f05_list.append(1.0)
                entity_prec_list.append(1.0)
                entity_rec_list.append(1.0)
                entity_fp_list.append(0)
                entity_fn_list.append(0)
            else:
                singleton_fp_matches += len_pred
                total_fp += len_pred
                entity_f05_list.append(0.0)
                entity_prec_list.append(0.0)
                entity_rec_list.append(0.0)
                entity_fp_list.append(len_pred)
                entity_fn_list.append(0)
        else:
            non_singleton_total += 1
            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            entity_fp_list.append(fp)
            entity_fn_list.append(fn)

            if cand_map is not None:
                cand_tp = len(gt_set & cand_set)
                cand_gt_recovered += cand_tp
                if cand_tp > 0:
                    cand_entity_recovered += 1

            if len_pred == 0:
                entity_f05_list.append(0.0)
                entity_prec_list.append(0.0)
                entity_rec_list.append(0.0)
            else:
                prec = tp / len_pred
                rec = tp / len_gt
                entity_prec_list.append(prec)
                entity_rec_list.append(rec)
                if (0.25 * prec + rec) > 0:
                    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                else:
                    f05 = 0.0
                entity_f05_list.append(f05)

                if gt_set == pred_set:
                    exact_match_count += 1

    macro_f05 = float(np.mean(entity_f05_list)) if entity_f05_list else 0.0
    macro_prec = float(np.mean(entity_prec_list)) if entity_prec_list else 0.0
    macro_rec = float(np.mean(entity_rec_list)) if entity_rec_list else 0.0
    exact_match_rate = exact_match_count / len(all_s1_ids) if all_s1_ids else 0.0

    singleton_acc = singleton_correct / singleton_total if singleton_total else 1.0
    pair_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    pair_rec = total_tp / total_gt_pairs if total_gt_pairs else 0.0

    cand_pair_rec = cand_gt_recovered / total_gt_pairs if (cand_map and total_gt_pairs) else 0.0
    cand_entity_rec = cand_entity_recovered / non_singleton_total if (cand_map and non_singleton_total) else 0.0

    c_arr = np.array(cand_lengths) if cand_lengths else np.array([0])

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "exact_match_rate": exact_match_rate,
        "exact_match_count": exact_match_count,
        "singleton_total": singleton_total,
        "singleton_accuracy": singleton_acc,
        "singleton_fp_matches": singleton_fp_matches,
        "pair_precision": pair_prec,
        "pair_recall": pair_rec,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "candidate_pair_recall": cand_pair_rec,
        "candidate_entity_recall": cand_entity_rec,
        "avg_candidates": float(np.mean(c_arr)),
        "median_candidates": float(np.median(c_arr)),
        "p95_candidates": float(np.percentile(c_arr, 95)),
        "p99_candidates": float(np.percentile(c_arr, 99)),
        "max_candidates": int(np.max(c_arr)),
        "zero_candidate_entities": int((c_arr == 0).sum()),
        "avg_fp_per_entity": float(np.mean(entity_fp_list)),
        "avg_fn_per_entity": float(np.mean(entity_fn_list))
    }


def print_validation_report(metrics: dict, title: str = "VALIDATION REPORT"):
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    print(f"OFFICIAL MACRO F0.5 : {metrics['macro_f05']:.4f}")
    print(f"Macro Precision     : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall        : {metrics['macro_recall']:.4f}")
    print(f"Exact Match Entity  : {metrics['exact_match_rate']:.2%} ({metrics['exact_match_count']:,} entities)")
    print(f"Singleton Accuracy  : {metrics['singleton_accuracy']:.2%} ({metrics['singleton_total']:,} total singletons)")
    print(f"Singleton FP Matches: {metrics['singleton_fp_matches']:,}")
    print(f"Pair Precision      : {metrics['pair_precision']:.4f}")
    print(f"Pair Recall         : {metrics['pair_recall']:.4f}")
    print(f"Pair Confusion      : TP={metrics['total_tp']:,} | FP={metrics['total_fp']:,} | FN={metrics['total_fn']:,}")
    print(f"Per-Entity Errors   : Avg FP={metrics['avg_fp_per_entity']:.3f} | Avg FN={metrics['avg_fn_per_entity']:.3f}")
    print("-" * 80)
    print("CANDIDATE BLOCKING METRICS:")
    print(f"Candidate Pair Rec  : {metrics['candidate_pair_recall']:.2%}")
    print(f"Candidate Entity Rec: {metrics['candidate_entity_recall']:.2%}")
    print(f"Avg Candidates/S1   : {metrics['avg_candidates']:.2f}")
    print(f"Median Candidates   : {metrics['median_candidates']:.1f}")
    print(f"P95 Candidates      : {metrics['p95_candidates']:.1f}")
    print(f"P99 Candidates      : {metrics['p99_candidates']:.1f}")
    print(f"Max Candidates      : {metrics['max_candidates']}")
    print(f"Zero-Cand Entities  : {metrics['zero_candidate_entities']:,}")
    print("=" * 80)
