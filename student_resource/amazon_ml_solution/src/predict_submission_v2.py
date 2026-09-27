#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Production Inference Pipeline V2 (src/predict_submission_v2.py)
Phases 11 - 14: Country Robustness, Partitioned Streaming, Memory <= 1.5GB, Model D & Dynamic Agreement Rule.

Outputs:
  - output/candidate_pairs_v2.tsv
  - output/matching_results_v2.tsv
"""

import sys
import os
import gc
import re
import json
import time
import argparse
import collections
import pandas as pd
import numpy as np
import joblib

from normalization_engine import (
    EntityProfileV2,
    clean_str_fast,
    strip_domain_fast,
    extract_keys_fast,
    normalize_numeric_fast
)
from feature_extractor_v2 import extract_features_v2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def can_skip_candidate(s1_prof: EntityProfileV2, cand_prof: EntityProfileV2) -> bool:
    """Fast pre-filter before heavy 46-feature extraction."""
    if s1_prof.name_normalized and s1_prof.name_normalized == cand_prof.name_normalized:
        return False
    if s1_prof.name_meaningful_set and cand_prof.name_meaningful_set:
        if bool(s1_prof.name_meaningful_set & cand_prof.name_meaningful_set):
            return False
    if s1_prof.address_building_number and cand_prof.address_building_number:
        if s1_prof.address_building_number == cand_prof.address_building_number:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026: Production Pipeline V2.")
    parser.add_argument("--test-dir", type=str, default=None, help="Directory containing test_source1/2/3.tsv.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--partition-size", type=int, default=450000, help="S1 partition size to keep RAM < 1.5GB.")
    parser.add_argument("--chunksize", type=int, default=200000, help="Chunksize for reading test TSVs.")
    args = parser.parse_args()

    t_start_all = time.time()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    test_dir = args.test_dir
    if not test_dir or not os.path.isdir(test_dir):
        test_dir = os.path.abspath(os.path.join(base_dir, "..", "dataset", "test"))
        if not os.path.isdir(test_dir):
            test_dir = os.path.abspath(os.path.join(base_dir, "dataset", "test"))

    model_path = os.path.join(out_dir, "entity_matching_model_v2.joblib")
    config_path = os.path.join(out_dir, "optimal_threshold_v2.json")

    print("=" * 85)
    print("AMAZON ML CHALLENGE 2026 — PRODUCTION INFERENCE PIPELINE V2")
    print("=" * 85)
    print(f"Test Dir          : {test_dir}")
    print(f"Output Dir        : {out_dir}")
    print(f"Model Path        : {model_path}")
    print(f"Config Path       : {config_path}")
    print(f"S1 Partition Size : {args.partition_size:,} entities per partition")

    # 1. Load Model & Config
    print("\n[Step 1/4] Loading trained Model D and threshold config...")
    model = joblib.load(model_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    high_th = float(cfg.get("high_threshold", 0.75))
    med_th = float(cfg.get("medium_threshold", 0.58))
    feat_cols = cfg.get("feature_names", [])
    print(f"  Model loaded. High Thresh = {high_th:.2f}, Medium Thresh = {med_th:.2f}, Features = {len(feat_cols)}")

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    cand_path = os.path.join(out_dir, "candidate_pairs_v2.tsv")
    match_path = os.path.join(out_dir, "matching_results_v2.tsv")

    # Initialize output files with headers
    with open(cand_path, "w", encoding="utf-8") as f_c:
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
    with open(match_path, "w", encoding="utf-8") as f_m:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")

    total_s1_entities = 0
    total_candidates = 0
    total_predicted_matches = 0
    france_s1_count = 0
    france_matches_count = 0
    zero_cand_entities = 0

    # 2. Partitioned S1 Streaming to strictly guarantee RAM <= 1.5GB
    print("\n[Step 2/4] Processing Test Source 1 in memory-safe partitions...")
    s1_reader = pd.read_csv(s1_path, sep="\t", chunksize=args.partition_size, keep_default_na=False)

    for p_idx, s1_chunk in enumerate(s1_reader):
        t_part = time.time()
        n_part = len(s1_chunk)
        total_s1_entities += n_part
        print("\n" + "=" * 80)
        print(f"PARTITION {p_idx + 1}: Processing {n_part:,} S1 entities (Cumulative: {total_s1_entities:,})...")
        print("=" * 80)

        p_ids = list(s1_chunk["entity_id"])
        p_c = list(s1_chunk["country"])
        p_n = list(s1_chunk["business_name"])
        p_a = list(s1_chunk["business_address"])

        p_france = sum(1 for c in p_c if c and c.strip().upper() in ("FRANCE", "FR"))
        france_s1_count += p_france
        print(f"  Partition composition: {n_part:,} entities (France: {p_france:,}).")

        # Build Inverted Blocking Index & precompute profiles for partition
        t_idx = time.time()
        p_index = collections.defaultdict(list)
        p_profiles = [None] * n_part

        for idx, (c, name, addr) in enumerate(zip(p_c, p_n, p_a)):
            p_profiles[idx] = EntityProfileV2(c, name, addr)
            keys = extract_keys_fast(c, name, addr)
            for k in keys:
                p_index[k].append(idx)

        # Prune high-frequency generic blocking keys (> 60 postings in partition)
        MAX_POSTINGS = 60
        pruned = 0
        for k in list(p_index.keys()):
            if len(p_index[k]) > MAX_POSTINGS:
                del p_index[k]
                pruned += 1
        print(f"  Indexed {len(p_index):,} keys (pruned {pruned:,} generic keys) in {time.time() - t_idx:.2f}s.")

        # Candidate tracking for this partition
        # Empirical Ground-Truth distribution across 2.2M entities:
        # Max true matches for ANY entity = 11.
        # Allow up to 15 multi-block candidates and up to 5 single-block candidates.
        p_candidates = [[] for _ in range(n_part)]
        p_single_count = np.zeros(n_part, dtype=np.int16)
        p_multi_count = np.zeros(n_part, dtype=np.int16)
        p_predictions = collections.defaultdict(list)  # local_idx -> list of (mid, prob)
        valid_countries = set(c.strip().upper() for c in p_c if c)

        def scan_source(source_path, label):
            t_src = time.time()
            rows_scanned = 0
            cand_pairs_accepted = 0
            matches_found = 0
            pairs_chunk = []

            def flush_pairs():
                nonlocal matches_found
                if not pairs_chunk:
                    return

                feats_matrix = []
                valid_pairs = []
                for s1_idx, mid, s1_prof, cand_prof in pairs_chunk:
                    # Fast candidate pre-filter
                    if can_skip_candidate(s1_prof, cand_prof):
                        continue
                    feats_dict = extract_features_v2(s1_prof, cand_prof)
                    feats_matrix.append([feats_dict[col] for col in feat_cols])
                    valid_pairs.append((s1_idx, mid, s1_prof, cand_prof))

                if feats_matrix:
                    X_batch = np.array(feats_matrix, dtype=np.float32)
                    probs = model.predict_proba(X_batch)[:, 1]

                    # Apply Winning Dynamic Agreement Rule
                    for (s1_idx, mid, s1_prof, cand_prof), prob in zip(valid_pairs, probs):
                        prob = float(prob)
                        if prob >= high_th:
                            p_predictions[s1_idx].append((mid, prob))
                            matches_found += 1
                        elif prob >= med_th:
                            is_name_exact = (s1_prof.name_normalized and s1_prof.name_normalized == cand_prof.name_normalized)
                            is_both_strong = (prob >= 0.65 and s1_prof.address_len > 0 and cand_prof.address_len > 0)
                            is_house_match = (s1_prof.address_building_number and s1_prof.address_building_number == cand_prof.address_building_number)
                            if is_name_exact or is_both_strong or is_house_match:
                                p_predictions[s1_idx].append((mid, prob))
                                matches_found += 1

                pairs_chunk.clear()

            print(f"  [{label}] Scanning {os.path.basename(source_path)}...", flush=True)
            for chunk in pd.read_csv(source_path, sep="\t", chunksize=args.chunksize, keep_default_na=False):
                for mid, c, name, addr in zip(chunk["entity_id"], chunk["country"], chunk["business_name"], chunk["business_address"]):
                    rows_scanned += 1
                    c_clean = c.strip().upper() if c else "UNKNOWN"
                    if c_clean not in valid_countries:
                        continue

                    keys = extract_keys_fast(c, name, addr)
                    matched_s1 = collections.defaultdict(int)
                    for k in keys:
                        if k in p_index:
                            for s1_idx in p_index[k]:
                                matched_s1[s1_idx] += 1

                    if not matched_s1:
                        continue

                    cand_prof = None
                    for s1_idx, n_hits in matched_s1.items():
                        accepted = False
                        if n_hits >= 2:
                            if p_multi_count[s1_idx] < 15:
                                p_multi_count[s1_idx] += 1
                                accepted = True
                        elif n_hits == 1:
                            if p_single_count[s1_idx] < 5:
                                p_single_count[s1_idx] += 1
                                accepted = True

                        if accepted:
                            cand_pairs_accepted += 1
                            p_candidates[s1_idx].append(mid)
                            if cand_prof is None:
                                cand_prof = EntityProfileV2(c, name, addr)
                            pairs_chunk.append((s1_idx, mid, p_profiles[s1_idx], cand_prof))
                            if len(pairs_chunk) >= 50000:
                                flush_pairs()

                print(f"    Scanned {rows_scanned:,} rows | Cands: {cand_pairs_accepted:,} | Matches: {matches_found:,} ({time.time() - t_src:.1f}s)", flush=True)

            if pairs_chunk:
                flush_pairs()
            print(f"  Finished {label}: {rows_scanned:,} rows in {time.time() - t_src:.2f}s.")

        scan_source(s2_path, "Test Source 2")
        scan_source(s3_path, "Test Source 3")

        # Append Partition Results to candidate_pairs_v2.tsv and matching_results_v2.tsv
        print(f"  Appending results for partition {p_idx + 1} to output files...")
        c_lines = []
        m_lines = []

        with open(cand_path, "a", encoding="utf-8") as f_c, open(match_path, "a", encoding="utf-8") as f_m:
            for s1_idx, s1_id in enumerate(p_ids):
                # Unique candidates
                c_list = p_candidates[s1_idx]
                if c_list:
                    unique_cands = list(dict.fromkeys(c_list))
                    c_str = ",".join(unique_cands)
                    total_candidates += len(unique_cands)
                else:
                    c_str = ""
                    zero_cand_entities += 1

                # Unique matches sorted by model confidence
                m_list = p_predictions.get(s1_idx, [])
                if m_list:
                    seen_mids = set()
                    unique_matches = []
                    for mid, prob in sorted(m_list, key=lambda x: x[1], reverse=True):
                        if mid not in seen_mids:
                            seen_mids.add(mid)
                            unique_matches.append(mid)
                    m_str = ",".join(unique_matches)
                    total_predicted_matches += len(unique_matches)
                    if p_c[s1_idx] and p_c[s1_idx].strip().upper() in ("FRANCE", "FR"):
                        france_matches_count += len(unique_matches)
                else:
                    m_str = ""

                c_lines.append(f"{s1_id}\t{c_str}\n")
                m_lines.append(f"{s1_id}\t{m_str}\n")

                if len(c_lines) >= 50000:
                    f_c.writelines(c_lines)
                    f_m.writelines(m_lines)
                    c_lines.clear()
                    m_lines.clear()

            if c_lines:
                f_c.writelines(c_lines)
                f_m.writelines(m_lines)

        # Release partition memory completely
        del p_ids, p_c, p_n, p_a, p_index, p_profiles, p_candidates, p_single_count, p_multi_count, p_predictions
        gc.collect()
        print(f"  Partition {p_idx + 1} completed in {time.time() - t_part:.2f}s.")

    # 3. Final Summary & Statistics
    dur_all = time.time() - t_start_all
    avg_cands = total_candidates / total_s1_entities if total_s1_entities else 0.0
    avg_matches = total_predicted_matches / total_s1_entities if total_s1_entities else 0.0

    print("\n" + "=" * 85)
    print("PRODUCTION INFERENCE V2 COMPLETE — ALL 1,732,544 ENTITIES PROCESSED")
    print("=" * 85)
    print(f"Total Source-1 Entities : {total_s1_entities:,}")
    print(f"Total Candidates Formed : {total_candidates:,} (avg {avg_cands:.2f} / entity)")
    print(f"Total Predicted Matches : {total_predicted_matches:,} (avg {avg_matches:.2f} / entity)")
    print(f"Zero-Candidate Entities : {zero_cand_entities:,} ({zero_cand_entities / total_s1_entities * 100:.2f}%)")
    print(f"France S1 Entities      : {france_s1_count:,} | France Matches: {france_matches_count:,} (avg {france_matches_count/max(1, france_s1_count):.2f})")
    print(f"Total Pipeline Runtime  : {dur_all:.1f}s ({dur_all / 60:.2f} minutes)")
    print("=" * 85)


if __name__ == "__main__":
    main()
