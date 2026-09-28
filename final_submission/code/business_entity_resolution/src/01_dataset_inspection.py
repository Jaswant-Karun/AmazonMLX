#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution Challenge
Dataset Inspection Script (01_dataset_inspection.py)

This script inspects all training and test TSV files without loading
the entire dataset into memory at once, using chunked reads with pandas.

Reports for each dataset:
  1. Number of rows
  2. Number of columns
  3. Column names
  4. Data types
  5. Missing values per column
  6. Duplicate entity_id count
  7. Number of unique countries
  8. Country frequency
  9. Five sample records

Additional report for train_ground_truth.tsv:
  1. Number of rows
  2. Number of singleton Source 1 entities (empty matched_entity_ids)
  3. Number of Source 1 entities having 1 match
  4. Number having multiple matches
  5. Number of S2 matches
  6. Number of S3 matches
  7. Number of total matched IDs
  8. Five example ground-truth rows
"""

import sys
import os
import gc
import collections
import argparse
import pandas as pd

# Ensure terminal handles UTF-8 characters on all platforms (including Windows cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def locate_dataset_dir(explicit_path=None):
    """Locate the dataset directory containing 'train' and 'test' subdirectories."""
    if explicit_path and os.path.isdir(explicit_path):
        return os.path.abspath(explicit_path)

    # Candidate search paths
    candidates = [
        "dataset",
        os.path.join("student_resource", "dataset"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "student_resource", "dataset"),
    ]

    for cand in candidates:
        abs_cand = os.path.abspath(cand)
        if os.path.isdir(abs_cand) and os.path.isdir(os.path.join(abs_cand, "train")):
            return abs_cand

    raise FileNotFoundError(
        "Could not find dataset directory with 'train' and 'test' subdirectories. "
        "Please provide --dataset-dir <path>."
    )


def format_size(bytes_size):
    """Format bytes into human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def inspect_source_tsv(filepath, chunksize=100000):
    """
    Inspect a Source TSV file (entity_id, business_name, business_address, country)
    using chunked reading to conserve memory.
    """
    filename = os.path.basename(filepath)
    file_size = os.path.getsize(filepath)

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print(f"Path: {filepath} ({format_size(file_size)})")
    print("=" * 80)

    total_rows = 0
    column_names = None
    column_dtypes = {}
    missing_counts = collections.defaultdict(int)
    country_counts = collections.Counter()
    seen_ids = set()
    dup_id_count = 0
    first_chunk_df = None

    for chunk_idx, chunk in enumerate(pd.read_csv(filepath, sep="\t", chunksize=chunksize, keep_default_na=False)):
        if chunk_idx == 0:
            first_chunk_df = chunk.head(5).copy()
            column_names = list(chunk.columns)
            for col in chunk.columns:
                column_dtypes[col] = str(chunk[col].dtype)

        total_rows += len(chunk)

        # Missing values (empty string or NaN)
        for col in chunk.columns:
            empty_mask = (chunk[col] == "") | chunk[col].isna()
            missing_counts[col] += int(empty_mask.sum())

        # Duplicate entity_id
        if "entity_id" in chunk.columns:
            for eid in chunk["entity_id"]:
                if eid in seen_ids:
                    dup_id_count += 1
                else:
                    seen_ids.add(eid)

        # Country frequency
        if "country" in chunk.columns:
            country_series = chunk["country"].astype(str).str.strip()
            country_counts.update(country_series.value_counts().to_dict())

    # Release ID set memory immediately
    del seen_ids
    gc.collect()

    num_cols = len(column_names) if column_names else 0

    print(f"\n1. Number of rows   : {total_rows:,}")
    print(f"2. Number of columns: {num_cols}")
    print(f"3. Column names     : {column_names}")

    print("\n4. Data types & 5. Missing values per column:")
    header = f"   {'Column Name':<20} | {'Data Type':<12} | {'Missing Count':<15} | {'Missing %':<10}"
    print(header)
    print("   " + "-" * (len(header) - 3))
    for col in column_names:
        miss = missing_counts[col]
        pct = (miss / total_rows * 100.0) if total_rows > 0 else 0.0
        dtype_str = column_dtypes.get(col, "unknown")
        print(f"   {col:<20} | {dtype_str:<12} | {miss:>13,} | {pct:>8.2f}%")

    print(f"\n6. Duplicate entity_id count: {dup_id_count:,}")

    unique_countries = len([c for c in country_counts if c != ""])
    print(f"\n7. Number of unique countries: {unique_countries}")
    print("8. Country frequency:")
    for country, count in country_counts.most_common():
        label = repr(country) if country == "" else country
        pct = (count / total_rows * 100.0) if total_rows > 0 else 0.0
        print(f"   - {label:<15}: {count:>12,} ({pct:>6.2f}%)")

    print("\n9. Five sample records:")
    if first_chunk_df is not None:
        for idx, row in first_chunk_df.iterrows():
            print(f"   Record #{idx + 1}:")
            for col in column_names:
                val = row[col]
                # truncate display if extremely long
                display_val = val if len(str(val)) < 120 else str(val)[:117] + "..."
                print(f"     {col:<18}: {display_val}")
            print()

    return {
        "filename": filename,
        "rows": total_rows,
        "cols": num_cols,
        "column_names": column_names,
        "dup_ids": dup_id_count,
        "unique_countries": unique_countries,
        "country_freq": dict(country_counts),
        "missing_counts": dict(missing_counts),
    }


def inspect_ground_truth(filepath, chunksize=100000):
    """
    Inspect train_ground_truth.tsv using chunked reading to conserve memory.
    Reports general TSV metrics plus specific matching distributions.
    """
    filename = os.path.basename(filepath)
    file_size = os.path.getsize(filepath)

    print("\n" + "=" * 80)
    print(f"GROUND TRUTH FILE: {filename}")
    print(f"Path: {filepath} ({format_size(file_size)})")
    print("=" * 80)

    total_rows = 0
    column_names = None
    column_dtypes = {}
    missing_counts = collections.defaultdict(int)
    seen_s1_ids = set()
    dup_s1_count = 0
    first_chunk_df = None

    # Ground truth specific counters
    singleton_count = 0
    one_match_count = 0
    multi_match_count = 0
    s2_match_count = 0
    s3_match_count = 0
    total_matched_ids = 0
    seen_s2_ids = set()
    seen_s3_ids = set()
    max_matches_in_row = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(filepath, sep="\t", chunksize=chunksize, keep_default_na=False)):
        if chunk_idx == 0:
            first_chunk_df = chunk.head(5).copy()
            column_names = list(chunk.columns)
            for col in chunk.columns:
                column_dtypes[col] = str(chunk[col].dtype)

        total_rows += len(chunk)

        # Missing values (empty string or NaN)
        for col in chunk.columns:
            empty_mask = (chunk[col] == "") | chunk[col].isna()
            missing_counts[col] += int(empty_mask.sum())

        for s1, m_str in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            # S1 duplicate tracking
            if s1 in seen_s1_ids:
                dup_s1_count += 1
            else:
                seen_s1_ids.add(s1)

            # Matched IDs parsing
            m_str = m_str.strip()
            if not m_str:
                singleton_count += 1
                continue

            ids = [x.strip() for x in m_str.split(",") if x.strip()]
            n_matches = len(ids)

            if n_matches == 0:
                singleton_count += 1
            elif n_matches == 1:
                one_match_count += 1
            else:
                multi_match_count += 1

            if n_matches > max_matches_in_row:
                max_matches_in_row = n_matches

            for mid in ids:
                total_matched_ids += 1
                if mid.startswith("S2-"):
                    s2_match_count += 1
                    seen_s2_ids.add(mid)
                elif mid.startswith("S3-"):
                    s3_match_count += 1
                    seen_s3_ids.add(mid)

    # Free memory
    del seen_s1_ids
    unique_s2_count = len(seen_s2_ids)
    unique_s3_count = len(seen_s3_ids)
    del seen_s2_ids
    del seen_s3_ids
    gc.collect()

    num_cols = len(column_names) if column_names else 0

    print("\n[GENERAL DATASET METRICS]")
    print(f"1. Number of rows   : {total_rows:,}")
    print(f"2. Number of columns: {num_cols}")
    print(f"3. Column names     : {column_names}")

    print("\n4. Data types & 5. Missing values per column:")
    header = f"   {'Column Name':<22} | {'Data Type':<12} | {'Missing Count':<15} | {'Missing %':<10}"
    print(header)
    print("   " + "-" * (len(header) - 3))
    for col in column_names:
        miss = missing_counts[col]
        pct = (miss / total_rows * 100.0) if total_rows > 0 else 0.0
        dtype_str = column_dtypes.get(col, "unknown")
        print(f"   {col:<22} | {dtype_str:<12} | {miss:>13,} | {pct:>8.2f}%")

    print(f"\n6. Duplicate source1_entity_id count: {dup_s1_count:,}")

    print("\n" + "-" * 80)
    print("[GROUND TRUTH MATCHING ANALYSIS]")
    print("-" * 80)
    print(f"1. Total Source 1 entities (rows) : {total_rows:,}")
    pct_single = (singleton_count / total_rows * 100.0) if total_rows > 0 else 0.0
    print(f"2. Singleton S1 entities (0 match): {singleton_count:,} ({pct_single:.2f}%)")
    pct_one = (one_match_count / total_rows * 100.0) if total_rows > 0 else 0.0
    print(f"3. Entities with exactly 1 match  : {one_match_count:,} ({pct_one:.2f}%)")
    pct_multi = (multi_match_count / total_rows * 100.0) if total_rows > 0 else 0.0
    print(f"4. Entities with multiple matches : {multi_match_count:,} ({pct_multi:.2f}%)")
    print(f"   - Max matches for a single S1  : {max_matches_in_row:,}")
    avg_matches = (total_matched_ids / total_rows) if total_rows > 0 else 0.0
    print(f"   - Average matches per S1 entity: {avg_matches:.2f}")

    pct_s2 = (s2_match_count / total_matched_ids * 100.0) if total_matched_ids > 0 else 0.0
    pct_s3 = (s3_match_count / total_matched_ids * 100.0) if total_matched_ids > 0 else 0.0
    print(f"\n5. Total S2 matches               : {s2_match_count:,} ({pct_s2:.2f}% of matched IDs)")
    print(f"   - Unique Source 2 IDs matched  : {unique_s2_count:,}")
    print(f"6. Total S3 matches               : {s3_match_count:,} ({pct_s3:.2f}% of matched IDs)")
    print(f"   - Unique Source 3 IDs matched  : {unique_s3_count:,}")
    print(f"7. Total matched IDs              : {total_matched_ids:,}")

    print("\n8. Five example ground-truth rows:")
    if first_chunk_df is not None:
        for idx, row in first_chunk_df.iterrows():
            s1_id = row["source1_entity_id"]
            m_ids = row["matched_entity_ids"]
            m_list = [x.strip() for x in m_ids.split(",") if x.strip()] if m_ids else []
            print(f"   Row #{idx + 1}:")
            print(f"     source1_entity_id : {s1_id}")
            print(f"     matched_count     : {len(m_list)}")
            display_m = m_ids if len(m_ids) < 100 else m_ids[:97] + "..."
            print(f"     matched_entity_ids: {display_m}")
            print()

    return {
        "filename": filename,
        "rows": total_rows,
        "cols": num_cols,
        "column_names": column_names,
        "dup_ids": dup_s1_count,
        "singletons": singleton_count,
        "one_match": one_match_count,
        "multi_match": multi_match_count,
        "s2_matches": s2_match_count,
        "s3_matches": s3_match_count,
        "total_matched_ids": total_matched_ids,
        "unique_s2": unique_s2_count,
        "unique_s3": unique_s3_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect datasets for the Amazon ML Challenge 2026 Business Entity Resolution."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to the dataset directory containing 'train' and 'test' folders.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000,
        help="Chunk size (number of rows) for reading large TSVs (default: 100,000).",
    )
    args = parser.parse_args()

    dataset_dir = locate_dataset_dir(args.dataset_dir)
    print(f"Located dataset directory at: {dataset_dir}")

    # Files to inspect in designated order
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    source_files = [
        os.path.join(train_dir, "train_source1.tsv"),
        os.path.join(train_dir, "train_source2.tsv"),
        os.path.join(train_dir, "train_source3.tsv"),
        os.path.join(test_dir, "test_source1.tsv"),
        os.path.join(test_dir, "test_source2.tsv"),
        os.path.join(test_dir, "test_source3.tsv"),
    ]

    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")

    results = []

    print("\n" + "#" * 80)
    print("# PART 1: SOURCE FILES INSPECTION (TRAIN & TEST)")
    print("#" * 80)

    for sf in source_files:
        if not os.path.isfile(sf):
            print(f"WARNING: File not found: {sf}")
            continue
        res = inspect_source_tsv(sf, chunksize=args.chunksize)
        results.append(res)

    print("\n" + "#" * 80)
    print("# PART 2: GROUND TRUTH INSPECTION (TRAIN)")
    print("#" * 80)

    gt_res = None
    if os.path.isfile(gt_file):
        gt_res = inspect_ground_truth(gt_file, chunksize=args.chunksize)
    else:
        print(f"WARNING: Ground truth file not found: {gt_file}")

    # Summary table across all files
    print("\n" + "=" * 90)
    print("COMPREHENSIVE SUMMARY OF ALL DATASETS")
    print("=" * 90)
    header = f"{'Dataset File':<28} | {'Rows':<12} | {'Cols':<6} | {'Duplicates':<12} | {'Unique Countries'}"
    print(header)
    print("-" * len(header))
    for r in results:
        countries_str = ", ".join([f"{k}:{v:,}" for k, v in r["country_freq"].items() if k])
        print(f"{r['filename']:<28} | {r['rows']:>10,} | {r['cols']:>4} | {r['dup_ids']:>10,} | {countries_str}")

    if gt_res:
        print("-" * len(header))
        print(f"{gt_res['filename']:<28} | {gt_res['rows']:>10,} | {gt_res['cols']:>4} | {gt_res['dup_ids']:>10,} | Ground Truth (Total Matches: {gt_res['total_matched_ids']:,})")
    print("=" * 90)
    print("\nDataset inspection completed successfully. No models trained. No submission created.")


if __name__ == "__main__":
    main()
