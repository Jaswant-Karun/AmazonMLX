#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Module: Batch Feature Extraction v2 (Phases 5, 6, 7)
src/extract_features_v2_batch.py

Extracts all 46 features from train_pairs.tsv and val_pairs.tsv using multi-core parallelization,
saving high-performance Parquet datasets:
  - output/train_features_v2.parquet
  - output/val_features_v2.parquet
"""

import sys
import os
import time
import argparse
import multiprocessing as mp
import pandas as pd
import numpy as np

# Ensure src is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from normalization_engine import EntityProfileV2
from feature_extractor_v2 import extract_features_v2, FEATURE_NAMES_V2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def _process_chunk_worker(chunk_tuples):
    """
    Worker function to process a batch of entity tuples with local profile caching.
    Tuple structure:
      (s1_id, tgt_id, tgt_src, s1_c, s1_n, s1_a, tgt_c, tgt_n, tgt_a, blocks, num_blocks, label)
    """
    cache = {}
    records = []
    for s1_c, s1_n, s1_a, tgt_c, tgt_n, tgt_a in chunk_tuples:
        k1 = (s1_c, s1_n, s1_a)
        if k1 not in cache:
            cache[k1] = EntityProfileV2(s1_c, s1_n, s1_a)
        p1 = cache[k1]

        k2 = (tgt_c, tgt_n, tgt_a)
        if k2 not in cache:
            cache[k2] = EntityProfileV2(tgt_c, tgt_n, tgt_a)
        p2 = cache[k2]

        feats = extract_features_v2(p1, p2)
        records.append([feats[col] for col in FEATURE_NAMES_V2])

    return records


def extract_features_from_tsv(input_tsv: str, output_parquet: str, num_workers: int = 8, chunksize: int = 25000):
    print("=" * 80)
    print(f"BATCH FEATURE EXTRACTION V2: {os.path.basename(input_tsv)} -> {os.path.basename(output_parquet)}")
    print(f"Workers: {num_workers} | Features: {len(FEATURE_NAMES_V2)} | Chunksize: {chunksize}")
    print("=" * 80)

    t0 = time.time()
    total_rows = 0

    meta_cols = [
        "source1_entity_id", "target_entity_id", "target_source",
        "matched_by_blocks", "num_blocks_matched", "label"
    ]

    all_dfs = []
    
    with mp.Pool(processes=num_workers) as pool:
        for chunk_idx, chunk in enumerate(pd.read_csv(input_tsv, sep="\t", chunksize=chunksize, keep_default_na=False)):
            c_t0 = time.time()
            chunk_len = len(chunk)
            total_rows += chunk_len

            # Extract tuple lists for parallel workers
            tuples_to_proc = list(zip(
                chunk["s1_country"],
                chunk["s1_business_name"],
                chunk["s1_business_address"],
                chunk["target_country"],
                chunk["target_business_name"],
                chunk["target_business_address"]
            ))

            # Split chunk into sub-chunks for pool workers
            subchunk_size = max(1, len(tuples_to_proc) // num_workers)
            subchunks = [
                tuples_to_proc[i:i + subchunk_size]
                for i in range(0, len(tuples_to_proc), subchunk_size)
            ]

            results = pool.map(_process_chunk_worker, subchunks)
            flattened = [item for sublist in results for item in sublist]

            # Construct DataFrame for this chunk
            feat_df = pd.DataFrame(flattened, columns=FEATURE_NAMES_V2, dtype=np.float32)
            
            # Combine with metadata columns
            for mc in meta_cols:
                if mc in chunk.columns:
                    feat_df[mc] = chunk[mc].values

            all_dfs.append(feat_df)
            print(f"  Chunk {chunk_idx + 1:>3}: {chunk_len:,} rows in {time.time() - c_t0:.2f}s "
                  f"(Cumulative: {total_rows:,} rows, {time.time() - t0:.1f}s elapsed)")

    print(f"\nConcatenating {len(all_dfs)} chunks...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"Saving Parquet to {output_parquet}...")
    final_df.to_parquet(output_parquet, index=False, engine="pyarrow", compression="snappy")
    
    dur = time.time() - t0
    print(f"SUCCESS: Extracted {len(final_df):,} rows x {final_df.shape[1]} cols in {dur:.2f}s ({len(final_df)/dur:.0f} rows/s).")
    print(f"File size: {os.path.getsize(output_parquet) / (1024 * 1024):.2f} MB")
    return final_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-only", action="store_true", help="Extract validation features only.")
    parser.add_argument("--train-only", action="store_true", help="Extract train features only.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers.")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(BASE_DIR), "output")
    os.makedirs(out_dir, exist_ok=True)

    val_tsv = os.path.join(out_dir, "val_pairs.tsv")
    val_parquet = os.path.join(out_dir, "val_features_v2.parquet")

    train_tsv = os.path.join(out_dir, "train_pairs.tsv")
    train_parquet = os.path.join(out_dir, "train_features_v2.parquet")

    if not args.train_only:
        extract_features_from_tsv(val_tsv, val_parquet, num_workers=args.workers)

    if not args.val_only:
        extract_features_from_tsv(train_tsv, train_parquet, num_workers=args.workers)


if __name__ == "__main__":
    main()
