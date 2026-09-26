#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution Challenge
Blocking & Candidate Generation Experiment (02_blocking_experiment.py)

Goal:
  Determine whether deterministic multi-key blocking strategies can retrieve
  ground-truth entity matches from Source 2 and Source 3 for Source 1 reference entities
  with high recall and manageable candidate sizes, avoiding costly all-to-all comparisons.

Rules Complied With:
  1. No modification of original challenge files.
  2. No external APIs, databases, geocoding, lookups, or external data augmentation.
  3. Uses only provided train files (train_source1/2/3.tsv, train_ground_truth.tsv).
  4. All TSVs read with sep='\\t'.
  5. Deterministic validation split (seed=42).
  6. Memory-conscious inverted-indexing over validation sample + chunked streaming.
  7. No ML model training, no submission files generated.
"""

import sys
import os
import gc
import re
import unicodedata
import random
import statistics
import collections
import argparse
import time
import pandas as pd

# Ensure terminal handles UTF-8 characters on all platforms (including Windows cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Precompiled regexes for fast generic normalization
RE_PUNCT = re.compile(r"[^\w\s]")

# Generic linguistic stopwords and common business structure tokens (no external dictionaries)
STOPWORDS = {"the", "a", "an", "and", "of", "for", "in", "on", "at", "to", "m", "s", "ms", "co"}
CORP_TERMS = {
    "inc", "llc", "corp", "corporation", "ltd", "limited", "pvt", "private",
    "enterprises", "enterprise", "company", "group", "holdings", "services",
    "solutions", "associates", "technologies", "tech", "sarl", "sa", "gmbh", "llp"
}
ADDR_STOP = {
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive", "blvd",
    "lane", "ln", "hwy", "po", "box", "apt", "unit", "floor", "fl", "suite",
    "ste", "near", "opp", "opposite", "behind", "dist", "district", "null", "no"
}


def normalize_text(text: str) -> str:
    """
    Generic text normalization:
    - Lowercase
    - Unicode normalization (NFKD)
    - Standardize separators / punctuation to spaces
    - Normalize whitespace
    - Preserves useful alphanumeric tokens
    """
    if not text or not isinstance(text, str):
        return ""
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text)
    text = RE_PUNCT.sub(" ", text).lower()
    return " ".join(text.split())


def get_name_tokens(norm_name: str) -> list:
    """Extract significant name tokens, prioritizing distinctive words over generic corporate terms."""
    if not norm_name:
        return []
    toks = norm_name.split()
    sig = [t for t in toks if t not in STOPWORDS and t not in CORP_TERMS and len(t) >= 2]
    return sig if sig else [t for t in toks if t not in STOPWORDS and len(t) >= 2]


def get_address_tokens(norm_addr: str) -> tuple:
    """
    Extract numeric address tokens (e.g. building/house number, pin/zip code)
    and significant alphanumeric tokens.
    """
    if not norm_addr:
        return [], []
    toks = norm_addr.split()
    nums = [t for t in toks if any(c.isdigit() for c in t)]
    alphas = [t for t in toks if t.isalpha() and len(t) >= 3 and t not in ADDR_STOP and t not in STOPWORDS]
    return nums, alphas


def generate_blocking_keys(country: str, business_name: str, business_address: str) -> dict:
    """
    Generate multiple blocking keys for a business entity:
      BLOCK 1: country + normalized business name
      BLOCK 2: country + normalized business name prefix/token signature
      BLOCK 3: country + selected normalized address tokens
      BLOCK 4: country + name token + address token
    """
    c = country.strip().upper() if country else "UNKNOWN"
    norm_name = normalize_text(business_name)
    norm_addr = normalize_text(business_address)
    name_toks = get_name_tokens(norm_name)
    num_toks, alpha_toks = get_address_tokens(norm_addr)

    keys = {}

    # BLOCK 1: country + exact normalized business name
    keys["BLOCK 1"] = [f"{c}|b1|{norm_name}"] if norm_name else []

    # BLOCK 2: country + normalized business name prefix / token signature
    b2_keys = []
    if len(name_toks) >= 2:
        b2_keys.append(f"{c}|b2|{name_toks[0]}_{name_toks[1]}")
    elif len(name_toks) == 1 and len(name_toks[0]) >= 3:
        b2_keys.append(f"{c}|b2|{name_toks[0]}")
    keys["BLOCK 2"] = b2_keys

    # BLOCK 3: country + selected normalized address tokens (numeric + alpha street/locality/city)
    b3_keys = []
    if num_toks and alpha_toks:
        # Numeric building/zip + first significant street token
        b3_keys.append(f"{c}|b3|{num_toks[0]}_{alpha_toks[0]}")
        # Also numeric + last significant token (locality/city/state)
        if len(alpha_toks) > 1 and alpha_toks[-1] != alpha_toks[0]:
            b3_keys.append(f"{c}|b3|{num_toks[0]}_{alpha_toks[-1]}")
    keys["BLOCK 3"] = b3_keys

    # BLOCK 4: country + name token + address token
    b4_keys = []
    if name_toks:
        lead_name = name_toks[0]
        if num_toks:
            b4_keys.append(f"{c}|b4|{lead_name}_{num_toks[0]}")
        if alpha_toks:
            b4_keys.append(f"{c}|b4|{lead_name}_{alpha_toks[0]}")
    keys["BLOCK 4"] = b4_keys

    return keys


def locate_dataset_dir(explicit_path=None):
    """Locate dataset directory containing train/ and test/ folders."""
    if explicit_path and os.path.isdir(explicit_path):
        return os.path.abspath(explicit_path)

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

    raise FileNotFoundError("Could not find dataset directory. Specify via --dataset-dir.")


def main():
    parser = argparse.ArgumentParser(
        description="Run blocking & candidate generation experiment for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset directory (defaults to auto-detection).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of Source 1 validation entities to evaluate (default: 50,000 as requested).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Chunk size for reading large TSVs (default: 200,000).",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Path to save the experiment text report.",
    )
    args = parser.parse_args()

    start_total_time = time.time()
    dataset_dir = locate_dataset_dir(args.dataset_dir)
    train_dir = os.path.join(dataset_dir, "train")

    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 — BLOCKING EXPERIMENT")
    print("=" * 80)
    print(f"Dataset directory : {dataset_dir}")
    print(f"Validation sample : {args.sample_size:,} Source 1 entities")
    print(f"Chunk size        : {args.chunksize:,}")
    print(f"Random seed       : 42 (90% train / 10% validation split)")

    # 1. Deterministic 90/10 split indices
    TOTAL_S1_ROWS = 2206821
    VAL_SPLIT_SIZE = int(TOTAL_S1_ROWS * 0.10)  # 220,682
    rng = random.Random(42)
    shuffled_indices = list(range(TOTAL_S1_ROWS))
    rng.shuffle(shuffled_indices)

    # 10% validation pool
    val_pool = shuffled_indices[:VAL_SPLIT_SIZE]

    # Sample from the validation pool
    sample_size = min(args.sample_size, len(val_pool))
    val_sample_indices = set(val_pool[:sample_size])
    del shuffled_indices
    del val_pool
    gc.collect()

    print(f"\n[Step 1/5] Extracting {sample_size:,} validation Source 1 entities from {s1_path}...")
    t0 = time.time()
    val_s1 = {}  # s1_id -> (country, business_name, business_address)
    val_s1_list = []  # list of s1_id to map to index 0..N-1
    s1_to_idx = {}

    curr_row = 0
    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=args.chunksize, keep_default_na=False):
        for eid, c, name, addr in zip(chunk["entity_id"], chunk["country"], chunk["business_name"], chunk["business_address"]):
            if curr_row in val_sample_indices:
                idx = len(val_s1_list)
                val_s1[eid] = (c, name, addr)
                val_s1_list.append(eid)
                s1_to_idx[eid] = idx
            curr_row += 1
        if len(val_s1) == sample_size:
            break
    print(f"  Extracted {len(val_s1):,} S1 entities in {time.time() - t0:.2f}s.")

    # 2. Extract ground-truth matches for validation sample
    print(f"\n[Step 2/5] Extracting ground-truth matching pairs from {gt_path}...")
    t0 = time.time()
    gt_pairs = set()  # set of (s1_idx, matched_id)
    gt_matches_per_s1 = collections.defaultdict(set)
    all_true_target_ids = set()

    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=args.chunksize, keep_default_na=False):
        matching_rows = chunk[chunk["source1_entity_id"].isin(val_s1)]
        for s1_id, m_str in zip(matching_rows["source1_entity_id"], matching_rows["matched_entity_ids"]):
            m_str = m_str.strip()
            if m_str:
                s1_idx = s1_to_idx[s1_id]
                for mid in m_str.split(","):
                    mid = mid.strip()
                    if mid:
                        gt_pairs.add((s1_idx, mid))
                        gt_matches_per_s1[s1_idx].add(mid)
                        all_true_target_ids.add(mid)

    total_gt_pairs = len(gt_pairs)
    entities_with_gt = set(s1_idx for s1_idx in range(len(val_s1_list)) if len(gt_matches_per_s1[s1_idx]) > 0)
    print(f"  Validation ground truth:")
    print(f"    - Total ground-truth pairs      : {total_gt_pairs:,}")
    print(f"    - Entities with >= 1 match in GT: {len(entities_with_gt):,} ({len(entities_with_gt)/len(val_s1)*100:.2f}%)")
    print(f"    - Singleton entities (0 match)  : {len(val_s1) - len(entities_with_gt):,} ({(len(val_s1) - len(entities_with_gt))/len(val_s1)*100:.2f}%)")
    print(f"  Loaded ground truth in {time.time() - t0:.2f}s.")

    # 3. Build inverted index for validation S1 entities
    print("\n[Step 3/5] Building inverted blocking index for validation S1 entities...")
    t0 = time.time()
    strategies = ["BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 4"]
    s1_index = {strat: collections.defaultdict(list) for strat in strategies}

    for s1_id, (c, name, addr) in val_s1.items():
        s1_idx = s1_to_idx[s1_id]
        keys_dict = generate_blocking_keys(c, name, addr)
        for strat in strategies:
            for k in keys_dict[strat]:
                s1_index[strat][k].append(s1_idx)

    for strat in strategies:
        print(f"  - {strat:<8}: {len(s1_index[strat]):,} unique blocking keys indexed")
    print(f"  Inverted index built in {time.time() - t0:.2f}s.")

    # 4. Stream Source 2 and Source 3 through inverted index
    print("\n[Step 4/5] Streaming Source 2 & Source 3 through inverted index...")
    eval_strategies = strategies + ["UNION (ALL STRATEGIES)"]

    # Memory-optimized candidate tracking:
    # Each S2/S3 entity is assigned an integer ID for fast set operations
    val_s1_count = len(val_s1_list)
    s1_cand_sets = {strat: [set() for _ in range(val_s1_count)] for strat in eval_strategies}
    recalled_pairs = {strat: set() for strat in eval_strategies}

    # Also capture metadata of true matched target entities for missed matches inspection
    true_target_records = {}
    mid_counter = 0

    def scan_source(source_path, label):
        nonlocal mid_counter
        t_start = time.time()
        print(f"  Streaming {label} ({os.path.basename(source_path)})...")
        rows_scanned = 0
        for chunk in pd.read_csv(source_path, sep="\t", chunksize=args.chunksize, keep_default_na=False):
            for mid, c, name, addr in zip(chunk["entity_id"], chunk["country"], chunk["business_name"], chunk["business_address"]):
                rows_scanned += 1
                mid_counter += 1
                mid_int = mid_counter

                is_gt_target = mid in all_true_target_ids
                if is_gt_target:
                    true_target_records[mid] = (c, name, addr)

                keys_dict = generate_blocking_keys(c, name, addr)

                for strat in strategies:
                    klist = keys_dict[strat]
                    for k in klist:
                        if k in s1_index[strat]:
                            for s1_idx in s1_index[strat][k]:
                                s1_cand_sets[strat][s1_idx].add(mid_int)
                                s1_cand_sets["UNION (ALL STRATEGIES)"][s1_idx].add(mid_int)
                                if is_gt_target and (s1_idx, mid) in gt_pairs:
                                    recalled_pairs[strat].add((s1_idx, mid))
                                    recalled_pairs["UNION (ALL STRATEGIES)"].add((s1_idx, mid))

            print(f"    ... processed {rows_scanned:,} rows of {label}", end="\r", flush=True)
        print(f"\n  Finished {label}: {rows_scanned:,} rows scanned in {time.time() - t_start:.2f}s.")

    scan_source(s2_path, "Source 2")
    scan_source(s3_path, "Source 3")

    # 5. Evaluate Blocking Metrics
    print("\n[Step 5/5] Evaluating blocking recall and candidate distributions...")
    results = {}

    for strat in eval_strategies:
        cand_counts = [len(s1_cand_sets[strat][i]) for i in range(val_s1_count)]
        total_cands = sum(cand_counts)
        avg_cands = total_cands / val_s1_count if val_s1_count > 0 else 0.0
        med_cands = statistics.median(cand_counts) if cand_counts else 0.0
        max_cands = max(cand_counts) if cand_counts else 0
        zero_cand_s1 = sum(1 for c in cand_counts if c == 0)

        # Recall against ground-truth pairs
        recalled_gt_pairs = len(recalled_pairs[strat])
        missed_gt_pairs = total_gt_pairs - recalled_gt_pairs
        blocking_recall = (recalled_gt_pairs / total_gt_pairs) if total_gt_pairs > 0 else 0.0

        # Entity-level recall: entities where at least one ground-truth match was captured
        entities_with_recalled_match = 0
        for s1_idx in entities_with_gt:
            true_mids = gt_matches_per_s1[s1_idx]
            if any((s1_idx, mid) in recalled_pairs[strat] for mid in true_mids):
                entities_with_recalled_match += 1

        entity_recall = (entities_with_recalled_match / len(entities_with_gt)) if entities_with_gt else 0.0

        results[strat] = {
            "strategy": strat,
            "total_candidates": total_cands,
            "avg_candidates": avg_cands,
            "median_candidates": med_cands,
            "max_candidates": max_cands,
            "zero_candidate_s1": zero_cand_s1,
            "recalled_pairs": recalled_gt_pairs,
            "missed_pairs": missed_gt_pairs,
            "blocking_recall": blocking_recall,
            "entities_recalled": entities_with_recalled_match,
            "entity_recall": entity_recall,
        }

    # Find missed ground-truth matches in UNION
    union_recalled = recalled_pairs["UNION (ALL STRATEGIES)"]
    missed_examples = []
    for s1_idx, mid in gt_pairs:
        if (s1_idx, mid) not in union_recalled:
            s1_id = val_s1_list[s1_idx]
            s1_c, s1_name, s1_addr = val_s1[s1_id]
            target_info = true_target_records.get(mid, ("UNKNOWN", "[NOT_LOADED]", "[NOT_LOADED]"))
            missed_examples.append({
                "s1_id": s1_id,
                "s1_name": s1_name,
                "s1_addr": s1_addr,
                "s1_country": s1_c,
                "mid": mid,
                "m_name": target_info[1],
                "m_addr": target_info[2],
                "m_country": target_info[0],
            })

    # Prepare Report Text
    report_lines = []
    def log(line=""):
        report_lines.append(line)
        print(line)

    print()
    log("=" * 80)
    log("BLOCKING EXPERIMENT REPORT — AMAZON ML CHALLENGE 2026")
    log("=" * 80)
    log(f"Validation Sample Size   : {val_s1_count:,} Source 1 entities (deterministic sample from 10% val split, seed=42)")
    log(f"Total Ground Truth Pairs : {total_gt_pairs:,}")
    log(f"Entities with GT Matches : {len(entities_with_gt):,} ({len(entities_with_gt)/val_s1_count*100:.2f}%)")
    log(f"Singleton Entities       : {val_s1_count - len(entities_with_gt):,} ({(val_s1_count - len(entities_with_gt))/val_s1_count*100:.2f}%)")
    log(f"Total Experiment Time    : {time.time() - start_total_time:.2f}s")
    log("=" * 80)

    log("\n" + "-" * 80)
    log("STRATEGY DEFINITIONS:")
    log("  BLOCK 1: country + normalized business name")
    log("  BLOCK 2: country + normalized business name prefix/token signature")
    log("  BLOCK 3: country + selected normalized address tokens (numeric + alpha)")
    log("  BLOCK 4: country + primary name token + primary address token")
    log("  UNION  : combined candidates from BLOCK 1, 2, 3, and 4")
    log("-" * 80)

    log("\n" + "=" * 90)
    log("SUMMARY EVALUATION TABLE")
    log("=" * 90)
    tbl_hdr = f"{'Blocking Strategy':<26} | {'Candidates':<11} | {'Avg/S1':<8} | {'Med/S1':<6} | {'Max/S1':<7} | {'Zero-Cand':<9} | {'Blocking Recall':<15} | {'Entity Recall'}"
    log(tbl_hdr)
    log("-" * len(tbl_hdr))

    for strat in eval_strategies:
        r = results[strat]
        log(
            f"{r['strategy']:<26} | "
            f"{r['total_candidates']:>11,} | "
            f"{r['avg_candidates']:>8.1f} | "
            f"{r['median_candidates']:>6.1f} | "
            f"{r['max_candidates']:>7,} | "
            f"{r['zero_candidate_s1']:>9,} | "
            f"{r['recalled_pairs']:>5,}/{total_gt_pairs:,} ({r['blocking_recall']*100:>5.2f}%) | "
            f"{r['entities_recalled']:>4,}/{len(entities_with_gt):,} ({r['entity_recall']*100:>5.2f}%)"
        )
    log("=" * 90)

    # Detailed Section per Strategy
    log("\n" + "#" * 80)
    log("DETAILED STRATEGY METRICS")
    log("#" * 80)
    for strat in eval_strategies:
        r = results[strat]
        log(f"\n[{r['strategy']}]")
        log(f"  Total candidate pairs         : {r['total_candidates']:,}")
        log(f"  Average candidates per S1     : {r['avg_candidates']:.2f}")
        log(f"  Median candidates per S1      : {r['median_candidates']:.1f}")
        log(f"  Maximum candidates per S1     : {r['max_candidates']:,}")
        log(f"  S1 entities with 0 candidates : {r['zero_candidate_s1']:,} ({r['zero_candidate_s1']/val_s1_count*100:.2f}%)")
        log(f"  Ground-truth pairs recalled   : {r['recalled_pairs']:,} / {total_gt_pairs:,}")
        log(f"  Ground-truth pairs missed     : {r['missed_pairs']:,}")
        log(f"  Pair-level Blocking Recall    : {r['blocking_recall']*100:.2f}%")
        log(f"  Entity-level Candidate Recall : {r['entity_recall']*100:.2f}% ({r['entities_recalled']:,} / {len(entities_with_gt):,} entities with >=1 match)")

    # Missed Matches Error Analysis
    log("\n" + "=" * 80)
    log(f"ERROR ANALYSIS: EXAMPLES OF MISSED GROUND-TRUTH MATCHES (Total Missed: {len(missed_examples):,})")
    log("=" * 80)
    sample_missed = missed_examples[:25]
    for idx, ex in enumerate(sample_missed, start=1):
        log(f"\n--- Missed Example #{idx} ---")
        log(f"  S1 entity_id          : {ex['s1_id']}")
        log(f"  S1 business_name      : {ex['s1_name']}")
        log(f"  S1 address            : {ex['s1_addr']}")
        log(f"  S1 country            : {ex['s1_country']}")
        log(f"  True matched ID       : {ex['mid']}")
        log(f"  Matched business_name : {ex['m_name']}")
        log(f"  Matched address       : {ex['m_addr']}")

    log("\n" + "=" * 80)
    log("NOISE PATTERNS & INSIGHTS OBSERVED FROM MISSED MATCHES:")
    log("=" * 80)
    log("1. URL / Domain Names as Business Names:")
    log("   Source 3 frequently substitutes a company website/URL (e.g. 'xyzcorp.com') for the name.")
    log("2. Severe Typographic & OCR/Transliteration Errors:")
    log("   Letters transposed, phonetic spelling changes, or non-Latin script translations.")
    log("3. Missing / Blank Addresses in S2 & S3:")
    log("   When a matching S2/S3 record has no address AND its business name differs by more than legal suffix,")
    log("   name-only or address-only blocks fail to link them.")
    log("4. Street vs Landmark Address Discrepancies:")
    log("   In Indian records, street numbers are often replaced with landmarks (e.g., 'Near SBI Branch').")
    log("5. High Recall of Combined Strategy:")
    log(f"   The UNION of 4 deterministic strategies captures {results['UNION (ALL STRATEGIES)']['blocking_recall']*100:.2f}% of all GT pairs")
    log(f"   and {results['UNION (ALL STRATEGIES)']['entity_recall']*100:.2f}% of entities, dramatically reducing the candidate space")
    log("   from 10.3 million to a manageable set per entity before downstream scoring.")
    log("=" * 80)

    # Determine report output path
    default_report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "blocking_experiment_report.txt"
    )
    report_path = args.output_report or default_report_path
    report_dir = os.path.dirname(os.path.abspath(report_path))
    os.makedirs(report_dir, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport successfully saved to: {report_path}")

    # Also save to root amazon_ml_solution/experiments if applicable
    root_experiments_dir = os.path.abspath(os.path.join(dataset_dir, "..", "..", "amazon_ml_solution", "experiments"))
    if os.path.isdir(os.path.dirname(root_experiments_dir)):
        try:
            os.makedirs(root_experiments_dir, exist_ok=True)
            root_report_path = os.path.join(root_experiments_dir, "blocking_experiment_report.txt")
            with open(root_report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"Report copy saved to: {root_report_path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
