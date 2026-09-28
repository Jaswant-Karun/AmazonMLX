#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Step 4: Generate Training and Validation Pairs (with Hard Negative Mining)

Author: Amazon ML Challenge Team
Date: September 2026

Description:
  Uses the winning union blocking rules (Blocks 1-4 + Blocks A-G) to generate
  candidate pairs for a clean 90/10 split of Source 1 entities:
    - Positive pairs (label=1): true matching pairs from train_ground_truth.tsv
    - Hard negative pairs (label=0): blocked candidates that are not true matches
  
  Outputs:
    - student_resource/amazon_ml_solution/output/train_pairs.tsv
    - student_resource/amazon_ml_solution/output/val_pairs.tsv
"""

import sys
import os
import re
import gc
import time
import random
import unicodedata
import argparse
import collections
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# ==============================================================================
# 1. Normalization & Blocking Constants
# ==============================================================================
RE_PUNCT = re.compile(r"[^\w\s]")
RE_DOMAIN_SUFFIX = re.compile(
    r"\.(?:com|org|net|in|co|info|biz|edu|gov|io|ai|tech|me|tv|us|de|fr|uk|ca|au)(?:/.*)?$",
    re.IGNORECASE
)
RE_WWW_PREFIX = re.compile(r"^(?:https?://)?(?:www\.)?", re.IGNORECASE)
RE_FORMER_NAME = re.compile(
    r"(?:formerly|aka|a/k/a|fka|f/k/a|dba|d/b/a|trading\s*as|t/a)[:\s]+(.*)",
    re.IGNORECASE
)
RE_ADDRESS_PREFIXES = re.compile(
    r"\b(?:door\s*no|building\s*no|flat\s*no|room\s*no|plot\s*no|house\s*no|h\.?no|kh\.?no|survey\s*no|pl\s*no|ward\s*no|unit\s*no|fl|floor|block|sector|no|flat|unit|suite|ste)\b\.?\s*[:#]?",
    re.IGNORECASE
)
RE_LEADING_ZEROS = re.compile(r"^0+([1-9]\d*)$")
RE_COMPOUND_NUM = re.compile(r"\b(\d+[/_-]\d+)\b")

STOPWORDS = {"the", "a", "an", "and", "of", "for", "in", "on", "at", "to", "m", "s", "ms", "co"}

STRUCTURAL_PREFIXES = {
    "dr", "shri", "smt", "m/s", "ms", "mr", "mrs", "shree", "sri",
    "prof", "messrs", "c/o", "d/o", "s/o", "w/o", "md", "ca", "adv"
}

STRUCTURAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "enterprises", "enterprise", "company", "group",
    "holdings", "services", "solutions", "associates", "technologies",
    "tech", "sarl", "sa", "gmbh", "llp", "pllc", "co", "sons"
}

ADDR_GENERIC = {
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive", "blvd",
    "lane", "ln", "hwy", "highway", "po", "box", "apt", "unit", "floor",
    "fl", "suite", "ste", "near", "opp", "opposite", "behind", "dist",
    "district", "null", "no", "door", "room", "flat", "plot", "bldg", "building"
}


def basic_normalize(text: str) -> str:
    if not text:
        return ""
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text)
    text = RE_PUNCT.sub(" ", text).lower()
    return " ".join(text.split())


def strip_accents(text: str) -> str:
    if not text or text.isascii():
        return text.lower() if text else ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def normalize_numeric_token(tok: str) -> str:
    if tok.isdigit():
        return str(int(tok))
    m = RE_LEADING_ZEROS.match(tok)
    return m.group(1) if m else tok


def generate_all_blocking_keys(c: str, name: str, addr: str) -> dict:
    if not isinstance(c, str):
        c = ""
    if not isinstance(name, str):
        name = ""
    if not isinstance(addr, str):
        addr = ""

    c = c.strip().upper() if c else "UNKNOWN"
    keys = {}

    compounds = RE_COMPOUND_NUM.findall(addr) if ("/" in addr or "-" in addr) else []
    comp_norm = []
    for comp in compounds:
        parts = re.split(r"[/_-]", comp)
        np = [normalize_numeric_token(p) for p in parts if p]
        if len(np) > 1:
            comp_norm.append("_".join(np))

    clean_addr = RE_ADDRESS_PREFIXES.sub(" ", addr)
    norm_addr = basic_normalize(clean_addr)
    addr_toks = norm_addr.split()

    numerics = list(comp_norm)
    for t in addr_toks:
        if t.isdigit():
            c_num = normalize_numeric_token(t)
            if c_num and len(c_num) <= 8:
                numerics.append(c_num)

    postals = [n for n in numerics if len(n) in (5, 6) and "_" not in n]
    bldgs = [n for n in numerics if len(n) <= 4 and "_" not in n]
    alpha_toks = [t for t in addr_toks if t.isalpha() and len(t) >= 3 and t not in ADDR_GENERIC and t not in STOPWORDS]

    has_domain = ("." in name or "www" in name.lower())
    clean_name = name
    split_dom_toks = []
    if has_domain:
        c_dom = RE_WWW_PREFIX.sub("", name.strip())
        c_dom = RE_DOMAIN_SUFFIX.sub("", c_dom)
        clean_name = c_dom
        split_dom_toks = [
            t.lower()
            for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+", c_dom)
            if len(t) >= 2 and t.lower() not in STOPWORDS
        ]

    norm_name = basic_normalize(clean_name)
    raw_name_toks = norm_name.split()
    name_toks = [t for t in raw_name_toks if t not in STOPWORDS and t not in STRUCTURAL_SUFFIXES and len(t) >= 2]

    s_toks = list(raw_name_toks)
    while s_toks and (s_toks[0] in STRUCTURAL_PREFIXES or s_toks[0] in STOPWORDS):
        s_toks.pop(0)
    while s_toks and (s_toks[-1] in STRUCTURAL_SUFFIXES or s_toks[-1] in STOPWORDS):
        s_toks.pop()
    struct_toks = [
        t for t in s_toks
        if t not in STRUCTURAL_PREFIXES and t not in STRUCTURAL_SUFFIXES and t not in STOPWORDS and len(t) >= 2
    ]

    if "former" in name.lower() or "aka" in name.lower() or "dba" in name.lower():
        m_former = RE_FORMER_NAME.search(name)
        if m_former:
            f_toks = basic_normalize(m_former.group(1)).split()
            while f_toks and (f_toks[0] in STRUCTURAL_PREFIXES or f_toks[0] in STOPWORDS):
                f_toks.pop(0)
            while f_toks and (f_toks[-1] in STRUCTURAL_SUFFIXES or f_toks[-1] in STOPWORDS):
                f_toks.pop()
            f_sig = [
                t for t in f_toks
                if t not in STRUCTURAL_PREFIXES and t not in STRUCTURAL_SUFFIXES and t not in STOPWORDS and len(t) >= 2
            ]
            if f_sig:
                struct_toks = f_sig

    # BLOCK 1: Country + exact normalized business name
    keys["B1"] = [f"{c}|b1|{norm_name}"] if norm_name else []

    # BLOCK 2: Country + first 2 name tokens
    b2 = []
    if len(name_toks) >= 2:
        b2.append(f"{c}|b2|{name_toks[0]}_{name_toks[1]}")
    elif len(name_toks) == 1 and len(name_toks[0]) >= 3:
        b2.append(f"{c}|b2|{name_toks[0]}")
    keys["B2"] = b2

    # BLOCK 3: Country + numeric + alpha address token
    b3 = []
    if numerics and alpha_toks:
        b3.append(f"{c}|b3|{numerics[0]}_{alpha_toks[0]}")
        if len(alpha_toks) > 1 and alpha_toks[-1] != alpha_toks[0]:
            b3.append(f"{c}|b3|{numerics[0]}_{alpha_toks[-1]}")
    keys["B3"] = b3

    # BLOCK 4: Country + name token + address token
    b4 = []
    if name_toks:
        lead = name_toks[0]
        if numerics:
            b4.append(f"{c}|b4|{lead}_{numerics[0]}")
        if alpha_toks:
            b4.append(f"{c}|b4|{lead}_{alpha_toks[0]}")
    keys["B4"] = b4

    # BLOCK A: Normalized numeric address key (compound numerics & leading zeros)
    b_a = []
    if comp_norm:
        for cn in comp_norm:
            b_a.append(f"{c}|ba|cmp_{cn}")
    if numerics and alpha_toks:
        b_a.append(f"{c}|ba|{numerics[0]}_{alpha_toks[0]}")
        if len(alpha_toks) > 1 and alpha_toks[-1] != alpha_toks[0]:
            b_a.append(f"{c}|ba|{numerics[0]}_{alpha_toks[-1]}")
    keys["BA"] = b_a

    # BLOCK B: Normalized domain & de-spaced business name key
    b_b = []
    despaced_domain = "".join(raw_name_toks)
    if has_domain and len(despaced_domain) >= 4:
        b_b.append(f"{c}|bb|{despaced_domain}")
    if len(split_dom_toks) >= 2:
        b_b.append(f"{c}|bb|{split_dom_toks[0]}_{split_dom_toks[1]}")
    if struct_toks:
        compact_struct = "".join(struct_toks)
        if len(compact_struct) >= 4 and compact_struct != despaced_domain:
            b_b.append(f"{c}|bb|{compact_struct}")
    keys["BB"] = b_b

    # BLOCK C: Structural token stripped business name signature
    b_c = []
    if len(struct_toks) >= 2:
        b_c.append(f"{c}|bc|{struct_toks[0]}_{struct_toks[1]}")
    elif len(struct_toks) == 1 and len(struct_toks[0]) >= 3:
        b_c.append(f"{c}|bc|{struct_toks[0]}")
    keys["BC"] = b_c

    # BLOCK D: Bounded 3-gram prefix & suffix
    b_d = []
    if struct_toks:
        core_tok = struct_toks[0]
        if len(core_tok) >= 4:
            b_d.append(f"{c}|bd|pre_{core_tok[:3]}")
            b_d.append(f"{c}|bd|suf_{core_tok[-3:]}")
    keys["BD"] = b_d

    # BLOCK E: Locality and street combinations
    b_e = []
    if postals and bldgs:
        b_e.append(f"{c}|be|p_{postals[0]}_b_{bldgs[0]}")
    if postals and alpha_toks:
        b_e.append(f"{c}|be|p_{postals[0]}_a_{alpha_toks[0]}")
    if bldgs and alpha_toks:
        b_e.append(f"{c}|be|b_{bldgs[0]}_a_{alpha_toks[0]}")
    keys["BE"] = b_e

    # BLOCK F: Cross-field physical address match
    b_f = []
    if comp_norm and postals:
        b_f.append(f"{c}|bf|{comp_norm[0]}_{postals[0]}")
    if comp_norm and alpha_toks:
        b_f.append(f"{c}|bf|{comp_norm[0]}_{alpha_toks[0]}")
    keys["BF"] = b_f

    # BLOCK G: Multilingual & Unicode normalization
    b_g = []
    has_indic = any(0x0900 <= ord(char) <= 0x0D7F for char in name)
    if has_indic:
        indic_chars = [char for char in name if unicodedata.category(char).startswith(("L", "M", "N"))]
        indic_sig = "".join(indic_chars[:6])
        if len(indic_sig) >= 3:
            b_g.append(f"{c}|bg|indic_{indic_sig}")
    elif not name.isascii():
        accent_stripped = strip_accents(name)
        accent_toks = basic_normalize(accent_stripped).split()
        if len(accent_toks) >= 2:
            b_g.append(f"{c}|bg|lat_{accent_toks[0]}_{accent_toks[1]}")
    keys["BG"] = b_g

    return keys


# ==============================================================================
# 2. Path Locator
# ==============================================================================
def locate_dataset_dir(explicit_path=None):
    if explicit_path and os.path.isdir(explicit_path):
        return os.path.abspath(explicit_path)

    candidates = [
        os.path.join("student_resource", "dataset"),
        "dataset",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "student_resource", "dataset"),
    ]
    for cand in candidates:
        abs_cand = os.path.abspath(cand)
        if os.path.isdir(abs_cand) and os.path.isdir(os.path.join(abs_cand, "train")):
            return abs_cand

    raise FileNotFoundError("Could not find dataset directory. Specify via --dataset-dir.")


# ==============================================================================
# 3. Candidate Pairs Generator Core (Simultaneous Single-Pass Streaming)
# ==============================================================================
def generate_pairs_simultaneous(
    train_s1,
    train_to_idx,
    s1_index_train,
    train_gt_matches,
    max_train_negatives,
    val_s1,
    val_to_idx,
    s1_index_val,
    val_gt_matches,
    max_val_negatives,
    s2_path,
    s3_path,
    chunksize=200000,
    seed=42,
):
    """
    Streams S2 and S3 ONCE through both train and validation inverted indexes.
    Returns:
      (train_pairs_list, val_pairs_list)
    """
    rng = random.Random(seed)

    train_pos = collections.defaultdict(dict)
    train_neg = collections.defaultdict(dict)

    val_pos = collections.defaultdict(dict)
    val_neg = collections.defaultdict(dict)

    valid_countries = set()
    for c, _, _ in train_s1.values():
        valid_countries.add(c.upper())
    for c, _, _ in val_s1.values():
        valid_countries.add(c.upper())

    def scan_source(source_path, label):
        t_start = time.time()
        print(f"\n  [STREAMING] {label} ({os.path.basename(source_path)})...", flush=True)
        rows_scanned = 0

        for chunk in pd.read_csv(source_path, sep="\t", chunksize=chunksize, keep_default_na=False):
            t_chunk_start = time.time()
            chunk_rows = len(chunk)

            for mid, c, name, addr in zip(chunk["entity_id"], chunk["country"], chunk["business_name"], chunk["business_address"]):
                rows_scanned += 1
                c_clean = c.strip().upper() if c else "UNKNOWN"
                if c_clean not in valid_countries:
                    continue

                keys_dict = generate_all_blocking_keys(c, name, addr)

                # --- 1. Check Train Index ---
                matched_train_blocks = collections.defaultdict(list)
                for b_code, b_keys in keys_dict.items():
                    for k in b_keys:
                        if k in s1_index_train:
                            for s1_idx in s1_index_train[k]:
                                matched_train_blocks[s1_idx].append(b_code)

                if matched_train_blocks:
                    for s1_idx, b_codes in matched_train_blocks.items():
                        blocks_unique = tuple(sorted(set(b_codes)))
                        if mid in train_gt_matches[s1_idx]:
                            train_pos[s1_idx][mid] = (mid, label, c, name, addr, blocks_unique)
                        else:
                            cur_negs = train_neg[s1_idx]
                            if len(cur_negs) < max_train_negatives * 3:
                                cur_negs[mid] = (mid, label, c, name, addr, blocks_unique)
                            elif len(blocks_unique) > 1:
                                for existing_mid, item in list(cur_negs.items()):
                                    if len(item[5]) == 1:
                                        del cur_negs[existing_mid]
                                        cur_negs[mid] = (mid, label, c, name, addr, blocks_unique)
                                        break

                # --- 2. Check Validation Index ---
                matched_val_blocks = collections.defaultdict(list)
                for b_code, b_keys in keys_dict.items():
                    for k in b_keys:
                        if k in s1_index_val:
                            for s1_idx in s1_index_val[k]:
                                matched_val_blocks[s1_idx].append(b_code)

                if matched_val_blocks:
                    for s1_idx, b_codes in matched_val_blocks.items():
                        blocks_unique = tuple(sorted(set(b_codes)))
                        if mid in val_gt_matches[s1_idx]:
                            val_pos[s1_idx][mid] = (mid, label, c, name, addr, blocks_unique)
                        else:
                            cur_negs = val_neg[s1_idx]
                            if len(cur_negs) < max_val_negatives:
                                cur_negs[mid] = (mid, label, c, name, addr, blocks_unique)

            elapsed_total = time.time() - t_start
            n_t_pos = sum(len(d) for d in train_pos.values())
            n_v_pos = sum(len(d) for d in val_pos.values())
            print(
                f"    [{label}] Scanned {rows_scanned:,} rows ({elapsed_total:.1f}s) "
                f"| Train pos: {n_t_pos:,} | Val pos: {n_v_pos:,}",
                flush=True
            )

        print(f"  Finished {label}: {rows_scanned:,} rows scanned in {time.time() - t_start:.2f}s.\n", flush=True)

    scan_source(s2_path, "Source 2")
    scan_source(s3_path, "Source 3")

    # Assemble Train Pairs
    print("  Assembling training pairs...", flush=True)
    train_pairs = []
    for s1_id, (s1_c, s1_name, s1_addr) in train_s1.items():
        s1_idx = train_to_idx[s1_id]
        for mid, (target_id, target_src, target_c, target_name, target_addr, b_tuple) in train_pos[s1_idx].items():
            train_pairs.append({
                "source1_entity_id": s1_id,
                "target_entity_id": target_id,
                "target_source": target_src,
                "s1_business_name": s1_name,
                "s1_business_address": s1_addr,
                "s1_country": s1_c,
                "target_business_name": target_name,
                "target_business_address": target_addr,
                "target_country": target_c,
                "matched_by_blocks": ",".join(b_tuple),
                "num_blocks_matched": len(b_tuple),
                "label": 1,
            })
        neg_items = list(train_neg[s1_idx].values())
        neg_items.sort(key=lambda x: len(x[5]), reverse=True)
        k = max_train_negatives if train_pos[s1_idx] else (max_train_negatives // 2)
        for target_id, target_src, target_c, target_name, target_addr, b_tuple in neg_items[:k]:
            train_pairs.append({
                "source1_entity_id": s1_id,
                "target_entity_id": target_id,
                "target_source": target_src,
                "s1_business_name": s1_name,
                "s1_business_address": s1_addr,
                "s1_country": s1_c,
                "target_business_name": target_name,
                "target_business_address": target_addr,
                "target_country": target_c,
                "matched_by_blocks": ",".join(b_tuple),
                "num_blocks_matched": len(b_tuple),
                "label": 0,
            })

    # Assemble Validation Pairs
    print("  Assembling validation pairs...", flush=True)
    val_pairs = []
    for s1_id, (s1_c, s1_name, s1_addr) in val_s1.items():
        s1_idx = val_to_idx[s1_id]
        for mid, (target_id, target_src, target_c, target_name, target_addr, b_tuple) in val_pos[s1_idx].items():
            val_pairs.append({
                "source1_entity_id": s1_id,
                "target_entity_id": target_id,
                "target_source": target_src,
                "s1_business_name": s1_name,
                "s1_business_address": s1_addr,
                "s1_country": s1_c,
                "target_business_name": target_name,
                "target_business_address": target_addr,
                "target_country": target_c,
                "matched_by_blocks": ",".join(b_tuple),
                "num_blocks_matched": len(b_tuple),
                "label": 1,
            })
        neg_items = list(val_neg[s1_idx].values())
        neg_items.sort(key=lambda x: len(x[5]), reverse=True)
        for target_id, target_src, target_c, target_name, target_addr, b_tuple in neg_items[:max_val_negatives]:
            val_pairs.append({
                "source1_entity_id": s1_id,
                "target_entity_id": target_id,
                "target_source": target_src,
                "s1_business_name": s1_name,
                "s1_business_address": s1_addr,
                "s1_country": s1_c,
                "target_business_name": target_name,
                "target_business_address": target_addr,
                "target_country": target_c,
                "matched_by_blocks": ",".join(b_tuple),
                "num_blocks_matched": len(b_tuple),
                "label": 0,
            })

    return train_pairs, val_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate Training & Validation Pairs for Amazon ML Challenge 2026.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to dataset directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save pair files.")
    parser.add_argument("--train-sample-size", type=int, default=50000, help="Number of S1 training entities (default: 50,000).")
    parser.add_argument("--val-sample-size", type=int, default=10000, help="Number of S1 validation entities (default: 10,000).")
    parser.add_argument("--max-negatives", type=int, default=8, help="Max hard negatives per S1 entity in train set (default: 8).")
    parser.add_argument("--chunksize", type=int, default=200000, help="Chunk size for reading TSVs (default: 200,000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    args = parser.parse_args()

    t_start_all = time.time()
    dataset_dir = locate_dataset_dir(args.dataset_dir)
    train_dir = os.path.join(dataset_dir, "train")

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 — GENERATE TRAINING & VALIDATION PAIRS (STEP 4)")
    print("=" * 80)
    print(f"Dataset directory     : {dataset_dir}")
    print(f"Output directory      : {output_dir}")
    print(f"Train sample size     : {args.train_sample_size:,} S1 entities")
    print(f"Validation sample size: {args.val_sample_size:,} S1 entities")
    print(f"Max negatives/entity  : {args.max_negatives}")
    print(f"Random seed           : {args.seed}")

    # Step 1: Deterministic 90/10 Split of Source 1
    TOTAL_S1_ROWS = 2206821
    VAL_SPLIT_SIZE = int(TOTAL_S1_ROWS * 0.10)  # 220,682
    rng = random.Random(args.seed)
    shuffled_indices = list(range(TOTAL_S1_ROWS))
    rng.shuffle(shuffled_indices)

    val_pool = shuffled_indices[:VAL_SPLIT_SIZE]
    train_pool = shuffled_indices[VAL_SPLIT_SIZE:]

    val_selected = set(val_pool[:args.val_sample_size])
    train_selected = set(train_pool[:args.train_sample_size])

    del shuffled_indices
    del val_pool
    del train_pool
    gc.collect()

    print(f"\n[Step 1/4] Extracting {len(train_selected):,} train and {len(val_selected):,} val entities from {s1_path}...")
    t0 = time.time()

    train_s1 = {}
    val_s1 = {}
    train_to_idx = {}
    val_to_idx = {}

    curr_row = 0
    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=args.chunksize, keep_default_na=False):
        for eid, c, name, addr in zip(chunk["entity_id"], chunk["country"], chunk["business_name"], chunk["business_address"]):
            if curr_row in train_selected:
                idx = len(train_to_idx)
                train_s1[eid] = (c, name, addr)
                train_to_idx[eid] = idx
            elif curr_row in val_selected:
                idx = len(val_to_idx)
                val_s1[eid] = (c, name, addr)
                val_to_idx[eid] = idx
            curr_row += 1

        if len(train_s1) == len(train_selected) and len(val_s1) == len(val_selected):
            break

    print(f"  Extracted {len(train_s1):,} train and {len(val_s1):,} val entities in {time.time() - t0:.2f}s.")

    # Step 2: Extract Ground-Truth Matches
    print(f"\n[Step 2/4] Reading ground truth from {gt_path}...")
    t0 = time.time()
    train_gt_matches = collections.defaultdict(set)
    val_gt_matches = collections.defaultdict(set)

    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=args.chunksize, keep_default_na=False):
        matching_rows = chunk[chunk["source1_entity_id"].isin(train_s1) | chunk["source1_entity_id"].isin(val_s1)]
        for s1_id, m_str in zip(matching_rows["source1_entity_id"], matching_rows["matched_entity_ids"]):
            m_str = m_str.strip()
            if m_str:
                m_list = [m.strip() for m in m_str.split(",") if m.strip()]
                if s1_id in train_to_idx:
                    train_gt_matches[train_to_idx[s1_id]].update(m_list)
                elif s1_id in val_to_idx:
                    val_gt_matches[val_to_idx[s1_id]].update(m_list)

    total_train_gt = sum(len(s) for s in train_gt_matches.values())
    total_val_gt = sum(len(s) for s in val_gt_matches.values())
    print(f"  Ground truth loaded: {total_train_gt:,} train matches, {total_val_gt:,} val matches in {time.time() - t0:.2f}s.")

    # Step 3: Build Combined Inverted Indexes
    print("\n[Step 3/4] Building inverted blocking indexes...")
    t0 = time.time()
    s1_index_train = collections.defaultdict(list)
    for s1_id, (c, name, addr) in train_s1.items():
        s1_idx = train_to_idx[s1_id]
        keys_dict = generate_all_blocking_keys(c, name, addr)
        for b_code, b_keys in keys_dict.items():
            for k in b_keys:
                s1_index_train[k].append(s1_idx)

    s1_index_val = collections.defaultdict(list)
    for s1_id, (c, name, addr) in val_s1.items():
        s1_idx = val_to_idx[s1_id]
        keys_dict = generate_all_blocking_keys(c, name, addr)
        for b_code, b_keys in keys_dict.items():
            for k in b_keys:
                s1_index_val[k].append(s1_idx)

    MAX_NGRAM_POSTINGS = 250
    pruned_t = 0
    for k in list(s1_index_train.keys()):
        if "|bd|" in k and len(s1_index_train[k]) > MAX_NGRAM_POSTINGS:
            del s1_index_train[k]
            pruned_t += 1
    pruned_v = 0
    for k in list(s1_index_val.keys()):
        if "|bd|" in k and len(s1_index_val[k]) > MAX_NGRAM_POSTINGS:
            del s1_index_val[k]
            pruned_v += 1

    print(
        f"  Indexes built in {time.time() - t0:.2f}s: "
        f"Train={len(s1_index_train):,} keys (pruned {pruned_t:,}), "
        f"Val={len(s1_index_val):,} keys (pruned {pruned_v:,})."
    )

    # Step 4: Stream S2 & S3 in a Single Pass
    print("\n[Step 4/4] Generating pairs via single-pass streaming of S2 and S3...")
    train_pairs, val_pairs = generate_pairs_simultaneous(
        train_s1=train_s1,
        train_to_idx=train_to_idx,
        s1_index_train=s1_index_train,
        train_gt_matches=train_gt_matches,
        max_train_negatives=args.max_negatives,
        val_s1=val_s1,
        val_to_idx=val_to_idx,
        s1_index_val=s1_index_val,
        val_gt_matches=val_gt_matches,
        max_val_negatives=50,
        s2_path=s2_path,
        s3_path=s3_path,
        chunksize=args.chunksize,
        seed=args.seed,
    )

    del s1_index_train
    del s1_index_val
    gc.collect()

    # Save to disk
    train_out_path = os.path.join(output_dir, "train_pairs.tsv")
    print(f"\n  Saving {len(train_pairs):,} train pairs to {train_out_path}...")
    df_train = pd.DataFrame(train_pairs)
    df_train.to_csv(train_out_path, sep="\t", index=False)
    del train_pairs
    del df_train
    gc.collect()

    val_out_path = os.path.join(output_dir, "val_pairs.tsv")
    print(f"  Saving {len(val_pairs):,} validation pairs to {val_out_path}...")
    df_val = pd.DataFrame(val_pairs)
    df_val.to_csv(val_out_path, sep="\t", index=False)
    del val_pairs
    del df_val
    gc.collect()

    # Mirror to secondary output directory if applicable
    alt_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "amazon_ml_solution", "output"))
    if os.path.isdir(os.path.dirname(alt_output_dir)) and alt_output_dir != output_dir:
        import shutil
        os.makedirs(alt_output_dir, exist_ok=True)
        try:
            shutil.copy2(train_out_path, os.path.join(alt_output_dir, "train_pairs.tsv"))
            shutil.copy2(val_out_path, os.path.join(alt_output_dir, "val_pairs.tsv"))
            print(f"  Mirrored pair files to {alt_output_dir}")
        except Exception as e:
            pass

    print("\n" + "=" * 80)
    print("STEP 4 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Total pipeline execution time: {time.time() - t_start_all:.2f}s.")
    print(f"Train pairs file: {train_out_path}")
    print(f"Val pairs file  : {val_out_path}")
    print("Ready for Step 5: Feature Engineering.")


if __name__ == "__main__":
    main()

