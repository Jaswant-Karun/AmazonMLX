#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution Challenge
Improved Blocking Experiment (03_improved_blocking_experiment.py)

Goal:
  Significantly improve candidate generation recall by resolving real-world noise patterns
  uncovered in the first experiment (leading zeros in addresses, web domains as names,
  honorific/legal prefixes & suffixes, OCR typos, address locality signals, cross-field
  fallback, and regional/accented Unicode scripts) using ONLY existing in-dataset information.

Rules Complied With:
  1. No modification of original challenge files or dataset/ utils/ directories.
  2. No external APIs, websites, business databases, geocoding, or external data.
  3. Uses only provided train files (train_source1/2/3.tsv, train_ground_truth.tsv).
  4. TSV read with sep='\\t'.
  5. Deterministic validation split (seed=42, 50,000 Source 1 sample).
  6. Memory-conscious flat candidate counting & high-throughput streaming (< 50MB RAM for tracking).
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

# Ensure terminal handles UTF-8 characters and flushes lines immediately on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# Precompiled regexes for fast normalization
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

# Structural and linguistic tokens for normalization (Documented explicitly)
STOPWORDS = {"the", "a", "an", "and", "of", "for", "in", "on", "at", "to", "m", "s", "ms", "co"}

# BLOCK C: Structural prefixes removed/normalized conservatively:
# - Honorifics / titles: Dr, Shri, Smt, Mr, Mrs, Prof, Shree, Sri, Messrs
# - Business / legal prefixes: M/s, Ms, C/o, D/o, S/o, W/o, Md, CA, Adv
STRUCTURAL_PREFIXES = {
    "dr", "shri", "smt", "m/s", "ms", "mr", "mrs", "shree", "sri",
    "prof", "messrs", "c/o", "d/o", "s/o", "w/o", "md", "ca", "adv"
}

# BLOCK C: Structural legal suffixes removed/normalized conservatively:
# - Corporate structures: Inc, Incorporated, LLC, LLP, PLLC, Corp, Corporation,
#   Ltd, Limited, Pvt, Private, Enterprises, Enterprise, Company, Co, Group,
#   Holdings, Services, Solutions, Associates, Technologies, Tech, Sarl, SA, GmbH, Sons
STRUCTURAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "enterprises", "enterprise", "company", "group",
    "holdings", "services", "solutions", "associates", "technologies",
    "tech", "sarl", "sa", "gmbh", "llp", "pllc", "co", "sons"
}

# Common address generic stopwords to ignore for locality blocking
ADDR_GENERIC = {
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive", "blvd",
    "lane", "ln", "hwy", "highway", "po", "box", "apt", "unit", "floor",
    "fl", "suite", "ste", "near", "opp", "opposite", "behind", "dist",
    "district", "null", "no", "door", "room", "flat", "plot", "bldg", "building"
}


def basic_normalize(text: str) -> str:
    """Standard lowercase, punct-to-space, whitespace normalized string preserving Unicode characters."""
    if not text:
        return ""
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text)
    text = RE_PUNCT.sub(" ", text).lower()
    return " ".join(text.split())


def strip_accents(text: str) -> str:
    """Strip accent diacritics while preserving base Latin characters (e.g., Béverage -> beverage)."""
    if not text or text.isascii():
        return text.lower() if text else ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def normalize_numeric_token(tok: str) -> str:
    """Strip leading zeros to canonical numeric form (e.g., '003940' -> '3940', '0010' -> '10')."""
    if tok.isdigit():
        return str(int(tok))
    m = RE_LEADING_ZEROS.match(tok)
    return m.group(1) if m else tok


def generate_all_blocking_keys(c: str, name: str, addr: str) -> dict:
    """
    High-performance, single-pass generation of all blocking keys:
      OLD BLOCKS (1 – 4)
      NEW BLOCKS (A – G)
    """
    if not isinstance(c, str):
        c = ""
    if not isinstance(name, str):
        name = ""
    if not isinstance(addr, str):
        addr = ""

    c = c.strip().upper() if c else "UNKNOWN"
    keys = {}

    # =========================================================================
    # 1. Address Processing
    # =========================================================================
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

    # =========================================================================
    # 2. Business Name Processing
    # =========================================================================
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

    # BLOCK C: Structural token normalization
    s_toks = list(raw_name_toks)
    while s_toks and (s_toks[0] in STRUCTURAL_PREFIXES or s_toks[0] in STOPWORDS):
        s_toks.pop(0)
    while s_toks and (s_toks[-1] in STRUCTURAL_SUFFIXES or s_toks[-1] in STOPWORDS):
        s_toks.pop()
    struct_toks = [
        t for t in s_toks
        if t not in STRUCTURAL_PREFIXES and t not in STRUCTURAL_SUFFIXES and t not in STOPWORDS and len(t) >= 2
    ]

    # Former name / alias check: 'Company formerly: OldName' or 'A aka B'
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

    # =========================================================================
    # 3. Key Generation: OLD STRATEGIES (1 – 4)
    # =========================================================================
    # BLOCK 1: Country + exact normalized business name
    keys["BLOCK 1"] = [f"{c}|b1|{norm_name}"] if norm_name else []

    # BLOCK 2: Country + normalized business name token signature (first 2 tokens)
    b2 = []
    if len(name_toks) >= 2:
        b2.append(f"{c}|b2|{name_toks[0]}_{name_toks[1]}")
    elif len(name_toks) == 1 and len(name_toks[0]) >= 3:
        b2.append(f"{c}|b2|{name_toks[0]}")
    keys["BLOCK 2"] = b2

    # BLOCK 3: Country + selected normalized address tokens
    b3 = []
    if numerics and alpha_toks:
        b3.append(f"{c}|b3|{numerics[0]}_{alpha_toks[0]}")
        if len(alpha_toks) > 1 and alpha_toks[-1] != alpha_toks[0]:
            b3.append(f"{c}|b3|{numerics[0]}_{alpha_toks[-1]}")
    keys["BLOCK 3"] = b3

    # BLOCK 4: Country + name token + address token
    b4 = []
    if name_toks:
        lead = name_toks[0]
        if numerics:
            b4.append(f"{c}|b4|{lead}_{numerics[0]}")
        if alpha_toks:
            b4.append(f"{c}|b4|{lead}_{alpha_toks[0]}")
    keys["BLOCK 4"] = b4

    # =========================================================================
    # 4. Key Generation: NEW STRATEGIES (A – G)
    # =========================================================================
    # BLOCK A: Normalized numeric address key (leading zeros stripped, compound numerics)
    b_a = []
    if comp_norm:
        for cn in comp_norm:
            b_a.append(f"{c}|ba|cmp_{cn}")
    if numerics and alpha_toks:
        b_a.append(f"{c}|ba|{numerics[0]}_{alpha_toks[0]}")
        if len(alpha_toks) > 1 and alpha_toks[-1] != alpha_toks[0]:
            b_a.append(f"{c}|ba|{numerics[0]}_{alpha_toks[-1]}")
    keys["BLOCK A"] = b_a

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
    keys["BLOCK B"] = b_b

    # BLOCK C: Name normalization with structural prefixes & legal suffixes
    b_c = []
    if len(struct_toks) >= 2:
        b_c.append(f"{c}|bc|{struct_toks[0]}_{struct_toks[1]}")
    elif len(struct_toks) == 1 and len(struct_toks[0]) >= 3:
        b_c.append(f"{c}|bc|{struct_toks[0]}")
    keys["BLOCK C"] = b_c

    # BLOCK D: Character n-gram blocking (prefix+suffix trigrams)
    b_d = []
    if struct_toks:
        lead = struct_toks[0]
        if len(lead) >= 5:
            b_d.append(f"{c}|bd|{lead[:3]}_{lead[-3:]}")
        elif len(lead) >= 4:
            b_d.append(f"{c}|bd|{lead[:4]}")
        if len(struct_toks) >= 2 and len(struct_toks[1]) >= 3:
            b_d.append(f"{c}|bd|{lead[:3]}_{struct_toks[1][:3]}")
    keys["BLOCK D"] = b_d

    # BLOCK E: Address locality tokens (postal/building + locality/state)
    b_e = []
    if postals and bldgs:
        b_e.append(f"{c}|be|p_{postals[0]}_b_{bldgs[0]}")
    if postals and alpha_toks:
        b_e.append(f"{c}|be|p_{postals[0]}_w_{alpha_toks[0]}")
    if bldgs and alpha_toks:
        b_e.append(f"{c}|be|b_{bldgs[0]}_c_{alpha_toks[-1]}")
    keys["BLOCK E"] = b_e

    # BLOCK F: Cross-field blocking (strong address signals for aliases / rebranding)
    b_f = []
    if numerics and len(alpha_toks) >= 2:
        b_f.append(f"{c}|bf|{numerics[0]}_{alpha_toks[0]}_{alpha_toks[-1]}")
    if comp_norm and alpha_toks:
        for cn in comp_norm:
            b_f.append(f"{c}|bf|cmp_{cn}_{alpha_toks[-1]}")
    if postals and bldgs and alpha_toks:
        b_f.append(f"{c}|bf|p_{postals[0]}_b_{bldgs[0]}_{alpha_toks[0]}")
    keys["BLOCK F"] = b_f

    # BLOCK G: Multilingual-safe & accent-normalized blocking
    b_g = []
    if not name.isascii():
        astrip = strip_accents(name)
        if astrip != norm_name:
            astrip_toks = [t for t in basic_normalize(astrip).split() if t not in STOPWORDS and len(t) >= 2]
            if len(astrip_toks) >= 2:
                b_g.append(f"{c}|bg_acc|{astrip_toks[0]}_{astrip_toks[1]}")
            elif len(astrip_toks) == 1:
                b_g.append(f"{c}|bg_acc|{astrip_toks[0]}")
        # Preserve Unicode letters (L), marks (M), numbers (N) so Indic matras remain intact
        u_chars = [ch for ch in name if unicodedata.category(ch).startswith(('L', 'M', 'N')) or ch.isspace()]
        u_clean = " ".join("".join(u_chars).split())
        u_toks = [t for t in u_clean.split() if not t.isascii() and len(t) >= 2]
        if len(u_toks) >= 2:
            b_g.append(f"{c}|bg_uni|{u_toks[0]}_{u_toks[1]}")
        elif len(u_toks) == 1:
            b_g.append(f"{c}|bg_uni|{u_toks[0]}")
    keys["BLOCK G"] = b_g

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
        description="Run improved blocking experiment for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of Source 1 validation entities to evaluate (default: 50,000).",
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
        help="Path to save the experiment report.",
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
    print("AMAZON ML CHALLENGE 2026 — IMPROVED BLOCKING EXPERIMENT")
    print("=" * 80)
    print(f"Dataset directory : {dataset_dir}")
    print(f"Validation sample : {args.sample_size:,} Source 1 entities")
    print(f"Chunk size        : {args.chunksize:,}")
    print(f"Random seed       : 42 (matching previous baseline)")

    # 1. Deterministic 90/10 split indices (identical to experiment 02)
    TOTAL_S1_ROWS = 2206821
    VAL_SPLIT_SIZE = int(TOTAL_S1_ROWS * 0.10)  # 220,682
    rng = random.Random(42)
    shuffled_indices = list(range(TOTAL_S1_ROWS))
    rng.shuffle(shuffled_indices)

    val_pool = shuffled_indices[:VAL_SPLIT_SIZE]
    sample_size = min(args.sample_size, len(val_pool))
    val_sample_indices = set(val_pool[:sample_size])
    del shuffled_indices
    del val_pool
    gc.collect()

    print(f"\n[Step 1/5] Extracting {sample_size:,} validation Source 1 entities from {s1_path}...")
    t0 = time.time()
    val_s1 = {}
    val_s1_list = []
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
    gt_pairs = set()
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

    old_strategies = ["BLOCK 1", "BLOCK 2", "BLOCK 3", "BLOCK 4"]
    new_strategies = ["BLOCK A", "BLOCK B", "BLOCK C", "BLOCK D", "BLOCK E", "BLOCK F", "BLOCK G"]
    all_strategies = old_strategies + new_strategies
    eval_strategies = old_strategies + ["OLD UNION"] + new_strategies + ["NEW UNION"]

    s1_index = {strat: collections.defaultdict(list) for strat in all_strategies}

    for s1_id, (c, name, addr) in val_s1.items():
        s1_idx = s1_to_idx[s1_id]
        keys_dict = generate_all_blocking_keys(c, name, addr)
        for strat in all_strategies:
            for k in keys_dict[strat]:
                s1_index[strat][k].append(s1_idx)

    # Prune unselective keys for BLOCK D to strictly keep candidate counts bounded
    MAX_NGRAM_POSTING = 250
    pruned_d = 0
    d_keys = list(s1_index["BLOCK D"].keys())
    for k in d_keys:
        if len(s1_index["BLOCK D"][k]) > MAX_NGRAM_POSTING:
            del s1_index["BLOCK D"][k]
            pruned_d += 1
    if pruned_d > 0:
        print(f"  [BLOCK D] Pruned {pruned_d:,} high-frequency keys (> {MAX_NGRAM_POSTING} postings) to keep candidates bounded.")

    for strat in all_strategies:
        print(f"  - {strat:<8}: {len(s1_index[strat]):,} unique blocking keys indexed")
    print(f"  Inverted index built in {time.time() - t0:.2f}s.")

    # 4. Stream Source 2 and Source 3 through inverted index
    print("\n[Step 4/5] Streaming Source 2 & Source 3 through inverted index...")
    val_s1_count = len(val_s1_list)

    # Memory-conscious flat candidate tracking:
    # Since every row in S2 and S3 is a globally unique entity ID,
    # candidate counts per S1 entity are tracked via integer counters.
    # Deduplication per entity is performed per row using fast local sets.
    cand_counts = {strat: [0] * val_s1_count for strat in eval_strategies}
    recalled_pairs = {strat: set() for strat in eval_strategies}

    true_target_records = {}
    valid_countries = set(c.upper() for c, _, _ in val_s1.values())

    def scan_source(source_path, label):
        t_start = time.time()
        print(f"  Streaming {label} ({os.path.basename(source_path)})...", flush=True)
        rows_scanned = 0
        t_last_log = time.time()
        rows_since_log = 0

        for chunk in pd.read_csv(source_path, sep="\t", chunksize=args.chunksize, keep_default_na=False):
            for mid, c, name, addr in zip(chunk["entity_id"], chunk["country"], chunk["business_name"], chunk["business_address"]):
                rows_scanned += 1
                rows_since_log += 1
                c_clean = c.strip().upper() if c else "UNKNOWN"
                if c_clean not in valid_countries:
                    continue

                is_gt_target = mid in all_true_target_ids
                if is_gt_target:
                    true_target_records[mid] = (c, name, addr)

                keys_dict = generate_all_blocking_keys(c, name, addr)

                matched_by_strat = {}
                for strat in all_strategies:
                    klist = keys_dict[strat]
                    m = None
                    for k in klist:
                        if k in s1_index[strat]:
                            inds = s1_index[strat][k]
                            if m is None:
                                m = inds
                            else:
                                if isinstance(m, list):
                                    m = set(m)
                                m.update(inds)
                    if m is not None:
                        matched_by_strat[strat] = m

                if not matched_by_strat:
                    continue

                matched_old_s1 = set()
                for strat in old_strategies:
                    if strat in matched_by_strat:
                        inds = matched_by_strat[strat]
                        for s1_idx in inds:
                            cand_counts[strat][s1_idx] += 1
                            if is_gt_target and (s1_idx, mid) in gt_pairs:
                                recalled_pairs[strat].add((s1_idx, mid))
                                recalled_pairs["OLD UNION"].add((s1_idx, mid))
                                recalled_pairs["NEW UNION"].add((s1_idx, mid))
                        matched_old_s1.update(inds)

                matched_new_s1 = set()
                for strat in new_strategies:
                    if strat in matched_by_strat:
                        inds = matched_by_strat[strat]
                        for s1_idx in inds:
                            cand_counts[strat][s1_idx] += 1
                            if is_gt_target and (s1_idx, mid) in gt_pairs:
                                recalled_pairs[strat].add((s1_idx, mid))
                                recalled_pairs["NEW UNION"].add((s1_idx, mid))
                        matched_new_s1.update(inds)

                for s1_idx in matched_old_s1:
                    cand_counts["OLD UNION"][s1_idx] += 1

                for s1_idx in (matched_old_s1 | matched_new_s1):
                    cand_counts["NEW UNION"][s1_idx] += 1

            if rows_since_log >= 1000000:
                elapsed_log = time.time() - t_last_log
                rate = rows_since_log / elapsed_log if elapsed_log > 0 else 0
                print(f"    ... processed {rows_scanned:,} rows of {label} (speed: {rate:,.0f} rows/s)", flush=True)
                t_last_log = time.time()
                rows_since_log = 0

        print(f"  Finished {label}: {rows_scanned:,} rows scanned in {time.time() - t_start:.2f}s.", flush=True)

    scan_source(s2_path, "Source 2")
    scan_source(s3_path, "Source 3")

    # 5. Evaluate Metrics
    print("\n[Step 5/5] Evaluating blocking recall and candidate distributions...", flush=True)
    results = {}

    for strat in eval_strategies:
        counts = cand_counts[strat]
        total_cands = sum(counts)
        avg_cands = total_cands / val_s1_count if val_s1_count > 0 else 0.0
        med_cands = statistics.median(counts) if counts else 0.0
        max_cands = max(counts) if counts else 0
        zero_cand_s1 = sum(1 for c in counts if c == 0)

        recalled_gt_pairs = len(recalled_pairs[strat])
        missed_gt_pairs = total_gt_pairs - recalled_gt_pairs
        blocking_recall = (recalled_gt_pairs / total_gt_pairs) if total_gt_pairs > 0 else 0.0

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

    # Analysis of new recoveries over OLD UNION
    old_union_recalled = recalled_pairs["OLD UNION"]
    new_union_recalled = recalled_pairs["NEW UNION"]
    newly_recovered_pairs = new_union_recalled - old_union_recalled

    new_block_recovery_counts = {}
    for strat in new_strategies:
        unique_recovered = recalled_pairs[strat] - old_union_recalled
        new_block_recovery_counts[strat] = len(unique_recovered)

    best_block = max(new_block_recovery_counts.items(), key=lambda x: x[1])

    # Candidate growth
    old_cands = results["OLD UNION"]["total_candidates"]
    new_cands = results["NEW UNION"]["total_candidates"]
    cand_growth_pct = ((new_cands - old_cands) / old_cands * 100.0) if old_cands > 0 else 0.0

    # Collect 35 examples of newly recovered pairs
    recovered_examples = []
    for s1_idx, mid in newly_recovered_pairs:
        s1_id = val_s1_list[s1_idx]
        s1_c, s1_name, s1_addr = val_s1[s1_id]
        target_info = true_target_records.get(mid, ("UNKNOWN", "[NOT_LOADED]", "[NOT_LOADED]"))
        methods = [strat for strat in new_strategies if (s1_idx, mid) in recalled_pairs[strat]]
        recovered_examples.append({
            "s1_id": s1_id,
            "s1_name": s1_name,
            "s1_addr": s1_addr,
            "s1_country": s1_c,
            "mid": mid,
            "m_name": target_info[1],
            "m_addr": target_info[2],
            "m_country": target_info[0],
            "methods": ", ".join(methods),
        })
        if len(recovered_examples) >= 40:
            break

    # Prepare Report Text
    report_lines = []
    def log(line=""):
        report_lines.append(line)
        print(line)

    print()
    log("=" * 85)
    log("IMPROVED BLOCKING EXPERIMENT REPORT — AMAZON ML CHALLENGE 2026")
    log("=" * 85)
    log(f"Validation Sample Size   : {val_s1_count:,} Source 1 entities (seed=42)")
    log(f"Total Ground Truth Pairs : {total_gt_pairs:,}")
    log(f"Entities with GT Matches : {len(entities_with_gt):,} ({len(entities_with_gt)/val_s1_count*100:.2f}%)")
    log(f"Singleton Entities       : {val_s1_count - len(entities_with_gt):,} ({(val_s1_count - len(entities_with_gt))/val_s1_count*100:.2f}%)")
    log(f"Total Experiment Time    : {time.time() - start_total_time:.2f}s")
    log("=" * 85)

    log("\n" + "-" * 85)
    log("STRATEGY DEFINITIONS:")
    log("  OLD STRATEGIES:")
    log("    BLOCK 1: Country + exact normalized business name")
    log("    BLOCK 2: Country + normalized business name token signature (first 2 tokens)")
    log("    BLOCK 3: Country + selected normalized address tokens")
    log("    BLOCK 4: Country + name token + address token")
    log("    OLD UNION: Combined (Blocks 1 + 2 + 3 + 4)")
    log("  NEW STRATEGIES:")
    log("    BLOCK A: Normalized numeric address key (leading zeros stripped, compound numerics)")
    log("    BLOCK B: Normalized domain & de-spaced business name key")
    log("    BLOCK C: Name normalization with structural prefixes (Dr, Shri, Formerly) & legal suffixes")
    log("    BLOCK D: Character n-gram blocking (prefix+suffix trigrams, bounded retrieval)")
    log("    BLOCK E: Address locality tokens (postal/building + locality/state)")
    log("    BLOCK F: Cross-field blocking (strong address signal for aliases/trading names/multilingual)")
    log("    BLOCK G: Multilingual-safe & accent-normalized blocking (regional scripts & French diacritics)")
    log("    NEW UNION: Combined (OLD UNION + Blocks A + B + C + D + E + F + G)")
    log("-" * 85)

    log("\n" + "=" * 105)
    log("SUMMARY EVALUATION TABLE ACROSS ALL STRATEGIES")
    log("=" * 105)
    tbl_hdr = f"{'Blocking Strategy':<26} | {'Candidates':<11} | {'Avg/S1':<8} | {'Med/S1':<6} | {'Max/S1':<7} | {'Zero-Cand':<9} | {'Blocking Recall':<15} | {'Entity Recall'}"
    log(tbl_hdr)
    log("-" * len(tbl_hdr))

    display_order = (
        old_strategies +
        ["OLD UNION"] +
        new_strategies +
        ["NEW UNION"]
    )

    for strat in display_order:
        r = results[strat]
        log(
            f"{strat:<26} | "
            f"{r['total_candidates']:>11,} | "
            f"{r['avg_candidates']:>8.1f} | "
            f"{r['median_candidates']:>6.1f} | "
            f"{r['max_candidates']:>7,} | "
            f"{r['zero_candidate_s1']:>9,} | "
            f"{r['recalled_pairs']:>5,}/{total_gt_pairs:,} ({r['blocking_recall']*100:>5.2f}%) | "
            f"{r['entities_recalled']:>4,}/{len(entities_with_gt):,} ({r['entity_recall']*100:>5.2f}%)"
        )
        if strat in ["OLD UNION", "NEW UNION"]:
            log("-" * len(tbl_hdr))
    log("=" * 105)

    # Union Comparison Summary
    old_r = results["OLD UNION"]
    new_r = results["NEW UNION"]
    log("\n" + "=" * 80)
    log("OLD UNION vs NEW UNION COMPARISON")
    log("=" * 80)
    log("OLD UNION:")
    log(f"  Pair recall       : {old_r['blocking_recall']*100:.2f}% ({old_r['recalled_pairs']:,} / {total_gt_pairs:,})")
    log(f"  Entity recall     : {old_r['entity_recall']*100:.2f}% ({old_r['entities_recalled']:,} / {len(entities_with_gt):,})")
    log(f"  Average candidates: {old_r['avg_candidates']:.1f}")
    log(f"  Median candidates : {old_r['median_candidates']:.1f}")
    log(f"  Maximum candidates: {old_r['max_candidates']:,}")
    log(f"  Zero-candidate S1 : {old_r['zero_candidate_s1']:,}")

    log("\nNEW UNION:")
    log(f"  Pair recall       : {new_r['blocking_recall']*100:.2f}% ({new_r['recalled_pairs']:,} / {total_gt_pairs:,})")
    log(f"  Entity recall     : {new_r['entity_recall']*100:.2f}% ({new_r['entities_recalled']:,} / {len(entities_with_gt):,})")
    log(f"  Average candidates: {new_r['avg_candidates']:.1f}")
    log(f"  Median candidates : {new_r['median_candidates']:.1f}")
    log(f"  Maximum candidates: {new_r['max_candidates']:,}")
    log(f"  Zero-candidate S1 : {new_r['zero_candidate_s1']:,}")

    log(f"\nNet Ground-Truth Pairs Recovered: +{len(newly_recovered_pairs):,} pairs")
    log(f"Missed Ground-Truth Pairs       : {old_r['missed_pairs']:,} -> {new_r['missed_pairs']:,} (-{old_r['missed_pairs'] - new_r['missed_pairs']:,})")
    log(f"Pair-Level Recall Improvement   : {old_r['blocking_recall']*100:.2f}% -> {new_r['blocking_recall']*100:.2f}% (+{(new_r['blocking_recall'] - old_r['blocking_recall'])*100:.2f}%)")
    log(f"Entity-Level Recall Improvement : {old_r['entity_recall']*100:.2f}% -> {new_r['entity_recall']*100:.2f}% (+{(new_r['entity_recall'] - old_r['entity_recall'])*100:.2f}%)")
    log(f"Candidate Growth                : {old_cands:,} -> {new_cands:,} (+{cand_growth_pct:.2f}%)")

    log("\n" + "=" * 80)
    log("BREAKDOWN: GROUND-TRUTH PAIRS RECOVERED BY EACH NEW BLOCK")
    log("=" * 80)
    sorted_blocks = sorted(new_block_recovery_counts.items(), key=lambda x: x[1], reverse=True)
    for b_name, count in sorted_blocks:
        log(f"  - {b_name:<10}: {count:>6,} previously missed ground-truth pairs recovered")
    log(f"\nChampion Strategy: {best_block[0]} recovered the largest number of previously missed pairs ({best_block[1]:,} pairs).")

    # 30 Recovered Examples
    log("\n" + "=" * 80)
    log(f"EXAMPLES OF PREVIOUSLY MISSED PAIRS RECOVERED BY NEW BLOCKING (Showing {min(30, len(recovered_examples))})")
    log("=" * 80)
    for idx, ex in enumerate(recovered_examples[:30], start=1):
        log(f"\n--- Recovered Example #{idx} ---")
        log(f"  S1 ID                   : {ex['s1_id']}")
        log(f"  S1 Name                 : {ex['s1_name']}")
        log(f"  S1 Address              : {ex['s1_addr']}")
        log(f"  Country                 : {ex['s1_country']}")
        log(f"  Matched S2/S3 ID        : {ex['mid']}")
        log(f"  Matched Name            : {ex['m_name']}")
        log(f"  Matched Address         : {ex['m_addr']}")
        log(f"  Recovered By Method(s)  : {ex['methods']}")

    # Save to report file
    default_report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "improved_blocking_experiment_report.txt"
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
            root_report_path = os.path.join(root_experiments_dir, "improved_blocking_experiment_report.txt")
            with open(root_report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"Report copy saved to: {root_report_path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
