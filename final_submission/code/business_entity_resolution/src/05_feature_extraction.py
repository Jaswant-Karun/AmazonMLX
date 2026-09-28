#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Step 5: Feature Engineering & Parquet Extraction (05_feature_extraction.py)

Author: Amazon ML Challenge Team
Date: September 2026

Description:
  Reads candidate pairs generated in Step 4 (train_pairs.tsv and val_pairs.tsv),
  extracts all 33 pairwise normalized lexical, phonetic, numeric, token, character,
  and cross-domain interaction features, and outputs high-performance Parquet datasets:
    - output/train_features.parquet
    - output/val_features.parquet
    - experiments/feature_extraction_report.txt
"""

import sys
import os
import gc
import re
import time
import math
import unicodedata
import argparse
import collections
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# ==============================================================================
# 1. Normalization & Tokenization Primitives
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

STOPWORDS = {
    "the", "a", "an", "and", "of", "for", "in", "on", "at", "to", "m", "s", "ms",
    "co", "by", "with", "from", "as", "is", "or", "into", "near", "opp", "behind"
}

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


def strip_domain_name(name: str) -> str:
    if not name or ("." not in name and "www" not in name.lower()):
        return name
    c_dom = RE_WWW_PREFIX.sub("", name.strip())
    c_dom = RE_DOMAIN_SUFFIX.sub("", c_dom)
    return c_dom


def fast_levenshtein(s1: str, s2: str, max_dist: int = 15) -> int:
    """Bounded fast Levenshtein distance in pure Python."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > max_dist:
        return max_dist + 1
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    if not len1:
        return min(len2, max_dist + 1)

    prev = list(range(len1 + 1))
    for j, c2 in enumerate(s2):
        curr = [j + 1] * (len1 + 1)
        min_row_val = curr[0]
        for i, c1 in enumerate(s1):
            cost = 0 if c1 == c2 else 1
            val = min(prev[i + 1] + 1, curr[i] + 1, prev[i] + cost)
            curr[i + 1] = val
            if val < min_row_val:
                min_row_val = val
        if min_row_val > max_dist:
            return max_dist + 1
        prev = curr
    return prev[len1]


def get_char_ngrams(s: str, n: int = 3) -> set:
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0.0


def containment_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    min_size = min(len(set1), len(set2))
    return intersection / min_size if min_size else 0.0


def longest_common_prefix_len(s1: str, s2: str) -> int:
    lim = min(len(s1), len(s2))
    idx = 0
    while idx < lim and s1[idx] == s2[idx]:
        idx += 1
    return idx


# ==============================================================================
# 2. Entity Profile Representation Cache
# ==============================================================================
class EntityProfile:
    __slots__ = (
        "raw_name", "raw_addr", "country",
        "norm_name", "norm_addr",
        "name_len", "addr_len",
        "despaced_name",
        "name_toks", "name_tok_set",
        "meaningful_name_toks", "meaningful_name_tok_set",
        "first_name_tok",
        "addr_toks", "addr_tok_set",
        "numerics", "numeric_set", "first_numeric",
        "locality_toks", "locality_tok_set",
        "name_char_ngrams", "addr_char_ngrams",
        "is_addr_missing"
    )

    def __init__(self, country: str, business_name: str, business_address: str):
        self.country = country.strip().upper() if country else "UNKNOWN"
        self.raw_name = business_name if isinstance(business_name, str) else ""
        self.raw_addr = business_address if isinstance(business_address, str) else ""

        # Normalization
        clean_name_dom = strip_domain_name(self.raw_name)
        self.norm_name = basic_normalize(clean_name_dom)
        clean_addr_pref = RE_ADDRESS_PREFIXES.sub(" ", self.raw_addr)
        self.norm_addr = basic_normalize(clean_addr_pref)

        self.name_len = len(self.norm_name)
        self.addr_len = len(self.norm_addr)
        self.is_addr_missing = 1 if self.addr_len == 0 else 0

        # Tokens
        raw_name_toks = self.norm_name.split()
        self.name_toks = raw_name_toks
        self.name_tok_set = set(raw_name_toks)
        self.despaced_name = "".join(raw_name_toks)

        # Meaningful tokens
        s_toks = list(raw_name_toks)
        while s_toks and (s_toks[0] in STRUCTURAL_PREFIXES or s_toks[0] in STOPWORDS):
            s_toks.pop(0)
        while s_toks and (s_toks[-1] in STRUCTURAL_SUFFIXES or s_toks[-1] in STOPWORDS):
            s_toks.pop()

        m_toks = [
            t for t in s_toks
            if t not in STRUCTURAL_PREFIXES and t not in STRUCTURAL_SUFFIXES and t not in STOPWORDS and len(t) >= 2
        ]
        self.meaningful_name_toks = m_toks
        self.meaningful_name_tok_set = set(m_toks)
        self.first_name_tok = m_toks[0] if m_toks else (raw_name_toks[0] if raw_name_toks else "")

        # Address decomposition
        compounds = RE_COMPOUND_NUM.findall(self.raw_addr) if ("/" in self.raw_addr or "-" in self.raw_addr) else []
        comp_norm = []
        for comp in compounds:
            parts = re.split(r"[/_-]", comp)
            np = [normalize_numeric_token(p) for p in parts if p]
            if len(np) > 1:
                comp_norm.append("_".join(np))

        addr_toks = self.norm_addr.split()
        self.addr_toks = addr_toks
        self.addr_tok_set = set(addr_toks)

        numerics = list(comp_norm)
        locality = []
        for t in addr_toks:
            if t.isdigit():
                c_num = normalize_numeric_token(t)
                if c_num and len(c_num) <= 8:
                    numerics.append(c_num)
            elif t.isalpha() and len(t) >= 3 and t not in ADDR_GENERIC and t not in STOPWORDS:
                locality.append(t)

        self.numerics = numerics
        self.numeric_set = set(numerics)
        self.first_numeric = numerics[0] if numerics else ""

        self.locality_toks = locality
        self.locality_tok_set = set(locality)

        # Character 3-grams
        self.name_char_ngrams = get_char_ngrams(self.norm_name, 3)
        self.addr_char_ngrams = get_char_ngrams(self.norm_addr, 3)


# ==============================================================================
# 3. Pairwise Feature Computation Function
# ==============================================================================
def compute_pairwise_features(s1: EntityProfile, cand: EntityProfile) -> dict:
    feats = {}

    # --- A. NAME FEATURES ---
    feats["name_exact_match"] = 1 if (s1.norm_name and s1.norm_name == cand.norm_name) else 0

    max_nl = max(s1.name_len, cand.name_len, 1)
    min_nl = min(s1.name_len, cand.name_len)
    feats["name_char_len_s1"] = s1.name_len
    feats["name_char_len_cand"] = cand.name_len
    feats["name_char_len_diff"] = abs(s1.name_len - cand.name_len)
    feats["name_char_len_ratio"] = min_nl / max_nl

    feats["name_char_sim"] = jaccard_similarity(s1.name_char_ngrams, cand.name_char_ngrams)

    if feats["name_exact_match"] == 1:
        feats["name_levenshtein_sim"] = 1.0
    else:
        cap = max(12, int(0.5 * max_nl))
        ld = fast_levenshtein(s1.norm_name, cand.norm_name, max_dist=cap)
        feats["name_levenshtein_sim"] = max(0.0, 1.0 - ld / max_nl)

    shared_toks = s1.meaningful_name_tok_set & cand.meaningful_name_tok_set
    feats["name_token_overlap"] = len(shared_toks)
    feats["name_token_jaccard"] = jaccard_similarity(s1.meaningful_name_tok_set, cand.meaningful_name_tok_set)
    feats["name_token_containment"] = containment_similarity(s1.meaningful_name_tok_set, cand.meaningful_name_tok_set)

    if s1.first_name_tok and cand.first_name_tok:
        if s1.first_name_tok == cand.first_name_tok:
            feats["name_first_token_sim"] = 1.0
        else:
            max_f = max(len(s1.first_name_tok), len(cand.first_name_tok), 1)
            ld_f = fast_levenshtein(s1.first_name_tok, cand.first_name_tok, max_dist=4)
            feats["name_first_token_sim"] = max(0.0, 1.0 - ld_f / max_f)
    else:
        feats["name_first_token_sim"] = 0.0

    lcp = longest_common_prefix_len(s1.norm_name, cand.norm_name)
    feats["name_prefix_sim"] = lcp / max_nl

    if s1.despaced_name and cand.despaced_name:
        if s1.despaced_name == cand.despaced_name:
            feats["name_alphanumeric_sim"] = 1.0
        else:
            max_dn = max(len(s1.despaced_name), len(cand.despaced_name), 1)
            cap_d = max(10, int(0.4 * max_dn))
            ld_dn = fast_levenshtein(s1.despaced_name, cand.despaced_name, max_dist=cap_d)
            feats["name_alphanumeric_sim"] = max(0.0, 1.0 - ld_dn / max_dn)
    else:
        feats["name_alphanumeric_sim"] = 0.0

    # --- B. ADDRESS FEATURES ---
    feats["addr_exact_match"] = 1 if (s1.norm_addr and s1.norm_addr == cand.norm_addr) else 0

    max_al = max(s1.addr_len, cand.addr_len, 1)
    min_al = min(s1.addr_len, cand.addr_len)
    feats["addr_char_len_diff"] = abs(s1.addr_len - cand.addr_len)
    feats["addr_char_len_ratio"] = min_al / max_al if (s1.addr_len and cand.addr_len) else 0.0

    feats["addr_char_sim"] = jaccard_similarity(s1.addr_char_ngrams, cand.addr_char_ngrams)

    if feats["addr_exact_match"] == 1:
        feats["addr_levenshtein_sim"] = 1.0
    elif s1.addr_len == 0 or cand.addr_len == 0:
        feats["addr_levenshtein_sim"] = 0.0
    else:
        cap_a = max(15, int(0.5 * max_al))
        ld_a = fast_levenshtein(s1.norm_addr, cand.norm_addr, max_dist=cap_a)
        feats["addr_levenshtein_sim"] = max(0.0, 1.0 - ld_a / max_al)

    shared_addr_toks = s1.addr_tok_set & cand.addr_tok_set
    feats["addr_token_overlap"] = len(shared_addr_toks)
    feats["addr_token_jaccard"] = jaccard_similarity(s1.addr_tok_set, cand.addr_tok_set)

    shared_nums = s1.numeric_set & cand.numeric_set
    feats["addr_numeric_overlap"] = len(shared_nums)
    feats["addr_numeric_jaccard"] = jaccard_similarity(s1.numeric_set, cand.numeric_set)

    if s1.first_numeric and cand.first_numeric:
        feats["addr_house_number_match"] = 1 if s1.first_numeric == cand.first_numeric else 0
    else:
        feats["addr_house_number_match"] = -1

    shared_loc = s1.locality_tok_set & cand.locality_tok_set
    feats["addr_locality_token_overlap"] = len(shared_loc)
    feats["addr_locality_jaccard"] = jaccard_similarity(s1.locality_tok_set, cand.locality_tok_set)

    # --- C. CROSS & COMPOSITIONAL FEATURES ---
    feats["country_match"] = 1 if (s1.country and s1.country == cand.country) else 0

    name_s = feats["name_levenshtein_sim"]
    addr_s = feats["addr_levenshtein_sim"]
    feats["name_addr_combined_sim"] = 0.65 * name_s + 0.35 * addr_s

    if (name_s + addr_s) > 0:
        feats["name_addr_harmonic_sim"] = 2 * (name_s * addr_s) / (name_s + addr_s)
    else:
        feats["name_addr_harmonic_sim"] = 0.0

    feats["both_strong"] = 1 if (name_s >= 0.70 and addr_s >= 0.60) else 0
    feats["name_strong_addr_weak"] = 1 if (name_s >= 0.80 and addr_s < 0.30) else 0
    feats["addr_strong_name_weak"] = 1 if (addr_s >= 0.80 and name_s < 0.40) else 0
    feats["cand_addr_missing"] = cand.is_addr_missing
    feats["s1_addr_missing"] = s1.is_addr_missing

    return feats


# ==============================================================================
# 4. Batch Feature Extraction Pipeline
# ==============================================================================
def process_pairs_file(input_path: str, output_parquet_path: str, chunksize: int = 100000):
    t_start = time.time()
    print(f"\nProcessing pairs from: {input_path}")
    print(f"Target Parquet output : {output_parquet_path}")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input pairs file not found: {input_path}")

    # Profile cache
    profile_cache = {}

    def get_profile(c, n, a):
        key = (c, n, a)
        if key not in profile_cache:
            profile_cache[key] = EntityProfile(c, n, a)
        return profile_cache[key]

    total_pairs = 0
    total_pos = 0
    chunks_processed = 0

    all_dfs = []

    for chunk in pd.read_csv(input_path, sep="\t", chunksize=chunksize, keep_default_na=False):
        t_c = time.time()
        chunk_len = len(chunk)
        total_pairs += chunk_len

        feat_list = []
        meta_s1_id = []
        meta_target_id = []
        meta_target_src = []
        meta_num_blocks = []
        labels = []

        has_num_blocks = "num_blocks_matched" in chunk.columns
        has_label = "label" in chunk.columns

        for row in chunk.itertuples(index=False):
            s1_prof = get_profile(row.s1_country, row.s1_business_name, row.s1_business_address)
            target_prof = get_profile(row.target_country, row.target_business_name, row.target_business_address)

            f = compute_pairwise_features(s1_prof, target_prof)

            # Extra features from candidate generation
            if has_num_blocks:
                f["num_blocks_matched"] = getattr(row, "num_blocks_matched", 1)
            else:
                f["num_blocks_matched"] = 1

            lbl = getattr(row, "label", -1) if has_label else -1
            if lbl == 1:
                total_pos += 1

            meta_s1_id.append(row.source1_entity_id)
            meta_target_id.append(row.target_entity_id)
            meta_target_src.append(row.target_source)
            labels.append(lbl)
            feat_list.append(f)

        df_feat = pd.DataFrame(feat_list)
        df_feat.insert(0, "source1_entity_id", meta_s1_id)
        df_feat.insert(1, "target_entity_id", meta_target_id)
        df_feat.insert(2, "target_source", meta_target_src)
        if has_label:
            df_feat["label"] = labels

        all_dfs.append(df_feat)
        chunks_processed += 1
        print(f"  Chunk {chunks_processed}: {chunk_len:,} pairs processed in {time.time() - t_c:.2f}s (Cache size: {len(profile_cache):,})")

        # Periodically prune cache if getting too large
        if len(profile_cache) > 500000:
            profile_cache.clear()
            gc.collect()

    print(f"  Concatenating {len(all_dfs)} chunks into final DataFrame...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    profile_cache.clear()
    gc.collect()

    # Cast datatypes for ultra-efficient storage
    float_cols = [c for c in final_df.columns if final_df[c].dtype == "float64"]
    for c in float_cols:
        final_df[c] = final_df[c].astype("float32")

    int_cols = [c for c in final_df.columns if final_df[c].dtype == "int64"]
    for c in int_cols:
        final_df[c] = final_df[c].astype("int16")

    if "label" in final_df.columns:
        final_df["label"] = final_df["label"].astype("int8")

    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    final_df.to_parquet(output_parquet_path, engine="pyarrow", compression="snappy", index=False)
    file_size_mb = os.path.getsize(output_parquet_path) / (1024 * 1024)

    print(f"  Saved Parquet: {output_parquet_path} ({file_size_mb:.2f} MB, {len(final_df):,} rows)")
    print(f"  Total processing time: {time.time() - t_start:.2f}s")
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Step 5: Feature Engineering for Candidate Pairs.")
    parser.add_argument("--train-pairs", type=str, default=None, help="Path to train_pairs.tsv.")
    parser.add_argument("--val-pairs", type=str, default=None, help="Path to val_pairs.tsv.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save feature parquet files.")
    parser.add_argument("--chunksize", type=int, default=100000, help="Chunk size for pair processing.")
    args = parser.parse_args()

    t_all = time.time()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(base_dir, "output")
    exp_dir = os.path.join(base_dir, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    train_pairs_path = args.train_pairs or os.path.join(out_dir, "train_pairs.tsv")
    val_pairs_path = args.val_pairs or os.path.join(out_dir, "val_pairs.tsv")

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 — STEP 5: FEATURE EXTRACTION PIPELINE")
    print("=" * 80)
    print(f"Train pairs input  : {train_pairs_path}")
    print(f"Val pairs input    : {val_pairs_path}")
    print(f"Output directory   : {out_dir}")

    # Process Train Pairs
    train_parquet_path = os.path.join(out_dir, "train_features.parquet")
    df_train = process_pairs_file(train_pairs_path, train_parquet_path, chunksize=args.chunksize)

    # Process Validation Pairs
    val_parquet_path = os.path.join(out_dir, "val_features.parquet")
    df_val = process_pairs_file(val_pairs_path, val_parquet_path, chunksize=args.chunksize)

    # Generate Feature Report
    report_path = os.path.join(exp_dir, "feature_extraction_report.txt")
    feature_cols = [c for c in df_train.columns if c not in ("source1_entity_id", "target_entity_id", "target_source", "label")]

    report_lines = [
        "=" * 85,
        "AMAZON ML CHALLENGE 2026 — STEP 5 FEATURE EXTRACTION REPORT",
        "=" * 85,
        f"Generated At          : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Train Pairs Processed : {len(df_train):,}",
        f"Train Positives (1)   : {(df_train['label'] == 1).sum():,} ({(df_train['label'] == 1).mean() * 100:.2f}%)",
        f"Train Negatives (0)   : {(df_train['label'] == 0).sum():,} ({(df_train['label'] == 0).mean() * 100:.2f}%)",
        f"Val Pairs Processed   : {len(df_val):,}",
        f"Val Positives (1)     : {(df_val['label'] == 1).sum():,} ({(df_val['label'] == 1).mean() * 100:.2f}%)",
        f"Val Negatives (0)     : {(df_val['label'] == 0).sum():,} ({(df_val['label'] == 0).mean() * 100:.2f}%)",
        f"Total Engineered Feats: {len(feature_cols)}",
        f"Train Parquet Path    : {train_parquet_path}",
        f"Val Parquet Path      : {val_parquet_path}",
        "=" * 85,
        "\nFEATURE DEFINITIONS & DISTRIBUTIONS (Train Set):",
        "-" * 85,
        f"{'Feature Name':<32} | {'Dtype':<8} | {'Min':>8} | {'Mean':>8} | {'Median':>8} | {'Max':>8} | {'NaNs':>5}",
        "-" * 85,
    ]

    for fn in feature_cols:
        s = df_train[fn]
        report_lines.append(f"{fn:<32} | {str(s.dtype):<8} | {s.min():>8.2f} | {s.mean():>8.2f} | {s.median():>8.2f} | {s.max():>8.2f} | {s.isnull().sum():>5}")

    report_lines.extend([
        "-" * 85,
        f"\nTotal Pipeline Execution Time: {time.time() - t_all:.2f}s",
        "=" * 85,
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nFeature extraction report saved to: {report_path}")
    print("\n" + "=" * 80)
    print("STEP 5 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Ready for Step 6: ML Model (LightGBM/GBDT Classifier).")


if __name__ == "__main__":
    main()
