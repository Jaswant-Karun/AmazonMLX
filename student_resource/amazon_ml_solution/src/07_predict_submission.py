#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Step 7 & 8: Test Candidate Retrieval, Prediction, and Submission Generation (07_predict_submission.py)

Author: Amazon ML Challenge Team
Date: September 2026

Description:
  1. Loads test_source1.tsv, test_source2.tsv, test_source3.tsv.
  2. Builds the winning inverted blocking index for test Source 1 entities.
  3. Streams test Source 2 and Source 3 to generate candidate pairs.
  4. Computes the 33 pairwise engineered features on candidate pairs.
  5. Applies the trained LightGBM model at the optimal validation decision threshold.
  6. Aggregates predictions per Source 1 entity into the official submission TSV format:
       source1_entity_id\tmatched_entity_ids
  7. Performs comprehensive verification checks (row count, format, order, nulls).

Outputs:
  - output/submission.tsv
  - experiments/submission_verification_report.txt
"""

import sys
import os
import gc
import re
import json
import time
import unicodedata
import argparse
import collections
import pandas as pd
import numpy as np
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# ==============================================================================
# 1. Normalization & Blocking Key Generation
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

    # BLOCK A: Normalized numeric address key
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
# 2. Similarity & Feature Extraction Primitives
# ==============================================================================
def fast_levenshtein(s1: str, s2: str, max_dist: int = 15) -> int:
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

        clean_name_dom = strip_domain_name(self.raw_name)
        self.norm_name = basic_normalize(clean_name_dom)
        clean_addr_pref = RE_ADDRESS_PREFIXES.sub(" ", self.raw_addr)
        self.norm_addr = basic_normalize(clean_addr_pref)

        self.name_len = len(self.norm_name)
        self.addr_len = len(self.norm_addr)
        self.is_addr_missing = 1 if self.addr_len == 0 else 0

        raw_name_toks = self.norm_name.split()
        self.name_toks = raw_name_toks
        self.name_tok_set = set(raw_name_toks)
        self.despaced_name = "".join(raw_name_toks)

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

        self.name_char_ngrams = get_char_ngrams(self.norm_name, 3)
        self.addr_char_ngrams = get_char_ngrams(self.norm_addr, 3)


def compute_pairwise_features(s1: EntityProfile, cand: EntityProfile) -> dict:
    feats = {}

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
# 3. Main Test Pipeline
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Step 7 & 8: Generate Competition Submission.")
    parser.add_argument("--test-dir", type=str, default=None, help="Directory containing test_source1/2/3.tsv.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model .joblib.")
    parser.add_argument("--threshold-path", type=str, default=None, help="Path to optimal_threshold.json.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for submission.tsv.")
    parser.add_argument("--chunksize", type=int, default=200000, help="Chunksize for reading test TSVs.")
    args = parser.parse_args()

    t_start_all = time.time()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(base_dir, "output")
    exp_dir = os.path.join(base_dir, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    # Locate test dir
    if args.test_dir and os.path.isdir(args.test_dir):
        test_dir = os.path.abspath(args.test_dir)
    else:
        test_dir = os.path.abspath(os.path.join(base_dir, "dataset", "test"))
        if not os.path.isdir(test_dir):
            test_dir = os.path.abspath(os.path.join(base_dir, "..", "dataset", "test"))

    model_path = args.model_path or os.path.join(out_dir, "entity_matching_model.joblib")
    threshold_path = args.threshold_path or os.path.join(out_dir, "optimal_threshold.json")

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 — TEST PREDICTION & SUBMISSION PIPELINE")
    print("=" * 80)
    print(f"Test directory     : {test_dir}")
    print(f"Model path         : {model_path}")
    print(f"Threshold config   : {threshold_path}")
    print(f"Output directory   : {out_dir}")

    # Load Model and Threshold
    print("\nLoading trained model and threshold config...")
    model = joblib.load(model_path)
    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_config = json.load(f)
    decision_threshold = float(threshold_config.get("optimal_threshold", 0.50))
    feature_names = threshold_config.get("feature_names", [])

    print(f"  Model loaded successfully.")
    print(f"  Optimal decision threshold: {decision_threshold:.4f}")
    print(f"  Expected feature count    : {len(feature_names)}")

    # Step 1: Load Test Source 1 Entities
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"\n[Step 1/4] Loading test Source 1 entities from {s1_path}...")
    t0 = time.time()
    test_s1_entities = {}
    test_s1_ordered_ids = []
    test_s1_to_idx = {}

    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=args.chunksize, keep_default_na=False):
        for eid, c, name, addr in zip(chunk["entity_id"], chunk["country"], chunk["business_name"], chunk["business_address"]):
            idx = len(test_s1_ordered_ids)
            test_s1_ordered_ids.append(eid)
            test_s1_to_idx[eid] = idx
            test_s1_entities[eid] = (c, name, addr)

    print(f"  Loaded {len(test_s1_ordered_ids):,} test S1 entities in {time.time() - t0:.2f}s.")

    # Step 2: Build Inverted Blocking Index for Test S1
    print("\n[Step 2/4] Building inverted blocking index for test entities...")
    t0 = time.time()
    s1_index = collections.defaultdict(list)
    s1_profiles = {}

    for eid, (c, name, addr) in test_s1_entities.items():
        s1_idx = test_s1_to_idx[eid]
        s1_profiles[s1_idx] = EntityProfile(c, name, addr)
        keys_dict = generate_all_blocking_keys(c, name, addr)
        for b_code, b_keys in keys_dict.items():
            for k in b_keys:
                s1_index[k].append(s1_idx)

    # Prune high-frequency n-grams
    MAX_NGRAM_POSTINGS = 250
    pruned = 0
    for k in list(s1_index.keys()):
        if "|bd|" in k and len(s1_index[k]) > MAX_NGRAM_POSTINGS:
            del s1_index[k]
            pruned += 1
    if pruned:
        print(f"  Pruned {pruned:,} high-frequency n-gram keys.")

    print(f"  Inverted index built with {len(s1_index):,} keys in {time.time() - t0:.2f}s.")

    # Step 3: Stream Test Source 2 and Source 3
    print("\n[Step 3/4] Streaming test Source 2 and Source 3 to score candidates...")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    predictions = collections.defaultdict(list)  # s1_idx -> list of target_id
    valid_countries = set(c.upper() for c, _, _ in test_s1_entities.values())

    def scan_test_source(source_path, label):
        t_src = time.time()
        print(f"\n  [STREAMING] {label} ({os.path.basename(source_path)})...", flush=True)
        rows_scanned = 0
        cand_pairs_scored = 0
        matches_found = 0

        def flush_pairs():
            nonlocal cand_pairs_scored, matches_found
            if not pairs_chunk:
                return
            cand_pairs_scored += len(pairs_chunk)
            feat_matrix = []
            pair_meta = []
            for s1_idx, mid, s1_prof, cand_prof, n_blocks in pairs_chunk:
                f = compute_pairwise_features(s1_prof, cand_prof)
                f["num_blocks_matched"] = n_blocks
                row_vec = [f.get(fname, 0.0) for fname in feature_names]
                feat_matrix.append(row_vec)
                pair_meta.append((s1_idx, mid))

            X_batch = np.array(feat_matrix, dtype=np.float32)
            probs = model.predict_proba(X_batch)[:, 1]

            for (s1_idx, mid), prob in zip(pair_meta, probs):
                if prob >= decision_threshold:
                    predictions[s1_idx].append((mid, float(prob)))
                    matches_found += 1
            pairs_chunk.clear()

        for chunk in pd.read_csv(source_path, sep="\t", chunksize=args.chunksize, keep_default_na=False):
            pairs_chunk = []
            for mid, c, name, addr in zip(chunk["entity_id"], chunk["country"], chunk["business_name"], chunk["business_address"]):
                rows_scanned += 1
                c_clean = c.strip().upper() if c else "UNKNOWN"
                if c_clean not in valid_countries:
                    continue

                keys_dict = generate_all_blocking_keys(c, name, addr)

                matched_s1_blocks = collections.defaultdict(list)
                for b_code, b_keys in keys_dict.items():
                    for k in b_keys:
                        if k in s1_index:
                            for s1_idx in s1_index[k]:
                                matched_s1_blocks[s1_idx].append(b_code)

                if not matched_s1_blocks:
                    continue

                cand_prof = EntityProfile(c, name, addr)
                for s1_idx, b_codes in matched_s1_blocks.items():
                    pairs_chunk.append((s1_idx, mid, s1_profiles[s1_idx], cand_prof, len(set(b_codes))))
                    if len(pairs_chunk) >= 50000:
                        flush_pairs()

            flush_pairs()

            print(
                f"    [{label}] Scanned {rows_scanned:,} rows | Cands scored: {cand_pairs_scored:,} | Matches: {matches_found:,} ({time.time() - t_src:.1f}s)",
                flush=True
            )

        print(f"  Finished {label}: {rows_scanned:,} rows scanned in {time.time() - t_src:.2f}s.")

    scan_test_source(s2_path, "Test Source 2")
    scan_test_source(s3_path, "Test Source 3")

    # Step 4: Assemble and Verify Official Submission TSV
    print("\n[Step 4/4] Assembling official submission file...")
    t0 = time.time()
    submission_rows = []
    entities_with_matches = 0
    total_predicted_matches = 0

    for s1_id in test_s1_ordered_ids:
        s1_idx = test_s1_to_idx[s1_id]
        cand_list = predictions.get(s1_idx, [])
        if cand_list:
            # Sort matches by probability descending and deduplicate by target_id
            seen_ids = set()
            unique_sorted_ids = []
            for mid, prob in sorted(cand_list, key=lambda x: x[1], reverse=True):
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    unique_sorted_ids.append(mid)

            matched_str = ",".join(unique_sorted_ids)
            entities_with_matches += 1
            total_predicted_matches += len(unique_sorted_ids)
        else:
            matched_str = ""

        submission_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": matched_str
        })

    df_submission = pd.DataFrame(submission_rows)
    submission_path = os.path.join(out_dir, "submission.tsv")
    print(f"  Writing submission TSV to: {submission_path}...")
    df_submission.to_csv(submission_path, sep="\t", index=False)
    print(f"  Submission written in {time.time() - t0:.2f}s.")

    # Also save to root directory for easy access
    root_submission_path = os.path.abspath(os.path.join(base_dir, "..", "submission.tsv"))
    try:
        df_submission.to_csv(root_submission_path, sep="\t", index=False)
        print(f"  Mirrored submission to: {root_submission_path}")
    except Exception:
        pass

    # Rigorous Verification Checks
    print("\n" + "=" * 80)
    print("SUBMISSION VERIFICATION CHECKS")
    print("=" * 80)
    total_test_rows = len(test_s1_ordered_ids)
    sub_rows = len(df_submission)
    print(f"1. Row Count Check        : {sub_rows:,} / {total_test_rows:,} (Must match exactly) -> {'PASS' if sub_rows == total_test_rows else 'FAIL'}")
    assert sub_rows == total_test_rows, "Error: Submission row count does not match test Source 1 row count!"

    col_names = list(df_submission.columns)
    print(f"2. Header Format Check     : {col_names} -> {'PASS' if col_names == ['source1_entity_id', 'matched_entity_ids'] else 'FAIL'}")
    assert col_names == ["source1_entity_id", "matched_entity_ids"], "Error: Column names must be ['source1_entity_id', 'matched_entity_ids']"

    null_count = df_submission["source1_entity_id"].isnull().sum()
    print(f"3. Null Entity ID Check    : {null_count} nulls -> {'PASS' if null_count == 0 else 'FAIL'}")
    assert null_count == 0, "Error: Null values found in source1_entity_id!"

    order_match = (df_submission["source1_entity_id"] == pd.Series(test_s1_ordered_ids)).all()
    print(f"4. Exact Row Order Check   : -> {'PASS' if order_match else 'FAIL'}")
    assert order_match, "Error: Submission rows do not match original test Source 1 order!"

    print(f"5. Entities with matches   : {entities_with_matches:,} ({(entities_with_matches/total_test_rows)*100:.2f}%)")
    print(f"6. Singleton entities (\"\")  : {total_test_rows - entities_with_matches:,} ({((total_test_rows-entities_with_matches)/total_test_rows)*100:.2f}%)")
    print(f"7. Total Predicted Matches : {total_predicted_matches:,}")

    # Generate verification report
    report_path = os.path.join(exp_dir, "submission_verification_report.txt")
    report_lines = [
        "=" * 85,
        "AMAZON ML CHALLENGE 2026 — SUBMISSION VERIFICATION REPORT",
        "=" * 85,
        f"Generated At            : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Submission File Path    : {submission_path}",
        f"Total Test S1 Entities  : {total_test_rows:,}",
        f"Total Submission Rows   : {sub_rows:,}",
        f"Entities With Matches   : {entities_with_matches:,} ({(entities_with_matches/total_test_rows)*100:.2f}%)",
        f"Singleton Entities      : {total_test_rows - entities_with_matches:,} ({((total_test_rows-entities_with_matches)/total_test_rows)*100:.2f}%)",
        f"Total Matched IDs       : {total_predicted_matches:,}",
        f"Decision Threshold Used : {decision_threshold:.4f}",
        f"Row Count Verification  : PASS",
        f"Header Verification     : PASS",
        f"Order Verification      : PASS",
        "=" * 85,
        "\nSAMPLE SUBMISSION PREDICTIONS (First 15 Rows):",
        "-" * 85,
    ]
    for idx, row in df_submission.head(15).iterrows():
        report_lines.append(f"{row['source1_entity_id']:<25} | {row['matched_entity_ids']}")

    report_lines.extend([
        "=" * 85,
        f"Total Execution Time: {time.time() - t_start_all:.2f}s",
        "=" * 85,
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Verification report written to: {report_path}")

    print("\n" + "=" * 80)
    print("STEP 7 & 8 COMPLETED SUCCESSFULLY! SUBMISSION READY FOR EVALUATION.")
    print("=" * 80)


if __name__ == "__main__":
    main()
