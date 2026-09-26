#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution
Feature Engineering Pipeline (03_feature_engineering.py)

Description:
  Generates robust, pairwise features for business entity resolution candidate pairs
  (Source 1 vs Source 2/Source 3) retrieved by the multi-key blocking strategies.

Normalization Suite:
  1. Unicode normalization (NFKD)
  2. Lowercasing
  3. Punctuation removal (Unicode-aware, preserves non-Latin scripts)
  4. Whitespace normalization
  5. Legal suffix normalization / removal (inc, llc, ltd, corp, pvt, etc.)
  6. Common prefix / honorific normalization (dr, shri, smt, m/s, formerly, etc.)
  7. URL / domain normalization (trinitycatholicchurch.com -> trinity catholic church)
  8. Leading-zero normalization for address numbers (0010 -> 10, 0330 -> 330)
  9. Multilingual Unicode tokenization

Feature Groups:
  - NAME FEATURES:
      * exact normalized name match
      * normalized name character lengths (S1, Cand, diff, ratio)
      * character similarity (3-gram Jaccard)
      * edit-distance similarity (normalized Levenshtein)
      * token overlap count
      * token Jaccard similarity
      * token containment
      * first meaningful token similarity
      * prefix similarity
      * normalized alphanumeric similarity (bridges URLs and spaced names)
  - ADDRESS FEATURES:
      * exact normalized address match
      * address character length diff & ratio
      * character similarity (3-gram Jaccard)
      * edit-distance similarity (normalized Levenshtein)
      * address token overlap count
      * address token Jaccard similarity
      * numeric token overlap count (leading-zero normalized)
      * numeric token Jaccard similarity
      * normalized house/building number match
      * locality/city/state token overlap
  - CROSS FEATURES:
      * country equality
      * name + address combined similarity
      * both name and address strong
      * name strong but address weak
      * address strong but name weak
      * missing-address indicators for S2/S3 and S1

Outputs:
  - amazon_ml_solution/experiments/03_validation_features.parquet
  - amazon_ml_solution/experiments/feature_engineering_report.txt
"""

import sys
import os
import gc
import re
import math
import time
import random
import unicodedata
import argparse
import collections
import pandas as pd
import numpy as np

# Ensure terminal handles UTF-8 characters on all platforms (including Windows cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# ==============================================================================
# 1. Normalization Constants & Precompiled Regular Expressions
# ==============================================================================

# Preserves Unicode alphanumeric characters across all world scripts (Latin, Indic, Cyrillic, etc.)
RE_PUNCT_UNICODE = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+", re.UNICODE)
RE_DESPACE = re.compile(r"[\W_]+", re.UNICODE)

# URL and Domain patterns
RE_URL_PREFIX = re.compile(r"^(?:https?://)?(?:www\.)?", re.IGNORECASE)
RE_DOMAIN_SUFFIX = re.compile(
    r"\.(?:com|org|net|in|co|info|biz|edu|gov|io|ai|tech|me|tv|us|de|fr|uk|ca|au|co\.in|org\.in|net\.in|ac\.in)(?:/.*)?$",
    re.IGNORECASE
)

# Address number normalization patterns
RE_LEADING_ZEROS = re.compile(r"^0+([1-9]\d*)$")
RE_COMPOUND_NUM = re.compile(r"\b(\d+[/_-]\d+[a-zA-Z]?)\b")
RE_ADDRESS_PREFIXES = re.compile(
    r"\b(?:door\s*no|building\s*no|flat\s*no|room\s*no|plot\s*no|house\s*no|h\.?no|kh\.?no|survey\s*no|pl\s*no|ward\s*no|unit\s*no|fl|floor|block|sector|no|flat|unit|suite|ste)\b\.?\s*[:#]?",
    re.IGNORECASE
)

# Linguistic stopwords (generic functional words)
STOPWORDS = {
    "the", "a", "an", "and", "of", "for", "in", "on", "at", "to", "by", "with",
    "from", "m", "s", "ms", "co"
}

# Legal suffixes to normalize or strip from business names
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "pvt", "private", "enterprises", "enterprise", "company", "group",
    "holdings", "holding", "services", "solutions", "associates", "associate",
    "technologies", "tech", "sarl", "sa", "gmbh", "llp", "pllc", "co", "sons",
    "industries", "industry", "intl", "international", "corp.", "inc.", "ltd."
}

# Honorifics & structural prefixes to normalize or strip from business names
STRUCTURAL_PREFIXES = {
    "dr", "shri", "smt", "m/s", "ms", "mr", "mrs", "shree", "sri",
    "prof", "messrs", "c/o", "d/o", "s/o", "w/o", "md", "ca", "adv",
    "formerly", "aka", "a/k/a", "fka", "f/k/a", "dba", "d/b/a"
}

# Generic address tokens to exclude when identifying specific localities/streets
ADDR_GENERIC = {
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive", "blvd",
    "lane", "ln", "hwy", "highway", "po", "box", "apt", "unit", "floor",
    "fl", "suite", "ste", "near", "opp", "opposite", "behind", "dist",
    "district", "null", "no", "door", "room", "flat", "plot", "bldg", "building",
    "main", "cross", "1st", "2nd", "3rd", "4th", "block", "sector", "phase"
}


# ==============================================================================
# 2. Text Normalization Functions
# ==============================================================================

def normalize_unicode(text: str) -> str:
    """Apply Unicode NFKD normalization."""
    if not text or not isinstance(text, str):
        return ""
    if not text.isascii():
        return unicodedata.normalize("NFKD", text)
    return text


def clean_text(text: str) -> str:
    """
    Combines Unicode normalization, lowercasing, punctuation stripping (preserving
    non-Latin multilingual Unicode characters), and whitespace normalization.
    """
    if not text or not isinstance(text, str):
        return ""
    text = normalize_unicode(text).lower()
    text = RE_PUNCT_UNICODE.sub(" ", text)
    return RE_WHITESPACE.sub(" ", text).strip()


def strip_domain_name(name: str) -> str:
    """
    Normalizes URLs/domains by stripping protocols and top-level domain extensions.
    Example: 'trinitycatholicchurch.com' -> 'trinitycatholicchurch'
             'www.amazon.in/shop' -> 'amazon'
    """
    if not name or not isinstance(name, str):
        return ""
    cleaned = RE_URL_PREFIX.sub("", name.strip())
    cleaned = RE_DOMAIN_SUFFIX.sub("", cleaned)
    return cleaned


def despace_alphanumeric(text: str) -> str:
    """
    Extracts all alphanumeric characters concatenated without spaces or punctuation.
    Example: 'Trinity Catholic Church' -> 'trinitycatholicchurch'
             'trinitycatholicchurch.com' (post-strip) -> 'trinitycatholicchurch'
    """
    if not text or not isinstance(text, str):
        return ""
    norm = normalize_unicode(text).lower()
    return RE_DESPACE.sub("", norm)


def normalize_numeric_token(tok: str) -> str:
    """
    Normalizes numeric strings by stripping leading zeros.
    Example: '0010' -> '10', '0330' -> '330', '0101' -> '101'
    """
    if tok.isdigit():
        return str(int(tok))
    m = RE_LEADING_ZEROS.match(tok)
    return m.group(1) if m else tok


def tokenize_multilingual(text: str) -> list:
    """
    Tokenizes text while cleanly preserving non-Latin multilingual Unicode scripts
    (Indic, Arabic, Cyrillic, Chinese, etc.).
    """
    if not text:
        return []
    clean = clean_text(text)
    return [t for t in clean.split() if t]


def extract_meaningful_name_tokens(tokens: list) -> list:
    """
    Strips leading structural prefixes (dr, shri, smt, etc.),
    trailing legal suffixes (inc, llc, ltd, etc.), and generic stopwords.
    """
    toks = list(tokens)
    while toks and (toks[0] in STRUCTURAL_PREFIXES or toks[0] in STOPWORDS):
        toks.pop(0)
    while toks and (toks[-1] in LEGAL_SUFFIXES or toks[-1] in STOPWORDS):
        toks.pop()
    meaningful = [t for t in toks if t not in LEGAL_SUFFIXES and t not in STRUCTURAL_PREFIXES and t not in STOPWORDS]
    return meaningful if meaningful else toks


def extract_address_components(addr: str) -> tuple:
    """
    Extracts normalized numeric tokens (building/zip/door numbers with leading zeros stripped)
    and locality/street alpha tokens from an address string.
    """
    if not addr or not isinstance(addr, str):
        return [], []
    
    # 1. Check compound numbers e.g. 010/2, 4-A
    compounds = RE_COMPOUND_NUM.findall(addr) if ("/" in addr or "-" in addr) else []
    comp_norm = []
    for comp in compounds:
        parts = re.split(r"[/_-]", comp)
        np = [normalize_numeric_token(p) for p in parts if p]
        if len(np) > 1:
            comp_norm.append("/".join(np))

    clean_addr = RE_ADDRESS_PREFIXES.sub(" ", addr)
    tokens = tokenize_multilingual(clean_addr)

    numerics = list(comp_norm)
    alpha_toks = []

    for t in tokens:
        if t.isdigit():
            c_num = normalize_numeric_token(t)
            if c_num and len(c_num) <= 8:
                numerics.append(c_num)
        elif t.isalpha() and len(t) >= 2 and t not in ADDR_GENERIC and t not in STOPWORDS:
            alpha_toks.append(t)

    return numerics, alpha_toks


# ==============================================================================
# 3. High-Performance Pairwise Similarity Primitives
# ==============================================================================

def fast_levenshtein(s1: str, s2: str, max_dist: int = 15) -> int:
    """
    Fast bounded Levenshtein distance in pure Python.
    Exits early if length difference exceeds max_dist or distance exceeds threshold.
    """
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
    """Extract character n-grams from a string."""
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard_similarity(set1: set, set2: set) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0.0


def containment_similarity(set1: set, set2: set) -> float:
    """Computes token containment (overlap coefficient) = |A ∩ B| / min(|A|, |B|)."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    min_size = min(len(set1), len(set2))
    return intersection / min_size if min_size else 0.0


def longest_common_prefix_len(s1: str, s2: str) -> int:
    """Computes length of the longest common prefix between two strings."""
    lim = min(len(s1), len(s2))
    idx = 0
    while idx < lim and s1[idx] == s2[idx]:
        idx += 1
    return idx


# ==============================================================================
# 4. In-Memory Entity Preprocessing Cache
# ==============================================================================

class EntityProfile:
    """Precomputed normalized representations and token sets for fast pairwise comparisons."""
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
        self.norm_name = clean_text(clean_name_dom)
        self.norm_addr = clean_text(self.raw_addr)

        self.name_len = len(self.norm_name)
        self.addr_len = len(self.norm_addr)
        self.is_addr_missing = 1 if self.addr_len == 0 else 0

        # Despaced alphanumeric string (handles URL vs spaced name matching)
        self.despaced_name = despace_alphanumeric(clean_name_dom)

        # Tokens
        all_name_toks = self.norm_name.split()
        self.name_toks = all_name_toks
        self.name_tok_set = set(all_name_toks)

        meaningful = extract_meaningful_name_tokens(all_name_toks)
        self.meaningful_name_toks = meaningful
        self.meaningful_name_tok_set = set(meaningful)
        self.first_name_tok = meaningful[0] if meaningful else (all_name_toks[0] if all_name_toks else "")

        # Address tokens and numerics
        numerics, locality = extract_address_components(self.raw_addr)
        self.numerics = numerics
        self.numeric_set = set(numerics)
        self.first_numeric = numerics[0] if numerics else ""

        self.locality_toks = locality
        self.locality_tok_set = set(locality)

        addr_toks = self.norm_addr.split()
        self.addr_toks = addr_toks
        self.addr_tok_set = set(addr_toks)

        # Character 3-grams
        self.name_char_ngrams = get_char_ngrams(self.norm_name, 3)
        self.addr_char_ngrams = get_char_ngrams(self.norm_addr, 3)


# ==============================================================================
# 5. Pairwise Feature Computation Core
# ==============================================================================

def compute_pairwise_features(s1: EntityProfile, cand: EntityProfile) -> dict:
    """
    Computes all engineered pairwise features between a Source 1 entity
    and a candidate entity (from Source 2 or Source 3).
    """
    feats = {}

    # -------------------------------------------------------------------------
    # A. NAME FEATURES
    # -------------------------------------------------------------------------
    # 1. Exact normalized name match
    feats["name_exact_match"] = 1 if (s1.norm_name and s1.norm_name == cand.norm_name) else 0

    # 2. Normalized name character lengths
    feats["name_char_len_s1"] = s1.name_len
    feats["name_char_len_cand"] = cand.name_len

    # 3. Normalized name length difference & ratio
    max_nl = max(s1.name_len, cand.name_len, 1)
    min_nl = min(s1.name_len, cand.name_len)
    feats["name_char_len_diff"] = abs(s1.name_len - cand.name_len)
    feats["name_char_len_ratio"] = min_nl / max_nl

    # 4. Character similarity (3-gram Jaccard)
    feats["name_char_sim"] = jaccard_similarity(s1.name_char_ngrams, cand.name_char_ngrams)

    # 5. Edit-distance similarity (normalized Levenshtein)
    if feats["name_exact_match"] == 1:
        feats["name_levenshtein_sim"] = 1.0
    else:
        # Bounded Levenshtein for performance: cap at max(12, int(0.5 * max_nl))
        cap = max(12, int(0.5 * max_nl))
        ld = fast_levenshtein(s1.norm_name, cand.norm_name, max_dist=cap)
        feats["name_levenshtein_sim"] = max(0.0, 1.0 - ld / max_nl)

    # 6. Token overlap count
    shared_toks = s1.meaningful_name_tok_set & cand.meaningful_name_tok_set
    feats["name_token_overlap"] = len(shared_toks)

    # 7. Token Jaccard similarity
    feats["name_token_jaccard"] = jaccard_similarity(s1.meaningful_name_tok_set, cand.meaningful_name_tok_set)

    # 8. Token containment
    feats["name_token_containment"] = containment_similarity(s1.meaningful_name_tok_set, cand.meaningful_name_tok_set)

    # 9. First meaningful token similarity
    if s1.first_name_tok and cand.first_name_tok:
        if s1.first_name_tok == cand.first_name_tok:
            feats["name_first_token_sim"] = 1.0
        else:
            max_f = max(len(s1.first_name_tok), len(cand.first_name_tok), 1)
            ld_f = fast_levenshtein(s1.first_name_tok, cand.first_name_tok, max_dist=4)
            feats["name_first_token_sim"] = max(0.0, 1.0 - ld_f / max_f)
    else:
        feats["name_first_token_sim"] = 0.0

    # 10. Prefix similarity
    lcp = longest_common_prefix_len(s1.norm_name, cand.norm_name)
    feats["name_prefix_sim"] = lcp / max_nl

    # 11. Normalized alphanumeric similarity (bridges URLs like 'trinitycatholicchurch.com' vs 'Trinity Catholic Church')
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

    # -------------------------------------------------------------------------
    # B. ADDRESS FEATURES
    # -------------------------------------------------------------------------
    # 1. Exact normalized address match
    feats["addr_exact_match"] = 1 if (s1.norm_addr and s1.norm_addr == cand.norm_addr) else 0

    # 2. Address character length difference & ratio
    max_al = max(s1.addr_len, cand.addr_len, 1)
    min_al = min(s1.addr_len, cand.addr_len)
    feats["addr_char_len_diff"] = abs(s1.addr_len - cand.addr_len)
    feats["addr_char_len_ratio"] = min_al / max_al if (s1.addr_len and cand.addr_len) else 0.0

    # 3. Character similarity (3-gram Jaccard)
    feats["addr_char_sim"] = jaccard_similarity(s1.addr_char_ngrams, cand.addr_char_ngrams)

    # 4. Edit-distance similarity (normalized Levenshtein)
    if feats["addr_exact_match"] == 1:
        feats["addr_levenshtein_sim"] = 1.0
    elif s1.addr_len == 0 or cand.addr_len == 0:
        feats["addr_levenshtein_sim"] = 0.0
    else:
        cap_a = max(15, int(0.4 * max_al))
        ld_a = fast_levenshtein(s1.norm_addr, cand.norm_addr, max_dist=cap_a)
        feats["addr_levenshtein_sim"] = max(0.0, 1.0 - ld_a / max_al)

    # 5. Address token overlap count
    shared_addr_toks = s1.addr_tok_set & cand.addr_tok_set
    feats["addr_token_overlap"] = len(shared_addr_toks)

    # 6. Address token Jaccard similarity
    feats["addr_token_jaccard"] = jaccard_similarity(s1.addr_tok_set, cand.addr_tok_set)

    # 7. Numeric token overlap count & Jaccard (leading-zero normalized: 0010 -> 10, etc.)
    shared_nums = s1.numeric_set & cand.numeric_set
    feats["addr_numeric_overlap"] = len(shared_nums)
    feats["addr_numeric_jaccard"] = jaccard_similarity(s1.numeric_set, cand.numeric_set)

    # 8. Normalized house/building number match (first numeric token in address)
    if s1.first_numeric and cand.first_numeric:
        feats["addr_house_number_match"] = 1.0 if s1.first_numeric == cand.first_numeric else 0.0
    else:
        feats["addr_house_number_match"] = -1.0  # -1 represents missing house number on one/both sides

    # 9. Locality/city/state token overlap
    feats["addr_locality_token_overlap"] = len(s1.locality_tok_set & cand.locality_tok_set)
    feats["addr_locality_jaccard"] = jaccard_similarity(s1.locality_tok_set, cand.locality_tok_set)

    # -------------------------------------------------------------------------
    # C. CROSS & INTERACTION FEATURES
    # -------------------------------------------------------------------------
    # 1. Country equality
    feats["country_match"] = 1 if (s1.country and s1.country == cand.country and s1.country != "UNKNOWN") else 0

    # 2. Name + Address combined similarity (composite harmonic & weighted mean)
    nl_sim = feats["name_levenshtein_sim"]
    al_sim = feats["addr_levenshtein_sim"]
    feats["name_addr_combined_sim"] = 0.55 * nl_sim + 0.45 * al_sim
    feats["name_addr_harmonic_sim"] = (2.0 * nl_sim * al_sim) / (nl_sim + al_sim + 1e-6)

    # 3. Both name and address strong
    feats["both_strong"] = 1 if (
        (feats["name_token_jaccard"] >= 0.5 or feats["name_levenshtein_sim"] >= 0.70) and
        (feats["addr_token_jaccard"] >= 0.4 or feats["addr_levenshtein_sim"] >= 0.60)
    ) else 0

    # 4. Name strong but address weak (e.g. branch in another street / missing address)
    feats["name_strong_addr_weak"] = 1 if (
        (feats["name_token_jaccard"] >= 0.70 or feats["name_levenshtein_sim"] >= 0.80) and
        (feats["addr_token_jaccard"] < 0.20 and feats["addr_levenshtein_sim"] < 0.35)
    ) else 0

    # 5. Address strong but name weak (e.g. business operating under different DBA / acronym)
    feats["addr_strong_name_weak"] = 1 if (
        (feats["addr_token_jaccard"] >= 0.65 or feats["addr_levenshtein_sim"] >= 0.70) and
        (feats["name_token_jaccard"] < 0.25 and feats["name_levenshtein_sim"] < 0.45)
    ) else 0

    # 6. Missing-address indicators for S2/S3 and S1
    feats["cand_addr_missing"] = cand.is_addr_missing
    feats["s1_addr_missing"] = s1.is_addr_missing

    return feats


# ==============================================================================
# 6. Blocking Keys Generation for Candidate Retrieval
# ==============================================================================

def generate_blocking_keys(profile: EntityProfile) -> dict:
    """
    Generates deterministic multi-key blocking signatures for fast candidate retrieval:
      - BLOCK 1: Country + exact normalized business name
      - BLOCK 2: Country + business name token signature (first 2 tokens)
      - BLOCK 3: Country + normalized numeric token + alpha address token
      - BLOCK 4: Country + leading name token + primary address token
      - BLOCK B: Country + despaced alphanumeric name (catches URLs vs spaced names)
    """
    c = profile.country
    keys = {}

    # BLOCK 1: Country + exact normalized name
    keys["B1"] = [f"{c}|b1|{profile.norm_name}"] if profile.norm_name else []

    # BLOCK 2: Country + first 2 meaningful name tokens
    b2 = []
    mt = profile.meaningful_name_toks
    if len(mt) >= 2:
        b2.append(f"{c}|b2|{mt[0]}_{mt[1]}")
    elif len(mt) == 1 and len(mt[0]) >= 3:
        b2.append(f"{c}|b2|{mt[0]}")
    keys["B2"] = b2

    # BLOCK 3: Country + numeric building number + alpha locality/street token
    b3 = []
    if profile.numerics and profile.locality_toks:
        b3.append(f"{c}|b3|{profile.numerics[0]}_{profile.locality_toks[0]}")
        if len(profile.locality_toks) > 1 and profile.locality_toks[-1] != profile.locality_toks[0]:
            b3.append(f"{c}|b3|{profile.numerics[0]}_{profile.locality_toks[-1]}")
    keys["B3"] = b3

    # BLOCK 4: Country + leading name token + primary address token
    b4 = []
    if mt:
        lead = mt[0]
        if profile.numerics:
            b4.append(f"{c}|b4|{lead}_{profile.numerics[0]}")
        if profile.locality_toks:
            b4.append(f"{c}|b4|{lead}_{profile.locality_toks[0]}")
    keys["B4"] = b4

    # BLOCK B: Country + despaced alphanumeric name (catches URLs vs spaced names)
    b_b = []
    if profile.despaced_name and len(profile.despaced_name) >= 4:
        b_b.append(f"{c}|bb|{profile.despaced_name}")
    keys["BB"] = b_b

    return keys


# ==============================================================================
# 7. Dataset Locator & Main Pipeline Execution
# ==============================================================================

def locate_dataset_dir(explicit_path=None):
    """Locate dataset directory containing train/ and test/ folders."""
    if explicit_path and os.path.isdir(explicit_path):
        return os.path.abspath(explicit_path)

    candidates = [
        os.path.join("student_resource", "dataset"),
        "dataset",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "student_resource", "dataset"),
    ]

    for cand in candidates:
        abs_cand = os.path.abspath(cand)
        if os.path.isdir(abs_cand) and os.path.isdir(os.path.join(abs_cand, "train")):
            return abs_cand

    raise FileNotFoundError("Could not locate challenge dataset directory.")


def main():
    parser = argparse.ArgumentParser(
        description="Feature Engineering Pipeline for Amazon ML Challenge 2026."
    )
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to dataset directory.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of deterministic Source 1 validation entities (default: 50,000).",
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=6,
        help="Maximum hard negatives per S1 entity (default: 6).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Chunk size for reading large TSVs (default: 200,000).",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default=None,
        help="Path to save the resulting features Parquet file.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Path to save the feature engineering report text file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    args = parser.parse_args()

    total_start_time = time.time()
    dataset_dir = locate_dataset_dir(args.dataset_dir)
    train_dir = os.path.join(dataset_dir, "train")

    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    # Determine default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_experiments_dir = os.path.abspath(os.path.join(script_dir, "..", "experiments"))
    os.makedirs(default_experiments_dir, exist_ok=True)

    parquet_path = args.output_parquet or os.path.join(default_experiments_dir, "03_validation_features.parquet")
    report_path = args.output_report or os.path.join(default_experiments_dir, "feature_engineering_report.txt")
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    print("=" * 85)
    print("AMAZON ML CHALLENGE 2026 — FEATURE ENGINEERING PIPELINE")
    print("=" * 85)
    print(f"Dataset directory : {dataset_dir}")
    print(f"Validation sample : {args.sample_size:,} Source 1 entities")
    print(f"Max hard negatives: {args.max_negatives} per entity")
    print(f"Output Parquet    : {parquet_path}")
    print(f"Output Report     : {report_path}")
    print(f"Random seed       : {args.seed}")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # Step 1: Deterministic S1 Sample Extraction (Matching 02_blocking_experiment.py)
    # -------------------------------------------------------------------------
    print(f"\n[Step 1/5] Extracting deterministic {args.sample_size:,} S1 validation entities...")
    t0 = time.time()
    TOTAL_S1_ROWS = 2206821
    VAL_SPLIT_SIZE = int(TOTAL_S1_ROWS * 0.10)  # 220,682
    rng = random.Random(args.seed)
    shuffled_indices = list(range(TOTAL_S1_ROWS))
    rng.shuffle(shuffled_indices)

    val_pool = shuffled_indices[:VAL_SPLIT_SIZE]
    sample_size = min(args.sample_size, len(val_pool))
    val_sample_indices = set(val_pool[:sample_size])
    del shuffled_indices
    del val_pool
    gc.collect()

    s1_profiles = {}  # s1_id -> EntityProfile
    s1_list = []      # list of s1_ids
    s1_to_idx = {}    # s1_id -> int index

    curr_row = 0
    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=args.chunksize, keep_default_na=False):
        for eid, c, name, addr in zip(chunk["entity_id"], chunk["country"], chunk["business_name"], chunk["business_address"]):
            if curr_row in val_sample_indices:
                idx = len(s1_list)
                prof = EntityProfile(c, name, addr)
                s1_profiles[eid] = prof
                s1_list.append(eid)
                s1_to_idx[eid] = idx
            curr_row += 1
        if len(s1_profiles) == sample_size:
            break

    print(f"  Extracted and profiled {len(s1_profiles):,} S1 entities in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # Step 2: Extract Ground-Truth Matches for Sample
    # -------------------------------------------------------------------------
    print(f"\n[Step 2/5] Extracting ground-truth matches from {gt_path}...")
    t0 = time.time()
    gt_matches_per_s1 = collections.defaultdict(set)  # s1_idx -> set of true matched_ids
    all_true_target_ids = set()

    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=args.chunksize, keep_default_na=False):
        matching_rows = chunk[chunk["source1_entity_id"].isin(s1_profiles)]
        for s1_id, m_str in zip(matching_rows["source1_entity_id"], matching_rows["matched_entity_ids"]):
            m_str = m_str.strip()
            if m_str:
                s1_idx = s1_to_idx[s1_id]
                for mid in m_str.split(","):
                    mid = mid.strip()
                    if mid:
                        gt_matches_per_s1[s1_idx].add(mid)
                        all_true_target_ids.add(mid)

    total_gt_pairs = sum(len(m) for m in gt_matches_per_s1.values())
    entities_with_gt = sum(1 for m in gt_matches_per_s1.values() if len(m) > 0)
    print(f"  Total ground-truth pairs in sample: {total_gt_pairs:,}")
    print(f"  Entities with >=1 match in GT     : {entities_with_gt:,} ({entities_with_gt / len(s1_profiles) * 100:.2f}%)")
    print(f"  Singleton entities (0 matches)    : {len(s1_profiles) - entities_with_gt:,}")
    print(f"  Loaded ground truth in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # Step 3: Build Inverted Index & Stream Candidates
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Building inverted blocking index and streaming candidates...")
    t0 = time.time()
    s1_index = collections.defaultdict(list)  # key -> list of s1_idx

    for s1_id, prof in s1_profiles.items():
        s1_idx = s1_to_idx[s1_id]
        keys_dict = generate_blocking_keys(prof)
        for b_code, b_keys in keys_dict.items():
            for k in b_keys:
                s1_index[k].append(s1_idx)

    print(f"  Indexed {len(s1_index):,} unique blocking keys across 5 blocking strategies.")

    positive_candidates = collections.defaultdict(dict)  # s1_idx -> mid -> (cand_profile, source_label, score)
    negative_candidates = collections.defaultdict(dict)  # s1_idx -> mid -> (cand_profile, source_label, score)
    valid_countries = set(prof.country for prof in s1_profiles.values())

    def scan_source(source_path, source_label):
        t_src = time.time()
        print(f"  Streaming {source_label} ({os.path.basename(source_path)})...", flush=True)
        rows_scanned = 0

        for chunk in pd.read_csv(source_path, sep="\t", chunksize=args.chunksize, keep_default_na=False):
            for mid, c, name, addr in zip(chunk["entity_id"], chunk["country"], chunk["business_name"], chunk["business_address"]):
                rows_scanned += 1
                c_clean = c.strip().upper() if c else "UNKNOWN"
                if c_clean not in valid_countries:
                    continue

                prof = None  # Lazily created if any key matches
                cand_keys_dict = None

                # Fast key check
                # Check Block 1 first
                norm_n = clean_text(strip_domain_name(name))
                k1 = f"{c_clean}|b1|{norm_n}" if norm_n else None

                matched_s1 = collections.defaultdict(int)
                if k1 and k1 in s1_index:
                    for s1_idx in s1_index[k1]:
                        matched_s1[s1_idx] += 3  # Higher weight for exact name match

                # If no Block 1 match or to find additional candidates, generate full keys
                prof = EntityProfile(c, name, addr)
                cand_keys_dict = generate_blocking_keys(prof)

                for b_code, b_keys in cand_keys_dict.items():
                    if b_code == "B1":
                        continue  # Already checked
                    weight = 2 if b_code in ("B4", "BB") else 1
                    for k in b_keys:
                        if k in s1_index:
                            for s1_idx in s1_index[k]:
                                matched_s1[s1_idx] += weight

                if not matched_s1:
                    continue

                for s1_idx, score in matched_s1.items():
                    is_true = (mid in gt_matches_per_s1[s1_idx])
                    if is_true:
                        positive_candidates[s1_idx][mid] = (prof, source_label, score)
                    else:
                        cur_negs = negative_candidates[s1_idx]
                        if len(cur_negs) < args.max_negatives * 2:
                            cur_negs[mid] = (prof, source_label, score)
                        elif score > 1:
                            # Replace lower-scoring candidate
                            for ex_mid, (_, _, ex_score) in list(cur_negs.items()):
                                if ex_score < score:
                                    del cur_negs[ex_mid]
                                    cur_negs[mid] = (prof, source_label, score)
                                    break

        print(f"  Finished {source_label}: {rows_scanned:,} records scanned in {time.time() - t_src:.2f}s.", flush=True)

    scan_source(s2_path, "source2")
    scan_source(s3_path, "source3")
    print(f"  Candidate search completed in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # Step 4: Assemble Balanced Candidate Pairs & Verify Ground Truth
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Assembling candidate pairs and extracting pairwise features...")
    t0 = time.time()

    pairs_to_process = []
    total_positives_found = 0

    for s1_id in s1_list:
        s1_idx = s1_to_idx[s1_id]
        s1_prof = s1_profiles[s1_id]

        # 1. Add ALL positive candidate matches (label = 1)
        for mid, (cand_prof, src_lbl, score) in positive_candidates[s1_idx].items():
            pairs_to_process.append((s1_id, s1_prof, mid, cand_prof, src_lbl, 1))
            total_positives_found += 1

        # 2. Add sampled hard negatives (label = 0)
        negs = list(negative_candidates[s1_idx].items())
        # Sort negatives by score descending (hardest negatives first)
        negs.sort(key=lambda x: x[1][2], reverse=True)
        selected_negs = negs[:args.max_negatives]

        for mid, (cand_prof, src_lbl, score) in selected_negs:
            pairs_to_process.append((s1_id, s1_prof, mid, cand_prof, src_lbl, 0))

    total_pairs = len(pairs_to_process)
    total_negatives = total_pairs - total_positives_found
    pos_pct = (total_positives_found / total_pairs * 100) if total_pairs else 0.0

    print(f"  Candidate pair assembly summary:")
    print(f"    - Total candidate pairs  : {total_pairs:,}")
    print(f"    - True positive pairs    : {total_positives_found:,} ({pos_pct:.2f}%)")
    print(f"    - Hard negative pairs    : {total_negatives:,} ({100 - pos_pct:.2f}%)")
    print(f"    - Recall of GT pairs     : {total_positives_found:,} / {total_gt_pairs:,} ({total_positives_found / total_gt_pairs * 100:.2f}%)")

    # -------------------------------------------------------------------------
    # Step 5: Compute Pairwise Features & Export Parquet
    # -------------------------------------------------------------------------
    t_feat = time.time()
    feature_rows = []
    meta_s1 = []
    meta_cand = []
    meta_src = []
    meta_labels = []

    report_pos_examples = []
    report_neg_examples = []

    for idx, (s1_id, s1_prof, cand_id, cand_prof, src_lbl, label) in enumerate(pairs_to_process):
        f_dict = compute_pairwise_features(s1_prof, cand_prof)

        meta_s1.append(s1_id)
        meta_cand.append(cand_id)
        meta_src.append(src_lbl)
        meta_labels.append(label)
        feature_rows.append(f_dict)

        # Collect diverse sample examples for the report
        if label == 1 and len(report_pos_examples) < 5:
            report_pos_examples.append({
                "s1_id": s1_id,
                "s1_name": s1_prof.raw_name,
                "s1_addr": s1_prof.raw_addr,
                "s1_country": s1_prof.country,
                "cand_id": cand_id,
                "cand_src": src_lbl,
                "cand_name": cand_prof.raw_name,
                "cand_addr": cand_prof.raw_addr,
                "cand_country": cand_prof.country,
                "name_exact": f_dict["name_exact_match"],
                "name_lev": f_dict["name_levenshtein_sim"],
                "name_jac": f_dict["name_token_jaccard"],
                "name_alpha": f_dict["name_alphanumeric_sim"],
                "addr_lev": f_dict["addr_levenshtein_sim"],
                "addr_num": f_dict["addr_numeric_overlap"],
                "both_strong": f_dict["both_strong"],
            })
        elif label == 0 and len(report_neg_examples) < 5:
            report_neg_examples.append({
                "s1_id": s1_id,
                "s1_name": s1_prof.raw_name,
                "s1_addr": s1_prof.raw_addr,
                "s1_country": s1_prof.country,
                "cand_id": cand_id,
                "cand_src": src_lbl,
                "cand_name": cand_prof.raw_name,
                "cand_addr": cand_prof.raw_addr,
                "cand_country": cand_prof.country,
                "name_exact": f_dict["name_exact_match"],
                "name_lev": f_dict["name_levenshtein_sim"],
                "name_jac": f_dict["name_token_jaccard"],
                "name_alpha": f_dict["name_alphanumeric_sim"],
                "addr_lev": f_dict["addr_levenshtein_sim"],
                "addr_num": f_dict["addr_numeric_overlap"],
                "both_strong": f_dict["both_strong"],
            })

    print(f"  Computed pairwise features for {total_pairs:,} candidate pairs in {time.time() - t_feat:.2f}s.")

    # Build DataFrame
    print("\n[Step 5/5] Building DataFrame and exporting to Parquet...")
    t_df = time.time()
    df_meta = pd.DataFrame({
        "s1_entity_id": meta_s1,
        "candidate_entity_id": meta_cand,
        "candidate_source": meta_src,
        "label": np.array(meta_labels, dtype=np.int8),
    })

    df_feats = pd.DataFrame(feature_rows)

    # Cast to memory-efficient types
    for col in df_feats.columns:
        if col.endswith("_match") or col.endswith("_missing") or col in ("both_strong", "name_strong_addr_weak", "addr_strong_name_weak", "country_match"):
            df_feats[col] = df_feats[col].astype(np.int8)
        elif col.endswith("_diff") or col.endswith("_overlap") or col.endswith("_len_s1") or col.endswith("_len_cand"):
            df_feats[col] = df_feats[col].astype(np.int16)
        else:
            df_feats[col] = df_feats[col].astype(np.float32)

    df_final = pd.concat([df_meta, df_feats], axis=1)
    del df_meta, df_feats, feature_rows
    gc.collect()

    # Save to Parquet
    df_final.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
    file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"  Parquet saved: {parquet_path} ({file_size_mb:.2f} MB) in {time.time() - t_df:.2f}s.")

    # -------------------------------------------------------------------------
    # Verification & Report Generation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("VERIFICATION CHECKS")
    print("=" * 85)

    # Check 1: Every ground truth pair in the candidate set must have label=1
    verified_pos = 0
    incorrect_labels = 0
    for s1_id, cand_id, lbl in zip(df_final["s1_entity_id"], df_final["candidate_entity_id"], df_final["label"]):
        s1_idx = s1_to_idx[s1_id]
        is_true = (cand_id in gt_matches_per_s1[s1_idx])
        if is_true:
            if lbl != 1:
                incorrect_labels += 1
            else:
                verified_pos += 1
        else:
            if lbl != 0:
                incorrect_labels += 1

    print(f"  Total candidate pairs evaluated   : {len(df_final):,}")
    print(f"  Verified positive ground-truth    : {verified_pos:,} (100.0% receiving label=1)")
    print(f"  Incorrectly labeled pairs         : {incorrect_labels}")
    assert incorrect_labels == 0, f"Error: Found {incorrect_labels} incorrectly labeled candidate pairs!"

    # Check 2: Missing values
    missing_stats = df_final.isnull().sum()
    total_missing = missing_stats.sum()
    print(f"  Total missing (NaN) values        : {total_missing}")
    assert total_missing == 0, "Error: Unexpected NaN values detected in feature dataset!"

    # Write Feature Engineering Report
    feature_names = [c for c in df_final.columns if c not in ("s1_entity_id", "candidate_entity_id", "candidate_source", "label")]

    report_lines = []
    def log(msg=""):
        report_lines.append(msg)

    log("=" * 85)
    log("AMAZON ML CHALLENGE 2026 — FEATURE ENGINEERING REPORT")
    log("=" * 85)
    log(f"Generated at                 : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Validation Sample Size       : {len(s1_profiles):,} Source 1 entities (deterministic seed=42)")
    log(f"Total Dataset Candidate Pairs: {len(df_final):,}")
    log(f"Positive Pairs (label=1)     : {total_positives_found:,} ({pos_pct:.2f}%)")
    log(f"Negative Pairs (label=0)     : {total_negatives:,} ({100 - pos_pct:.2f}%)")
    log(f"Total Ground-Truth In Sample : {total_gt_pairs:,}")
    log(f"Candidate Positive Coverage  : {total_positives_found:,} / {total_gt_pairs:,} ({total_positives_found / total_gt_pairs * 100:.2f}%)")
    log(f"Label Verification Check     : 100% of candidate ground-truth matches confirmed label=1 (0 errors)")
    log(f"Missing Value Count (NaNs)   : 0 (all features fully populated)")
    log(f"Output Parquet Path          : {parquet_path} ({file_size_mb:.2f} MB)")
    log("=" * 85)

    log("\n" + "#" * 85)
    log(f"FEATURE DEFINITIONS & SUMMARY ({len(feature_names)} features)")
    log("#" * 85)
    log(f"{'Feature Name':<32} | {'Dtype':<8} | {'Min':>8} | {'Mean':>8} | {'Median':>8} | {'Max':>8} | {'NaNs':>5}")
    log("-" * 85)
    for fn in feature_names:
        s = df_final[fn]
        log(f"{fn:<32} | {str(s.dtype):<8} | {s.min():>8.2f} | {s.mean():>8.2f} | {s.median():>8.2f} | {s.max():>8.2f} | {s.isnull().sum():>5}")

    log("\n" + "=" * 85)
    log("SAMPLE POSITIVE PAIRS (label = 1)")
    log("=" * 85)
    for idx, ex in enumerate(report_pos_examples, start=1):
        log(f"\n--- Positive Example #{idx} ---")
        log(f"  S1 Entity ID        : {ex['s1_id']} ({ex['s1_country']})")
        log(f"  S1 Business Name    : {ex['s1_name']}")
        log(f"  S1 Address          : {ex['s1_addr']}")
        log(f"  Candidate ID        : {ex['cand_id']} ({ex['cand_src']}, {ex['cand_country']})")
        log(f"  Candidate Name      : {ex['cand_name']}")
        log(f"  Candidate Address   : {ex['cand_addr']}")
        log(f"  Key Features:")
        log(f"    - name_exact_match        : {ex['name_exact']}")
        log(f"    - name_levenshtein_sim    : {ex['name_lev']:.3f}")
        log(f"    - name_token_jaccard      : {ex['name_jac']:.3f}")
        log(f"    - name_alphanumeric_sim   : {ex['name_alpha']:.3f}")
        log(f"    - addr_levenshtein_sim    : {ex['addr_lev']:.3f}")
        log(f"    - addr_numeric_overlap    : {ex['addr_num']}")
        log(f"    - both_strong             : {ex['both_strong']}")

    log("\n" + "=" * 85)
    log("SAMPLE HARD NEGATIVE PAIRS (label = 0)")
    log("=" * 85)
    for idx, ex in enumerate(report_neg_examples, start=1):
        log(f"\n--- Negative Example #{idx} ---")
        log(f"  S1 Entity ID        : {ex['s1_id']} ({ex['s1_country']})")
        log(f"  S1 Business Name    : {ex['s1_name']}")
        log(f"  S1 Address          : {ex['s1_addr']}")
        log(f"  Candidate ID        : {ex['cand_id']} ({ex['cand_src']}, {ex['cand_country']})")
        log(f"  Candidate Name      : {ex['cand_name']}")
        log(f"  Candidate Address   : {ex['cand_addr']}")
        log(f"  Key Features:")
        log(f"    - name_exact_match        : {ex['name_exact']}")
        log(f"    - name_levenshtein_sim    : {ex['name_lev']:.3f}")
        log(f"    - name_token_jaccard      : {ex['name_jac']:.3f}")
        log(f"    - name_alphanumeric_sim   : {ex['name_alpha']:.3f}")
        log(f"    - addr_levenshtein_sim    : {ex['addr_lev']:.3f}")
        log(f"    - addr_numeric_overlap    : {ex['addr_num']}")
        log(f"    - both_strong             : {ex['both_strong']}")

    log("\n" + "=" * 85)
    log(f"Total Pipeline Execution Time: {time.time() - total_start_time:.2f}s")
    log("=" * 85)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nFeature engineering report written to: {report_path}")

    # Also save a copy under root experiments folder if different
    root_experiments_dir = os.path.abspath(os.path.join(dataset_dir, "..", "..", "amazon_ml_solution", "experiments"))
    if os.path.isdir(os.path.dirname(root_experiments_dir)) and root_experiments_dir != default_experiments_dir:
        try:
            os.makedirs(root_experiments_dir, exist_ok=True)
            root_report = os.path.join(root_experiments_dir, "feature_engineering_report.txt")
            with open(root_report, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
        except Exception:
            pass


if __name__ == "__main__":
    main()
