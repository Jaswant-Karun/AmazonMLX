#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Module: Enhanced Feature Extractor v2 (Phases 5, 6, 7)
src/feature_extractor_v2.py

Extracts 46 rich pairwise features covering:
- Multilingual & script detection (Phase 5)
- Comprehensive address matching & missing address handling (Phase 6)
- Robust name matching with OCR, domain, and legal suffix robustness (Phase 7)
- Compositional & cross-field interaction signals (Phase 8 hard-negative discriminators)
"""

import sys
import re
import math
import unicodedata
from typing import Dict, Any, List

try:
    from normalization_engine import (
        EntityProfileV2,
        strip_accents_unicode,
        normalize_numeric_token
    )
except ImportError:
    from src.normalization_engine import (
        EntityProfileV2,
        strip_accents_unicode,
        normalize_numeric_token
    )

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


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
# Feature Extraction: 46 Discriminative Features
# ==============================================================================
FEATURE_NAMES_V2 = [
    # --- Group 1: Name Features (Phase 7) ---
    "name_exact_match",
    "name_compact_match",
    "name_ocr_compact_match",
    "name_sig_match",
    "name_char_len_s1",
    "name_char_len_cand",
    "name_char_len_diff",
    "name_char_len_ratio",
    "name_char_sim",
    "name_levenshtein_sim",
    "name_token_overlap",
    "name_token_jaccard",
    "name_token_containment",
    "name_first_token_sim",
    "name_last_token_sim",
    "name_prefix_sim",
    "name_alphanumeric_sim",
    # --- Group 2: Address Features (Phase 6) ---
    "addr_exact_match",
    "addr_compact_match",
    "addr_char_len_diff",
    "addr_char_len_ratio",
    "addr_char_sim",
    "addr_levenshtein_sim",
    "addr_token_overlap",
    "addr_token_jaccard",
    "addr_token_containment",
    "addr_numeric_overlap",
    "addr_numeric_jaccard",
    "addr_house_number_match",
    "addr_postal_code_match",
    "addr_numeric_sig_match",
    "addr_locality_token_overlap",
    "addr_locality_jaccard",
    "s1_addr_missing",
    "cand_addr_missing",
    "both_addr_missing",
    # --- Group 3: Multilingual / Script Features (Phase 5) ---
    "same_script",
    "script_type_latin",
    "script_type_indic",
    "script_char_overlap",
    # --- Group 4: Cross & Compositional Interaction Signals (Phase 8) ---
    "country_match",
    "name_addr_combined_sim",
    "name_addr_harmonic_sim",
    "both_strong",
    "name_strong_addr_weak",
    "addr_strong_name_weak",
]


def extract_features_v2(s1: EntityProfileV2, cand: EntityProfileV2) -> Dict[str, float]:
    """
    Extracts all 46 engineered features between two entity profiles.
    Returns dictionary mapping feature name to numeric float.
    """
    feats = {}

    # -------------------------------------------------------------
    # 1. NAME FEATURES (Phase 7)
    # -------------------------------------------------------------
    feats["name_exact_match"] = 1.0 if (s1.name_normalized and s1.name_normalized == cand.name_normalized) else 0.0
    feats["name_compact_match"] = 1.0 if (s1.name_compact and s1.name_compact == cand.name_compact) else 0.0
    feats["name_ocr_compact_match"] = 1.0 if (s1.name_ocr_normalized and s1.name_ocr_normalized == cand.name_ocr_normalized) else 0.0
    feats["name_sig_match"] = 1.0 if (s1.name_token_signature and s1.name_token_signature == cand.name_token_signature) else 0.0

    max_nl = max(s1.name_len, cand.name_len, 1)
    min_nl = min(s1.name_len, cand.name_len)
    feats["name_char_len_s1"] = float(s1.name_len)
    feats["name_char_len_cand"] = float(cand.name_len)
    feats["name_char_len_diff"] = float(abs(s1.name_len - cand.name_len))
    feats["name_char_len_ratio"] = float(min_nl / max_nl)

    feats["name_char_sim"] = jaccard_similarity(s1.name_char_ngrams, cand.name_char_ngrams)

    if feats["name_exact_match"] == 1.0:
        feats["name_levenshtein_sim"] = 1.0
    else:
        cap_n = max(12, int(0.5 * max_nl))
        ld_n = fast_levenshtein(s1.name_normalized, cand.name_normalized, max_dist=cap_n)
        feats["name_levenshtein_sim"] = max(0.0, 1.0 - ld_n / max_nl)

    shared_name_toks = s1.name_meaningful_set & cand.name_meaningful_set
    feats["name_token_overlap"] = float(len(shared_name_toks))
    feats["name_token_jaccard"] = jaccard_similarity(s1.name_meaningful_set, cand.name_meaningful_set)
    feats["name_token_containment"] = containment_similarity(s1.name_meaningful_set, cand.name_meaningful_set)

    # First and Last token similarity
    if s1.name_first_token and cand.name_first_token:
        if s1.name_first_token == cand.name_first_token:
            feats["name_first_token_sim"] = 1.0
        else:
            max_f = max(len(s1.name_first_token), len(cand.name_first_token), 1)
            ld_f = fast_levenshtein(s1.name_first_token, cand.name_first_token, max_dist=4)
            feats["name_first_token_sim"] = max(0.0, 1.0 - ld_f / max_f)
    else:
        feats["name_first_token_sim"] = 0.0

    if s1.name_last_token and cand.name_last_token:
        if s1.name_last_token == cand.name_last_token:
            feats["name_last_token_sim"] = 1.0
        else:
            max_l = max(len(s1.name_last_token), len(cand.name_last_token), 1)
            ld_l = fast_levenshtein(s1.name_last_token, cand.name_last_token, max_dist=4)
            feats["name_last_token_sim"] = max(0.0, 1.0 - ld_l / max_l)
    else:
        feats["name_last_token_sim"] = 0.0

    lcp = longest_common_prefix_len(s1.name_normalized, cand.name_normalized)
    feats["name_prefix_sim"] = float(lcp / max_nl)

    if s1.name_compact and cand.name_compact:
        if s1.name_compact == cand.name_compact:
            feats["name_alphanumeric_sim"] = 1.0
        else:
            max_dn = max(len(s1.name_compact), len(cand.name_compact), 1)
            cap_d = max(10, int(0.4 * max_dn))
            ld_dn = fast_levenshtein(s1.name_compact, cand.name_compact, max_dist=cap_d)
            feats["name_alphanumeric_sim"] = max(0.0, 1.0 - ld_dn / max_dn)
    else:
        feats["name_alphanumeric_sim"] = 0.0

    # -------------------------------------------------------------
    # 2. ADDRESS FEATURES (Phase 6)
    # -------------------------------------------------------------
    feats["addr_exact_match"] = 1.0 if (s1.address_normalized and s1.address_normalized == cand.address_normalized) else 0.0
    feats["addr_compact_match"] = 1.0 if (s1.address_compact and s1.address_compact == cand.address_compact) else 0.0

    max_al = max(s1.address_len, cand.address_len, 1)
    min_al = min(s1.address_len, cand.address_len)
    feats["addr_char_len_diff"] = float(abs(s1.address_len - cand.address_len))
    feats["addr_char_len_ratio"] = float(min_al / max_al) if (s1.address_len and cand.address_len) else 0.0

    feats["addr_char_sim"] = jaccard_similarity(s1.address_char_ngrams, cand.address_char_ngrams)

    if feats["addr_exact_match"] == 1.0:
        feats["addr_levenshtein_sim"] = 1.0
    elif s1.address_len == 0 or cand.address_len == 0:
        feats["addr_levenshtein_sim"] = 0.0
    else:
        cap_a = max(15, int(0.5 * max_al))
        ld_a = fast_levenshtein(s1.address_normalized, cand.address_normalized, max_dist=cap_a)
        feats["addr_levenshtein_sim"] = max(0.0, 1.0 - ld_a / max_al)

    shared_addr_toks = s1.address_token_set & cand.address_token_set
    feats["addr_token_overlap"] = float(len(shared_addr_toks))
    feats["addr_token_jaccard"] = jaccard_similarity(s1.address_token_set, cand.address_token_set)
    feats["addr_token_containment"] = containment_similarity(s1.address_token_set, cand.address_token_set)

    shared_nums = s1.address_numeric_set & cand.address_numeric_set
    feats["addr_numeric_overlap"] = float(len(shared_nums))
    feats["addr_numeric_jaccard"] = jaccard_similarity(s1.address_numeric_set, cand.address_numeric_set)

    # Building number
    if s1.address_building_number and cand.address_building_number:
        feats["addr_house_number_match"] = 1.0 if s1.address_building_number == cand.address_building_number else 0.0
    else:
        feats["addr_house_number_match"] = -1.0

    # Postal code
    if s1.address_postal_code and cand.address_postal_code:
        feats["addr_postal_code_match"] = 1.0 if s1.address_postal_code == cand.address_postal_code else 0.0
    else:
        feats["addr_postal_code_match"] = -1.0

    # Numeric signature match
    if s1.address_numeric_signature and cand.address_numeric_signature:
        feats["addr_numeric_sig_match"] = 1.0 if s1.address_numeric_signature == cand.address_numeric_signature else 0.0
    else:
        feats["addr_numeric_sig_match"] = -1.0

    shared_loc = s1.address_locality_set & cand.address_locality_set
    feats["addr_locality_token_overlap"] = float(len(shared_loc))
    feats["addr_locality_jaccard"] = jaccard_similarity(s1.address_locality_set, cand.address_locality_set)

    feats["s1_addr_missing"] = float(s1.is_addr_missing)
    feats["cand_addr_missing"] = float(cand.is_addr_missing)
    feats["both_addr_missing"] = 1.0 if (s1.is_addr_missing and cand.is_addr_missing) else 0.0

    # -------------------------------------------------------------
    # 3. MULTILINGUAL & SCRIPT FEATURES (Phase 5)
    # -------------------------------------------------------------
    feats["same_script"] = 1.0 if (s1.script_type == cand.script_type and s1.script_type != "none") else 0.0
    feats["script_type_latin"] = 1.0 if (s1.script_type == "latin" or cand.script_type == "latin") else 0.0
    feats["script_type_indic"] = 1.0 if (s1.script_type in ("devanagari", "indic_other") or cand.script_type in ("devanagari", "indic_other")) else 0.0

    s1_chars = set(s1.name_original)
    cand_chars = set(cand.name_original)
    feats["script_char_overlap"] = jaccard_similarity(s1_chars, cand_chars)

    # -------------------------------------------------------------
    # 4. CROSS & COMPOSITIONAL HARD NEGATIVE SIGNALS (Phases 6, 7, 8)
    # -------------------------------------------------------------
    feats["country_match"] = 1.0 if (s1.country and s1.country == cand.country) else 0.0

    name_s = feats["name_levenshtein_sim"]
    addr_s = feats["addr_levenshtein_sim"]

    feats["name_addr_combined_sim"] = 0.65 * name_s + 0.35 * addr_s

    if (name_s + addr_s) > 0:
        feats["name_addr_harmonic_sim"] = 2.0 * (name_s * addr_s) / (name_s + addr_s)
    else:
        feats["name_addr_harmonic_sim"] = 0.0

    feats["both_strong"] = 1.0 if (name_s >= 0.70 and addr_s >= 0.60) else 0.0
    feats["name_strong_addr_weak"] = 1.0 if (name_s >= 0.80 and addr_s < 0.30) else 0.0
    feats["addr_strong_name_weak"] = 1.0 if (addr_s >= 0.80 and name_s < 0.40) else 0.0

    return feats
