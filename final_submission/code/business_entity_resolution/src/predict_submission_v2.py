#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Module: High-Performance Candidate Retrieval & ML Inference Pipeline v2 (Phases 11-15)
src/predict_submission_v2.py

Generates:
  - output/candidate_pairs_v2.tsv
  - output/matching_results_v2.tsv

Key Specifications:
  - Memory: Peak RAM <= 1.5 GB via 4 partitions of test Source 1
  - Speed: Vectorized batch inference with direct feature compilation (~4 mins total)
  - Multilingual & Country: Native support for France, US, India, and arbitrary countries
  - ML Model: Entity Matching Model v2 (Model D - 46 features)
  - Decision Logic: Winning Dynamic Agreement Rule (Macro F0.5 = 0.95215, 98.38% precision)
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
# 1. High-Speed Regex & Normalization Primitives
# ==============================================================================
RE_PUNCT = re.compile(r"[^\w\s]")
RE_DOMAIN_SUFFIX = re.compile(
    r"\.(?:com|org|net|in|co|info|biz|edu|gov|io|ai|tech|me|tv|us|de|fr|uk|ca|au|eu)(?:/.*)?$",
    re.IGNORECASE
)
RE_WWW_PREFIX = re.compile(r"^(?:https?://)?(?:www\.)?", re.IGNORECASE)
RE_FORMER_NAME = re.compile(
    r"(?:formerly|aka|a/k/a|fka|f/k/a|dba|d/b/a|trading\s*as|t/a|nom\s*commercial)[:\s]+(.*)",
    re.IGNORECASE
)
RE_ADDRESS_PREFIXES = re.compile(
    r"\b(?:door\s*no|building\s*no|flat\s*no|room\s*no|plot\s*no|house\s*no|h\.?no|kh\.?no|"
    r"survey\s*no|pl\s*no|ward\s*no|unit\s*no|fl|floor|block|sector|no|flat|unit|suite|ste|"
    r"batiment|bat|etage|porte|numero|num|n°)\b\.?\s*[:#]?",
    re.IGNORECASE
)
RE_LEADING_ZEROS = re.compile(r"^0+([1-9]\d*)$")
RE_COMPOUND_NUM = re.compile(r"\b(\d+[/_-]\d+)\b")

STOPWORDS = {
    "the", "a", "an", "and", "of", "for", "in", "on", "at", "to", "m", "s", "ms",
    "co", "by", "with", "from", "as", "is", "or", "into", "near", "opp", "behind",
    "de", "des", "du", "la", "le", "les", "et", "en", "au", "aux", "sur", "dans", "par", "pour"
}

STRUCTURAL_PREFIXES = {
    "dr", "doctor", "shri", "smt", "m/s", "ms", "mr", "mrs", "shree", "sri",
    "prof", "professor", "messrs", "c/o", "d/o", "s/o", "w/o", "md", "ca", "adv",
    "me", "maitre", "docteur", "monsieur", "madame"
}

STRUCTURAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "enterprises", "enterprise", "company", "group",
    "holdings", "services", "solutions", "associates", "technologies",
    "tech", "sarl", "sa", "sasu", "eurl", "sci", "snc", "gmbh", "llp", "pllc", "co", "sons"
}

ADDR_GENERIC = {
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive", "blvd",
    "lane", "ln", "hwy", "highway", "po", "box", "apt", "unit", "floor",
    "fl", "suite", "ste", "near", "opp", "opposite", "behind", "dist",
    "district", "null", "no", "door", "room", "flat", "plot", "bldg", "building",
    "r", "rue", "bd", "bvd", "boulevard", "imp", "impasse", "all", "allee",
    "chem", "chemin", "rte", "route", "crs", "cours", "res", "residence", "cedex"
}

ADDRESS_ABBREVIATIONS = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "hwy": "highway",
    "r": "rue", "imp": "impasse", "all": "allee", "chem": "chemin", "rte": "route",
    "crs": "cours", "res": "residence", "bat": "batiment", "etg": "etage",
    "cdx": "cedex"
}

OCR_DIGIT_MAP = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '5': 's', '8': 'b', '@': 'a', '$': 's'})


def basic_normalize(text: str) -> str:
    if not text:
        return ""
    if not text.isascii():
        nfkd = unicodedata.normalize("NFKD", text)
        res = [c for c in nfkd if not (0x0300 <= ord(c) <= 0x036F)]
        text = unicodedata.normalize("NFKC", "".join(res))
    text = RE_PUNCT.sub(" ", text).lower()
    return " ".join(text.split())


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


def detect_script(text: str) -> str:
    if not text:
        return "none"
    has_latin = False
    for char in text:
        cp = ord(char)
        if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            has_latin = True
        elif 0x0900 <= cp <= 0x097F:
            return "devanagari"
        elif 0x0980 <= cp <= 0x0D7F:
            return "indic_other"
    return "latin" if has_latin else "other"


# Fast blocking key extraction
def generate_blocking_keys(c: str, name: str, addr: str) -> dict:
    if not isinstance(c, str): c = ""
    if not isinstance(name, str): name = ""
    if not isinstance(addr, str): addr = ""

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

    clean_name = strip_domain_name(name)
    if any(kw in clean_name.lower() for kw in ("former", "aka", "dba", "t/a", "commercial")):
        m_former = RE_FORMER_NAME.search(clean_name)
        if m_former and len(m_former.group(1).strip()) > 3:
            clean_name = m_former.group(1).strip()

    norm_name = basic_normalize(clean_name)
    raw_name_toks = norm_name.split()
    name_toks = [t for t in raw_name_toks if t not in STOPWORDS and t not in STRUCTURAL_SUFFIXES and len(t) >= 2]

    # B1: Exact normalized name
    keys["B1"] = [f"{c}|b1|{norm_name}"] if norm_name else []

    # B2: First 2 tokens
    b2 = []
    if len(name_toks) >= 2:
        b2.append(f"{c}|b2|{name_toks[0]}_{name_toks[1]}")
    elif len(name_toks) == 1 and len(name_toks[0]) >= 3:
        b2.append(f"{c}|b2|{name_toks[0]}")
    keys["B2"] = b2

    # B3: Address numerics + alpha
    b3 = []
    if numerics and alpha_toks:
        b3.append(f"{c}|b3|{numerics[0]}_{alpha_toks[0]}")
        if len(alpha_toks) > 1 and alpha_toks[-1] != alpha_toks[0]:
            b3.append(f"{c}|b3|{numerics[0]}_{alpha_toks[-1]}")
    keys["B3"] = b3

    # B4: Name token + address token
    b4 = []
    if name_toks:
        lead = name_toks[0]
        if numerics:
            b4.append(f"{c}|b4|{lead}_{numerics[0]}")
        if alpha_toks:
            b4.append(f"{c}|b4|{lead}_{alpha_toks[0]}")
    keys["B4"] = b4

    # BA: Compact name / OCR match
    ba = []
    compact = "".join(c_ch for c_ch in norm_name if c_ch.isalnum())
    if len(compact) >= 5:
        ba.append(f"{c}|ba|{compact}")
        ocr_comp = compact.translate(OCR_DIGIT_MAP)
        if ocr_comp != compact and len(ocr_comp) >= 5:
            ba.append(f"{c}|ba|{ocr_comp}")
    keys["BA"] = ba

    # BC: Building + Postal
    bc = []
    if bldgs and postals:
        bc.append(f"{c}|bc|{bldgs[0]}_{postals[0]}")
    keys["BC"] = bc

    return keys


# ==============================================================================
# 2. Fast Profile & Direct Feature Computation
# ==============================================================================
def fast_levenshtein(s1: str, s2: str, max_dist: int = 15) -> int:
    if s1 == s2: return 0
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > max_dist: return max_dist + 1
    if len1 > len2: s1, s2, len1, len2 = s2, s1, len2, len1
    if not len1: return min(len2, max_dist + 1)
    prev = list(range(len1 + 1))
    for j, c2 in enumerate(s2):
        curr = [j + 1] * (len1 + 1)
        min_row = curr[0]
        for i, c1 in enumerate(s1):
            cost = 0 if c1 == c2 else 1
            val = min(prev[i + 1] + 1, curr[i] + 1, prev[i] + cost)
            curr[i + 1] = val
            if val < min_row: min_row = val
        if min_row > max_dist: return max_dist + 1
        prev = curr
    return prev[len1]


def jaccard(s1: set, s2: set) -> float:
    if not s1 or not s2: return 0.0
    u = len(s1 | s2)
    return len(s1 & s2) / u if u else 0.0


def containment(s1: set, s2: set) -> float:
    if not s1 or not s2: return 0.0
    m = min(len(s1), len(s2))
    return len(s1 & s2) / m if m else 0.0


def lcp_len(s1: str, s2: str) -> int:
    lim = min(len(s1), len(s2))
    i = 0
    while i < lim and s1[i] == s2[i]: i += 1
    return i


class FastProfile:
    __slots__ = (
        "country", "raw_name", "raw_addr", "script",
        "norm_name", "compact_name", "ocr_name", "name_len",
        "name_toks", "name_tok_set", "meaningful_toks", "meaningful_set",
        "sig_name", "first_tok", "last_tok", "name_ngrams",
        "norm_addr", "compact_addr", "addr_len", "is_addr_missing",
        "addr_tok_set", "numerics", "numeric_set", "bldg", "postal", "num_sig",
        "loc_set", "addr_ngrams"
    )

    def __init__(self, country: str, business_name: str, business_address: str):
        self.country = country.strip().upper() if country else "UNKNOWN"
        self.raw_name = business_name if isinstance(business_name, str) else ""
        self.raw_addr = business_address if isinstance(business_address, str) else ""
        self.script = detect_script(self.raw_name)

        # Name
        clean_n = strip_domain_name(self.raw_name)
        if any(kw in clean_n.lower() for kw in ("former", "aka", "dba", "t/a", "commercial")):
            m_f = RE_FORMER_NAME.search(clean_n)
            if m_f and len(m_f.group(1).strip()) > 3:
                clean_n = m_f.group(1).strip()

        norm_n = basic_normalize(clean_n)
        self.norm_name = norm_n
        self.name_len = len(norm_n)
        self.compact_name = "".join(c for c in norm_n if c.isalnum())
        self.ocr_name = self.compact_name.translate(OCR_DIGIT_MAP)

        raw_toks = norm_n.split()
        self.name_toks = raw_toks
        self.name_tok_set = set(raw_toks)

        filt = [t for t in raw_toks if t not in STRUCTURAL_PREFIXES and t not in STRUCTURAL_SUFFIXES and t not in STOPWORDS and len(t) >= 2]
        self.meaningful_toks = filt
        self.meaningful_set = set(filt)
        self.sig_name = "_".join(sorted(self.meaningful_set))
        self.first_tok = filt[0] if filt else (raw_toks[0] if raw_toks else "")
        self.last_tok = filt[-1] if filt else (raw_toks[-1] if raw_toks else "")

        if len(norm_n) < 3:
            self.name_ngrams = {norm_n} if norm_n else set()
        else:
            self.name_ngrams = {norm_n[i:i + 3] for i in range(len(norm_n) - 2)}

        # Address
        clean_a = RE_ADDRESS_PREFIXES.sub(" ", self.raw_addr)
        norm_a = basic_normalize(clean_a)
        tokens_a = []
        for t in norm_a.split():
            if t in ADDRESS_ABBREVIATIONS:
                tokens_a.append(ADDRESS_ABBREVIATIONS[t])
            elif t.isdigit():
                tokens_a.append(normalize_numeric_token(t))
            else:
                tokens_a.append(t)
        norm_a_exp = " ".join(tokens_a)

        self.norm_addr = norm_a_exp
        self.addr_len = len(norm_a_exp)
        self.is_addr_missing = 1.0 if self.addr_len == 0 else 0.0
        self.compact_addr = "".join(c for c in norm_a_exp if c.isalnum())
        self.addr_tok_set = set(tokens_a)

        compounds = RE_COMPOUND_NUM.findall(self.raw_addr) if ("/" in self.raw_addr or "-" in self.raw_addr) else []
        comp_norm = []
        for comp in compounds:
            parts = re.split(r"[/_-]", comp)
            np = [normalize_numeric_token(p) for p in parts if p]
            if len(np) > 1: comp_norm.append("_".join(np))

        numerics = list(comp_norm)
        locality = []
        for t in tokens_a:
            if t.isdigit():
                c_num = normalize_numeric_token(t)
                if c_num and len(c_num) <= 8: numerics.append(c_num)
            elif t.isalpha() and len(t) >= 3 and t not in ADDR_GENERIC and t not in STOPWORDS:
                locality.append(t)

        self.numerics = numerics
        self.numeric_set = set(numerics)
        self.num_sig = "_".join(numerics[:4]) if numerics else ""
        postals = [n for n in numerics if len(n) in (5, 6) and "_" not in n]
        bldgs = [n for n in numerics if len(n) <= 4 and "_" not in n]
        self.postal = postals[0] if postals else ""
        self.bldg = bldgs[0] if bldgs else ""
        self.loc_set = set(locality)

        if len(norm_a_exp) < 3:
            self.addr_ngrams = {norm_a_exp} if norm_a_exp else set()
        else:
            self.addr_ngrams = {norm_a_exp[i:i + 3] for i in range(len(norm_a_exp) - 2)}


# Pre-allocated feature computation directly into 46 float values matching feature_names
def compute_direct_46_features(s1: FastProfile, cand: FastProfile) -> list:
    # 1. Names
    name_exact = 1.0 if (s1.norm_name and s1.norm_name == cand.norm_name) else 0.0
    s1_nl = s1.name_len
    c_nl = cand.name_len
    max_nl = max(s1_nl, c_nl, 1)
    min_nl = min(s1_nl, c_nl)
    name_diff = float(abs(s1_nl - c_nl))
    name_ratio = float(min_nl / max_nl)
    name_char_s = jaccard(s1.name_ngrams, cand.name_ngrams)

    if name_exact == 1.0:
        name_lev_s = 1.0
    else:
        cap_n = max(12, int(0.5 * max_nl))
        ld_n = fast_levenshtein(s1.norm_name, cand.norm_name, max_dist=cap_n)
        name_lev_s = max(0.0, 1.0 - ld_n / max_nl)

    sh_toks = s1.meaningful_set & cand.meaningful_set
    n_tok_over = float(len(sh_toks))
    n_tok_jacc = jaccard(s1.meaningful_set, cand.meaningful_set)

    if s1.first_tok and cand.first_tok:
        if s1.first_tok == cand.first_tok:
            n_first_s = 1.0
        else:
            max_f = max(len(s1.first_tok), len(cand.first_tok), 1)
            n_first_s = max(0.0, 1.0 - fast_levenshtein(s1.first_tok, cand.first_tok, max_dist=4) / max_f)
    else:
        n_first_s = 0.0

    n_pfx_s = float(lcp_len(s1.norm_name, cand.norm_name) / max_nl)

    if s1.compact_name and cand.compact_name:
        if s1.compact_name == cand.compact_name:
            n_alphan_s = 1.0
        else:
            max_dn = max(len(s1.compact_name), len(cand.compact_name), 1)
            cap_d = max(10, int(0.4 * max_dn))
            n_alphan_s = max(0.0, 1.0 - fast_levenshtein(s1.compact_name, cand.compact_name, max_dist=cap_d) / max_dn)
    else:
        n_alphan_s = 0.0

    # 2. Addresses
    addr_exact = 1.0 if (s1.norm_addr and s1.norm_addr == cand.norm_addr) else 0.0
    s1_al = s1.addr_len
    c_al = cand.addr_len
    max_al = max(s1_al, c_al, 1)
    min_al = min(s1_al, c_al)
    addr_diff = float(abs(s1_al - c_al))
    addr_ratio = float(min_al / max_al) if (s1_al and c_al) else 0.0
    addr_char_s = jaccard(s1.addr_ngrams, cand.addr_ngrams)

    if addr_exact == 1.0:
        addr_lev_s = 1.0
    elif s1_al == 0 or c_al == 0:
        addr_lev_s = 0.0
    else:
        cap_a = max(15, int(0.5 * max_al))
        addr_lev_s = max(0.0, 1.0 - fast_levenshtein(s1.norm_addr, cand.norm_addr, max_dist=cap_a) / max_al)

    sh_addr = s1.addr_tok_set & cand.addr_tok_set
    a_tok_over = float(len(sh_addr))
    a_tok_jacc = jaccard(s1.addr_tok_set, cand.addr_tok_set)

    sh_num = s1.numeric_set & cand.numeric_set
    a_num_over = float(len(sh_num))
    a_num_jacc = jaccard(s1.numeric_set, cand.numeric_set)

    if s1.bldg and cand.bldg:
        a_house_m = 1.0 if s1.bldg == cand.bldg else 0.0
    else:
        a_house_m = -1.0

    sh_loc = s1.loc_set & cand.loc_set
    a_loc_over = float(len(sh_loc))
    a_loc_jacc = jaccard(s1.loc_set, cand.loc_set)

    # 3. Cross & Interactions
    c_match = 1.0 if (s1.country and s1.country == cand.country) else 0.0
    comb_s = 0.65 * name_lev_s + 0.35 * addr_lev_s
    harm_s = (2.0 * name_lev_s * addr_lev_s) / (name_lev_s + addr_lev_s) if (name_lev_s + addr_lev_s) > 0 else 0.0

    both_str = 1.0 if (name_lev_s >= 0.70 and addr_lev_s >= 0.60) else 0.0
    name_str_addr_wk = 1.0 if (name_lev_s >= 0.80 and addr_lev_s < 0.30) else 0.0
    addr_str_name_wk = 1.0 if (addr_lev_s >= 0.80 and name_lev_s < 0.40) else 0.0

    # 4. Phase 7 Name v2
    name_comp_m = 1.0 if (s1.compact_name and s1.compact_name == cand.compact_name) else 0.0
    name_ocr_m = 1.0 if (s1.ocr_name and s1.ocr_name == cand.ocr_name) else 0.0
    name_sig_m = 1.0 if (s1.sig_name and s1.sig_name == cand.sig_name) else 0.0
    n_tok_cont = containment(s1.meaningful_set, cand.meaningful_set)

    if s1.last_tok and cand.last_tok:
        if s1.last_tok == cand.last_tok:
            n_last_s = 1.0
        else:
            max_l = max(len(s1.last_tok), len(cand.last_tok), 1)
            n_last_s = max(0.0, 1.0 - fast_levenshtein(s1.last_tok, cand.last_tok, max_dist=4) / max_l)
    else:
        n_last_s = 0.0

    # 5. Phase 6 Addr v2
    addr_comp_m = 1.0 if (s1.compact_addr and s1.compact_addr == cand.compact_addr) else 0.0
    a_tok_cont = containment(s1.addr_tok_set, cand.addr_tok_set)
    a_post_m = 1.0 if (s1.postal and s1.postal == cand.postal) else (-1.0 if not (s1.postal and cand.postal) else 0.0)
    a_num_sig_m = 1.0 if (s1.num_sig and s1.num_sig == cand.num_sig) else (-1.0 if not (s1.num_sig and cand.num_sig) else 0.0)
    both_addr_miss = 1.0 if (s1.is_addr_missing and cand.is_addr_missing) else 0.0

    # 6. Phase 5 Multilingual
    same_sc = 1.0 if (s1.script == cand.script and s1.script != "none") else 0.0
    sc_lat = 1.0 if (s1.script == "latin" or cand.script == "latin") else 0.0
    sc_ind = 1.0 if (s1.script in ("devanagari", "indic_other") or cand.script in ("devanagari", "indic_other")) else 0.0
    sc_char_over = jaccard(set(s1.raw_name), set(cand.raw_name))

    # Exact order matching feature_names in optimal_threshold_v2.json
    return [
        name_exact, float(s1_nl), float(c_nl), name_diff, name_ratio, name_char_s,
        name_lev_s, n_tok_over, n_tok_jacc, n_first_s, n_pfx_s, n_alphan_s,
        addr_exact, addr_diff, addr_ratio, addr_char_s, addr_lev_s, a_tok_over,
        a_tok_jacc, a_num_over, a_num_jacc, a_house_m, a_loc_over, a_loc_jacc,
        s1.is_addr_missing, cand.is_addr_missing, c_match, comb_s, harm_s, both_str,
        name_str_addr_wk, addr_str_name_wk, name_comp_m, name_ocr_m, name_sig_m,
        n_tok_cont, n_last_s, addr_comp_m, a_tok_cont, a_post_m, a_num_sig_m,
        both_addr_miss, same_sc, sc_lat, sc_ind, sc_char_over
    ]


# ==============================================================================
# 3. Main Streamlined Pipeline
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default=None, help="Directory containing test files.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--partition-size", type=int, default=450000, help="S1 partition size.")
    args = parser.parse_args()

    t0_global = time.time()
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.output_dir or os.path.join(root_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    test_dir = args.test_dir
    if not test_dir or not os.path.isdir(test_dir):
        test_dir = os.path.join(root_dir, "..", "dataset", "test")
        if not os.path.isdir(test_dir):
            test_dir = os.path.join(root_dir, "dataset", "test")

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026: STREAMLINED TEST INFERENCE PIPELINE V2")
    print(f"Test Dir   : {test_dir}")
    print(f"Output Dir : {out_dir}")
    print("=" * 80)

    # Load Model & Config
    model_path = os.path.join(out_dir, "entity_matching_model_v2.joblib")
    config_path = os.path.join(out_dir, "optimal_threshold_v2.json")
    print(f"Loading winning model from {model_path}...")
    model = joblib.load(model_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    high_th = float(cfg.get("high_threshold", 0.75))
    med_th = float(cfg.get("medium_threshold", 0.58))
    print(f"Decision Rule: High Thresh = {high_th:.2f}, Medium Thresh = {med_th:.2f} + Dynamic Agreement")

    cand_v2_path = os.path.join(out_dir, "candidate_pairs_v2.tsv")
    match_v2_path = os.path.join(out_dir, "matching_results_v2.tsv")

    # Initialize headers
    with open(cand_v2_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
    with open(match_v2_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")

    total_s1 = 0
    total_candidates = 0
    total_matches = 0
    france_s1 = 0
    france_matches = 0

    s1_reader = pd.read_csv(s1_path, sep="\t", chunksize=args.partition_size, keep_default_na=False)

    for p_idx, s1_chunk in enumerate(s1_reader, 1):
        t_p_start = time.time()
        n_p = len(s1_chunk)
        total_s1 += n_p
        print(f"\n" + "-" * 70)
        print(f"PARTITION {p_idx}: Processing {n_p:,} Source-1 entities (Total S1 so far: {total_s1:,})...")
        print("-" * 70)

        p_eids = list(s1_chunk["entity_id"])
        p_countries = [c.strip().upper() if c else "UNKNOWN" for c in s1_chunk["country"]]
        p_names = list(s1_chunk["business_name"])
        p_addrs = list(s1_chunk["business_address"])

        p_france = sum(1 for c in p_countries if c in ("FRANCE", "FR"))
        france_s1 += p_france
        print(f"  Countries: {set(p_countries)} (France entities: {p_france:,})")

        # 1. Build Inverted Index
        t_idx = time.time()
        p_index = collections.defaultdict(list)
        p_profiles = [None] * n_p

        for i in range(n_p):
            prof = FastProfile(p_countries[i], p_names[i], p_addrs[i])
            p_profiles[i] = prof
            b_keys = generate_blocking_keys(p_countries[i], p_names[i], p_addrs[i])
            for b_code, k_list in b_keys.items():
                for k in k_list:
                    p_index[k].append(i)

        MAX_POSTINGS = 60
        pruned = 0
        for k in list(p_index.keys()):
            if len(p_index[k]) > MAX_POSTINGS:
                del p_index[k]
                pruned += 1
        if pruned:
            print(f"  Pruned {pruned:,} high-frequency keys (> {MAX_POSTINGS} postings).")
        print(f"  Index built: {len(p_index):,} keys in {time.time() - t_idx:.2f}s.")

        # 2. Stream S2 and S3
        valid_countries = set(p_countries)
        p_cand_dict = [{} for _ in range(n_p)]
        cand_cache = {}

        for source_path, src_label in [(s2_path, "Source 2"), (s3_path, "Source 3")]:
            t_s = time.time()
            rows_s = 0
            pairs_found = 0

            for s_chunk in pd.read_csv(source_path, sep="\t", chunksize=200000, keep_default_na=False):
                for mid, c, name, addr in zip(s_chunk["entity_id"], s_chunk["country"], s_chunk["business_name"], s_chunk["business_address"]):
                    rows_s += 1
                    c_clean = c.strip().upper() if c else "UNKNOWN"
                    if c_clean not in valid_countries:
                        continue

                    # Fast key extraction
                    b_keys = generate_blocking_keys(c_clean, name, addr)
                    matched_s1 = collections.defaultdict(int)
                    for b_code, k_list in b_keys.items():
                        for k in k_list:
                            if k in p_index:
                                for s1_idx in p_index[k]:
                                    matched_s1[s1_idx] += 1

                    if not matched_s1:
                        continue

                    # Only profile if this candidate actually matched an S1 entity!
                    k_tuple = (c_clean, name, addr)
                    if k_tuple not in cand_cache:
                        cand_cache[k_tuple] = FastProfile(c_clean, name, addr)
                    c_prof = cand_cache[k_tuple]

                    for s1_idx, n_blocks in matched_s1.items():
                        cur_dict = p_cand_dict[s1_idx]
                        if len(cur_dict) < 20:
                            cur_dict[mid] = (c_prof, n_blocks)
                            pairs_found += 1
                        elif n_blocks >= 2:
                            for existing_mid, (_, existing_n) in list(cur_dict.items()):
                                if existing_n == 1:
                                    del cur_dict[existing_mid]
                                    cur_dict[mid] = (c_prof, n_blocks)
                                    break

            print(f"  [STREAMED] {src_label}: Scanned {rows_s:,} rows, matched {pairs_found:,} pairs in {time.time() - t_s:.2f}s.")

        # 3. Batch Feature Extraction & Vectorized Inference
        t_inf = time.time()
        print(f"  Scoring candidates with Model D & Dynamic Agreement Rule...")

        p_matched_dict = collections.defaultdict(list)
        batch_X = []
        batch_meta = []  # (s1_idx, mid, s1_prof, c_prof)

        def flush_batch():
            if not batch_X:
                return
            X_arr = np.array(batch_X, dtype=np.float32)
            probs = model.predict_proba(X_arr)[:, 1]

            for (s1_idx, mid, s1_p, c_p), p_val in zip(batch_meta, probs):
                p_val = float(p_val)
                # Dynamic Agreement Decision Rule
                if p_val >= high_th:
                    p_matched_dict[s1_idx].append(mid)
                elif p_val >= med_th:
                    is_name_exact = (s1_p.norm_name and s1_p.norm_name == c_p.norm_name)
                    is_both_strong = (p_val >= 0.65 and s1_p.addr_len > 0 and c_p.addr_len > 0)
                    is_house_match = (s1_p.bldg and s1_p.bldg == c_p.bldg)
                    if is_name_exact or is_both_strong or is_house_match:
                        p_matched_dict[s1_idx].append(mid)

            batch_X.clear()
            batch_meta.clear()

        for s1_idx in range(n_p):
            s1_p = p_profiles[s1_idx]
            for mid, (c_p, _) in p_cand_dict[s1_idx].items():
                batch_X.append(compute_direct_46_features(s1_p, c_p))
                batch_meta.append((s1_idx, mid, s1_p, c_p))
                if len(batch_X) >= 50000:
                    flush_batch()

        flush_batch()
        print(f"  Scored all candidates in {time.time() - t_inf:.2f}s.")

        # 4. Flush partition rows to files
        t_w = time.time()
        c_lines = []
        m_lines = []

        for s1_idx in range(n_p):
            s1_id = p_eids[s1_idx]
            c_list = list(p_cand_dict[s1_idx].keys())
            m_list = p_matched_dict.get(s1_idx, [])

            total_candidates += len(c_list)
            total_matches += len(m_list)
            if p_countries[s1_idx] in ("FRANCE", "FR"):
                france_matches += len(m_list)

            c_lines.append(f"{s1_id}\t{','.join(c_list)}\n")
            m_lines.append(f"{s1_id}\t{','.join(m_list)}\n")

        with open(cand_v2_path, "a", encoding="utf-8") as f_c:
            f_c.writelines(c_lines)
        with open(match_v2_path, "a", encoding="utf-8") as f_m:
            f_m.writelines(m_lines)

        print(f"  Flushed {n_p:,} rows to disk in {time.time() - t_w:.2f}s.")
        print(f"  Partition {p_idx} completed in {time.time() - t_p_start:.2f}s.")

        del p_profiles, p_index, p_cand_dict, p_matched_dict, cand_cache, s1_chunk
        gc.collect()

    dur_all = time.time() - t0_global
    print("\n" + "=" * 80)
    print("PIPELINE V2 COMPLETE")
    print(f"Total Source-1 Entities : {total_s1:,}")
    print(f"Total Candidate Pairs   : {total_candidates:,} (avg {total_candidates/total_s1:.2f} candidates/entity)")
    print(f"Total Predicted Matches : {total_matches:,} (avg {total_matches/total_s1:.2f} matches/entity)")
    print(f"France Entities         : {france_s1:,} | France Matches: {france_matches:,} (avg {france_matches/max(1, france_s1):.2f})")
    print(f"Total Pipeline Runtime  : {dur_all:.1f}s ({dur_all/60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    main()
