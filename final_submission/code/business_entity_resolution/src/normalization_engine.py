#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Module: Enhanced Normalization Engine (Phase 4 & 5)
src/normalization_engine.py

Provides robust, multi-representation entity profiling for:
- US, India, and France businesses
- Multilingual and Unicode script detection
- Multiple non-destructive representations for names and addresses
- Comprehensive legal suffix, honorific, and domain handling
"""

import sys
import re
import unicodedata
from typing import Tuple, List, Set, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# ==============================================================================
# Regex Patterns
# ==============================================================================
RE_PUNCT = re.compile(r"[^\w\s]")
RE_WHITESPACE = re.compile(r"\s+")
RE_ALPHANUMERIC = re.compile(r"[a-z0-9]+")
RE_NUMERIC = re.compile(r"\d+")
RE_LEADING_ZEROS = re.compile(r"^0+([1-9]\d*)$")
RE_COMPOUND_NUM = re.compile(r"\b(\d+[/_-]\d+)\b")

RE_DOMAIN_SUFFIX = re.compile(
    r"\.(?:com|org|net|in|co|info|biz|edu|gov|io|ai|tech|me|tv|us|de|fr|uk|ca|au|eu|nl)(?:/.*)?$",
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

# Common OCR / Leet digit substitutions for names (e.g. 8uildcon -> buildcon)
OCR_DIGIT_MAP = str.maketrans({
    '0': 'o',
    '1': 'l',
    '3': 'e',
    '5': 's',
    '8': 'b',
    '@': 'a',
    '$': 's'
})

# ==============================================================================
# Domain Vocabularies
# ==============================================================================
STOPWORDS = {
    "the", "a", "an", "and", "of", "for", "in", "on", "at", "to", "m", "s", "ms",
    "co", "by", "with", "from", "as", "is", "or", "into", "near", "opp", "behind",
    # French stopwords
    "de", "des", "du", "la", "le", "les", "et", "en", "au", "aux", "sur", "dans", "par", "pour"
}

COMMON_HONORIFICS = {
    "dr", "doctor", "shri", "smt", "m/s", "ms", "mr", "mrs", "shree", "sri",
    "prof", "professor", "messrs", "c/o", "d/o", "s/o", "w/o", "md", "ca", "adv",
    "me", "maitre", "docteur", "monsieur", "madame"
}

LEGAL_SUFFIXES = {
    # US / UK / International
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "enterprises", "enterprise", "company", "group", "holdings", "services",
    "solutions", "associates", "technologies", "tech", "llp", "pllc", "gmbh", "co", "sons",
    # India
    "pvt", "private", "proprietorship", "firm", "trust",
    # France
    "sarl", "sas", "sa", "sasu", "eurl", "sci", "snc", "gie", "sca", "scs",
    "selarl", "earl", "scp", "association", "ets", "etablissements"
}

ADDRESS_ABBREVIATIONS = {
    # US / English
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "hwy": "highway",
    "pkwy": "parkway", "pl": "place", "sq": "square", "apt": "apartment",
    "ste": "suite", "bldg": "building", "fl": "floor", "dist": "district",
    # French
    "r": "rue", "imp": "impasse", "all": "allee", "chem": "chemin", "rte": "route",
    "crs": "cours", "res": "residence", "bat": "batiment", "etg": "etage",
    "bp": "boitepostale", "cdx": "cedex"
}

ADDRESS_GENERIC_STOP = {
    "street", "road", "avenue", "drive", "boulevard", "lane", "highway", "place",
    "square", "apartment", "suite", "building", "floor", "district", "null", "no",
    "door", "room", "flat", "plot", "rue", "impasse", "allee", "chemin", "route",
    "cours", "residence", "batiment", "etage", "cedex", "near", "opp", "opposite", "behind"
}


# ==============================================================================
# Helper Functions
# ==============================================================================
def detect_script(text: str) -> str:
    """Detects Unicode script of text: 'latin', 'devanagari', 'indic_other', 'other'."""
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
        elif 0x0600 <= cp <= 0x06FF:
            return "arabic"
    return "latin" if has_latin else "other"


def strip_accents_unicode(text: str) -> str:
    """Normalizes Unicode characters to NFKD and strips European/Latin diacritical accents without altering Indic/regional scripts."""
    if not text or text.isascii():
        return text.lower() if text else ""
    nfkd = unicodedata.normalize("NFKD", text)
    res = []
    for c in nfkd:
        cp = ord(c)
        # Only strip European combining diacritical marks (U+0300 to U+036F)
        if 0x0300 <= cp <= 0x036F:
            continue
        res.append(c)
    return unicodedata.normalize("NFKC", "".join(res)).lower()


def strip_domain_name(name: str) -> str:
    """Safely extracts business name from domain string (e.g. trinitycatholicchurch.com -> trinitycatholicchurch)."""
    if not name or ("." not in name and "www" not in name.lower()):
        return name
    c_dom = RE_WWW_PREFIX.sub("", name.strip())
    c_dom = RE_DOMAIN_SUFFIX.sub("", c_dom)
    return c_dom


def normalize_numeric_token(tok: str) -> str:
    """Removes leading zeros from numeric strings (e.g. '0010' -> '10')."""
    if tok.isdigit():
        return str(int(tok))
    m = RE_LEADING_ZEROS.match(tok)
    return m.group(1) if m else tok


def normalize_address_text(addr: str) -> str:
    """Applies abbreviation expansion and leading-zero stripping to address text."""
    if not addr:
        return ""
    addr = strip_accents_unicode(addr)
    addr = RE_ADDRESS_PREFIXES.sub(" ", addr)
    addr = RE_PUNCT.sub(" ", addr)
    tokens = addr.split()
    norm_tokens = []
    for t in tokens:
        if t in ADDRESS_ABBREVIATIONS:
            norm_tokens.append(ADDRESS_ABBREVIATIONS[t])
        elif t.isdigit():
            norm_tokens.append(normalize_numeric_token(t))
        else:
            norm_tokens.append(t)
    return " ".join(norm_tokens)


# ==============================================================================
# EntityProfileV2: Multi-Representation Profile (Phase 4)
# ==============================================================================
class EntityProfileV2:
    """
    Rich entity profile containing multiple complementary representations:
      - name_original, name_normalized, name_alphanumeric, name_tokenized,
        name_token_signature, name_compact, name_ocr_normalized
      - address_original, address_normalized, address_alphanumeric,
        address_tokenized, address_numeric_signature, address_postal_code,
        address_building_number, address_locality_tokens
      - script_type, country
    """
    __slots__ = (
        "country",
        # Names
        "name_original",
        "name_normalized",
        "name_alphanumeric",
        "name_tokenized",
        "name_token_set",
        "name_meaningful_tokens",
        "name_meaningful_set",
        "name_token_signature",
        "name_compact",
        "name_ocr_normalized",
        "name_first_token",
        "name_last_token",
        "name_char_ngrams",
        "name_len",
        "script_type",
        # Addresses
        "address_original",
        "address_normalized",
        "address_alphanumeric",
        "address_tokenized",
        "address_token_set",
        "address_numeric_signature",
        "address_numeric_set",
        "address_building_number",
        "address_postal_code",
        "address_locality_tokens",
        "address_locality_set",
        "address_compact",
        "address_char_ngrams",
        "address_len",
        "is_addr_missing"
    )

    def __init__(self, country: str, business_name: str, business_address: str):
        self.country = country.strip().upper() if isinstance(country, str) and country.strip() else "UNKNOWN"
        raw_name = business_name if isinstance(business_name, str) else ""
        raw_addr = business_address if isinstance(business_address, str) else ""

        self.name_original = raw_name
        self.address_original = raw_addr

        # Script detection
        self.script_type = detect_script(raw_name)

        # ---------------------------------------------------------
        # NAME MULTI-REPRESENTATIONS
        # ---------------------------------------------------------
        # 1. Strip domain if present
        clean_name = strip_domain_name(raw_name)

        # 2. Check for former/trading names
        if any(kw in clean_name.lower() for kw in ("former", "aka", "dba", "t/a", "commercial")):
            m_former = RE_FORMER_NAME.search(clean_name)
            if m_former and len(m_former.group(1).strip()) > 3:
                clean_name = m_former.group(1).strip()

        # 3. Unicode NFKD accent-stripped
        name_no_accents = strip_accents_unicode(clean_name)
        norm_name_punct = RE_PUNCT.sub(" ", name_no_accents).lower()
        norm_name_str = " ".join(norm_name_punct.split())
        self.name_normalized = norm_name_str
        self.name_len = len(norm_name_str)

        # 4. Tokenized representation
        raw_tokens = norm_name_str.split()
        self.name_tokenized = raw_tokens
        self.name_token_set = set(raw_tokens)

        # 5. Meaningful tokens (strip honorifics, legal suffixes, stopwords)
        filtered = [
            t for t in raw_tokens
            if t not in COMMON_HONORIFICS and t not in LEGAL_SUFFIXES and t not in STOPWORDS and len(t) >= 2
        ]
        self.name_meaningful_tokens = filtered
        self.name_meaningful_set = set(filtered)

        # 6. Token signature (sorted meaningful tokens)
        self.name_token_signature = "_".join(sorted(self.name_meaningful_set))

        # 7. Compact & Alphanumeric
        self.name_compact = "".join(c for c in norm_name_str if c.isalnum())
        self.name_alphanumeric = " ".join("".join(c if c.isalnum() else " " for c in norm_name_str).split())

        # 8. OCR-tolerant representation
        self.name_ocr_normalized = self.name_compact.translate(OCR_DIGIT_MAP)

        # 9. First and Last token
        self.name_first_token = filtered[0] if filtered else (raw_tokens[0] if raw_tokens else "")
        self.name_last_token = filtered[-1] if filtered else (raw_tokens[-1] if raw_tokens else "")

        # 10. Character 3-grams
        if len(norm_name_str) < 3:
            self.name_char_ngrams = {norm_name_str} if norm_name_str else set()
        else:
            self.name_char_ngrams = {norm_name_str[i:i + 3] for i in range(len(norm_name_str) - 2)}

        # ---------------------------------------------------------
        # ADDRESS MULTI-REPRESENTATIONS
        # ---------------------------------------------------------
        norm_addr_str = normalize_address_text(raw_addr)
        self.address_normalized = norm_addr_str
        self.address_len = len(norm_addr_str)
        self.is_addr_missing = 1 if self.address_len == 0 else 0

        # Alphanumeric & Compact
        self.address_compact = "".join(c for c in norm_addr_str if c.isalnum())
        self.address_alphanumeric = " ".join("".join(c if c.isalnum() else " " for c in norm_addr_str).split())

        # Tokenized representation
        addr_tokens = norm_addr_str.split()
        self.address_tokenized = addr_tokens
        self.address_token_set = set(addr_tokens)

        # Compound numbers & numerics
        compounds = RE_COMPOUND_NUM.findall(raw_addr) if ("/" in raw_addr or "-" in raw_addr) else []
        comp_norm = []
        for comp in compounds:
            parts = re.split(r"[/_-]", comp)
            np = [normalize_numeric_token(p) for p in parts if p]
            if len(np) > 1:
                comp_norm.append("_".join(np))

        numerics = list(comp_norm)
        locality = []
        for t in addr_tokens:
            if t.isdigit():
                c_num = normalize_numeric_token(t)
                if c_num and len(c_num) <= 8:
                    numerics.append(c_num)
            elif t.isalpha() and len(t) >= 3 and t not in ADDRESS_GENERIC_STOP and t not in STOPWORDS:
                locality.append(t)

        self.address_numeric_signature = "_".join(numerics[:4]) if numerics else ""
        self.address_numeric_set = set(numerics)

        # Building number & Postal code
        postals = [n for n in numerics if len(n) in (5, 6) and "_" not in n]
        bldgs = [n for n in numerics if len(n) <= 4 and "_" not in n]
        self.address_postal_code = postals[0] if postals else ""
        self.address_building_number = bldgs[0] if bldgs else ""

        self.address_locality_tokens = locality
        self.address_locality_set = set(locality)

        # Character 3-grams
        if len(norm_addr_str) < 3:
            self.address_char_ngrams = {norm_addr_str} if norm_addr_str else set()
        else:
            self.address_char_ngrams = {norm_addr_str[i:i + 3] for i in range(len(norm_addr_str) - 2)}


if __name__ == "__main__":
    # Test suite demonstrating normalization across edge cases
    test_cases = [
        ("US", "Jarlan Bold LLC", "123 Main St, New York, NY 10001"),
        ("US", "JARLAN BHOCLD [LLC]", "0123 Main Street, Suite 4B, New York 10001"),
        ("India", "8uildcon-Textile Private Limited", "Plot No. 45/48, GIDC Industrial Area, Surat"),
        ("India", "Buildcon Textile Pvt Ltd", "45-48 GIDC Ind Area, Surat 395002"),
        ("France", "Société d'Exploitation SARL", "14 Rue de la Paix, 75002 Paris"),
        ("France", "SOCIETE D EXPLOITATION", "14 r de la paix, Paris Cedex 02"),
        ("India", "श्री एग्रो इंडस्ट्रीज", "एमआईडीसी, नागपुर"),
    ]

    print("--- Normalization Engine Test Cases ---")
    for country, name, addr in test_cases:
        p = EntityProfileV2(country, name, addr)
        print(f"\nCountry: {p.country} | Script: {p.script_type}")
        print(f"  Orig Name  : {p.name_original} -> Norm: '{p.name_normalized}'")
        print(f"  Compact    : '{p.name_compact}' | OCR: '{p.name_ocr_normalized}' | Sig: '{p.name_token_signature}'")
        print(f"  Meaningful : {p.name_meaningful_tokens}")
        print(f"  Orig Addr  : {p.address_original} -> Norm: '{p.address_normalized}'")
        print(f"  Bldg: '{p.address_building_number}' | Postal: '{p.address_postal_code}' | Locality: {p.address_locality_tokens[:3]}")
