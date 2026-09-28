#!/usr/bin/env python3
"""
EntityGraph: Multilingual Query Understanding & Landmark Extraction Engine
Supports English, Tamil (தமிழ்), Hindi (हिन्दी), and French (Français).
"""

import sys
import re
import unicodedata
from typing import Dict, Any, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# OCR digit-to-char mapping for business name repairs
OCR_REPAIR_MAP = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '5': 's', '8': 'b', '@': 'a', '$': 's'})

# Multilingual spatial landmark prepositions
SPATIAL_MARKERS = {
    "en": [
        r"\b(?:near|nearby|close to|next to|beside|opposite|opp\.?|behind|in front of|at|around)\b",
        r"\b(?:facing|adjacent to|cross from)\b"
    ],
    "ta": [
        r"(?:பக்கத்துல|அருகில்|அருகே|எதிரில்|எதிரே|பின்னாடி|பின்புறம்|முன்னாடி|அருகில)",
        r"(?:கிட்ட|பக்கம்)"
    ],
    "hi": [
        r"(?:के पास|पास|के सामने|सामने|के पीछे|पीछे|के बगल में|बगल में)",
    ],
    "fr": [
        r"\b(?:près de|proche de|à côté de|en face de|derrière|devant)\b"
    ]
}

# Categorical keywords across languages
CATEGORY_KEYWORDS = {
    "food_dining": [
        "restaurant", "hotel", "cafe", "tea", "coffee", "bakery", "sweets", "tiffin",
        "mess", "bhojanalaya", "dhaba", "food", "snacks", "bar", "bistrot", "boulangerie",
        "சாப்பாடு", "உணவகம்", "ஹோட்டல்", "டீ", "காபி", "பேக்கரி", "சாப்பாட்டு",
        "खाना", "होटल", "रेस्तरां", "चाय", "कॉफ़ी", "मिठाई", "ढाबा"
    ],
    "healthcare": [
        "pharmacy", "medical", "medicals", "chemist", "clinic", "hospital", "pharma",
        "lab", "diagnostic", "doctor", "health", "care", "pharmacie",
        "மருந்தகம்", "மருத்துவமனை", "கிளினிக்", "டாக்டர்", "மருந்து",
        "दवा", "फार्मेसी", "अस्पताल", "क्लीनिक", "डॉक्टर", "औषधालय"
    ],
    "education": [
        "college", "school", "university", "institute", "academy", "classes", "vidyalaya",
        "ecole", "universite", "lycee",
        "கல்லூரி", "பள்ளி", "பல்கலைக்கழகம்", "கல்வி",
        "कॉलेज", "स्कूल", "विश्वविद्यालय", "संस्थान", "विद्यालय"
    ],
    "retail_grocery": [
        "supermarket", "store", "mart", "provisions", "groceries", "textiles", "silks",
        "departmental", "magasin", "marche",
        "மளிகை", "கடை", "அங்காடி", "துணிக்கடை", "ஸ்டோர்",
        "किराना", "दुकान", "बाजार", "स्टोर", "वस्त्र"
    ],
    "automotive_services": [
        "motors", "garage", "service", "auto", "puncture", "tyres", "fuels", "petrol",
        "வண்டி", "பட்டறை", "மோட்டார்",
        "मोटर", "गैराज", "सर्विस", "पेट्रोल"
    ]
}

# Common transliteration mappings for Indian / Colloquial search terms
TAMIL_COLLOQUIAL_MAP = {
    "சாப்பாடு": "restaurant food",
    "நல்ல சாப்பாடு": "quality restaurant meals",
    "கல்லூரி": "college institute",
    "பள்ளி": "school",
    "மருந்தகம்": "pharmacy medicals",
    "மருத்துவமனை": "hospital healthcare",
    "டீ": "tea shop cafe",
    "காபி": "coffee cafe",
    "கடை": "store shop"
}

HINDI_COLLOQUIAL_MAP = {
    "खाना": "food restaurant",
    "चाय": "tea stall cafe",
    "दवा": "medical pharmacy",
    "अस्पताल": "hospital clinic",
    "दुकान": "store shop"
}

TYPO_CORRECTIONS = {
    "restarant": "restaurant",
    "restuarant": "restaurant",
    "resturant": "restaurant",
    "restraunt": "restaurant",
    "hotal": "hotel",
    "colege": "college",
    "clg": "college",
    "medicls": "medicals",
    "chemst": "chemist",
    "shope": "shop",
    "gandhipurm": "gandhipuram",
    "coimbator": "coimbatore",
    "banglore": "bangalore",
    "cheenai": "chennai"
}


def detect_query_script(text: str) -> str:
    """Detect dominant script of input query."""
    if not text:
        return "latin"
    ta_count = 0
    hi_count = 0
    latin_count = 0
    for ch in text:
        cp = ord(ch)
        if 0x0B80 <= cp <= 0x0BFF:
            ta_count += 1
        elif 0x0900 <= cp <= 0x097F:
            hi_count += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            latin_count += 1

    if ta_count > 0 and ta_count >= hi_count:
        return "tamil"
    if hi_count > 0 and hi_count > ta_count:
        return "hindi"
    return "latin"


def parse_multilingual_query(raw_query: str) -> Dict[str, Any]:
    """
    Parses a user query in any language and extracts:
      - detected script / language
      - target business name / entity keywords
      - target landmark / locality
      - inferred business category
      - cleaned search terms for vector and token retrieval
    """
    cleaned = raw_query.strip()
    script = detect_query_script(cleaned)

    # 1. Typo correction for latin tokens
    tokens = cleaned.split()
    corrected_tokens = [TYPO_CORRECTIONS.get(t.lower(), t) for t in tokens]
    normalized_text = " ".join(corrected_tokens)

    business_part = ""
    landmark_part = ""
    spatial_relation = ""

    # 2. Extract spatial markers & split landmark vs. business
    matched_marker = None
    all_markers = (
        SPATIAL_MARKERS["en"] +
        SPATIAL_MARKERS["ta"] +
        SPATIAL_MARKERS["hi"] +
        SPATIAL_MARKERS["fr"]
    )

    for marker_pat in all_markers:
        m = re.search(marker_pat, normalized_text, re.IGNORECASE)
        if m:
            matched_marker = m.group(0)
            prefix = normalized_text[:m.start()].strip()
            suffix = normalized_text[m.end():].strip()

            # Analyze which side is landmark vs business
            # In English: "near PSG college tea shop" -> prefix empty, landmark "PSG college", business "tea shop"
            # In English: "tea shop near PSG college" -> business "tea shop", landmark "PSG college"
            # In Tamil: "PSG கல்லூரி பக்கத்துல டீ கடை" -> landmark "PSG கல்லூரி", business "டீ கடை"
            if not prefix and suffix:
                # "near <X> <Y>"
                suffix_words = suffix.split()
                if len(suffix_words) >= 3:
                    # heuristic: first half landmark, second half business
                    midpoint = max(1, len(suffix_words) - 2)
                    landmark_part = " ".join(suffix_words[:midpoint])
                    business_part = " ".join(suffix_words[midpoint:])
                else:
                    landmark_part = suffix
                    business_part = ""
            elif prefix and suffix:
                # Check for Indic SOV order (Landmark + Marker + Business)
                if script in ("tamil", "hindi"):
                    landmark_part = prefix
                    business_part = suffix
                else:
                    business_part = prefix
                    landmark_part = suffix
            elif prefix and not suffix:
                business_part = prefix
                landmark_part = ""

            spatial_relation = matched_marker
            break

    if not matched_marker:
        business_part = normalized_text
        landmark_part = ""

    # 3. Detect Inferred Category
    full_lower = normalized_text.lower()
    inferred_category = "general"
    category_matches = []
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in full_lower:
                category_matches.append(cat)
                inferred_category = cat
                break
        if category_matches:
            break

    # 4. Multilingual Expansion
    expanded_search_terms = [business_part] if business_part else []
    if script == "tamil":
        for ta_key, en_val in TAMIL_COLLOQUIAL_MAP.items():
            if ta_key in full_lower:
                expanded_search_terms.append(en_val)
    elif script == "hindi":
        for hi_key, en_val in HINDI_COLLOQUIAL_MAP.items():
            if hi_key in full_lower:
                expanded_search_terms.append(en_val)

    return {
        "raw_query": raw_query,
        "normalized_query": normalized_text,
        "script": script,
        "business_name": business_part.strip(),
        "landmark": landmark_part.strip(),
        "spatial_relation": spatial_relation.strip(),
        "inferred_category": inferred_category,
        "expanded_terms": list(dict.fromkeys(expanded_search_terms))
    }


if __name__ == "__main__":
    queries = [
        "near PSG college tea shop",
        "கல்லூரி பக்கத்துல நல்ல சாப்பாடு",
        "Shri medicals near bus stand",
        "restarant near gandhipurm",
        "boulangerie près de rue de la paix",
        "Apollo pharmacy opposite hospital",
        "एमआईडीसी के पास एग्रो इंडस्ट्रीज"
    ]
    print("=== NLP Parser Demo ===")
    for q in queries:
        res = parse_multilingual_query(q)
        print(f"\nQuery: {q}")
        print(f"  Script: {res['script']} | Category: {res['inferred_category']}")
        print(f"  Business: '{res['business_name']}' | Landmark: '{res['landmark']}' | Rel: '{res['spatial_relation']}'")
