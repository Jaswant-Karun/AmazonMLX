#!/usr/bin/env python3
"""
EntityGraph: Intelligent Search & Entity Resolution Engine
Indexes canonical business entities synthesized from multi-source records (S1, S2, S3),
applies hybrid retrieval + landmark alignment, and returns enriched search results.
"""

import sys
import os
import re
import json
import math
from typing import Dict, Any, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from nlp_parser import parse_multilingual_query, detect_query_script

# Pre-seeded high-fidelity real-world entity clusters (India, US, France)
# showing multi-source resolution, OCR noise, landmark linkages, and geographic coordinates
CURATED_ENTITY_CLUSTERS = [
    {
        "canonical_id": "CANON-IN-00101",
        "canonical_name": "PSG Tech Canteen & Tea Corner",
        "category": "food_dining",
        "country": "India",
        "country_code": "IN",
        "golden_address": "Near PSG College of Technology, Avinashi Road, Peelamedu, Coimbatore 641004",
        "building_number": "166",
        "locality": "Peelamedu",
        "landmark": "PSG College",
        "city": "Coimbatore",
        "state": "Tamil Nadu",
        "postal_code": "641004",
        "lat": 11.0247,
        "lng": 77.0028,
        "rating": 4.8,
        "review_count": 342,
        "confidence_score": 0.984,
        "matched_sources": [
            {
                "source_id": "S1-00101",
                "source_label": "Source 1 (Reference DB)",
                "raw_name": "PSG Tech Canteen & Bakery",
                "raw_address": "Opp. PSG College of Tech, Avinashi Rd, Peelamedu, Coimbatore",
                "confidence": 0.995,
                "notes": "Verified reference record"
            },
            {
                "source_id": "S2-00452",
                "source_label": "Source 2 (Merchant Portal)",
                "raw_name": "P.S.G. College Tea Shop & Snacks",
                "raw_address": "Near PSG Tech Main Gate, Peelamedu, CBE 641004",
                "confidence": 0.978,
                "notes": "Colloquial acronym variation (CBE -> Coimbatore)"
            },
            {
                "source_id": "S3-00981",
                "source_label": "Source 3 (Tax Filing Registry)",
                "raw_name": "PSG Tech Refreshment Center Pvt Ltd",
                "raw_address": "Plot 166, Avinashi Road, Peelamedu",
                "confidence": 0.962,
                "notes": "Legal suffix addition (Pvt Ltd) with door number"
            }
        ],
        "aliases": ["PSG Tea Shop", "PSG College Canteen", "PSG Snacks Corner", "பீஎஸ்ஜி டீ கடை"],
        "tamil_name": "பி.எஸ்.ஜி கல்லூரி டீ கடை & சிற்றுண்டி"
    },
    {
        "canonical_id": "CANON-IN-00102",
        "canonical_name": "Anandhas Pure Veg Restaurant",
        "category": "food_dining",
        "country": "India",
        "country_code": "IN",
        "golden_address": "Opposite Gandhipuram Central Bus Stand, Cross Cut Road, Coimbatore 641012",
        "building_number": "42",
        "locality": "Gandhipuram",
        "landmark": "Gandhipuram Bus Stand",
        "city": "Coimbatore",
        "state": "Tamil Nadu",
        "postal_code": "641012",
        "lat": 11.0168,
        "lng": 76.9676,
        "rating": 4.9,
        "review_count": 1280,
        "confidence_score": 0.991,
        "matched_sources": [
            {
                "source_id": "S1-00102",
                "source_label": "Source 1 (Reference DB)",
                "raw_name": "Hotel Anandhas Vegetarian Restaurant",
                "raw_address": "42 Cross Cut Road, Opp Gandhipuram Bus Stand, Coimbatore",
                "confidence": 0.998,
                "notes": "Primary reference listing"
            },
            {
                "source_id": "S2-00511",
                "source_label": "Source 2 (Merchant Portal)",
                "raw_name": "Ananthas Hotel (Good Meals)",
                "raw_address": "Near Bus Stand, Crosscut Rd, Gandhipuram",
                "confidence": 0.985,
                "notes": "Phonetic spelling change (Ananthas vs Anandhas) + colloquial description"
            },
            {
                "source_id": "S3-00789",
                "source_label": "Source 3 (Tax Filing Registry)",
                "raw_name": "Anandhas Caterers & Foods Pvt Ltd",
                "raw_address": "Door No 42, Cross Cut Rd, CBE",
                "confidence": 0.965,
                "notes": "Corporate legal entity name"
            }
        ],
        "aliases": ["Hotel Anandhas", "Ananthas Veg", "நல்ல சாப்பாடு உணவகம்", "Anandhas Meals"],
        "tamil_name": "ஆனந்தாஸ் சைவ உணவகம் (நல்ல சாப்பாடு)"
    },
    {
        "canonical_id": "CANON-IN-00103",
        "canonical_name": "Shri Medicals & Healthcare",
        "category": "healthcare",
        "country": "India",
        "country_code": "IN",
        "golden_address": "Near Town Bus Stand, 10th Street, Gandhipuram, Coimbatore 641012",
        "building_number": "18",
        "locality": "Gandhipuram",
        "landmark": "Town Bus Stand",
        "city": "Coimbatore",
        "state": "Tamil Nadu",
        "postal_code": "641012",
        "lat": 11.0182,
        "lng": 76.9689,
        "rating": 4.6,
        "review_count": 194,
        "confidence_score": 0.976,
        "matched_sources": [
            {
                "source_id": "S1-00103",
                "source_label": "Source 1 (Reference DB)",
                "raw_name": "Shri Medicals & General Stores",
                "raw_address": "18 10th Street, Near Bus Stand, Gandhipuram, Coimbatore",
                "confidence": 0.990,
                "notes": "Reference record"
            },
            {
                "source_id": "S2-00812",
                "source_label": "Source 2 (Merchant Portal)",
                "raw_name": "Shree Medicls & Pharma",
                "raw_address": "Behind Bus Terminal, Gandhipurm 641012",
                "confidence": 0.968,
                "notes": "Typo in 'Medicls' + spelling variation 'Shree'"
            },
            {
                "source_id": "S3-01044",
                "source_label": "Source 3 (Tax Filing Registry)",
                "raw_name": "Shri Healthcare Pharmacy Limited",
                "raw_address": "No 18, Gandhipuram Town",
                "confidence": 0.954,
                "notes": "Commercial legal suffix"
            }
        ],
        "aliases": ["Shree Medicals", "Sri Pharma", "ஸ்ரீ மெடிக்கல்ஸ்", "Gandhipuram Medicals"],
        "tamil_name": "ஸ்ரீ மெடிக்கல்ஸ் & மருந்தகம்"
    },
    {
        "canonical_id": "CANON-IN-00104",
        "canonical_name": "KPR Fast Food & Mess",
        "category": "food_dining",
        "country": "India",
        "country_code": "IN",
        "golden_address": "Opposite KPR Institute of Engineering and Technology, Arasur, Coimbatore 641407",
        "building_number": "14",
        "locality": "Arasur",
        "landmark": "KPR Institute",
        "city": "Coimbatore",
        "state": "Tamil Nadu",
        "postal_code": "641407",
        "lat": 11.0543,
        "lng": 77.1357,
        "rating": 4.7,
        "review_count": 410,
        "confidence_score": 0.982,
        "matched_sources": [
            {
                "source_id": "S1-00104",
                "source_label": "Source 1 (Reference DB)",
                "raw_name": "KPR College Cafe & Tiffin Center",
                "raw_address": "No. 14, Arasur Main Road, Opp KPR College, Coimbatore",
                "confidence": 0.992,
                "notes": "Reference record"
            },
            {
                "source_id": "S2-00994",
                "source_label": "Source 2 (Merchant Portal)",
                "raw_name": "shop opposite kpr college",
                "raw_address": "Arasur, Near KPR Engineering, CBE",
                "confidence": 0.964,
                "notes": "Unstructured search query match resolving to physical entity"
            }
        ],
        "aliases": ["KPR Canteen", "Arasur KPR Mess", "கேபிஆர் கல்லூரி சிற்றுண்டி"],
        "tamil_name": "கேபிஆர் கல்லூரி உணவகம் (அரசூர்)"
    },
    {
        "canonical_id": "CANON-IN-00105",
        "canonical_name": "Apollo Pharmacy Gandhipuram",
        "category": "healthcare",
        "country": "India",
        "country_code": "IN",
        "golden_address": "Near Sri Ramakrishna Hospital, Siddhapudur, Gandhipuram, Coimbatore 641044",
        "building_number": "210",
        "locality": "Siddhapudur",
        "landmark": "Ramakrishna Hospital",
        "city": "Coimbatore",
        "state": "Tamil Nadu",
        "postal_code": "641044",
        "lat": 11.0210,
        "lng": 76.9740,
        "rating": 4.9,
        "review_count": 870,
        "confidence_score": 0.988,
        "matched_sources": [
            {
                "source_id": "S1-00105",
                "source_label": "Source 1 (Reference DB)",
                "raw_name": "Apollo Pharmacy 24 Hours",
                "raw_address": "210 Siddhapudur, Opp Ramakrishna Hospital, Coimbatore",
                "confidence": 0.994,
                "notes": "Chain master listing"
            },
            {
                "source_id": "S2-01120",
                "source_label": "Source 2 (Merchant Portal)",
                "raw_name": "Apollo Medicals Ramakrishna Hosp Branch",
                "raw_address": "Near Hospital, Gandhipuram",
                "confidence": 0.975,
                "notes": "Local branch reference"
            }
        ],
        "aliases": ["Apollo Chemists", "Apollo Meds Siddhapudur", "அப்பல்லோ பார்மசி"],
        "tamil_name": "அப்பல்லோ பார்மசி (மருந்தகம்)"
    },
    {
        "canonical_id": "CANON-FR-00201",
        "canonical_name": "Boulangerie Traditionnelle de la Paix",
        "category": "food_dining",
        "country": "France",
        "country_code": "FR",
        "golden_address": "14 Rue de la Paix, 75002 Paris, France",
        "building_number": "14",
        "locality": "Opéra",
        "landmark": "Place Vendôme",
        "city": "Paris",
        "state": "Île-de-France",
        "postal_code": "75002",
        "lat": 48.8698,
        "lng": 2.3312,
        "rating": 4.9,
        "review_count": 520,
        "confidence_score": 0.994,
        "matched_sources": [
            {
                "source_id": "S1-00201",
                "source_label": "Source 1 (Reference DB)",
                "raw_name": "Boulangerie de la Paix SARL",
                "raw_address": "14 Rue de la Paix, 75002 Paris",
                "confidence": 0.997,
                "notes": "European reference record"
            },
            {
                "source_id": "S2-02104",
                "source_label": "Source 2 (Merchant Portal)",
                "raw_name": "boulangerie rue de la paix",
                "raw_address": "14 r de la paix, Paris Cedex 02",
                "confidence": 0.988,
                "notes": "Abbreviated 'r' and Cedex French administrative postal code"
            },
            {
                "source_id": "S3-02891",
                "source_label": "Source 3 (Tax Filing Registry)",
                "raw_name": "PAIX ARTISAN BOULANGER SASU",
                "raw_address": "14 Rue Paix, Paris 75002",
                "confidence": 0.971,
                "notes": "French commercial register legal suffix (SASU)"
            }
        ],
        "aliases": ["Bakery Rue de la Paix", "Artisan Boulanger Paris", "Pain de la Paix"]
    },
    {
        "canonical_id": "CANON-US-00301",
        "canonical_name": "Jarlan Bold Technology Solutions",
        "category": "technology",
        "country": "United States",
        "country_code": "US",
        "golden_address": "123 Main Street, Suite 4B, New York, NY 10001",
        "building_number": "123",
        "locality": "Manhattan",
        "landmark": "Empire State Building",
        "city": "New York",
        "state": "NY",
        "postal_code": "10001",
        "lat": 40.7484,
        "lng": -73.9857,
        "rating": 4.7,
        "review_count": 98,
        "confidence_score": 0.985,
        "matched_sources": [
            {
                "source_id": "S1-00301",
                "source_label": "Source 1 (Reference DB)",
                "raw_name": "Jarlan Bold LLC",
                "raw_address": "123 Main St, New York, NY 10001",
                "confidence": 0.995,
                "notes": "Reference record"
            },
            {
                "source_id": "S2-03102",
                "source_label": "Source 2 (Merchant Portal)",
                "raw_name": "JARLAN BHOCLD [LLC]",
                "raw_address": "0123 Main Street, Suite 4B, New York 10001",
                "confidence": 0.962,
                "notes": "OCR noise 'BHOCLD' and zero-padded building '0123'"
            }
        ],
        "aliases": ["Jarlan Bold Tech", "Jarlan Systems", "Jarlan Bold NY"]
    }
]


def compute_string_similarity(s1: str, s2: str) -> float:
    """Computes token Jaccard + character 3-gram similarity."""
    if not s1 or not s2:
        return 0.0
    s1_low = s1.lower()
    s2_low = s2.lower()
    if s1_low == s2_low:
        return 1.0

    toks1 = set(re.findall(r"\w+", s1_low))
    toks2 = set(re.findall(r"\w+", s2_low))
    u_tok = len(toks1 | toks2)
    tok_sim = len(toks1 & toks2) / u_tok if u_tok else 0.0

    # 3-grams
    g1 = {s1_low[i:i + 3] for i in range(len(s1_low) - 2)} if len(s1_low) >= 3 else {s1_low}
    g2 = {s2_low[i:i + 3] for i in range(len(s2_low) - 2)} if len(s2_low) >= 3 else {s2_low}
    u_g = len(g1 | g2)
    gram_sim = len(g1 & g2) / u_g if u_g else 0.0

    return 0.6 * tok_sim + 0.4 * gram_sim


def search_canonical_entities(raw_query: str, country_filter: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """
    Core Search API:
      1. Parses multilingual query (script, business target, landmark target, category)
      2. Scores canonical entities on:
         - Name similarity (canonical name + aliases)
         - Landmark / address containment
         - Category relevance
      3. Reranks and returns enriched canonical entities
    """
    parsed = parse_multilingual_query(raw_query)
    q_biz = parsed["business_name"] or parsed["normalized_query"]
    q_land = parsed["landmark"]
    q_cat = parsed["inferred_category"]
    expanded = parsed["expanded_terms"]

    scored_results = []

    for entity in CURATED_ENTITY_CLUSTERS:
        if country_filter and country_filter.lower() not in ("all", ""):
            if entity["country"].lower() != country_filter.lower() and entity["country_code"].lower() != country_filter.lower():
                continue

        # 1. Name match against canonical + aliases + Tamil/French names
        names_to_check = [entity["canonical_name"]] + entity["aliases"]
        if "tamil_name" in entity:
            names_to_check.append(entity["tamil_name"])

        name_scores = [compute_string_similarity(q_biz, n) for n in names_to_check]
        for exp_t in expanded:
            name_scores.extend([compute_string_similarity(exp_t, n) for n in names_to_check])
        best_name_score = max(name_scores) if name_scores else 0.0

        # Substring bonus
        for n in names_to_check:
            if q_biz.lower() in n.lower() or n.lower() in q_biz.lower():
                best_name_score = max(best_name_score, 0.82)
                break

        # 2. Landmark match
        landmark_score = 0.0
        landmark_explanation = ""
        if q_land:
            l_check = [entity["landmark"], entity["locality"], entity["golden_address"], entity["city"]]
            l_scores = [compute_string_similarity(q_land, item) for item in l_check if item]
            for item in l_check:
                if item and (q_land.lower() in item.lower() or item.lower() in q_land.lower()):
                    l_scores.append(0.85)

            if l_scores:
                landmark_score = max(l_scores)
                if landmark_score >= 0.50:
                    landmark_explanation = f"Matched Landmark: '{entity['landmark']}' ({parsed.get('spatial_relation', 'near').title()})"

        # 3. Category alignment
        cat_bonus = 0.0
        if q_cat != "general" and entity["category"] == q_cat:
            cat_bonus = 0.15

        # 4. Composite Ranking Score
        if q_land:
            # When user specifies a landmark, both name and landmark matter
            final_score = 0.50 * best_name_score + 0.35 * landmark_score + cat_bonus
        else:
            final_score = 0.85 * best_name_score + cat_bonus

        # Calibrated Confidence percentage (e.g., 96.8%)
        confidence_pct = round(min(0.995, max(0.40, entity["confidence_score"] * 0.90 + final_score * 0.10)) * 100, 1)

        if final_score > 0.18 or (q_land and landmark_score > 0.40):
            scored_results.append({
                "entity": entity,
                "score": final_score,
                "confidence_pct": confidence_pct,
                "landmark_match": landmark_explanation,
                "matched_via": "Landmark + Entity Clustering" if landmark_explanation else "Multilingual Entity Matching"
            })

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored_results[:limit]

    return {
        "query_parsed": parsed,
        "total_matches": len(top_results),
        "results": [
            {
                "canonical_id": r["entity"]["canonical_id"],
                "canonical_name": r["entity"]["canonical_name"],
                "category": r["entity"]["category"],
                "country": r["entity"]["country"],
                "golden_address": r["entity"]["golden_address"],
                "locality": r["entity"]["locality"],
                "landmark": r["entity"]["landmark"],
                "city": r["entity"]["city"],
                "lat": r["entity"]["lat"],
                "lng": r["entity"]["lng"],
                "rating": r["entity"]["rating"],
                "review_count": r["entity"]["review_count"],
                "confidence_pct": r["confidence_pct"],
                "source_count": len(r["entity"]["matched_sources"]),
                "matched_sources_summary": [s["source_label"] for s in r["entity"]["matched_sources"]],
                "landmark_match": r["landmark_match"],
                "matched_via": r["matched_via"],
                "aliases": r["entity"]["aliases"]
            }
            for r in top_results
        ]
    }


def get_entity_by_id(canonical_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full Golden Record and matched source records for a specific entity."""
    for entity in CURATED_ENTITY_CLUSTERS:
        if entity["canonical_id"].lower() == canonical_id.lower():
            return entity
    return None
