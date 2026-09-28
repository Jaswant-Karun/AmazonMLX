#!/usr/bin/env python3
"""
EntityGraph: FastAPI Backend Application
Powers multilingual entity search, landmark extraction, and interactive knowledge graphs.
"""

import sys
import os
from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from nlp_parser import parse_multilingual_query
from search_engine import search_canonical_entities, get_entity_by_id, CURATED_ENTITY_CLUSTERS
from graph_engine import build_entity_graph

app = FastAPI(
    title="EntityGraph API",
    description="Multilingual Business Entity Resolution & Knowledge Graph Search Platform",
    version="2.0.0"
)

# Enable CORS for local Next.js / Vite frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "EntityGraph API",
        "version": "2.0.0",
        "description": "Intelligent Global Business Discovery & Entity Resolution",
        "endpoints": {
            "search": "/api/search?q=...&country=...",
            "entity_details": "/api/entity/{id}",
            "entity_graph": "/api/graph/{id}",
            "demo_queries": "/api/demo-queries",
            "stats": "/api/stats"
        }
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "entities_indexed": len(CURATED_ENTITY_CLUSTERS),
        "model": "LightGBM Model D (46 features) + Dynamic Agreement"
    }


@app.get("/api/search")
def search(
    q: str = Query(..., description="Multilingual search query with optional landmarks"),
    country: Optional[str] = Query(None, description="Country filter ('India', 'France', 'United States', or 'all')"),
    limit: int = Query(10, ge=1, le=50)
):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query string 'q' cannot be empty")
    return search_canonical_entities(raw_query=q, country_filter=country, limit=limit)


@app.get("/api/entity/{canonical_id}")
def get_entity(canonical_id: str):
    entity = get_entity_by_id(canonical_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity '{canonical_id}' not found")
    return entity


@app.get("/api/graph/{canonical_id}")
def get_graph(canonical_id: str):
    graph = build_entity_graph(canonical_id)
    if not graph:
        raise HTTPException(status_code=404, detail=f"Entity graph for '{canonical_id}' not found")
    return graph


@app.get("/api/demo-queries")
def get_demo_queries():
    return [
        {
            "id": "q1",
            "language": "English",
            "flag": "🇮🇳",
            "query": "near PSG college tea shop",
            "description": "Landmark extraction ('PSG college') + Category ('tea shop')",
            "target_entity": "PSG Tech Canteen & Tea Corner",
            "canonical_id": "CANON-IN-00101"
        },
        {
            "id": "q2",
            "language": "Tamil (தமிழ்)",
            "flag": "🇮🇳",
            "query": "கல்லூரி பக்கத்துல நல்ல சாப்பாடு",
            "description": "Regional script search ('கல்லூரி' -> college, 'நல்ல சாப்பாடு' -> quality meals)",
            "target_entity": "Anandhas Pure Veg Restaurant",
            "canonical_id": "CANON-IN-00102"
        },
        {
            "id": "q3",
            "language": "English (Typo)",
            "flag": "🇮🇳",
            "query": "Shri medicals near bus stand",
            "description": "Fuzzy matching ('Shree' vs 'Shri') + landmark alignment ('bus stand')",
            "target_entity": "Shri Medicals & Healthcare",
            "canonical_id": "CANON-IN-00103"
        },
        {
            "id": "q4",
            "language": "English (Colloquial)",
            "flag": "🇮🇳",
            "query": "shop opposite kpr college",
            "description": "Unstructured search resolving to physical institution",
            "target_entity": "KPR Fast Food & Mess",
            "canonical_id": "CANON-IN-00104"
        },
        {
            "id": "q5",
            "language": "French (Français)",
            "flag": "🇫🇷",
            "query": "boulangerie rue de la paix",
            "description": "Zero-shot European address resolution (Paris, France)",
            "target_entity": "Boulangerie Traditionnelle de la Paix",
            "canonical_id": "CANON-FR-00201"
        },
        {
            "id": "q6",
            "language": "English (US)",
            "flag": "🇺🇸",
            "query": "jarlan bold llc suite 4b",
            "description": "US corporate name resolution with OCR repair ('BHOCLD' -> 'Bold')",
            "target_entity": "Jarlan Bold Technology Solutions",
            "canonical_id": "CANON-US-00301"
        }
    ]


@app.get("/api/stats")
def get_stats():
    return {
        "platform_name": "EntityGraph",
        "total_records_processed": "2,206,821 Source Records",
        "canonical_entities_formed": "1,732,544 Golden Entities",
        "precision_macro_f05": 0.95215,
        "pairwise_precision": "98.38%",
        "singleton_fp_rate": "3.85%",
        "supported_languages": ["English", "Tamil (தமிழ்)", "Hindi (हिन्दी)", "French (Français)"],
        "model_architecture": "LightGBM Model D (46 Pairwise Features)",
        "decision_rule": "Dual-Threshold Dynamic Agreement Rule (T_high=0.75, T_med=0.58)",
        "average_query_latency_ms": 14.2
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
