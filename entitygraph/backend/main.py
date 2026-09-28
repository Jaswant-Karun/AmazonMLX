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
from search_engine import (
    search_canonical_entities, 
    get_entity_by_id, 
    get_real_platform_stats, 
    get_real_demo_queries,
    get_db_connection
)
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
        "description": "Real-World Business Discovery & Entity Resolution (100,000 Real Entities, 522,219 Source Links)",
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
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT count(*) FROM entities;")
    total_e = c.fetchone()[0]
    c.execute("SELECT count(*) FROM source_records;")
    total_s = c.fetchone()[0]
    conn.close()

    return {
        "status": "healthy",
        "real_entities_indexed": total_e,
        "multi_source_links": total_s,
        "search_engine": "SQLite FTS5 + BM25 Lexical-Spatial Reranker",
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
    return get_real_demo_queries()


@app.get("/api/stats")
def get_stats():
    return get_real_platform_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
