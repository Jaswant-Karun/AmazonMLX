#!/usr/bin/env python3
"""
EntityGraph: FastAPI Backend Application
Multilingual entity search, explainable matching evidence,
conflict radar, identity passports, batch enterprise resolution,
and human-in-the-loop review queues.
"""

import sys
import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    get_human_review_queue,
    submit_review_decision,
    get_audit_logs,
    resolve_batch_records,
    get_preloaded_vendor_batch,
    get_db_connection
)
from graph_engine import build_entity_graph

app = FastAPI(
    title="EntityGraph API",
    description="Enterprise Business Identity Resolution & Intelligence Platform",
    version="2.2.0"
)

# Enable CORS for local Next.js / Vite frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewActionRequest(BaseModel):
    canonical_id: str
    decision: str  # CONFIRMED | SEPARATED | MERGED | REJECTED
    reviewer: Optional[str] = "Admin Auditor"
    reason: Optional[str] = ""


class BatchResolveRequest(BaseModel):
    records: List[Dict[str, Any]]


@app.get("/")
def root():
    return {
        "service": "EntityGraph API",
        "version": "2.2.0",
        "description": "Enterprise Business Identity Resolution & Intelligence",
        "endpoints": {
            "search": "/api/search?q=...&country=...",
            "entity_details": "/api/entity/{id}",
            "entity_graph": "/api/graph/{id}",
            "passport": "/api/passport/{id}",
            "review_queue": "/api/review-queue",
            "audit_logs": "/api/audit-logs",
            "batch_resolve": "/api/batch-resolve",
            "batch_sample": "/api/batch-sample",
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
        "model": "LightGBM Model D (46 features) + Dynamic Agreement",
        "competition_submission_score": "0.356 Macro F0.5"
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


@app.get("/api/passport/{canonical_id}")
def get_passport(canonical_id: str):
    entity = get_entity_by_id(canonical_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity '{canonical_id}' not found")
    return entity.get("identity_passport")


@app.get("/api/review-queue")
def get_review_queue(limit: int = Query(15, ge=1, le=50)):
    """Returns human-in-the-loop audit queue containing borderline or conflict entities."""
    return get_human_review_queue(limit=limit)


@app.post("/api/review-action")
def review_action(action: ReviewActionRequest):
    """Submits a human decision: CONFIRMED, SEPARATED, MERGED, REJECTED."""
    return submit_review_decision(action.canonical_id, action.decision, action.reviewer, action.reason)


@app.get("/api/audit-logs")
def get_audit_trail(limit: int = Query(20, ge=1, le=100)):
    """Returns the immutable human review audit log trail."""
    return get_audit_logs(limit=limit)


@app.post("/api/batch-resolve")
def batch_resolve(req: BatchResolveRequest):
    """Resolves an array of unstructured messy vendor/customer records into Golden Profiles."""
    if not req.records:
        raise HTTPException(status_code=400, detail="Records array cannot be empty")
    return resolve_batch_records(req.records)


@app.get("/api/batch-sample")
def get_sample_batch():
    """Returns realistic preloaded sample messy vendor records for 1-click batch resolution."""
    return get_preloaded_vendor_batch()


@app.get("/api/demo-queries")
def get_demo_queries():
    return get_real_demo_queries()


@app.get("/api/stats")
def get_stats():
    return get_real_platform_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
