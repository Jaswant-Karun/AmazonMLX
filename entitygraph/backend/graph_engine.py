#!/usr/bin/env python3
"""
EntityGraph: Graph Construction Engine
Constructs semantic identity graph topologies (Nodes & Edges)
with explicit semantic relationship types:
SAME_ENTITY, ALIAS_OF, BRANCH_OF, SHARED_ADDRESS, SOURCE_RECORD, PROXIMITY_LANDMARK.
"""

import sys
import os
from typing import Dict, Any, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from search_engine import get_entity_by_id


def build_entity_graph(canonical_id: str) -> Optional[Dict[str, Any]]:
    """
    Constructs an explainable knowledge graph structure with semantic relationship types:
      - Central Golden Record Node (Canonical Entity)
      - Satellite Source Records (Source 1, Source 2, Source 3)
      - Semantic Typed Edges: SAME_ENTITY, ALIAS_OF, BRANCH_OF, SHARED_ADDRESS
    """
    entity = get_entity_by_id(canonical_id)
    if not entity:
        return None

    nodes = []
    edges = []

    # 1. Central Golden Record Node
    canon_node_id = f"node-{entity['canonical_id']}"
    nodes.append({
        "id": canon_node_id,
        "type": "canonical",
        "data": {
            "label": entity["canonical_name"],
            "type_label": "Golden Record (Canonical)",
            "subtitle": f"{entity['city']}, {entity['country']}",
            "confidence": f"{entity.get('evidence', {}).get('overall_confidence', 96.5)}%",
            "rating": entity["rating"],
            "is_golden": True,
            "badge": "Consolidated Profile",
            "category": entity["category"]
        },
        "position": {"x": 420, "y": 250},
        "style": {
            "background": "#0f172a",
            "color": "#ffffff",
            "border": "2px solid #10b981",
            "borderRadius": "14px",
            "padding": "18px",
            "boxShadow": "0 0 30px rgba(16, 185, 129, 0.45)"
        }
    })

    # 2. Source Record Nodes (Left side)
    src_y_positions = [100, 250, 400]
    colors = {
        "Source 1": {"bg": "#064e3b", "border": "#34d399", "glow": "rgba(52, 211, 153, 0.3)"},
        "Source 2": {"bg": "#78350f", "border": "#fbbf24", "glow": "rgba(251, 191, 36, 0.3)"},
        "Source 3": {"bg": "#701a75", "border": "#f472b6", "glow": "rgba(244, 114, 182, 0.3)"}
    }

    for i, src in enumerate(entity["matched_sources"]):
        s_id = src["source_id"]
        s_node_id = f"node-{s_id}"
        prefix = "Source 1" if s_id.startswith("S1") else ("Source 2" if s_id.startswith("S2") else "Source 3")
        c_scheme = colors.get(prefix, {"bg": "#1f2937", "border": "#9ca3af", "glow": "transparent"})

        y_pos = src_y_positions[i] if i < len(src_y_positions) else 250 + (i * 75)

        nodes.append({
            "id": s_node_id,
            "type": "source_record",
            "data": {
                "label": src["raw_name"],
                "type_label": src["source_label"],
                "subtitle": src["raw_address"],
                "confidence": f"{int(src.get('confidence', 0.95) * 100)}%",
                "notes": src.get("notes", "Raw multi-source candidate"),
                "source_id": s_id
            },
            "position": {"x": 60, "y": y_pos},
            "style": {
                "background": c_scheme["bg"],
                "color": "#f3f4f6",
                "border": f"1.5px solid {c_scheme['border']}",
                "borderRadius": "12px",
                "padding": "14px",
                "boxShadow": f"0 4px 18px {c_scheme['glow']}"
            }
        })

        # Determine explicit semantic relationship type
        conf_val = src.get("confidence", 0.95)
        has_conflict = entity.get("conflict_radar", {}).get("has_conflict", False)

        if i == 0:
            rel_type = "CANONICAL_SOURCE"
            edge_color = "#10b981"
        elif has_conflict:
            rel_type = "POSSIBLE_BRANCH / REVIEW"
            edge_color = "#f59e0b"
        elif conf_val >= 0.90:
            rel_type = "SAME_ENTITY"
            edge_color = "#34d399"
        elif conf_val >= 0.75:
            rel_type = "ALIAS_OF"
            edge_color = "#38bdf8"
        else:
            rel_type = "POSSIBLE_RELATION"
            edge_color = "#a855f7"

        edges.append({
            "id": f"edge-{s_id}-{entity['canonical_id']}",
            "source": s_node_id,
            "target": canon_node_id,
            "animated": True,
            "label": f"{rel_type} ({int(conf_val * 100)}%)",
            "style": {"stroke": edge_color, "strokeWidth": 2.5},
            "labelStyle": {"fill": "#ffffff", "fontWeight": 700, "fontSize": "11px", "background": "rgba(0,0,0,0.7)"}
        })

    # 3. Address & Physical Landmark Nodes (Right side)
    addr_node_id = f"node-addr-{entity['canonical_id']}"
    nodes.append({
        "id": addr_node_id,
        "type": "address",
        "data": {
            "label": entity["golden_address"],
            "type_label": "Physical Address Premises",
            "postal_code": entity["postal_code"],
            "locality": entity["locality"]
        },
        "position": {"x": 780, "y": 160},
        "style": {
            "background": "#0f172a",
            "color": "#94a3b8",
            "border": "1.5px solid #38bdf8",
            "borderRadius": "12px",
            "padding": "14px"
        }
    })

    edges.append({
        "id": f"edge-canon-addr",
        "source": canon_node_id,
        "target": addr_node_id,
        "label": "SHARED_ADDRESS",
        "style": {"stroke": "#38bdf8", "strokeWidth": 2},
        "labelStyle": {"fill": "#93c5fd", "fontWeight": 700, "fontSize": "11px"}
    })

    # Landmark Node
    if entity.get("landmark"):
        landmark_node_id = f"node-land-{entity['canonical_id']}"
        nodes.append({
            "id": landmark_node_id,
            "type": "landmark",
            "data": {
                "label": entity["landmark"],
                "type_label": "Proximity Landmark Anchor",
                "locality": entity["locality"]
            },
            "position": {"x": 820, "y": 340},
            "style": {
                "background": "#3b0764",
                "color": "#e9d5ff",
                "border": "1.5px solid #c084fc",
                "borderRadius": "12px",
                "padding": "14px"
            }
        })

        edges.append({
            "id": f"edge-addr-land",
            "source": addr_node_id,
            "target": landmark_node_id,
            "label": "PROXIMITY_LANDMARK",
            "style": {"stroke": "#c084fc", "strokeWidth": 1.5, "strokeDasharray": "5,5"},
            "labelStyle": {"fill": "#e9d5ff", "fontWeight": 600, "fontSize": "11px"}
        })

    # 4. Aliases Cluster Node
    if entity.get("aliases"):
        alias_node_id = f"node-alias-{entity['canonical_id']}"
        nodes.append({
            "id": alias_node_id,
            "type": "alias_cluster",
            "data": {
                "label": " • ".join(entity["aliases"][:3]),
                "type_label": "Resolved Trading Aliases",
                "count": len(entity["aliases"])
            },
            "position": {"x": 440, "y": 480},
            "style": {
                "background": "#18181b",
                "color": "#a1a1aa",
                "border": "1.5px dashed #71717a",
                "borderRadius": "10px",
                "padding": "12px"
            }
        })

        edges.append({
            "id": f"edge-canon-alias",
            "source": canon_node_id,
            "target": alias_node_id,
            "label": "ALIAS_OF",
            "style": {"stroke": "#71717a", "strokeWidth": 1.5},
            "labelStyle": {"fill": "#a1a1aa", "fontWeight": 600, "fontSize": "11px"}
        })

    return {
        "canonical_id": entity["canonical_id"],
        "canonical_name": entity["canonical_name"],
        "nodes": nodes,
        "edges": edges,
        "semantic_types": ["SAME_ENTITY", "ALIAS_OF", "BRANCH_OF", "SHARED_ADDRESS", "PROXIMITY_LANDMARK"],
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "source_count": len(entity["matched_sources"]),
            "alias_count": len(entity.get("aliases", [])),
            "consensus_confidence": f"{entity.get('evidence', {}).get('overall_confidence', 96.5)}%"
        }
    }
