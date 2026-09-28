#!/usr/bin/env python3
"""
EntityGraph: Graph Construction Engine
Constructs React Flow / Vis.js compatible graph topologies (Nodes & Edges)
illustrating cross-source entity resolution, attribute consensus, and physical links.
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
    Constructs a visual knowledge graph structure for an entity cluster:
      - Central Golden Record Node (Canonical Entity)
      - Satellite Source Records (Source 1, Source 2, Source 3)
      - Alias Nodes
      - Physical Address & Landmark Nodes
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
            "confidence": f"{entity['confidence_score'] * 100:.1f}%",
            "rating": entity["rating"],
            "is_golden": True,
            "badge": "Verified Cluster",
            "category": entity["category"]
        },
        "position": {"x": 400, "y": 250},
        "style": {
            "background": "#1e1b4b",
            "color": "#ffffff",
            "border": "2px solid #818cf8",
            "borderRadius": "12px",
            "padding": "16px",
            "boxShadow": "0 0 25px rgba(99, 102, 241, 0.45)"
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

        y_pos = src_y_positions[i] if i < len(src_y_positions) else 250 + (i * 70)

        nodes.append({
            "id": s_node_id,
            "type": "source_record",
            "data": {
                "label": src["raw_name"],
                "type_label": src["source_label"],
                "subtitle": src["raw_address"],
                "confidence": f"{src['confidence'] * 100:.1f}%",
                "notes": src["notes"],
                "source_id": s_id
            },
            "position": {"x": 50, "y": y_pos},
            "style": {
                "background": c_scheme["bg"],
                "color": "#f3f4f6",
                "border": f"1.5px solid {c_scheme['border']}",
                "borderRadius": "10px",
                "padding": "12px",
                "boxShadow": f"0 4px 15px {c_scheme['glow']}"
            }
        })

        # Edge from Source Node to Canonical Node
        edges.append({
            "id": f"edge-{s_id}-{entity['canonical_id']}",
            "source": s_node_id,
            "target": canon_node_id,
            "animated": True,
            "label": f"{src['confidence'] * 100:.1f}% Match",
            "style": {"stroke": c_scheme["border"], "strokeWidth": 2.5},
            "labelStyle": {"fill": "#e0e7ff", "fontWeight": 600, "fontSize": "11px"}
        })

    # 3. Address & Physical Landmark Nodes (Right side)
    addr_node_id = f"node-addr-{entity['canonical_id']}"
    nodes.append({
        "id": addr_node_id,
        "type": "address",
        "data": {
            "label": entity["golden_address"],
            "type_label": "Canonical Physical Address",
            "postal_code": entity["postal_code"],
            "locality": entity["locality"]
        },
        "position": {"x": 750, "y": 160},
        "style": {
            "background": "#0f172a",
            "color": "#94a3b8",
            "border": "1.5px solid #38bdf8",
            "borderRadius": "10px",
            "padding": "12px"
        }
    })

    edges.append({
        "id": f"edge-canon-addr",
        "source": canon_node_id,
        "target": addr_node_id,
        "label": "Located At",
        "style": {"stroke": "#38bdf8", "strokeWidth": 2},
        "labelStyle": {"fill": "#93c5fd", "fontSize": "11px"}
    })

    # Landmark Node
    if entity.get("landmark"):
        landmark_node_id = f"node-land-{entity['canonical_id']}"
        nodes.append({
            "id": landmark_node_id,
            "type": "landmark",
            "data": {
                "label": entity["landmark"],
                "type_label": "Geographic Landmark Anchor",
                "locality": entity["locality"]
            },
            "position": {"x": 800, "y": 340},
            "style": {
                "background": "#3b0764",
                "color": "#e9d5ff",
                "border": "1.5px solid #c084fc",
                "borderRadius": "10px",
                "padding": "12px"
            }
        })

        edges.append({
            "id": f"edge-addr-land",
            "source": addr_node_id,
            "target": landmark_node_id,
            "label": "Adjacent Landmark",
            "style": {"stroke": "#c084fc", "strokeWidth": 1.5, "strokeDasharray": "5,5"},
            "labelStyle": {"fill": "#e9d5ff", "fontSize": "11px"}
        })

    # 4. Aliases Node
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
            "position": {"x": 420, "y": 470},
            "style": {
                "background": "#18181b",
                "color": "#a1a1aa",
                "border": "1.5px dashed #71717a",
                "borderRadius": "8px",
                "padding": "10px"
            }
        })

        edges.append({
            "id": f"edge-canon-alias",
            "source": canon_node_id,
            "target": alias_node_id,
            "label": "Known Aliases",
            "style": {"stroke": "#71717a", "strokeWidth": 1.5},
            "labelStyle": {"fill": "#a1a1aa", "fontSize": "11px"}
        })

    return {
        "canonical_id": entity["canonical_id"],
        "canonical_name": entity["canonical_name"],
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "source_count": len(entity["matched_sources"]),
            "alias_count": len(entity.get("aliases", [])),
            "consensus_confidence": f"{entity['confidence_score'] * 100:.1f}%"
        }
    }
