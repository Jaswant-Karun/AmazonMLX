import React, { useState } from 'react';
import { Network, Database, MapPin, Tag, ShieldCheck, Zap, Info, ChevronRight, X } from 'lucide-react';

export default function InteractiveGraph({ graphData, onClose }) {
  const [selectedNode, setSelectedNode] = useState(null);

  if (!graphData) return null;

  const { nodes, edges, canonical_name, stats } = graphData;

  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: '520px',
      background: 'radial-gradient(circle at 50% 50%, #111425 0%, #08090f 100%)',
      borderRadius: '16px',
      border: '1px solid rgba(99, 102, 241, 0.3)',
      overflow: 'hidden',
      boxShadow: '0 0 40px rgba(0, 0, 0, 0.6)'
    }}>
      {/* Header Bar */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        padding: '12px 20px',
        background: 'rgba(10, 12, 20, 0.85)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        zIndex: 10
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Network size={18} color="#818cf8" />
          <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>Entity Resolution Graph</span>
          <span className="badge-verified">{stats.consensus_confidence} Consensus</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            ({stats.source_count} Sources Linked • {stats.alias_count} Aliases)
          </span>
        </div>

        {onClose && (
          <button 
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '4px' }}
          >
            <X size={18} />
          </button>
        )}
      </div>

      {/* SVG Canvas with Interactive Nodes & Edges */}
      <svg 
        style={{ width: '100%', height: '100%', cursor: 'grab' }}
        viewBox="0 0 950 540"
      >
        <defs>
          <linearGradient id="gradCentral" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#4f46e5" />
            <stop offset="100%" stopColor="#7c3aed" />
          </linearGradient>
          <filter id="glowCentral" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Render Edges */}
        {edges.map(edge => {
          const sourceNode = nodes.find(n => n.id === edge.source);
          const targetNode = nodes.find(n => n.id === edge.target);
          if (!sourceNode || !targetNode) return null;

          const sx = sourceNode.position.x + 80;
          const sy = sourceNode.position.y + 35;
          const tx = targetNode.position.x + 80;
          const ty = targetNode.position.y + 35;
          const mx = (sx + tx) / 2;
          const my = (sy + ty) / 2;

          return (
            <g key={edge.id}>
              <line
                x1={sx}
                y1={sy}
                x2={tx}
                y2={ty}
                stroke={edge.style?.stroke || "#6366f1"}
                strokeWidth={edge.style?.strokeWidth || 2}
                strokeDasharray={edge.style?.strokeDasharray || "none"}
                strokeOpacity={0.65}
              />
              {/* Edge label bubble */}
              {edge.label && (
                <g transform={`translate(${mx}, ${my})`}>
                  <rect 
                    x="-42" 
                    y="-11" 
                    width="84" 
                    height="20" 
                    rx="6" 
                    fill="#0a0c16" 
                    stroke={edge.style?.stroke || "#6366f1"} 
                    strokeWidth="1"
                    strokeOpacity="0.8"
                  />
                  <text
                    x="0"
                    y="3"
                    textAnchor="middle"
                    fill="#e0e7ff"
                    fontSize="9.5px"
                    fontWeight="600"
                    fontFamily="Inter, sans-serif"
                  >
                    {edge.label}
                  </text>
                </g>
              )}
            </g>
          );
        })}

        {/* Render Nodes */}
        {nodes.map(node => {
          const isSelected = selectedNode?.id === node.id;
          const isCanon = node.type === 'canonical';
          const isSource = node.type === 'source_record';
          const isLandmark = node.type === 'landmark';

          const w = isCanon ? 190 : 160;
          const h = isCanon ? 75 : 65;

          return (
            <g
              key={node.id}
              transform={`translate(${node.position.x}, ${node.position.y})`}
              onClick={() => setSelectedNode(node)}
              style={{ cursor: 'pointer' }}
            >
              {/* Node Card Container */}
              <rect
                x="0"
                y="0"
                width={w}
                height={h}
                rx={isCanon ? "14" : "10"}
                fill={isCanon ? "url(#gradCentral)" : (node.style?.background || "#1e293b")}
                stroke={isSelected ? "#ffffff" : (node.style?.border?.split(" ")[2] || "#818cf8")}
                strokeWidth={isSelected ? 3 : 1.8}
                filter={isCanon ? "url(#glowCentral)" : "none"}
              />

              {/* Node Type Label */}
              <text
                x="12"
                y="18"
                fill={isCanon ? "#e0e7ff" : "#94a3b8"}
                fontSize="9px"
                fontWeight="700"
                textTransform="uppercase"
                letterSpacing="0.05em"
              >
                {node.data.type_label}
              </text>

              {/* Main Node Text */}
              <text
                x="12"
                y="38"
                fill="#ffffff"
                fontSize={isCanon ? "12px" : "11px"}
                fontWeight="700"
              >
                {node.data.label.length > 22 ? node.data.label.slice(0, 22) + '...' : node.data.label}
              </text>

              {/* Subtitle / Confidence */}
              <text
                x="12"
                y="55"
                fill={isCanon ? "#c7d2fe" : "#cbd5e1"}
                fontSize="9.5px"
                fontWeight="500"
              >
                {node.data.confidence ? `Consensus: ${node.data.confidence}` : (node.data.subtitle ? (node.data.subtitle.slice(0, 24) + '...') : '')}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Selected Node Inspector Drawer */}
      {selectedNode && (
        <div style={{
          position: 'absolute',
          bottom: '16px',
          right: '16px',
          width: '320px',
          background: 'rgba(15, 18, 30, 0.95)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(99, 102, 241, 0.4)',
          borderRadius: '12px',
          padding: '16px',
          boxShadow: '0 10px 30px rgba(0, 0, 0, 0.6)',
          zIndex: 20
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.7rem', fontWeight: 700, color: '#818cf8', textTransform: 'uppercase' }}>
              {selectedNode.data.type_label}
            </span>
            <button 
              onClick={() => setSelectedNode(null)}
              style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
            >
              <X size={14} />
            </button>
          </div>

          <div style={{ fontWeight: 700, fontSize: '1rem', color: '#ffffff', marginBottom: '6px' }}>
            {selectedNode.data.label}
          </div>

          {selectedNode.data.subtitle && (
            <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '8px' }}>
              {selectedNode.data.subtitle}
            </div>
          )}

          {selectedNode.data.notes && (
            <div style={{
              background: 'rgba(255, 255, 255, 0.04)',
              borderLeft: '3px solid #818cf8',
              padding: '6px 10px',
              fontSize: '0.75rem',
              color: '#cbd5e1',
              borderRadius: '0 6px 6px 0',
              marginBottom: '8px'
            }}>
              <strong>Resolution Logic:</strong> {selectedNode.data.notes}
            </div>
          )}

          {selectedNode.data.confidence && (
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: '#34d399', fontWeight: 600 }}>
              <span>Pairwise Model Confidence</span>
              <span>{selectedNode.data.confidence}</span>
            </div>
          )}
        </div>
      )}

      {/* Legend Footer */}
      <div style={{
        position: 'absolute',
        bottom: '12px',
        left: '20px',
        display: 'flex',
        gap: '16px',
        fontSize: '0.72rem',
        color: '#94a3b8'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#6366f1' }}></div>
          <span>Golden Canonical Record</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#10b981' }}></div>
          <span>Source 1 Reference</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#f59e0b' }}></div>
          <span>Source 2 (OCR/Noisy)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#c084fc' }}></div>
          <span>Landmark Anchor</span>
        </div>
      </div>
    </div>
  );
}
