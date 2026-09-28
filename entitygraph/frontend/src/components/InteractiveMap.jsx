import React from 'react';
import { MapPin, Navigation, Compass, Layers, CheckCircle } from 'lucide-react';

export default function InteractiveMap({ results, activeEntityId, onSelectEntity, parsedQuery }) {
  const activeEntity = results.find(r => r.canonical_id === activeEntityId) || results[0];

  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: '480px',
      background: 'linear-gradient(180deg, #0b0f19 0%, #05070c 100%)',
      borderRadius: '16px',
      border: '1px solid var(--border-subtle)',
      overflow: 'hidden',
      boxShadow: '0 8px 30px rgba(0,0,0,0.5)'
    }}>
      {/* Map Control Bar */}
      <div style={{
        position: 'absolute',
        top: '12px',
        left: '12px',
        right: '12px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'rgba(15, 19, 32, 0.85)',
        backdropFilter: 'blur(12px)',
        border: '1px solid var(--border-subtle)',
        borderRadius: '10px',
        padding: '8px 16px',
        zIndex: 5
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.82rem', fontWeight: 600 }}>
          <Compass size={16} color="#818cf8" />
          <span>Geocoded Entity Alignment</span>
          {parsedQuery?.landmark && (
            <span style={{ background: 'rgba(192, 132, 252, 0.2)', color: '#d8b4fe', padding: '2px 8px', borderRadius: '6px', fontSize: '0.72rem' }}>
              Anchor: {parsedQuery.landmark}
            </span>
          )}
        </div>

        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {activeEntity ? `${activeEntity.city}, ${activeEntity.country}` : 'Global Coverage'}
        </div>
      </div>

      {/* Styled SVG Map Canvas */}
      <svg style={{ width: '100%', height: '100%' }} viewBox="0 0 600 450">
        {/* Subtle Map Grid lines */}
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255, 255, 255, 0.03)" strokeWidth="1" />
          </pattern>
          <radialGradient id="radarGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(99, 102, 241, 0.15)" />
            <stop offset="100%" stopColor="transparent" />
          </radialGradient>
        </defs>

        <rect width="600" height="450" fill="url(#grid)" />

        {/* Proximity / Radar Circles centered on active entity */}
        <circle cx="300" cy="225" r="140" fill="url(#radarGlow)" stroke="rgba(99, 102, 241, 0.2)" strokeWidth="1" strokeDasharray="4,4" />
        <circle cx="300" cy="225" r="80" stroke="rgba(99, 102, 241, 0.3)" strokeWidth="1" />
        <circle cx="300" cy="225" r="30" stroke="rgba(16, 185, 129, 0.4)" strokeWidth="1.5" />

        {/* Landmark Pin if identified in query */}
        {parsedQuery?.landmark && (
          <g transform="translate(360, 160)">
            <circle cx="0" cy="0" r="18" fill="rgba(192, 132, 252, 0.2)" stroke="#c084fc" strokeWidth="1.5" />
            <circle cx="0" cy="0" r="5" fill="#c084fc" />
            <rect x="-55" y="-30" width="110" height="18" rx="5" fill="#1e1035" stroke="#c084fc" strokeWidth="0.8" />
            <text x="0" y="-18" textAnchor="middle" fill="#e9d5ff" fontSize="9px" fontWeight="600">
              📍 {parsedQuery.landmark}
            </text>
          </g>
        )}

        {/* Render Business Entity Pins */}
        {results.map((r, i) => {
          const isSelected = r.canonical_id === activeEntity?.canonical_id;
          // Offset locations slightly around center for layout
          const px = 300 + (i % 2 === 0 ? 1 : -1) * (i * 45) + (i === 0 ? 0 : 20);
          const py = 225 + (i > 1 ? 55 : -40) + (i === 0 ? 0 : -15);

          return (
            <g
              key={r.canonical_id}
              transform={`translate(${px}, ${py})`}
              onClick={() => onSelectEntity(r.canonical_id)}
              style={{ cursor: 'pointer' }}
            >
              {/* Outer pulsing ring for selected pin */}
              {isSelected && (
                <circle cx="0" cy="-12" r="22" fill="none" stroke="#6366f1" strokeWidth="2" opacity="0.6">
                  <animate attributeName="r" values="16;26;16" dur="2.5s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.8;0.2;0.8" dur="2.5s" repeatCount="indefinite" />
                </circle>
              )}

              {/* Pin Head */}
              <circle
                cx="0"
                cy="-12"
                r={isSelected ? "14" : "11"}
                fill={isSelected ? "#4f46e5" : "#1e293b"}
                stroke={isSelected ? "#ffffff" : "#818cf8"}
                strokeWidth={isSelected ? "2.5" : "1.5"}
              />

              <text x="0" y="-8" textAnchor="middle" fill="#ffffff" fontSize="10px" fontWeight="700">
                {i + 1}
              </text>

              {/* Label Card */}
              <g transform="translate(0, 15)">
                <rect
                  x="-70"
                  y="0"
                  width="140"
                  height="26"
                  rx="6"
                  fill="#0b0f19"
                  stroke={isSelected ? "#818cf8" : "rgba(255,255,255,0.1)"}
                  strokeWidth="1"
                />
                <text x="0" y="12" textAnchor="middle" fill="#ffffff" fontSize="9px" fontWeight="700">
                  {r.canonical_name.length > 18 ? r.canonical_name.slice(0, 18) + '...' : r.canonical_name}
                </text>
                <text x="0" y="21" textAnchor="middle" fill="#34d399" fontSize="8px" fontWeight="600">
                  {r.confidence_pct}% Match
                </text>
              </g>
            </g>
          );
        })}
      </svg>

      {/* Selected Entity Card Overlay */}
      {activeEntity && (
        <div style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          right: '12px',
          background: 'rgba(15, 19, 32, 0.92)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(99, 102, 241, 0.4)',
          borderRadius: '12px',
          padding: '12px 18px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '3px' }}>
              <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>{activeEntity.canonical_name}</span>
              <span className="badge-verified">{activeEntity.confidence_pct}% Consensus</span>
            </div>
            <div style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
              📍 {activeEntity.golden_address}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              onClick={() => onSelectEntity(activeEntity.canonical_id)}
              className="btn-primary" 
              style={{ fontSize: '0.78rem', padding: '6px 14px' }}
            >
              View Knowledge Graph
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
