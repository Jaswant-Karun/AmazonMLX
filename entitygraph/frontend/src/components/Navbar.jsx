import React from 'react';
import { Network, Sparkles, Activity, ShieldCheck, Globe, Database } from 'lucide-react';

export default function Navbar({ onReset, stats }) {
  return (
    <nav className="glass-panel" style={{ margin: '16px 24px', padding: '12px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div 
        onClick={onReset}
        style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}
      >
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '12px',
          background: 'linear-gradient(135deg, #4f46e5, #9333ea)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 15px rgba(99, 102, 241, 0.5)'
        }}>
          <Network size={22} color="#ffffff" />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em' }} className="gradient-text">
              ENTITYGRAPH
            </span>
            <span style={{
              fontSize: '0.65rem',
              fontWeight: 700,
              background: 'rgba(99, 102, 241, 0.2)',
              color: '#a5b4fc',
              border: '1px solid rgba(99, 102, 241, 0.4)',
              padding: '1px 6px',
              borderRadius: '6px',
              textTransform: 'uppercase'
            }}>
              v2.0 PRO
            </span>
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            Multilingual Resolution & Knowledge Discovery
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          background: 'rgba(255, 255, 255, 0.03)',
          border: '1px solid var(--border-subtle)',
          padding: '6px 14px',
          borderRadius: '10px',
          fontSize: '0.78rem'
        }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: '#34d399' }}>
            <ShieldCheck size={14} /> 95.21% Macro F0.5
          </span>
          <span style={{ color: 'var(--border-subtle)' }}>|</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: '#818cf8' }}>
            <Database size={14} /> 2.2M Records
          </span>
          <span style={{ color: 'var(--border-subtle)' }}>|</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: '#fbbf24' }}>
            <Globe size={14} /> 4 Languages
          </span>
        </div>

        <button 
          onClick={onReset}
          className="btn-secondary" 
          style={{ fontSize: '0.8rem', padding: '6px 14px' }}
        >
          <Sparkles size={14} color="#818cf8" /> Demo Queries
        </button>
      </div>
    </nav>
  );
}
