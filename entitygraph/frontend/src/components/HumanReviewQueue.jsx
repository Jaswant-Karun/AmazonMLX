import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, AlertTriangle, CheckCircle2, XCircle, Split, 
  Merge, RefreshCw, Filter, ArrowUpRight, Search, Eye
} from 'lucide-react';

export default function HumanReviewQueue({ onSelectEntity, onOpenDetails }) {
  const [queue, setQueue] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [filterMode, setFilterMode] = useState('ALL'); // 'ALL' | 'NEEDS_REVIEW' | 'AUTO_MATCHED'

  const fetchQueue = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/review-queue?limit=20');
      const data = await res.json();
      setQueue(data);
    } catch (e) {
      console.error("Failed to fetch review queue:", e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchQueue();
  }, []);

  const handleDecision = async (canonicalId, decision) => {
    try {
      await fetch('http://127.0.0.1:8000/api/review-action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ canonical_id: canonicalId, decision })
      });
      setQueue(prev => prev.map(item => 
        item.canonical_id === canonicalId ? { ...item, status: decision } : item
      ));
    } catch (e) {
      console.error("Failed to submit decision:", e);
    }
  };

  const filteredItems = queue.filter(item => {
    if (filterMode === 'NEEDS_REVIEW') return item.status === 'NEEDS_REVIEW' || item.conflict_detected;
    if (filterMode === 'AUTO_MATCHED') return item.status === 'AUTO_MATCHED' || item.status === 'CONFIRMED';
    return true;
  });

  return (
    <div className="glass-panel" style={{ padding: '28px', marginBottom: '40px' }}>
      {/* Queue Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span className="badge-gold">Human-in-the-Loop Audit</span>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              Resolution Confidence Discrepancy & Conflict Queue
            </span>
          </div>
          <h2 style={{ fontSize: '1.45rem', fontWeight: 800, color: '#ffffff' }}>
            Entity Review & Conflict Radar Queue
          </h2>
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <div style={{ display: 'flex', background: 'rgba(255,255,255,0.04)', borderRadius: '10px', padding: '3px', border: '1px solid rgba(255,255,255,0.08)' }}>
            <button
              onClick={() => setFilterMode('ALL')}
              style={{
                background: filterMode === 'ALL' ? '#4f46e5' : 'transparent',
                color: filterMode === 'ALL' ? '#ffffff' : '#94a3b8',
                border: 'none',
                borderRadius: '6px',
                padding: '5px 12px',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              All Records
            </button>
            <button
              onClick={() => setFilterMode('NEEDS_REVIEW')}
              style={{
                background: filterMode === 'NEEDS_REVIEW' ? '#d97706' : 'transparent',
                color: filterMode === 'NEEDS_REVIEW' ? '#ffffff' : '#94a3b8',
                border: 'none',
                borderRadius: '6px',
                padding: '5px 12px',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              ⚠️ Needs Review
            </button>
            <button
              onClick={() => setFilterMode('AUTO_MATCHED')}
              style={{
                background: filterMode === 'AUTO_MATCHED' ? '#059669' : 'transparent',
                color: filterMode === 'AUTO_MATCHED' ? '#ffffff' : '#94a3b8',
                border: 'none',
                borderRadius: '6px',
                padding: '5px 12px',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              ✓ Auto-Matched
            </button>
          </div>

          <button
            onClick={fetchQueue}
            className="btn-secondary"
            style={{ padding: '6px 12px', fontSize: '0.75rem', gap: '4px' }}
          >
            <RefreshCw size={13} className={isLoading ? 'spin' : ''} /> Refresh Queue
          </button>
        </div>
      </div>

      {/* Queue Table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.84rem' }}>
          <thead>
            <tr style={{ background: 'rgba(255, 255, 255, 0.04)', textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
              <th style={{ padding: '12px 14px' }}>Entity ID</th>
              <th style={{ padding: '12px 14px' }}>Business Name</th>
              <th style={{ padding: '12px 14px' }}>Address & Jurisdiction</th>
              <th style={{ padding: '12px 14px' }}>Confidence</th>
              <th style={{ padding: '12px 14px' }}>Conflict Audit Status</th>
              <th style={{ padding: '12px 14px', textAlign: 'right' }}>Reviewer Action</th>
            </tr>
          </thead>
          <tbody>
            {filteredItems.map((item, idx) => (
              <tr 
                key={item.canonical_id} 
                style={{ 
                  borderBottom: '1px solid rgba(255,255,255,0.04)',
                  background: idx % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)'
                }}
              >
                <td style={{ padding: '14px', fontWeight: 700, color: '#38bdf8' }}>
                  <code>{item.canonical_id}</code>
                </td>

                <td style={{ padding: '14px', fontWeight: 700, color: '#ffffff' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span>{item.business_name}</span>
                    <button
                      onClick={() => onOpenDetails(item.canonical_id)}
                      title="Inspect Entity Investigation"
                      style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', padding: '2px' }}
                    >
                      <Eye size={14} />
                    </button>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 400 }}>
                    {item.matched_source_count} Ingested Representations
                  </div>
                </td>

                <td style={{ padding: '14px', color: '#cbd5e1', maxWidth: '240px' }}>
                  <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {item.business_address}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#64748b' }}>
                    {item.country} • {item.category}
                  </div>
                </td>

                <td style={{ padding: '14px' }}>
                  <span style={{ 
                    color: item.confidence_pct >= 90 ? '#34d399' : (item.confidence_pct >= 75 ? '#fbbf24' : '#f87171'),
                    fontWeight: 800
                  }}>
                    {item.confidence_pct}%
                  </span>
                </td>

                <td style={{ padding: '14px' }}>
                  {item.status === 'NEEDS_REVIEW' ? (
                    <span style={{ 
                      background: 'rgba(245, 158, 11, 0.15)', 
                      border: '1px solid #f59e0b', 
                      color: '#fbbf24', 
                      padding: '3px 8px', 
                      borderRadius: '6px', 
                      fontSize: '0.72rem', 
                      fontWeight: 700,
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}>
                      <AlertTriangle size={12} /> Needs Review
                    </span>
                  ) : item.status === 'CONFIRMED' ? (
                    <span style={{ 
                      background: 'rgba(16, 185, 129, 0.15)', 
                      border: '1px solid #10b981', 
                      color: '#34d399', 
                      padding: '3px 8px', 
                      borderRadius: '6px', 
                      fontSize: '0.72rem', 
                      fontWeight: 700
                    }}>
                      ✓ Confirmed
                    </span>
                  ) : item.status === 'SEPARATED' ? (
                    <span style={{ 
                      background: 'rgba(239, 68, 68, 0.15)', 
                      border: '1px solid #ef4444', 
                      color: '#f87171', 
                      padding: '3px 8px', 
                      borderRadius: '6px', 
                      fontSize: '0.72rem', 
                      fontWeight: 700
                    }}>
                      Split / Branch
                    </span>
                  ) : (
                    <span style={{ 
                      background: 'rgba(56, 189, 248, 0.12)', 
                      border: '1px solid #38bdf8', 
                      color: '#38bdf8', 
                      padding: '3px 8px', 
                      borderRadius: '6px', 
                      fontSize: '0.72rem', 
                      fontWeight: 700
                    }}>
                      Auto-Matched
                    </span>
                  )}
                  <div style={{ fontSize: '0.68rem', color: '#94a3b8', marginTop: '2px' }}>
                    {item.flag_reason}
                  </div>
                </td>

                <td style={{ padding: '14px', textAlign: 'right' }}>
                  <div style={{ display: 'inline-flex', gap: '6px' }}>
                    <button
                      onClick={() => handleDecision(item.canonical_id, 'CONFIRMED')}
                      title="Confirm Golden Match"
                      style={{
                        background: 'rgba(16, 185, 129, 0.2)',
                        border: '1px solid #10b981',
                        color: '#34d399',
                        borderRadius: '6px',
                        padding: '4px 8px',
                        fontSize: '0.72rem',
                        fontWeight: 700,
                        cursor: 'pointer'
                      }}
                    >
                      Confirm
                    </button>
                    <button
                      onClick={() => handleDecision(item.canonical_id, 'SEPARATED')}
                      title="Split into Separate Branch"
                      style={{
                        background: 'rgba(239, 68, 68, 0.15)',
                        border: '1px solid #ef4444',
                        color: '#f87171',
                        borderRadius: '6px',
                        padding: '4px 8px',
                        fontSize: '0.72rem',
                        fontWeight: 700,
                        cursor: 'pointer'
                      }}
                    >
                      Split
                    </button>
                    <button
                      onClick={() => onOpenDetails(item.canonical_id)}
                      className="btn-secondary"
                      style={{ padding: '4px 8px', fontSize: '0.72rem' }}
                    >
                      Audit
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
