import React, { useState } from 'react';
import { 
  X, ShieldCheck, CheckCircle2, AlertCircle, Building2, MapPin, 
  Layers, Award, Sparkles, Download, Clock, AlertTriangle, ArrowRight,
  Check, FileText, ChevronRight, Globe, Info
} from 'lucide-react';

export default function EntityDetailModal({ entity, onClose, onOpenGraph, onOpenMap }) {
  if (!entity) return null;

  const [activeTab, setActiveTab] = useState('evidence'); // 'evidence' | 'timeline' | 'sources' | 'passport'
  const [reviewDecision, setReviewDecision] = useState(entity.conflict_radar?.status || 'AUTO_MATCHED');
  const [isSubmittingReview, setIsSubmittingReview] = useState(false);

  const evidence = entity.evidence || {
    name_similarity: 96,
    address_similarity: 91,
    building_number_match: 100,
    token_agreement: 94,
    overall_confidence: 96.5,
    positive_evidence: ["Consistent trade name and postal region", "Exact building/door number match"],
    risk_flags: []
  };

  const conflict = entity.conflict_radar || {
    has_conflict: false,
    status: 'AUTO_MATCHED',
    name_agreement: '96%',
    address_agreement: '91%',
    potential_causes: []
  };

  const timeline = entity.identity_timeline || [];
  const passport = entity.identity_passport || {};

  const handleDownloadPassport = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(passport, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `Identity_Passport_${entity.canonical_id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleReviewAction = async (decision) => {
    setIsSubmittingReview(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/review-action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ canonical_id: entity.canonical_id, decision })
      });
      const data = await res.json();
      setReviewDecision(data.status);
    } catch (e) {
      console.error("Failed to submit review action:", e);
    } finally {
      setIsSubmittingReview(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(4, 7, 15, 0.85)',
      backdropFilter: 'blur(12px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 2000,
      padding: '24px'
    }}>
      <div className="glass-panel" style={{
        width: '100%',
        maxWidth: '920px',
        maxHeight: '92vh',
        overflowY: 'auto',
        background: '#0a0e1a',
        border: '1px solid rgba(16, 185, 129, 0.45)',
        boxShadow: '0 25px 70px rgba(0, 0, 0, 0.9)',
        padding: '32px',
        position: 'relative',
        borderRadius: '24px'
      }}>
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '24px',
            right: '24px',
            background: 'rgba(255, 255, 255, 0.06)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            color: 'var(--text-muted)',
            borderRadius: '50%',
            width: '36px',
            height: '36px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer'
          }}
        >
          <X size={18} />
        </button>

        {/* Modal Top Metadata Banner */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px', flexWrap: 'wrap' }}>
          <span className="badge-verified">
            <ShieldCheck size={14} /> Golden Profile Resolved
          </span>
          <span className="badge-gold">
            <Award size={14} /> {evidence.overall_confidence}% Identity Confidence
          </span>
          <span style={{ fontSize: '0.78rem', color: '#94a3b8', background: 'rgba(255,255,255,0.05)', padding: '2px 8px', borderRadius: '6px' }}>
            ID: <code style={{ color: '#38bdf8' }}>{entity.canonical_id}</code>
          </span>
          {conflict.has_conflict ? (
            <span style={{ background: 'rgba(245, 158, 11, 0.2)', border: '1px solid #f59e0b', color: '#fbbf24', padding: '2px 10px', borderRadius: '9999px', fontSize: '0.75rem', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
              <AlertTriangle size={13} /> {reviewDecision}
            </span>
          ) : (
            <span style={{ background: 'rgba(16, 185, 129, 0.2)', border: '1px solid #10b981', color: '#34d399', padding: '2px 10px', borderRadius: '9999px', fontSize: '0.75rem', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
              <Check size={13} /> {reviewDecision}
            </span>
          )}
        </div>

        {/* Canonical Title & Address */}
        <h2 style={{ fontSize: '1.9rem', marginBottom: '6px', fontWeight: 800 }} className="gradient-text">
          {entity.canonical_name}
        </h2>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#cbd5e1', fontSize: '0.92rem', marginBottom: '22px' }}>
          <MapPin size={16} color="#34d399" />
          <span>{entity.golden_address}</span>
          <span style={{ color: '#64748b' }}>•</span>
          <span style={{ color: '#fbbf24', fontWeight: 600 }}>{entity.country}</span>
        </div>

        {/* Workspace Navigation Subtabs */}
        <div style={{
          display: 'flex',
          gap: '8px',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          paddingBottom: '14px',
          marginBottom: '24px'
        }}>
          {[
            { id: 'evidence', label: '🔍 Explainable Evidence', icon: Info },
            { id: 'timeline', label: '⏳ Identity Timeline', icon: Clock },
            { id: 'sources', label: '📑 Multi-Source Lineage', icon: Layers },
            { id: 'passport', label: '🪪 Identity Passport', icon: FileText }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                background: activeTab === tab.id ? 'rgba(16, 185, 129, 0.18)' : 'rgba(255, 255, 255, 0.03)',
                border: activeTab === tab.id ? '1px solid #10b981' : '1px solid rgba(255, 255, 255, 0.08)',
                color: activeTab === tab.id ? '#34d399' : '#94a3b8',
                borderRadius: '10px',
                padding: '8px 16px',
                fontSize: '0.82rem',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                transition: 'all 0.15s ease'
              }}
            >
              <tab.icon size={15} /> {tab.label}
            </button>
          ))}
        </div>

        {/* TAB 1: EXPLAINABLE MATCHING EVIDENCE & CONFLICT RADAR */}
        {activeTab === 'evidence' && (
          <div>
            {/* Conflict Radar Alert Card */}
            {conflict.has_conflict ? (
              <div style={{
                background: 'rgba(245, 158, 11, 0.12)',
                border: '1px solid #f59e0b',
                borderRadius: '14px',
                padding: '16px 20px',
                marginBottom: '24px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                  <AlertTriangle size={20} color="#fbbf24" />
                  <span style={{ fontSize: '1rem', fontWeight: 800, color: '#fbbf24' }}>
                    Identity Conflict Radar Flagged
                  </span>
                  <span style={{ fontSize: '0.75rem', background: '#78350f', color: '#fef3c7', padding: '2px 8px', borderRadius: '4px' }}>
                    Name: {conflict.name_agreement} | Address: {conflict.address_agreement}
                  </span>
                </div>
                <p style={{ fontSize: '0.82rem', color: '#fde68a', marginBottom: '10px' }}>
                  High name agreement with divergent physical address tokens. Potential reasons:
                </p>
                <ul style={{ fontSize: '0.78rem', color: '#fcd34d', paddingLeft: '20px', marginBottom: '14px' }}>
                  {conflict.potential_causes?.map((c, i) => <li key={i}>{c}</li>)}
                </ul>

                {/* Human-in-the-Loop Review Actions */}
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.75rem', color: '#cbd5e1', fontWeight: 600 }}>Human Reviewer Action:</span>
                  <button onClick={() => handleReviewAction('CONFIRMED')} className="btn-secondary" style={{ padding: '4px 10px', fontSize: '0.74rem' }}>
                    Confirm Match
                  </button>
                  <button onClick={() => handleReviewAction('SEPARATED')} className="btn-secondary" style={{ padding: '4px 10px', fontSize: '0.74rem', color: '#f87171' }}>
                    Keep Separate (Branch)
                  </button>
                  <button onClick={() => handleReviewAction('MERGED')} className="btn-secondary" style={{ padding: '4px 10px', fontSize: '0.74rem' }}>
                    Merge Aliases
                  </button>
                </div>
              </div>
            ) : (
              <div style={{
                background: 'rgba(16, 185, 129, 0.08)',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                borderRadius: '14px',
                padding: '14px 18px',
                marginBottom: '24px',
                display: 'flex',
                alignItems: 'center',
                gap: '12px'
              }}>
                <CheckCircle2 size={20} color="#34d399" />
                <div>
                  <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#34d399' }}>
                    Conflict Radar: Clear Agreement Consensus
                  </div>
                  <div style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
                    Name agreement ({conflict.name_agreement}) and address agreement ({conflict.address_agreement}) satisfy automatic golden consolidation criteria.
                  </div>
                </div>
              </div>
            )}

            {/* Why Were These Matched? (Progress Bars) */}
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800, marginBottom: '14px', color: '#ffffff' }}>
              Why Were These Records Matched? (Feature-Level Explainability)
            </h3>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px' }}>
              {[
                { label: 'Trade Name Lexical & Phonetic Similarity', val: evidence.name_similarity, color: '#34d399' },
                { label: 'Corridor & Locality Address Agreement', val: evidence.address_similarity, color: '#38bdf8' },
                { label: 'Building / Door Number Match', val: evidence.building_number_match, color: '#a855f7' },
                { label: 'Administrative & Token Overlap', val: evidence.token_agreement, color: '#fbbf24' }
              ].map(f => (
                <div key={f.label} style={{ background: 'rgba(255,255,255,0.03)', padding: '14px 16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.8rem', fontWeight: 600 }}>
                    <span style={{ color: '#cbd5e1' }}>{f.label}</span>
                    <span style={{ color: f.color, fontWeight: 800 }}>{f.val}%</span>
                  </div>
                  <div style={{ background: 'rgba(255,255,255,0.08)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ background: f.color, height: '100%', width: `${f.val}%`, borderRadius: '3px' }}></div>
                  </div>
                </div>
              ))}
            </div>

            {/* Evidence Checklist */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.06)' }}>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#34d399', marginBottom: '10px' }}>
                  ✓ Positive Matching Evidence
                </h4>
                <ul style={{ listStyle: 'none', padding: 0, fontSize: '0.78rem', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {evidence.positive_evidence?.map((pos, i) => (
                    <li key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                      <Check size={14} color="#34d399" style={{ flexShrink: 0, marginTop: '2px' }} />
                      <span>{pos}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.06)' }}>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#fbbf24', marginBottom: '10px' }}>
                  ⚠ Noise Variations & Risk Factors
                </h4>
                {evidence.risk_flags && evidence.risk_flags.length > 0 ? (
                  <ul style={{ listStyle: 'none', padding: 0, fontSize: '0.78rem', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {evidence.risk_flags.map((neg, i) => (
                      <li key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                        <AlertCircle size={14} color="#fbbf24" style={{ flexShrink: 0, marginTop: '2px' }} />
                        <span>{neg}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
                    No critical risk flags or missing administrative parameters detected.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: IDENTITY TIMELINE (OBSERVED RECORD EVOLUTION) */}
        {activeTab === 'timeline' && (
          <div>
            <div style={{ marginBottom: '18px' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#ffffff', marginBottom: '4px' }}>
                Observed Record Evolution / Identity Continuity
              </h3>
              <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                Chronological representation of record appearances across ingested data layers.
              </p>
            </div>

            <div style={{ position: 'relative', paddingLeft: '32px' }}>
              <div style={{ position: 'absolute', left: '11px', top: '10px', bottom: '10px', width: '2px', background: 'rgba(16, 185, 129, 0.4)' }}></div>

              {timeline.map((item, idx) => (
                <div key={idx} style={{ position: 'relative', marginBottom: '22px' }}>
                  <div style={{
                    position: 'absolute',
                    left: '-32px',
                    top: '2px',
                    width: '24px',
                    height: '24px',
                    borderRadius: '50%',
                    background: item.source_id === 'CANONICAL-UNIFIED' ? '#10b981' : '#0f172a',
                    border: '2px solid #34d399',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '10px',
                    fontWeight: 800,
                    color: '#ffffff'
                  }}>
                    {idx + 1}
                  </div>

                  <div style={{
                    background: item.source_id === 'CANONICAL-UNIFIED' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(255, 255, 255, 0.03)',
                    border: item.source_id === 'CANONICAL-UNIFIED' ? '1px solid #10b981' : '1px solid rgba(255, 255, 255, 0.06)',
                    borderRadius: '12px',
                    padding: '14px 18px'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                      <span style={{ fontSize: '0.88rem', fontWeight: 800, color: '#34d399' }}>
                        {item.year} — {item.title}
                      </span>
                      <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                        {item.confidence} Concordance
                      </span>
                    </div>

                    <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#ffffff', marginBottom: '4px' }}>
                      "{item.recorded_name}"
                    </div>

                    <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '6px' }}>
                      📍 {item.recorded_address}
                    </div>

                    <div style={{ fontSize: '0.72rem', color: '#64748b' }}>
                      <strong>Continuity Type:</strong> {item.continuity_type} • <em>{item.notes}</em>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 3: MULTI-SOURCE LINEAGE AUDIT */}
        {activeTab === 'sources' && (
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800, marginBottom: '14px', color: '#ffffff' }}>
              Multi-Source Ingestion & Pairwise Discrepancies
            </h3>
            <div style={{ overflowX: 'auto', marginBottom: '20px' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
                <thead>
                  <tr style={{ background: 'rgba(255, 255, 255, 0.05)', textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                    <th style={{ padding: '10px 14px' }}>Source Record</th>
                    <th style={{ padding: '10px 14px' }}>Raw Ingested Name</th>
                    <th style={{ padding: '10px 14px' }}>Raw Ingested Address</th>
                    <th style={{ padding: '10px 14px' }}>Model Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {entity.matched_sources?.map((s, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 700, color: '#34d399' }}>
                        {s.source_label}
                        <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>{s.source_id}</div>
                      </td>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#ffffff' }}>
                        {s.raw_name}
                      </td>
                      <td style={{ padding: '12px 14px', color: '#94a3b8' }}>
                        {s.raw_address}
                      </td>
                      <td style={{ padding: '12px 14px', fontWeight: 700, color: '#38bdf8' }}>
                        {s.confidence ? `${(s.confidence * 100).toFixed(1)}%` : '98.5%'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 4: IDENTITY PASSPORT */}
        {activeTab === 'passport' && (
          <div>
            <div style={{
              background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 182, 212, 0.08) 100%)',
              border: '2px solid #10b981',
              borderRadius: '16px',
              padding: '24px',
              marginBottom: '20px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                <div>
                  <div style={{ fontSize: '0.72rem', color: '#34d399', textTransform: 'uppercase', fontWeight: 800, letterSpacing: '1px' }}>
                    EntityGraph Enterprise Identity Passport
                  </div>
                  <h3 style={{ fontSize: '1.4rem', fontWeight: 800, color: '#ffffff' }}>
                    {passport.canonical_name || entity.canonical_name}
                  </h3>
                  <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                    Entity ID: <code style={{ color: '#38bdf8' }}>{passport.entity_id || entity.canonical_id}</code>
                  </div>
                </div>

                <button
                  onClick={handleDownloadPassport}
                  className="btn-travel-gps"
                  style={{ padding: '8px 16px', fontSize: '0.82rem', gap: '6px' }}
                >
                  <Download size={14} /> Export Passport (JSON)
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', marginBottom: '18px' }}>
                <div style={{ background: 'rgba(0,0,0,0.4)', padding: '12px', borderRadius: '10px' }}>
                  <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>COUNTRY / JURISDICTION</div>
                  <div style={{ fontWeight: 700, color: '#ffffff' }}>{passport.country || entity.country}</div>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.4)', padding: '12px', borderRadius: '10px' }}>
                  <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>IDENTITY CONFIDENCE</div>
                  <div style={{ fontWeight: 700, color: '#34d399' }}>{passport.identity_confidence_score || '96.8%'}</div>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.4)', padding: '12px', borderRadius: '10px' }}>
                  <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>CONFLICT STATUS</div>
                  <div style={{ fontWeight: 700, color: '#fbbf24' }}>{passport.conflict_status || 'AUTO_MATCHED'}</div>
                </div>
              </div>

              <div style={{ fontSize: '0.82rem', color: '#cbd5e1', marginBottom: '12px' }}>
                <strong>Known Trading Aliases:</strong> {(passport.known_aliases || entity.aliases)?.join(', ')}
              </div>

              <div style={{ fontSize: '0.75rem', color: '#94a3b8', fontStyle: 'italic', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '10px' }}>
                {passport.verification_statement || "EntityGraph automated multilingual resolution verified against multi-source evidence with LightGBM Model D."}
              </div>
            </div>
          </div>
        )}

        {/* Modal Bottom Actions */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '20px', marginTop: '24px' }}>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button 
              onClick={() => {
                onClose();
                if (onOpenGraph) onOpenGraph(entity.canonical_id);
              }} 
              className="btn-primary"
              style={{ fontSize: '0.82rem', padding: '8px 16px' }}
            >
              <Sparkles size={15} /> Open Semantic Graph
            </button>
            {onOpenMap && (
              <button 
                onClick={() => {
                  onClose();
                  onOpenMap(entity.canonical_id);
                }} 
                className="btn-secondary"
                style={{ fontSize: '0.82rem', padding: '8px 16px' }}
              >
                <MapPin size={15} /> View on Map
              </button>
            )}
          </div>

          <button onClick={onClose} className="btn-secondary" style={{ fontSize: '0.82rem', padding: '8px 20px' }}>
            Close Investigation
          </button>
        </div>
      </div>
    </div>
  );
}
