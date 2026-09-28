import React, { useState } from 'react';
import { 
  FileSpreadsheet, Upload, Download, Sparkles, CheckCircle2, 
  AlertTriangle, PlusCircle, RefreshCw, Eye, ArrowRight, ShieldCheck
} from 'lucide-react';

export default function BatchResolver({ onOpenEntityDetails }) {
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [batchResults, setBatchResults] = useState(null);

  const handleLoadSampleBatch = async () => {
    setIsProcessing(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/batch-sample');
      const sampleRecords = await res.json();
      setInputText(JSON.stringify(sampleRecords, null, 2));

      // Automatically execute resolution
      const resolveRes = await fetch('http://127.0.0.1:8000/api/batch-resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ records: sampleRecords })
      });
      const data = await resolveRes.json();
      setBatchResults(data);
    } catch (e) {
      console.error("Batch resolution failed:", e);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunBatchResolution = async () => {
    if (!inputText.trim()) return;
    setIsProcessing(true);

    try {
      let records = [];
      try {
        records = JSON.parse(inputText);
      } catch (err) {
        // Fallback: parse CSV lines (name, address, country)
        const lines = inputText.split('\n').map(l => l.trim()).filter(Boolean);
        records = lines.map((line, idx) => {
          const parts = line.split(',');
          return {
            id: `ROW-${idx + 101}`,
            name: parts[0]?.trim() || `Record ${idx + 1}`,
            address: parts[1]?.trim() || '',
            country: parts[2]?.trim() || 'India'
          };
        });
      }

      const res = await fetch('http://127.0.0.1:8000/api/batch-resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ records })
      });
      const data = await res.json();
      setBatchResults(data);
    } catch (e) {
      console.error("Batch execution failed:", e);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleExportCsv = () => {
    if (!batchResults || !batchResults.resolved_records) return;

    const headers = ["Input ID", "Raw Name", "Raw Address", "Resolved Golden ID", "Resolved Golden Name", "Golden Address", "Confidence", "Status", "Audit Note"];
    const rows = batchResults.resolved_records.map(r => [
      `"${r.input_id}"`,
      `"${r.raw_name}"`,
      `"${r.raw_address}"`,
      `"${r.resolved_canonical_id}"`,
      `"${r.resolved_golden_name}"`,
      `"${r.golden_address}"`,
      `"${r.confidence_pct}%"`,
      `"${r.status}"`,
      `"${r.conflict_reason}"`
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `EntityGraph_Resolved_Master_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <div className="glass-panel" style={{ padding: '32px', marginBottom: '40px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
            <span className="badge-verified">
              <ShieldCheck size={14} /> Enterprise Batch Processor
            </span>
            <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
              Multi-Source Ingestion, Deduplication & Golden Master Generation
            </span>
          </div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: '#ffffff' }}>
            Batch Entity Resolution & Golden Master Ingestion
          </h2>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={handleLoadSampleBatch}
            disabled={isProcessing}
            className="btn-secondary"
            style={{ padding: '8px 16px', fontSize: '0.8rem', gap: '6px' }}
          >
            <Sparkles size={14} color="#34d399" /> Load Sample Messy Batch
          </button>
          {batchResults && (
            <button
              onClick={handleExportCsv}
              className="btn-travel-gps"
              style={{ padding: '8px 16px', fontSize: '0.8rem', gap: '6px' }}
            >
              <Download size={14} /> Export Resolved CSV
            </button>
          )}
        </div>
      </div>

      {/* Input Textarea Area */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <label style={{ fontSize: '0.82rem', fontWeight: 700, color: '#cbd5e1' }}>
            Paste Vendor Records (JSON Array or CSV lines: <code>name, address, country</code>):
          </label>
          <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
            Supports cross-source noise, abbreviations, and missing fields
          </span>
        </div>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder={`[\n  {\n    "id": "VEND-101",\n    "name": "Jamnagar Producers",\n    "address": "Bedi Gate Jamnagar",\n    "country": "India"\n  }\n]`}
          rows={6}
          style={{
            width: '100%',
            background: 'rgba(10, 14, 26, 0.95)',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            borderRadius: '12px',
            color: '#34d399',
            fontFamily: 'monospace',
            fontSize: '0.85rem',
            padding: '14px',
            outline: 'none',
            resize: 'vertical'
          }}
        />

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '10px' }}>
          <button
            onClick={handleRunBatchResolution}
            disabled={isProcessing || !inputText.trim()}
            className="btn-primary"
            style={{ padding: '8px 20px', fontSize: '0.85rem', gap: '8px' }}
          >
            {isProcessing ? <RefreshCw size={15} className="spin" /> : <Upload size={15} />}
            {isProcessing ? 'Resolving Batch...' : 'Execute Batch Resolution'}
          </button>
        </div>
      </div>

      {/* Summary Scorecard Cards */}
      {batchResults && batchResults.summary && (
        <div style={{ marginBottom: '28px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)', textAlign: 'center' }}>
              <div style={{ fontSize: '1.6rem', fontWeight: 900, color: '#ffffff' }}>
                {batchResults.summary.total_ingested}
              </div>
              <div style={{ fontSize: '0.74rem', color: '#94a3b8', marginTop: '4px' }}>Total Ingested Records</div>
            </div>

            <div style={{ background: 'rgba(16,185,129,0.08)', padding: '16px', borderRadius: '14px', border: '1px solid rgba(16,185,129,0.3)', textAlign: 'center' }}>
              <div style={{ fontSize: '1.6rem', fontWeight: 900, color: '#34d399' }}>
                {batchResults.summary.auto_resolved_golden}
              </div>
              <div style={{ fontSize: '0.74rem', color: '#34d399', marginTop: '4px', fontWeight: 600 }}>Resolved to Golden Profiles</div>
            </div>

            <div style={{ background: 'rgba(245,158,11,0.08)', padding: '16px', borderRadius: '14px', border: '1px solid rgba(245,158,11,0.3)', textAlign: 'center' }}>
              <div style={{ fontSize: '1.6rem', fontWeight: 900, color: '#fbbf24' }}>
                {batchResults.summary.conflicts_flagged}
              </div>
              <div style={{ fontSize: '0.74rem', color: '#fbbf24', marginTop: '4px', fontWeight: 600 }}>Conflict Radar Flags</div>
            </div>

            <div style={{ background: 'rgba(56,189,248,0.08)', padding: '16px', borderRadius: '14px', border: '1px solid rgba(56,189,248,0.3)', textAlign: 'center' }}>
              <div style={{ fontSize: '1.6rem', fontWeight: 900, color: '#38bdf8' }}>
                {batchResults.summary.new_clusters_created}
              </div>
              <div style={{ fontSize: '0.74rem', color: '#38bdf8', marginTop: '4px', fontWeight: 600 }}>New Identity Clusters</div>
            </div>
          </div>
        </div>
      )}

      {/* Detailed Batch Mapping Table */}
      {batchResults && batchResults.resolved_records && (
        <div>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#ffffff', marginBottom: '14px' }}>
            Resolved Entity Concordance Mapping
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
              <thead>
                <tr style={{ background: 'rgba(255, 255, 255, 0.05)', textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  <th style={{ padding: '10px 14px' }}>Input Record</th>
                  <th style={{ padding: '10px 14px' }}>Resolved Golden Entity</th>
                  <th style={{ padding: '10px 14px' }}>Synthesized Address</th>
                  <th style={{ padding: '10px 14px' }}>Confidence</th>
                  <th style={{ padding: '10px 14px' }}>Resolution Status</th>
                  <th style={{ padding: '10px 14px', textAlign: 'right' }}>Audit</th>
                </tr>
              </thead>
              <tbody>
                {batchResults.resolved_records.map((r, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <td style={{ padding: '12px 14px' }}>
                      <div style={{ fontWeight: 700, color: '#ffffff' }}>{r.raw_name}</div>
                      <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                        ID: {r.input_id} • 📍 {r.raw_address || 'N/A'}
                      </div>
                    </td>

                    <td style={{ padding: '12px 14px' }}>
                      <div style={{ fontWeight: 700, color: '#34d399' }}>{r.resolved_golden_name}</div>
                      <div style={{ fontSize: '0.72rem', color: '#38bdf8' }}>
                        <code>{r.resolved_canonical_id}</code>
                      </div>
                    </td>

                    <td style={{ padding: '12px 14px', color: '#cbd5e1', maxWidth: '240px' }}>
                      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {r.golden_address}
                      </div>
                    </td>

                    <td style={{ padding: '12px 14px', fontWeight: 800, color: r.confidence_pct >= 75 ? '#34d399' : '#fbbf24' }}>
                      {r.confidence_pct}%
                    </td>

                    <td style={{ padding: '12px 14px' }}>
                      {r.status === 'RESOLVED_GOLDEN' ? (
                        <span style={{ background: 'rgba(16,185,129,0.15)', color: '#34d399', border: '1px solid #10b981', padding: '2px 8px', borderRadius: '4px', fontSize: '0.72rem', fontWeight: 700 }}>
                          ✓ Golden Match
                        </span>
                      ) : r.status === 'NEEDS_REVIEW' ? (
                        <span style={{ background: 'rgba(245,158,11,0.15)', color: '#fbbf24', border: '1px solid #f59e0b', padding: '2px 8px', borderRadius: '4px', fontSize: '0.72rem', fontWeight: 700 }}>
                          ⚠ Conflict: Review
                        </span>
                      ) : (
                        <span style={{ background: 'rgba(56,189,248,0.15)', color: '#38bdf8', border: '1px solid #38bdf8', padding: '2px 8px', borderRadius: '4px', fontSize: '0.72rem', fontWeight: 700 }}>
                          + New Cluster
                        </span>
                      )}
                      <div style={{ fontSize: '0.68rem', color: '#94a3b8', marginTop: '2px' }}>
                        {r.conflict_reason}
                      </div>
                    </td>

                    <td style={{ padding: '12px 14px', textAlign: 'right' }}>
                      {r.resolved_canonical_id.startsWith('S1-') && (
                        <button
                          onClick={() => onOpenEntityDetails(r.resolved_canonical_id)}
                          className="btn-secondary"
                          style={{ padding: '4px 10px', fontSize: '0.72rem' }}
                        >
                          Inspect
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
