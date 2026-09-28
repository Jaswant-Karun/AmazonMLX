import React, { useState } from 'react';
import { Terminal, Copy, Check, Code2, Globe, Database, ArrowRight } from 'lucide-react';

export default function DeveloperConsole() {
  const [copiedKey, setCopiedKey] = useState(null);
  const [selectedLang, setSelectedLang] = useState('curl'); // 'curl' | 'python' | 'javascript'

  const copyToClipboard = (text, key) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const SNIPPETS = {
    search: {
      title: "1. Multilingual Search & Entity Resolution",
      curl: `curl -X GET "http://127.0.0.1:8000/api/search?q=Jamnagar+shop+om+shoping&country=India"`,
      python: `import requests\n\nres = requests.get(\n    "http://127.0.0.1:8000/api/search",\n    params={"q": "Jamnagar shop om shoping", "country": "India"}\n)\nprint(res.json())`,
      javascript: `const res = await fetch("http://127.0.0.1:8000/api/search?q=Jamnagar+shop+om+shoping&country=India");\nconst data = await res.json();\nconsole.log(data);`
    },
    passport: {
      title: "2. Fetch Verified Business Identity Passport",
      curl: `curl -X GET "http://127.0.0.1:8000/api/passport/S1-138436105"`,
      python: `import requests\n\npassport = requests.get("http://127.0.0.1:8000/api/passport/S1-138436105").json()\nprint("Confidence:", passport["identity_confidence_score"])`,
      javascript: `const res = await fetch("http://127.0.0.1:8000/api/passport/S1-138436105");\nconst passport = await res.json();\nconsole.log(passport);`
    },
    batch: {
      title: "3. Enterprise Batch Entity Resolution",
      curl: `curl -X POST "http://127.0.0.1:8000/api/batch-resolve" \\\n  -H "Content-Type: application/json" \\\n  -d '{"records": [{"name": "Jamnagar Producers", "address": "Bedi Gate"}]}'`,
      python: `import requests\n\npayload = {\n    "records": [\n        {"id": "V-01", "name": "Jamnagar Producers", "address": "Bedi Gate Jamnagar"},\n        {"id": "V-02", "name": "Sweet Book Store Birch", "address": "Birch St"}\n    ]\n}\nres = requests.post("http://127.0.0.1:8000/api/batch-resolve", json=payload)\nprint(res.json()["summary"])`,
      javascript: `const payload = {\n  records: [{ id: "V-01", name: "Jamnagar Producers", address: "Bedi Gate" }]\n};\nconst res = await fetch("http://127.0.0.1:8000/api/batch-resolve", {\n  method: "POST",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify(payload)\n});\nconsole.log(await res.json());`
    },
    review: {
      title: "4. Human Review Queue & Audit Logging",
      curl: `curl -X POST "http://127.0.0.1:8000/api/review-action" \\\n  -H "Content-Type: application/json" \\\n  -d '{"canonical_id": "S1-138436105", "decision": "CONFIRMED", "reviewer": "Auditor"}'`,
      python: `import requests\n\nres = requests.post(\n    "http://127.0.0.1:8000/api/review-action",\n    json={"canonical_id": "S1-138436105", "decision": "CONFIRMED"}\n)\nprint(res.json())`,
      javascript: `const res = await fetch("http://127.0.0.1:8000/api/review-action", {\n  method: "POST",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify({ canonical_id: "S1-138436105", decision: "CONFIRMED" })\n});\nconsole.log(await res.json());`
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '32px', marginBottom: '40px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
            <span className="badge-verified">
              <Terminal size={14} /> Developer API Console
            </span>
            <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
              REST API Integration & Verification SDK
            </span>
          </div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: '#ffffff' }}>
            Interactive REST API Reference & Code Snippets
          </h2>
        </div>

        {/* Language Tabs */}
        <div style={{ display: 'flex', background: 'rgba(255,255,255,0.04)', borderRadius: '10px', padding: '3px', border: '1px solid rgba(255,255,255,0.08)' }}>
          {[
            { id: 'curl', label: 'cURL' },
            { id: 'python', label: 'Python (requests)' },
            { id: 'javascript', label: 'JavaScript (fetch)' }
          ].map(lang => (
            <button
              key={lang.id}
              onClick={() => setSelectedLang(lang.id)}
              style={{
                background: selectedLang === lang.id ? '#10b981' : 'transparent',
                color: selectedLang === lang.id ? '#ffffff' : '#94a3b8',
                border: 'none',
                borderRadius: '6px',
                padding: '6px 14px',
                fontSize: '0.78rem',
                fontWeight: 700,
                cursor: 'pointer'
              }}
            >
              {lang.label}
            </button>
          ))}
        </div>
      </div>

      {/* Snippets Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {Object.entries(SNIPPETS).map(([key, item]) => {
          const code = item[selectedLang];
          const isCopied = copiedKey === key;

          return (
            <div key={key} style={{
              background: 'rgba(10, 14, 26, 0.95)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: '14px',
              padding: '18px',
              position: 'relative'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <span style={{ fontSize: '0.84rem', fontWeight: 700, color: '#34d399' }}>
                  {item.title}
                </span>
                <button
                  onClick={() => copyToClipboard(code, key)}
                  style={{
                    background: isCopied ? 'rgba(16,185,129,0.2)' : 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: isCopied ? '#34d399' : '#94a3b8',
                    borderRadius: '6px',
                    padding: '4px 8px',
                    fontSize: '0.72rem',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}
                >
                  {isCopied ? <><Check size={12} /> Copied</> : <><Copy size={12} /> Copy</>}
                </button>
              </div>

              <pre style={{
                background: 'rgba(0,0,0,0.5)',
                padding: '12px',
                borderRadius: '8px',
                color: '#e2e8f0',
                fontSize: '0.76rem',
                fontFamily: 'monospace',
                overflowX: 'auto',
                margin: 0,
                lineHeight: 1.45
              }}>
                <code>{code}</code>
              </pre>
            </div>
          );
        })}
      </div>
    </div>
  );
}
