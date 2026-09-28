import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import InteractiveGraph from './components/InteractiveGraph';
import InteractiveMap from './components/InteractiveMap';
import EntityDetailModal from './components/EntityDetailModal';
import { 
  Search, Sparkles, MapPin, Building2, ShieldCheck, Layers, 
  ArrowRight, Compass, RefreshCw, Star, Tag, Award, Globe, Database,
  SlidersHorizontal, CheckCircle2, ChevronRight, Zap
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCountry, setSelectedCountry] = useState('All');
  const [searchResults, setSearchResults] = useState([]);
  const [parsedQuery, setParsedQuery] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeEntityId, setActiveEntityId] = useState(null);
  const [activeGraphData, setActiveGraphData] = useState(null);
  const [selectedEntityForModal, setSelectedEntityForModal] = useState(null);
  const [demoQueries, setDemoQueries] = useState([]);
  const [platformStats, setPlatformStats] = useState(null);
  const [viewMode, setViewMode] = useState('split'); // 'split' | 'graph'

  // Fetch initial demos and platform stats
  useEffect(() => {
    fetch(`${API_BASE}/api/demo-queries`)
      .then(r => r.json())
      .then(data => setDemoQueries(data))
      .catch(e => console.error("Could not load demos:", e));

    fetch(`${API_BASE}/api/stats`)
      .then(r => r.json())
      .then(data => setPlatformStats(data))
      .catch(e => console.error("Could not load stats:", e));

    // Pre-populate with a rich default search
    handleSearch("near PSG college tea shop", "India");
  }, []);

  const handleSearch = async (queryText = searchQuery, country = selectedCountry) => {
    if (!queryText.trim()) return;
    setIsLoading(true);
    setSearchQuery(queryText);

    try {
      const countryParam = country && country !== 'All' ? `&country=${encodeURIComponent(country)}` : '';
      const res = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(queryText)}${countryParam}`);
      const data = await res.json();

      setSearchResults(data.results || []);
      setParsedQuery(data.query_parsed || null);

      if (data.results && data.results.length > 0) {
        const topEntityId = data.results[0].canonical_id;
        setActiveEntityId(topEntityId);
        loadGraph(topEntityId);
      } else {
        setActiveEntityId(null);
        setActiveGraphData(null);
      }
    } catch (err) {
      console.error("Search failed:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadGraph = async (canonicalId) => {
    try {
      const res = await fetch(`${API_BASE}/api/graph/${canonicalId}`);
      const data = await res.json();
      setActiveGraphData(data);
    } catch (e) {
      console.error("Failed to load graph:", e);
    }
  };

  const openEntityDetails = async (canonicalId) => {
    try {
      const res = await fetch(`${API_BASE}/api/entity/${canonicalId}`);
      const data = await res.json();
      setSelectedEntityForModal(data);
    } catch (e) {
      console.error("Failed to load entity details:", e);
    }
  };

  const handleSelectEntity = (canonicalId) => {
    setActiveEntityId(canonicalId);
    loadGraph(canonicalId);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navigation */}
      <Navbar 
        onReset={() => handleSearch("near PSG college tea shop", "India")} 
        stats={platformStats}
      />

      {/* Main Container */}
      <main style={{ maxWidth: '1440px', margin: '0 auto', padding: '0 24px', width: '100%', flex: 1 }}>
        
        {/* Search Hero Section */}
        <section style={{ textAlign: 'center', margin: '30px 0 24px 0' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <span className="badge-verified">
              <Zap size={13} /> Zero-Shot Multilingual Resolution
            </span>
            <span className="badge-gold">
              <Award size={13} /> Model D (46 Features)
            </span>
          </div>

          <h1 style={{ fontSize: '2.5rem', fontWeight: 800, marginBottom: '8px', letterSpacing: '-0.03em' }}>
            Discover Any Business in <span className="gradient-text-vibrant">Any Language</span>
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '1rem', maxWidth: '680px', margin: '0 auto 24px auto' }}>
            Search colloquial landmarks, regional Tamil/Hindi scripts, and misspelled names. 
            EntityGraph unifies multi-source fragments into verified Golden Records.
          </p>

          {/* Omnibar Input */}
          <div style={{
            maxWidth: '760px',
            margin: '0 auto',
            position: 'relative',
            display: 'flex',
            alignItems: 'center',
            background: 'var(--bg-elevated)',
            border: '2px solid rgba(99, 102, 241, 0.4)',
            borderRadius: '16px',
            padding: '6px 8px 6px 20px',
            boxShadow: '0 10px 40px rgba(99, 102, 241, 0.22)',
            transition: 'all 0.2s ease'
          }}>
            <Search size={22} color="#818cf8" style={{ marginRight: '12px', flexShrink: 0 }} />
            <input 
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search e.g. 'near PSG college tea shop', 'கல்லூரி பக்கத்துல நல்ல சாப்பாடு'..."
              style={{
                background: 'transparent',
                border: 'none',
                outline: 'none',
                color: '#ffffff',
                fontSize: '1.05rem',
                width: '100%',
                fontWeight: 500
              }}
            />

            {isLoading && (
              <div style={{ marginRight: '12px', animation: 'spin 1s linear infinite' }}>
                <RefreshCw size={18} color="#818cf8" />
              </div>
            )}

            <button 
              onClick={() => handleSearch()}
              className="btn-primary"
              style={{ flexShrink: 0, padding: '10px 24px', fontSize: '0.95rem' }}
            >
              Search
            </button>
          </div>

          {/* Demo Query Pills */}
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'center',
            gap: '8px',
            maxWidth: '900px',
            margin: '18px auto 0 auto'
          }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Sparkles size={13} color="#f59e0b" /> Try Samples:
            </span>
            {demoQueries.map(demo => (
              <button
                key={demo.id}
                onClick={() => handleSearch(demo.query, demo.language.includes('France') ? 'France' : (demo.language.includes('US') ? 'United States' : 'India'))}
                style={{
                  background: searchQuery === demo.query ? 'rgba(99, 102, 241, 0.3)' : 'rgba(255, 255, 255, 0.04)',
                  border: searchQuery === demo.query ? '1px solid #818cf8' : '1px solid var(--border-subtle)',
                  borderRadius: '9999px',
                  color: searchQuery === demo.query ? '#ffffff' : '#cbd5e1',
                  fontSize: '0.75rem',
                  padding: '4px 12px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  transition: 'all 0.15s ease'
                }}
              >
                <span>{demo.flag}</span>
                <span style={{ fontWeight: 600 }}>{demo.query}</span>
              </button>
            ))}
          </div>

          {/* Parsed NLP Explanation Card */}
          {parsedQuery && (
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '16px',
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid rgba(56, 189, 248, 0.3)',
              borderRadius: '12px',
              padding: '8px 18px',
              margin: '18px auto 0 auto',
              fontSize: '0.8rem',
              color: '#94a3b8'
            }}>
              <div>
                <strong style={{ color: '#38bdf8' }}>Detected Script:</strong> {parsedQuery.script.toUpperCase()}
              </div>
              {parsedQuery.landmark && (
                <div>
                  <strong style={{ color: '#c084fc' }}>Extracted Landmark:</strong> "{parsedQuery.landmark}"
                </div>
              )}
              {parsedQuery.business_name && (
                <div>
                  <strong style={{ color: '#34d399' }}>Target Business:</strong> "{parsedQuery.business_name}"
                </div>
              )}
              <div>
                <strong style={{ color: '#fbbf24' }}>Category:</strong> {parsedQuery.inferred_category.replace('_', ' ').toUpperCase()}
              </div>
            </div>
          )}
        </section>

        {/* View Switcher Controls */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px',
          borderBottom: '1px solid var(--border-subtle)',
          paddingBottom: '12px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontWeight: 700, fontSize: '1.1rem' }}>Search Results</span>
            <span style={{ background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: '6px', fontSize: '0.8rem', color: '#94a3b8' }}>
              {searchResults.length} Verified Clusters
            </span>
          </div>

          <div style={{ display: 'flex', gap: '8px', background: 'rgba(255,255,255,0.03)', padding: '4px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
            <button
              onClick={() => setViewMode('split')}
              style={{
                background: viewMode === 'split' ? '#4f46e5' : 'transparent',
                color: viewMode === 'split' ? '#ffffff' : '#94a3b8',
                border: 'none',
                borderRadius: '8px',
                padding: '6px 14px',
                fontSize: '0.8rem',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <Compass size={14} /> Split (Results + Map)
            </button>

            <button
              onClick={() => setViewMode('graph')}
              style={{
                background: viewMode === 'graph' ? '#4f46e5' : 'transparent',
                color: viewMode === 'graph' ? '#ffffff' : '#94a3b8',
                border: 'none',
                borderRadius: '8px',
                padding: '6px 14px',
                fontSize: '0.8rem',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <Layers size={14} /> Fullscreen Entity Graph
            </button>
          </div>
        </div>

        {/* View Mode 1: Fullscreen Entity Graph */}
        {viewMode === 'graph' && activeGraphData && (
          <div style={{ marginBottom: '40px' }}>
            <InteractiveGraph 
              graphData={activeGraphData} 
              onClose={() => setViewMode('split')}
            />
          </div>
        )}

        {/* View Mode 2: Split View (Results on Left, Map & Mini-Graph on Right) */}
        {viewMode === 'split' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(12, 1fr)',
            gap: '24px',
            marginBottom: '40px',
            alignItems: 'start'
          }}>
            {/* Left Column: Business Cards List */}
            <div style={{ gridColumn: 'span 5', display: 'flex', flexDirection: 'column', gap: '14px' }}>
              {searchResults.length === 0 ? (
                <div className="glass-panel" style={{ padding: '36px', textAlign: 'center', color: '#94a3b8' }}>
                  No matching business entities found. Try one of the sample queries above!
                </div>
              ) : (
                searchResults.map((entity, idx) => {
                  const isSelected = entity.canonical_id === activeEntityId;
                  return (
                    <div 
                      key={entity.canonical_id}
                      onClick={() => handleSelectEntity(entity.canonical_id)}
                      className="glass-panel"
                      style={{
                        padding: '18px',
                        cursor: 'pointer',
                        borderColor: isSelected ? '#818cf8' : 'var(--border-subtle)',
                        background: isSelected ? 'rgba(30, 37, 60, 0.85)' : 'var(--bg-card)',
                        boxShadow: isSelected ? '0 0 25px rgba(99, 102, 241, 0.25)' : 'none'
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '6px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{
                            width: '24px',
                            height: '24px',
                            borderRadius: '50%',
                            background: isSelected ? '#4f46e5' : 'rgba(255,255,255,0.06)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '0.75rem',
                            fontWeight: 700,
                            color: '#ffffff'
                          }}>
                            {idx + 1}
                          </span>
                          <span className="badge-verified">{entity.confidence_pct}% Consensus</span>
                        </div>
                        <span style={{ fontSize: '0.75rem', color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 600 }}>
                          <Star size={13} fill="#fbbf24" /> {entity.rating} ({entity.review_count})
                        </span>
                      </div>

                      <h3 style={{ fontSize: '1.15rem', color: '#ffffff', marginBottom: '6px' }}>
                        {entity.canonical_name}
                      </h3>

                      <div style={{ fontSize: '0.8rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '10px' }}>
                        <MapPin size={14} color="#38bdf8" style={{ flexShrink: 0 }} />
                        <span>{entity.golden_address}</span>
                      </div>

                      {entity.landmark_match && (
                        <div style={{
                          background: 'rgba(192, 132, 252, 0.12)',
                          border: '1px solid rgba(192, 132, 252, 0.3)',
                          padding: '4px 10px',
                          borderRadius: '8px',
                          fontSize: '0.75rem',
                          color: '#e9d5ff',
                          marginBottom: '10px',
                          fontWeight: 500
                        }}>
                          🎯 {entity.landmark_match}
                        </div>
                      )}

                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '10px' }}>
                        <div style={{ display: 'flex', gap: '6px' }}>
                          {entity.matched_sources_summary?.slice(0, 2).map((s, i) => (
                            <span key={i} className="badge-source">{s.split(' ')[0]} {s.split(' ')[1]}</span>
                          ))}
                          {entity.source_count > 2 && (
                            <span className="badge-source">+{entity.source_count - 2} more</span>
                          )}
                        </div>

                        <div style={{ display: 'flex', gap: '8px' }}>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              openEntityDetails(entity.canonical_id);
                            }}
                            style={{
                              background: 'transparent',
                              border: '1px solid rgba(255,255,255,0.15)',
                              color: '#cbd5e1',
                              borderRadius: '8px',
                              padding: '4px 10px',
                              fontSize: '0.75rem',
                              cursor: 'pointer'
                            }}
                          >
                            Audit Lineage
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSelectEntity(entity.canonical_id);
                              setViewMode('graph');
                            }}
                            className="btn-primary"
                            style={{ padding: '4px 10px', fontSize: '0.75rem', borderRadius: '8px' }}
                          >
                            Graph
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            {/* Right Column: Interactive Map & Entity Graph Preview */}
            <div style={{ gridColumn: 'span 7', display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <InteractiveMap 
                results={searchResults}
                activeEntityId={activeEntityId}
                onSelectEntity={handleSelectEntity}
                parsedQuery={parsedQuery}
              />

              {activeGraphData && (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#818cf8', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Layers size={15} /> Entity Graph Topology Preview
                    </span>
                    <button 
                      onClick={() => setViewMode('graph')}
                      style={{ background: 'none', border: 'none', color: '#38bdf8', fontSize: '0.78rem', cursor: 'pointer', fontWeight: 600 }}
                    >
                      Open Fullscreen Canvas →
                    </button>
                  </div>
                  <InteractiveGraph graphData={activeGraphData} />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Benchmark & Platform Stats Footer Card */}
        {platformStats && (
          <section className="glass-panel" style={{ padding: '24px', margin: '40px 0 60px 0' }}>
            <div style={{ textAlign: 'center', marginBottom: '20px' }}>
              <h3 style={{ fontSize: '1.35rem', marginBottom: '4px' }}>
                EntityGraph Engine Benchmarks
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                Powered by LightGBM Model D (46 Pairwise Features) with Dynamic Agreement Decision Rule
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#34d399' }}>{platformStats.precision_macro_f05}</div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Official Macro F0.5 Score</div>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#818cf8' }}>{platformStats.pairwise_precision}</div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Pairwise Model Precision</div>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#fbbf24' }}>{platformStats.singleton_fp_rate}</div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Singleton False Positive Rate</div>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#38bdf8' }}>{platformStats.average_query_latency_ms} ms</div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Average Search Latency</div>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Entity Details Modal */}
      {selectedEntityForModal && (
        <EntityDetailModal 
          entity={selectedEntityForModal}
          onClose={() => setSelectedEntityForModal(null)}
          onOpenGraph={(id) => {
            handleSelectEntity(id);
            setViewMode('graph');
          }}
        />
      )}
    </div>
  );
}
