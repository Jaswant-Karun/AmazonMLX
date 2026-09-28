import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar';
import InteractiveGraph from './components/InteractiveGraph';
import InteractiveMap from './components/InteractiveMap';
import EntityDetailModal from './components/EntityDetailModal';
import { 
  Search, Sparkles, MapPin, Building2, ShieldCheck, Layers, 
  ArrowRight, Compass, RefreshCw, Star, Tag, Award, Globe, Database,
  SlidersHorizontal, CheckCircle2, ChevronRight, Zap, Navigation,
  Car, Eye, ExternalLink, LocateFixed, Store, Utensils, HeartPulse, Laptop
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
  const [viewMode, setViewMode] = useState('split'); // 'split' | 'gps' | 'graph'
  const [gpsDestinationId, setGpsDestinationId] = useState(null);
  const [isGpsTravelActive, setIsGpsTravelActive] = useState(false);

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

    // Pre-populate with a real dataset search
    handleSearch("Jamnagar shop om shoping", "India");
  }, []);

  const handleSearch = async (queryText = searchQuery, country = selectedCountry) => {
    if (!queryText.trim()) return;
    setIsLoading(true);
    setSearchQuery(queryText);
    setSelectedCountry(country);

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

  // Immediate 1-Click GPS Travel Trigger
  const handleTriggerTravel = (canonicalId) => {
    setActiveEntityId(canonicalId);
    setGpsDestinationId(canonicalId);
    setIsGpsTravelActive(true);
  };

  // Active selected entity object
  const activeEntity = searchResults.find(r => r.canonical_id === (gpsDestinationId || activeEntityId)) || searchResults[0];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Luxury Navigation */}
      <Navbar 
        onReset={() => handleSearch("Jamnagar shop om shoping", "India")} 
        stats={platformStats}
      />

      {/* Main Container */}
      <main style={{ maxWidth: '1440px', margin: '0 auto', padding: '0 24px', width: '100%', flex: 1 }}>
        
        {/* Search Hero Section */}
        <section style={{ textAlign: 'center', margin: '24px 0 20px 0' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <span className="badge-verified">
              <Zap size={13} /> Zero-Shot Multilingual Resolution
            </span>
            <span className="badge-gold">
              <Award size={13} /> Model D (46 Topological Features)
            </span>
          </div>

          <h1 style={{ fontSize: '2.6rem', fontWeight: 900, marginBottom: '8px', letterSpacing: '-0.03em' }}>
            Explore Verified Real Places with <span className="gradient-text-emerald">Live GPS Travel</span>
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '1.02rem', maxWidth: '720px', margin: '0 auto 20px auto', lineHeight: 1.5 }}>
            Search colloquial landmarks, regional Tamil/Hindi scripts, and misspelled shop names. 
            EntityGraph unifies multi-source fragments into verified Golden Records and routes your journey.
          </p>

          {/* Omnibar Input */}
          <div style={{
            maxWidth: '820px',
            margin: '0 auto',
            position: 'relative',
            display: 'flex',
            alignItems: 'center',
            background: 'rgba(14, 20, 36, 0.95)',
            backdropFilter: 'blur(20px)',
            border: '2px solid rgba(16, 185, 129, 0.4)',
            borderRadius: '18px',
            padding: '6px 8px 6px 18px',
            boxShadow: '0 12px 40px rgba(16, 185, 129, 0.22)',
            transition: 'all 0.2s ease'
          }}>
            <Search size={22} color="#34d399" style={{ marginRight: '12px', flexShrink: 0 }} />
            <input 
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search shops, tea stalls, medicals, houses... e.g. 'Jamnagar om shop', 'Coimbatore hotel near PSG'"
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
                <RefreshCw size={18} color="#34d399" />
              </div>
            )}

            {/* Country Selector Inside Search Bar */}
            <select
              value={selectedCountry}
              onChange={(e) => {
                setSelectedCountry(e.target.value);
                handleSearch(searchQuery, e.target.value);
              }}
              style={{
                background: 'rgba(255, 255, 255, 0.06)',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                borderRadius: '10px',
                color: '#cbd5e1',
                fontSize: '0.8rem',
                padding: '8px 12px',
                marginRight: '8px',
                cursor: 'pointer',
                outline: 'none'
              }}
            >
              <option value="All" style={{ background: '#0b0f19', color: '#fff' }}>🌍 All Regions</option>
              <option value="India" style={{ background: '#0b0f19', color: '#fff' }}>🇮🇳 India</option>
              <option value="France" style={{ background: '#0b0f19', color: '#fff' }}>🇫🇷 France</option>
              <option value="United States" style={{ background: '#0b0f19', color: '#fff' }}>🇺🇸 USA</option>
            </select>

            <button 
              onClick={() => handleSearch()}
              className="btn-travel-gps"
              style={{ flexShrink: 0, padding: '10px 22px', fontSize: '0.92rem' }}
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
            maxWidth: '960px',
            margin: '16px auto 0 auto'
          }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Sparkles size={13} color="#f59e0b" /> Try Dataset Queries:
            </span>
            {demoQueries.map(demo => (
              <button
                key={demo.id}
                onClick={() => handleSearch(demo.query, demo.language.includes('France') ? 'France' : (demo.language.includes('US') ? 'United States' : 'India'))}
                style={{
                  background: searchQuery === demo.query ? 'rgba(16, 185, 129, 0.25)' : 'rgba(255, 255, 255, 0.04)',
                  border: searchQuery === demo.query ? '1px solid #10b981' : '1px solid var(--border-subtle)',
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
              background: 'rgba(14, 20, 36, 0.85)',
              border: '1px solid rgba(56, 189, 248, 0.3)',
              borderRadius: '12px',
              padding: '8px 20px',
              margin: '14px auto 0 auto',
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
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontWeight: 800, fontSize: '1.2rem', color: '#ffffff' }}>Search Results</span>
            <span style={{ background: 'rgba(255,255,255,0.06)', padding: '3px 10px', borderRadius: '8px', fontSize: '0.8rem', color: '#34d399', fontWeight: 700 }}>
              {searchResults.length} Verified Real Places
            </span>
          </div>

          <div style={{ display: 'flex', gap: '8px', background: 'rgba(255,255,255,0.04)', padding: '4px', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
            <button
              onClick={() => setViewMode('split')}
              style={{
                background: viewMode === 'split' ? '#4f46e5' : 'transparent',
                color: viewMode === 'split' ? '#ffffff' : '#94a3b8',
                border: 'none',
                borderRadius: '8px',
                padding: '7px 16px',
                fontSize: '0.82rem',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <Compass size={15} /> Split View (List + GPS Map)
            </button>

            <button
              onClick={() => {
                setViewMode('gps');
                if (activeEntity) handleTriggerTravel(activeEntity.canonical_id);
              }}
              style={{
                background: viewMode === 'gps' ? 'linear-gradient(135deg, #059669 0%, #10b981 100%)' : 'transparent',
                color: viewMode === 'gps' ? '#ffffff' : '#94a3b8',
                border: 'none',
                borderRadius: '8px',
                padding: '7px 16px',
                fontSize: '0.82rem',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                boxShadow: viewMode === 'gps' ? '0 0 20px rgba(16,185,129,0.55)' : 'none'
              }}
            >
              <Navigation size={15} /> 🧭 Fullscreen GPS Navigator
            </button>

            <button
              onClick={() => setViewMode('graph')}
              style={{
                background: viewMode === 'graph' ? '#4f46e5' : 'transparent',
                color: viewMode === 'graph' ? '#ffffff' : '#94a3b8',
                border: 'none',
                borderRadius: '8px',
                padding: '7px 16px',
                fontSize: '0.82rem',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <Layers size={15} /> Fullscreen Graph
            </button>
          </div>
        </div>

        {/* View Mode 1: Fullscreen GPS Travel Navigator */}
        {viewMode === 'gps' && (
          <div style={{ marginBottom: '40px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div>
                <h2 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#ffffff', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Navigation size={20} color="#34d399" /> Turn-by-Turn GPS Travel Mode
                </h2>
                <p style={{ fontSize: '0.84rem', color: '#94a3b8' }}>
                  Live street-level GPS routing & vehicle navigation simulation to <strong style={{ color: '#34d399' }}>{activeEntity?.canonical_name}</strong>
                </p>
              </div>
              <button 
                onClick={() => setViewMode('split')}
                className="btn-secondary"
                style={{ padding: '8px 16px', fontSize: '0.82rem' }}
              >
                Exit Fullscreen Mode
              </button>
            </div>
            <div style={{ height: '700px' }}>
              <InteractiveMap 
                results={searchResults}
                activeEntityId={activeEntityId}
                onSelectEntity={handleSelectEntity}
                parsedQuery={parsedQuery}
                gpsDestinationId={gpsDestinationId}
                isGpsModeActive={true}
                onExitGps={() => {
                  setIsGpsTravelActive(false);
                  setGpsDestinationId(null);
                }}
                onOpenDetails={openEntityDetails}
              />
            </div>
          </div>
        )}

        {/* View Mode 2: Fullscreen Entity Graph */}
        {viewMode === 'graph' && activeGraphData && (
          <div style={{ marginBottom: '40px' }}>
            <InteractiveGraph 
              graphData={activeGraphData} 
              onClose={() => setViewMode('split')}
            />
          </div>
        )}

        {/* View Mode 3: Split View (Results on Left, Map & Mini-Graph on Right) */}
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
                <div className="glass-panel" style={{ padding: '40px', textAlign: 'center', color: '#94a3b8' }}>
                  No matching business entities found. Try one of the dataset sample queries above!
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
                        padding: '18px 20px',
                        cursor: 'pointer',
                        borderColor: isSelected ? '#10b981' : 'var(--border-subtle)',
                        background: isSelected ? 'rgba(16, 26, 46, 0.95)' : 'var(--bg-card)',
                        boxShadow: isSelected ? '0 0 30px rgba(16, 185, 129, 0.28)' : 'none',
                        transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)'
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{
                            width: '26px',
                            height: '26px',
                            borderRadius: '50%',
                            background: isSelected ? '#10b981' : 'rgba(255,255,255,0.06)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '0.75rem',
                            fontWeight: 800,
                            color: '#ffffff'
                          }}>
                            {idx + 1}
                          </span>
                          <span className="badge-verified">{entity.confidence_pct}% Consensus</span>
                        </div>
                        <span style={{ fontSize: '0.78rem', color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 700 }}>
                          <Star size={14} fill="#fbbf24" /> {entity.rating} ({entity.review_count})
                        </span>
                      </div>

                      <h3 style={{ fontSize: '1.2rem', color: '#ffffff', marginBottom: '6px', fontWeight: 800 }}>
                        {entity.canonical_name}
                      </h3>

                      <div style={{ fontSize: '0.82rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px' }}>
                        <MapPin size={15} color="#38bdf8" style={{ flexShrink: 0 }} />
                        <span>{entity.golden_address}</span>
                      </div>

                      {entity.landmark && (
                        <div style={{
                          background: 'rgba(192, 132, 252, 0.12)',
                          border: '1px solid rgba(192, 132, 252, 0.3)',
                          padding: '5px 12px',
                          borderRadius: '8px',
                          fontSize: '0.75rem',
                          color: '#e9d5ff',
                          marginBottom: '12px',
                          fontWeight: 600,
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '6px'
                        }}>
                          🎯 Landmark: {entity.landmark}
                        </div>
                      )}

                      {/* Card Bottom Actions */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '12px' }}>
                        <div style={{ display: 'flex', gap: '6px' }}>
                          {entity.matched_sources_summary?.slice(0, 2).map((s, i) => (
                            <span key={i} className="badge-source">{s.split(' ')[0]} {s.split(' ')[1]}</span>
                          ))}
                        </div>

                        <div style={{ display: 'flex', gap: '8px' }}>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleTriggerTravel(entity.canonical_id);
                            }}
                            className="btn-travel-gps"
                            style={{
                              padding: '5px 12px',
                              fontSize: '0.75rem',
                              fontWeight: 800
                            }}
                          >
                            <Navigation size={13} /> Travel GPS
                          </button>

                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              openEntityDetails(entity.canonical_id);
                            }}
                            style={{
                              background: 'rgba(255,255,255,0.05)',
                              border: '1px solid rgba(255,255,255,0.15)',
                              color: '#cbd5e1',
                              borderRadius: '8px',
                              padding: '5px 10px',
                              fontSize: '0.75rem',
                              fontWeight: 600,
                              cursor: 'pointer'
                            }}
                          >
                            Audit
                          </button>

                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSelectEntity(entity.canonical_id);
                              setViewMode('graph');
                            }}
                            className="btn-primary"
                            style={{ padding: '5px 10px', fontSize: '0.75rem', borderRadius: '8px' }}
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

            {/* Right Column: Interactive Map & Mini-Graph */}
            <div style={{ gridColumn: 'span 7', display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {/* Selected Entity Fast-Travel Bar */}
              {activeEntity && (
                <div style={{
                  background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%)',
                  border: '1px solid #10b981',
                  borderRadius: '16px',
                  padding: '12px 18px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  boxShadow: '0 8px 25px rgba(16, 185, 129, 0.2)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{
                      width: '36px',
                      height: '36px',
                      borderRadius: '50%',
                      background: '#10b981',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#ffffff',
                      boxShadow: '0 0 15px rgba(16,185,129,0.8)'
                    }}>
                      <Car size={18} />
                    </div>
                    <div>
                      <div style={{ fontSize: '0.72rem', color: '#34d399', textTransform: 'uppercase', fontWeight: 800 }}>
                        Active Selection
                      </div>
                      <div style={{ fontSize: '1rem', fontWeight: 800, color: '#ffffff' }}>
                        {activeEntity.canonical_name}
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={() => handleTriggerTravel(activeEntity.canonical_id)}
                    className="btn-travel-gps"
                    style={{ padding: '8px 18px', fontSize: '0.82rem', gap: '6px' }}
                  >
                    <Navigation size={15} /> Travel to this Place Now
                  </button>
                </div>
              )}

              {/* Map Component */}
              <div style={{ height: '560px' }}>
                <InteractiveMap 
                  results={searchResults}
                  activeEntityId={activeEntityId}
                  onSelectEntity={handleSelectEntity}
                  parsedQuery={parsedQuery}
                  gpsDestinationId={gpsDestinationId}
                  isGpsModeActive={isGpsTravelActive}
                  onExitGps={() => {
                    setIsGpsTravelActive(false);
                    setGpsDestinationId(null);
                  }}
                  onOpenDetails={openEntityDetails}
                />
              </div>

              {/* Entity Graph Preview */}
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
              <h3 style={{ fontSize: '1.35rem', marginBottom: '4px', fontWeight: 800 }}>
                EntityGraph Engine Benchmarks & Dataset Metrics
              </h3>
              <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
                Verified against 2,208,614 multi-source raw records across India, US, and France
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#34d399' }}>95.21%</div>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '4px' }}>Macro F0.5 Accuracy</div>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#818cf8' }}>{platformStats.total_canonical_entities?.toLocaleString()}</div>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '4px' }}>Indexed Golden Entities</div>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#38bdf8' }}>{platformStats.total_source_records?.toLocaleString()}</div>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '4px' }}>Multi-Source Records</div>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#fbbf24' }}>&lt; 5 ms</div>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '4px' }}>FTS5 Resolution Latency</div>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Detail Audit Modal */}
      {selectedEntityForModal && (
        <EntityDetailModal 
          entity={selectedEntityForModal} 
          onClose={() => setSelectedEntityForModal(null)} 
        />
      )}
    </div>
  );
}
