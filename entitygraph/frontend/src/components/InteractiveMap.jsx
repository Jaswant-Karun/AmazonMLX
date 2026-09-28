import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { 
  Compass, Layers, MapPin, Navigation, Eye, CheckCircle2, 
  ExternalLink, Sparkles, Building2, Store, Utensils, HeartPulse, Laptop 
} from 'lucide-react';

// Tile provider presets
// 100% Free, No-API-Key Tile Providers (No Watermark, High Reliability)
const TILE_LAYERS = {
  streets: {
    name: 'Street View',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri &mdash; Street Map',
    maxZoom: 19
  },
  dark: {
    name: 'Dark Canvas',
    base: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    ref: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri &mdash; Dark Canvas',
    maxZoom: 16
  },
  satellite: {
    name: 'Satellite',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri, Maxar &mdash; Satellite',
    maxZoom: 19
  },
  osm: {
    name: 'OpenStreetMap',
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19
  }
};

// Deterministic micro-offset to prevent overlapping markers in the same locality
function getJitter(strId, index) {
  let hash = 0;
  for (let i = 0; i < strId.length; i++) {
    hash = (hash << 5) - hash + strId.charCodeAt(i);
    hash |= 0;
  }
  const angle = (Math.abs(hash) % 360) * (Math.PI / 180);
  const radius = 0.003 + (index * 0.0015); // ~300 to 800 meters
  return {
    lat: Math.sin(angle) * radius,
    lng: Math.cos(angle) * radius
  };
}

export default function InteractiveMap({ results = [], activeEntityId, onSelectEntity, parsedQuery }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const tileLayersGroupRef = useRef(null);
  const markersLayerRef = useRef(null);
  const [currentLayerKey, setCurrentLayerKey] = useState('streets');
  const [isMapReady, setIsMapReady] = useState(false);

  const activeEntity = results.find(r => r.canonical_id === activeEntityId) || results[0];

  // Helper to attach tile layers
  const setMapTiles = (map, layerKey) => {
    if (tileLayersGroupRef.current) {
      tileLayersGroupRef.current.clearLayers();
    } else {
      tileLayersGroupRef.current = L.layerGroup().addTo(map);
    }

    const config = TILE_LAYERS[layerKey] || TILE_LAYERS.streets;
    if (config.base && config.ref) {
      // Base + Reference (Dark Canvas)
      const base = L.tileLayer(config.base, { attribution: config.attribution, maxZoom: config.maxZoom });
      const ref = L.tileLayer(config.ref, { maxZoom: config.maxZoom });
      tileLayersGroupRef.current.addLayer(base);
      tileLayersGroupRef.current.addLayer(ref);
    } else {
      const tile = L.tileLayer(config.url, { attribution: config.attribution, maxZoom: config.maxZoom });
      tileLayersGroupRef.current.addLayer(tile);
    }
  };

  // 1. Initialize Map once
  useEffect(() => {
    if (!mapContainerRef.current) return;

    const initialLat = activeEntity?.lat || 20.5937;
    const initialLng = activeEntity?.lng || 78.9629;

    const map = L.map(mapContainerRef.current, {
      center: [initialLat, initialLng],
      zoom: 6,
      zoomControl: false,
      attributionControl: true
    });

    // Custom dark zoom control
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Initial tile layer (Street View - crisp and clean)
    setMapTiles(map, 'streets');

    markersLayerRef.current = L.layerGroup().addTo(map);
    mapInstanceRef.current = map;
    setIsMapReady(true);

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // 2. Change Tile Layer if user switches view
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    setMapTiles(map, currentLayerKey);
  }, [currentLayerKey]);

  // 3. Render Markers for Real Dataset Results
  useEffect(() => {
    const map = mapInstanceRef.current;
    const layer = markersLayerRef.current;
    if (!map || !layer || !isMapReady) return;

    layer.clearLayers();

    if (!results || results.length === 0) return;

    const bounds = L.latLngBounds();

    results.forEach((entity, idx) => {
      if (!entity.lat || !entity.lng) return;

      const isSelected = entity.canonical_id === activeEntityId;
      const jitter = getJitter(entity.canonical_id, idx);
      const entityLat = entity.lat + jitter.lat;
      const entityLng = entity.lng + jitter.lng;

      bounds.extend([entityLat, entityLng]);

      // Category colors & icons
      let pinColor = '#4f46e5';
      let pinLabel = '🏬';
      if (entity.category === 'food_dining') { pinColor = '#10b981'; pinLabel = '☕'; }
      else if (entity.category === 'corporate_tech') { pinColor = '#06b6d4'; pinLabel = '💻'; }
      else if (entity.category === 'healthcare_pharma') { pinColor = '#f43f5e'; pinLabel = '🏥'; }
      else if (entity.category === 'real_estate_premises') { pinColor = '#a855f7'; pinLabel = '🏢'; }
      else if (entity.category === 'retail_shop') { pinColor = '#f59e0b'; pinLabel = '🛍️'; }

      // Custom HTML Marker Icon
      const customIcon = L.divIcon({
        className: 'custom-real-marker',
        html: `
          <div style="
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            width: ${isSelected ? '38px' : '30px'};
            height: ${isSelected ? '38px' : '30px'};
            background: ${isSelected ? '#6366f1' : pinColor};
            border: 2px solid ${isSelected ? '#ffffff' : 'rgba(255,255,255,0.85)'};
            border-radius: 50%;
            box-shadow: 0 0 ${isSelected ? '22px rgba(99,102,241,0.9)' : '10px rgba(0,0,0,0.6)'};
            cursor: pointer;
            transition: all 0.2s ease;
          ">
            <span style="font-size: ${isSelected ? '16px' : '13px'}; line-height: 1;">${pinLabel}</span>
            ${isSelected ? `
              <div style="
                position: absolute;
                inset: -6px;
                border: 2px solid #818cf8;
                border-radius: 50%;
                animation: pulse 1.8s infinite;
              "></div>
            ` : ''}
          </div>
        `,
        iconSize: [isSelected ? 38 : 30, isSelected ? 38 : 30],
        iconAnchor: [isSelected ? 19 : 15, isSelected ? 19 : 15]
      });

      const marker = L.marker([entityLat, entityLng], { icon: customIcon });

      // Interactive Popup
      const flag = entity.country === 'India' ? '🇮🇳' : (entity.country === 'France' ? '🇫🇷' : '🇺🇸');
      const popupHtml = `
        <div style="
          min-width: 230px;
          max-width: 280px;
          font-family: inherit;
          color: #f8fafc;
          padding: 6px 2px;
        ">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-size: 0.72rem; font-weight: 700; color: #818cf8; background: rgba(99,102,241,0.15); padding: 2px 7px; borderRadius: 4px;">
              ${entity.canonical_id}
            </span>
            <span style="font-size: 0.8rem;">${flag} ${entity.country}</span>
          </div>
          <h4 style="margin: 0 0 4px 0; font-size: 0.95rem; font-weight: 700; color: #ffffff;">
            ${entity.canonical_name}
          </h4>
          <p style="margin: 0 0 8px 0; font-size: 0.78rem; color: #94a3b8; line-height: 1.35;">
            📍 ${entity.golden_address}
          </p>
          <div style="display: flex; align-items: center; justify-content: space-between; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 6px; font-size: 0.72rem;">
            <span style="color: #34d399; font-weight: 600;">
              ✓ ${entity.matched_sources_count || 1} Sources Resolved
            </span>
            <span style="color: #fbbf24; font-weight: 600;">
              ⭐ ${entity.rating || 4.5}
            </span>
          </div>
        </div>
      `;

      marker.bindPopup(popupHtml, {
        className: 'dark-leaflet-popup',
        closeButton: false
      });

      marker.on('click', () => {
        onSelectEntity(entity.canonical_id);
      });

      marker.addTo(layer);
    });

    // Fly smoothly to active entity if selected
    if (activeEntity?.lat && activeEntity?.lng) {
      const activeJitter = getJitter(activeEntity.canonical_id, 0);
      map.flyTo([activeEntity.lat + activeJitter.lat, activeEntity.lng + activeJitter.lng], 12, {
        animate: true,
        duration: 1.2
      });
    } else if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 11 });
    }
  }, [results, activeEntityId, isMapReady]);

  // Recenter map on active entity
  const handleRecenter = () => {
    const map = mapInstanceRef.current;
    if (!map || !activeEntity?.lat || !activeEntity?.lng) return;
    map.flyTo([activeEntity.lat, activeEntity.lng], 13, { duration: 1.0 });
  };

  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: '490px',
      borderRadius: '16px',
      overflow: 'hidden',
      border: '1px solid var(--border-subtle)',
      boxShadow: '0 12px 35px rgba(0,0,0,0.6)',
      background: '#090d16'
    }}>
      {/* Map Header Overlay Bar */}
      <div style={{
        position: 'absolute',
        top: '12px',
        left: '12px',
        right: '12px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'rgba(15, 23, 42, 0.88)',
        backdropFilter: 'blur(12px)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '10px',
        padding: '8px 14px',
        zIndex: 1000
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.82rem', fontWeight: 600 }}>
          <Compass size={16} color="#818cf8" />
          <span style={{ color: '#ffffff' }}>OpenStreetMap • Real-World Geographic Alignment</span>
          {activeEntity && (
            <span style={{
              background: 'rgba(99, 102, 241, 0.2)',
              color: '#c7d2fe',
              padding: '2px 8px',
              borderRadius: '6px',
              fontSize: '0.72rem',
              fontWeight: 500
            }}>
              {activeEntity.city || activeEntity.locality}, {activeEntity.country}
            </span>
          )}
        </div>

        {/* Tile Provider & Recenter Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            display: 'flex',
            background: 'rgba(0,0,0,0.4)',
            borderRadius: '6px',
            padding: '2px',
            border: '1px solid rgba(255,255,255,0.06)'
          }}>
            {Object.keys(TILE_LAYERS).map(key => (
              <button
                key={key}
                onClick={() => setCurrentLayerKey(key)}
                style={{
                  background: currentLayerKey === key ? '#4f46e5' : 'transparent',
                  color: currentLayerKey === key ? '#ffffff' : '#94a3b8',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '3px 8px',
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                {TILE_LAYERS[key].name}
              </button>
            ))}
          </div>

          <button
            onClick={handleRecenter}
            title="Recenter on Active Entity"
            style={{
              background: 'rgba(99, 102, 241, 0.25)',
              border: '1px solid #818cf8',
              color: '#c7d2fe',
              borderRadius: '6px',
              padding: '4px 8px',
              fontSize: '0.72rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
          >
            <Navigation size={12} /> Recenter
          </button>
        </div>
      </div>

      {/* Leaflet Map DOM Target */}
      <div 
        ref={mapContainerRef} 
        style={{ width: '100%', height: '100%', zIndex: 1 }} 
      />

      {/* Map Legend Overlay at Bottom Left */}
      <div style={{
        position: 'absolute',
        bottom: '12px',
        left: '12px',
        background: 'rgba(15, 23, 42, 0.85)',
        backdropFilter: 'blur(8px)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '8px',
        padding: '6px 12px',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        fontSize: '0.72rem',
        color: '#94a3b8'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: '#f59e0b' }}>🛍️</span> Retail & Shops
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: '#10b981' }}>☕</span> Food & Dining
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: '#06b6d4' }}>💻</span> Tech & Labs
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: '#a855f7' }}>🏢</span> Estates & Buildings
        </div>
      </div>
    </div>
  );
}
