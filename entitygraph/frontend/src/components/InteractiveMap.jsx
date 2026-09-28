import React, { useEffect, useRef, useState, useMemo } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { 
  Compass, Layers, MapPin, Navigation, Eye, CheckCircle2, 
  ExternalLink, Sparkles, Building2, Store, Utensils, HeartPulse, Laptop,
  Play, Pause, RotateCcw, Gauge, Clock, ArrowUpRight, CornerUpRight, 
  CornerUpLeft, Flag, Award, Volume2, X, ChevronRight
} from 'lucide-react';

// Free, No-API-Key Tile Providers (No Watermark, Ultra High Reliability)
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
  const radius = 0.003 + (index * 0.0015);
  return {
    lat: Math.sin(angle) * radius,
    lng: Math.cos(angle) * radius
  };
}

// Calculate compass bearing between two points
function calculateBearing(lat1, lng1, lat2, lng2) {
  const dLng = (lng2 - lng1) * (Math.PI / 180);
  const y = Math.sin(dLng) * Math.cos(lat2 * (Math.PI / 180));
  const x = Math.cos(lat1 * (Math.PI / 180)) * Math.sin(lat2 * (Math.PI / 180)) -
            Math.sin(lat1 * (Math.PI / 180)) * Math.cos(lat2 * (Math.PI / 180)) * Math.cos(dLng);
  const brng = (Math.atan2(y, x) * (180 / Math.PI) + 360) % 360;
  return brng;
}

// Calculate distance in kilometers
function calculateDistanceKm(lat1, lng1, lat2, lng2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * (Math.PI / 180);
  const dLng = (lng2 - lng1) * (Math.PI / 180);
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * (Math.PI / 180)) * Math.cos(lat2 * (Math.PI / 180)) *
            Math.sin(dLng / 2) * Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

// Generate realistic road waypoints from origin to destination
function generateRoadRoute(destLat, destLng, entityName, locality, landmark) {
  // Origin ~ 2.4 km southwest
  const originLat = destLat - 0.0165;
  const originLng = destLng - 0.0195;

  const keyWaypoints = [
    { lat: originLat, lng: originLng, instruction: "Head Northeast on Outer Ring Road", turn: "straight" },
    { lat: originLat + 0.0045, lng: originLng + 0.0035, instruction: "Continue straight past Highway Junction (1.2 km)", turn: "straight" },
    { lat: originLat + 0.0090, lng: originLng + 0.0080, instruction: `In 350m, turn right onto Central Avenue towards ${locality || 'Commercial Zone'}`, turn: "right" },
    { lat: originLat + 0.0105, lng: originLng + 0.0135, instruction: `Turn right onto Central Avenue`, turn: "right" },
    { lat: originLat + 0.0135, lng: originLng + 0.0160, instruction: `In 200m, turn left near ${landmark || 'Main Commercial Center'}`, turn: "left" },
    { lat: originLat + 0.0150, lng: originLng + 0.0175, instruction: `Turn left onto Business Driveway`, turn: "left" },
    { lat: destLat, lng: destLng, instruction: `Arriving at ${entityName} on the right`, turn: "arrive" }
  ];

  // Interpolate dense sub-points (total 240 steps for butter-smooth animation)
  const fullSteps = [];
  for (let i = 0; i < keyWaypoints.length - 1; i++) {
    const wp1 = keyWaypoints[i];
    const wp2 = keyWaypoints[i + 1];
    const stepsBetween = 40;

    for (let s = 0; s < stepsBetween; s++) {
      const t = s / stepsBetween;
      const lat = wp1.lat + (wp2.lat - wp1.lat) * t;
      const lng = wp1.lng + (wp2.lng - wp1.lng) * t;
      fullSteps.push({
        lat,
        lng,
        instruction: wp1.instruction,
        turn: wp1.turn,
        targetWp: wp2
      });
    }
  }

  // Push final destination point
  fullSteps.push({
    lat: destLat,
    lng: destLng,
    instruction: `Arrived at ${entityName}!`,
    turn: "arrive",
    targetWp: keyWaypoints[keyWaypoints.length - 1]
  });

  return {
    origin: { lat: originLat, lng: originLng },
    destination: { lat: destLat, lng: destLng },
    keyWaypoints,
    steps: fullSteps,
    totalDistanceKm: calculateDistanceKm(originLat, originLng, destLat, destLng) * 1.35
  };
}

export default function InteractiveMap({ 
  results = [], 
  activeEntityId, 
  onSelectEntity, 
  parsedQuery,
  gpsDestinationId,
  onOpenDetails
}) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const tileLayersGroupRef = useRef(null);
  const markersLayerRef = useRef(null);
  const routeLayerRef = useRef(null);
  const vehicleMarkerRef = useRef(null);

  const [currentLayerKey, setCurrentLayerKey] = useState('streets');
  const [isMapReady, setIsMapReady] = useState(false);

  // GPS Simulation State
  const [isGpsActive, setIsGpsActive] = useState(false);
  const [isNavigating, setIsNavigating] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [simulationSpeed, setSimulationSpeed] = useState(1); // 1x, 2x, 4x
  const [hasArrived, setHasArrived] = useState(false);
  const [cameraFollow, setCameraFollow] = useState(true);

  const activeEntity = useMemo(() => {
    return results.find(r => r.canonical_id === (gpsDestinationId || activeEntityId)) || results[0];
  }, [results, activeEntityId, gpsDestinationId]);

  // Generate current road route
  const currentRoute = useMemo(() => {
    if (!activeEntity?.lat || !activeEntity?.lng) return null;
    return generateRoadRoute(
      activeEntity.lat, 
      activeEntity.lng, 
      activeEntity.canonical_name, 
      activeEntity.locality, 
      activeEntity.landmark
    );
  }, [activeEntity]);

  // Helper to attach tile layers
  const setMapTiles = (map, layerKey) => {
    if (tileLayersGroupRef.current) {
      tileLayersGroupRef.current.clearLayers();
    } else {
      tileLayersGroupRef.current = L.layerGroup().addTo(map);
    }

    const config = TILE_LAYERS[layerKey] || TILE_LAYERS.streets;
    if (config.base && config.ref) {
      const base = L.tileLayer(config.base, { attribution: config.attribution, maxZoom: config.maxZoom });
      const ref = L.tileLayer(config.ref, { maxZoom: config.maxZoom });
      tileLayersGroupRef.current.addLayer(base);
      tileLayersGroupRef.current.addLayer(ref);
    } else {
      const tile = L.tileLayer(config.url, { attribution: config.attribution, maxZoom: config.maxZoom });
      tileLayersGroupRef.current.addLayer(tile);
    }
  };

  // 1. Initialize Map Once
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

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    setMapTiles(map, 'streets');

    markersLayerRef.current = L.layerGroup().addTo(map);
    routeLayerRef.current = L.layerGroup().addTo(map);
    mapInstanceRef.current = map;
    setIsMapReady(true);

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // 2. Change Tile Layer
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    setMapTiles(map, currentLayerKey);
  }, [currentLayerKey]);

  // 3. Render Normal Dataset Markers
  useEffect(() => {
    const map = mapInstanceRef.current;
    const layer = markersLayerRef.current;
    if (!map || !layer || !isMapReady) return;

    layer.clearLayers();

    if (!results || results.length === 0) return;

    const bounds = L.latLngBounds();

    results.forEach((entity, idx) => {
      if (!entity.lat || !entity.lng) return;

      const isSelected = entity.canonical_id === activeEntity?.canonical_id;
      const jitter = getJitter(entity.canonical_id, idx);
      const entityLat = entity.lat + jitter.lat;
      const entityLng = entity.lng + jitter.lng;

      bounds.extend([entityLat, entityLng]);

      let pinColor = '#4f46e5';
      let pinLabel = '🏬';
      if (entity.category === 'food_dining') { pinColor = '#10b981'; pinLabel = '☕'; }
      else if (entity.category === 'corporate_tech') { pinColor = '#06b6d4'; pinLabel = '💻'; }
      else if (entity.category === 'healthcare_pharma') { pinColor = '#f43f5e'; pinLabel = '🏥'; }
      else if (entity.category === 'real_estate_premises') { pinColor = '#a855f7'; pinLabel = '🏢'; }
      else if (entity.category === 'retail_shop') { pinColor = '#f59e0b'; pinLabel = '🛍️'; }

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

      const flag = entity.country === 'India' ? '🇮🇳' : (entity.country === 'France' ? '🇫🇷' : '🇺🇸');
      const popupHtml = `
        <div style="min-width: 230px; max-width: 280px; font-family: inherit; color: #f8fafc; padding: 6px 2px;">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-size: 0.72rem; font-weight: 700; color: #818cf8; background: rgba(99,102,241,0.15); padding: 2px 7px; border-radius: 4px;">
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

      marker.bindPopup(popupHtml, { className: 'dark-leaflet-popup', closeButton: false });
      marker.on('click', () => onSelectEntity(entity.canonical_id));
      marker.addTo(layer);
    });

    if (!isGpsActive) {
      if (activeEntity?.lat && activeEntity?.lng) {
        const activeJitter = getJitter(activeEntity.canonical_id, 0);
        map.flyTo([activeEntity.lat + activeJitter.lat, activeEntity.lng + activeJitter.lng], 12, { duration: 1.0 });
      } else if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [40, 40], maxZoom: 11 });
      }
    }
  }, [results, activeEntity, isMapReady, isGpsActive]);

  // 4. GPS Navigation Mode: Draw Neon Route & Vehicle Marker
  useEffect(() => {
    const map = mapInstanceRef.current;
    const rLayer = routeLayerRef.current;
    if (!map || !rLayer || !isMapReady) return;

    rLayer.clearLayers();

    if (!isGpsActive || !currentRoute) return;

    // Draw full road polyline (Glow background)
    const latLngs = currentRoute.steps.map(s => [s.lat, s.lng]);
    
    // Outer glow polyline
    L.polyline(latLngs, {
      color: '#06b6d4',
      weight: 8,
      opacity: 0.45,
      lineCap: 'round',
      lineJoin: 'round'
    }).addTo(rLayer);

    // Inner bright polyline
    L.polyline(latLngs, {
      color: '#38bdf8',
      weight: 4,
      opacity: 0.95,
      dashArray: '8, 6',
      lineCap: 'round'
    }).addTo(rLayer);

    // Origin Marker (Start)
    const originIcon = L.divIcon({
      className: 'gps-origin-marker',
      html: `
        <div style="background: #10b981; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid white; box-shadow: 0 0 15px rgba(16,185,129,0.8); font-size: 12px; font-weight: 700;">
          🏁
        </div>
      `,
      iconSize: [28, 28],
      iconAnchor: [14, 14]
    });
    L.marker([currentRoute.origin.lat, currentRoute.origin.lng], { icon: originIcon }).addTo(rLayer);

    // Destination Marker (Finish)
    const destIcon = L.divIcon({
      className: 'gps-dest-marker',
      html: `
        <div style="background: #ef4444; color: white; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid white; box-shadow: 0 0 20px rgba(239,68,68,0.9); font-size: 15px;">
          📍
        </div>
      `,
      iconSize: [34, 34],
      iconAnchor: [17, 17]
    });
    L.marker([currentRoute.destination.lat, currentRoute.destination.lng], { icon: destIcon }).addTo(rLayer);

    // Initialize vehicle marker at current step
    const step = currentRoute.steps[currentStepIndex] || currentRoute.steps[0];
    const nextStep = currentRoute.steps[Math.min(currentStepIndex + 1, currentRoute.steps.length - 1)];
    const heading = calculateBearing(step.lat, step.lng, nextStep.lat, nextStep.lng);

    const vehicleIcon = L.divIcon({
      className: 'gps-vehicle-marker',
      html: `
        <div style="
          position: relative;
          width: 44px;
          height: 44px;
          display: flex;
          align-items: center;
          justify-content: center;
          transform: rotate(${heading}deg);
          transition: transform 0.15s ease-out;
        ">
          <!-- Radar beam glow ahead -->
          <div style="
            position: absolute;
            top: -24px;
            width: 32px;
            height: 32px;
            background: radial-gradient(circle, rgba(56,189,248,0.5) 0%, transparent 70%);
            border-radius: 50%;
          "></div>
          <!-- Vehicle body -->
          <div style="
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            border: 2px solid #ffffff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px rgba(99,102,241,0.95);
            font-size: 16px;
          ">
            🚘
          </div>
        </div>
      `,
      iconSize: [44, 44],
      iconAnchor: [22, 22]
    });

    const vMarker = L.marker([step.lat, step.lng], { icon: vehicleIcon, zIndexOffset: 1000 });
    vMarker.addTo(rLayer);
    vehicleMarkerRef.current = vMarker;

    // Zoom into route on activation
    const routeBounds = L.latLngBounds(latLngs);
    map.fitBounds(routeBounds, { padding: [60, 60], maxZoom: 15 });
  }, [isGpsActive, currentRoute, isMapReady]);

  // 5. GPS Simulation Animation Loop
  useEffect(() => {
    if (!isGpsActive || !isNavigating || !currentRoute) return;

    const intervalTime = 60 / simulationSpeed;
    const timer = setInterval(() => {
      setCurrentStepIndex(prev => {
        if (prev >= currentRoute.steps.length - 1) {
          setIsNavigating(false);
          setHasArrived(true);
          return prev;
        }

        const next = prev + 1;
        const curStep = currentRoute.steps[next];
        const futureStep = currentRoute.steps[Math.min(next + 1, currentRoute.steps.length - 1)];
        const heading = calculateBearing(curStep.lat, curStep.lng, futureStep.lat, futureStep.lng);

        // Update vehicle position smoothly
        if (vehicleMarkerRef.current) {
          vehicleMarkerRef.current.setLatLng([curStep.lat, curStep.lng]);
          
          const iconElem = vehicleMarkerRef.current.getElement();
          if (iconElem) {
            const inner = iconElem.querySelector('.gps-vehicle-marker > div') || iconElem.firstChild;
            if (inner && inner.style) {
              inner.style.transform = `rotate(${heading}deg)`;
            }
          }
        }

        // Camera follow
        if (cameraFollow && mapInstanceRef.current) {
          mapInstanceRef.current.setView([curStep.lat, curStep.lng], 16, { animate: true, duration: 0.1 });
        }

        return next;
      });
    }, intervalTime);

    return () => clearInterval(timer);
  }, [isGpsActive, isNavigating, currentRoute, simulationSpeed, cameraFollow]);

  // Start GPS Travel
  const startTravelSimulation = () => {
    setIsGpsActive(true);
    setHasArrived(false);
    setCurrentStepIndex(0);
    setIsNavigating(true);
  };

  const pauseResumeTravel = () => {
    setIsNavigating(prev => !prev);
  };

  const resetTravel = () => {
    setCurrentStepIndex(0);
    setHasArrived(false);
    setIsNavigating(false);
    if (vehicleMarkerRef.current && currentRoute) {
      vehicleMarkerRef.current.setLatLng([currentRoute.origin.lat, currentRoute.origin.lng]);
    }
  };

  const exitGpsMode = () => {
    setIsGpsActive(false);
    setIsNavigating(false);
    setHasArrived(false);
    setCurrentStepIndex(0);
    if (routeLayerRef.current) {
      routeLayerRef.current.clearLayers();
    }
    if (activeEntity?.lat && mapInstanceRef.current) {
      mapInstanceRef.current.setView([activeEntity.lat, activeEntity.lng], 12);
    }
  };

  // Turn calculation for current step
  const currentStep = currentRoute?.steps[currentStepIndex] || currentRoute?.steps[0];
  const progressRatio = currentRoute ? (currentStepIndex / (currentRoute.steps.length - 1)) : 0;
  const remainingDistKm = currentRoute ? Math.max(0, (currentRoute.totalDistanceKm * (1 - progressRatio))).toFixed(2) : '0.0';
  const remainingTimeMins = Math.max(1, Math.round(parseFloat(remainingDistKm) * 1.5));
  const currentSpeedKmH = isNavigating ? Math.round(42 + (Math.sin(currentStepIndex * 0.1) * 8) * simulationSpeed) : 0;

  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: '520px',
      borderRadius: '16px',
      overflow: 'hidden',
      border: '1px solid var(--border-glow)',
      boxShadow: '0 16px 40px rgba(0,0,0,0.7)',
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
        background: 'rgba(15, 23, 42, 0.90)',
        backdropFilter: 'blur(12px)',
        border: '1px solid rgba(255,255,255,0.12)',
        borderRadius: '12px',
        padding: '8px 14px',
        zIndex: 1000
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.82rem', fontWeight: 600 }}>
          <Compass size={16} color="#818cf8" />
          <span style={{ color: '#ffffff' }}>OpenStreetMap • Real-World Geographic Alignment</span>
          {activeEntity && (
            <span style={{
              background: 'rgba(99, 102, 241, 0.25)',
              color: '#c7d2fe',
              padding: '2px 8px',
              borderRadius: '6px',
              fontSize: '0.72rem',
              fontWeight: 600,
              border: '1px solid rgba(99, 102, 241, 0.3)'
            }}>
              {activeEntity.city || activeEntity.locality}, {activeEntity.country}
            </span>
          )}
        </div>

        {/* GPS Mode Trigger & Layer Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {!isGpsActive ? (
            <button
              onClick={startTravelSimulation}
              style={{
                background: 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
                color: '#ffffff',
                border: 'none',
                borderRadius: '8px',
                padding: '6px 12px',
                fontSize: '0.75rem',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                boxShadow: '0 0 15px rgba(16,185,129,0.5)'
              }}
            >
              <Navigation size={14} /> Start GPS Travel
            </button>
          ) : (
            <button
              onClick={exitGpsMode}
              style={{
                background: 'rgba(239, 68, 68, 0.25)',
                color: '#fca5a5',
                border: '1px solid #ef4444',
                borderRadius: '8px',
                padding: '6px 12px',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <X size={14} /> Exit GPS Mode
            </button>
          )}

          <div style={{
            display: 'flex',
            background: 'rgba(0,0,0,0.5)',
            borderRadius: '6px',
            padding: '2px',
            border: '1px solid rgba(255,255,255,0.08)'
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
                  padding: '3px 7px',
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                {TILE_LAYERS[key].name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* TOP-LEFT GPS TURN-BY-TURN HUD (When in GPS Navigation Mode) */}
      {isGpsActive && currentStep && (
        <div style={{
          position: 'absolute',
          top: '64px',
          left: '12px',
          zIndex: 1000,
          background: 'rgba(15, 23, 42, 0.94)',
          backdropFilter: 'blur(16px)',
          border: '1px solid #38bdf8',
          borderRadius: '14px',
          padding: '12px 16px',
          boxShadow: '0 10px 30px rgba(6, 182, 212, 0.35)',
          maxWidth: '380px',
          color: '#ffffff'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '10px',
              background: '#0284c7',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '20px',
              flexShrink: 0
            }}>
              {currentStep.turn === 'right' ? <CornerUpRight size={22} color="#ffffff" /> : 
               currentStep.turn === 'left' ? <CornerUpLeft size={22} color="#ffffff" /> : 
               currentStep.turn === 'arrive' ? <Flag size={22} color="#facc15" /> : 
               <ArrowUpRight size={22} color="#ffffff" />}
            </div>

            <div>
              <div style={{ fontSize: '0.72rem', color: '#38bdf8', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.5px' }}>
                {currentStep.turn === 'arrive' ? 'Target Destination' : 'Next Navigation Cue'}
              </div>
              <div style={{ fontSize: '0.92rem', fontWeight: 700, lineHeight: 1.3 }}>
                {currentStep.instruction}
              </div>
            </div>
          </div>

          {/* Real-time Progress Bar */}
          <div style={{ marginTop: '10px', background: 'rgba(255,255,255,0.1)', height: '4px', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{
              background: 'linear-gradient(90deg, #38bdf8 0%, #34d399 100%)',
              height: '100%',
              width: `${progressRatio * 100}%`,
              transition: 'width 0.1s linear'
            }}></div>
          </div>
        </div>
      )}

      {/* BOTTOM GPS COCKPIT CONTROL BAR (When in GPS Mode) */}
      {isGpsActive && (
        <div style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          right: '12px',
          zIndex: 1000,
          background: 'rgba(15, 23, 42, 0.95)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: '14px',
          padding: '10px 18px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          boxShadow: '0 10px 30px rgba(0,0,0,0.8)'
        }}>
          {/* Trip Metrics */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Gauge size={18} color="#38bdf8" />
              <div>
                <div style={{ fontSize: '0.65rem', color: '#94a3b8', textTransform: 'uppercase' }}>Current Speed</div>
                <div style={{ fontSize: '1rem', fontWeight: 800, color: '#ffffff' }}>{currentSpeedKmH} km/h</div>
              </div>
            </div>

            <div style={{ width: '1px', height: '26px', background: 'rgba(255,255,255,0.1)' }}></div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Navigation size={18} color="#34d399" />
              <div>
                <div style={{ fontSize: '0.65rem', color: '#94a3b8', textTransform: 'uppercase' }}>Remaining</div>
                <div style={{ fontSize: '1rem', fontWeight: 800, color: '#ffffff' }}>{remainingDistKm} km</div>
              </div>
            </div>

            <div style={{ width: '1px', height: '26px', background: 'rgba(255,255,255,0.1)' }}></div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Clock size={18} color="#fbbf24" />
              <div>
                <div style={{ fontSize: '0.65rem', color: '#94a3b8', textTransform: 'uppercase' }}>ETA</div>
                <div style={{ fontSize: '1rem', fontWeight: 800, color: '#ffffff' }}>{remainingTimeMins} min</div>
              </div>
            </div>
          </div>

          {/* Navigation Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              onClick={() => setCameraFollow(prev => !prev)}
              style={{
                background: cameraFollow ? 'rgba(56,189,248,0.2)' : 'rgba(255,255,255,0.05)',
                border: cameraFollow ? '1px solid #38bdf8' : '1px solid rgba(255,255,255,0.1)',
                color: cameraFollow ? '#38bdf8' : '#94a3b8',
                borderRadius: '8px',
                padding: '6px 10px',
                fontSize: '0.72rem',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              🎥 Lock Camera
            </button>

            {/* Speed Multipliers */}
            <div style={{ display: 'flex', background: 'rgba(0,0,0,0.5)', borderRadius: '6px', padding: '2px' }}>
              {[1, 2, 4].map(s => (
                <button
                  key={s}
                  onClick={() => setSimulationSpeed(s)}
                  style={{
                    background: simulationSpeed === s ? '#4f46e5' : 'transparent',
                    color: simulationSpeed === s ? '#ffffff' : '#94a3b8',
                    border: 'none',
                    borderRadius: '4px',
                    padding: '3px 8px',
                    fontSize: '0.7rem',
                    fontWeight: 700,
                    cursor: 'pointer'
                  }}
                >
                  {s}x
                </button>
              ))}
            </div>

            <button
              onClick={pauseResumeTravel}
              style={{
                background: isNavigating ? 'rgba(245,158,11,0.2)' : 'rgba(16,185,129,0.2)',
                border: isNavigating ? '1px solid #f59e0b' : '1px solid #10b981',
                color: isNavigating ? '#fbbf24' : '#34d399',
                borderRadius: '8px',
                padding: '6px 12px',
                fontSize: '0.75rem',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              {isNavigating ? <><Pause size={14} /> Pause</> : <><Play size={14} /> Resume</>}
            </button>

            <button
              onClick={resetTravel}
              title="Restart Route"
              style={{
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.12)',
                color: '#ffffff',
                borderRadius: '8px',
                padding: '6px 10px',
                fontSize: '0.75rem',
                cursor: 'pointer'
              }}
            >
              <RotateCcw size={14} />
            </button>
          </div>
        </div>
      )}

      {/* ARRIVAL CELEBRATION MODAL BANNER */}
      {hasArrived && activeEntity && (
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'rgba(0,0,0,0.7)',
          backdropFilter: 'blur(8px)',
          zIndex: 2000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px'
        }}>
          <div style={{
            background: 'linear-gradient(135deg, #0d111d 0%, #171d33 100%)',
            border: '2px solid #34d399',
            boxShadow: '0 0 40px rgba(52,211,153,0.4)',
            borderRadius: '16px',
            padding: '24px 32px',
            textAlign: 'center',
            maxWidth: '460px',
            color: '#ffffff'
          }}>
            <div style={{ fontSize: '38px', marginBottom: '8px' }}>🎉</div>
            <h3 style={{ fontSize: '1.35rem', fontWeight: 800, marginBottom: '6px' }}>
              You Have Arrived!
            </h3>
            <p style={{ fontSize: '1.05rem', color: '#34d399', fontWeight: 700, marginBottom: '4px' }}>
              {activeEntity.canonical_name}
            </p>
            <p style={{ fontSize: '0.82rem', color: '#94a3b8', marginBottom: '16px', lineHeight: 1.4 }}>
              📍 {activeEntity.golden_address}
            </p>

            <div style={{
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: '10px',
              padding: '10px',
              fontSize: '0.78rem',
              color: '#cbd5e1',
              marginBottom: '20px',
              display: 'flex',
              justifyContent: 'space-around'
            }}>
              <div>
                <span style={{ color: '#94a3b8', display: 'block', fontSize: '0.68rem' }}>DOOR / BLDG</span>
                <strong>{activeEntity.building_number || 'N/A'}</strong>
              </div>
              <div style={{ borderLeft: '1px solid rgba(255,255,255,0.1)' }}></div>
              <div>
                <span style={{ color: '#94a3b8', display: 'block', fontSize: '0.68rem' }}>LANDMARK</span>
                <strong>{activeEntity.landmark || 'Commercial Hub'}</strong>
              </div>
              <div style={{ borderLeft: '1px solid rgba(255,255,255,0.1)' }}></div>
              <div>
                <span style={{ color: '#94a3b8', display: 'block', fontSize: '0.68rem' }}>CONFIDENCE</span>
                <strong style={{ color: '#34d399' }}>{activeEntity.confidence_pct}%</strong>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
              <button
                onClick={() => {
                  setHasArrived(false);
                  if (onOpenDetails) onOpenDetails(activeEntity.canonical_id);
                }}
                className="btn-primary"
                style={{ padding: '8px 16px', fontSize: '0.82rem', borderRadius: '10px' }}
              >
                Inspect Multi-Source Matches
              </button>
              <button
                onClick={() => setHasArrived(false)}
                className="btn-secondary"
                style={{ padding: '8px 16px', fontSize: '0.82rem', borderRadius: '10px' }}
              >
                Close View
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Leaflet Map DOM Target */}
      <div 
        ref={mapContainerRef} 
        style={{ width: '100%', height: '100%', zIndex: 1 }} 
      />

      {/* Default Map Legend at Bottom Left (When not in GPS mode) */}
      {!isGpsActive && (
        <div style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          background: 'rgba(15, 23, 42, 0.88)',
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
      )}
    </div>
  );
}
