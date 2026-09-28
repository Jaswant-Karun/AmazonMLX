import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { 
  Compass, Layers, MapPin, Navigation, Eye, CheckCircle2, 
  Sparkles, Building2, Store, Utensils, HeartPulse, Laptop,
  Play, Pause, RotateCcw, Gauge, Clock, ArrowUpRight, CornerUpRight, 
  CornerUpLeft, Flag, Award, Volume2, VolumeX, X, ChevronRight,
  FastForward, LocateFixed, Car, ShieldCheck
} from 'lucide-react';

// Tile Providers (100% Free, Zero API Key Required, No Watermark)
const TILE_LAYERS = {
  streets: {
    name: 'Street View',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri &mdash; World Street Map',
    maxZoom: 19
  },
  dark: {
    name: 'Dark Cockpit',
    base: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    ref: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri &mdash; Dark Luxury Canvas',
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

// Deterministic jitter to prevent markers overlapping exactly at same lat/lng
function getJitter(strId, index) {
  let hash = 0;
  for (let i = 0; i < (strId || '').length; i++) {
    hash = (hash << 5) - hash + strId.charCodeAt(i);
    hash |= 0;
  }
  const angle = (Math.abs(hash) % 360) * (Math.PI / 180);
  const radius = 0.0025 + (index * 0.0012);
  return {
    lat: Math.sin(angle) * radius,
    lng: Math.cos(angle) * radius
  };
}

// Calculate compass bearing between two coordinates
function calculateBearing(lat1, lng1, lat2, lng2) {
  const dLng = (lng2 - lng1) * (Math.PI / 180);
  const y = Math.sin(dLng) * Math.cos(lat2 * (Math.PI / 180));
  const x = Math.cos(lat1 * (Math.PI / 180)) * Math.sin(lat2 * (Math.PI / 180)) -
            Math.sin(lat1 * (Math.PI / 180)) * Math.cos(lat2 * (Math.PI / 180)) * Math.cos(dLng);
  const brng = (Math.atan2(y, x) * (180 / Math.PI) + 360) % 360;
  return brng;
}

// Haversine distance in km
function calculateDistanceKm(lat1, lng1, lat2, lng2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * (Math.PI / 180);
  const dLng = (lng2 - lng1) * (Math.PI / 180);
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * (Math.PI / 180)) * Math.cos(lat2 * (Math.PI / 180)) *
            Math.sin(dLng / 2) * Math.sin(dLng / 2);
  return R * (2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)));
}

// Speech synthesis helper for GPS voice announcements
function speakGuidance(text, isMuted) {
  if (isMuted || !('speechSynthesis' in window)) return;
  try {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    utterance.volume = 0.9;
    window.speechSynthesis.speak(utterance);
  } catch (e) {
    // Ignore speech errors
  }
}

export default function InteractiveMap({ 
  results = [], 
  activeEntityId, 
  onSelectEntity, 
  parsedQuery,
  gpsDestinationId,
  isGpsModeActive = false,
  onExitGps,
  onOpenDetails
}) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const tileLayersGroupRef = useRef(null);
  const markersLayerRef = useRef(null);
  const routeLayerRef = useRef(null);
  const vehicleMarkerRef = useRef(null);

  const [currentLayerKey, setCurrentLayerKey] = useState('dark');
  const [isMapReady, setIsMapReady] = useState(false);

  // GPS Simulation State
  const [isGpsActive, setIsGpsActive] = useState(false);
  const [isNavigating, setIsNavigating] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [simulationSpeed, setSimulationSpeed] = useState(2); // 1x, 2x, 5x
  const [hasArrived, setHasArrived] = useState(false);
  const [cameraFollow, setCameraFollow] = useState(true);
  const [isVoiceMuted, setIsVoiceMuted] = useState(false);
  const [lastAnnouncedStep, setLastAnnouncedStep] = useState(-1);

  // Route storage from OSRM or fallback
  const [routeData, setRouteData] = useState(null);
  const [isLoadingRoute, setIsLoadingRoute] = useState(false);

  // Identify the target destination entity
  const targetEntity = useMemo(() => {
    const idToFind = gpsDestinationId || activeEntityId;
    return results.find(r => r.canonical_id === idToFind) || results[0];
  }, [results, activeEntityId, gpsDestinationId]);

  // Set tile layer
  const setMapTiles = useCallback((map, layerKey) => {
    if (!tileLayersGroupRef.current) {
      tileLayersGroupRef.current = L.layerGroup().addTo(map);
    } else {
      tileLayersGroupRef.current.clearLayers();
    }

    const config = TILE_LAYERS[layerKey] || TILE_LAYERS.dark;
    if (config.base && config.ref) {
      const base = L.tileLayer(config.base, { attribution: config.attribution, maxZoom: config.maxZoom });
      const ref = L.tileLayer(config.ref, { maxZoom: config.maxZoom });
      tileLayersGroupRef.current.addLayer(base);
      tileLayersGroupRef.current.addLayer(ref);
    } else {
      const tile = L.tileLayer(config.url, { attribution: config.attribution, maxZoom: config.maxZoom });
      tileLayersGroupRef.current.addLayer(tile);
    }
  }, []);

  // 1. Initialize Map Once
  useEffect(() => {
    if (!mapContainerRef.current) return;

    const initialLat = targetEntity?.lat || 20.5937;
    const initialLng = targetEntity?.lng || 78.9629;

    const map = L.map(mapContainerRef.current, {
      center: [initialLat, initialLng],
      zoom: 12,
      zoomControl: false,
      attributionControl: false
    });

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    setMapTiles(map, 'dark');

    markersLayerRef.current = L.layerGroup().addTo(map);
    routeLayerRef.current = L.layerGroup().addTo(map);
    mapInstanceRef.current = map;
    setIsMapReady(true);

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Change Tile Layer
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    setMapTiles(map, currentLayerKey);
  }, [currentLayerKey, setMapTiles]);

  // Fetch or Compute Road Route (Real OSRM with dense road steps & synthetic fallback)
  const buildRouteForEntity = useCallback(async (dest) => {
    if (!dest?.lat || !dest?.lng) return null;

    setIsLoadingRoute(true);
    const destLat = dest.lat;
    const destLng = dest.lng;

    // Realistic origin ~ 2.8 km away along a street vector
    const originLat = destLat - 0.0185;
    const originLng = destLng - 0.0210;

    let points = [];
    let instructions = [];

    try {
      // Attempt real OSRM road network query with 2.5s timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2500);

      const url = `https://router.project-osrm.org/route/v1/driving/${originLng},${originLat};${destLng},${destLat}?overview=full&geometries=geojson&steps=true`;
      const res = await fetch(url, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        if (data.routes && data.routes.length > 0) {
          const route = data.routes[0];
          const coords = route.geometry.coordinates; // [lng, lat]
          
          // Interpolate dense sub-steps along the road
          for (let i = 0; i < coords.length - 1; i++) {
            const p1 = coords[i];
            const p2 = coords[i + 1];
            const subSteps = 12;
            for (let s = 0; s < subSteps; s++) {
              const t = s / subSteps;
              points.push({
                lat: p1[1] + (p2[1] - p1[1]) * t,
                lng: p1[0] + (p2[0] - p1[0]) * t,
                instruction: `Driving towards ${dest.canonical_name}`,
                turn: 'straight'
              });
            }
          }
          points.push({
            lat: destLat,
            lng: destLng,
            instruction: `Arrived at ${dest.canonical_name}`,
            turn: 'arrive'
          });

          // Extract maneuvers
          if (route.legs && route.legs[0]?.steps) {
            instructions = route.legs[0].steps.map(step => ({
              text: step.maneuver.instruction || `Head ${step.maneuver.modifier || 'forward'} on ${step.name || 'Road'}`,
              modifier: step.maneuver.modifier || 'straight',
              distMeters: Math.round(step.distance)
            }));
          }
        }
      }
    } catch (e) {
      // Fallback below
    }

    // High-Resolution Fallback if OSRM is unreachable
    if (points.length === 0) {
      const landmarkText = dest.landmark || 'Commercial Hub';
      const localityText = dest.locality || 'City Centre';

      const keyWps = [
        { lat: originLat, lng: originLng, instruction: `Start journey from Outer Ring Road`, turn: 'straight' },
        { lat: originLat + 0.0055, lng: originLng + 0.0040, instruction: `Continue past Highway Overpass (900m)`, turn: 'straight' },
        { lat: originLat + 0.0105, lng: originLng + 0.0090, instruction: `In 250m, Turn Right onto Main Arterial Road`, turn: 'right' },
        { lat: originLat + 0.0125, lng: originLng + 0.0145, instruction: `Turn Right onto ${localityText} Main Road`, turn: 'right' },
        { lat: originLat + 0.0155, lng: originLng + 0.0180, instruction: `Turn Left near landmark: ${landmarkText}`, turn: 'left' },
        { lat: destLat, lng: destLng, instruction: `Arriving at ${dest.canonical_name} on right`, turn: 'arrive' }
      ];

      for (let i = 0; i < keyWps.length - 1; i++) {
        const wp1 = keyWps[i];
        const wp2 = keyWps[i + 1];
        const count = 35;
        for (let s = 0; s < count; s++) {
          const t = s / count;
          points.push({
            lat: wp1.lat + (wp2.lat - wp1.lat) * t,
            lng: wp1.lng + (wp2.lng - wp1.lng) * t,
            instruction: wp1.instruction,
            turn: wp1.turn
          });
        }
      }
      points.push({
        lat: destLat,
        lng: destLng,
        instruction: `Arrived at ${dest.canonical_name}!`,
        turn: 'arrive'
      });
    }

    const totalDist = calculateDistanceKm(originLat, originLng, destLat, destLng) * 1.32;
    const resultRoute = {
      origin: { lat: originLat, lng: originLng },
      destination: { lat: destLat, lng: destLng },
      steps: points,
      totalDistanceKm: totalDist,
      instructions
    };

    setRouteData(resultRoute);
    setIsLoadingRoute(false);
    return resultRoute;
  }, []);

  // When gpsDestinationId or isGpsModeActive changes, automatically start GPS travel!
  useEffect(() => {
    if (!targetEntity) return;

    if (gpsDestinationId || isGpsModeActive) {
      buildRouteForEntity(targetEntity).then(() => {
        setIsGpsActive(true);
        setIsNavigating(true);
        setCurrentStepIndex(0);
        setHasArrived(false);
        speakGuidance(`Starting GPS navigation to ${targetEntity.canonical_name}`, isVoiceMuted);
      });
    }
  }, [gpsDestinationId, isGpsModeActive, targetEntity, buildRouteForEntity, isVoiceMuted]);

  // Render normal dataset markers when not in GPS navigation
  useEffect(() => {
    const map = mapInstanceRef.current;
    const layer = markersLayerRef.current;
    if (!map || !layer || !isMapReady) return;

    layer.clearLayers();
    if (!results || results.length === 0) return;

    const bounds = L.latLngBounds();

    results.forEach((entity, idx) => {
      if (!entity.lat || !entity.lng) return;

      const isSelected = entity.canonical_id === targetEntity?.canonical_id;
      const jitter = getJitter(entity.canonical_id, idx);
      const entityLat = entity.lat + jitter.lat;
      const entityLng = entity.lng + jitter.lng;

      bounds.extend([entityLat, entityLng]);

      let pinColor = '#6366f1';
      let pinEmoji = '🏬';
      if (entity.category === 'food_dining') { pinColor = '#10b981'; pinEmoji = '☕'; }
      else if (entity.category === 'corporate_tech') { pinColor = '#06b6d4'; pinEmoji = '💻'; }
      else if (entity.category === 'healthcare_pharma') { pinColor = '#f43f5e'; pinEmoji = '🏥'; }
      else if (entity.category === 'real_estate_premises') { pinColor = '#a855f7'; pinEmoji = '🏢'; }
      else if (entity.category === 'retail_shop') { pinColor = '#f59e0b'; pinEmoji = '🛍️'; }

      const customIcon = L.divIcon({
        className: 'custom-real-marker',
        html: `
          <div style="
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            width: ${isSelected ? '42px' : '32px'};
            height: ${isSelected ? '42px' : '32px'};
            background: ${isSelected ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)' : pinColor};
            border: 2px solid ${isSelected ? '#ffffff' : 'rgba(255,255,255,0.85)'};
            border-radius: 50%;
            box-shadow: ${isSelected ? '0 0 25px rgba(16,185,129,0.95)' : '0 4px 12px rgba(0,0,0,0.6)'};
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
          ">
            <span style="font-size: ${isSelected ? '18px' : '14px'}; line-height: 1;">${pinEmoji}</span>
            ${isSelected ? `
              <div style="
                position: absolute;
                inset: -7px;
                border: 2px solid #34d399;
                border-radius: 50%;
                animation: pulse 1.6s infinite;
              "></div>
            ` : ''}
          </div>
        `,
        iconSize: [isSelected ? 42 : 32, isSelected ? 42 : 32],
        iconAnchor: [isSelected ? 21 : 16, isSelected ? 21 : 16]
      });

      const marker = L.marker([entityLat, entityLng], { icon: customIcon });

      const flag = entity.country === 'India' ? '🇮🇳' : (entity.country === 'France' ? '🇫🇷' : '🇺🇸');
      const popupHtml = `
        <div style="min-width: 240px; max-width: 290px; color: #f8fafc; font-family: inherit;">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-size: 0.72rem; font-weight: 700; color: #34d399; background: rgba(16,185,129,0.15); padding: 2px 8px; border-radius: 4px;">
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
          <div style="display: flex; align-items: center; justify-content: space-between; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 6px;">
            <span style="color: #34d399; font-size: 0.75rem; font-weight: 700;">
              ✓ ${entity.confidence_pct || 96}% Consensus
            </span>
            <span style="color: #fbbf24; font-size: 0.75rem; font-weight: 700;">
              ⭐ ${entity.rating || 4.5}
            </span>
          </div>
        </div>
      `;

      marker.bindPopup(popupHtml, { className: 'dark-leaflet-popup', closeButton: false });
      marker.on('click', () => {
        if (onSelectEntity) onSelectEntity(entity.canonical_id);
      });
      marker.addTo(layer);
    });

    if (!isGpsActive) {
      if (targetEntity?.lat && targetEntity?.lng) {
        const jitter = getJitter(targetEntity.canonical_id, 0);
        map.flyTo([targetEntity.lat + jitter.lat, targetEntity.lng + jitter.lng], 13, { duration: 1.0 });
      } else if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [50, 50], maxZoom: 12 });
      }
    }
  }, [results, targetEntity, isMapReady, isGpsActive, onSelectEntity]);

  // Draw Route Polyline & Vehicle when in GPS Mode
  useEffect(() => {
    const map = mapInstanceRef.current;
    const rLayer = routeLayerRef.current;
    if (!map || !rLayer || !isMapReady) return;

    rLayer.clearLayers();

    if (!isGpsActive || !routeData || !routeData.steps || routeData.steps.length === 0) return;

    const latLngs = routeData.steps.map(s => [s.lat, s.lng]);

    // Outer Neon Glow Polyline
    L.polyline(latLngs, {
      color: '#10b981',
      weight: 10,
      opacity: 0.35,
      lineCap: 'round',
      lineJoin: 'round'
    }).addTo(rLayer);

    // Inner Crisp Core Polyline
    L.polyline(latLngs, {
      color: '#34d399',
      weight: 4,
      opacity: 0.95,
      dashArray: '8, 6',
      lineCap: 'round'
    }).addTo(rLayer);

    // Origin Starting Flag Marker
    const startIcon = L.divIcon({
      className: 'gps-start-pin',
      html: `
        <div style="background: #10b981; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid white; box-shadow: 0 0 20px rgba(16,185,129,0.85); font-size: 14px;">
          🏁
        </div>
      `,
      iconSize: [32, 32],
      iconAnchor: [16, 16]
    });
    L.marker([routeData.origin.lat, routeData.origin.lng], { icon: startIcon }).addTo(rLayer);

    // Target Destination Flag Marker
    const destIcon = L.divIcon({
      className: 'gps-dest-pin',
      html: `
        <div style="background: #ef4444; color: white; width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid white; box-shadow: 0 0 25px rgba(239,68,68,0.95); font-size: 18px;">
          📍
        </div>
      `,
      iconSize: [38, 38],
      iconAnchor: [19, 19]
    });
    L.marker([routeData.destination.lat, routeData.destination.lng], { icon: destIcon }).addTo(rLayer);

    // Vehicle Marker with Real-time Rotational Heading
    const curStep = routeData.steps[currentStepIndex] || routeData.steps[0];
    const nxtStep = routeData.steps[Math.min(currentStepIndex + 1, routeData.steps.length - 1)];
    const initialHeading = calculateBearing(curStep.lat, curStep.lng, nxtStep.lat, nxtStep.lng);

    const vehicleIcon = L.divIcon({
      className: 'gps-active-car-marker',
      html: `
        <div style="
          position: relative;
          width: 50px;
          height: 50px;
          display: flex;
          align-items: center;
          justify-content: center;
          transform: rotate(${initialHeading}deg);
          transition: transform 0.12s ease-out;
        ">
          <!-- Headlights Cone Beam -->
          <div style="
            position: absolute;
            top: -30px;
            width: 36px;
            height: 38px;
            background: linear-gradient(to top, rgba(56, 189, 248, 0.6) 0%, rgba(56, 189, 248, 0.0) 100%);
            clip-path: polygon(30% 100%, 70% 100%, 100% 0%, 0% 0%);
            pointer-events: none;
          "></div>
          <!-- Vehicle Disc -->
          <div style="
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
            border: 2px solid #ffffff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 25px rgba(16, 185, 129, 0.95);
            font-size: 18px;
          ">
            🚘
          </div>
        </div>
      `,
      iconSize: [50, 50],
      iconAnchor: [25, 25]
    });

    const vMarker = L.marker([curStep.lat, curStep.lng], { icon: vehicleIcon, zIndexOffset: 2000 });
    vMarker.addTo(rLayer);
    vehicleMarkerRef.current = vMarker;

    // Center map view on starting area
    map.fitBounds(L.latLngBounds(latLngs), { padding: [70, 70], maxZoom: 15 });
  }, [isGpsActive, routeData, isMapReady]);

  // GPS Simulation Animation Loop
  useEffect(() => {
    if (!isGpsActive || !isNavigating || !routeData || !routeData.steps) return;

    const intervalTime = Math.max(30, 80 / simulationSpeed);
    const timer = setInterval(() => {
      setCurrentStepIndex(prev => {
        if (prev >= routeData.steps.length - 1) {
          setIsNavigating(false);
          setHasArrived(true);
          speakGuidance(`You have arrived at ${targetEntity?.canonical_name}`, isVoiceMuted);
          return prev;
        }

        const next = prev + 1;
        const cur = routeData.steps[next];
        const ahead = routeData.steps[Math.min(next + 1, routeData.steps.length - 1)];
        const heading = calculateBearing(cur.lat, cur.lng, ahead.lat, ahead.lng);

        // Update car position & rotation
        if (vehicleMarkerRef.current) {
          vehicleMarkerRef.current.setLatLng([cur.lat, cur.lng]);
          const elem = vehicleMarkerRef.current.getElement();
          if (elem) {
            const inner = elem.querySelector('.gps-active-car-marker > div') || elem.firstChild;
            if (inner && inner.style) {
              inner.style.transform = `rotate(${heading}deg)`;
            }
          }
        }

        // Voice cue at major turn events
        if (cur.turn !== 'straight' && next !== lastAnnouncedStep && next % 30 === 0) {
          speakGuidance(cur.instruction, isVoiceMuted);
          setLastAnnouncedStep(next);
        }

        // Camera follow
        if (cameraFollow && mapInstanceRef.current) {
          mapInstanceRef.current.setView([cur.lat, cur.lng], 16, { animate: true, duration: 0.1 });
        }

        return next;
      });
    }, intervalTime);

    return () => clearInterval(timer);
  }, [isGpsActive, isNavigating, routeData, simulationSpeed, cameraFollow, isVoiceMuted, lastAnnouncedStep, targetEntity]);

  // User Actions
  const handleStartTravel = () => {
    if (!targetEntity) return;
    buildRouteForEntity(targetEntity).then(() => {
      setIsGpsActive(true);
      setIsNavigating(true);
      setCurrentStepIndex(0);
      setHasArrived(false);
      speakGuidance(`Navigating to ${targetEntity.canonical_name}`, isVoiceMuted);
    });
  };

  const handlePauseResume = () => {
    setIsNavigating(prev => !prev);
  };

  const handleReset = () => {
    setCurrentStepIndex(0);
    setHasArrived(false);
    setIsNavigating(false);
    if (vehicleMarkerRef.current && routeData) {
      vehicleMarkerRef.current.setLatLng([routeData.origin.lat, routeData.origin.lng]);
    }
  };

  const handleJumpToEnd = () => {
    if (!routeData || !routeData.steps) return;
    const lastIdx = routeData.steps.length - 1;
    setCurrentStepIndex(lastIdx);
    setIsNavigating(false);
    setHasArrived(true);
    const lastStep = routeData.steps[lastIdx];
    if (vehicleMarkerRef.current) {
      vehicleMarkerRef.current.setLatLng([lastStep.lat, lastStep.lng]);
    }
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setView([lastStep.lat, lastStep.lng], 17);
    }
    speakGuidance(`You have arrived at ${targetEntity?.canonical_name}`, isVoiceMuted);
  };

  const handleExitGps = () => {
    setIsGpsActive(false);
    setIsNavigating(false);
    setHasArrived(false);
    setCurrentStepIndex(0);
    if (routeLayerRef.current) {
      routeLayerRef.current.clearLayers();
    }
    if (onExitGps) onExitGps();
  };

  // Metrics calculation
  const curStep = routeData?.steps[currentStepIndex] || routeData?.steps[0];
  const progressRatio = routeData?.steps?.length ? (currentStepIndex / (routeData.steps.length - 1)) : 0;
  const remainingDistKm = routeData ? Math.max(0, (routeData.totalDistanceKm * (1 - progressRatio))).toFixed(2) : '0.00';
  const remainingTimeMins = Math.max(1, Math.round(parseFloat(remainingDistKm) * 1.6));
  const currentSpeedKmH = isNavigating ? Math.round(48 + (Math.sin(currentStepIndex * 0.15) * 12) * (simulationSpeed / 1.5)) : 0;

  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: '100%',
      minHeight: '520px',
      borderRadius: '20px',
      overflow: 'hidden',
      border: isGpsActive ? '2px solid #10b981' : '1px solid rgba(99, 102, 241, 0.35)',
      boxShadow: isGpsActive ? '0 0 35px rgba(16, 185, 129, 0.35)' : '0 20px 50px rgba(0,0,0,0.8)',
      background: '#070a12',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* TOP FLOATING OVERLAY BAR */}
      <div style={{
        position: 'absolute',
        top: '14px',
        left: '14px',
        right: '14px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'rgba(11, 15, 25, 0.92)',
        backdropFilter: 'blur(16px)',
        border: '1px solid rgba(255, 255, 255, 0.12)',
        borderRadius: '14px',
        padding: '10px 16px',
        zIndex: 1000,
        boxShadow: '0 8px 30px rgba(0,0,0,0.6)'
      }}>
        {/* Active Destination Information */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', minWidth: 0 }}>
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '10px',
            background: isGpsActive ? 'linear-gradient(135deg, #10b981, #059669)' : 'linear-gradient(135deg, #6366f1, #4f46e5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: isGpsActive ? '0 0 15px rgba(16,185,129,0.7)' : '0 0 15px rgba(99,102,241,0.5)',
            flexShrink: 0
          }}>
            {isGpsActive ? <Car size={18} color="#ffffff" /> : <MapPin size={18} color="#ffffff" />}
          </div>

          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.95rem', fontWeight: 800, color: '#ffffff', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {targetEntity?.canonical_name || 'Select a Destination'}
              </span>
              {targetEntity && (
                <span className="badge-verified" style={{ padding: '1px 7px', fontSize: '0.68rem' }}>
                  {targetEntity.confidence_pct || 96}% Match
                </span>
              )}
            </div>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {targetEntity?.golden_address || 'Search and select any business from the real dataset'}
            </div>
          </div>
        </div>

        {/* Action Controls & Tile Switcher */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0 }}>
          {!isGpsActive ? (
            <button
              onClick={handleStartTravel}
              disabled={isLoadingRoute}
              style={{
                background: 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
                color: '#ffffff',
                border: 'none',
                borderRadius: '10px',
                padding: '8px 16px',
                fontSize: '0.82rem',
                fontWeight: 800,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                boxShadow: '0 0 20px rgba(16, 185, 129, 0.65)',
                transition: 'all 0.2s ease'
              }}
            >
              <Navigation size={15} /> 
              {isLoadingRoute ? 'Routing...' : 'Start GPS Travel'}
            </button>
          ) : (
            <button
              onClick={handleExitGps}
              style={{
                background: 'rgba(239, 68, 68, 0.2)',
                border: '1px solid #ef4444',
                color: '#fca5a5',
                borderRadius: '10px',
                padding: '7px 14px',
                fontSize: '0.78rem',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <X size={14} /> Exit GPS Mode
            </button>
          )}

          {/* Tile Layer Selector */}
          <div style={{
            display: 'flex',
            background: 'rgba(0,0,0,0.6)',
            borderRadius: '8px',
            padding: '3px',
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
                  borderRadius: '6px',
                  padding: '4px 8px',
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

      {/* TOP-LEFT GPS TURN-BY-TURN HUD (Active During Travel) */}
      {isGpsActive && curStep && (
        <div style={{
          position: 'absolute',
          top: '74px',
          left: '14px',
          zIndex: 1000,
          background: 'rgba(11, 15, 25, 0.94)',
          backdropFilter: 'blur(20px)',
          border: '1px solid #34d399',
          borderRadius: '16px',
          padding: '14px 18px',
          boxShadow: '0 12px 35px rgba(16, 185, 129, 0.35)',
          maxWidth: '380px',
          color: '#ffffff'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{
              width: '44px',
              height: '44px',
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 15px rgba(16, 185, 129, 0.7)',
              flexShrink: 0
            }}>
              {curStep.turn === 'right' ? <CornerUpRight size={24} color="#ffffff" /> : 
               curStep.turn === 'left' ? <CornerUpLeft size={24} color="#ffffff" /> : 
               curStep.turn === 'arrive' ? <Flag size={24} color="#facc15" /> : 
               <ArrowUpRight size={24} color="#ffffff" />}
            </div>

            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2px' }}>
                <span style={{ fontSize: '0.68rem', color: '#34d399', textTransform: 'uppercase', fontWeight: 800, letterSpacing: '0.6px' }}>
                  {curStep.turn === 'arrive' ? 'Target Destination' : 'Next Driving Maneuver'}
                </span>
                <button
                  onClick={() => setIsVoiceMuted(prev => !prev)}
                  title={isVoiceMuted ? 'Unmute GPS Voice' : 'Mute GPS Voice'}
                  style={{ background: 'none', border: 'none', color: isVoiceMuted ? '#94a3b8' : '#34d399', cursor: 'pointer' }}
                >
                  {isVoiceMuted ? <VolumeX size={15} /> : <Volume2 size={15} />}
                </button>
              </div>

              <div style={{ fontSize: '0.92rem', fontWeight: 700, lineHeight: 1.3 }}>
                {curStep.instruction}
              </div>
            </div>
          </div>

          {/* Progress bar */}
          <div style={{ marginTop: '12px', background: 'rgba(255,255,255,0.08)', height: '5px', borderRadius: '3px', overflow: 'hidden' }}>
            <div style={{
              background: 'linear-gradient(90deg, #10b981 0%, #34d399 100%)',
              height: '100%',
              width: `${progressRatio * 100}%`,
              transition: 'width 0.1s linear',
              boxShadow: '0 0 10px #34d399'
            }}></div>
          </div>
        </div>
      )}

      {/* BOTTOM GPS COCKPIT CONTROL BAR */}
      {isGpsActive && (
        <div style={{
          position: 'absolute',
          bottom: '14px',
          left: '14px',
          right: '14px',
          zIndex: 1000,
          background: 'rgba(11, 15, 25, 0.95)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: '16px',
          padding: '12px 20px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          boxShadow: '0 16px 40px rgba(0,0,0,0.85)'
        }}>
          {/* Trip Metrics */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '22px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Gauge size={22} color="#34d399" />
              <div>
                <div style={{ fontSize: '0.65rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>Speed</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#ffffff' }}>{currentSpeedKmH} <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>km/h</span></div>
              </div>
            </div>

            <div style={{ width: '1px', height: '28px', background: 'rgba(255,255,255,0.1)' }}></div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Navigation size={22} color="#38bdf8" />
              <div>
                <div style={{ fontSize: '0.65rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>Remaining</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#ffffff' }}>{remainingDistKm} <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>km</span></div>
              </div>
            </div>

            <div style={{ width: '1px', height: '28px', background: 'rgba(255,255,255,0.1)' }}></div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Clock size={22} color="#fbbf24" />
              <div>
                <div style={{ fontSize: '0.65rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>ETA</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#ffffff' }}>{remainingTimeMins} <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>min</span></div>
              </div>
            </div>
          </div>

          {/* Navigation Cockpit Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              onClick={() => setCameraFollow(prev => !prev)}
              style={{
                background: cameraFollow ? 'rgba(16,185,129,0.2)' : 'rgba(255,255,255,0.05)',
                border: cameraFollow ? '1px solid #10b981' : '1px solid rgba(255,255,255,0.1)',
                color: cameraFollow ? '#34d399' : '#94a3b8',
                borderRadius: '8px',
                padding: '6px 12px',
                fontSize: '0.75rem',
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '5px'
              }}
            >
              <LocateFixed size={14} /> Follow Car
            </button>

            {/* Speed multipliers */}
            <div style={{ display: 'flex', background: 'rgba(0,0,0,0.5)', borderRadius: '8px', padding: '2px' }}>
              {[1, 2, 5].map(s => (
                <button
                  key={s}
                  onClick={() => setSimulationSpeed(s)}
                  style={{
                    background: simulationSpeed === s ? '#10b981' : 'transparent',
                    color: simulationSpeed === s ? '#ffffff' : '#94a3b8',
                    border: 'none',
                    borderRadius: '6px',
                    padding: '4px 8px',
                    fontSize: '0.72rem',
                    fontWeight: 700,
                    cursor: 'pointer'
                  }}
                >
                  {s}x
                </button>
              ))}
            </div>

            <button
              onClick={handlePauseResume}
              style={{
                background: isNavigating ? 'rgba(245,158,11,0.2)' : 'rgba(16,185,129,0.2)',
                border: isNavigating ? '1px solid #f59e0b' : '1px solid #10b981',
                color: isNavigating ? '#fbbf24' : '#34d399',
                borderRadius: '8px',
                padding: '6px 14px',
                fontSize: '0.78rem',
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
              onClick={handleJumpToEnd}
              title="Fast forward to arrival"
              style={{
                background: 'rgba(56,189,248,0.15)',
                border: '1px solid rgba(56,189,248,0.4)',
                color: '#38bdf8',
                borderRadius: '8px',
                padding: '6px 10px',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}
            >
              <FastForward size={14} /> Arrive
            </button>

            <button
              onClick={handleReset}
              title="Restart route"
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
      {hasArrived && targetEntity && (
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'rgba(4, 7, 14, 0.82)',
          backdropFilter: 'blur(10px)',
          zIndex: 2500,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px'
        }}>
          <div style={{
            background: 'linear-gradient(135deg, #0b1120 0%, #111e38 100%)',
            border: '2px solid #34d399',
            boxShadow: '0 0 50px rgba(52,211,153,0.45)',
            borderRadius: '20px',
            padding: '28px 36px',
            textAlign: 'center',
            maxWidth: '480px',
            color: '#ffffff'
          }}>
            <div style={{ fontSize: '42px', marginBottom: '8px' }}>🏁 🎉</div>
            <h3 style={{ fontSize: '1.45rem', fontWeight: 800, marginBottom: '6px' }}>
              You Have Arrived!
            </h3>
            <p style={{ fontSize: '1.15rem', color: '#34d399', fontWeight: 800, marginBottom: '4px' }}>
              {targetEntity.canonical_name}
            </p>
            <p style={{ fontSize: '0.84rem', color: '#94a3b8', marginBottom: '18px', lineHeight: 1.4 }}>
              📍 {targetEntity.golden_address}
            </p>

            <div style={{
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: '12px',
              padding: '12px',
              fontSize: '0.8rem',
              color: '#cbd5e1',
              marginBottom: '22px',
              display: 'flex',
              justifyContent: 'space-around'
            }}>
              <div>
                <span style={{ color: '#94a3b8', display: 'block', fontSize: '0.7rem', fontWeight: 600 }}>DOOR / UNIT</span>
                <strong>{targetEntity.building_number || 'N/A'}</strong>
              </div>
              <div style={{ borderLeft: '1px solid rgba(255,255,255,0.1)' }}></div>
              <div>
                <span style={{ color: '#94a3b8', display: 'block', fontSize: '0.7rem', fontWeight: 600 }}>LANDMARK</span>
                <strong>{targetEntity.landmark || 'Main Road'}</strong>
              </div>
              <div style={{ borderLeft: '1px solid rgba(255,255,255,0.1)' }}></div>
              <div>
                <span style={{ color: '#94a3b8', display: 'block', fontSize: '0.7rem', fontWeight: 600 }}>CONSENSUS</span>
                <strong style={{ color: '#34d399' }}>{targetEntity.confidence_pct || 96}%</strong>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
              <button
                onClick={() => {
                  setHasArrived(false);
                  if (onOpenDetails) onOpenDetails(targetEntity.canonical_id);
                }}
                className="btn-primary"
                style={{ padding: '9px 18px', fontSize: '0.85rem', borderRadius: '10px' }}
              >
                Inspect Entity Graph & Sources
              </button>
              <button
                onClick={() => setHasArrived(false)}
                className="btn-secondary"
                style={{ padding: '9px 18px', fontSize: '0.85rem', borderRadius: '10px' }}
              >
                Close View
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Leaflet Map Target DOM */}
      <div 
        ref={mapContainerRef} 
        style={{ width: '100%', height: '100%', minHeight: '520px', flex: 1, zIndex: 1 }} 
      />

      {/* Non-GPS Map Category Legend */}
      {!isGpsActive && (
        <div style={{
          position: 'absolute',
          bottom: '14px',
          left: '14px',
          background: 'rgba(11, 15, 25, 0.88)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '10px',
          padding: '8px 14px',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          gap: '14px',
          fontSize: '0.75rem',
          color: '#94a3b8'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ color: '#f59e0b' }}>🛍️</span> Retail & Shops
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ color: '#10b981' }}>☕</span> Food & Cafes
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ color: '#06b6d4' }}>💻</span> Tech & Corporate
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ color: '#a855f7' }}>🏢</span> Complexes
          </div>
        </div>
      )}
    </div>
  );
}
