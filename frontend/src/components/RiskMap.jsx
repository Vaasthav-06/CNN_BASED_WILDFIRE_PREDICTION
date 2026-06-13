// src/components/RiskMap.jsx
import { useEffect, useRef } from 'react'
import {
  MapContainer, TileLayer, Rectangle, CircleMarker,
  Tooltip, useMap, Pane
} from 'react-leaflet'
import L from 'leaflet'

const RISK_COLORS = {
  Low:      '#22c55e',
  Moderate: '#eab308',
  High:     '#f97316',
  Extreme:  '#ef4444',
}

// ── Heatmap layer using Leaflet's built-in canvas renderer ─────────────────
function FirmsHeatLayer({ points }) {
  const map = useMap()
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!points || points.length === 0) return

    // Use Leaflet CircleMarkers on a dedicated pane for performance
    const pane = map.getPane('heatPane') || map.createPane('heatPane')
    pane.style.zIndex = 350

    const markers = points.map(p => {
      const intensity = Math.min(1, (p.frp || 1) / 50)
      return L.circleMarker([p.lat, p.lon], {
        radius:      3 + intensity * 5,
        fillColor:   `hsl(${30 - intensity * 30}, 100%, ${60 - intensity * 20}%)`,
        fillOpacity: 0.55 + intensity * 0.3,
        stroke:      false,
        pane:        'heatPane',
      })
    })

    const layer = L.layerGroup(markers).addTo(map)
    return () => { map.removeLayer(layer) }
  }, [map, points])

  return null
}

// ── Animated pulse rings for high/extreme risk ─────────────────────────────
function PulseLayer({ regions }) {
  const map = useMap()

  useEffect(() => {
    if (!regions || regions.length === 0) return
    const pulseMarkers = []

    regions
      .filter(r => r.fire_risk_label === 'High' || r.fire_risk_label === 'Extreme')
      .forEach(r => {
        const color = RISK_COLORS[r.fire_risk_label]
        const icon = L.divIcon({
          className: '',
          html: `<div class="pulse-ring" style="--pulse-color:${color}"></div>`,
          iconSize:   [40, 40],
          iconAnchor: [20, 20],
        })
        const m = L.marker([r.center.lat, r.center.lon], { icon, interactive: false })
        m.addTo(map)
        pulseMarkers.push(m)
      })

    return () => { pulseMarkers.forEach(m => map.removeLayer(m)) }
  }, [map, regions])

  return null
}

// ── Main map component ─────────────────────────────────────────────────────
export default function RiskMap({
  geoFeatures,    // list of RegionGeoFeature
  riskMap,        // { regionKey → RegionRiskResponse }
  firmsPoints,    // { regionKey → [FirePoint] } 
  selected,       // regionKey | null
  onSelect,
  layers,         // { regions, heatmap, pulses, labels }
}) {
  // Build enriched list for pulse layer
  const enrichedRegions = (geoFeatures || []).map(f => ({
    ...f,
    fire_risk_label: riskMap[f.region]?.fire_risk_label ?? 'Low',
  }))

  // Flatten all FIRMS points when heatmap is on
  const allPoints = layers.heatmap
    ? Object.values(firmsPoints || {}).flat()
    : []

  return (
    <MapContainer
      center={[24, 84]}
      zoom={5}
      scrollWheelZoom
      style={{ width: '100%', height: '100%' }}
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; <a href="https://carto.com/">CARTO</a>'
        maxZoom={18}
      />

      {/* FIRMS heatmap */}
      {layers.heatmap && <FirmsHeatLayer points={allPoints} />}

      {/* Pulse rings */}
      {layers.pulses && <PulseLayer regions={enrichedRegions} />}

      {/* Region bounding-box rectangles */}
      {layers.regions && (geoFeatures || []).map(f => {
        const risk  = riskMap[f.region]?.fire_risk_label ?? 'Low'
        const color = RISK_COLORS[risk]
        const isSelected = selected === f.region
        const b = f.bounds

        return (
          <Rectangle
            key={f.region}
            bounds={[[b.lat_min, b.lon_min], [b.lat_max, b.lon_max]]}
            pathOptions={{
              color,
              fillColor:   color,
              fillOpacity: isSelected ? 0.25 : 0.10,
              weight:      isSelected ? 2.5 : 1.5,
              dashArray:   isSelected ? undefined : '4 4',
            }}
            eventHandlers={{ click: () => onSelect(f.region) }}
          >
            {layers.labels && (
              <Tooltip
                permanent
                direction="center"
                className="region-tooltip"
              >
                <span style={{ fontSize: 11, fontWeight: 600 }}>
                  {f.name.split(' ')[0]}
                  <br />
                  <span style={{ color, fontWeight: 700 }}>{risk}</span>
                  {' '}{((riskMap[f.region]?.fire_probability ?? 0) * 100).toFixed(1)}%
                </span>
              </Tooltip>
            )}
          </Rectangle>
        )
      })}

      {/* Centre dot markers */}
      {(geoFeatures || []).map(f => {
        const risk  = riskMap[f.region]?.fire_risk_label ?? 'Low'
        const color = RISK_COLORS[risk]
        return (
          <CircleMarker
            key={`dot-${f.region}`}
            center={[f.center.lat, f.center.lon]}
            radius={selected === f.region ? 7 : 5}
            pathOptions={{
              color,
              fillColor:   color,
              fillOpacity: 0.9,
              weight:      2,
            }}
            eventHandlers={{ click: () => onSelect(f.region) }}
          />
        )
      })}
    </MapContainer>
  )
}