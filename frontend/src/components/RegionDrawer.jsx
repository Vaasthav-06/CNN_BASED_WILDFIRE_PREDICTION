// src/components/RegionDrawer.jsx
import { useEffect, useState } from 'react'
import { fetchRegionTimeline, fetchRegionRisk } from '../api.js'
import RiskSparkline from './RiskSparkline.jsx'

const MONTH_NAMES = [
  '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]

const RISK_COLORS = {
  Low:      '#22c55e',
  Moderate: '#eab308',
  High:     '#f97316',
  Extreme:  '#ef4444',
}

export default function RegionDrawer({ geo, riskData, onClose }) {
  // geo: RegionGeoFeature from /regions/geo
  // riskData: RegionRiskResponse from /regions/risk
  const [timeline, setTimeline] = useState([])
  const [loading,  setLoading]  = useState(true)

  useEffect(() => {
    if (!geo) return
    setLoading(true)
    fetchRegionTimeline(geo.region)
      .then(d => setTimeline(d.timeline || []))
      .catch(() => setTimeline([]))
      .finally(() => setLoading(false))
  }, [geo?.region])

  if (!geo) return null

  const risk   = riskData?.fire_risk_label ?? 'Low'
  const prob   = riskData?.fire_probability ?? 0
  const color  = RISK_COLORS[risk]

  return (
    <div className="region-drawer">
      <div className="drawer-header">
        <div>
          <div className="drawer-name">{geo.name}</div>
          <div className="drawer-meta">{geo.state} · {geo.area_sq_km.toLocaleString()} km²</div>
        </div>
        <button className="drawer-close" onClick={onClose}>✕</button>
      </div>

      {/* Risk hero */}
      <div className="drawer-risk-hero" style={{ borderColor: color }}>
        <div className="drawer-risk-prob" style={{ color }}>
          {(prob * 100).toFixed(1)}%
        </div>
        <div className="drawer-risk-label" style={{ color }}>
          {risk} Risk
        </div>
      </div>

      {/* Info grid */}
      <div className="drawer-grid">
        <div className="drawer-cell">
          <div className="drawer-cell-label">Forest Type</div>
          <div className="drawer-cell-value">{geo.forest_type}</div>
        </div>
        <div className="drawer-cell">
          <div className="drawer-cell-label">Fire Season</div>
          <div className="drawer-cell-value">
            {MONTH_NAMES[geo.fire_season_start]} – {MONTH_NAMES[geo.fire_season_end]}
          </div>
        </div>
        <div className="drawer-cell">
          <div className="drawer-cell-label">Avg Fires / Year</div>
          <div className="drawer-cell-value">{geo.avg_fires_per_year}</div>
        </div>
        <div className="drawer-cell">
          <div className="drawer-cell-label">Elevation Center</div>
          <div className="drawer-cell-value">
            {geo.center.lat.toFixed(3)}°N, {geo.center.lon.toFixed(3)}°E
          </div>
        </div>
      </div>

      {/* Sparkline */}
      <div className="drawer-section">
        <div className="drawer-section-title">FIRE ACTIVITY (36 MONTHS)</div>
        {loading
          ? <div className="loading">Loading timeline…</div>
          : <RiskSparkline data={timeline} />
        }
      </div>

      {/* Bounding box */}
      <div className="drawer-section">
        <div className="drawer-section-title">BOUNDING BOX</div>
        <div className="bbox-grid">
          {[
            ['N', geo.bounds.lat_max.toFixed(3) + '°'],
            ['S', geo.bounds.lat_min.toFixed(3) + '°'],
            ['E', geo.bounds.lon_max.toFixed(3) + '°'],
            ['W', geo.bounds.lon_min.toFixed(3) + '°'],
          ].map(([dir, val]) => (
            <div key={dir} className="bbox-cell">
              <span className="bbox-dir">{dir}</span>
              <span className="bbox-val">{val}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}