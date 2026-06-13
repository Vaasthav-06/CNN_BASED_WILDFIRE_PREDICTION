// src/App.jsx
import { useState, useEffect, useCallback } from 'react'
import Navbar       from './components/Navbar.jsx'
import RiskMap      from './components/RiskMap.jsx'
import RegionDrawer from './components/RegionDrawer.jsx'
import LayerToggle  from './components/LayerToggle.jsx'
import {
  fetchHealth, fetchAllRegionsRisk, fetchRegionsGeo,
  fetchRegionFirms, predictTabular,
} from './api.js'

const DEFAULT_FORM = {
  region:                     'similipal',
  temperature_2m_mean:         38.5,
  temperature_2m_max:          43.0,
  relative_humidity_2m_mean:   18.0,
  precipitation_sum:            0.0,
  wind_speed_10m_max:          35.0,
  wind_speed_10m_mean:         18.0,
  soil_moisture_0_1cm_mean:     0.08,
  vpd:                          4.2,
}

const RISK_COLORS = {
  Low: '#22c55e', Moderate: '#eab308', High: '#f97316', Extreme: '#ef4444',
}

export default function App() {
  const [health,     setHealth]     = useState(null)
  const [geoFeats,   setGeoFeats]   = useState([])   // RegionGeoFeature[]
  const [riskMap,    setRiskMap]    = useState({})    // { region → RiskResponse }
  const [firmsMap,   setFirmsMap]   = useState({})   // { region → FirePoint[] }
  const [selected,   setSelected]   = useState(null)
  const [layers,     setLayers]     = useState({
    regions: true, heatmap: true, pulses: true, labels: true,
  })
  const [form,       setForm]       = useState(DEFAULT_FORM)
  const [result,     setResult]     = useState(null)
  const [loading,    setLoading]    = useState(false)
  const [predError,  setPredError]  = useState(null)
  const [activeTab,  setActiveTab]  = useState('map')  // 'map' | 'predict'

  // ── Initial data load ──────────────────────────────────────────────────
  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => {})

    // Load geo + risk in parallel, then load FIRMS for all regions
    Promise.all([fetchRegionsGeo(), fetchAllRegionsRisk()])
      .then(([geo, risk]) => {
        setGeoFeats(geo.features ?? [])

        const rm = {}
        ;(risk.regions ?? []).forEach(r => { rm[r.region] = r })
        setRiskMap(rm)

        // Load FIRMS for every region concurrently
        const regions = (geo.features ?? []).map(f => f.region)
        regions.forEach(r => {
          fetchRegionFirms(r)
            .then(d => setFirmsMap(prev => ({ ...prev, [r]: d.points ?? [] })))
            .catch(() => {})
        })
      })
      .catch(console.error)
  }, [])

  // ── Layer toggle ───────────────────────────────────────────────────────
  const handleLayerChange = useCallback((key, val) => {
    setLayers(prev => ({ ...prev, [key]: val }))
  }, [])

  // ── Map click ─────────────────────────────────────────────────────────
  const handleMapSelect = useCallback(regionKey => {
    setSelected(prev => prev === regionKey ? null : regionKey)
    setForm(f => ({ ...f, region: regionKey }))
  }, [])

  // ── Prediction ────────────────────────────────────────────────────────
  async function handlePredict() {
    setLoading(true); setPredError(null); setResult(null)
    try {
      setResult(await predictTabular(form))
    } catch (e) {
      setPredError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleInput(e) {
    const { name, value } = e.target
    setForm(f => ({ ...f, [name]: isNaN(value) || value === '' ? value : parseFloat(value) }))
  }

  // ── Drawer geo for selected region ────────────────────────────────────
  const selectedGeo  = geoFeats.find(f => f.region === selected) ?? null
  const selectedRisk = selected ? riskMap[selected] ?? null : null

  const maxImp = result?.feature_importances
    ? Math.max(...Object.values(result.feature_importances))
    : 1

  return (
    <div className="app">
      <Navbar health={health} />

      <div className="main">
        {/* ── Map panel ─────────────────────────────────────── */}
        <div className="map-panel">
          <RiskMap
            geoFeatures = {geoFeats}
            riskMap     = {riskMap}
            firmsPoints = {firmsMap}
            selected    = {selected}
            onSelect    = {handleMapSelect}
            layers      = {layers}
          />

          {/* Layer toggle — floats top-left over map */}
          <div className="map-overlay-tl">
            <LayerToggle layers={layers} onChange={handleLayerChange} />
          </div>

          {/* Region drawer — slides in from right over map on mobile,
              or appears as map overlay panel on desktop */}
          {selectedGeo && (
            <div className="drawer-overlay">
              <RegionDrawer
                geo      = {selectedGeo}
                riskData = {selectedRisk}
                onClose  = {() => setSelected(null)}
              />
            </div>
          )}
        </div>

        {/* ── Sidebar ───────────────────────────────────────── */}
        <aside className="sidebar">

          {/* Tab switcher */}
          <div className="tab-bar">
            <button
              className={`tab-btn ${activeTab === 'map' ? 'active' : ''}`}
              onClick={() => setActiveTab('map')}
            >
              Risk Overview
            </button>
            <button
              className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`}
              onClick={() => setActiveTab('predict')}
            >
              Predict
            </button>
          </div>

          {/* ── Risk overview tab ──────────────────────────── */}
          {activeTab === 'map' && (
            <div className="sidebar-section">
              <h3>Region Risk</h3>
              {Object.keys(riskMap).length === 0
                ? <p className="loading">Loading…</p>
                : geoFeats.map(f => {
                  const r = riskMap[f.region]
                  if (!r) return null
                  return (
                    <div
                      key={f.region}
                      className={`region-card ${selected === f.region ? 'active' : ''}`}
                      onClick={() => handleMapSelect(f.region)}
                    >
                      <div>
                        <div className="region-name">{f.name}</div>
                        <div className="region-prob">
                          P(fire) = {(r.fire_probability * 100).toFixed(2)}%
                          <span style={{ marginLeft: 6, color: 'var(--muted)' }}>
                            · {f.state}
                          </span>
                        </div>
                      </div>
                      <span className={`risk-badge risk-${r.fire_risk_label}`}>
                        {r.fire_risk_label}
                      </span>
                    </div>
                  )
                })
              }

              {/* Legend */}
              <div className="legend">
                <div className="legend-title">RISK LEGEND</div>
                {Object.entries(RISK_COLORS).map(([label, color]) => (
                  <div key={label} className="legend-row">
                    <span className="legend-dot" style={{ background: color }} />
                    <span>{label}</span>
                    <span className="legend-range">
                      {label === 'Low'      ? '0–25%'
                      : label === 'Moderate' ? '25–50%'
                      : label === 'High'     ? '50–75%'
                      : '75–100%'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Predict tab ────────────────────────────────── */}
          {activeTab === 'predict' && (
            <div className="sidebar-section">
              <h3>Predict Fire Risk</h3>
              <div className="predict-form">
                <div className="form-row">
                  <label>REGION</label>
                  <select name="region" value={form.region} onChange={handleInput}>
                    {geoFeats.map(f =>
                      <option key={f.region} value={f.region}>{f.name}</option>
                    )}
                  </select>
                </div>
                {[
                  ['temperature_2m_mean',       'Temp Mean (°C)'],
                  ['temperature_2m_max',         'Temp Max (°C)'],
                  ['relative_humidity_2m_mean',  'Humidity (%)'],
                  ['precipitation_sum',          'Precipitation (mm)'],
                  ['wind_speed_10m_max',         'Wind Max (km/h)'],
                  ['wind_speed_10m_mean',        'Wind Mean (km/h)'],
                  ['soil_moisture_0_1cm_mean',   'Soil Moisture (m³/m³)'],
                  ['vpd',                        'VPD (kPa)'],
                ].map(([key, label]) => (
                  <div className="form-row" key={key}>
                    <label>{label.toUpperCase()}</label>
                    <input
                      type="number" name={key}
                      value={form[key]} onChange={handleInput} step="any"
                    />
                  </div>
                ))}
                <button
                  className="btn-predict"
                  onClick={handlePredict}
                  disabled={loading}
                >
                  {loading ? 'Running…' : 'Run Prediction'}
                </button>
                {predError && <p className="error-msg">{predError}</p>}
              </div>

              {result && (
                <div className="result-box">
                  <div
                    className="result-prob"
                    style={{ color: RISK_COLORS[result.fire_risk_label] }}
                  >
                    {(result.fire_probability * 100).toFixed(2)}%
                  </div>
                  <div className="result-label">Fire Probability</div>
                  <div className="result-meta">
                    Risk: <strong>{result.fire_risk_label}</strong>
                    {' · '}Confidence: {(result.confidence * 100).toFixed(1)}%
                    <br />Model: {result.model_used}
                  </div>
                  {result.feature_importances && (
                    <div style={{ marginTop: '1rem' }}>
                      <div style={{
                        fontSize: '0.65rem', color: 'var(--muted)',
                        letterSpacing: '0.08em', marginBottom: '0.6rem'
                      }}>
                        TOP FEATURE IMPORTANCES
                      </div>
                      {Object.entries(result.feature_importances).map(([k, v]) => (
                        <div className="imp-row" key={k}>
                          <span className="imp-name">{k}</span>
                          <div className="imp-bar-bg">
                            <div
                              className="imp-bar-fill"
                              style={{ width: `${(v / maxImp) * 100}%` }}
                            />
                          </div>
                          <span className="imp-val">{v.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}