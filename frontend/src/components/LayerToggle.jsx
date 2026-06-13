// src/components/LayerToggle.jsx

export default function LayerToggle({ layers, onChange }) {
  // layers: { regions: bool, heatmap: bool, pulses: bool }
  const items = [
    { key: 'regions',  label: 'Region Bounds',    icon: '⬜' },
    { key: 'heatmap',  label: 'FIRMS Heatmap',    icon: '🔥' },
    { key: 'pulses',   label: 'Risk Pulses',       icon: '📡' },
    { key: 'labels',   label: 'Region Labels',     icon: '🏷️' },
  ]

  return (
    <div className="layer-toggle-panel">
      <div className="layer-toggle-title">MAP LAYERS</div>
      {items.map(({ key, label, icon }) => (
        <label key={key} className="layer-toggle-row">
          <span className="layer-icon">{icon}</span>
          <span className="layer-label">{label}</span>
          <div
            className={`toggle-switch ${layers[key] ? 'on' : 'off'}`}
            onClick={() => onChange(key, !layers[key])}
          >
            <div className="toggle-knob" />
          </div>
        </label>
      ))}
    </div>
  )
}