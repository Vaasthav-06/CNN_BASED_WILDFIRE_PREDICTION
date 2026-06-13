// src/components/RiskSparkline.jsx

export default function RiskSparkline({ data = [], width = 260, height = 48 }) {
  if (!data || data.length < 2) {
    return (
      <div style={{ color: 'var(--muted)', fontSize: '0.7rem', padding: '8px 0' }}>
        No timeline data
      </div>
    )
  }

  const counts  = data.map(d => d.count)
  const maxVal  = Math.max(...counts, 1)
  const minVal  = Math.min(...counts)
  const pad     = 4

  const pts = data.map((d, i) => {
    const x = pad + (i / (data.length - 1)) * (width - pad * 2)
    const y = pad + (1 - (d.count - minVal) / (maxVal - minVal || 1)) * (height - pad * 2)
    return [x, y]
  })

  const pathD = pts
    .map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    .join(' ')

  // Filled area under the line
  const areaD =
    pathD +
    ` L${pts[pts.length - 1][0].toFixed(1)},${height - pad}` +
    ` L${pts[0][0].toFixed(1)},${height - pad} Z`

  // Find peak month
  const peakIdx = counts.indexOf(maxVal)
  const [px, py] = pts[peakIdx]

  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: '100%', height: height, display: 'block', overflow: 'visible' }}
      >
        <defs>
          <linearGradient id="spark-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="var(--accent)" stopOpacity="0.4" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {/* Area fill */}
        <path d={areaD} fill="url(#spark-grad)" />
        {/* Line */}
        <path d={pathD} fill="none" stroke="var(--accent)" strokeWidth="1.5" />
        {/* Peak dot */}
        <circle cx={px} cy={py} r="3" fill="var(--accent)" />
        <title>{data[peakIdx]?.month}: {maxVal} fires</title>
      </svg>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        fontSize: '0.6rem', color: 'var(--muted)', marginTop: 2
      }}>
        <span>{data[0]?.month}</span>
        <span style={{ color: 'var(--accent)' }}>peak {maxVal}</span>
        <span>{data[data.length - 1]?.month}</span>
      </div>
    </div>
  )
}