const BASE = '/api'

async function _get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`)
  return res.json()
}

async function _post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`)
  return res.json()
}

export const fetchHealth           = ()           => _get('/health')
export const fetchAllRegionsRisk   = ()           => _get('/regions/risk')
export const fetchRegionsGeo       = ()           => _get('/regions/geo')
export const fetchRegionFirms      = (r, days)    => _get(`/regions/${r}/firms?days_back=${days ?? 1825}`)
export const fetchRegionTimeline   = (r)          => _get(`/regions/${r}/timeline`)
export const fetchRegionRisk       = (r)          => _get(`/regions/${r}/risk`)

export const predictTabular = (payload, model = 'ensemble') =>
  _post(`/predict/tabular?model=${model}&explain=true`, payload)