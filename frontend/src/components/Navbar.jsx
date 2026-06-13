export default function Navbar({ health }) {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="dot" />
        Wildfire Prediction System
      </div>
      <div className="navbar-status">
        {health
          ? <span className={health.status === 'ok' ? 'status-ok' : 'status-deg'}>
              {health.status === 'ok' ? '● API Online' : '⚠ API Degraded'}
            </span>
          : <span>Connecting…</span>
        }
      </div>
    </nav>
  )
}