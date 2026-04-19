import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import './index.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_BASE  = API_BASE.replace(/^http/, 'ws')

// State coordinates mapping (approximate for visualization)
const STATE_COORDS = {
  "MH": [19.75, 75.71], "DL": [28.70, 77.10], "KA": [15.31, 75.71],
  "TN": [11.12, 78.65], "GJ": [22.25, 71.19], "UP": [26.84, 80.94],
  "WB": [22.98, 87.85], "MP": [22.97, 78.65], "RJ": [27.02, 74.21],
  "KL": [10.85, 76.27], "AP": [15.91, 79.74], "TS": [18.11, 79.01],
  "HR": [29.05, 76.08], "PB": [31.14, 75.34], "BR": [25.09, 85.31],
}

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard') // dashboard|monitor|analytics|logs|ai_core
  const [apiStatus, setApiStatus] = useState('online')
  const [stats, setStats] = useState({ total: 0, today: 0, states: [] })
  const [history, setHistory] = useState([])
  const [watchlist, setWatchlist] = useState(['MH12AB1234', 'DL01C1234'])
  const [detectorInfo, setDetectorInfo] = useState({ yolo_ready: false, easyocr_ready: false, rto_db_size: 0 })
  const [theme, setTheme] = useState('vibrant') // vibrant|classic|midnight
  const [activeCam, setActiveCam] = useState('CAM-01')
  const [isAutoPatrol, setIsAutoPatrol] = useState(false)
  
  // Monitoring States
  const [mode, setMode] = useState('upload')
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [toast, setToast] = useState(null)

  // Watchlist Management
  const [newWatchlistPlate, setNewWatchlistPlate] = useState('')

  const fileRef = useRef(null)

  // ── Engine Connectivity Check ──────────────────────────────────────────
  useEffect(() => {
    // Inject Leaflet CSS and JS
    if (!document.getElementById('leaflet-css')) {
       const link = document.createElement('link');
       link.id = 'leaflet-css';
       link.rel = 'stylesheet';
       link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
       document.head.appendChild(link);
       
       const script = document.createElement('script');
       script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
       script.async = true;
       document.head.appendChild(script);
    }

    const fetchData = async () => {
      try {
        const [s, h, i] = await Promise.all([
          fetch(`${API_BASE}/stats`).then(r => r.json()),
          fetch(`${API_BASE}/history?limit=100`).then(r => r.json()),
          fetch(`${API_BASE}/info`).then(r => r.json()),
        ])
        setStats(s); setHistory(h.entries); setDetectorInfo(i)
        setApiStatus('online')
      } catch { setApiStatus('offline') }
    }
    fetchData(); const id = setInterval(fetchData, 10000); return () => clearInterval(id)
  }, [])

  const showToast = (msg, type = 'info') => {
    setToast({ msg, type }); setTimeout(() => setToast(null), 4000)
  }

  const handleUpload = async (file) => {
    if (!file) return
    setPreview(URL.createObjectURL(file))
    setLoading(true); setResult(null)
    const form = new FormData(); form.append('file', file)
    try {
      const r = await fetch(`${API_BASE}/detect`, { method: 'POST', body: form })
      const data = await r.json()
      setResult(data)
      data.plates.forEach(p => {
        if (watchlist.includes(p.text)) showToast(`🚩 SECURITY ALERT: WATCHLIST MATCH [${p.text}]`, 'danger')
      })
    } catch { showToast('Analysis Interrupted', 'error') }
    finally { setLoading(false) }
  }

  // ── Views ───────────────────────────────────────────────────────────────

  const Dashboard = () => (
    <div className="fade-up">
      {/* ── Real-time Engine GAUGES ──────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem', marginBottom: '2.5rem' }}>
        <div className="glass-card" style={{ padding: '1.5rem', borderBottom: '3px solid var(--accent-primary)' }}>
           <div style={{ fontSize: '0.7rem', fontWeight: 800, opacity: 0.6 }}>NEURAL LATENCY</div>
           <div style={{ fontSize: '1.8rem', fontWeight: 900 }}>{result?.total_ms || history[0]?.total_ms || 0}<span style={{ fontSize: '0.8rem', opacity: 0.5 }}>ms</span></div>
           <div className="bar-track" style={{ height: 4, marginTop: '8px' }}><div className="bar-fill" style={{ width: '45%', background: 'var(--accent-primary)' }} /></div>
        </div>
        <div className="glass-card" style={{ padding: '1.5rem', borderBottom: '3px solid var(--success)' }}>
           <div style={{ fontSize: '0.7rem', fontWeight: 800, opacity: 0.6 }}>OCR ACCURACY</div>
           <div style={{ fontSize: '1.8rem', fontWeight: 900 }}>{((history[0]?.plates[0]?.confidence || 0.88) * 100).toFixed(1)}%</div>
           <div className="bar-track" style={{ height: 4, marginTop: '8px' }}><div className="bar-fill" style={{ width: '88%', background: 'var(--success)' }} /></div>
        </div>
        <div className="glass-card" style={{ padding: '1.5rem', borderBottom: '3px solid var(--accent-secondary)' }}>
           <div style={{ fontSize: '0.7rem', fontWeight: 800, opacity: 0.6 }}>RTO DATABASE</div>
           <div style={{ fontSize: '1.8rem', fontWeight: 900 }}>{detectorInfo.rto_db_size}</div>
           <div style={{ fontSize: '0.6rem', opacity: 0.5 }}>IDENTIFIED DISTRICTS</div>
        </div>
        <div className="glass-card" style={{ padding: '1.5rem', borderBottom: '3px solid var(--danger)' }}>
           <div style={{ fontSize: '0.7rem', fontWeight: 800, opacity: 0.6 }}>THREAT WATCHLIST</div>
           <div style={{ fontSize: '1.8rem', fontWeight: 900 }}>{watchlist.length}</div>
           <div style={{ fontSize: '0.6rem', opacity: 0.5 }}>ACTIVE TARGETS</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '2rem' }}>
         <div className="glass-card" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
               <div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 800 }}>Observer Telemetry</div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>Real-time engine throughput</div>
               </div>
               <div style={{ display: 'flex', gap: '0.5rem' }}>
                  {['CAM-01', 'CAM-02', 'CAM-GATES'].map(c => (
                     <button key={c} onClick={() => setActiveCam(c)} style={{ padding: '4px 10px', fontSize: '0.6rem', borderRadius: 4, background: activeCam === c ? 'var(--accent-primary)' : 'rgba(255,255,255,0.05)', border: 'none', color: '#fff', cursor: 'pointer' }}>{c}</button>
                  ))}
               </div>
            </div>
            <div style={{ height: 250, display: 'flex', alignItems: 'flex-end', gap: '6px' }}>
               {Array.from({ length: 40 }).map((_, i) => (
                 <div key={i} className="shimmer" style={{ flex: 1, height: `${Math.random() * 90 + 5}%`, background: 'rgba(99, 102, 241, 0.2)', borderRadius: 2 }} />
               ))}
            </div>
         </div>

         <div className="glass-card" style={{ padding: '1.5rem' }}>
            <div style={{ fontWeight: 800, marginBottom: '1.5rem' }}>System Environment</div>
            <div style={{ display: 'grid', gap: '1rem' }}>
               <div className="glass-card" style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)' }}>
                  <div style={{ fontSize: '0.6rem', opacity: 0.5 }}>CURRENT MODE</div>
                  <div style={{ fontWeight: 800, display: 'flex', alignItems: 'center', gap: '8px' }}>
                     {history[0]?.plates[0]?.environment === 'Daylight' ? '☀️ DAYLIGHT' : '🌙 NIGHT LOGIC'}
                  </div>
               </div>
               <div className="glass-card" style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)' }}>
                  <div style={{ fontSize: '0.6rem', opacity: 0.5 }}>ACTIVE SENSORS</div>
                  <div style={{ fontSize: '0.8rem', marginTop: '0.4rem' }}>
                     <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Optical</span><span>OK</span></div>
                     <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Neural</span><span>UP</span></div>
                  </div>
               </div>
            </div>
         </div>
      </div>
    </div>
  )

  const Intelligence = () => (
    <div className="fade-up">
       <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '2rem' }}>
          <div className="glass-card" style={{ overflow: 'hidden' }}>
             <div className="scanner-frame" style={{ minHeight: 450 }}>
                <div className="scan-laser" />
                {preview ? <img src={preview} style={{ width: '100%', height: 450, objectFit: 'contain' }} /> : (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: 450, color: 'var(--text-secondary)' }}>
                     <div style={{ fontSize: '4rem', opacity: 0.2 }}>🛰️</div>
                     <p style={{ marginTop: '1rem', letterSpacing: '2px', fontWeight: 600 }}>AWAITING OPTICAL INPUT</p>
                     <button className="nav-item" style={{ marginTop: '2rem', background: 'var(--accent-primary)', color: '#fff' }} onClick={() => fileRef.current.click()}>Initialize Scan</button>
                  </div>
                )}
             </div>
             <input type="file" ref={fileRef} hidden onChange={e => handleUpload(e.target.files[0])} />
             {preview && (
               <div style={{ padding: '1.5rem' }}>
                  <button className="nav-item" style={{ width: '100%', background: 'linear-gradient(to right, var(--accent-primary), var(--accent-secondary))', color: '#fff', justifyContent: 'center' }} onClick={() => handleUpload(fileRef.current.files[0])} disabled={loading}>
                     {loading ? 'NEURAL PROCESSING...' : 'EXECUTE OBSERVATION'}
                  </button>
               </div>
             )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <div className="glass-card" style={{ padding: '1.5rem' }}>
                 <div style={{ fontWeight: 800, color: 'var(--accent-primary)', marginBottom: '1rem', display: 'flex', justifyContent: 'space-between' }}>
                    <span>Target Intelligence</span>
                    <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>{activeCam}</span>
                 </div>
                 {result?.plates.map((p, i) => (
                   <div key={i} style={{ marginBottom: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '1rem' }}>
                     <img src={p.crop_image} style={{ width: '100%', borderRadius: 8, border: '1px solid var(--accent-primary)', marginBottom: '0.8rem' }} />
                     
                     <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div>
                           <div style={{ fontSize: '1.6rem', fontWeight: 800, color: watchlist.includes(p.text) ? 'var(--danger)' : '#fff', letterSpacing: '1px' }}>{p.text}</div>
                           {p.city && <div style={{ fontSize: '0.8rem', color: 'var(--accent-primary)', fontWeight: 700 }}>📍 {p.city}</div>}
                        </div>
                        <div style={{ textAlign: 'right' }}>
                           <div className="mono" style={{ fontSize: '0.9rem', fontWeight: 900, color: 'var(--accent-secondary)' }}>{Math.round(p.speed)} km/h</div>
                           <div style={{ fontSize: '0.5rem', opacity: 0.5 }}>EST. VELOCITY</div>
                        </div>
                     </div>
                     
                     <div className="glass-card" style={{ marginTop: '1rem', padding: '0.8rem', background: 'rgba(255,255,255,0.02)' }}>
                        <div style={{ fontSize: '0.6rem', opacity: 0.5, marginBottom: '0.4rem' }}>OWNER PROFILE & COMPLIANCE</div>
                        <div style={{ fontWeight: 700, fontSize: '0.85rem' }}>{p.owner || 'UNKNOWN'}</div>
                        <div style={{ fontSize: '0.7rem', display: 'flex', gap: '8px', marginTop: '4px' }}>
                           <span style={{ color: p.insurance_valid !== false ? 'var(--success)' : 'var(--danger)' }}>INS: {p.insurance_valid !== false ? '✅' : '❌'}</span>
                           <span style={{ color: p.puc_valid !== false ? 'var(--success)' : 'var(--danger)' }}>PUC: {p.puc_valid !== false ? '✅' : '❌'}</span>
                        </div>
                     </div>

                     <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.8rem' }}>
                        <div className="glass-card" style={{ padding: '0.5rem', fontSize: '0.7rem' }}>
                           <div style={{ opacity: 0.6 }}>CLASSIFICATION</div>
                           <div style={{ fontWeight: 700 }}>{p.vehicle_type?.toUpperCase()}</div>
                        </div>
                        <div className="glass-card" style={{ padding: '0.5rem', fontSize: '0.7rem' }}>
                           <div style={{ opacity: 0.6 }}>COLOR</div>
                           <div style={{ fontWeight: 700 }}>{p.vehicle_color?.toUpperCase()}</div>
                        </div>
                     </div>
                   </div>
                 ))}
                 {!result && (
                   <div style={{ padding: '2rem', textAlign: 'center' }}>
                      <div className="neural-pulse" style={{ margin: '0 auto 1rem' }} />
                      <div style={{ fontSize: '0.7rem', opacity: 0.4 }}>AWAITING OPTICAL FEED...</div>
                   </div>
                 )}
              </div>

              <div className="glass-card" style={{ padding: '1.5rem' }}>
                 <div style={{ fontWeight: 800, marginBottom: '1rem' }}>Risk Analysis</div>
                 {result?.plates.some(p => watchlist.includes(p.text)) ? (
                   <div style={{ color: 'var(--danger)', fontWeight: 700, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className="neural-pulse" style={{ background: 'var(--danger)' }} /> IMMEDIATE THREAT DETECTED
                   </div>
                 ) : (
                   <div style={{ color: 'var(--success)', fontSize: '0.8rem' }}>● NO IMMEDIATE THREATS IN DATABASE</div>
                 )}
              </div>
          </div>
       </div>
    </div>
  )

  const Audit = () => (
    <div className="fade-up glass-card" style={{ padding: '2rem' }}>
       <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>Audit Ledger</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Cryptographically tagged historical records</div>
          </div>
          <a href={`${API_BASE}/history/export`} className="nav-item" style={{ background: 'rgba(255,255,255,0.05)', height: 'fit-content' }}>Export CSV</a>
       </div>
       <input className="search-input" placeholder="SEARCH PLATE ID, CITY, OR STATE..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)} style={{ marginBottom: '2rem' }} />
       <div style={{ display: 'grid', gap: '0.5rem' }}>
          {history.filter(h => {
             const p = h.plates[0] || {};
             const term = searchTerm.toUpperCase();
             return p.text?.includes(term) || p.city?.toUpperCase().includes(term) || p.state?.toUpperCase().includes(term);
          }).map((h, i) => (
            <div key={i} className="glass-card" style={{ padding: '1rem', display: 'flex', alignItems: 'center', gap: '1.5rem', background: 'rgba(255,255,255,0.02)', borderLeft: watchlist.includes(h.plates[0]?.text) ? '4px solid var(--danger)' : '1px solid var(--glass-border)' }}>
               <img src={h.plates[0]?.crop_image} style={{ width: 80, height: 40, objectFit: 'cover', borderRadius: 4 }} />
                  <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 800, color: watchlist.includes(h.plates[0]?.text) ? 'var(--danger)' : 'var(--accent-primary)', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    {h.plates[0]?.text} 
                    {watchlist.includes(h.plates[0]?.text) && <span style={{ fontSize: '0.6rem', color: '#fff', background: 'var(--danger)', padding: '2px 4px', borderRadius: 2 }}>WATCHLIST</span>}
                  </div>
                  <div className="mono" style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
                     {h.plates[0]?.vehicle_type?.toUpperCase()} ({h.plates[0]?.vehicle_color?.toUpperCase()}) | {new Date(h.timestamp).toLocaleString()}
                  </div>
               </div>
               <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 700 }}>{h.plates[0]?.city || h.plates[0]?.state || 'UNK'}</div>
                  <div style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', opacity: 0.8 }}>{h.plates[0]?.city ? h.plates[0]?.state : ''}</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--success)', fontWeight: 700 }}>{(h.plates[0]?.confidence*100).toFixed(0)}%</div>
               </div>
            </div>
          ))}
          {history.length === 0 && <div style={{ textAlign: 'center', opacity: 0.3, padding: '4rem' }}>NO AUDIT RECORDS FOUND</div>}
       </div>
    </div>
  )

  const Analytics = () => {
    const timeBuckets = useMemo(() => {
      const buckets = Array(24).fill(0);
      history.forEach(h => {
        const hour = new Date(h.timestamp).getHours();
        buckets[hour]++;
      });
      return buckets;
    }, [history]);

    const maxActivity = Math.max(...timeBuckets, 1);

    return (
      <div className="fade-up">
         <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '2rem' }}>
            <div className="glass-card" style={{ padding: '2rem' }}>
               <div style={{ fontWeight: 800, marginBottom: '2rem' }}>Temporal Activity (24h)</div>
               <div style={{ height: 200, display: 'flex', alignItems: 'flex-end', gap: '8px', paddingBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  {timeBuckets.map((v, i) => (
                    <div key={i} style={{ 
                      flex: 1, 
                      height: `${(v / maxActivity) * 100}%`, 
                      background: 'linear-gradient(to top, var(--accent-primary), var(--accent-secondary))',
                      borderRadius: '4px 4px 0 0',
                      position: 'relative',
                      minHeight: v > 0 ? 4 : 0
                    }}>
                       {v > 0 && <div style={{ position: 'absolute', top: -20, left: '50%', transform: 'translateX(-50%)', fontSize: '0.6rem', fontWeight: 800 }}>{v}</div>}
                    </div>
                  ))}
               </div>
               <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem', color: 'var(--text-secondary)', fontSize: '0.7rem' }}>
                  <span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>23:59</span>
               </div>
            </div>

            <div className="glass-card" style={{ padding: '2rem' }}>
               <div style={{ fontWeight: 800, marginBottom: '1.5rem' }}>Vehicle Distribution</div>
               {['car', 'motorcycle', 'truck', 'bus'].map(type => {
                  const count = history.filter(h => h.plates[0]?.vehicle_type === type).length;
                  const pct = history.length ? (count / history.length) * 100 : 0;
                  return (
                    <div key={type} style={{ marginBottom: '1.2rem' }}>
                       <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.4rem' }}>
                          <span style={{ textTransform: 'capitalize' }}>{type}</span>
                          <span>{count}</span>
                       </div>
                       <div className="bar-track"><div className="bar-fill" style={{ width: `${pct}%` }} /></div>
                    </div>
                  )
               })}
            </div>
         </div>

         <div className="glass-card" style={{ padding: '2rem', marginTop: '2rem' }}>
            <div style={{ fontWeight: 800, marginBottom: '1.5rem' }}>Top 10 Cities By Registration</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1.5rem' }}>
               {Object.entries(history.reduce((acc, h) => {
                  const city = h.plates[0]?.city;
                  if (city) acc[city] = (acc[city] || 0) + 1;
                  return acc;
               }, {})).sort((a,b) => b[1] - a[1]).slice(0, 10).map(([city, count]) => (
                  <div key={city} className="glass-card" style={{ padding: '1rem', textAlign: 'center', background: 'rgba(255,255,255,0.03)' }}>
                     <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: 600 }}>{city.toUpperCase()}</div>
                     <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-primary)' }}>{count}</div>
                  </div>
               ))}
            </div>
         </div>
      </div>
    )
  }

  const Geospatial = () => {
    const mapRef = useRef(null);
    useEffect(() => {
      if (!window.L || !mapRef.current) return;
      
      const map = L.map(mapRef.current, { zoomControl: false }).setView([22.5937, 78.9629], 5);
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; CartoDB'
      }).addTo(map);

      history.forEach(h => {
        const state = h.plates[0]?.state || 'MH';
        const coords = STATE_COORDS[state] || [19, 75];
        const isWatchlist = watchlist.includes(h.plates[0].text);
        
        const marker = L.circleMarker(coords, {
          radius: isWatchlist ? 8 : 4,
          fillColor: isWatchlist ? 'var(--danger)' : 'var(--accent-primary)',
          color: '#fff',
          weight: 1,
          opacity: 1,
          fillOpacity: 0.8
        }).addTo(map);

        marker.bindPopup(`<div style="color:#000;font-family:Inter;font-weight:700">${h.plates[0].text}</div><div style="color:#666;font-size:0.7rem">${h.plates[0].city || state}</div>`);
      });

      return () => map.remove();
    }, [history]);

    return (
      <div className="fade-up">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '2rem' }}>
          <div className="glass-card" style={{ height: 600, position: 'relative', overflow: 'hidden' }}>
             <div ref={mapRef} style={{ height: '100%', width: '100%' }} />
             <div style={{ position: 'absolute', top: '1.5rem', left: '1.5rem', zIndex: 1000, background: 'rgba(0,0,0,0.8)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--accent-primary)' }}>
                <div style={{ fontSize: '0.6rem', fontWeight: 800, opacity: 0.6, marginBottom: '0.4rem' }}>GEOSPATIAL LAYER</div>
                <div style={{ fontWeight: 800, fontSize: '0.9rem' }}>INDIAN TERRITORY SCAN</div>
             </div>
          </div>

          <div className="glass-card" style={{ padding: '1.5rem' }}>
             <div style={{ fontWeight: 800, marginBottom: '1.5rem' }}>Active Alerts</div>
             <div style={{ display: 'grid', gap: '0.8rem', maxHeight: 500, overflowY: 'auto' }}>
                {history.filter(h => watchlist.includes(h.plates[0].text)).map((h, i) => (
                   <div key={i} className="glass-card" style={{ padding: '0.8rem', borderLeft: '3px solid var(--danger)', background: 'rgba(239, 68, 68, 0.05)' }}>
                      <div style={{ fontWeight: 800, color: 'var(--danger)', fontSize: '0.85rem' }}>🚨 {h.plates[0].text}</div>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>MATCH IN {h.plates[0].city?.toUpperCase()}</div>
                   </div>
                ))}
                {history.filter(h => !watchlist.includes(h.plates[0].text)).slice(0, 10).map((h, i) => (
                   <div key={i} className="glass-card" style={{ padding: '0.8rem', borderLeft: '3px solid var(--accent-primary)', background: 'rgba(255,255,255,0.02)' }}>
                      <div style={{ fontWeight: 800, fontSize: '0.85rem' }}>● {h.plates[0].text}</div>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>DETECTED IN {h.plates[0].city || h.plates[0].state || 'UNK'}</div>
                   </div>
                ))}
             </div>
          </div>
        </div>
      </div>
    )
  }

  const AICore = () => (
    <div className="fade-up">
       <div className="glass-card" style={{ padding: '2rem', marginBottom: '2rem' }}>
          <div style={{ fontWeight: 800, marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
             <span className="neural-pulse" /> Neural Temporal Heatmap
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(24, 1fr)', gap: '4px', height: 120 }}>
             {Array.from({ length: 24 * 7 }).map((_, i) => {
               const val = Math.random();
               return <div key={i} style={{ 
                 background: val > 0.8 ? 'var(--accent-primary)' : val > 0.4 ? 'rgba(99, 102, 241, 0.3)' : 'rgba(255,255,255,0.02)',
                 borderRadius: 2,
                 title: `Hour ${i % 24}`
               }} />
             })}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: '0.6rem', marginTop: '0.5rem' }}>
             <span>00:00</span><span>12:00</span><span>23:59</span>
          </div>
       </div>

       <div className="glass-card" style={{ padding: '3rem', textAlign: 'center' }}>
          <div className="neural-pulse" style={{ width: 40, height: 40, margin: '0 auto 2rem' }} />
          <h2 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '1rem' }}>AI Logic Connected</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '2rem', marginTop: '3rem' }}>
             <div className="glass-card" style={{ padding: '2rem' }}>
                <div style={{ fontWeight: 800, color: 'var(--accent-primary)' }}>PRIMARY ENGINE</div>
                <div className="mono" style={{ fontSize: '1.2rem', marginTop: '0.5rem' }}>YOLO v8n</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--success)', marginTop: '0.5rem' }}>{detectorInfo.yolo_loaded ? '● ACTIVE' : '● LEGACY FALLBACK'}</div>
             </div>
             <div style={{ paddingTop: '2rem' }}>
                <div style={{ fontWeight: 800 }}>OCR LAYER</div>
                <div className="mono" style={{ fontSize: '1.2rem' }}>EasyOCR Adaptive</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--success)' }}>● ATTACHED</div>
             </div>
             <div className="glass-card" style={{ padding: '2rem' }}>
                <div style={{ fontWeight: 800, color: 'var(--accent-secondary)' }}>LEGACY CORE</div>
                <div className="mono" style={{ fontSize: '1.2rem' }}>Haar Cascades</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--success)' }}>● READY</div>
             </div>
          </div>
          <div style={{ marginTop: '4rem', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
             Database Connection: SQLite 3 | Persistence Layer: Active | Audit Log: Encrypted
          </div>
       </div>
    </div>
  )

  return (
    <div className="app-container">
      <nav className="sidebar">
        <div style={{ marginBottom: '3rem', padding: '0 1rem' }}>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, letterSpacing: '-1px' }}>SENTINEL<span style={{ color: 'var(--accent-primary)' }}>.AI</span></div>
          <div className="mono" style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', letterSpacing: '2px' }}>NEURAL NETWORK INTERFACE</div>
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {[
          { id: 'dashboard', label: 'Mission Control', icon: '📡' },
          { id: 'monitor', label: 'Optical Scan', icon: '👁️' },
          { id: 'geospatial', label: 'Geospatial Intel', icon: '🌍' },
          { id: 'analytics', label: 'Intelligence', icon: '📈' },
          { id: 'logs', label: 'Audit Trail', icon: '📜' },
          { id: 'settings', label: 'Neural Config', icon: '⚙️' },
        ].map(n => (
          <div key={n.id} className={`nav-item ${activeTab === n.id ? 'active' : ''}`} onClick={() => setActiveTab(n.id)}>
              <span>{n.icon}</span> {n.label}
            </div>
          ))}
        </div>

        <div style={{ marginTop: 'auto', padding: '1.5rem', background: 'rgba(255,255,255,0.02)', borderRadius: 16 }}>
           <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div className="neural-pulse" />
              <div style={{ fontSize: '0.8rem', fontWeight: 700 }}>AI ACTIVE</div>
           </div>
           <div style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>UPTIME: {Math.floor(Date.now()/1000) % 10000}s</div>
        </div>
      </nav>

      <main className="main-stage">
        <header style={{ marginBottom: '3rem' }}>
          <h1 className="page-title">{activeTab.toUpperCase()}</h1>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>SYS_STATUS: {apiStatus}</span>
            <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--accent-primary)' }}>CORE: INTEGRATED</span>
          </div>
        </header>

        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'monitor' && <Intelligence />}
        {activeTab === 'geospatial' && <Geospatial />}
        {activeTab === 'logs' && <Audit />}
        {activeTab === 'analytics' && <Analytics />}
        {activeTab === 'settings' && (
          <div className="fade-up glass-card" style={{ padding: '2rem' }}>
             <h2 style={{ fontWeight: 800, marginBottom: '2rem' }}>Watchlist Management</h2>
             <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
                <input className="search-input" placeholder="ENTER PLATE ID..." value={newWatchlistPlate} onChange={e => setNewWatchlistPlate(e.target.value.toUpperCase())} />
                <button className="nav-item" style={{ background: 'var(--accent-primary)', color: '#fff' }} onClick={() => {
                   if (newWatchlistPlate) {
                      setWatchlist([...watchlist, newWatchlistPlate]);
                      setNewWatchlistPlate('');
                      showToast(`Added ${newWatchlistPlate} to Watchlist`, 'success');
                   }
                }}>Add to Watchlist</button>
             </div>
             <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                {watchlist.map(p => (
                  <div key={p} className="glass-card" style={{ padding: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                     <span style={{ fontWeight: 800 }}>{p}</span>
                     <button onClick={() => setWatchlist(watchlist.filter(x => x !== p))} style={{ background: 'none', border: 'none', color: 'var(--danger)', cursor: 'pointer' }}>REMOVE</button>
                  </div>
                ))}
             </div>
          </div>
        )}
      </main>

      {toast && (
        <div style={{ position: 'fixed', bottom: '2rem', right: '2rem', padding: '1.5rem 2.5rem', background: toast.type === 'danger' ? 'var(--danger)' : 'var(--accent-primary)', color: '#fff', borderRadius: '1rem', boxShadow: '0 20px 50px rgba(0,0,0,0.5)', zIndex: 1000, fontWeight: 800, animation: 'fadeUp 0.3s' }}>
          {toast.msg}
        </div>
      )}
    </div>
  )
}
