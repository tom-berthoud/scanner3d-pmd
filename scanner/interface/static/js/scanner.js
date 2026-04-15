/* ============================================================
   Scanner 3D — Common JavaScript
   LEDs, logging, SSE, frame polling, UI updates, scan trigger
   ============================================================ */

// ---- LED control ----
const LED_MAP = {
  IDLE:        { orange:'off',   red:'off' },
  CALIBRATING: { orange:'blink', red:'off' },
  SCANNING:    { orange:'on',    red:'off' },
  PROCESSING:  { orange:'blink-fast', red:'off' },
  EXPORTING:   { orange:'blink', red:'off' },
  COMPLETE:    { orange:'off',   red:'off' },
  ERROR:       { orange:'off',   red:'on'  },
};
const LED_ID = { orange:'led-orange', red:'led-red' };

function applyLeds(state) {
  const p = LED_MAP[state] || LED_MAP.IDLE;
  for (const [color, mode] of Object.entries(p)) {
    const el = document.getElementById(LED_ID[color]);
    if (!el) continue;
    el.className = 'led';
    const cls = color === 'orange' ? 'o' : 'r';
    if (mode === 'on')         el.classList.add(cls);
    if (mode === 'blink')      el.classList.add(cls, 'blink');
    if (mode === 'blink-fast') el.classList.add(cls, 'blink-fast');
  }
}

// ---- Logging ----
function log(msg, cls) {
  cls = cls || '';
  const box = document.getElementById('log-box');
  if (!box) return;
  const ts  = new Date().toLocaleTimeString('fr', { hour12: false });
  const row = document.createElement('div');
  row.className = cls;
  row.innerHTML = '<span class="log-ts">' + ts + '</span><span class="log-msg">' + msg + '</span>';
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

// ---- Frame polling ----
let _poll = null;
function startPolling() {
  const fc = document.getElementById('frame-card');
  if (fc) fc.classList.add('show');
  if (_poll) return;
  _poll = setInterval(function() {
    const img = document.getElementById('live-frame');
    if (img) img.src = '/scan/frame/latest?t=' + Date.now();
  }, 600);
}
function stopPolling() { clearInterval(_poll); _poll = null; }

// ---- Kiosk UI update ----
function updateKiosk(d) {
  const kb = document.getElementById('kiosk-btn');
  const kr = document.querySelector('.kiosk-ring .ring-fill');
  const ks = document.querySelector('.kiosk-status .state');
  const km = document.querySelector('.kiosk-status .msg');
  const ns = document.getElementById('kiosk-new-scan');

  if (!kb) return;

  const pct = d.progress || 0;
  const state = d.state || 'IDLE';

  // Update ring progress (circumference = 2*PI*88 ≈ 553)
  if (kr) {
    const offset = 553 - (553 * pct / 100);
    kr.style.strokeDashoffset = offset;
    kr.className = 'ring-fill';
    if (state === 'SCANNING' || state === 'PROCESSING' || state === 'EXPORTING') kr.classList.add('scanning');
    if (state === 'COMPLETE') kr.classList.add('complete');
  }

  // Update button state
  kb.className = 'kiosk-btn';
  const busy = ['SCANNING', 'PROCESSING', 'EXPORTING'].includes(state);
  kb.disabled = busy;

  if (state === 'COMPLETE') {
    kb.classList.add('usb-mode');
    kb.innerHTML = '<i class="bi bi-usb-drive"></i>COPIER<br>USB';
    kb.disabled = false;
    kb.onclick = function() { if (typeof window.copyToUsb === 'function') window.copyToUsb(); };
    if (ns) ns.style.display = '';
  } else if (busy) {
    kb.classList.add('scanning');
    kb.innerHTML = '<i class="bi bi-arrow-repeat"></i>' + pct + '%';
  } else if (state === 'ERROR') {
    kb.classList.add('error');
    kb.innerHTML = '<i class="bi bi-exclamation-triangle"></i>ERREUR';
    kb.disabled = false;
    kb.onclick = startScan;
    if (ns) ns.style.display = '';
  } else {
    kb.innerHTML = '<i class="bi bi-play-fill"></i>SCAN';
    kb.onclick = startScan;
    if (ns) ns.style.display = 'none';
  }

  if (ks) ks.textContent = state;
  if (km) km.textContent = d.message || '';
}

// ---- Main UI update ----
function updateUI(d) {
  const sl  = document.getElementById('state-label');
  const sm  = document.getElementById('state-message');
  const pb  = document.getElementById('progress-bar');
  const pct = document.getElementById('progress-pct');
  const btn = document.getElementById('btn-scan');
  const txt = document.getElementById('btn-scan-text');
  const dl  = document.getElementById('btn-download');
  const fl  = document.getElementById('frame-label');
  const vs  = document.getElementById('viewer-status');

  if (sl && d.state) {
    sl.textContent = d.state;
    sl.className   = 'state-name ' + d.state.toLowerCase();
  }
  if (sm && d.message) sm.textContent = d.message;
  if (pb && d.progress !== undefined) {
    pb.style.width    = d.progress + '%';
    if (pct) pct.textContent   = d.progress + '%';
    if (fl) fl.textContent = d.progress + '%';
  }
  applyLeds(d.state || 'IDLE');

  if (btn) {
    var busy = ['SCANNING','PROCESSING','EXPORTING'].includes(d.state);
    btn.disabled   = busy;
    btn.className  = 'btn-scan' + (busy ? ' active' : '');
    if (txt) txt.textContent = busy ? '\u2B1B  ACQUISITION...' : '\u25B6  LANCER LE SCAN';
  }

  if (d.state === 'SCANNING') {
    if (pb) pb.classList.add('active');
    startPolling();
  }
  if (['COMPLETE','ERROR','IDLE'].includes(d.state)) {
    if (pb) pb.classList.remove('active');
    stopPolling();
    if (d.state === 'COMPLETE') {
      var img = document.getElementById('live-frame');
      if (img) img.src = '/scan/frame/latest?t=' + Date.now();
      if (dl) dl.classList.remove('off');
      var usbBtn = document.getElementById('btn-usb');
      if (usbBtn) usbBtn.classList.remove('off');
      if (vs) vs.textContent = 'CHARGEMENT...';
      log('Scan termin\u00e9.', 'log-ok');
    }
    if (d.state === 'ERROR') log(d.message || 'Erreur', 'log-err');
  }
  if (d.message) log(d.message);

  // Update kiosk view
  updateKiosk(d);
}

// ---- SSE ----
function connectSSE() {
  var es = new EventSource('/scan/stream');
  es.onmessage = function(e) {
    try {
      var d = JSON.parse(e.data);
      updateUI(d);
      if (d.state === 'COMPLETE' && typeof loadModel === 'function') loadModel();
    } catch(_) {}
  };
  es.onerror = function() { es.close(); setTimeout(connectSSE, 3000); };
}

// ---- Start scan ----
async function startScan() {
  var btn = document.getElementById('btn-scan');
  var kb = document.getElementById('kiosk-btn');
  if (btn) btn.disabled = true;
  if (kb) kb.disabled = true;
  try {
    var r = await fetch('/scan/start', { method:'POST' });
    var d = await r.json();
    if (!r.ok) {
      showToast(d.error || r.statusText, 'error');
      if (btn) btn.disabled = false;
      if (kb) kb.disabled = false;
    } else {
      startPolling();
      log('D\u00e9marrage de l\'acquisition...');
    }
  } catch(e) {
    showToast('Erreur r\u00e9seau : ' + e, 'error');
    if (btn) btn.disabled = false;
    if (kb) kb.disabled = false;
  }
}

// ---- Toast notifications ----
function showToast(msg, type) {
  type = type || 'info';
  var container = document.getElementById('toast-container');
  if (!container) return;
  var el = document.createElement('div');
  el.className = 'toast-msg ' + type;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(function() { el.remove(); }, 4000);
}

// ---- Load 3D model (stub, overridden by Three.js module) ----
function loadModel() {
  if (typeof window._loadModel === 'function') {
    window._loadModel();
    return;
  }
  var t = setInterval(function() {
    if (typeof window._loadModel === 'function') {
      clearInterval(t);
      window._loadModel();
    }
  }, 50);
}

// ---- Gear panel ----
function toggleGearPanel() {
  var panel = document.getElementById('gear-panel');
  var overlay = document.getElementById('gear-overlay');
  if (!panel) return;
  panel.classList.toggle('open');
  if (overlay) overlay.classList.toggle('open');
}

// ---- Mode detection ----
function detectMode() {
  var params = new URLSearchParams(window.location.search);
  if (params.get('mode') === 'kiosk') {
    document.body.setAttribute('data-mode', 'kiosk');
  }
}
