/* ============================================================
   Scanner 3D — Common JavaScript
   LEDs, logging, SSE, artifact polling, UI updates, scan trigger
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

// ---- Artifact polling (fallback when SSE is unavailable) ----
let _poll = null;
let _artifactPoll = null;
let _selectedArtifact = (
  window.SCANNER_ARTIFACT_KINDS && window.SCANNER_ARTIFACT_KINDS.length
)
  ? window.SCANNER_ARTIFACT_KINDS[0]
  : 'extract_left';
let _artifacts = {};
const RING_CIRCUMFERENCE = 553; // 2*PI*r, r=88

const PROCESSING_STEP_ORDER = [
  'extract', 'fit', 'triangulate', 'fuse', 'merge', 'regression', 'outliers', 'caps', 'mesh'
];
let _processingStep = null;

function setProcessingStep(step) {
  if (!step) return;
  _processingStep = step;
  var idx = PROCESSING_STEP_ORDER.indexOf(step);
  document.querySelectorAll('#proc-steps .proc-step').forEach(function(el) {
    var s = el.getAttribute('data-step');
    var sidx = PROCESSING_STEP_ORDER.indexOf(s);
    el.classList.remove('active', 'done');
    if (sidx >= 0 && idx >= 0 && sidx < idx) el.classList.add('done');
    if (s === step) el.classList.add('active');
  });
}

function completeProcessingSteps() {
  _processingStep = 'mesh';
  document.querySelectorAll('#proc-steps .proc-step').forEach(function(el) {
    el.classList.remove('active');
    el.classList.add('done');
  });
}

function resetProcessingSteps() {
  _processingStep = null;
  document.querySelectorAll('#proc-steps .proc-step').forEach(function(el) {
    el.classList.remove('active', 'done');
  });
}

function inferProcessingStep(message) {
  var m = (message || '').toLowerCase();
  if (m.indexOf('extracting') >= 0 || m.indexOf('extract ') >= 0) return 'extract';
  if (m.indexOf('fitting shared camera frame') >= 0 || m.indexOf('fit') >= 0) return 'fit';
  if (m.indexOf('triangulating') >= 0 || m.indexOf('triangulation') >= 0) return 'triangulate';
  if (m.indexOf('fusing per-step profiles') >= 0 || m.indexOf('fuse') >= 0) return 'fuse';
  if (m.indexOf('merging') >= 0 || m.indexOf('merge') >= 0) return 'merge';
  if (m.indexOf('regression') >= 0 || m.indexOf('local polynomial') >= 0) return 'regression';
  if (m.indexOf('outlier') >= 0 || m.indexOf('filtering') >= 0) return 'outliers';
  if (m.indexOf('caps') >= 0) return 'caps';
  if (m.indexOf('mesh') >= 0 || m.indexOf('poisson') >= 0 || m.indexOf('exporting') >= 0) return 'mesh';
  return null;
}

function isImageArtifact(kind) {
  return kind && kind.indexOf('extract_') === 0;
}

function artifactUrl(kind) {
  return '/scan/artifact/' + encodeURIComponent(kind) + '?t=' + Date.now();
}

function setArtifactTabs() {
  document.querySelectorAll('.artifact-tab').forEach(function(btn) {
    var kind = btn.getAttribute('data-kind');
    var artifact = _artifacts[kind];
    btn.classList.toggle('active', kind === _selectedArtifact);
    btn.classList.toggle('available', !!(artifact && artifact.available));
    btn.disabled = !!(artifact && artifact.path === null && !artifact.available);
  });
}

// ---- Stage display (3 overlapping layers: image / 3D canvas / placeholder) ----
function showStageLayer(which) {
  var img = document.getElementById('live-frame');
  var canvas = document.getElementById('stl-canvas');
  var placeholder = document.getElementById('viewer-placeholder');
  if (img) img.style.display = (which === 'image') ? 'block' : 'none';
  if (canvas) canvas.style.display = (which === 'canvas') ? 'block' : 'none';
  if (placeholder) placeholder.style.display = (which === 'placeholder') ? 'flex' : 'none';
}

function showImageArtifact(kind) {
  var img = document.getElementById('live-frame');
  var stage = document.getElementById('artifact-stage');
  var label = document.getElementById('frame-label');
  var artifact = _artifacts[kind] || {};
  if (artifact.available) {
    showStageLayer('image');
    if (img) img.src = artifactUrl(kind);
    if (stage) stage.textContent = 'EXTRACTION';
  } else {
    showStageLayer('placeholder');
    if (stage) stage.textContent = 'EN ATTENTE';
  }
  if (label) label.textContent = artifact.label || kind;
}

function loadSelectedArtifact() {
  var artifact = _artifacts[_selectedArtifact];
  setArtifactTabs();
  if (isImageArtifact(_selectedArtifact)) {
    showImageArtifact(_selectedArtifact);
    return;
  }
  var stage = document.getElementById('artifact-stage');
  var label = document.getElementById('frame-label');
  if (label) label.textContent = artifact ? artifact.label : _selectedArtifact;
  if (stage) stage.textContent = artifact && artifact.available ? 'MODÈLE' : 'EN ATTENTE';
  if (!artifact || !artifact.available) {
    showStageLayer('placeholder');
    var vs = document.getElementById('viewer-status');
    if (vs) vs.textContent = '·';
    return;
  }
  if (typeof window._loadArtifact === 'function') {
    window._loadArtifact(_selectedArtifact, artifact.media_type || 'model/stl');
  }
}

function applyArtifacts(artifacts) {
  _artifacts = artifacts || {};
  setArtifactTabs();
  loadSelectedArtifact();
}

async function refreshArtifacts() {
  try {
    var resp = await fetch('/scan/artifacts?t=' + Date.now());
    if (!resp.ok) return;
    applyArtifacts(await resp.json());
  } catch (_) {}
}

function selectArtifact(kind) {
  _selectedArtifact = kind;
  loadSelectedArtifact();
}

function startPolling() {
  if (_poll) return;
  _poll = setInterval(refreshArtifacts, 800);
}
function stopPolling() { clearInterval(_poll); _poll = null; }

function startArtifactPolling() {
  if (_artifactPoll) return;
  _artifactPoll = setInterval(refreshArtifacts, 1500);
}
function stopArtifactPolling() { clearInterval(_artifactPoll); _artifactPoll = null; }

// ---- Progress ring + scan button ----
function updateRing(state, pct) {
  var fill = document.querySelector('.scan-ring .ring-fill');
  if (!fill) return;
  var offset = RING_CIRCUMFERENCE - (RING_CIRCUMFERENCE * (pct || 0) / 100);
  fill.style.strokeDashoffset = offset;
  fill.className = 'ring-fill';
  if (['SCANNING', 'PROCESSING', 'EXPORTING'].includes(state)) fill.classList.add('scanning');
  if (state === 'COMPLETE') fill.classList.add('complete');
  if (state === 'ERROR') fill.classList.add('error');
}

function setScanButton(state, pct, blockedByDoor) {
  var btn = document.getElementById('btn-scan');
  var txt = document.getElementById('btn-scan-text');
  if (!btn) return;
  var icon = btn.querySelector('i');
  var busy = ['SCANNING', 'PROCESSING', 'EXPORTING'].includes(state);
  btn.className = 'scan-btn' + (busy ? ' scanning' : (state === 'ERROR' ? ' error' : ''));
  btn.disabled = busy || !!blockedByDoor;

  if (busy) {
    if (icon) icon.className = 'bi bi-arrow-repeat spin';
    if (txt) txt.textContent = (pct || 0) + '%';
  } else if (blockedByDoor) {
    if (icon) icon.className = 'bi bi-door-open';
    if (txt) txt.textContent = 'PORTE';
  } else if (state === 'ERROR') {
    if (icon) icon.className = 'bi bi-arrow-clockwise';
    if (txt) txt.textContent = 'RÉESSAYER';
  } else if (state === 'COMPLETE') {
    if (icon) icon.className = 'bi bi-arrow-clockwise';
    if (txt) txt.textContent = 'NOUVEAU';
  } else {
    if (icon) icon.className = 'bi bi-play-fill';
    if (txt) txt.textContent = 'SCAN';
  }
}

// ---- Main UI update ----
function updateUI(d) {
  const sl  = document.getElementById('state-label');
  const sm  = document.getElementById('state-message');
  const pb  = document.getElementById('progress-bar');
  const pct = document.getElementById('progress-pct');
  const dl  = document.getElementById('btn-download');
  const ds  = document.getElementById('door-state');
  const dw  = document.getElementById('door-open-warn');

  var state = d.state || 'IDLE';
  var progress = (d.progress !== undefined) ? d.progress : 0;
  var blockedByDoor = !!d.door_interlock_enabled && !!d.door_open;

  if (sl && d.state) {
    sl.textContent = d.state;
    sl.className   = 'state-name ' + d.state.toLowerCase();
  }
  if (sm && d.message) sm.textContent = d.message;
  if (pb && d.progress !== undefined) {
    pb.style.width = progress + '%';
    if (pct) pct.textContent = progress + '%';
  }
  applyLeds(state);
  updateRing(state, progress);
  setScanButton(state, progress, blockedByDoor);

  if (state === 'SCANNING') {
    resetProcessingSteps();
    if (pb) pb.classList.add('active');
    startPolling();
  }
  if (['PROCESSING', 'EXPORTING'].includes(state)) {
    startArtifactPolling();
  }
  if (['COMPLETE', 'ERROR', 'IDLE'].includes(state)) {
    if (pb) pb.classList.remove('active');
    stopPolling();
    stopArtifactPolling();
    if (state === 'COMPLETE') {
      completeProcessingSteps();
      refreshArtifacts();
      if (dl) dl.classList.remove('off');
      var usbBtn = document.getElementById('btn-usb');
      if (usbBtn) usbBtn.classList.remove('off');
      log('Scan terminé.', 'log-ok');
    }
    if (state === 'ERROR') log(d.message || 'Erreur', 'log-err');
  }
  if (d.message) log(d.message);

  if (state === 'PROCESSING' || state === 'EXPORTING') {
    var inferred = inferProcessingStep(d.message || '');
    if (inferred) setProcessingStep(inferred);
  }
  if (state === 'ERROR' || state === 'IDLE') {
    if (!_processingStep || state === 'IDLE') resetProcessingSteps();
  }

  if (ds) {
    if (!d.door_interlock_enabled) ds.textContent = 'Porte: interlock désactivé';
    else ds.textContent = 'Porte: ' + (d.door_open ? 'OUVERTE' : 'fermée');
  }
  if (dw) dw.style.display = blockedByDoor ? 'block' : 'none';
}

// ---- SSE ----
function connectSSE() {
  var es = new EventSource('/scan/stream');
  es.onmessage = function(e) {
    try {
      var d = JSON.parse(e.data);
      if (d.artifacts) applyArtifacts(d.artifacts);
      updateUI(d);
      if (d.state === 'COMPLETE') selectArtifact('mesh');
    } catch(_) {}
  };
  es.onerror = function() { es.close(); setTimeout(connectSSE, 3000); };
}

// ---- Start scan ----
async function startScan() {
  var btn = document.getElementById('btn-scan');
  if (btn) btn.disabled = true;
  try {
    var r = await fetch('/scan/start', { method:'POST' });
    var d = await r.json();
    if (!r.ok) {
      showToast(d.error || r.statusText, 'error');
      if (btn) btn.disabled = false;
    } else {
      startPolling();
      log('Démarrage de l\'acquisition…');
    }
  } catch(e) {
    showToast('Erreur réseau : ' + e, 'error');
    if (btn) btn.disabled = false;
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

// ---- Load 3D model (stub, overridden by the lazy Three.js loader) ----
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

window.selectArtifact = selectArtifact;
window.refreshArtifacts = refreshArtifacts;
