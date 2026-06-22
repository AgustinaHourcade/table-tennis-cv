// Detectamos si estamos en producción (GitHub Pages) para apuntar al backend de Hugging Face
let API_BASE_URL = '';
if (window.location.hostname.includes('github.io')) {
  // URL de tu servidor en Hugging Face
  API_BASE_URL = 'https://agustinah-table-tennis-cv.hf.space';
} else if (window.location.port !== '8555' && window.location.hostname === 'localhost' || window.location.protocol === 'file:') {
  // Si estás usando Live Server (puerto 8000) o abriendo el archivo localmente
  API_BASE_URL = 'http://localhost:8555';
}
/* ═══════════════════════════════════════════════════════════════
   TT Vision Dashboard — Main Application Script
   ═══════════════════════════════════════════════════════════════ */

// ── DOM Refs ──
const video       = document.getElementById('video-src');
const canvas      = document.getElementById('overlay-canvas');
const ctx         = canvas.getContext('2d');
const playerWrap  = document.getElementById('player-wrapper');
const idleState   = document.getElementById('player-idle-state');
const loadingOv   = document.getElementById('loading-overlay');

// Controls
const btnPlay       = document.getElementById('btn-play');
const playIcon      = document.getElementById('play-icon');
const btnPrev       = document.getElementById('btn-prev-frame');
const btnNext       = document.getElementById('btn-next-frame');
const ctrlTime      = document.getElementById('ctrl-time');
const ctrlProgress  = document.getElementById('ctrl-progress');
const progressFill  = document.getElementById('ctrl-progress-fill');
const btnMute       = document.getElementById('btn-mute');
const ctrlVolume    = document.getElementById('ctrl-volume');
const exportBtn     = document.getElementById('export-btn');
const exportVidBtn  = document.getElementById('export-vid-btn');

// Metrics
const confSlider    = document.getElementById('confidence-slider');
const confDisplay   = document.getElementById('confidence-display');
const metricsEmptyBanner = document.getElementById('metrics-empty-banner');

// Dropzone
const dropzone      = document.getElementById('dropzone');
const fileInput     = document.getElementById('file-input');
const dropzoneError = document.getElementById('dropzone-error');

// Idle upload link
const idleUploadLink = document.getElementById('idle-upload-link');

// ── State ──
let videoData = null;
let forceRedraw = true;
let isPlaying = false;
let activeDemo = null;
let hasVideoLoaded = false;
let maxPlaybackTime = Infinity; // Limit playback to JSON data duration
let lastMediaTime = 0; // Precise media time from requestVideoFrameCallback

const layerState = {
  detection: true,
  pose: true,
  segmentation: false   // #6: Segmentación OFF por defecto
};

let confidenceThreshold = 0.45;

// ── COCO Skeleton Connections ──
const POSE_CONNECTIONS = [
  [0,1],[0,2],[1,3],[2,4],
  [5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16]
];

// ── Colors ──
const COLOR_DET  = '#1D9E75';
const COLOR_POSE = '#378ADD';
const COLOR_SEG  = '#D85A30';
const COLOR_SEG_FILL = 'rgba(216, 90, 48, 0.25)';

// Custom detection classes
const CUSTOM_CLASSES = new Set(['TT Table', 'TT Net', 'TT Racket']);

// ── Demo Config ── (#3: 4 demos)
const DEMOS = [
  { key: 'demo_01', name: 'Demo 01', file: 'demo_01.mp4' },
  { key: 'demo_02', name: 'Demo 02', file: 'demo_02.mp4' },
  { key: 'demo_03', name: 'Demo 03', file: 'demo_03.mp4' },
  { key: 'demo_04', name: 'Demo 04', file: 'demo_04.mp4' }
];

/* ═══════════════════════════════════
   Enable / Disable Controls
   ═══════════════════════════════════ */
function setControlsEnabled(enabled) {
  btnPlay.disabled = !enabled;
  btnPrev.disabled = !enabled;
  btnNext.disabled = !enabled;
  btnMute.disabled = !enabled;
  ctrlVolume.disabled = !enabled;

  if (enabled) {
    ctrlTime.classList.remove('disabled');
    ctrlProgress.classList.remove('disabled');
    exportBtn.disabled = false;
    exportVidBtn.disabled = false;
  } else {
    ctrlTime.classList.add('disabled');
    ctrlTime.textContent = '—:— / —:—';
    ctrlProgress.classList.add('disabled');
    progressFill.style.width = '0%';
    exportBtn.disabled = true;
    exportVidBtn.disabled = true;
  }
}

/* ═══════════════════════════════════
   Canvas Sizing
   ═══════════════════════════════════ */
function resizeCanvas() {
  if (videoData) {
    canvas.width = videoData.video_info.width;
    canvas.height = videoData.video_info.height;
  } else if (video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  } else {
    canvas.width = 1920;
    canvas.height = 1080;
  }
  // Log para debugging de desfasaje de dimensiones
  if (video.videoWidth && videoData) {
    const vw = video.videoWidth, vh = video.videoHeight;
    const jw = videoData.video_info.width, jh = videoData.video_info.height;
    if (vw !== jw || vh !== jh) {
      console.warn('[TT-CV] Dimension mismatch: video=' + vw + 'x' + vh + ', JSON=' + jw + 'x' + jh +
        '. Video will be stretched to match JSON (OpenCV raw matrix) dimensions to ensure overlay alignment.');
    }
  }
  forceRedraw = true;
}

window.addEventListener('resize', resizeCanvas);

/* ═══════════════════════════════════
   Render Loop
   ═══════════════════════════════════ */

// Use requestVideoFrameCallback for frame-accurate sync
function startVideoFrameSync() {
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    function onFrame(now, metadata) {
      lastMediaTime = metadata.mediaTime;
      forceRedraw = true;
      video.requestVideoFrameCallback(onFrame);
    }
    video.requestVideoFrameCallback(onFrame);
  }
}

function renderLoop() {
  if (!video.src || (video.paused && !forceRedraw)) {
    requestAnimationFrame(renderLoop);
    return;
  }

  // Use precise media time if available, fallback to currentTime
  const mediaTime = lastMediaTime > 0 ? lastMediaTime : video.currentTime;

  // Enforce playback limit: pause when reaching max time
  if (video.currentTime >= maxPlaybackTime) {
    video.pause();
    video.currentTime = maxPlaybackTime - 0.01;
    isPlaying = false;
    updatePlayIcon();
    forceRedraw = true;
  }

  // Calculate current frame index — VFR-safe strategy:
  // 1. If frames have timestamp_ms (from updated backend), use binary search
  //    on timestamps to find the closest frame. This correctly handles Variable
  //    Frame Rate videos and survives seeking.
  // 2. Fallback to mediaTime * fps for legacy JSON without timestamps.
  let currentFrame = 0;
  if (videoData && videoData.frames && videoData.frames.length > 0) {
    const mediaTimeMs = mediaTime * 1000;
    if (videoData.frames[0] && typeof videoData.frames[0].timestamp_ms === 'number') {
      // VFR-safe: binary search for the frame with closest timestamp
      currentFrame = findFrameByTimestamp(videoData.frames, mediaTimeMs);
    } else {
      // Fallback: time-based calculation (works for CFR, may drift for VFR)
      const fps = videoData.video_info.fps;
      currentFrame = Math.floor(mediaTime * fps);
    }
    currentFrame = Math.max(0, Math.min(currentFrame, videoData.frames.length - 1));
  }

  // Clear and draw video frame
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (videoData && videoData.frames) {
    const frameData = videoData.frames[currentFrame];

    if (frameData) {
      // Use the exact dimensions from the JSON to ensure 1:1 mapping with OpenCV's matrix.
      // Do NOT use video.videoWidth because the browser might apply Display Aspect Ratio (DAR)
      // scaling, which would cause double-scaling since OpenCV (and thus YOLO) processes
      // the raw pixel matrix ignoring DAR.
      const refWidth = videoData.video_info.width;
      const refHeight = videoData.video_info.height;
      const scaleX = canvas.width / refWidth;
      const scaleY = canvas.height / refHeight;

      // Draw order: segmentation (back), detection (mid), pose (front)
      if (layerState.segmentation && frameData.segmentations) {
        drawSegmentations(frameData.segmentations, scaleX, scaleY);
      }
      if (layerState.detection && frameData.detections) {
        drawDetections(frameData.detections, scaleX, scaleY);
      }
      if (layerState.pose && frameData.poses) {
        drawPoses(frameData.poses, scaleX, scaleY);
      }
    }
  }

  // Update time display and progress
  updateTimeDisplay();

  forceRedraw = false;
  requestAnimationFrame(renderLoop);
}

/* ═══════════════════════════════════
   Drawing Functions
   ═══════════════════════════════════ */
function drawDetections(detections, scaleX, scaleY) {
  detections
    .filter(d => d.confidence >= confidenceThreshold)
    .forEach(d => {
      const [x1, y1, x2, y2] = d.bbox;
      const sx1 = x1 * scaleX, sy1 = y1 * scaleY;
      const sw  = (x2 - x1) * scaleX, sh = (y2 - y1) * scaleY;

      // Bounding box
      ctx.strokeStyle = COLOR_DET;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
      ctx.strokeRect(sx1, sy1, sw, sh);

      // Corner accents
      const cornerLen = Math.min(sw, sh) * 0.15;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      // Top-left
      ctx.moveTo(sx1, sy1 + cornerLen); ctx.lineTo(sx1, sy1); ctx.lineTo(sx1 + cornerLen, sy1);
      // Top-right
      ctx.moveTo(sx1 + sw - cornerLen, sy1); ctx.lineTo(sx1 + sw, sy1); ctx.lineTo(sx1 + sw, sy1 + cornerLen);
      // Bottom-right
      ctx.moveTo(sx1 + sw, sy1 + sh - cornerLen); ctx.lineTo(sx1 + sw, sy1 + sh); ctx.lineTo(sx1 + sw - cornerLen, sy1 + sh);
      // Bottom-left
      ctx.moveTo(sx1 + cornerLen, sy1 + sh); ctx.lineTo(sx1, sy1 + sh); ctx.lineTo(sx1, sy1 + sh - cornerLen);
      ctx.stroke();
      ctx.lineWidth = 1.5;

      // Label pill
      const label = d.class_name + ' ' + d.confidence.toFixed(2);
      ctx.font = '11px "DM Mono", monospace';
      const tw = ctx.measureText(label).width;
      const pillH = 18;
      const pillW = tw + 10;

      ctx.fillStyle = COLOR_DET;
      roundRect(ctx, sx1, sy1 - pillH, pillW, pillH, 3);
      ctx.fill();

      ctx.fillStyle = '#000';
      ctx.fillText(label, sx1 + 5, sy1 - 5);
    });
}

function drawPoses(poses, scaleX, scaleY) {
  poses.forEach(person => {
    const kps = person.keypoints;
    if (!kps || kps.length === 0) return;

    // Skeleton lines
    ctx.strokeStyle = COLOR_POSE;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    ctx.globalAlpha = 0.7;
    POSE_CONNECTIONS.forEach(([a, b]) => {
      if (a >= kps.length || b >= kps.length) return;
      const ka = kps[a], kb = kps[b];
      if (ka.confidence > 0.3 && kb.confidence > 0.3) {
        ctx.beginPath();
        ctx.moveTo(ka.x * scaleX, ka.y * scaleY);
        ctx.lineTo(kb.x * scaleX, kb.y * scaleY);
        ctx.stroke();
      }
    });
    ctx.globalAlpha = 1.0;

    // Keypoints
    kps.forEach(kp => {
      if (kp.confidence > 0.3) {
        const r = 2 + kp.confidence * 2;
        ctx.beginPath();
        ctx.arc(kp.x * scaleX, kp.y * scaleY, r, 0, Math.PI * 2);
        ctx.fillStyle = COLOR_POSE;
        ctx.fill();
        // Bright center dot
        ctx.beginPath();
        ctx.arc(kp.x * scaleX, kp.y * scaleY, r * 0.4, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.globalAlpha = 0.7;
        ctx.fill();
        ctx.globalAlpha = 1.0;
      }
    });
  });
}

function drawSegmentations(segmentations, scaleX, scaleY) {
  segmentations
    .filter(s => s.confidence >= confidenceThreshold)
    .forEach(s => {
      if (!s.polygon || s.polygon.length < 3) return;

      ctx.beginPath();
      ctx.moveTo(s.polygon[0][0] * scaleX, s.polygon[0][1] * scaleY);
      s.polygon.slice(1).forEach(([x, y]) => {
        ctx.lineTo(x * scaleX, y * scaleY);
      });
      ctx.closePath();

      // Semitransparent fill
      ctx.fillStyle = COLOR_SEG_FILL;
      ctx.fill();

      // Border
      ctx.strokeStyle = COLOR_SEG;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Label at centroid
      const cx = s.polygon.reduce((sum, p) => sum + p[0], 0) / s.polygon.length * scaleX;
      const cy = s.polygon.reduce((sum, p) => sum + p[1], 0) / s.polygon.length * scaleY;

      const label = s.class_name + ' ' + s.confidence.toFixed(2);
      ctx.font = '11px "DM Mono", monospace';
      const tw = ctx.measureText(label).width;

      // Label background
      ctx.fillStyle = 'rgba(13,13,13,0.7)';
      roundRect(ctx, cx - tw/2 - 5, cy - 8, tw + 10, 18, 3);
      ctx.fill();

      ctx.fillStyle = COLOR_SEG;
      ctx.fillText(label, cx - tw/2, cy + 4);
    });
}

// Rounded rect helper
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// Binary search for closest frame by timestamp (VFR-safe).
// Frames are sorted by timestamp_ms (ascending, as read sequentially from video).
// Returns the index of the frame whose timestamp is <= targetMs (or the closest one).
function findFrameByTimestamp(frames, targetMs) {
  let lo = 0, hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >>> 1; // ceil division to avoid infinite loop
    if (frames[mid].timestamp_ms <= targetMs) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

/* ═══════════════════════════════════
   Data Loading
   ═══════════════════════════════════ */
async function loadVideoData(videoKey) {
  const jsonPath = API_BASE_URL + '/videos/processed/processed_' + videoKey + '_data.json';
  try {
    const res = await fetch(jsonPath);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    videoData = await res.json();
    // Set max playback time to match JSON data duration (typically 20s)
    if (videoData && videoData.video_info) {
      maxPlaybackTime = videoData.video_info.total_frames / videoData.video_info.fps;
    }
    updateMetricsPanel(videoData);
    buildTimeline(videoData);
    resizeCanvas();
  } catch (err) {
    console.warn('No se pudo cargar el JSON de datos:', err);
    videoData = null;
    maxPlaybackTime = Infinity;
    clearMetricsPanel();
    clearTimeline();
  }
}

function onVideoLoaded() {
  hasVideoLoaded = true;
  idleState.style.display = 'none';
  setControlsEnabled(true);
  metricsEmptyBanner.style.display = 'none';
  // Remove empty-value class from all metric values
  document.querySelectorAll('.metric-value.empty-value').forEach(el => {
    el.classList.remove('empty-value');
  });
}

function selectDemo(demoKey) {
  if (activeDemo === demoKey) return;
  activeDemo = demoKey;

  // Update card active states
  document.querySelectorAll('.demo-card').forEach(c => {
    c.classList.toggle('active', c.dataset.key === demoKey);
  });

  // Show skeleton loaders on metrics
  showSkeletonMetrics();

  // Load video
  const demo = DEMOS.find(d => d.key === demoKey);
  lastMediaTime = 0; // Reset precise time tracker
  video.src = API_BASE_URL + '/videos/original/' + demo.file + '?v=' + new Date().getTime();
  video.load();

  video.addEventListener('loadedmetadata', function onMeta() {
    video.removeEventListener('loadedmetadata', onMeta);
    onVideoLoaded();
    resizeCanvas();
    forceRedraw = true;
    startVideoFrameSync(); // Start frame-accurate sync
    
    // Explicitly call play to force Chrome to buffer the hidden video
    video.play().then(() => {
      isPlaying = true;
      updatePlayIcon();
    }).catch(err => console.error("Error reproduciendo video:", err));
  }, { once: true });

  // Load JSON data
  loadVideoData(demoKey);
}

/* ═══════════════════════════════════
   Metrics Panel Updates
   ═══════════════════════════════════ */
function showSkeletonMetrics() {
  document.getElementById('metric-total-det').innerHTML  = '<div class="skeleton skeleton-value"></div>';
  document.getElementById('metric-persons').innerHTML    = '<div class="skeleton skeleton-value"></div>';
  document.getElementById('metric-conf-avg').innerHTML   = '<div class="skeleton skeleton-value"></div>';
  document.getElementById('metric-fps').innerHTML        = '<div class="skeleton skeleton-value"></div>';
  document.getElementById('metric-fps-tooltip').textContent = '';
  document.getElementById('bar-chart').innerHTML = '<div class="skeleton skeleton-bar"></div><div class="skeleton skeleton-bar"></div><div class="skeleton skeleton-bar"></div>';
  document.getElementById('summary-box').innerHTML = '<span class="arrow">→</span> Cargando datos...';
}

function clearMetricsPanel() {
  const setEmpty = (id) => {
    const el = document.getElementById(id);
    el.textContent = '—';
    el.classList.add('empty-value');
  };
  setEmpty('metric-total-det');
  setEmpty('metric-persons');
  setEmpty('metric-conf-avg');
  setEmpty('metric-fps');
  document.getElementById('metric-fps-tooltip').textContent = '';
  document.getElementById('bar-chart').innerHTML = '';
  document.getElementById('summary-box').innerHTML = '<span class="arrow">→</span> Sin datos JSON disponibles. Video reproducido sin overlays.';
}

function updateMetricsPanel(data) {
  const info = data.video_info;
  const frames = data.frames;

  // Total custom detections
  let totalDet = 0;
  let maxPersons = 0;
  const classCounts = {};

  frames.forEach(f => {
    // Count custom detections
    if (f.detections) {
      f.detections.forEach(d => {
        if (CUSTOM_CLASSES.has(d.class_name)) totalDet++;
        classCounts[d.class_name] = (classCounts[d.class_name] || 0) + 1;
      });
    }
    // Max persons
    if (f.poses) {
      maxPersons = Math.max(maxPersons, f.poses.length);
    }
    // Segmentation class counts
    if (f.segmentations) {
      f.segmentations.forEach(s => {
        classCounts[s.class_name] = (classCounts[s.class_name] || 0) + 1;
      });
    }
  });

  // Remove empty-value from all metrics
  document.querySelectorAll('.metric-value').forEach(el => el.classList.remove('empty-value'));

  // Total detections
  document.getElementById('metric-total-det').textContent = totalDet.toLocaleString();

  // Persons tracked
  document.getElementById('metric-persons').textContent = maxPersons;

  // Confidence avg
  const confAvg = info.custom_classes_conf_avg;
  const confEl = document.getElementById('metric-conf-avg');
  let badgeClass = 'green';
  if (confAvg < 0.70) badgeClass = 'red';
  else if (confAvg < 0.85) badgeClass = 'yellow';
  confEl.innerHTML = confAvg.toFixed(2) + '<span class="conf-badge ' + badgeClass + '">' + (badgeClass === 'green' ? 'Alto' : badgeClass === 'yellow' ? 'Medio' : 'Bajo') + '</span>';

  // Inference / fps
  const inferMs = info.inference_time_ms_avg;
  const inferFps = Math.round(1000 / inferMs);
  document.getElementById('metric-fps').innerHTML = inferMs.toFixed(0) + 'ms<span class="metric-unit"> · ' + inferFps + 'fps</span>';
  document.getElementById('metric-fps-tooltip').textContent = 'Medido durante procesamiento offline';

  // Hide empty banner
  metricsEmptyBanner.style.display = 'none';

  // Bar chart
  buildBarChart(classCounts);

  // Summary
  const summaryBox = document.getElementById('summary-box');
  summaryBox.innerHTML =
    '<span class="arrow">→</span> Se detectaron <strong>' + totalDet + '</strong> instancias en <strong>' + info.total_frames + '</strong> frames.<br>' +
    '&nbsp;&nbsp;Pose estimada en <strong>' + maxPersons + '</strong> persona' + (maxPersons !== 1 ? 's' : '') + '.<br>' +
    '&nbsp;&nbsp;Confianza promedio: <strong>' + confAvg.toFixed(2) + '</strong> · <strong>' + inferMs.toFixed(0) + 'ms</strong> por frame.';
}

function buildBarChart(classCounts) {
  const container = document.getElementById('bar-chart');
  container.innerHTML = '';

  const entries = Object.entries(classCounts).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    container.innerHTML = '<div style="font-family:DM Mono,monospace;font-size:10px;color:var(--text-muted);text-align:center;padding:12px;">Sin detecciones</div>';
    return;
  }

  const maxCount = entries[0][1];

  entries.forEach(([className, count]) => {
    const isCustom = CUSTOM_CLASSES.has(className);
    const color = isCustom ? COLOR_DET : COLOR_SEG;
    const pct = (count / maxCount * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML =
      '<span class="bar-class-name" title="' + className + '">' + className + '</span>' +
      '<div class="bar-track"><div class="bar-fill" style="width:0%;background:' + color + ';"></div></div>' +
      '<span class="bar-count">' + count + '</span>';
    container.appendChild(row);

    // Animate bar fill
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        row.querySelector('.bar-fill').style.width = pct + '%';
      });
    });
  });
}

/* ═══════════════════════════════════
   Timeline
   ═══════════════════════════════════ */
const TIMELINE_BUCKETS = 100;

function buildTimeline(data) {
  const container = document.getElementById('timeline-container');
  container.innerHTML = '';

  if (!data || !data.frames || data.frames.length === 0) return;

  const buckets = Array.from({ length: TIMELINE_BUCKETS }, () => ({ det: 0, pose: 0, seg: 0 }));
  const totalFrames = data.frames.length;

  data.frames.forEach((f, i) => {
    const b = Math.min(Math.floor(i / totalFrames * TIMELINE_BUCKETS), TIMELINE_BUCKETS - 1);
    if (f.detections && f.detections.length > 0) buckets[b].det++;
    if (f.poses && f.poses.length > 0)           buckets[b].pose++;
    if (f.segmentations && f.segmentations.length > 0) buckets[b].seg++;
  });

  const maxVal = Math.max(
    ...buckets.map(b => Math.max(b.det, b.pose, b.seg)),
    1
  );

  const rows = [
    { key: 'det',  label: 'DET',  color: COLOR_DET },
    { key: 'pose', label: 'POSE', color: COLOR_POSE },
    { key: 'seg',  label: 'SEG',  color: COLOR_SEG }
  ];

  rows.forEach(rowConf => {
    const row = document.createElement('div');
    row.className = 'timeline-row';

    const label = document.createElement('span');
    label.className = 'timeline-row-label';
    label.textContent = rowConf.label;
    row.appendChild(label);

    const barsWrap = document.createElement('div');
    barsWrap.className = 'timeline-bars-wrapper';

    buckets.forEach(b => {
      const bar = document.createElement('div');
      bar.className = 'timeline-bar';
      const val = b[rowConf.key];
      if (val > 0) {
        bar.classList.add('has-data');
        bar.style.background = rowConf.color;
        bar.style.opacity = Math.max(0.3, val / maxVal);
      } else {
        bar.style.background = 'var(--border)';
        bar.style.opacity = '0.3';
      }
      barsWrap.appendChild(bar);
    });

    // Playhead (only on first row with shared positioning)
    if (rowConf.key === 'det') {
      const playhead = document.createElement('div');
      playhead.className = 'timeline-playhead';
      playhead.id = 'timeline-playhead';
      barsWrap.appendChild(playhead);
    }

    row.appendChild(barsWrap);
    container.appendChild(row);
  });
}

function updateTimelinePlayhead() {
  const playhead = document.getElementById('timeline-playhead');
  if (!playhead || !video.duration) return;
  const pct = (video.currentTime / video.duration * 100).toFixed(2);
  playhead.style.left = pct + '%';
}

function clearTimeline() {
  document.getElementById('timeline-container').innerHTML = '';
}

/* ═══════════════════════════════════
   Playback Controls
   ═══════════════════════════════════ */
function updatePlayIcon() {
  if (isPlaying) {
    playIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
  } else {
    playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
  }
}

function updateTimeDisplay() {
  if (!hasVideoLoaded) return;
  const cur = Math.min(video.currentTime, maxPlaybackTime) || 0;
  const dur = Math.min(video.duration, maxPlaybackTime) || 0;
  ctrlTime.textContent = formatTime(cur) + ' / ' + formatTime(dur);

  if (dur > 0) {
    progressFill.style.width = (cur / dur * 100).toFixed(2) + '%';
  }

  updateTimelinePlayhead();
}

function formatTime(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return m + ':' + (s < 10 ? '0' : '') + s;
}

btnPlay.addEventListener('click', () => {
  if (!hasVideoLoaded) return;
  if (video.paused) {
    video.play().catch(() => {});
    isPlaying = true;
  } else {
    video.pause();
    isPlaying = false;
    forceRedraw = true;
  }
  updatePlayIcon();
});

// #5: Retroceder 5 segundos
btnPrev.addEventListener('click', () => {
  if (!hasVideoLoaded) return;
  video.currentTime = Math.max(0, video.currentTime - 5);
  if (video.currentTime >= maxPlaybackTime) video.currentTime = maxPlaybackTime - 0.01;
  forceRedraw = true;
});

// #5: Adelantar 5 segundos
btnNext.addEventListener('click', () => {
  if (!hasVideoLoaded) return;
  const dur = Math.min(video.duration, maxPlaybackTime) || 0;
  video.currentTime = Math.min(dur, video.currentTime + 5);
  forceRedraw = true;
});

ctrlProgress.addEventListener('click', (e) => {
  if (!hasVideoLoaded || !video.duration) return;
  const rect = ctrlProgress.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  const dur = Math.min(video.duration, maxPlaybackTime) || 0;
  video.currentTime = pct * dur;
  forceRedraw = true;
});

// Mute / Volume
btnMute.addEventListener('click', () => {
  video.muted = !video.muted;
  ctrlVolume.value = video.muted ? 0 : video.volume;
});

ctrlVolume.addEventListener('input', () => {
  video.volume = parseFloat(ctrlVolume.value);
  video.muted = video.volume === 0;
});

video.addEventListener('ended', () => {
  isPlaying = false;
  updatePlayIcon();
  forceRedraw = true;
});

// #13: Idle upload link scrolls to dropzone
idleUploadLink.addEventListener('click', () => {
  const dropzoneSection = document.getElementById('dropzone-section');
  dropzoneSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
  // Briefly highlight the dropzone
  dropzone.classList.add('drag-over');
  setTimeout(() => dropzone.classList.remove('drag-over'), 1500);
});

/* ═══════════════════════════════════
   Toggles & Confidence
   ═══════════════════════════════════ */
document.querySelectorAll('.toggle-switch input').forEach(input => {
  input.addEventListener('change', () => {
    const layer = input.dataset.layer;
    layerState[layer] = input.checked;
    forceRedraw = true;
  });
});

confSlider.addEventListener('input', () => {
  confidenceThreshold = parseFloat(confSlider.value);
  confDisplay.textContent = confidenceThreshold.toFixed(2);
  confSlider.style.setProperty('--val', (confidenceThreshold * 100) + '%');
  forceRedraw = true;
});

/* ═══════════════════════════════════
   Export Frame
   ═══════════════════════════════════ */
exportBtn.addEventListener('click', () => {
  if (!hasVideoLoaded) return;

  // Force a clean render before export
  forceRedraw = true;

  canvas.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'frame_' + Math.floor(video.currentTime * 1000) + 'ms.png';
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
});

let mediaRecorder;
let recordedChunks = [];

exportVidBtn.addEventListener('click', async () => {
  if (!hasVideoLoaded) return;
  
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    return; // Already recording
  }

  const originalHtml = exportVidBtn.innerHTML;
  exportVidBtn.innerHTML = '<span style="display:inline-block; width:8px; height:8px; background:#ff4444; border-radius:50%; margin-right:6px;"></span><span style="font-size: 11px;">Grabando...</span>';
  exportVidBtn.disabled = true;

  video.currentTime = 0;
  
  await new Promise(r => { 
    video.addEventListener('seeked', r, {once: true}); 
  });
  
  const stream = canvas.captureStream(30);
  const options = { mimeType: 'video/webm; codecs=vp9', videoBitsPerSecond: 5000000 };
  if (!MediaRecorder.isTypeSupported(options.mimeType)) {
    options.mimeType = 'video/webm';
  }
  
  mediaRecorder = new MediaRecorder(stream, options);
  
  recordedChunks = [];
  mediaRecorder.ondataavailable = e => {
    if (e.data.size > 0) recordedChunks.push(e.data);
  };
  
  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'video_exportado.webm';
    a.click();
    URL.revokeObjectURL(url);
    
    exportVidBtn.innerHTML = originalHtml;
    exportVidBtn.disabled = false;
  };
  
  video.play().catch(()=>{});
  isPlaying = true;
  updatePlayIcon();
  
  mediaRecorder.start();
  
  video.addEventListener('ended', function stopRecording() {
    video.removeEventListener('ended', stopRecording);
    if (mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }
  }, {once: true});
});

/* ═══════════════════════════════════
   Demo Cards Generation
   ═══════════════════════════════════ */
function buildDemoCards() {
  const grid = document.getElementById('demo-cards-grid');
  grid.innerHTML = '';

  DEMOS.forEach((demo, idx) => {
    const card = document.createElement('div');
    card.className = 'demo-card animate-in';
    card.dataset.key = demo.key;

    // #2: No chips/tags — only title + metadata
    card.innerHTML =
      '<div class="demo-card-thumb">' +
        '<div class="thumb-placeholder" id="thumb-' + demo.key + '">' +
          '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" style="color:var(--text-muted)">' +
            '<rect x="2" y="2" width="20" height="20" rx="2"/>' +
            '<polygon points="10 8 16 12 10 16 10 8"/>' +
          '</svg>' +
        '</div>' +
        '<canvas class="thumb-canvas" id="thumb-canvas-' + demo.key + '" style="display:none;position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"></canvas>' +
        '<div class="demo-card-hover-overlay">' +
          '<div class="play-icon-circle"><svg viewBox="0 0 24 24"><polygon points="8 5 19 12 8 19 8 5"/></svg></div>' +
        '</div>' +
      '</div>' +
      '<div class="demo-card-info">' +
        '<div class="demo-card-name">' + demo.name + '</div>' +
        '<div class="demo-card-meta" id="meta-' + demo.key + '"><span id="duration-' + demo.key + '">—</span> · ' + demo.file + '</div>' +
      '</div>';

    card.addEventListener('click', () => selectDemo(demo.key));
    grid.appendChild(card);

    // Try loading thumbnail
    loadThumbnail(demo);
  });
}

function loadThumbnail(demo) {
  const thumbVideo = document.createElement('video');
  thumbVideo.crossOrigin = 'anonymous';
  thumbVideo.muted = true;
  thumbVideo.preload = 'metadata';
  thumbVideo.src = API_BASE_URL + '/videos/original/' + demo.file + '?v=' + new Date().getTime();

  thumbVideo.addEventListener('loadeddata', () => {
    thumbVideo.currentTime = 0.5; // Seek to 0.5s for a good thumbnail
  });

  thumbVideo.addEventListener('seeked', () => {
    const thumbCanvas = document.getElementById('thumb-canvas-' + demo.key);
    const placeholder = document.getElementById('thumb-' + demo.key);
    if (thumbCanvas) {
      thumbCanvas.width = thumbVideo.videoWidth;
      thumbCanvas.height = thumbVideo.videoHeight;
      const tCtx = thumbCanvas.getContext('2d');
      tCtx.drawImage(thumbVideo, 0, 0);
      thumbCanvas.style.display = 'block';
      if (placeholder) placeholder.style.display = 'none';
    }

    // Set duration
    const durEl = document.getElementById('duration-' + demo.key);
    if (durEl && thumbVideo.duration) {
      durEl.textContent = formatTime(thumbVideo.duration);
    }

    // Cleanup
    thumbVideo.src = '';
    thumbVideo.load();
  });

  thumbVideo.addEventListener('error', () => {
    // Keep placeholder visible
  });
}

/* ═══════════════════════════════════
   Dropzone & File Upload
   ═══════════════════════════════════ */
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('drag-over');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files.length > 0) handleUpload(files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    handleUpload(fileInput.files[0]);
    fileInput.value = '';
  }
});

function handleUpload(file) {
  dropzoneError.classList.remove('visible');

  if (!file.type.startsWith('video/')) {
    showDropzoneError('El archivo debe ser un video.');
    return;
  }

    // Se eliminó la validación de duración porque algunos videos reportan duraciones erróneas.

    // Valid — start loading sequence
    activeDemo = null;
    document.querySelectorAll('.demo-card').forEach(c => c.classList.remove('active'));
    videoData = null;
    clearMetricsPanel();
    clearTimeline();

    startLoadingSequence(file);
}

function showDropzoneError(msg) {
  dropzoneError.textContent = msg;
  dropzoneError.classList.add('visible');
}

/* ═══════════════════════════════════
   Loading Sequence (Upload)
   ═══════════════════════════════════ */
const LOADING_STEPS = []; // Retained to avoid breaking other scripts if any

async function startLoadingSequence(file) {
  loadingOv.classList.add('active');
  idleState.style.display = 'none';
  
  const progressText = document.getElementById('real-progress-text');
  const progressBar = document.getElementById('real-progress-bar');
  progressText.textContent = "Subiendo archivo...";
  progressBar.style.width = "0%";

  showSkeletonMetrics();

  const formData = new FormData();
  formData.append("file", file);

  // Progreso simulado mientras el backend procesa de forma síncrona
  let fakeProgress = 0;
  let dotCount = 0;
  progressText.textContent = "Aplicando inferencia YOLO";
  const fakeInterval = setInterval(() => {
    fakeProgress += (90 - fakeProgress) * 0.1; 
    progressBar.style.width = fakeProgress + "%";
    
    dotCount = (dotCount + 1) % 4;
    progressText.textContent = "Aplicando inferencia YOLO" + ".".repeat(dotCount);
  }, 500);

  let uploadRes;
  try {
    uploadRes = await fetch(API_BASE_URL + '/api/upload_video', {
      method: 'POST',
      body: formData
    });
    if (!uploadRes.ok) throw new Error("Error en el servidor al procesar video");
  } catch (err) {
    clearInterval(fakeInterval);
    loadingOv.classList.remove('active');
    showDropzoneError(err.message);
    clearMetricsPanel();
    return;
  }

  const result = await uploadRes.json();
  clearInterval(fakeInterval);

  if (result.status === "error") {
    loadingOv.classList.remove('active');
    showDropzoneError(result.message || "Error procesando video");
    clearMetricsPanel();
    return;
  }

  progressBar.style.width = "100%";
  progressText.textContent = "Compilando resultados...";

  // Create local URL for video (evita descargar/guardar en servidor)
  const localVideoUrl = URL.createObjectURL(file);

  setTimeout(() => {
    loadingOv.classList.remove('active');
    videoData = result.data;
    updateMetricsPanel(videoData);
    buildTimeline(videoData);
    
    video.src = localVideoUrl;
    video.load();
  }, 800);

  video.addEventListener('loadedmetadata', function onReady() {
    video.removeEventListener('loadedmetadata', onReady);
    onVideoLoaded();
    
    // Reset time tracker for the new upload
    lastMediaTime = 0;
    startVideoFrameSync(); // Enable frame-accurate sync for uploads too
    
    video.play().then(() => {
      isPlaying = true;
      updatePlayIcon();
    }).catch(err => console.error("Error reproduciendo video subido:", err));
    
    resizeCanvas();
    forceRedraw = true;
  }, { once: true });
}

function typewrite(el, text, charDelay) {
  return new Promise(resolve => {
    let i = 0;
    function tick() {
      if (i < text.length) {
        el.textContent += text.charAt(i);
        i++;
        setTimeout(tick, charDelay);
      } else {
        resolve();
      }
    }
    tick();
  });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/* ═══════════════════════════════════
   Initialization
   ═══════════════════════════════════ */
function init() {
  buildDemoCards();
  resizeCanvas();
  setControlsEnabled(false); // Start with controls disabled
  requestAnimationFrame(renderLoop);
}

document.addEventListener('DOMContentLoaded', init);