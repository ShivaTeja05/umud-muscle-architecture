/* UMUD offline annotator.  No external dependencies; served by tools/annotator/serve.py.
 *
 * Coordinates: native image pixels of the task image, x right, y down (continuous, a click at
 * screen point p maps to (p - t) / s).  Scale: px_per_cm is the VERTICAL scale bar, applied to
 * both axes (square pixels), exactly as the organiser does.
 *   MT = |y_deep - y_superficial| * mm/px          (x locked to the first click)
 *   FL = sum of polyline segment lengths * mm/px    (points may lie outside the image)
 *   PA = acute angle between line(f0,f1) and line(a0,a1), in degrees
 * Each parameter = mean of its (up to) 3 measurements.
 *
 * Phases (task queue): 'calib' (data/calib_manifest.json, 12 benchmark images with hidden expert
 * answers; must be finished or explicitly skipped first), then 'clips' (the 28 Telemed clip tasks),
 * then 'test' (the other 169 test tasks).  A task may carry `need` (dynamic calibration frames:
 * MT optional) and view-only `frames`.  Calibration tasks never show model values.
 */
'use strict';
(() => {
  const LS_KEY = 'umud_annotator_v1';
  const NEED = 3;
  const DEFAULT_NEED = { mt: NEED, fl: NEED, pa: NEED };
  const TOOLS = ['mt', 'fl', 'pa'];
  const TEST_HEADER = ['image_id', 'pa_deg', 'fl_mm', 'mt_mm', 'clicks_json', 'task_id', 'clip', 'n_pa', 'n_fl',
    'n_mt', 'complete', 'flag', 'note', 'px_per_cm', 'active_s', 'updated'];
  const CALIB_HEADER = ['image_id', 'kind', 'pa_deg', 'pa_h_deg', 'fl_mm', 'fl_px', 'mt_mm', 'mt_px', 'clicks_json',
    'task_id', 'n_pa', 'n_fl', 'n_mt', 'need_mt', 'complete', 'flag', 'note', 'px_per_cm', 'active_s', 'updated'];
  const PHASE_NAMES = { calib: 'Calibration', clips: 'Telemed clips', test: 'Other test images' };
  const NAMES = { mt: 'MT', fl: 'FL', pa: 'PA' };
  const UNITS = { mt: 'mm', fl: 'mm', pa: 'deg' };
  const COLORS = { mt: '#22d3ee', fl: '#facc15', pa: '#f472b6', apo: '#4ade80', pending: '#ffffff' };
  const RANGES = { pa: [5, 45], fl: [30, 200], mt: [10, 50] };
  const REVIEW_DELTA = { pa: 4, fl: 15, mt: 2 }; // flag large disagreement with the model hint
  const GUIDE_FRACS = [0.25, 0.5, 0.75];
  const SNAP_PX = 8; // screen px for MT guide snapping
  const HIT_PX = 7; // screen px to grab an existing vertex
  const DRAG_PX = 5; // movement before a click becomes a pan
  const IDLE_MS = 45000; // stop the per-image timer after this much inactivity
  const DEFAULT_SETTINGS = {
    bright: 0, contrast: 1, gamma: 1, autoEach: false, guides: true, loupe: true,
    scale: false, labels: true, trackpad: false, lastTask: null,
  };

  const $ = (id) => document.getElementById(id);
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');
  const loupe = $('loupe');
  const lctx = loupe.getContext('2d');

  let manifest = null;
  let calibManifest = null; // null when data/calib_manifest.json is absent (no calibration phase)
  let imagesById = {};
  let tasks = [];
  let taskIndexById = {};
  let pendingGateIdx = -1;
  let state = { version: 1, labels: {}, times: {}, settings: { ...DEFAULT_SETTINGS }, updated: 0 };
  const cur = { idx: -1, frame: 0, tool: 'mt', pending: [], entry: null, src: null, adj: null, adjCtx: null, out: null };
  const view = { s: 1, tx: 0, ty: 0, userMoved: false }; // userMoved: zoomed/panned since the last fit
  const mouse = { x: 0, y: 0, inside: false, down: null, drag: null, hover: null };
  const undoStacks = {};
  const redoStacks = {};
  const imgCache = new Map();
  let spaceHeld = false;
  let rafId = 0;
  let lastInteraction = Date.now();
  let serverOK = null;
  let serverTimer = 0;
  let toastTimer = 0;

  // ---------- small helpers ----------
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const r2 = (v) => Math.round(v * 100) / 100;
  const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '–');
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const task = () => tasks[cur.idx];
  const taskEntry = (t = task()) => imagesById[t.image_id];
  const mmPerPx = (t = task()) => 10 / taskEntry(t).px_per_cm;
  const toImg = (p) => [(p.x - view.tx) / view.s, (p.y - view.ty) / view.s];
  const toScr = (q, T = view) => ({ x: q[0] * T.s + T.tx, y: q[1] * T.s + T.ty });
  const taskById = (id) => tasks[taskIndexById[id]];
  const needOf = (t = task()) => (t && t.need) || DEFAULT_NEED;
  const isCalib = (t = task()) => !!t && t.phase === 'calib';
  const viewFrames = (t) => t.frames || t.members;
  const calibTasks = () => tasks.filter((t) => t.phase === 'calib');

  function getLabel(id, create) {
    let lab = state.labels[id];
    if (!lab && create) {
      lab = { mt: [], fl: [], pa: [], flag: false, note: '', created: Date.now(), updated: Date.now() };
      state.labels[id] = lab;
    }
    return lab || null;
  }

  // ---------- measurement math ----------
  const mtValue = (m, mmpp) => Math.abs(m[1][1] - m[0][1]) * mmpp;
  function flValue(pts, mmpp) {
    let len = 0;
    for (let i = 1; i < pts.length; i++) len += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    return len * mmpp;
  }
  function angleBetween(p0, p1, q0, q1) {
    const a = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);
    const b = Math.atan2(q1[1] - q0[1], q1[0] - q0[0]);
    let d = Math.abs(a - b) * 180 / Math.PI % 180;
    if (d > 90) d = 180 - d;
    return d;
  }
  const paValue = (m) => angleBetween(m[0], m[1], m[2], m[3]);
  // fascicle line (first 2 PA clicks) against the image horizontal; recorded for the calibration
  // scorer (the dynamic benchmark's convention), never shown
  const paHValue = (m) => angleBetween(m[0], m[1], [0, 0], [1, 0]);
  function value(tool, m, mmpp) {
    if (tool === 'mt') return mtValue(m, mmpp);
    if (tool === 'fl') return flValue(m, mmpp);
    return paValue(m);
  }
  const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
  function summarize(lab, mmpp, need = DEFAULT_NEED) {
    const out = {};
    for (const tool of TOOLS) {
      const vals = lab ? lab[tool].map((m) => value(tool, m, mmpp)) : [];
      out[tool] = mean(vals);
      out['n_' + tool] = vals.length;
      out['vals_' + tool] = vals;
    }
    const items = (tool) => (lab ? lab[tool] : []);
    out.pa_h = mean(items('pa').map(paHValue));
    out.fl_px = mean(items('fl').map((m) => flValue(m, 1)));
    out.mt_px = mean(items('mt').map((m) => mtValue(m, 1)));
    out.complete = TOOLS.every((t) => out['n_' + t] >= (need[t] ?? NEED));
    return out;
  }
  function status(id) {
    const lab = state.labels[id];
    if (!lab) return 'none';
    if (lab.flag) return 'flag';
    const need = needOf(taskById(id));
    const n = TOOLS.map((t) => lab[t].length);
    if (TOOLS.every((t, i) => n[i] >= (need[t] ?? NEED))) return 'done';
    return n.some((k) => k > 0) || lab.note ? 'partial' : 'none';
  }
  const handled = (id) => ['done', 'flag'].includes(status(id));
  const calibPending = () => calibTasks().some((t) => !handled(t.task_id));
  function gateBlocks(idx) {
    const t = tasks[idx];
    return !!t && !isCalib(t) && calibPending() && !state.settings.calibSkipped;
  }
  function lineIntersection(p0, p1, q0, q1) {
    const d1x = p1[0] - p0[0], d1y = p1[1] - p0[1], d2x = q1[0] - q0[0], d2y = q1[1] - q0[1];
    const den = d1x * d2y - d1y * d2x;
    if (Math.abs(den) < 1e-9) return null;
    const t = ((q0[0] - p0[0]) * d2y - (q0[1] - p0[1]) * d2x) / den;
    return [p0[0] + t * d1x, p0[1] + t * d1y];
  }
  function guideXs(entry) {
    const [x1, , x2] = entry.viewport;
    return GUIDE_FRACS.map((f) => x1 + f * (x2 - x1));
  }

  // ---------- persistence ----------
  function touch(lab) {
    const now = Date.now();
    if (lab) lab.updated = now;
    state.updated = now;
    saveLocal();
    scheduleServerSave();
  }
  function saveLocal() {
    state.settings.ts = Date.now(); // settings (last task, contrast...) merge by this stamp
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(state));
      return true;
    } catch (err) {
      return false;
    }
  }
  function scheduleServerSave(delay = 700) {
    clearTimeout(serverTimer);
    serverTimer = setTimeout(saveServer, delay);
  }
  async function saveServer() {
    const local = saveLocal();
    try {
      const res = await fetch('api/state', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Annotator': '1' },
        body: JSON.stringify({ state }),
      });
      const js = await res.json();
      if (!res.ok || !js.ok) throw new Error(js.error || res.status);
      serverOK = true;
      setSaveStatus(`saved ${js.saved} (browser${local ? '' : ' FAILED'} + disk)`, !local);
      if (js.newer_on_disk && js.newer_on_disk.length) await pullNewer(js.newer_on_disk.length);
    } catch (err) {
      serverOK = false;
      setSaveStatus(`disk save failed (${err.message}); browser ${local ? 'ok' : 'FAILED'}`, true);
    }
  }
  async function pullNewer(n) {
    // Another tab/session saved newer labels for some images: merge them in (newest wins).
    try {
      const res = await fetch('api/state', { headers: { 'X-Annotator': '1' }, cache: 'no-store' });
      if (!res.ok) return;
      const disk = await res.json();
      const settings = state.settings;
      state = mergeStates(state, { ...disk, updated: 0 });
      state.settings = settings;
      for (const lab of Object.values(state.labels)) for (const t of TOOLS) lab[t] = lab[t] || [];
      saveLocal();
      toast(`merged ${n} newer label(s) from disk: is another annotator tab open?`);
      refresh();
    } catch (err) { /* keep local state */ }
  }
  function setSaveStatus(text, bad) {
    const el = $('save-status');
    el.textContent = text;
    el.classList.toggle('err', !!bad);
  }
  function mergeStates(a, b) {
    // Per-task merge, newest label wins; settings from the newer state.
    const [older, newer] = (a.updated || 0) <= (b.updated || 0) ? [a, b] : [b, a];
    const labels = { ...(older.labels || {}) };
    for (const [id, lab] of Object.entries(newer.labels || {})) {
      if (!labels[id] || (lab.updated || 0) >= (labels[id].updated || 0)) labels[id] = lab;
    }
    const times = { ...(older.times || {}) };
    for (const [id, ms] of Object.entries(newer.times || {})) times[id] = Math.max(ms, times[id] || 0);
    const sa = a.settings || {}, sb = b.settings || {};
    const [s0, s1] = (sa.ts || 0) <= (sb.ts || 0) ? [sa, sb] : [sb, sa];
    return {
      version: 1, labels, times,
      settings: { ...DEFAULT_SETTINGS, ...s0, ...s1 },
      updated: Math.max(a.updated || 0, b.updated || 0),
    };
  }

  // ---------- CSV ----------
  function csvCell(v) {
    const s = v === null || v === undefined ? '' : String(v);
    return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }
  // Test CSV (labels.csv format) or calibration CSV (calib_labels.csv format); both follow the
  // manifest order so they are byte-identical to labels.py.
  function buildCSV(kind = 'test') {
    const calib = kind === 'calib';
    const lines = [(calib ? CALIB_HEADER : TEST_HEADER).join(',')];
    const list = calib ? (calibManifest ? calibManifest.tasks : []) : manifest.tasks;
    const v = (x) => (Number.isFinite(x) ? x.toFixed(3) : '');
    for (const t of list) {
      const lab = state.labels[t.task_id];
      if (!lab || (!TOOLS.some((k) => lab[k].length) && !lab.flag && !lab.note)) continue;
      const entry = taskEntry(t);
      const need = needOf(t);
      const s = summarize(lab, mmPerPx(t), need);
      const clicks = JSON.stringify({ task_id: t.task_id, px_per_cm: entry.px_per_cm, mt: lab.mt, fl: lab.fl, pa: lab.pa });
      const active = Math.round((state.times[t.task_id] || 0) / 1000);
      const stamp = new Date(lab.updated || 0).toISOString();
      if (calib) {
        lines.push([t.image_id, t.kind || '', v(s.pa), v(s.pa_h), v(s.fl), v(s.fl_px), v(s.mt), v(s.mt_px), clicks,
          t.task_id, s.n_pa, s.n_fl, s.n_mt, need.mt ?? NEED, s.complete ? 1 : 0, lab.flag ? 1 : 0, lab.note || '',
          entry.px_per_cm, active, stamp].map(csvCell).join(','));
        continue;
      }
      for (const member of t.members) {
        lines.push([member, v(s.pa), v(s.fl), v(s.mt), clicks, t.task_id, t.clip ?? '', s.n_pa, s.n_fl, s.n_mt,
          s.complete ? 1 : 0, lab.flag ? 1 : 0, lab.note || '', entry.px_per_cm, active, stamp].map(csvCell).join(','));
      }
    }
    return lines.join('\n') + '\n';
  }
  function parseCSV(text) {
    const rows = [];
    let row = [], field = '', q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) {
        if (c === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; } else q = false;
        } else field += c;
      } else if (c === '"') q = true;
      else if (c === ',') { row.push(field); field = ''; }
      else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
      else if (c !== '\r') field += c;
    }
    if (field.length || row.length) { row.push(field); rows.push(row); }
    return rows;
  }
  function exportCSV(kind = 'test') {
    const blob = new Blob([buildCSV(kind)], { type: 'text/csv' });
    const a = document.createElement('a');
    const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
    a.href = URL.createObjectURL(blob);
    a.download = kind === 'calib' ? `umud_calib_labels_${stamp}.csv` : `umud_hand_labels_${stamp}.csv`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }
  function validPts(arr, n) {
    return Array.isArray(arr) && arr.every((m) => Array.isArray(m) && (n ? m.length === n : m.length >= 2)
      && m.every((p) => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite)));
  }
  function importCSV(text) {
    const rows = parseCSV(text.replace(/^﻿/, ''));
    if (!rows.length) return toast('empty CSV');
    const head = rows[0].map((h) => h.trim());
    const col = (name) => head.indexOf(name);
    const ci = col('clicks_json');
    if (ci < 0) return toast('CSV has no clicks_json column');
    const incoming = {};
    for (const r of rows.slice(1)) {
      if (!r[ci]) continue;
      let obj;
      try { obj = JSON.parse(r[ci]); } catch (err) { continue; }
      const tid = obj.task_id || (col('task_id') >= 0 && r[col('task_id')]) || r[col('image_id')];
      if (!(tid in taskIndexById) || incoming[tid]) continue;
      const mt = obj.mt || [], fl = obj.fl || [], pa = obj.pa || [];
      if (!validPts(mt, 2) || !validPts(fl, 0) || !validPts(pa, 4)) continue;
      incoming[tid] = {
        mt, fl, pa,
        flag: col('flag') >= 0 && r[col('flag')] === '1',
        note: col('note') >= 0 ? r[col('note')] : '',
        created: Date.now(), updated: Date.now(),
      };
    }
    const ids = Object.keys(incoming);
    if (!ids.length) return toast('no valid rows found in CSV');
    const clash = ids.filter((id) => state.labels[id] && status(id) !== 'none').length;
    if (!window.confirm(`Import ${ids.length} images from CSV?` + (clash ? `\n${clash} already have labels here and will be OVERWRITTEN.` : ''))) return;
    for (const id of ids) {
      if (state.labels[id]) pushUndo(id);
      state.labels[id] = incoming[id];
    }
    touch(null);
    toast(`imported ${ids.length} images`);
    refresh();
  }

  // ---------- undo / redo ----------
  function snap(id) {
    const lab = getLabel(id, true);
    return JSON.stringify({ mt: lab.mt, fl: lab.fl, pa: lab.pa });
  }
  function pushUndo(id = task().task_id) {
    (undoStacks[id] = undoStacks[id] || []).push(snap(id));
    if (undoStacks[id].length > 200) undoStacks[id].shift();
    redoStacks[id] = [];
  }
  function restore(id, json) {
    const lab = getLabel(id, true);
    Object.assign(lab, JSON.parse(json));
    touch(lab);
  }
  function undo() {
    const id = task().task_id;
    if (cur.pending.length) { cur.pending.pop(); return refresh(); }
    const st = undoStacks[id];
    if (!st || !st.length) return toast('nothing to undo');
    (redoStacks[id] = redoStacks[id] || []).push(snap(id));
    restore(id, st.pop());
    refresh();
  }
  function redo() {
    const id = task().task_id;
    const st = redoStacks[id];
    if (!st || !st.length) return toast('nothing to redo');
    (undoStacks[id] = undoStacks[id] || []).push(snap(id));
    restore(id, st.pop());
    refresh();
  }

  // ---------- editing ----------
  function firstIncompleteTool(lab, need = needOf()) {
    const required = TOOLS.filter((t) => (need[t] ?? NEED) > 0);
    if (!lab) return required[0] || 'mt';
    return required.find((t) => lab[t].length < (need[t] ?? NEED)) || cur.tool;
  }
  function setTool(tool) {
    if (cur.tool !== tool) cur.pending = [];
    cur.tool = tool;
    refresh();
  }
  function commit(tool, pts) {
    const id = task().task_id;
    const lab = getLabel(id, true);
    const wasComplete = status(id) === 'done';
    const calibWasPending = calibPending();
    pushUndo(id);
    lab[tool].push(pts.map((p) => [r2(p[0]), r2(p[1])]));
    cur.pending = [];
    touch(lab);
    if (lab[tool].length >= NEED) cur.tool = firstIncompleteTool(lab);
    if (!wasComplete && summarize(lab, mmPerPx(), needOf()).complete) {
      if (isCalib() && calibWasPending && !calibPending()) toast('Calibration finished! Next: the 28 Telemed clips. Press N.', 'ok');
      else toast('Image finished: check the review panel, then N for next', 'ok');
    }
    refresh();
  }
  function deleteMeasurement(tool, i) {
    const id = task().task_id;
    const lab = getLabel(id, true);
    pushUndo(id);
    lab[tool].splice(i, 1);
    cur.pending = [];
    cur.tool = tool;
    touch(lab);
    refresh();
  }
  function snapMTx(x) {
    if (!state.settings.guides) return x;
    let best = x, bestD = Infinity;
    for (const gx of guideXs(taskEntry())) {
      const d = Math.abs(gx - x) * view.s;
      if (d < bestD) { bestD = d; best = gx; }
    }
    return bestD <= SNAP_PX ? best : x;
  }
  function handleClick(p) {
    const q = toImg(p);
    const tool = cur.tool;
    const lab = getLabel(task().task_id, false);
    if (lab && lab[tool].length >= NEED && !cur.pending.length) {
      const need = needOf();
      const other = TOOLS.find((t) => lab[t].length < (need[t] ?? NEED) && (need[t] ?? NEED) > 0);
      return toast(`${NAMES[tool]} already has 3. ` + (other ? `Press ${other === 'mt' ? 'M' : other === 'fl' ? 'F' : 'P'} for ${NAMES[other]}, ` : '') + 'or delete one.');
    }
    if (cur.pending.length) {
      const last = toScr(cur.pending[cur.pending.length - 1]);
      if (Math.hypot(last.x - p.x, last.y - p.y) < 4) return; // double-click / jitter duplicate
    }
    if (tool === 'mt') {
      if (!cur.pending.length) cur.pending = [[snapMTx(q[0]), q[1]]];
      else return commit('mt', [cur.pending[0], [cur.pending[0][0], q[1]]]);
    } else if (tool === 'fl') {
      cur.pending.push(q);
    } else {
      cur.pending.push(q);
      if (cur.pending.length === 4) return commit('pa', cur.pending);
    }
    refresh();
  }
  function finishFL() {
    if (cur.tool !== 'fl') return;
    if (cur.pending.length < 2) return toast('a fascicle needs at least 2 points');
    commit('fl', cur.pending);
  }
  function cancelPending() {
    if (cur.pending.length) { cur.pending = []; refresh(); }
  }
  function hitVertex(p) {
    const lab = getLabel(task().task_id, false);
    if (!lab) return null;
    let best = null, bestD = HIT_PX;
    for (const tool of TOOLS) {
      lab[tool].forEach((m, mi) => m.forEach((q, pi) => {
        const s = toScr(q);
        const d = Math.hypot(s.x - p.x, s.y - p.y);
        if (d < bestD) { bestD = d; best = { tool, mi, pi }; }
      }));
    }
    return best;
  }
  function moveVertex(hit, q) {
    const lab = getLabel(task().task_id, true);
    const m = lab[hit.tool][hit.mi];
    const qq = [r2(q[0]), r2(q[1])];
    if (hit.tool === 'mt') {
      if (hit.pi === 0) { m[0] = qq; m[1] = [qq[0], m[1][1]]; } else m[1] = [m[0][0], qq[1]];
    } else m[hit.pi] = qq;
  }

  // ---------- view ----------
  function fit() {
    const e = cur.entry;
    if (!e) return;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    view.userMoved = false; // refit on resize until the user zooms or pans
    if (W < 80 || H < 80) return requestDraw(); // layout not ready yet
    view.s = Math.min((W - 40) / e.width, (H - 40) / e.height);
    view.tx = (W - e.width * view.s) / 2;
    view.ty = (H - e.height * view.s) / 2;
    requestDraw();
  }
  function zoomAt(px, py, f) {
    const s2 = clamp(view.s * f, 0.05, 60);
    view.tx = px - (px - view.tx) * (s2 / view.s);
    view.ty = py - (py - view.ty) * (s2 / view.s);
    view.s = s2;
    view.userMoved = true;
    requestDraw();
  }

  // ---------- images ----------
  function loadImageEl(entry) {
    if (imgCache.has(entry.image_id)) return imgCache.get(entry.image_id);
    const pr = new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => reject(new Error('cannot load ' + entry.file));
      im.src = 'data/' + entry.file;
    });
    imgCache.set(entry.image_id, pr);
    while (imgCache.size > 8) imgCache.delete(imgCache.keys().next().value);
    pr.catch(() => imgCache.delete(entry.image_id));
    return pr;
  }
  async function showImage(entry, keepView) {
    const token = (showImage.token = (showImage.token || 0) + 1);
    let im;
    try { im = await loadImageEl(entry); } catch (err) { return toast(err.message); }
    if (token !== showImage.token) return;
    const c = document.createElement('canvas');
    c.width = entry.width; c.height = entry.height;
    const cc = c.getContext('2d', { willReadFrequently: true });
    cc.drawImage(im, 0, 0, entry.width, entry.height);
    cur.src = cc.getImageData(0, 0, entry.width, entry.height);
    cur.adj = document.createElement('canvas');
    cur.adj.width = entry.width; cur.adj.height = entry.height;
    cur.adjCtx = cur.adj.getContext('2d');
    cur.out = cur.adjCtx.createImageData(entry.width, entry.height);
    cur.entry = entry;
    if (state.settings.autoEach) autoContrast(false);
    applyAdjust();
    if (!keepView) fit();
    requestDraw();
  }
  function applyAdjust() {
    if (!cur.src) return;
    const { bright: b, contrast: c, gamma: g } = state.settings;
    const lut = new Uint8ClampedArray(256);
    for (let v = 0; v < 256; v++) {
      const y = clamp((v / 255 - 0.5) * c + 0.5 + b, 0, 1);
      lut[v] = Math.round(Math.pow(y, 1 / g) * 255);
    }
    const s = cur.src.data, o = cur.out.data;
    for (let i = 0; i < s.length; i += 4) {
      o[i] = lut[s[i]]; o[i + 1] = lut[s[i + 1]]; o[i + 2] = lut[s[i + 2]]; o[i + 3] = 255;
    }
    cur.adjCtx.putImageData(cur.out, 0, 0);
    requestDraw();
  }
  function autoContrast(apply = true) {
    if (!cur.src || !cur.entry) return;
    const [x1, y1, x2, y2] = cur.entry.viewport;
    const w = cur.entry.width, d = cur.src.data;
    const hist = new Uint32Array(256);
    let n = 0;
    for (let y = Math.max(0, y1); y < Math.min(cur.entry.height, y2); y += 2) {
      for (let x = Math.max(0, x1); x < Math.min(w, x2); x += 2) { hist[d[(y * w + x) * 4]]++; n++; }
    }
    const pct = (p) => { let acc = 0; for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= p * n) return v; } return 255; };
    const lo = pct(0.01) / 255, hi = Math.max(pct(0.995) / 255, lo + 0.05);
    const c = clamp(1 / (hi - lo), 0.4, 4);
    const b = clamp(-0.5 - (lo - 0.5) * c, -0.6, 0.6);
    state.settings.contrast = r2(c);
    state.settings.bright = r2(b);
    syncControls();
    if (apply) { applyAdjust(); saveLocal(); }
  }

  // ---------- drawing ----------
  function requestDraw() {
    if (!rafId) rafId = requestAnimationFrame(draw);
  }
  function strokePath(c, pts, T, color, width = 1.5, dash = null) {
    if (pts.length < 2) return;
    c.save();
    c.strokeStyle = color;
    c.lineWidth = width;
    if (dash) c.setLineDash(dash);
    c.beginPath();
    pts.forEach((q, i) => { const s = toScr(q, T); if (i) c.lineTo(s.x, s.y); else c.moveTo(s.x, s.y); });
    c.stroke();
    c.restore();
  }
  function dot(c, q, T, color, r = 2.5) {
    const s = toScr(q, T);
    c.fillStyle = color;
    c.beginPath();
    c.arc(s.x, s.y, r, 0, Math.PI * 2);
    c.fill();
  }
  function text(c, str, x, y, color) {
    c.save();
    c.font = '600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    c.lineWidth = 3;
    c.strokeStyle = 'rgba(0,0,0,.85)';
    c.strokeText(str, x, y);
    c.fillStyle = color;
    c.fillText(str, x, y);
    c.restore();
  }
  function drawPA(c, m, T, opts, label) {
    const [f0, f1, a0, a1] = m;
    strokePath(c, [f0, f1], T, COLORS.pa, opts.w);
    if (a1) strokePath(c, [a0, a1], T, COLORS.apo, opts.w);
    if (a1) {
      const X = lineIntersection(f0, f1, a0, a1);
      if (X) {
        const near = (seg) => (Math.hypot(seg[0][0] - X[0], seg[0][1] - X[1]) < Math.hypot(seg[1][0] - X[0], seg[1][1] - X[1]) ? seg[0] : seg[1]);
        const span = Math.hypot(f1[0] - f0[0], f1[1] - f0[1]) + Math.hypot(a1[0] - a0[0], a1[1] - a0[1]);
        if (Math.hypot(near([f0, f1])[0] - X[0], near([f0, f1])[1] - X[1]) < 3 * span + 50) {
          strokePath(c, [near([f0, f1]), X], T, COLORS.pa, 1, [4, 4]);
          strokePath(c, [near([a0, a1]), X], T, COLORS.apo, 1, [4, 4]);
        }
      }
    }
    if (label && opts.labels) { const s = toScr(f1, T); text(c, label, s.x + 6, s.y - 6, COLORS.pa); }
  }
  function drawOverlays(c, T, opts) {
    const t = task();
    const lab = getLabel(t.task_id, false);
    const mmpp = mmPerPx(t);
    if (lab) {
      lab.mt.forEach((m, i) => {
        strokePath(c, m, T, COLORS.mt, opts.w);
        for (const q of m) { const s = toScr(q, T); c.strokeStyle = COLORS.mt; c.lineWidth = opts.w; c.beginPath(); c.moveTo(s.x - 6, s.y); c.lineTo(s.x + 6, s.y); c.stroke(); }
        if (opts.labels) { const s = toScr([m[0][0], (m[0][1] + m[1][1]) / 2], T); text(c, `MT${i + 1} ${fmt(mtValue(m, mmpp))}`, s.x + 6, s.y, COLORS.mt); }
      });
      lab.fl.forEach((m, i) => {
        strokePath(c, m, T, COLORS.fl, opts.w);
        if (opts.labels) { const k = m[Math.floor((m.length - 1) / 2)]; const s = toScr(k, T); text(c, `FL${i + 1} ${fmt(flValue(m, mmpp))}`, s.x + 6, s.y - 6, COLORS.fl); }
      });
      lab.pa.forEach((m, i) => drawPA(c, m, T, opts, `PA${i + 1} ${fmt(paValue(m))}°`));
      if (opts.dots) {
        for (const tool of TOOLS) lab[tool].forEach((m) => m.forEach((q) => dot(c, q, T, tool === 'fl' ? COLORS.fl : tool === 'mt' ? COLORS.mt : COLORS.pa)));
      }
    }
    // pending measurement + rubber band to the cursor
    const pend = cur.pending;
    const mq = toImg(mouse);
    if (pend.length) {
      if (cur.tool === 'mt') {
        const p1 = [pend[0][0], mq[1]];
        strokePath(c, [pend[0], p1], T, COLORS.pending, opts.w, [5, 3]);
        dot(c, pend[0], T, COLORS.mt, 3);
      } else if (cur.tool === 'fl') {
        strokePath(c, pend, T, COLORS.fl, opts.w);
        if (mouse.inside) strokePath(c, [pend[pend.length - 1], mq], T, COLORS.pending, opts.w, [5, 3]);
        pend.forEach((q) => dot(c, q, T, COLORS.fl, 3));
      } else {
        if (pend.length >= 2) strokePath(c, [pend[0], pend[1]], T, COLORS.pa, opts.w);
        if (pend.length === 1 && mouse.inside) strokePath(c, [pend[0], mq], T, COLORS.pending, opts.w, [5, 3]);
        if (pend.length === 3 && mouse.inside) drawPA(c, [pend[0], pend[1], pend[2], mq], T, { ...opts, labels: false });
        pend.forEach((q, i) => dot(c, q, T, i < 2 ? COLORS.pa : COLORS.apo, 3));
      }
    }
    if (opts.hover && mouse.hover && lab) {
      const q = lab[mouse.hover.tool][mouse.hover.mi][mouse.hover.pi];
      const s = toScr(q, T);
      c.strokeStyle = '#fff'; c.lineWidth = 1.5;
      c.beginPath(); c.arc(s.x, s.y, 6, 0, Math.PI * 2); c.stroke();
    }
  }
  function liveValue() {
    const pend = cur.pending;
    const mq = toImg(mouse);
    const mmpp = mmPerPx();
    if (!pend.length) return null;
    if (cur.tool === 'mt') return `${fmt(Math.abs(mq[1] - pend[0][1]) * mmpp)} mm`;
    if (cur.tool === 'fl') return `${fmt(flValue([...pend, mq], mmpp))} mm`;
    if (pend.length === 3) return `${fmt(angleBetween(pend[0], pend[1], pend[2], mq))}°`;
    return null;
  }
  function draw() {
    rafId = 0;
    const W = canvas.clientWidth, H = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#08090c';
    ctx.fillRect(0, 0, W, H);
    if (!cur.adj || !cur.entry || !task()) return;
    const e = cur.entry;
    ctx.imageSmoothingEnabled = view.s < 2.5;
    ctx.drawImage(cur.adj, view.tx, view.ty, e.width * view.s, e.height * view.s);
    ctx.strokeStyle = 'rgba(255,255,255,.22)';
    ctx.lineWidth = 1;
    ctx.strokeRect(view.tx - 0.5, view.ty - 0.5, e.width * view.s + 1, e.height * view.s + 1);
    const te = taskEntry();
    const [vx1, vy1, vx2, vy2] = te.viewport;
    if (state.settings.scale) {
      const step = te.px_per_cm;
      ctx.save();
      ctx.strokeStyle = 'rgba(250,250,250,.28)';
      ctx.setLineDash([2, 4]);
      for (let k = 0, y = vy1; y <= e.height + 0.5; k++, y += step) {
        const s0 = toScr([vx1, y]), s1 = toScr([vx2, y]);
        ctx.beginPath(); ctx.moveTo(s0.x, s0.y); ctx.lineTo(s1.x, s1.y); ctx.stroke();
        text(ctx, `${k} cm`, s0.x + 3, s0.y - 3, 'rgba(255,255,255,.75)');
      }
      for (let x = vx1; x <= vx2 + 0.5; x += step) {
        const s0 = toScr([x, vy1]), s1 = toScr([x, vy2]);
        ctx.beginPath(); ctx.moveTo(s0.x, s0.y); ctx.lineTo(s1.x, s1.y); ctx.stroke();
      }
      ctx.restore();
    }
    if (state.settings.guides) {
      ctx.save();
      ctx.strokeStyle = 'rgba(34,211,238,.38)';
      ctx.setLineDash([6, 6]);
      for (const gx of guideXs(te)) {
        const s0 = toScr([gx, vy1]), s1 = toScr([gx, vy2]);
        ctx.beginPath(); ctx.moveTo(s0.x, s0.y); ctx.lineTo(s1.x, s1.y); ctx.stroke();
      }
      ctx.restore();
    }
    // MT: show the locked vertical line under the cursor before the first click
    if (cur.tool === 'mt' && !cur.pending.length && mouse.inside && !mouse.down) {
      const x = snapMTx(toImg(mouse)[0]);
      const s0 = toScr([x, 0]), s1 = toScr([x, e.height]);
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,.18)';
      ctx.beginPath(); ctx.moveTo(s0.x, s0.y); ctx.lineTo(s1.x, s1.y); ctx.stroke();
      ctx.restore();
    }
    drawOverlays(ctx, view, { w: 1.5, labels: state.settings.labels, dots: true, hover: true });
    const live = liveValue();
    if (live && mouse.inside) text(ctx, live, mouse.x + 14, mouse.y + 18, '#fff');
    drawLoupe(W);
  }
  function drawLoupe(W) {
    const show = state.settings.loupe && mouse.inside && cur.adj && !mouse.down;
    loupe.classList.toggle('hidden', !show);
    if (!show) return;
    loupe.classList.toggle('left', mouse.x > W - 230 && mouse.y < 230);
    const size = loupe.width;
    const ls = Math.max(3, view.s * 4); // screen px per image px inside the loupe
    const q = toImg(mouse);
    const T = { s: ls, tx: size / 2 - q[0] * ls, ty: size / 2 - q[1] * ls };
    lctx.setTransform(1, 0, 0, 1, 0, 0);
    lctx.fillStyle = '#000';
    lctx.fillRect(0, 0, size, size);
    lctx.imageSmoothingEnabled = false;
    lctx.drawImage(cur.adj, T.tx, T.ty, cur.entry.width * ls, cur.entry.height * ls);
    drawOverlays(lctx, T, { w: 1, labels: false, dots: false, hover: false });
    lctx.strokeStyle = 'rgba(255,80,80,.9)';
    lctx.lineWidth = 1;
    lctx.beginPath();
    lctx.moveTo(size / 2, size / 2 - 14); lctx.lineTo(size / 2, size / 2 - 3);
    lctx.moveTo(size / 2, size / 2 + 3); lctx.lineTo(size / 2, size / 2 + 14);
    lctx.moveTo(size / 2 - 14, size / 2); lctx.lineTo(size / 2 - 3, size / 2);
    lctx.moveTo(size / 2 + 3, size / 2); lctx.lineTo(size / 2 + 14, size / 2);
    lctx.stroke();
  }

  // ---------- panel ----------
  function instruction() {
    const t = task();
    const note = t.note ? `<span class="inote">${esc(t.note)}</span><br>` : '';
    return note + instructionCore(t);
  }
  function instructionCore(t) {
    const lab = getLabel(t.task_id, false);
    const need = needOf(t);
    const n = (tool) => (lab ? lab[tool].length : 0);
    const k = n(cur.tool) + 1;
    const p = cur.pending.length;
    const optional = (need[cur.tool] ?? NEED) === 0;
    if (lab && summarize(lab, mmPerPx(t), need).complete && !p && n(cur.tool) >= NEED) {
      const mtOptional = (need.mt ?? NEED) === 0 && n('mt') < NEED;
      return (mtOptional ? 'FL and PA are done (MT is optional here: <kbd>M</kbd> if you want to add it). ' : 'All measurements done. ')
        + 'Check the review panel (right), adjust points by dragging if needed, then press <kbd>N</kbd>.';
    }
    if (n(cur.tool) >= NEED && !p) return `<b class="${cur.tool}">${NAMES[cur.tool]}</b> has 3. Switch tool: <kbd>M</kbd> MT, <kbd>F</kbd> FL, <kbd>P</kbd> PA.`;
    if (cur.tool === 'mt') {
      const where = ['left', 'middle', 'right'][Math.min(k, 3) - 1];
      const opt = optional ? ' <i>(optional on this image)</i>' : '';
      return p === 0
        ? `<b class="mt">MT ${k}/3</b>${opt} click the <b>inner</b> (lower) edge of the <b>superficial</b> aponeurosis near the ${where} guide.`
        : `<b class="mt">MT ${k}/3</b>${opt} click the <b>inner</b> (upper) edge of the <b>first deep</b> aponeurosis (x is locked). <kbd>Esc</kbd> cancels.`;
    }
    if (cur.tool === 'fl') {
      return p === 0
        ? `<b class="fl">FL ${k}/3</b> trace a fascicle from the deep to the superficial aponeurosis: click points along it, outside the image is fine. Prefer fascicles near the centre (least extrapolation).`
        : `<b class="fl">FL ${k}/3</b> ${p} point${p > 1 ? 's' : ''}. Keep clicking along the fascicle; <b>double-click</b> or <kbd>Enter</kbd> to finish, <kbd>Backspace</kbd> removes the last point.`;
    }
    return p < 2
      ? `<b class="pa">PA ${k}/3</b> click 2 points along a fascicle close to its <b>insertion</b> into the deep aponeurosis (${p}/2). Use fascicles other than the FL ones.`
      : `<b class="pa">PA ${k}/3</b> now click 2 points along the <b>deep aponeurosis</b> at that insertion (${p - 2}/2).`;
  }
  function renderTaskInfo() {
    const t = task();
    const e = taskEntry(t);
    const shown = cur.entry || e;
    const inPhase = tasks.filter((x) => x.phase === t.phase);
    const parts = [`<span class="tag phase phase-${t.phase}">${PHASE_NAMES[t.phase]} ${inPhase.indexOf(t) + 1}/${inPhase.length}</span>`,
      `<span class="id">${esc(t.task_id)}</span>`];
    if (!isCalib(t)) parts.push(`<span class="tag">${esc(e.group)}</span>`);
    if (t.kind === 'dynamic') parts.push('<span class="tag warn" title="calf-raise video frame">dynamic: FL + PA (MT optional)</span>');
    parts.push(`<span class="tag${e.scale_trusted ? '' : ' warn'}" title="vertical scale, used for both axes">${fmt(e.px_per_cm, 2)} px/cm${e.scale_trusted ? '' : ' (heuristic!)'}</span>`,
      `<span class="tag">${e.width}&times;${e.height}</span>`);
    const frames = viewFrames(t);
    if (t.clip !== null && t.clip !== undefined) parts.push(`<span class="tag">clip ${t.clip}: ${t.members.length} frames${shown.image_id !== t.task_id ? `, viewing ${esc(shown.image_id)}` : ''}</span>`);
    else if (frames.length > 1) parts.push(`<span class="tag">${frames.length} video frames${shown.image_id !== t.task_id ? `, viewing frame ${frames.indexOf(shown.image_id) + 1}` : ''}</span>`);
    if (state.settings.calibSkipped && !isCalib(t) && calibPending()) parts.push('<span class="tag warn">calibration skipped</span>');
    $('task-info').innerHTML = parts.join('');
  }
  function renderPanel() {
    const t = task();
    const id = t.task_id;
    const lab = getLabel(id, false);
    const mmpp = mmPerPx(t);
    const need = needOf(t);
    const s = summarize(lab, mmpp, need);
    for (const tool of TOOLS) {
      const btn = document.querySelector(`.tool[data-tool="${tool}"]`);
      const optional = (need[tool] ?? NEED) === 0;
      btn.classList.toggle('active', cur.tool === tool);
      btn.classList.toggle('full', s['n_' + tool] >= NEED);
      btn.classList.toggle('optional', optional);
      $('cnt-' + tool).textContent = optional ? `${s['n_' + tool]}/3 opt.` : `${s['n_' + tool]}/3`;
      $('mean-' + tool).textContent = fmt(s[tool]);
    }
    const warns = [];
    for (const tool of TOOLS) {
      const v = s[tool];
      if (Number.isFinite(v) && (v < RANGES[tool][0] || v > RANGES[tool][1])) warns.push(`${NAMES[tool]} ${fmt(v)} outside the stated test range ${RANGES[tool][0]}-${RANGES[tool][1]} ${UNITS[tool]}`);
    }
    if (lab) lab.mt.forEach((m, i) => { if (m[1][1] < m[0][1]) warns.push(`MT${i + 1}: deep point is above the superficial one (fine if intended)`); });
    $('warnings').innerHTML = warns.map(esc).join('<br>');
    const rows = [];
    for (const tool of ['pa', 'fl', 'mt']) {
      const color = tool === 'pa' ? COLORS.pa : COLORS[tool];
      for (let i = 0; i < NEED; i++) {
        const v = s['vals_' + tool][i];
        rows.push(v === undefined
          ? `<tr class="empty"><td><span class="sw" style="background:${color}"></span>${NAMES[tool]}${i + 1}</td><td class="num">–</td><td></td></tr>`
          : `<tr><td><span class="sw" style="background:${color}"></span>${NAMES[tool]}${i + 1}</td><td class="num">${fmt(v)} ${UNITS[tool]}</td><td class="num"><button class="del" data-tool="${tool}" data-i="${i}" title="delete">&times;</button></td></tr>`);
      }
    }
    $('meas-table').tBodies[0].innerHTML = rows.join('');
    renderReview(s, t);
    const noteEl = $('note');
    if (document.activeElement !== noteEl) noteEl.value = lab ? lab.note || '' : '';
    $('ck-flag').checked = !!(lab && lab.flag);
    $('instruction').innerHTML = instruction();
    renderFrames(t);
  }
  function renderReview(s, t) {
    const card = $('review');
    if (!s.complete) { card.classList.add('hidden'); return; }
    card.classList.remove('hidden');
    if (isCalib(t)) return renderCalibReview(s);
    $('review-title').textContent = 'Review (shown only after the image is finished)';
    const hint = taskEntry(t).hint;
    const rows = ['<tr><th></th><th>you</th><th>model</th><th>&Delta;</th></tr>'];
    for (const tool of ['pa', 'fl', 'mt']) {
      const mine = s[tool];
      const key = tool === 'pa' ? 'pa_deg' : tool + '_mm';
      const other = hint ? hint[key] : NaN;
      const d = mine - other;
      const big = Number.isFinite(d) && Math.abs(d) > REVIEW_DELTA[tool];
      rows.push(`<tr><td>${NAMES[tool]}</td><td class="num">${fmt(mine)}</td><td class="num">${fmt(other)}</td><td class="num${big ? ' flag' : ''}">${Number.isFinite(d) ? (d > 0 ? '+' : '') + fmt(d) : '–'}</td></tr>`);
    }
    $('review-table').tBodies[0].innerHTML = rows.join('');
    const geo = 1.065 * s.mt / Math.sin(s.pa * Math.PI / 180);
    const ratio = s.fl / geo;
    const notes = [`Geometry check: 1.065&middot;MT/sin(PA) = ${fmt(geo)} mm vs your FL ${fmt(s.fl)} mm${ratio < 0.75 || ratio > 1.33 ? ' <span style="color:var(--warn)">(inconsistent, re-check PA/FL)</span>' : ''}.`];
    const spread = (tool) => { const v = s['vals_' + tool]; return Math.max(...v) - Math.min(...v); };
    if (spread('pa') > 8) notes.push(`PA values spread ${fmt(spread('pa'))}&deg;.`);
    if (spread('mt') > 4) notes.push(`MT values spread ${fmt(spread('mt'))} mm (fine if the muscle tapers).`);
    if (!hint) notes.push('No model hints in the manifest.');
    else notes.push(`Orange &Delta;: beyond ${REVIEW_DELTA.pa}&deg; / ${REVIEW_DELTA.fl} mm / ${REVIEW_DELTA.mt} mm: re-check which aponeurosis you used. The model can be wrong.`);
    $('review-notes').innerHTML = notes.join('<br>');
  }
  function renderCalibReview(s) {
    // Calibration: no model values and no expert answers, only your own values and the same
    // self-consistency check as on the test images.
    $('review-title').textContent = 'Review (calibration image)';
    const rows = ['<tr><th></th><th>you</th><th>spread of your 3</th></tr>'];
    for (const tool of ['pa', 'fl', 'mt']) {
      const v = s['vals_' + tool];
      const spread = v.length > 1 ? Math.max(...v) - Math.min(...v) : NaN;
      rows.push(`<tr><td>${NAMES[tool]}</td><td class="num">${fmt(s[tool])}</td><td class="num">${fmt(spread)}</td></tr>`);
    }
    $('review-table').tBodies[0].innerHTML = rows.join('');
    const notes = [];
    if (s.n_mt >= NEED) {
      const geo = 1.065 * s.mt / Math.sin(s.pa * Math.PI / 180);
      const ratio = s.fl / geo;
      notes.push(`Geometry check: 1.065&middot;MT/sin(PA) = ${fmt(geo)} mm vs your FL ${fmt(s.fl)} mm${ratio < 0.75 || ratio > 1.33 ? ' <span style="color:var(--warn)">(inconsistent, re-check PA/FL)</span>' : ''}.`);
    }
    notes.push('Calibration image: no model values are shown, and the expert answers stay hidden. Measure it exactly as you would a test image, then press <kbd>N</kbd>.');
    $('review-notes').innerHTML = notes.join('<br>');
  }
  function renderFrames(t) {
    const el = $('frames');
    const frames = viewFrames(t);
    if (frames.length <= 1) { el.classList.add('hidden'); return; }
    el.classList.remove('hidden');
    el.innerHTML = 'Frame: ' + frames.map((m, i) => `<button data-frame="${i}" class="${i === cur.frame ? 'on' : ''}" title="${esc(m)}">${i + 1}</button>`).join('') + ' <span class="small">(<kbd>,</kbd> <kbd>.</kbd>; label is on frame 1)</span>';
  }
  function renderProgress() {
    const counts = {};
    for (const ph of Object.keys(PHASE_NAMES)) counts[ph] = { done: 0, partial: 0, flag: 0, none: 0, n: 0 };
    const cells = document.querySelectorAll('#grids .cell');
    tasks.forEach((t, i) => {
      const st = status(t.task_id);
      counts[t.phase][st]++;
      counts[t.phase].n++;
      const cell = cells[i];
      if (cell) cell.className = `cell ${st}${t.clip !== null && t.clip !== undefined ? ' clip' : ''}${isCalib(t) ? ' calib' : ''}${i === cur.idx ? ' cur' : ''}`;
    });
    const c = counts.calib, test = { done: counts.clips.done + counts.test.done, flag: counts.clips.flag + counts.test.flag, n: counts.clips.n + counts.test.n };
    const parts = [];
    if (c.n) parts.push(`calib <b>${c.done + c.flag}</b>/${c.n}` + (state.settings.calibSkipped && c.done + c.flag < c.n ? ' (skipped)' : ''));
    parts.push(`clips <b>${counts.clips.done}</b>/${counts.clips.n}`);
    parts.push(`test <b>${test.done}</b>/${test.n}` + (test.flag ? ` (${test.flag} flagged)` : ''));
    $('progress').innerHTML = parts.join(' &middot; ');
    const doneTimes = tasks.filter((t) => status(t.task_id) === 'done').map((t) => state.times[t.task_id] || 0).filter((ms) => ms > 0);
    const totalMs = Object.values(state.times).reduce((a, b) => a + b, 0);
    const avg = doneTimes.length ? mean(doneTimes) : NaN;
    const remaining = tasks.filter((t) => !handled(t.task_id)).length;
    const mins = (ms) => `${Math.floor(ms / 60000)}m${String(Math.round(ms / 1000) % 60).padStart(2, '0')}s`;
    $('stats').innerHTML = `This image ${mins(state.times[task().task_id] || 0)} &middot; total ${mins(totalMs)}`
      + (Number.isFinite(avg) ? ` &middot; avg ${Math.round(avg / 1000)} s/image &middot; ETA (all remaining) ${Math.round(avg * remaining / 60000)} min` : '')
      + `<br>Grid: grey todo, amber partial, green done, red flagged; outlined = clip (label copied to 5 frames); violet outline = calibration.`;
  }
  function syncControls() {
    const st = state.settings;
    $('sl-bright').value = st.bright;
    $('sl-contrast').value = st.contrast;
    $('sl-gamma').value = st.gamma;
    $('ck-autoeach').checked = st.autoEach;
    $('ck-guides').checked = st.guides;
    $('ck-loupe').checked = st.loupe;
    $('ck-scale').checked = st.scale;
    $('ck-labels').checked = st.labels;
    $('ck-trackpad').checked = st.trackpad;
  }
  function refresh() {
    if (!task()) return;
    renderTaskInfo();
    renderPanel();
    renderProgress();
    requestDraw();
  }
  function toast(msg, kind = '') {
    const el = $('toast');
    el.textContent = msg;
    el.style.borderColor = kind === 'ok' ? 'var(--ok)' : 'var(--warn)';
    el.classList.remove('hidden');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add('hidden'), kind === 'ok' ? 3500 : 2600);
  }

  // ---------- navigation ----------
  function firstUnfinishedCalib() {
    const list = calibTasks();
    const start = isCalib() ? list.indexOf(task()) : -1;
    for (let k = 1; k <= list.length; k++) {
      const t = list[(start + k + list.length) % list.length];
      if (!handled(t.task_id)) return taskIndexById[t.task_id];
    }
    return -1;
  }
  function showGate(idx) {
    // Test tasks wait until the calibration is finished (done or flagged) or explicitly skipped.
    pendingGateIdx = idx;
    const list = calibTasks();
    const left = list.filter((t) => !handled(t.task_id)).length;
    $('gate-text').innerHTML = `<b>${left}</b> of ${list.length} calibration images are not finished yet. `
      + 'They have known expert answers (hidden from you). Measuring them first tells us how accurate your '
      + 'measurements are, per parameter, so your Telemed labels can be weighted correctly.';
    $('gate').classList.remove('hidden');
    if (cur.idx < 0) { const j = firstUnfinishedCalib(); if (j >= 0) openTask(j); }
  }
  function closeGate() { $('gate').classList.add('hidden'); pendingGateIdx = -1; }
  function continueCalibration() {
    // stay on the current calibration image if it is unfinished, otherwise go to the next unfinished one
    closeGate();
    if (!(isCalib() && !handled(task().task_id))) nextUnfinished();
    canvas.focus();
  }
  function skipCalibration() {
    if (!window.confirm('Skip the calibration and start the test images?\n\nWithout it we cannot measure your error and bias, so your labels will get a low, generic weight. You can come back to the calibration images at any time from the progress grid.')) return;
    const idx = pendingGateIdx;
    state.settings.calibSkipped = true;
    touch(null);
    closeGate();
    openTask(idx >= 0 ? idx : cur.idx + 1);
  }
  function openTask(idx, opts = {}) {
    if (!tasks.length) return;
    idx = clamp(idx, 0, tasks.length - 1);
    if (gateBlocks(idx)) return showGate(idx);
    if (cur.pending.length && idx !== cur.idx) toast('unfinished measurement discarded');
    cur.pending = [];
    if (idx !== cur.idx) { cur.adj = null; cur.entry = null; cur.src = null; }
    cur.idx = idx;
    cur.frame = 0;
    const t = task();
    cur.tool = firstIncompleteTool(getLabel(t.task_id, false), needOf(t));
    state.settings.lastTask = t.task_id;
    saveLocal();
    mouse.hover = null;
    showImage(taskEntry(t), false);
    // preload neighbours
    for (const j of [idx + 1, idx + 2, idx - 1]) if (tasks[j]) loadImageEl(taskEntry(tasks[j])).catch(() => {});
    refresh();
    if (t.note) toast(t.note);
    if (!opts.keepFocus) canvas.focus();
  }
  function nextUnfinished() {
    if (calibPending() && !state.settings.calibSkipped) {
      const j = firstUnfinishedCalib();
      if (j >= 0) return openTask(j);
    }
    for (let k = 1; k <= tasks.length; k++) {
      const j = (cur.idx + k) % tasks.length;
      if (!handled(tasks[j].task_id)) return openTask(j);
    }
    toast('every image is done or flagged', 'ok');
  }
  function showFrame(i) {
    const t = task();
    const frames = viewFrames(t);
    if (frames.length <= 1) return;
    cur.frame = (i + frames.length) % frames.length;
    showImage(imagesById[frames[cur.frame]], true).then(refresh);
  }
  function jump(text) {
    const s = text.trim();
    if (!s) return;
    let idx = tasks.findIndex((t) => t.task_id.toLowerCase().startsWith(s.toLowerCase()));
    if (idx < 0 && /^\d+$/.test(s)) {
      const n = parseInt(s, 10);
      const entry = manifest.images.find((e) => e.number === n);
      if (entry) idx = taskIndexById[entry.task_id];
    }
    if (idx >= 0 && idx !== undefined) openTask(idx);
    else toast(`no image matches "${s}"`);
  }

  // ---------- events ----------
  function canvasPoint(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top, inside: e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom };
  }
  canvas.addEventListener('mousedown', (e) => {
    lastInteraction = Date.now();
    canvas.focus();
    const p = canvasPoint(e);
    Object.assign(mouse, { x: p.x, y: p.y, inside: true });
    if (!cur.adj) return; // image still loading
    if (e.button === 0 && !spaceHeld && !cur.pending.length) {
      const hit = hitVertex(p);
      if (hit) {
        mouse.drag = { hit, start: p, moved: false, before: snap(task().task_id) };
        return;
      }
    }
    mouse.down = { x: p.x, y: p.y, button: e.button, tx: view.tx, ty: view.ty, moved: false, pan: e.button !== 0 || spaceHeld };
    if (mouse.down.pan) canvas.classList.add('panning');
    e.preventDefault();
  });
  window.addEventListener('mousemove', (e) => {
    const p = canvasPoint(e);
    mouse.x = p.x; mouse.y = p.y; mouse.inside = p.inside;
    lastInteraction = Date.now();
    const ins = $('instruction');
    ins.classList.toggle('faded', p.inside && p.y > canvas.clientHeight - ins.offsetHeight - 30 && p.x < ins.offsetWidth + 30);
    if (mouse.drag) {
      if (!mouse.drag.moved && Math.hypot(p.x - mouse.drag.start.x, p.y - mouse.drag.start.y) < 3) return;
      mouse.drag.moved = true;
      moveVertex(mouse.drag.hit, toImg(p));
      renderPanel();
      requestDraw();
      return;
    }
    if (mouse.down) {
      const dx = p.x - mouse.down.x, dy = p.y - mouse.down.y;
      if (!mouse.down.moved && Math.hypot(dx, dy) > DRAG_PX) mouse.down.moved = true;
      if (mouse.down.moved) {
        view.userMoved = true;
        canvas.classList.add('panning');
        view.tx = mouse.down.tx + dx;
        view.ty = mouse.down.ty + dy;
      }
    } else if (p.inside) {
      const hv = cur.pending.length ? null : hitVertex(p);
      if (JSON.stringify(hv) !== JSON.stringify(mouse.hover)) mouse.hover = hv;
      canvas.classList.toggle('hovering', !!hv);
    }
    requestDraw();
  });
  window.addEventListener('mouseup', (e) => {
    const p = canvasPoint(e);
    if (mouse.drag) {
      const d = mouse.drag;
      mouse.drag = null;
      if (d.moved) {
        const id = task().task_id;
        (undoStacks[id] = undoStacks[id] || []).push(d.before);
        redoStacks[id] = [];
        touch(getLabel(id, true));
        refresh();
      } else handleClick(p);
      return;
    }
    if (mouse.down) {
      const d = mouse.down;
      mouse.down = null;
      canvas.classList.remove('panning');
      if (!d.moved && d.button === 0 && !d.pan) handleClick(p);
      requestDraw();
    }
  });
  canvas.addEventListener('dblclick', (e) => { e.preventDefault(); if (cur.tool === 'fl') finishFL(); });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('mouseleave', () => { mouse.inside = false; requestDraw(); });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    lastInteraction = Date.now();
    const p = canvasPoint(e);
    Object.assign(mouse, { x: p.x, y: p.y });
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
    if (state.settings.trackpad && !e.ctrlKey) {
      view.tx -= e.deltaX * unit;
      view.ty -= e.deltaY * unit;
      view.userMoved = true;
      requestDraw();
      return;
    }
    const speed = e.ctrlKey ? 0.01 : 0.0015;
    zoomAt(p.x, p.y, Math.exp(-e.deltaY * unit * speed));
    if (!e.ctrlKey && e.deltaX) { view.tx -= e.deltaX * unit; requestDraw(); }
  }, { passive: false });

  window.addEventListener('keydown', (e) => {
    lastInteraction = Date.now();
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' && e.target.type === 'range' && e.key.startsWith('Arrow')) return;
    if ((tag === 'INPUT' && e.target.type === 'text') || tag === 'TEXTAREA') {
      if (e.key === 'Escape') e.target.blur();
      if (e.key === 'Enter' && e.target.id === 'jump') { jump(e.target.value); e.target.value = ''; e.target.blur(); }
      return;
    }
    const help = $('help');
    if (!help.classList.contains('hidden')) {
      if (['Escape', '?', '/', 'h', 'H'].includes(e.key)) help.classList.add('hidden');
      return;
    }
    if (!$('gate').classList.contains('hidden')) {
      if (e.key === 'Escape') closeGate();
      if (e.key === 'Enter') continueCalibration();
      return;
    }
    const k = e.key;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && (k === 'z' || k === 'Z')) { e.preventDefault(); return e.shiftKey ? redo() : undo(); }
    if (mod && (k === 'y' || k === 'Y')) { e.preventDefault(); return redo(); }
    if (mod) return; // leave browser shortcuts alone
    switch (k) {
      case 'm': case 'M': case '1': setTool('mt'); break;
      case 'f': case 'F': case '2': setTool('fl'); break;
      case 'p': case 'P': case '3': setTool('pa'); break;
      case 'Enter': finishFL(); break;
      case 'Escape': cancelPending(); break;
      case 'Backspace': case 'Delete': e.preventDefault(); undo(); break;
      case 'n': openTask(cur.idx + 1); break;
      case 'N': nextUnfinished(); break;
      case 'ArrowRight': e.preventDefault(); openTask(cur.idx + 1); break;
      case 'b': case 'B': openTask(cur.idx - 1); break;
      case 'ArrowLeft': e.preventDefault(); openTask(cur.idx - 1); break;
      case '0': case 'z': case 'Z': fit(); break;
      case '+': case '=': zoomAt(canvas.clientWidth / 2, canvas.clientHeight / 2, 1.25); break;
      case '-': case '_': zoomAt(canvas.clientWidth / 2, canvas.clientHeight / 2, 0.8); break;
      case 'a': case 'A': autoContrast(true); break;
      case 'g': case 'G': toggleSetting('guides'); break;
      case 'l': case 'L': toggleSetting('loupe'); break;
      case 'k': case 'K': toggleSetting('scale'); break;
      case 'v': case 'V': toggleSetting('labels'); break;
      case ',': case '<': showFrame(cur.frame - 1); break;
      case '.': case '>': showFrame(cur.frame + 1); break;
      case 'x': case 'X': setFlag(!(getLabel(task().task_id, false) || {}).flag); break;
      case '?': case '/': case 'h': case 'H': help.classList.remove('hidden'); break;
      case ' ': spaceHeld = true; e.preventDefault(); break;
      default: return;
    }
  });
  window.addEventListener('keyup', (e) => { if (e.key === ' ') spaceHeld = false; });
  window.addEventListener('blur', () => { spaceHeld = false; });

  function toggleSetting(name) {
    state.settings[name] = !state.settings[name];
    syncControls();
    saveLocal();
    requestDraw();
  }
  function setFlag(on) {
    const lab = getLabel(task().task_id, true);
    lab.flag = !!on;
    touch(lab);
    refresh();
    if (on) toast('image flagged (counts as handled; still exported with whatever was measured)');
  }

  function bindUI() {
    $('btn-prev').onclick = () => openTask(cur.idx - 1);
    $('btn-next').onclick = () => openTask(cur.idx + 1);
    $('btn-next-todo').onclick = () => nextUnfinished();
    $('btn-help').onclick = () => $('help').classList.remove('hidden');
    $('btn-help-close').onclick = () => $('help').classList.add('hidden');
    $('help').addEventListener('click', (e) => { if (e.target.id === 'help') $('help').classList.add('hidden'); });
    document.querySelectorAll('.tool').forEach((b) => { b.onclick = () => { setTool(b.dataset.tool); canvas.focus(); }; });
    $('meas-table').addEventListener('click', (e) => {
      const b = e.target.closest('button.del');
      if (b) { deleteMeasurement(b.dataset.tool, parseInt(b.dataset.i, 10)); canvas.focus(); }
    });
    $('frames').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-frame]');
      if (b) { showFrame(parseInt(b.dataset.frame, 10)); canvas.focus(); }
    });
    const slider = (id, key) => $(id).addEventListener('input', (e) => { state.settings[key] = parseFloat(e.target.value); applyAdjust(); saveLocal(); });
    slider('sl-bright', 'bright');
    slider('sl-contrast', 'contrast');
    slider('sl-gamma', 'gamma');
    $('btn-auto').onclick = () => { autoContrast(true); canvas.focus(); };
    $('btn-reset-adj').onclick = () => { Object.assign(state.settings, { bright: 0, contrast: 1, gamma: 1 }); syncControls(); applyAdjust(); saveLocal(); canvas.focus(); };
    const check = (id, key, after) => $(id).addEventListener('change', (e) => { state.settings[key] = e.target.checked; saveLocal(); if (after) after(); requestDraw(); });
    check('ck-autoeach', 'autoEach', () => { if (state.settings.autoEach) autoContrast(true); });
    check('ck-guides', 'guides');
    check('ck-loupe', 'loupe');
    check('ck-scale', 'scale');
    check('ck-labels', 'labels');
    check('ck-trackpad', 'trackpad');
    $('ck-flag').addEventListener('change', (e) => setFlag(e.target.checked));
    $('note').addEventListener('input', (e) => { const lab = getLabel(task().task_id, true); lab.note = e.target.value; touch(lab); renderProgress(); });
    $('btn-export').onclick = () => { exportCSV('test'); saveServer(); };
    $('btn-export-calib').onclick = () => { exportCSV('calib'); saveServer(); };
    $('btn-export-calib').classList.toggle('hidden', !calibManifest);
    $('btn-gate-back').onclick = () => continueCalibration();
    $('btn-gate-skip').onclick = () => skipCalibration();
    $('btn-import').onclick = () => $('file-import').click();
    $('file-import').addEventListener('change', async (e) => {
      const f = e.target.files[0];
      if (f) importCSV(await f.text());
      e.target.value = '';
    });
    // Return keyboard focus to the canvas after any button / checkbox click (Space, arrows).
    document.addEventListener('click', (e) => {
      if (e.target.closest('button, input[type=checkbox]') && !e.target.closest('#help')) setTimeout(() => canvas.focus(), 0);
    });
    const grids = $('grids');
    const cell = (t, i) => `<div class="cell" data-i="${i}" title="${esc(t.task_id)}${t.clip !== null && t.clip !== undefined ? ` (clip ${t.clip})` : ''}${t.kind === 'dynamic' ? ' (dynamic)' : ''}"></div>`;
    grids.innerHTML = Object.keys(PHASE_NAMES).map((ph) => {
      const items = tasks.map((t, i) => [t, i]).filter(([t]) => t.phase === ph);
      if (!items.length) return '';
      return `<div class="grid-title small">${PHASE_NAMES[ph]} (${items.length})</div><div class="grid">${items.map(([t, i]) => cell(t, i)).join('')}</div>`;
    }).join('');
    grids.addEventListener('click', (e) => { const c = e.target.closest('.cell'); if (c) openTask(parseInt(c.dataset.i, 10)); });
    new ResizeObserver(() => { if (!view.userMoved) fit(); requestDraw(); }).observe($('stage'));
    window.addEventListener('beforeunload', () => saveLocal());
    // per-image active timer
    setInterval(() => {
      if (!task() || document.visibilityState !== 'visible' || Date.now() - lastInteraction > IDLE_MS) return;
      const id = task().task_id;
      state.times[id] = (state.times[id] || 0) + 1000;
      renderProgress();
    }, 1000);
    setInterval(() => { saveLocal(); if (serverOK !== false) scheduleServerSave(0); }, 30000);
  }

  // ---------- init ----------
  async function init() {
    try {
      const res = await fetch('data/manifest.json', { cache: 'no-store' });
      if (!res.ok) throw new Error(res.status);
      manifest = await res.json();
    } catch (err) {
      document.body.innerHTML = '<p style="padding:20px">Cannot load data/manifest.json. Run <code>tools/annotator/prepare.py</code>, then open this page through <code>tools/annotator/serve.py</code> (http://127.0.0.1:8765/).</p>';
      return;
    }
    try {
      const res = await fetch('data/calib_manifest.json', { cache: 'no-store' });
      calibManifest = res.ok ? await res.json() : null;
    } catch (err) { calibManifest = null; }
    if (calibManifest && !(calibManifest.tasks || []).length) calibManifest = null;
    imagesById = Object.fromEntries([...manifest.images, ...(calibManifest ? calibManifest.images : [])].map((e) => [e.image_id, e]));
    // Queue: calibration first, then the Telemed clip tasks, then every other test task.
    const calib = calibManifest ? calibManifest.tasks.map((t) => Object.assign(t, { phase: 'calib' })) : [];
    const clips = manifest.tasks.filter((t) => t.clip !== null && t.clip !== undefined).map((t) => Object.assign(t, { phase: 'clips' }));
    const rest = manifest.tasks.filter((t) => t.clip === null || t.clip === undefined).map((t) => Object.assign(t, { phase: 'test' }));
    tasks = [...calib, ...clips, ...rest];
    taskIndexById = Object.fromEntries(tasks.map((t, i) => [t.task_id, i]));
    let local = null;
    try { local = JSON.parse(localStorage.getItem(LS_KEY) || 'null'); } catch (err) { local = null; }
    let server = null;
    try {
      const res = await fetch('api/state', { headers: { 'X-Annotator': '1' }, cache: 'no-store' });
      if (res.ok) { server = await res.json(); serverOK = true; }
    } catch (err) { serverOK = false; }
    const empty = { version: 1, labels: {}, times: {}, settings: { ...DEFAULT_SETTINGS }, updated: 0 };
    state = mergeStates(local && local.labels ? local : empty, server && server.labels ? server : empty);
    for (const lab of Object.values(state.labels)) for (const t of TOOLS) lab[t] = lab[t] || [];
    bindUI();
    syncControls();
    const n = Object.keys(state.labels).length;
    setSaveStatus(n ? `restored ${n} labelled images` : (serverOK ? 'ready (disk autosave on)' : 'ready (browser autosave only)'), serverOK === false);
    const last = state.settings.lastTask;
    openTask(last && last in taskIndexById ? taskIndexById[last] : 0);
    if (!n) $('help').classList.remove('hidden');
  }
  // read-only hooks for debugging / automated checks
  window.__umud = { state: () => state, csv: () => buildCSV('test'), csvCalib: () => buildCSV('calib'), task: () => task(), view: () => ({ ...view }), importText: (t) => importCSV(t) };
  init();
})();
