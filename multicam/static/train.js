'use strict';
// Live dashboard of a training run. One poll a second of /api/train/live (only what is new),
// canvas charts that glide to their new view, numbers that count up, step/ETA interpolated between log rows.
const $ = (s, r = document) => r.querySelector(s);
const el = (h) => { const t = document.createElement('template'); t.innerHTML = h.trim(); return t.content.firstChild; };
const C = { acc: '#5aa9ff', ok: '#3ecf8e', bad: '#ff6470', warn: '#f5b84b', vio: '#a58bff', cy: '#4fd6d6', pink: '#ff7ac6', dim: '#8a94a4' };
const PAL = [C.acc, C.ok, C.warn, C.vio, C.cy, C.pink, C.bad];
const pct = (v, d = 1) => v == null ? '—' : (100 * v).toFixed(d) + ' %';
const f3 = v => v == null ? '—' : (+v).toFixed(3);
const mmss = s => { s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); return h ? h + ' ч ' + String(m).padStart(2, '0') + ' м' : m + ' м ' + String(s % 60).padStart(2, '0') + ' с'; };
const hhmm = d => String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');

const S = { run: new URLSearchParams(location.search).get('run') || '', rows: [], epochs: [], hw: [], restarts: [], n: 0, en: 0, hwT: 0,
  status: {}, fetchedAt: performance.now(), smooth: 0.85, range: 0, hidden: new Set(), ok: false, newEpoch: null, ver: 0 };

// ---- data ------------------------------------------------------------------------------------------------------
function addRows(rs) {
  for (const r of rs) {
    const last = S.rows[S.rows.length - 1];
    if (last && r.step <= last.step) {                 // a restart from an older checkpoint: the overwritten rows go
      S.restarts.push(r.step);
      while (S.rows.length && S.rows[S.rows.length - 1].step >= r.step) S.rows.pop();
    }
    S.rows.push(r);
  }
}
function addEpochs(es) {
  for (const e of es) {
    while (S.epochs.length && S.epochs[S.epochs.length - 1].step >= e.step) S.epochs.pop();
    S.epochs.push(e); S.newEpoch = e.step;
  }
}
function ema(v, a) {
  if (!a) return v.slice();
  const o = new Array(v.length); let m = null;
  for (let i = 0; i < v.length; i++) { const x = v[i]; if (x == null || !isFinite(x)) { o[i] = m; continue; } m = m == null ? x : a * m + (1 - a) * x; o[i] = m; }
  return o;
}
const quant = (a, q) => { const s = a.filter(x => x != null && isFinite(x)).sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))] : 0; };

// ---- charts ----------------------------------------------------------------------------------------------------
const CHARTS = [];
class Chart {
  // pts(): [{x, ...values}], series: [{k,label,color}], opt: {fmt, xfmt, smooth, area, yfix:[lo,hi], marks, live, refs}
  constructor(host, pts, series, opt = {}) {
    this.host = host; this.pts = pts; this.series = series; this.o = opt;
    this.cv = document.createElement('canvas'); host.appendChild(this.cv); this.ctx = this.cv.getContext('2d');
    this.v = null; this.dirty = true; this.hx = null;
    new ResizeObserver(() => { this.resize(); }).observe(host); this.resize(); this.onscreen = true;
    new IntersectionObserver(es => { this.onscreen = es[0].isIntersecting; this.dirty = true; }).observe(host);
    this.cv.addEventListener('mousemove', e => { const r = this.cv.getBoundingClientRect(); this.hx = e.clientX - r.left; this.hy = e.clientY; this.hxc = e.clientX; this.dirty = true; });
    this.cv.addEventListener('mouseleave', () => { this.hx = null; $('#tip').style.display = 'none'; this.dirty = true; });
    CHARTS.push(this);
  }
  resize() { const r = this.host.getBoundingClientRect(), d = devicePixelRatio || 1; this.W = r.width; this.H = r.height;
    this.cv.width = Math.round(r.width * d); this.cv.height = Math.round(r.height * d); this.ctx.setTransform(d, 0, 0, d, 0, 0); this.dirty = true; }
  target() {
    const P = this.pts(); if (!P.length) return null;
    const live = this.series.filter(s => !S.hidden.has(this.host.id + s.k));
    const xmax = P[P.length - 1].x; let xmin = P[0].x;
    const win = this.o.window != null ? this.o.window : S.range;
    if (win) xmin = Math.max(xmin, xmax - win);
    const vis = P.filter(p => p.x >= xmin - 1e-9);
    const a = this.o.smooth === false ? 0 : S.smooth;
    const cols = {}, raws = {}; let all = [];
    for (const s of live) {
      const raw = vis.map(p => p[s.k]); raws[s.k] = raw;
      cols[s.k] = a ? ema(P.map(p => p[s.k]), a).slice(P.length - vis.length) : raw;
      all = all.concat(cols[s.k].filter(x => x != null), a ? raw.filter(x => x != null && isFinite(x)) : []);
    }
    let lo, hi;
    if (this.o.yfix) [lo, hi] = this.o.yfix;
    else { const pos = all.every(x => x >= 0); lo = quant(all, a ? .02 : 0); hi = quant(all, a ? .98 : 1);
      if (!a) { /* raw values only: the whole range */ }
      if (this.o.zero) lo = Math.min(lo, 0);
      const pad = (hi - lo || Math.abs(hi) || 1) * .1; lo -= pad; hi += pad; if (pos && lo < 0) lo = 0; }
    return { xmin, xmax, lo, hi, vis, cols, raws, live, a };
  }
  frame() {
    if (!this.onscreen) return false;
    if (this._ver !== S.ver) { const T0 = this.target(); if (!T0) return false; this.T = T0; this._ver = S.ver; }
    const T = this.T; if (!T) return false;
    if (!this.v) { this.v = { xmin: T.xmin, xmax: T.xmax, lo: T.lo, hi: T.hi }; this.dirty = true; }
    const v = this.v, k = 0.16; let moving = false;
    for (const key of ['xmin', 'xmax', 'lo', 'hi']) { const d = T[key] - v[key]; if (Math.abs(d) > (T.hi - T.lo) * 1e-4 + 1e-9) { v[key] += d * k; moving = true; } else v[key] = T[key]; }
    if (moving || this.dirty) { this.draw(); this.dirty = false; return moving; }
    return false;
  }
  draw() {
    const { ctx: g, W, H, T, v } = this; const L = 46, R = 8, Tp = 8, B = 18;
    g.clearRect(0, 0, W, H);
    const X = x => L + (x - v.xmin) / Math.max(1e-9, v.xmax - v.xmin) * (W - L - R), Y = y => Tp + (1 - (y - v.lo) / Math.max(1e-12, v.hi - v.lo)) * (H - Tp - B);
    const fmt = this.o.fmt || (y => Math.abs(y) >= 100 ? y.toFixed(0) : Math.abs(y) >= 10 ? y.toFixed(1) : y.toFixed(3));
    g.font = '11px system-ui'; g.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {                       // horizontal grid + y labels
      const y = v.lo + (v.hi - v.lo) * i / 4, py = Y(y);
      g.strokeStyle = 'rgba(255,255,255,.06)'; g.lineWidth = 1; g.beginPath(); g.moveTo(L, py); g.lineTo(W - R, py); g.stroke();
      g.fillStyle = C.dim; g.textAlign = 'right'; g.fillText(fmt(y), L - 6, py);
    }
    g.textAlign = 'center'; g.textBaseline = 'alphabetic';
    for (let i = 0; i <= 4; i++) { const x = v.xmin + (v.xmax - v.xmin) * i / 4; g.fillStyle = C.dim; g.fillText((this.o.xfmt || (x => Math.round(x)))(x, v), L + (W - L - R) * i / 4, H - 4); }
    g.save(); g.beginPath(); g.rect(L, Tp - 2, W - L - R, H - Tp - B + 4); g.clip();
    for (const rf of this.o.refs || []) {                // dashed reference lines (v1, YOLO ...)
      if (rf.y < v.lo || rf.y > v.hi) continue; g.strokeStyle = rf.color; g.globalAlpha = .55; g.setLineDash([5, 5]); g.beginPath(); g.moveTo(L, Y(rf.y)); g.lineTo(W - R, Y(rf.y)); g.stroke();
      g.setLineDash([]); g.globalAlpha = .9; g.fillStyle = rf.color; g.textAlign = 'left'; g.fillText(rf.label, L + 4, Y(rf.y) - 4); g.globalAlpha = 1;
    }
    if (this.o.marks !== false) for (const rs of S.restarts) {   // restarts of the run
      if (rs < v.xmin || rs > v.xmax) continue; g.strokeStyle = C.warn; g.globalAlpha = .5; g.setLineDash([3, 4]);
      g.beginPath(); g.moveTo(X(rs), Tp); g.lineTo(X(rs), H - B); g.stroke(); g.setLineDash([]); g.fillStyle = C.warn; g.textAlign = 'left'; g.fillText('перезапуск', X(rs) + 3, Tp + 10); g.globalAlpha = 1;
    }
    const xs = T.vis.map(p => p.x);
    T.live.forEach((s, si) => {
      const raw = T.raws[s.k], sm = T.cols[s.k], col = s.color;
      if (T.a && this.o.rawline !== false) {                                     // faint raw
        g.strokeStyle = col; g.globalAlpha = .22; g.lineWidth = 1; g.beginPath(); let st = false;
        raw.forEach((y, i) => { if (y == null || !isFinite(y)) return; const px = X(xs[i]), py = Y(y); st ? g.lineTo(px, py) : (g.moveTo(px, py), st = true); }); g.stroke(); g.globalAlpha = 1;
      }
      const path = () => { g.beginPath(); let st = false, px0 = 0, py0 = 0;
        sm.forEach((y, i) => { if (y == null || !isFinite(y)) return; const px = X(xs[i]), py = Y(y);
          if (!st) { g.moveTo(px, py); st = true; } else { const mx = (px0 + px) / 2, my = (py0 + py) / 2; g.quadraticCurveTo(px0, py0, mx, my); }
          px0 = px; py0 = py; }); if (st) g.lineTo(px0, py0); return st; };
      if (si === 0 && this.o.area !== false && path()) {                        // gradient fill under the first line
        const last = xs.length - 1; g.lineTo(X(xs[last]), H - B); g.lineTo(X(xs[0]), H - B); g.closePath();
        const gr = g.createLinearGradient(0, Tp, 0, H - B); gr.addColorStop(0, col + '40'); gr.addColorStop(1, col + '00'); g.fillStyle = gr; g.fill();
      }
      g.strokeStyle = col; g.lineWidth = 2; g.lineJoin = 'round'; if (path()) g.stroke();
      const li = sm.length - 1; if (sm[li] != null) { const px = X(xs[li]), py = Y(sm[li]);   // pulsing head dot
        const ph = (performance.now() % 1600) / 1600; g.fillStyle = col; g.beginPath(); g.arc(px, py, 3.2, 0, 7); g.fill();
        g.globalAlpha = .35 * (1 - ph); g.beginPath(); g.arc(px, py, 3.2 + ph * 9, 0, 7); g.fill(); g.globalAlpha = 1; }
    });
    if (this.o.markers) T.live.forEach(s => { g.fillStyle = s.color; T.vis.forEach(p => { if (p[s.k] != null) { g.beginPath(); g.arc(X(p.x), Y(p[s.k]), 3, 0, 7); g.fill(); } }); });
    g.restore();
    if (this.hx != null && this.hx >= L && this.hx <= W - R) this.hover(X, Y, fmt, L, R);
    if (this.o.live) this.dirty = true;                                          // the pulsing dot keeps animating
  }
  hover(X, Y, fmt, L, R) {
    const { ctx: g, T, H } = this; const xs = T.vis.map(p => p.x); let bi = 0, bd = 1e18;
    xs.forEach((x, i) => { const d = Math.abs(X(x) - this.hx); if (d < bd) { bd = d; bi = i; } });
    const px = X(xs[bi]); g.strokeStyle = 'rgba(255,255,255,.35)'; g.lineWidth = 1; g.beginPath(); g.moveTo(px, 6); g.lineTo(px, H - 18); g.stroke();
    let h = '<b>' + (this.o.xhover ? this.o.xhover(xs[bi]) : 'шаг ' + xs[bi]) + '</b>';
    T.live.forEach(s => { const y = T.cols[s.k][bi], r = T.raws[s.k][bi]; if (y == null) return;
      g.fillStyle = s.color; g.beginPath(); g.arc(px, Y(y), 4, 0, 7); g.fill();
      h += '<br><i style="background:' + s.color + '"></i>' + s.label + ': <b>' + fmt(y) + '</b>' + (T.a && r != null && Math.abs(r - y) > 1e-9 ? ' <span style="color:#8a94a4">(' + fmt(r) + ')</span>' : ''); });
    const tip = $('#tip'); tip.innerHTML = h; tip.style.display = 'block';
    const tw = tip.offsetWidth; tip.style.left = Math.min(innerWidth - tw - 8, this.hxc + 14) + 'px'; tip.style.top = (this.hy + 14) + 'px';
  }
}
function loop() {
  for (const c of CHARTS) { const m = c.frame(); if (m) c.dirty = true; }
  animateNumbers(); interpolate(); requestAnimationFrame(loop);
}

// ---- animated numbers ------------------------------------------------------------------------------------------
const NUMS = [];
function num(node, fmt) { const o = { node, fmt, cur: null, tgt: null }; NUMS.push(o); return o; }
function setNum(o, v) { o.tgt = v; if (o.cur == null) o.cur = v; }
function animateNumbers() { for (const o of NUMS) { if (o.tgt == null) continue; o.cur += (o.tgt - o.cur) * .18; if (Math.abs(o.tgt - o.cur) < Math.abs(o.tgt) * 1e-4 + 1e-9) o.cur = o.tgt; const t = o.fmt(o.cur); if (o.node.textContent !== t) o.node.textContent = t; } }

// ---- build the page --------------------------------------------------------------------------------------------
const NAMES = { obj: 'уверенность', l1: 'рамка L1', giou: 'рамка GIoU', bce: 'маска BCE', dice: 'маска Dice', cen: 'центры людей', size: 'размеры людей', bnd: 'границы между людьми',
  state: 'состояние', place: 'положение', zone: 'зона', reid: 'ReID (учителя)', supcon: 'ReID (сравнение)', rel: 'связь треков', same: '«тот же человек»',
  dn_obj: 'DN уверенность', dn_l1: 'DN рамка L1', dn_giou: 'DN GIoU', dn_bce: 'DN маска BCE', dn_dice: 'DN маска Dice' };
const GROUPS = [['Детекция: рамки', ['obj', 'l1', 'giou']], ['Маски', ['bce', 'dice']], ['Люди на карте', ['cen', 'size', 'bnd']],
  ['Состояние, положение, зона', ['state', 'place', 'zone']], ['ReID и память', ['reid', 'supcon', 'rel', 'same']], ['Подсказки DN', ['dn_obj', 'dn_l1', 'dn_giou', 'dn_bce', 'dn_dice']]];
const QUAL = [{ k: 'recall', label: 'найдено', color: C.acc }, { k: 'precision', label: 'точность', color: C.ok }, { k: 'f1', label: 'F1', color: '#ffffff' },
  { k: 'hall', label: 'найдено в зале', color: C.pink }, { k: 'small', label: 'мелкие', color: C.warn }, { k: 'mask', label: 'маска IoU', color: C.vio },
  { k: 'idf1', label: 'IDF1 (трекинг)', color: C.cy }, { k: 'owner', label: 'экзамен владельца', color: C.dim },
  { k: 'f1c', label: 'F1 без обрывков SAM', color: '#b8ffcf' }];
// the test by the teacher (SAM 3.1, held-out 23.09) when there is one; older runs had only the owner's exam
const TQ = e => e.test && e.test.recall != null ? e.test : null;
const epochPts = () => { let lastT = {}; return S.epochs.map(e => { const t = TQ(e), x = e.exam || {}, tr = e.tracking || {}; if (tr.idf1 != null) lastT = tr;
  return t ? { x: e.step, recall: t.recall, precision: t.precision, f1: t.f1, hall: t.recall_hall, small: t.recall_small, mask: t.mask_iou_median, idf1: tr.idf1, owner: x.recall, f1c: t.f1_clean }
           : { x: e.step, recall: x.recall, precision: x.precision, small: x.small_recall, mask: x.mask_iou_median, idf1: tr.idf1 }; }); };
const rowPts = () => S.rows.map(r => Object.assign({ x: r.step, sps: (r.gpu_s || 0) + (r.data_s || 0) }, r));
const hwPts = () => S.hw.map(s => Object.assign({ x: s.ts }, s));
const hwx = (x, v) => { const d = Math.round(v.xmax - x); return d <= 0 ? 'сейчас' : d >= 60 ? '−' + Math.round(d / 60) + ' м' : '−' + d + ' с'; };

let built = false;
const KP = {};
function build() {
  built = true;
  const legend = $('#leg_q');
  QUAL.forEach(s => { const b = el('<span><i style="background:' + s.color + '"></i>' + s.label + '</span>'); b.onclick = () => { const key = 'ch_quality' + s.k; S.hidden.has(key) ? S.hidden.delete(key) : S.hidden.add(key); b.classList.toggle('off'); S.ver++; CHARTS.forEach(c => c.dirty = true); }; legend.appendChild(b); });
  new Chart($('#ch_quality'), epochPts, QUAL, { fmt: y => (100 * y).toFixed(0) + ' %', smooth: false, markers: true, yfix: [0, 1], live: true, window: 0,
    refs: [{ y: .95, color: C.ok, label: 'цель 95 %' }, { y: .90, color: C.dim, label: '90 %' }] });
  // kpis
  const K = $('#kpis');
  for (const [id, label, fmt] of [['recall', 'найдено', v => pct(v)], ['precision', 'точность', v => pct(v)], ['mask', 'маска IoU', v => v.toFixed(3)], ['idf1', 'IDF1 трекинга', v => pct(v)],
    ['gpu', 'загрузка карты', v => v.toFixed(0) + ' %'], ['vram', 'память карты', v => v.toFixed(1) + ' ГБ'], ['power', 'мощность', v => v.toFixed(0) + ' Вт'], ['temp', 'температура', v => v.toFixed(0) + ' °C']]) {
    const k = el('<div class="kpi"><em>' + label + '</em><b>—</b><span class="d flat">&nbsp;</span></div>'); K.appendChild(k);
    KP[id] = { box: k, n: num($('b', k), fmt), d: $('.d', k) };
  }
  // hardware charts
  const hw = $('#hw');
  for (const [id, title, keys, opt] of [['h_gpu', 'Загрузка видеокарты, %', [{ k: 'gpu', label: 'GPU', color: C.ok }], { yfix: [0, 100], fmt: y => y.toFixed(0) }],
    ['h_vram', 'Память карты, МБ', [{ k: 'vram', label: 'занято', color: C.acc }], { fmt: y => y.toFixed(0) }],
    ['h_pow', 'Мощность, Вт', [{ k: 'power', label: 'Вт', color: C.warn }], { zero: true, fmt: y => y.toFixed(0) }],
    ['h_temp', 'Температура, °C', [{ k: 'temp', label: '°C', color: C.bad }], { fmt: y => y.toFixed(0) }],
    ['h_cpu', 'Процессор, %', [{ k: 'cpu', label: 'CPU', color: C.vio }], { yfix: [0, 100], fmt: y => y.toFixed(0) }],
    ['h_ram', 'Оперативная память, ГБ', [{ k: 'ram', label: 'ГБ', color: C.cy }], { fmt: y => y.toFixed(1) }],
    ['h_clk', 'Частота ядра, МГц', [{ k: 'clock', label: 'МГц', color: C.pink }], { fmt: y => y.toFixed(0) }]]) {
    const cell = el('<div class="cell"><div class="t"><b>' + title + '</b><span></span></div><div class="chart" id="' + id + '"></div></div>'); hw.appendChild(cell);
    const ch = new Chart($('.chart', cell), hwPts, keys, Object.assign({ window: 600, xfmt: hwx, xhover: x => new Date(x * 1000).toLocaleTimeString(), marks: false, live: true }, opt));
    ch.valEl = $('.t span', cell); ch.valKey = keys[0].k; ch.valFmt = opt.fmt;
  }
  const sp = $('#speed');
  for (const [id, title, keys, opt] of [['s_sps', 'Секунд на шаг', [{ k: 'sps', label: 'с/шаг', color: C.acc }, { k: 'gpu_s', label: 'на карте', color: C.warn }], { zero: true, fmt: y => y.toFixed(2) }],
    ['s_lr', 'Скорость обучения (lr)', [{ k: 'lr', label: 'lr', color: C.vio }], { smooth: false, fmt: y => y.toExponential(1) }],
    ['s_mem', 'Память процесса обучения, ГБ', [{ k: 'mem_gb', label: 'ГБ', color: C.cy }], { fmt: y => y.toFixed(2) }]]) {
    const cell = el('<div class="cell"><div class="t"><b>' + title + '</b><span></span></div><div class="chart" id="' + id + '"></div></div>'); sp.appendChild(cell);
    const ch = new Chart($('.chart', cell), rowPts, keys, opt); ch.valEl = $('.t span', cell); ch.valKey = keys[0].k; ch.valFmt = opt.fmt; ch.rows = true;
  }
  const L = $('#losses');
  for (const [title, keys] of GROUPS) {
    L.appendChild(el('<h3>' + title + '</h3>')); const g = el('<div class="grid g3"></div>'); L.appendChild(g);
    keys.forEach((k, i) => { const cell = el('<div class="cell"><div class="t"><b>' + NAMES[k] + '</b><span></span></div><div class="chart" id="l_' + k + '"></div></div>'); g.appendChild(cell);
      const ch = new Chart($('.chart', cell), rowPts, [{ k, label: NAMES[k], color: PAL[i % PAL.length] }], { fmt: y => Math.abs(y) >= 10 ? y.toFixed(1) : y.toFixed(3) });
      ch.valEl = $('.t span', cell); ch.valKey = k; ch.valFmt = y => (+y).toFixed(3); ch.rows = true; ch.trend = true; });
  }
}
function trendText(vals) {          // smoothed change over the last quarter of the visible history, lower is better for losses
  const s = ema(vals, S.smooth || .5).filter(x => x != null); if (s.length < 8) return '';
  const a = s[s.length - 1], b = s[Math.max(0, s.length - 1 - Math.max(4, Math.floor(s.length / 4)))]; if (!b) return '';
  const d = (a - b) / Math.abs(b) * 100; return Math.abs(d) < 1 ? '<span class="flat"> →</span>' : d < 0 ? '<span class="up"> ▼ ' + Math.abs(d).toFixed(0) + '%</span>' : '<span class="down"> ▲ ' + d.toFixed(0) + '%</span>';
}
function refreshSmall() {
  for (const c of CHARTS) { if (!c.valEl) continue; const P = c.pts(); if (!P.length) continue; const last = P[P.length - 1][c.valKey]; if (last == null) continue;
    c.valEl.innerHTML = (c.valFmt ? c.valFmt(last) : last) + (c.trend ? trendText(P.map(p => p[c.valKey])) : ''); }
}
function renderEpochs() {
  const tb = $('#ep tbody'); const E = S.epochs; if (!E.length) { tb.innerHTML = ''; return; }
  const T = e => TQ(e) || {};
  const cols = [e => T(e).recall, e => T(e).precision, e => T(e).f1, e => T(e).recall_hall, e => T(e).recall_out, e => T(e).recall_small, e => T(e).mask_iou_median,
    e => e.exam && e.exam.recall, e => e.tracking && e.tracking.idf1, e => e.tracking && e.tracking.switches, e => e.tracking && e.tracking.found];
  const best = cols.map((f, i) => { const v = E.map(f).filter(x => x != null); return v.length ? (i === 9 ? Math.min(...v) : Math.max(...v)) : null; });
  const fm = [pct, pct, pct, pct, pct, pct, f3, pct, pct, v => v, pct];
  const html = E.slice().reverse().map(e => '<tr' + (e.step === S.newEpoch ? ' class="newrow"' : '') + '><td>' + e.epoch + (e.frozen ? ' ❄' : '') + '</td><td>' + e.step + '</td><td>' + (e.time || '') + '</td>' +
    cols.map((f, i) => { const v = f(e); return '<td' + (v != null && v === best[i] ? ' class="best"' : '') + '>' + (v == null ? '—' : fm[i](v)) + '</td>'; }).join('') + '</tr>').join('');
  if (tb._h !== html) { tb.innerHTML = html; tb._h = html; }
  // kpis from the last epoch with deltas against the previous one
  const l = E[E.length - 1], p = E.length > 1 ? E[E.length - 2] : null;
  const Q = e => e ? (TQ(e) || e.exam || {}) : {};
  const trk = E.filter(e => e.tracking && e.tracking.idf1 != null), lt = trk[trk.length - 1], pt = trk.length > 1 ? trk[trk.length - 2] : null;
  [['recall', Q(l).recall, p && Q(p).recall, 1], ['precision', Q(l).precision, p && Q(p).precision, 1],
   ['mask', Q(l).mask_iou_median, p && Q(p).mask_iou_median, 0], ['idf1', lt && lt.tracking.idf1, pt && pt.tracking.idf1, 1]].forEach(([id, v, pv, isp]) => {
    if (v == null) return; setNum(KP[id].n, v); if (pv == null) return; const d = v - pv;
    KP[id].d.className = 'd ' + (Math.abs(d) < 1e-4 ? 'flat' : d > 0 ? 'up' : 'down');
    KP[id].d.textContent = (d > 0 ? '▲ +' : d < 0 ? '▼ ' : '= ') + (isp ? (100 * d).toFixed(1) + ' п.п.' : d.toFixed(3)) + ' к прошлой эпохе (' + l.epoch + ')'; });
}
function renderHwKpi() {
  const s = S.hw[S.hw.length - 1]; if (!s) return;
  [['gpu', s.gpu, '%'], ['vram', s.vram != null ? s.vram / 1024 : null, ''], ['power', s.power, ''], ['temp', s.temp, '']].forEach(([id, v, u]) => { if (v != null && KP[id]) setNum(KP[id].n, v); });
  if (s.vram_total && KP.vram) KP.vram.d.textContent = 'из ' + (s.vram_total / 1024).toFixed(0) + ' ГБ';
  if (s.power_limit && KP.power) KP.power.d.textContent = 'предел ' + s.power_limit.toFixed(0) + ' Вт';
  if (KP.gpu) KP.gpu.d.textContent = s.clock ? s.clock.toFixed(0) + ' МГц' : '';
}

// ---- live interpolation between log rows --------------------------------------------------------------------------
let H = {};
function stepsPerSec() {            // the real pace: wall-clock seconds per step over the last rows of this process, tests included
  const r = S.rows; let k = r.length - 1; if (k < 1) return null;
  let a = k; while (a > 0 && r[a - 1].t < r[a].t && r[a - 1].step < r[a].step && k - a < 24) a--;
  if (a === k) return null; const dt = r[k].t - r[a].t, ds = r[k].step - r[a].step; return ds > 0 && dt > 0 ? dt / ds : null;
}
function interpolate() {
  const st = S.status; if (!st.steps) return;
  const sps = stepsPerSec() || st.step_s || 3; const age = (st.age_s || 0) + (performance.now() - S.fetchedAt) / 1000;
  const training = st.phase === 'training'; const live = training ? Math.min(st.step + 20, st.step + age / sps) : st.step;
  $('#step_live').textContent = Math.floor(live).toLocaleString('ru');
  $('#prog').style.width = (100 * live / st.steps).toFixed(2) + '%';
  const eta = (st.steps - live) * sps; $('#eta').textContent = training ? mmss(eta) : '—';
  const endAt = new Date(Date.now() + eta * 1000); $('#etaclock').textContent = training ? hhmm(endAt) : '—';
  if (st.until && training) { const [hh, mm] = st.until.split(':').map(Number), now = new Date(), stop = new Date(now.getFullYear(), now.getMonth(), now.getDate(), hh, mm);
    $('#note').textContent = 'остановка по расписанию в ' + st.until + (stop < endAt && stop > now ? ' — до конца расписания не дойдёт' : ''); } else $('#note').textContent = training ? 'остановки по времени нет' : '';
}

// ---- fetch loop ---------------------------------------------------------------------------------------------------
let busy = false;
async function tick() {
  if (busy) return; busy = true;
  try {
    const r = await fetch('/api/train/live?run=' + encodeURIComponent(S.run) + '&n=' + S.n + '&en=' + S.en + '&hw=' + S.hwT, { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status); const d = await r.json();
    if (d.reset) { S.rows = []; S.epochs = []; S.restarts = []; }
    if (!S.run) S.run = d.run;
    if (!new URLSearchParams(location.search).get('run') && d.newest && d.newest !== S.run) { location.reload(); return; }   // the queue started the next variant
    if (!$('#run').options.length) { d.runs.forEach(x => { const o = document.createElement('option'); o.textContent = x; o.selected = x === d.run; $('#run').appendChild(o); }); }
    S.n = d.n; S.en = d.en; addRows(d.log); addEpochs(d.epochs);
    if (d.hw.length) { S.hw = S.hw.concat(d.hw).slice(-1800); S.hwT = d.hw[d.hw.length - 1].ts; }
    S.status = d.status || {}; S.fetchedAt = performance.now();
    if (!built) build();
    $('#steps').textContent = (S.status.steps || 0).toLocaleString('ru'); $('#epoch').textContent = S.status.epoch || '—'; $('#sps').textContent = (stepsPerSec() || S.status.step_s || 0).toFixed(2);
    const ph = S.status.phase || ''; $('#phase').textContent = ph === 'training' ? 'обучение' : ph.startsWith('testing') ? 'тест по учителю, эпоха ' + ph.split(' ').pop()
      : ph.startsWith('tracking') ? 'тест трекинга, эпоха ' + ph.split(' ').pop() : ph.startsWith("owner's") ? 'экзамен владельца, эпоха ' + ph.split(' ').pop() : (ph || 'нет данных');
    const fresh = (S.status.age_s || 999) < 400; $('#phase').className = 'pill ' + (fresh ? 'ok' : 'bad');
    const total = S.status.elapsed_s; $('#elapsed').textContent = total ? mmss(total) : '—';
    S.ok = true; $('#conn').textContent = 'онлайн'; $('#conn').className = 'pill ok'; $('#live').className = 'live on';
    S.ver++; if (d.log.length || d.epochs.length || d.reset) { CHARTS.forEach(c => c.dirty = true); refreshSmall(); }
    if (d.epochs.length || d.reset) renderEpochs();
    renderHwKpi(); if (d.hw.length) CHARTS.forEach(c => { if (c.valEl && !c.rows) { const P = c.pts(), l = P[P.length - 1]; if (l && l[c.valKey] != null) c.valEl.textContent = c.valFmt(l[c.valKey]); } });
  } catch (e) { S.ok = false; $('#conn').textContent = 'нет связи'; $('#conn').className = 'pill bad'; $('#live').className = 'live'; }
  busy = false;
}
$('#smooth').oninput = e => { S.smooth = e.target.value / 100; S.ver++; $('#smooth_v').textContent = S.smooth.toFixed(2); CHARTS.forEach(c => c.dirty = true); refreshSmall(); };
$('#range').onchange = e => { S.range = +e.target.value; S.ver++; CHARTS.forEach(c => c.dirty = true); };
$('#run').onchange = e => { location.search = '?run=' + e.target.value; };
$('#smooth').value = S.smooth * 100; $('#smooth_v').textContent = S.smooth.toFixed(2);
tick(); setInterval(tick, 1000); requestAnimationFrame(loop);

// ---- the variant queue ------------------------------------------------------------------------------------------
const QSTATE = { running: 'идёт', done: 'дошёл до конца', stopped: 'остановлен', failed: 'упал' };
async function queueTick() {
  try {
    const r = await fetch('/api/train/queue', { cache: 'no-store' }); if (!r.ok) return; const d = await r.json();
    const f = (v, s) => v == null ? '—' : pct(v) + (s != null ? ' <small>(шаг ' + s + ')</small>' : '');
    const html = (d.items || []).map(i => '<tr' + (i.state === 'running' ? ' class="newrow"' : '') + '><td><a href="?run=' + i.name + '">' + i.name + '</a></td><td>' + (i.note || '') +
      '</td><td>' + (QSTATE[i.state] || 'ждёт') + (i.started ? ' <small>с ' + i.started + '</small>' : '') + '</td><td>' + f(i.best_f1, i.best_step) + '</td><td>' + f(i.last_f1, i.last_step) +
      '</td><td>' + (i.reason || '') + '</td></tr>').join('');
    const tb = $('#queue tbody'); if (tb && tb._h !== html) { tb.innerHTML = html; tb._h = html; }
  } catch (e) { }
}
queueTick(); setInterval(queueTick, 10000);
