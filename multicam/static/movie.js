'use strict';
/* The working day as a film.
 *
 * Two files, one per camera, cut on one clock: position t in either file is the same
 * moment in the shop, so the players only have to stand at the same place. Every person
 * is outlined by the detector and numbered by the machine; the owner watches at speed and
 * corrects where he sees it go wrong. He is never asked who somebody is.
 *
 * One way in for every correction: click the person (or the empty spot where somebody has
 * no outline) and a menu next to him says what can be wrong. The keys do the same things
 * faster, on whoever is under the mouse.
 *
 * Outlines are drawn for the frame the browser actually shows (requestVideoFrameCallback),
 * not for currentTime: at 8x the picture lags the clock and the masks ran ahead of people.
 */
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const hhmmss = new Intl.DateTimeFormat('ru-RU', {hour:'2-digit',minute:'2-digit',second:'2-digit',timeZone:'Asia/Bangkok'});
const hhmm = new Intl.DateTimeFormat('ru-RU', {hour:'2-digit',minute:'2-digit',timeZone:'Asia/Bangkok'});
const PALETTE = ['#7ae1b3','#f5a3c7','#9ad0ff','#ffd479','#c3a6ff','#8fe388','#ff9f80','#7fd4d0','#e0b0ff','#b8d96b','#ffa0a0','#6fb3ff'];
const PASSER = '#c9d6e3';
const RATES = [1, 2, 4, 8, 12, 16];
const SLOW = 2;          // speed around a moment the machine is unsure of
const HOLD = 0.35;       // an outline stays up this long; the detector looked every 0.12 s
const LANE = 150;        // seconds either side in the lane under the players
const FLUSH = 8000;      // ms between reports of what was watched
const RAW = 2560;        // width of the original frame: the unit of "missed" marks
const CAMS = ['cam1', 'cam2'];
const KIND = {customer: 'покупатель', staff: 'сотрудник', passer: 'прохожий'};

let day = null, view = null, rate = 8, playing = false, sel = null, hover = null, mouse = null;
let skipEmpty = true, slowDoubts = true, passers = true, busy = false, review = null, pausedByMenu = false;
let numbered = {}, everyone = {}, activeSeconds = 0, span = null, lastT = null, lastFlush = 0, pending = [];
let unit = 1280;
let draft = null, lastDraw = null, lastAction = null, menuAt = null, lastWarm = null, draftSeq = 0;
let drawnIndex = {cam1: [], cam2: [], fix: {}};
const clips = new Map();           // clip -> {state, at, frames: {cam1: [], cam2: []}}
const thumbs = new Map();          // part -> {h, canvas}
const drawn = {cam1: [], cam2: []};
const shownAt = {cam1: null, cam2: null};
const videos = {cam1: $('#v1'), cam2: $('#v2')};
const overlays = {cam1: $('#o1'), cam2: $('#o2')};
const blanks = {cam1: $('#n1'), cam2: $('#n2')};
const menu = $('#menu');

async function api(url, options) {
  const response = await fetch(url, options);
  let body = {};
  try { body = await response.json(); } catch (nothing) { body = {}; }
  if (!response.ok) {
    const failure = Error(body.error || `Сервер ответил ${response.status}`);
    failure.status = response.status;
    throw failure;
  }
  return body;
}

function notice(text, bad) {
  const box = $('#notice');
  box.textContent = text || '';
  box.classList.toggle('error', !!bad);
}

const clockOf = t => view ? hhmmss.format(new Date((view.start + t) * 1000)) : '—';
function lasting(seconds) {
  seconds = Math.max(0, Math.round(seconds));
  if (seconds < 60) return `${seconds} с`;
  const m = Math.floor(seconds / 60), s = seconds % 60;
  if (m < 60) return s ? `${m} мин ${s} с` : `${m} мин`;
  return `${Math.floor(m / 60)} ч ${m % 60} мин`;
}

/* ---------- the model: parts, people, where the owner cut ---------- */

function partOf(track, t) {
  const cuts = view.cuts[track];
  let id = track + '@';
  if (cuts) for (const [at, pid] of cuts) { if (t >= at - 1e-6) id = pid; else break; }
  return id;
}

const part = pid => view.parts[pid];                  // [person, first, last, shop, false]
const personIdOf = pid => { const p = part(pid); return p && !p[4] ? p[0] : null; };
const personOf = pid => { const id = personIdOf(pid); return id ? view.persons[id] || null : null; };
const colour = person => person && person.color !== null ? PALETTE[person.color % PALETTE.length] : PASSER;
function nameOf(person) {
  if (!person) return 'не человек';
  if (person.n === null) return KIND[person.kind];
  return `#${person.n}${person.kind === 'staff' ? ' сотр.' : ''}`;
}

function adopt(next) {
  view = next;
  numbered = {}; everyone = {};
  for (const [pid, p] of Object.entries(view.parts)) {
    if (p[4] || !view.persons[p[0]]) continue;
    (everyone[p[0]] = everyone[p[0]] || []).push([p[1], p[2], pid]);
    if (view.persons[p[0]].n !== null) (numbered[p[0]] = numbered[p[0]] || []).push([p[1], p[2], pid]);
  }
  activeSeconds = view.activity.reduce((sum, [a, b]) => sum + b - a, 0);
  drawnIndex = {cam1: [], cam2: [], fix: {}};
  for (const d of view.drawn || []) {
    const samples = d.samples.slice().sort((a, b) => a[0] - b[0]);
    if (d.kind === 'new') drawnIndex[d.cam].push({k: d.track, samples});
    else (drawnIndex.fix[d.replaces] = drawnIndex.fix[d.replaces] || []).push(...samples);
  }
  $('#undo').disabled = !view.can_undo && !lastDraw;
  progress();
  drawDay();
}

function watchedActive() {
  let total = 0;
  for (const [a, b] of view.watched.spans) {
    for (const [c, d] of view.activity) {
      const lo = Math.max(a, c), hi = Math.min(b, d);
      if (hi > lo) total += hi - lo;
    }
  }
  return total;
}

function progress() {
  const people = Object.values(view.persons).filter(p => p.n !== null).length;
  const edits = Object.values(view.edits).reduce((a, b) => a + b, 0);
  const seen = watchedActive();
  const share = activeSeconds ? Math.round(100 * seen / activeSeconds) : 0;
  $('#progress').textContent = `просмотрено ${lasting(seen)} из ${lasting(activeSeconds)} (${share}%) · людей ${people} · правок ${edits}`
    + (view.stale.length ? ` · ${view.stale.length} треков пересчитаны ночью, правки на них не действуют` : '');
}

/* ---------- outlines of one clip, fetched when the film gets near it ---------- */

function clipsNear(t, ahead) {
  return view.clips.filter(c => c.to + 20 >= t && c.from - ahead <= t).map(c => c.clip);
}

async function loadClip(clip) {
  const had = clips.get(clip);
  if (had && !(had.state === 'failed' && Date.now() - had.at > 20000)) return;   // a dropped tunnel gets another go
  const entry = {state: 'loading', frames: {cam1: [], cam2: []}, at: Date.now()};
  clips.set(clip, entry);
  try {
    const body = await api(`/api/movie/${day}/clip/${encodeURIComponent(clip)}`);
    unit = body.unit || unit;
    const byTime = {cam1: new Map(), cam2: new Map()};
    for (const track of body.tracks) {
      const cam = track.cam === 1 ? 'cam1' : 'cam2';
      track.t.forEach((t, i) => {
        const bucket = byTime[cam].get(t) || [];
        bucket.push({k: track.k, rings: track.p[i]});
        byTime[cam].set(t, bucket);
      });
    }
    for (const cam of CAMS) {
      entry.frames[cam] = [...byTime[cam].entries()].sort((a, b) => a[0] - b[0]).map(([t, items]) => ({t, items}));
    }
    entry.state = 'ready';
  } catch (error) {
    entry.state = 'failed';
    entry.at = Date.now();
    notice('Не загрузились обводки ' + clip + ': ' + error.message + ' — попробую ещё раз', true);
  }
}

let lastEvict = 0;
function evict(t) {
  // a whole day of outlines would not fit in a tab: keep the windows around now, drop the rest
  if (performance.now() - lastEvict < 5000) return;
  lastEvict = performance.now();
  for (const c of view.clips) {
    const entry = clips.get(c.clip);
    if (entry && entry.state !== 'loading' && (c.to < t - 900 || c.from > t + 1800)) clips.delete(c.clip);
  }
}

function frameAt(cam, t) {
  let best = null;
  for (const clip of clipsNear(t, 0)) {
    const entry = clips.get(clip);
    if (!entry || entry.state !== 'ready') continue;
    const frames = entry.frames[cam];
    let lo = 0, hi = frames.length - 1, found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (frames[mid].t <= t + 0.001) { found = mid; lo = mid + 1; } else hi = mid - 1;
    }
    if (found >= 0 && t - frames[found].t <= HOLD && (!best || frames[found].t > best.t)) best = frames[found];
  }
  return best;
}

/* ---------- which frame is on screen ---------- */

function followFrames(cam) {
  const v = videos[cam];
  if (!('requestVideoFrameCallback' in v)) return;
  const step = (now, meta) => { shownAt[cam] = meta.mediaTime; v.requestVideoFrameCallback(step); };
  v.requestVideoFrameCallback(step);
}
CAMS.forEach(followFrames);

function onScreen(cam) {
  // the moment of the frame the browser painted; currentTime runs ahead of it at speed
  const v = videos[cam], t = shownAt[cam];
  if (v.paused || v.seeking) return v.currentTime;      // standing still, the picture is the clock
  return t !== null && Math.abs(t - v.currentTime) < 3 ? t : v.currentTime;
}

/* ---------- drawing ---------- */

function boxOf(rings) {
  let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
  for (const r of rings) for (let i = 0; i < r.length; i += 2) {
    x1 = Math.min(x1, r[i]); x2 = Math.max(x2, r[i]);
    y1 = Math.min(y1, r[i + 1]); y2 = Math.max(y2, r[i + 1]);
  }
  return [x1, y1, x2, y2];
}

function inside(rings, x, y) {
  for (const r of rings) {
    let hit = false;
    for (let i = 0, j = r.length - 2; i < r.length; j = i, i += 2) {
      const xi = r[i], yi = r[i + 1], xj = r[j], yj = r[j + 1];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) hit = !hit;
    }
    if (hit) return true;
  }
  return false;
}

function trace(ctx, rings, k) {
  ctx.beginPath();
  for (const r of rings) {
    for (let i = 0; i < r.length; i += 2) {
      const x = r[i] * k, y = r[i + 1] * k;
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }
    ctx.closePath();
  }
}

const badNear = (pid, t) => view.badmask.some(m => m.part === pid && Math.abs(m.at - t) < 1.5);

function render() {
  for (const cam of CAMS) {
    const t = onScreen(cam);
    const canvas = overlays[cam], rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
    const width = Math.round(rect.width * ratio), height = Math.round(rect.height * ratio);
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    const k = width / unit;
    const frame = frameAt(cam, t);
    drawn[cam] = [];
    const items = [];
    if (frame) for (const item of frame.items) {
      const fix = (drawnIndex.fix[item.k] || []).find(([at]) => Math.abs(at - frame.t) <= 0.02);
      items.push(fix ? {k: item.k, rings: fix[1], at: frame.t, hand: true} : {...item, at: frame.t});
    }
    for (const d of drawnIndex[cam]) {
      let latest = null;
      for (const sample of d.samples) { if (sample[0] <= t + 0.001) latest = sample; else break; }
      if (latest && t - latest[0] <= HOLD) items.push({k: d.k, rings: latest[1], at: latest[0], hand: true});
    }
    if (items.length) {
      for (const item of items) {
        const pid = partOf(item.k, t), p = part(pid);
        if (!p) continue;
        const person = personOf(pid), isFalse = !!p[4];
        if (person && person.n === null && !passers && !(sel && sel.pid === pid)) continue;
        const entry = {pid, rings: item.rings, person, isFalse, cam, box: boxOf(item.rings), bad: badNear(pid, t),
                       hand: !!item.hand, at: item.at};
        drawn[cam].push(entry);
        trace(ctx, item.rings, k);
        const focus = (sel && sel.pid === pid) || (hover && hover.pid === pid);
        if (isFalse) {
          ctx.setLineDash([5 * ratio, 4 * ratio]);
          ctx.strokeStyle = '#ff8a80'; ctx.lineWidth = 1.5 * ratio; ctx.stroke();
          ctx.setLineDash([]);
          continue;
        }
        const c = colour(person), passer = person && person.n === null;
        ctx.globalAlpha = focus ? 0.5 : passer ? 0.22 : 0.33; ctx.fillStyle = c; ctx.fill('nonzero'); ctx.globalAlpha = 1;
        if (entry.bad) { ctx.setLineDash([4 * ratio, 3 * ratio]); ctx.strokeStyle = '#ffd24d'; ctx.lineWidth = 2.5 * ratio; }
        else if (passer && !focus) {
          // a pale line vanished on the bright gallery floor and read as "no outline":
          // dark under light shows on any background
          ctx.strokeStyle = '#000a'; ctx.lineWidth = 3 * ratio; ctx.stroke();
          ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4 * ratio;
        } else { ctx.strokeStyle = focus ? '#fff' : c; ctx.lineWidth = (focus ? 3 : 1.6) * ratio; }
        ctx.stroke(); ctx.setLineDash([]);
        capture(cam, pid, entry.box, t - item.at);
      }
      ctx.font = `600 ${Math.round(13 * ratio)}px system-ui,sans-serif`;
      ctx.textBaseline = 'bottom';
      for (const entry of drawn[cam]) {
        const passer = entry.person && entry.person.n === null;
        if (passer && !(hover && hover.pid === entry.pid) && !(sel && sel.pid === entry.pid)) continue;
        const label = entry.isFalse ? 'не человек' : nameOf(entry.person) + (entry.bad ? ' · обводка?' : '') + (entry.hand ? ' ✎' : '');
        const x = entry.box[0] * k, y = Math.max(16 * ratio, entry.box[1] * k - 2 * ratio);
        const w = ctx.measureText(label).width + 8 * ratio;
        ctx.fillStyle = '#000b'; ctx.fillRect(x, y - 16 * ratio, w, 16 * ratio);
        ctx.fillStyle = entry.isFalse ? '#ff8a80' : entry.bad ? '#ffd24d' : colour(entry.person); ctx.fillText(label, x + 4 * ratio, y);
      }
    }
    const raw = width / RAW;
    for (const mark of view.missed) {
      if (mark.cam !== cam || Math.abs(mark.at - t) > 1.5) continue;
      ctx.strokeStyle = '#ff5f56'; ctx.lineWidth = 2.5 * ratio;
      ctx.beginPath(); ctx.arc(mark.x * raw, mark.y * raw, 16 * ratio, 0, 2 * Math.PI); ctx.stroke();
      ctx.fillStyle = '#ff5f56'; ctx.font = `600 ${Math.round(12 * ratio)}px system-ui`;
      ctx.fillText('без обводки', mark.x * raw + 19 * ratio, mark.y * raw);
    }
    if (draft && draft.cam === cam) {
      // the outline being made: white, dashed, with the clicks that shaped it
      if (draft.rings) {
        trace(ctx, draft.rings, raw);
        ctx.globalAlpha = 0.35; ctx.fillStyle = '#fff'; ctx.fill('nonzero'); ctx.globalAlpha = 1;
        ctx.strokeStyle = '#000a'; ctx.lineWidth = 4 * ratio; ctx.stroke();
        ctx.setLineDash([6 * ratio, 4 * ratio]); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2 * ratio; ctx.stroke(); ctx.setLineDash([]);
      }
      for (const [x, y, label] of draft.points) {
        ctx.beginPath(); ctx.arc(x * raw, y * raw, 6 * ratio, 0, 2 * Math.PI);
        ctx.fillStyle = label ? '#3ddc84' : '#ff5f56'; ctx.fill();
        ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5 * ratio; ctx.stroke();
      }
      if (draft.busy && draft.points.length) {
        const [x, y] = draft.points[draft.points.length - 1];
        ctx.fillStyle = '#fff'; ctx.font = `600 ${Math.round(13 * ratio)}px system-ui`;
        ctx.fillText('обвожу…', x * raw + 10 * ratio, y * raw - 10 * ratio);
      }
    }
    const hole = view.videos[cam].holes.some(([a, b]) => t >= a && t < b);
    const labelled = view.covered.some(([a, b]) => t >= a && t < b);
    blanks[cam].textContent = view.videos[cam].stale ? 'Фильм этой камеры пересобирается с исправленной синхронизацией'
      : !view.videos[cam].ready ? 'Фильм этой камеры ещё собирается'
      : hole ? 'Камера здесь ничего не записала' : !labelled ? 'Это время ночью ещё не разметили — обводок нет' : '';
  }
}

function capture(cam, pid, box, age) {
  // the best look at every part, cut from the film while it plays: the cards need no server
  const s = videos[cam].videoWidth ? videos[cam].videoWidth / unit : 640 / unit;
  const [x1, y1, x2, y2] = box.map(v => v * s);
  const h = y2 - y1;
  if (age > 0.1 || h < 30) return;
  const had = thumbs.get(pid);
  if (had && had.h >= h) return;
  const video = videos[cam];
  if (video.readyState < 2) return;
  const canvas = had ? had.canvas : document.createElement('canvas');
  canvas.width = 48; canvas.height = 96;
  const pad = 0.12 * h, sx = x1 - pad, sy = y1 - pad, sw = x2 - x1 + 2 * pad, sh = h + 2 * pad;
  const fit = Math.min(48 / sw, 96 / sh);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 48, 96);
  try {
    ctx.drawImage(video, sx, sy, sw, sh, (48 - sw * fit) / 2, (96 - sh * fit) / 2, sw * fit, sh * fit);
    thumbs.set(pid, {h, canvas});
  } catch (nothing) { /* the frame is not decodable yet */ }
}

function thumbOf(person) {
  let best = null;
  for (const [, , pid] of everyone[person] || []) {
    const t = thumbs.get(pid);
    if (t && (!best || t.h > best.h)) best = t;
  }
  return best;
}

function card(id, note, onClick) {
  const p = view.persons[id];
  const button = document.createElement('button');
  button.className = 'card';
  button.style.borderColor = colour(p);
  const pic = document.createElement('canvas'); pic.width = 48; pic.height = 96;
  const thumb = thumbOf(id);
  if (thumb) pic.getContext('2d').drawImage(thumb.canvas, 0, 0);
  button.append(pic);
  const text = document.createElement('span');
  text.innerHTML = `<span class="n" style="color:${colour(p)}">${esc(nameOf(p))}</span><br><span class="when">${esc(note)}</span>`;
  button.append(text);
  button.addEventListener('click', onClick);
  return button;
}

/* ---------- the day line and the lane of people around now ---------- */

const dayImage = document.createElement('canvas');
function drawDay() {
  // the day itself changes only on an edit or a report of watching: drawn once, aside
  const canvas = $('#dayline'), rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  canvas.width = dayImage.width = Math.round(rect.width * ratio);
  canvas.height = dayImage.height = Math.round(rect.height * ratio);
  if (!view) return;
  const ctx = dayImage.getContext('2d'), W = dayImage.width, H = dayImage.height, x = t => t / view.duration * W;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#3a2a1c';                 // not labelled by the night passes yet
  let from = 0;
  for (const [a, b] of view.covered.concat([[view.duration, view.duration]])) {
    if (a > from) ctx.fillRect(x(from), H * 0.35, x(a) - x(from), H * 0.4);
    from = Math.max(from, b);
  }
  ctx.fillStyle = '#26323c';
  for (const [a, b] of view.activity) ctx.fillRect(x(a), H * 0.35, Math.max(1, x(b) - x(a)), H * 0.4);
  ctx.fillStyle = '#3f9f73';
  for (const [a, b] of view.watched.spans) ctx.fillRect(x(a), H * 0.35, Math.max(1, x(b) - x(a)), H * 0.4);
  ctx.fillStyle = '#e0a93c';
  for (const d of view.doubts) ctx.fillRect(x(d), H * 0.78, Math.max(1, ratio), H * 0.16);
  ctx.fillStyle = '#ff5f56';
  for (const m of view.missed.concat(view.badmask)) ctx.fillRect(x(m.at) - ratio, H * 0.05, 2 * ratio, H * 0.26);
  ctx.fillStyle = '#b0bcc6'; ctx.font = `${Math.round(11 * ratio)}px system-ui`; ctx.textBaseline = 'top';
  const firstHour = Math.ceil(view.start / 3600) * 3600;
  for (let s = firstHour; s < view.start + view.duration; s += 3600) {
    const px = x(s - view.start);
    ctx.fillRect(px, 0, ratio, H * 0.3);
    ctx.fillText(hhmm.format(new Date(s * 1000)), px + 3 * ratio, 1 * ratio);
  }
}

function drawCursor(t) {
  const canvas = $('#dayline');
  if (!view || !canvas.width) return;
  const ctx = canvas.getContext('2d'), ratio = window.devicePixelRatio || 1;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(dayImage, 0, 0);
  ctx.fillStyle = '#fff';
  ctx.fillRect(t / view.duration * canvas.width - ratio, 0, 2 * ratio, canvas.height);
}

let laneRows = [];
function drawLane(t) {
  const canvas = $('#lane'), rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!view) return;
  const lo = t - LANE, hi = t + LANE, x = s => (s - lo) / (hi - lo) * W;
  const here = Object.entries(numbered)
    .map(([person, list]) => [person, list.filter(([a, b]) => b >= lo && a <= hi)])
    .filter(([, list]) => list.length)
    .sort((a, b) => view.persons[a[0]].first - view.persons[b[0]].first);
  const rows = [];
  laneRows = [];
  const rowH = 13 * ratio, gap = 3 * ratio;
  for (const [person, list] of here) {
    const a = Math.min(...list.map(p => p[0])), b = Math.max(...list.map(p => p[1]));
    let row = rows.findIndex(end => end < a - 4);
    if (row < 0) { row = rows.length; rows.push(b); } else rows[row] = b;
    if ((row + 1) * (rowH + gap) > H) continue;
    const y = row * (rowH + gap) + gap, p = view.persons[person];
    ctx.fillStyle = colour(p);
    for (const [s, e, pid] of list) {
      ctx.globalAlpha = sel && personIdOf(sel.pid) === person ? 1 : 0.8;
      ctx.fillRect(x(s), y, Math.max(2 * ratio, x(e) - x(s)), rowH);
      laneRows.push({x1: x(s), x2: x(e), y, h: rowH, at: s, pid});
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#0e1419'; ctx.font = `600 ${Math.round(11 * ratio)}px system-ui`; ctx.textBaseline = 'top';
    ctx.fillText(nameOf(p), Math.max(x(a), 0) + 3 * ratio, y + ratio);
  }
  ctx.fillStyle = '#e0a93c';
  for (const d of view.doubts) if (d >= lo && d <= hi) ctx.fillRect(x(d), H - 6 * ratio, ratio * 1.5, 6 * ratio);
  ctx.fillStyle = '#fff';
  ctx.fillRect(W / 2 - ratio, 0, 2 * ratio, H);
}

/* ---------- cards under the players: who is here, who just left ---------- */

function visibleNow() {
  const now = {cam1: new Set(), cam2: new Set()};
  for (const cam of CAMS) for (const e of drawn[cam]) if (e.person) now[cam].add(personIdOf(e.pid));
  return now;
}

function lastBefore(id, t) {
  return Math.max(...(everyone[id] || []).filter(([a]) => a < t).map(([, b]) => Math.min(b, t)), -Infinity);
}

function strip(t) {
  const now = visibleNow();
  const here = new Set([...now.cam1, ...now.cam2].filter(id => view.persons[id].n !== null));
  const gone = Object.keys(numbered)
    .filter(id => !here.has(id) && numbered[id].some(([a, b]) => b < t && b > t - 240))
    .map(id => [id, lastBefore(id, t)]).sort((a, b) => b[1] - a[1]).slice(0, 10);
  $('#now').replaceChildren(...[...here].map(id => card(id, 'с ' + clockOf(view.persons[id].first).slice(0, 5),
    () => { if (sel && !sel.empty) same(sel.pid, id); })));
  $('#gone').replaceChildren(...gone.map(([id, left]) => card(id, 'ушёл ' + lasting(t - left) + ' назад',
    () => { if (sel && !sel.empty) same(sel.pid, id); else seek(Math.max(0, left - 2)); })));
}

/* ---------- the menu: everything that can be wrong, next to the person ---------- */

function candidates(t, except) {
  // who this might really be: people who left a moment ago, people the other camera sees now,
  // people who came in just before. Most recent first.
  const now = visibleNow(), out = new Map();
  const cam = sel ? sel.cam : 'cam1', other = cam === 'cam1' ? 'cam2' : 'cam1';
  for (const id of now[other]) if (id !== except) out.set(id, [0, 'на другой камере']);
  for (const [id, list] of Object.entries(everyone)) {
    if (id === except || out.has(id) || now[cam].has(id)) continue;
    const p = view.persons[id];
    if (p.n === null && !list.some(([, b]) => b > t - 60 && b < t)) continue;
    const left = lastBefore(id, t);
    if (left > t - 240 && left < t) out.set(id, [t - left, 'ушёл ' + lasting(t - left) + ' назад']);
    else if (p.first > t - 60 && p.first <= t && !now[cam].has(id)) out.set(id, [t - p.first, 'пришёл ' + lasting(t - p.first) + ' назад']);
  }
  return [...out.entries()].sort((a, b) => a[1][0] - b[1][0]).slice(0, 10);
}

function menuParts() {
  menu.hidden = false;
  menu.replaceChildren();
  const add = (html, cls) => { const d = document.createElement('div'); if (cls) d.className = cls; d.innerHTML = html; menu.append(d); return d; };
  const act = (label, key, fn, cls) => {
    const b = document.createElement('button');
    b.className = cls || '';
    b.innerHTML = `${esc(label)}${key ? ` <kbd>${key}</kbd>` : ''}`;
    b.addEventListener('click', event => { event.stopPropagation(); fn(); });
    return b;
  };
  return {add, act};
}

function placeMenu(event) {
  // beside the person, never on him: right of what is being outlined if it fits, else left
  const box = $('.screens').getBoundingClientRect();
  if (event) menuAt = [event.clientX - box.left, event.clientY - box.top];
  const [x, y] = menuAt || [box.width / 2, 40];
  let [lo, hi] = [x, x];
  if (draft && draft.rings) {
    const canvas = overlays[draft.cam].getBoundingClientRect(), k = canvas.width / RAW;
    const [x1, , x2] = boxOf(draft.rings);
    lo = Math.min(lo, canvas.left - box.left + x1 * k);
    hi = Math.max(hi, canvas.left - box.left + x2 * k);
  }
  const w = menu.offsetWidth, h = menu.offsetHeight;
  const left = hi + 16 + w <= box.width ? hi + 16 : Math.max(4, lo - 16 - w);
  menu.style.left = Math.min(box.width - w - 4, left) + 'px';
  menu.style.top = Math.max(4, Math.min(box.height - h - 4, y - 20)) + 'px';
}

function openMenu(target, event) {
  if (target.empty) { startDraft(target.cam, target.x, target.y, event); return; }
  sel = target;
  if (playing) { pausedByMenu = true; setPlaying(false); }
  warm();
  const t = onScreen(target.cam);
  const {add, act} = menuParts();
  const p = part(target.pid), person = personOf(target.pid);
  if (p[4]) {
    add('Помечено: <b>не человек</b>.', 'title');
    const row = add('', 'acts');
    row.append(act('Это человек — вернуть', 'X', toggleFalse), act('Закрыть', 'Esc', closeMenu, 'quiet'));
  } else {
    const where = person ? `${clockOf(person.first).slice(0, 8)}–${clockOf(person.last).slice(0, 8)}` : '';
    add(`<span class="dot" style="background:${colour(person)}"></span><b>${esc(nameOf(person))}</b>`
      + (person && person.n !== null ? ` · ${KIND[person.kind]}` : '') + ` <span class="muted">${where}</span>`, 'title');
    const cands = candidates(t, personIdOf(target.pid));
    add('Кто это на самом деле? <span class="muted">Выберите или щёлкните его на видео</span>', 'q');
    const row = add('', 'cands');
    if (cands.length) row.append(...cands.map(([id, [, note]]) => card(id, note, () => same(target.pid, id))));
    else row.innerHTML = '<span class="muted">рядом по времени никого — щёлкните его на видео, если он виден</span>';
    const acts = add('', 'acts');
    acts.append(act('Дальше это другой человек', 'S', split),
                act('Это не человек', 'X', toggleFalse),
                act('Поправить обводку', 'E', () => startFix(target)),
                act(badNear(target.pid, t) ? 'Обводка нормальная' : 'Кривая, пусть ночь', 'B', badMask));
    const kinds = add('<span class="muted">Кто:</span>', 'kinds');
    for (const [kind, key] of [['customer', 'C'], ['staff', 'W'], ['passer', 'P']]) {
      kinds.append(act(KIND[kind], key, () => setKind(kind), person && person.kind === kind ? 'on' : ''));
    }
    add('', 'acts').append(act('Закрыть', 'Esc', closeMenu, 'quiet'));
  }
  placeMenu(event);
  info();
}

/* ---------- outlining by clicks: the model draws, the owner points ---------- */

function startDraft(cam, x, y, event) {
  // a click where nobody is outlined: the model outlines whoever stands there
  sel = null;
  if (playing) { pausedByMenu = true; setPlaying(false); }
  draft = {cam, at: onScreen(cam), points: [[Math.round(x), Math.round(y), 1]], rings: null, busy: false};
  placeMenu(event);
  runDraft();
}

function startFix(target) {
  // the machine's outline is wrong: start from it and let clicks reshape it
  const e = drawn[target.cam].find(d => d.pid === target.pid);
  if (!e) { notice('Этой обводки в кадре уже нет — остановите видео на ней', true); return; }
  const s = RAW / unit, track = target.pid.slice(0, target.pid.lastIndexOf('@'));
  sel = null;
  // the fix belongs to the frame the detector saw, which is the one under the outline now
  draft = {cam: target.cam, at: e.at, points: [], box: e.box.map(v => v * s), replaces: track,
           rings: e.rings.map(r => r.map(v => Math.round(v * s))), busy: false};
  showDraft();
}

function draftPoint(x, y, label) {
  draft.points.push([Math.round(x), Math.round(y), label]);
  runDraft();
}

async function runDraft() {
  if (!draft) return;
  const mine = ++draftSeq;
  draft.busy = true; draft.error = null;
  showDraft();
  try {
    const body = {cam: draft.cam, at: draft.at, points: draft.points};
    if (draft.box) body.box = draft.box;
    const got = await api(`/api/movie/${day}/segment`, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                                         body: JSON.stringify(body)});
    if (!draft || mine !== draftSeq) return;
    draft.rings = got.rings.length ? got.rings : null;
    draft.frame = got.frame; draft.ms = got.ms; draft.model = got.model;
    if (!got.rings.length) draft.error = 'Здесь модель никого не видит — щёлкните точнее по человеку';
  } catch (error) {
    if (draft && mine === draftSeq) draft.error = error.message;
  } finally {
    if (draft && mine === draftSeq) { draft.busy = false; showDraft(); }
  }
}

function showDraft() {
  if (!draft) return;
  const {add, act} = menuParts();
  add(`<b>${draft.replaces ? 'Поправить обводку' : 'Обвести человека'}</b>`, 'title');
  const state = draft.busy ? 'Обвожу…'
    : draft.error ? `<span style="color:#ffb3ab">${esc(draft.error)}</span>`
    : draft.rings ? `Готово${draft.ms ? ` за ${(draft.ms.total / 1000).toFixed(1)} с` : ''}. Если что-то не так — поправьте щелчками.`
    : 'Щёлкните по нему на кадре.';
  add(state, 'q');
  add('<span class="muted">Щелчок по человеку — добавить часть. <kbd>Shift</kbd>+щелчок или правая кнопка — убрать лишнее.</span>');
  const acts = add('', 'acts');
  const save = act(draft.replaces ? 'Сохранить на этом кадре' : 'Сохранить и вести дальше', 'Enter', saveDraft, 'on');
  save.disabled = !draft.rings || draft.busy;
  acts.append(save);
  if (draft.points.length) acts.append(act('Заново', '', () => { draft.points = []; draft.rings = draft.replaces ? draft.rings : null; runDraft(); }));
  if (!draft.replaces) acts.append(act('Не выходит — отметить для ночи', 'M', () => missed({cam: draft.cam, x: draft.points[0][0], y: draft.points[0][1]})));
  acts.append(act('Отмена', 'Esc', cancelDraft, 'quiet'));
  placeMenu();
}

function cancelDraft() {
  draft = null;
  closeMenu();
}

async function saveDraft() {
  if (!draft || !draft.rings || draft.busy) return;
  const d = draft;
  try {
    const made = await api(`/api/movie/${day}/draw`, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cam: d.cam, at: d.at, rings: d.rings, points: d.points, replaces: d.replaces || null, frame: d.frame})});
    lastDraw = made.id; lastAction = 'draw';
    draft = null;
    adopt(await api(`/api/movie/${day}`));
    if (d.replaces) {
      notice('Обводка исправлена на этом кадре. Z — отменить.');
      closeMenu();
    } else {
      notice('Обведён. Машина сама ведёт его по фильму вперёд и назад — полоса внизу растёт. Z — отменить.');
      openMenu({pid: made.track + '@', cam: d.cam});              // who is he? the usual question, now answerable
    }
  } catch (error) {
    notice(error.message, true);
  }
}

function warm() {
  // the heavy part of outlining is reading the frame: do it the moment the film stops
  if (!view || !videos.cam1.src) return;
  const t = videos.cam1.currentTime;
  if (lastWarm !== null && Math.abs(lastWarm - t) < 0.05) return;
  lastWarm = t;
  fetch(`/api/movie/${day}/warm`, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                   body: JSON.stringify({at: t})}).catch(() => {});
}

function closeMenu(resume = true) {
  menu.hidden = true;
  sel = null;
  draft = null;
  if (pausedByMenu) { pausedByMenu = false; if (resume) setPlaying(true); }
  info();
}

/* ---------- playing ---------- */

function effectiveRate(t) {
  if (review) return 1;
  if (slowDoubts && rate > SLOW) {
    for (const d of view.doubts) if (t >= d - 1.0 && t <= d + 2.0) return SLOW;
  }
  return rate;
}

function activeAt(t) {
  for (const [a, b] of view.activity) { if (t < a) return [false, a]; if (t <= b) return [true, b]; }
  return [false, null];
}

function seek(t) {
  closeSpan();
  t = Math.max(0, Math.min(view.duration - 0.1, t));
  for (const cam of CAMS) if (videos[cam].src) videos[cam].currentTime = t;
  lastT = null;
  for (const clip of clipsNear(t, 120)) loadClip(clip);
}

function setPlaying(on) {
  playing = on;
  for (const cam of CAMS) {
    const v = videos[cam];
    if (!v.src) continue;
    if (on) v.play().catch(() => {}); else v.pause();
  }
  $('#play').textContent = on ? '❚❚ Пауза · пробел' : '▶ Пуск · пробел';
  if (!on) { closeSpan(); warm(); }
}

function setRate(value) {
  rate = value;
  for (const b of document.querySelectorAll('#rates button')) b.classList.toggle('on', +b.dataset.rate === rate);
}

function closeSpan() {
  if (span && span[1] - span[0] > 0.3) pending.push([+span[0].toFixed(2), +span[1].toFixed(2), span[2]]);
  span = null;
}

async function flush(force) {
  closeSpan();
  if (!pending.length || (!force && Date.now() - lastFlush < FLUSH)) return;
  const spans = pending.splice(0);
  lastFlush = Date.now();
  try {
    view.watched = await api(`/api/movie/${day}/watched`, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                                            body: JSON.stringify({spans})});
    progress();
    drawDay();
  } catch (error) {
    pending.unshift(...spans);
  }
}

let lastSlow = 0;
function tick() {
  requestAnimationFrame(tick);
  if (!view || !videos.cam1.src) return;
  const master = videos.cam1, t = master.currentTime;
  if (playing && master.paused && !document.hidden && master.readyState >= 2) {
    master.play().catch(() => {});      // the browser pauses a muted video in a background tab
  }
  if (playing) {
    if (review && t >= review.until) { setRate(review.rate); review = null; }
    if (skipEmpty && !review) {
      const [on, next] = activeAt(t);
      if (!on) {
        if (next === null) { setPlaying(false); notice('Дальше в этот день никого нет'); }
        else if (next - t > 1.5) { seek(next - 1); return; }
      }
    }
    const eff = effectiveRate(t);
    if (Math.abs(master.playbackRate - eff) > 1e-3) master.playbackRate = eff;
    const follower = videos.cam2;
    if (follower.src) {
      const drift = t - follower.currentTime;
      if (Math.abs(drift) > 1.5) follower.currentTime = t;
      else follower.playbackRate = Math.max(0.25, eff * (1 + Math.max(-0.25, Math.min(0.25, drift * 0.8))));
      if (follower.paused && !master.paused) follower.play().catch(() => {});
    }
    if (lastT !== null && t > lastT && t - lastT < 3 * Math.max(eff, 1)) {
      if (!span) span = [lastT, t, eff]; else { span[1] = t; span[2] = Math.max(span[2], eff); }
    } else closeSpan();
    lastT = t;
    if (performance.now() - lastSlow > 250) {
      lastSlow = performance.now();
      $('#rateNow').textContent = eff !== rate ? `${eff}× — машина здесь не уверена` : review ? 'показываю правку на 1×' : '';
    }
    if (Date.now() - lastFlush > FLUSH) flush();
    for (const clip of clipsNear(t, 60 * Math.max(eff, 1))) loadClip(clip);
  } else {
    const follower = videos.cam2;
    if (follower.src && Math.abs(follower.currentTime - t) > 0.05 && !follower.seeking) follower.currentTime = t;
  }
  $('#clock').textContent = clockOf(t);
  evict(t);
  render();
  drawCursor(t);
  if (performance.now() - lastCards > 250) { lastCards = performance.now(); strip(t); drawLane(t); }
}
let lastCards = 0;

/* ---------- pointing ---------- */

function filmPoint(cam, event) {
  const rect = overlays[cam].getBoundingClientRect();
  const fx = (event.clientX - rect.left) / rect.width, fy = (event.clientY - rect.top) / rect.height;
  return {u: [fx * unit, fy * unit * 9 / 16], raw: [fx * RAW, fy * RAW * 9 / 16]};
}

function hit(cam, x, y) {
  let best = null;
  for (const e of drawn[cam]) {
    if (inside(e.rings, x, y)) return e;
    const [x1, y1, x2, y2] = e.box, pad = unit / 120;
    if (x >= x1 - pad && x <= x2 + pad && y >= y1 - pad && y <= y2 + pad) {
      const area = (x2 - x1) * (y2 - y1);
      if (!best || area < best.area) best = {...e, area};
    }
  }
  return best;
}

function info() {
  const box = $('#info'), target = hover;
  if (!view || !target || sel) { box.innerHTML = '&nbsp;'; return; }
  const p = part(target.pid), person = personOf(target.pid);
  box.innerHTML = p && p[4] ? '<b>не человек</b> — щёлкните, чтобы вернуть'
    : `<b style="color:${colour(person)}">${esc(nameOf(person))}</b>`
      + (person ? ` · ${KIND[person.kind]} · ${clockOf(person.first)}–${clockOf(person.last)} (${lasting(person.last - person.first)})` : '')
      + ' · щёлкните — что не так';
}

for (const cam of CAMS) {
  const canvas = overlays[cam];
  canvas.addEventListener('mousemove', event => {
    if (!view) return;
    const {u, raw} = filmPoint(cam, event);
    mouse = {cam, x: raw[0], y: raw[1]};
    const found = hit(cam, u[0], u[1]);
    const before = hover && hover.pid;
    hover = found ? {pid: found.pid, cam} : null;
    canvas.style.cursor = found ? 'pointer' : 'crosshair';
    if ((hover && hover.pid) !== before) info();
  });
  canvas.addEventListener('mouseleave', () => { mouse = null; hover = null; info(); });
  canvas.addEventListener('contextmenu', event => {
    if (!draft || draft.cam !== cam) return;
    event.preventDefault();
    const {raw} = filmPoint(cam, event);
    draftPoint(raw[0], raw[1], 0);
  });
  canvas.addEventListener('click', event => {
    if (!view) return;
    const {u, raw} = filmPoint(cam, event);
    if (draft) {
      if (draft.cam === cam) draftPoint(raw[0], raw[1], event.shiftKey || event.altKey ? 0 : 1);
      else notice('Сначала закончите обводку на другой камере: Enter — сохранить, Esc — отмена', true);
      return;
    }
    const found = hit(cam, u[0], u[1]);
    if (sel && !sel.empty && found && found.pid !== sel.pid) {       // the menu is open: "it is him"
      const target = personIdOf(found.pid) || found.pid;
      if (target !== personIdOf(sel.pid)) { same(sel.pid, target); return; }
    }
    if (found) openMenu({pid: found.pid, cam}, event);
    else openMenu({empty: true, cam, x: raw[0], y: raw[1]}, event);
  });
}
document.addEventListener('click', event => {
  if (!menu.hidden && !draft && !menu.contains(event.target) && !event.target.closest('.screen,.card')) closeMenu();
});

/* ---------- corrections ---------- */

async function edit(body) {
  if (busy) return null;
  busy = true;
  try {
    const next = await api(`/api/movie/${day}`, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                                  body: JSON.stringify({...body, revision: view.revision})});
    const did = next.did;
    adopt(next);
    lastAction = 'edit';
    return did;
  } catch (error) {
    notice(error.message, true);
    if (error.status === 409) {
      try { adopt(await api(`/api/movie/${day}`)); } catch (nothing) { /* keep the old view */ }
    }
    return null;
  } finally {
    busy = false;
  }
}

const targetOf = () => (sel && !sel.empty ? sel : hover);

async function same(pid, target) {
  const a = personOf(pid), b = view.persons[target] || personOf(target);
  const did = await edit({action: 'same', part: pid, target});
  if (did) notice(`Склеено: ${nameOf(a)} → ${nameOf(b)}. Z — отменить.`);
  closeMenu();
}

async function split() {
  const target = targetOf();
  if (!target) { notice('Наведите мышь на человека и нажмите S', true); return; }
  const t = onScreen(target.cam), was = playing || pausedByMenu ? effectiveRate(t) : 1;
  const running = playing || pausedByMenu;
  const did = await edit({action: 'split', part: target.pid, at: t, rate: running ? was : 1});
  closeMenu(false);
  if (!did) { if (running) setPlaying(true); return; }
  const person = personOf(did.part);
  notice(`С ${clockOf(did.at)} это другой человек — ${nameOf(person)}. Если это кто-то из ушедших — щёлкните его и выберите. Z — отменить.`);
  if (running && was > 1.5) {            // show the cut at normal speed, then carry on
    review = {until: did.at + 1.5, rate};
    seek(did.at - 1.5);
    setRate(1);
  }
  if (running) setPlaying(true);
}

async function toggleFalse() {
  const target = targetOf();
  if (!target) { notice('Наведите мышь на то, что обведено зря, и нажмите X', true); return; }
  const did = await edit({action: 'false', part: target.pid});
  if (did) notice(did.now ? 'Помечено: не человек. X ещё раз — вернуть.' : 'Снова человек.');
  closeMenu();
}

async function setKind(kind) {
  const target = targetOf();
  if (!target || !personOf(target.pid)) { notice('Наведите мышь на человека', true); return; }
  const did = await edit({action: 'kind', part: target.pid, kind});
  if (did) notice(`Это ${KIND[did.kind]} — на всего человека.`);
  closeMenu();
}

async function badMask() {
  const target = targetOf();
  if (!target) { notice('Наведите мышь на кривую обводку и нажмите B', true); return; }
  const did = await edit({action: 'badmask', part: target.pid, at: onScreen(target.cam)});
  if (did) notice(did.now ? 'Отмечено: здесь обводка кривая — эти кадры не пойдут в обучение.' : 'Отметка кривой обводки снята.');
  closeMenu();
}

async function missed(where) {
  const spot = where || (mouse ? {cam: mouse.cam, x: mouse.x, y: mouse.y} : null);
  if (!spot) { notice('Наведите мышь на человека без обводки и нажмите M', true); return; }
  const did = await edit({action: 'missed', cam: spot.cam, at: onScreen(spot.cam), x: spot.x, y: spot.y});
  if (did) notice(did.now ? 'Отмечен человек без обводки: машина поищет его здесь.' : 'Отметка убрана.');
  closeMenu();
}

async function undo() {
  if (lastAction === 'draw' && lastDraw) {
    try {
      await api(`/api/movie/${day}/undraw`, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                            body: JSON.stringify({id: lastDraw})});
      lastDraw = null; lastAction = null;
      adopt(await api(`/api/movie/${day}`));
      notice('Обводка отменена.');
    } catch (error) { notice(error.message, true); }
    return;
  }
  const did = await edit({action: 'undo'});
  if (did) notice('Отменено.');
}

/* ---------- keys, buttons, loading ---------- */

// The physical key, whatever the layout. Some remote desktops and automation send only the
// character, so fall back to it -- Russian letters included, they sit on the same keys.
const BY_CHAR = {' ': 'Space', '[': 'BracketLeft', 'х': 'BracketLeft', ']': 'BracketRight', 'ъ': 'BracketRight',
  ',': 'Comma', 'б': 'Comma', '.': 'Period', 'ю': 'Period', '/': 'Slash', '?': 'Slash',
  s: 'KeyS', 'ы': 'KeyS', x: 'KeyX', 'ч': 'KeyX', w: 'KeyW', 'ц': 'KeyW', m: 'KeyM', 'ь': 'KeyM',
  z: 'KeyZ', 'я': 'KeyZ', g: 'KeyG', 'п': 'KeyG', b: 'KeyB', 'и': 'KeyB', p: 'KeyP', 'з': 'KeyP', c: 'KeyC', 'с': 'KeyC',
  e: 'KeyE', 'у': 'KeyE', enter: 'Enter'};
function codeOf(event) {
  if (event.code) return event.code;
  const key = (event.key || '').toLowerCase();
  if (/^[0-9]$/.test(key)) return 'Digit' + key;
  if (/^(arrow(left|right|up|down)|escape)$/.test(key)) return event.key;
  return BY_CHAR[key] || '';
}

document.addEventListener('keydown', event => {
  const typing = event.target.closest && event.target.closest('select,input,textarea');
  if (!view || typing || event.metaKey || event.ctrlKey || event.altKey) return;
  const code = codeOf(event);
  if ($('#helpBox').open && code !== 'Escape') return;
  const t = videos.cam1.currentTime;
  const act = {
    Space: () => { if (draft) return; if (!menu.hidden) closeMenu(); else setPlaying(!playing); },
    Enter: () => { if (draft) saveDraft(); },
    NumpadEnter: () => { if (draft) saveDraft(); },
    KeyE: () => { const target = targetOf(); if (target && !draft) startFix(target); },
    BracketLeft: () => setRate(RATES[Math.max(0, RATES.indexOf(rate) - 1)]),
    BracketRight: () => setRate(RATES[Math.min(RATES.length - 1, RATES.indexOf(rate) + 1)]),
    ArrowLeft: () => seek(t - (event.shiftKey ? 10 : 2)),
    ArrowRight: () => seek(t + (event.shiftKey ? 10 : 2)),
    Comma: () => { setPlaying(false); seek(t - view.tick); },
    Period: () => { setPlaying(false); seek(t + view.tick); },
    KeyS: split, KeyX: toggleFalse, KeyB: badMask, KeyZ: undo,
    KeyW: () => setKind('staff'), KeyP: () => setKind('passer'), KeyC: () => setKind('customer'),
    KeyM: () => missed(draft && !draft.replaces && draft.points.length ? {cam: draft.cam, x: draft.points[0][0], y: draft.points[0][1]} : null),
    KeyG: () => { passers = !passers; $('#passers').checked = passers; },
    Escape: () => { if ($('#helpBox').open) $('#helpBox').close(); else if (draft) cancelDraft(); else closeMenu(); },
    Slash: () => $('#helpBox').showModal(),
  }[code];
  const digit = /^Digit([1-6])$/.exec(code);
  if (digit) { setRate(RATES[+digit[1] - 1]); event.preventDefault(); return; }
  if (act) { event.preventDefault(); act(); }
});

$('#play').addEventListener('click', () => setPlaying(!playing));
$('#undo').addEventListener('click', undo);
$('#help').addEventListener('click', () => $('#helpBox').showModal());
for (const close of document.querySelectorAll('[data-close]')) close.addEventListener('click', () => close.closest('dialog').close());
$('#skipEmpty').addEventListener('change', e => { skipEmpty = e.target.checked; });
$('#slowDoubts').addEventListener('change', e => { slowDoubts = e.target.checked; });
$('#passers').addEventListener('change', e => { passers = e.target.checked; });
$('#rates').replaceChildren(...RATES.map(r => {
  const b = document.createElement('button');
  b.textContent = r + '×'; b.dataset.rate = r;
  b.addEventListener('click', () => setRate(r));
  return b;
}));
setRate(rate);

function pointTo(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX - rect.left) * (canvas.width / rect.width), (event.clientY - rect.top) * (canvas.height / rect.height)];
}
let dragging = false;
const dayline = $('#dayline');
dayline.addEventListener('mousedown', event => { dragging = true; closeMenu(false); seek(pointTo(dayline, event)[0] / dayline.width * view.duration); });
window.addEventListener('mousemove', event => { if (dragging) seek(pointTo(dayline, event)[0] / dayline.width * view.duration); });
window.addEventListener('mouseup', () => { dragging = false; });
$('#lane').addEventListener('click', event => {
  const [x, y] = pointTo($('#lane'), event);
  const bar = laneRows.find(r => x >= r.x1 - 3 && x <= r.x2 + 3 && y >= r.y && y <= r.y + r.h);
  if (bar) { closeMenu(false); seek(Math.max(0, bar.at - 1.5)); }
});
window.addEventListener('resize', drawDay);
window.addEventListener('pagehide', () => {
  closeSpan();
  if (pending.length) navigator.sendBeacon(`/api/movie/${day}/watched`,
    new Blob([JSON.stringify({spans: pending.splice(0)})], {type: 'application/json'}));
});
document.addEventListener('visibilitychange', () => { if (document.hidden) flush(true); });

async function open(which) {
  await flush(true);
  setPlaying(false);
  closeMenu(false);
  day = which; clips.clear(); thumbs.clear(); hover = null;
  notice('Загружаю день…');
  try {
    adopt(await api(`/api/movie/${day}`));
  } catch (error) {
    notice(error.message, true);
    return;
  }
  unit = view.unit || unit;
  for (const cam of CAMS) {
    const v = videos[cam];
    if (view.videos[cam].ready) v.src = view.videos[cam].url; else v.removeAttribute('src');
  }
  const first = view.activity.length ? Math.max(0, view.activity[0][0] - 1) : 0;
  let saved = NaN;
  try { saved = +(localStorage.getItem('movie-at-' + day) || NaN); } catch (nothing) { /* private mode */ }
  seek(Number.isFinite(saved) ? saved : first);
  warm();          // the outlining model takes half a minute to wake up: let it start now
  notice(view.videos.cam1.ready ? 'Пробел — пуск. Скорость — цифрами 1…6.' : 'Фильм этого дня ещё собирается.');
}
setInterval(async () => {
  if (!view || busy || draft || !(view.drawn || []).some(d => d.carry && ['queued', 'running'].includes(d.carry.state))) return;
  try { adopt(await api(`/api/movie/${day}`)); } catch (nothing) { /* next time */ }
}, 3000);
setInterval(() => { try { if (view && videos.cam1.src) localStorage.setItem('movie-at-' + day, videos.cam1.currentTime.toFixed(1)); } catch (nothing) { /* private mode */ } }, 3000);

(async () => {
  try {
    const {days} = await api('/api/movie');
    const select = $('#day');
    select.replaceChildren(...days.map(d => new Option(`${d.slice(6)}.${d.slice(4, 6)}.${d.slice(0, 4)}`, d)));
    const wanted = new URLSearchParams(location.search).get('day');
    const chosen = days.includes(wanted) ? wanted : days[0];
    if (!chosen) { notice('Ни один день ещё не собран в фильм.'); return; }
    select.value = chosen;
    select.addEventListener('change', () => open(select.value));
    await open(chosen);
  } catch (error) {
    notice(error.message, true);
  }
  requestAnimationFrame(tick);
})();
