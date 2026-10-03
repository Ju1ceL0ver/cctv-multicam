'use strict';
/* Labelling a whole working day by watching it.
 *
 * Nothing here is drawn by hand: the detector outlined every person on every third frame
 * and the tracker chained those outlines into tracks. A click selects a whole track, a
 * digit says who it is, and the answer covers every frame of it. The two cameras run on one
 * clock -- camera 1's -- because camera 2 sees the same moment `offset` seconds later.
 */
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const hhmmss = new Intl.DateTimeFormat('ru-RU', {hour:'2-digit',minute:'2-digit',second:'2-digit',timeZone:'Asia/Bangkok'});
const hhmm = new Intl.DateTimeFormat('ru-RU', {hour:'2-digit',minute:'2-digit',timeZone:'Asia/Bangkok'});
const clock = t => hhmmss.format(new Date(t * 1000));
const PALETTE = ['#7ae1b3','#f5a3c7','#9ad0ff','#ffd479','#c3a6ff','#8fe388','#ff9f80','#7fd4d0','#e0b0ff','#b8d96b','#ffa0a0','#6fb3ff'];
const SPECIAL = {_passer: {name: 'прохожий', colour: '#7d8794'}, _false: {name: 'не человек', colour: '#8a3b3b'}};
const PLAIN = '#e7eef5';            // a track nobody has answered about yet
const SHORT = 10;                   // seconds: the passer-by sweep offers these
const RAW_WIDTH = 2560;
const RATES = [0.5, 1, 2, 4, 8, 16];

let day = null, data = null, store = null, byKey = {}, stale = new Set();
let T = 0, playing = false, rate = 1, sel = null, longOnly = false, blocked = false, loading = false;
const boxes = {}, asked = new Set(), spans = new Map();
const videos = {cam1: $('#v1'), cam2: $('#v2')};
const overlays = {cam1: $('#o1'), cam2: $('#o2')};
const missing = {cam1: $('#n1'), cam2: $('#n2')};
const outbox = [];
let sending = false, tape = null, tapeKey = '';

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

/* ---------- the day's model ---------- */

const shift = cam => cam === 'cam2' ? data.offset : 0;
const trackOf = key => byKey[key];
const record = key => store.tracks[key] || {cuts: [], who: [null]};
const parts = key => record(key).who.length;

function partAt(key, t) {
  const cuts = record(key).cuts;
  let i = 0;
  while (i < cuts.length && t >= cuts[i]) i += 1;
  return i;
}

function partRange(key, part) {
  const track = trackOf(key), cuts = record(key).cuts;
  return [part ? cuts[part - 1] : track.first, part < cuts.length ? cuts[part] : track.last];
}

function nameOf(who) {
  if (!who) return '?';
  if (SPECIAL[who]) return SPECIAL[who].name;
  const person = store.people[who];
  return person ? person.name : who;
}

function colourOf(who) {
  if (!who) return PLAIN;
  if (SPECIAL[who]) return SPECIAL[who].colour;
  const person = store.people[who];
  return person ? PALETTE[person.color % PALETTE.length] : PLAIN;
}

/** Where each person was seen, from the owner's own answers -- used to offer the people
 *  who are around right now first, and to draw the day's line. */
function recount() {
  spans.clear();
  for (const [key, item] of Object.entries(store.tracks)) {
    if (!trackOf(key)) continue;
    item.who.forEach((who, part) => {
      if (!who) return;
      const [from, to] = partRange(key, part);
      if (!spans.has(who)) spans.set(who, []);
      spans.get(who).push([from, to]);
    });
  }
  tapeKey = '';
}

const nearby = who => (spans.get(who) || []).some(([a, b]) => b > T - 120 && a < T + 120);

function roster() {
  const ids = Object.keys(store.people);
  const last = who => Math.max(...(spans.get(who) || [[-1e12]]).map(s => s[1]));
  return ids.sort((a, b) => (nearby(b) - nearby(a)) || (last(b) - last(a)));
}

/* ---------- video ---------- */

const segmentAt = (cam, wall) =>
  (data.segments[cam] || []).find(s => wall >= s.start && wall < s.start + s.duration);

function ensure(cam) {
  const video = videos[cam], wall = T + shift(cam), segment = segmentAt(cam, wall);
  if (!segment || !segment.ready) {
    missing[cam].textContent = !segment ? 'В это время записи нет'
      : segment.broken ? 'Запись этого куска повреждена — файл не дописан, кадров нет'
      : 'Облегчённая копия этого куска ещё готовится — вернитесь сюда позже';
    if (video.dataset.seg) { video.removeAttribute('src'); video.load(); video.dataset.seg = ''; }
    return;
  }
  missing[cam].textContent = '';
  const want = Math.max(0, wall - segment.start - segment.origin);
  if (video.dataset.seg !== segment.name) {
    video.dataset.seg = segment.name;
    video.pending = want;
    video.src = segment.url;
    video.load();
    return;
  }
  if (video.readyState < 1) { video.pending = want; return; }
  if (Math.abs(video.currentTime - want) > (playing ? 0.35 : 0.06)) video.currentTime = want;
  if (video.playbackRate !== rate) video.playbackRate = rate;
  if (playing && video.paused) video.play().catch(() => {});
  if (!playing && !video.paused) video.pause();
}

for (const cam of ['cam1', 'cam2']) {
  const video = videos[cam];
  video.addEventListener('loadedmetadata', () => {
    if (video.pending != null) { video.currentTime = video.pending; video.pending = null; }
    if (playing) video.play().catch(() => {});
  });
  video.addEventListener('ended', () => {
    const segment = segmentAt(cam, T + shift(cam));
    if (playing && segment) T = segment.start + segment.duration + 0.05 - shift(cam);
  });
}

function driving() {
  for (const cam of ['cam1', 'cam2']) {
    const video = videos[cam];
    if (video.dataset.seg && video.readyState >= 2 && !video.paused && !video.ended) return cam;
  }
  return null;
}

/* ---------- boxes ---------- */

function want(clip) {
  if (boxes[clip] || asked.has(clip)) return;
  asked.add(clip);
  api(`/api/dayplayer/${day}/boxes?clip=${encodeURIComponent(clip)}`)
    .then(got => { boxes[clip] = got; })
    .catch(error => { asked.delete(clip); notice(error.message, true); });
}

function boxAt(track, t) {
  const rows = (boxes[track.clip] || {})[track.key];
  if (!rows || !rows.length) return null;
  let low = 0, high = rows.length - 1;
  while (low < high) { const mid = (low + high + 1) >> 1; if (rows[mid][0] <= t) low = mid; else high = mid - 1; }
  const before = rows[low], after = rows[Math.min(low + 1, rows.length - 1)];
  if (t < before[0] - 0.4 || t > after[0] + 0.4) return null;
  if (after === before || after[0] - before[0] > 1.2) return before.slice(1);
  const k = Math.min(1, Math.max(0, (t - before[0]) / (after[0] - before[0])));
  return [1, 2, 3, 4].map(i => before[i] + (after[i] - before[i]) * k);
}

const showing = cam => data.tracks.filter(t => t.cam === cam && T >= t.first - 0.3 && T <= t.last + 0.3);

function overlay(cam) {
  const canvas = overlays[cam], width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const paint = canvas.getContext('2d');
  paint.clearRect(0, 0, width, height);
  if (!videos[cam].dataset.seg) return;
  const scale = width / RAW_WIDTH;
  for (const track of showing(cam)) {
    want(track.clip);
    const box = boxAt(track, T);
    if (!box) continue;
    const part = partAt(track.key, T), who = record(track.key).who[part];
    const chosen = sel && sel.key === track.key && sel.part === part;
    const colour = stale.has(track.key) ? '#ff8f86' : colourOf(who);
    const x = box[0] * scale, y = box[1] * scale, w = (box[2] - box[0]) * scale, h = (box[3] - box[1]) * scale;
    paint.setLineDash(who ? [] : [7, 5]);
    paint.lineWidth = chosen ? 4 : 2;
    paint.strokeStyle = colour;
    paint.strokeRect(x, y, w, h);
    paint.setLineDash([]);
    const text = stale.has(track.key) ? 'пересчитан ночью' : nameOf(who);
    paint.font = `${chosen ? 700 : 500} 13px system-ui,sans-serif`;
    const pad = 4, tw = paint.measureText(text).width + pad * 2;
    paint.fillStyle = colour;
    paint.fillRect(x, Math.max(0, y - 18), tw, 17);
    paint.fillStyle = '#0d1116';
    paint.fillText(text, x + pad, Math.max(12, y - 5));
  }
}

function pick(cam, event) {
  const canvas = overlays[cam], rect = canvas.getBoundingClientRect();
  const scale = canvas.clientWidth / RAW_WIDTH;
  const px = (event.clientX - rect.left) / scale, py = (event.clientY - rect.top) / scale;
  let best = null, area = Infinity;
  for (const track of showing(cam)) {
    const box = boxAt(track, T);
    if (!box || px < box[0] || px > box[2] || py < box[1] || py > box[3]) continue;
    const size = (box[2] - box[0]) * (box[3] - box[1]);
    if (size < area) { area = size; best = track; }
  }
  if (best) select(best.key, T);
  else { sel = null; paintSelected(); }
}

/* ---------- selection and moving through the day ---------- */

function select(key, at) {
  sel = {key, part: partAt(key, at)};
  paintSelected();
  paintRoster();
}

function seek(to) {
  T = Math.max(data.windows[0].start - 1, Math.min(to, data.windows[data.windows.length - 1].end + 1));
  for (const window of data.windows) if (T >= window.start - 5 && T <= window.end + 5) want(window.clip);
}

function unanswered(track) {
  if (longOnly && track.last - track.first < SHORT) return false;
  return record(track.key).who.some(who => !who);
}

/** The main loop of the day: the next track nobody has named yet, in the order people
 *  appear. Dead time is never watched -- the tool jumps over it. */
function jump(direction) {
  const list = data.tracks.filter(unanswered).sort((a, b) =>
    (b.floor > 0) - (a.floor > 0) || (b.last - b.first) - (a.last - a.first) || a.first - b.first);
  if (!list.length) { notice('Все треки разобраны. День готов.'); return; }
  const at = sel ? list.findIndex(t => t.key === sel.key) : -1;
  const index = ((at < 0 ? (direction > 0 ? -1 : 0) : at) + direction + list.length) % list.length;
  const track = list[index];
  playing = false;
  seek(track.first + 0.3);
  const part = record(track.key).who.findIndex(who => !who);
  sel = {key: track.key, part: Math.max(0, part)};
  paintSelected();
  paintRoster();
  paintPlay();
}

/* ---------- answers ---------- */

function send(job) { outbox.push(job); pump(); }

async function pump() {
  if (sending || blocked || !outbox.length) return;
  sending = true;
  const job = outbox[0];
  try {
    const result = await api(`/api/dayplayer/${day}`, {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...job.body(), revision: store.revision})});
    outbox.shift();
    sending = false;
    store.revision = result.revision;
    if (result.people) store.people = result.people;
    if (result.next) store.next = result.next;
    if (result.store) { store = result.store; recount(); }
    for (const [key, item] of Object.entries(result.changed || {})) store.tracks[key] = item;
    if (job.done) job.done(result);
    $('#undo').disabled = !result.can_undo;
    paintAll();
    pump();
  } catch (error) {
    sending = false;
    blocked = true;
    if (job.revert) job.revert();
    recount();
    notice('Не сохранено: ' + error.message + ' Обновите страницу, этот ответ не записан.', true);
    paintAll();
  }
}

function answer(who) {
  if (blocked) { notice('Сначала обновите страницу — последний ответ не сохранился.', true); return; }
  if (!sel) { notice('Сначала выберите человека на видео или нажмите Tab.'); return; }
  if (stale.has(sel.key)) { notice('Этот трек пересчитан ночью; обновите страницу.', true); return; }
  const key = sel.key, part = sel.part;
  const track = trackOf(key);
  const before = store.tracks[key] ? JSON.parse(JSON.stringify(store.tracks[key])) : null;
  const item = store.tracks[key] || (store.tracks[key] = {cuts: [], who: [null], fp: track.fp});
  item.who[part] = who;
  send({body: () => ({action: 'assign', track: key, part, who}),
        revert: () => { if (before) store.tracks[key] = before; else delete store.tracks[key]; }});
  recount();
  if (who && !SPECIAL[who]) grid(who); else jump(1);
}

function cutHere() {
  if (!sel) { notice('Сначала выберите трек.'); return; }
  const track = trackOf(sel.key);
  if (!(T > track.first && T < track.last)) { notice('Резать можно только внутри выбранного трека.'); return; }
  const key = sel.key;
  const before = store.tracks[key] ? JSON.parse(JSON.stringify(store.tracks[key])) : null;
  const item = store.tracks[key] || (store.tracks[key] = {cuts: [], who: [null], fp: track.fp});
  const part = partAt(key, T);
  item.cuts.splice(part, 0, Math.round(T * 100) / 100);
  item.who.splice(part + 1, 0, item.who[part]);
  send({body: () => ({action: 'cut', track: key, at: Math.round(T * 100) / 100}),
        revert: () => { if (before) store.tracks[key] = before; else delete store.tracks[key]; }});
  sel = {key, part: part + 1};
  recount();
  notice('Трек разрезан. Теперь назовите вторую половину.');
  paintAll();
}

function newPerson() {
  if (!sel) { notice('Сначала выберите человека на видео.'); return; }
  $('#personName').value = '';
  $('#nameBox').showModal();
  setTimeout(() => $('#personName').focus(), 0);
}

// Enter in the name field submits the form explicitly: the implicit submission of a lone
// text field is not something to rely on inside a dialog.
$('#personName').addEventListener('keydown', event => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  $('#nameForm').requestSubmit($('#nameForm').querySelector('button[value=ok]'));
});

$('#nameForm').addEventListener('submit', async event => {
  if (event.submitter && event.submitter.value === 'cancel') return;
  const name = $('#personName').value.trim();
  const kind = $('#nameForm').querySelector('input[name=kind]:checked').value;
  if (!name) return;
  const keep = sel;
  try {
    const made = await api(`/api/dayplayer/${day}`, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'person', revision: store.revision, name, kind})});
    store.revision = made.revision;
    store.people = made.people;
    store.next = made.next;
    sel = keep;
    answer('P' + (made.next - 1));           // the id the server has just handed out
  } catch (error) {
    notice(error.message, true);
  }
});

function undo() {
  if (blocked || !data) return;
  send({body: () => ({action: 'undo'}), done: () => { sel = null; notice('Отменено'); }});
}

/* ---------- the passer-by sweep ---------- */

const card = (t, extra) =>
  `<button data-key="${esc(t.key)}" class="${extra || ''}">
     <img src="/portrait/${encodeURIComponent(t.clip)}/${t.piece}" alt="" loading="lazy">
     <span class="${t.cam}">${t.cam === 'cam1' ? 'камера 1' : 'камера 2'}</span> · ${esc(clock(t.first))}<br>
     ${Math.round(t.last - t.first)} с${t.distance != null ? ` · ${t.distance.toFixed(2)}` : ''}</button>`;

/** Everything short that never set foot on the shop floor is the mall gallery behind the
 *  glass -- 2305 of 2874 tracks on 17.09. One look at a sample, one key, and it is gone. */
const gallery = () => data.tracks.filter(t => t.floor === 0 && t.last - t.first < SHORT
  && record(t.key).who.every(who => !who));

function sweep() {
  if (!data.measured) { notice('Пол магазина ещё не посчитан для этого дня.', true); return; }
  const list = gallery();
  if (!list.length) { notice('Галерея уже снята.'); return; }
  const sample = list.slice().sort(() => Math.random() - 0.5).slice(0, 60);
  $('#sweepWhat').innerHTML = `Треков, которые <b>ни разу не ступали на пол магазина</b> и короче 10 секунд:
    <b>${list.length}</b> из ${data.tracks.length}. Это люди в галерее за стеклом. Ниже случайные ${sample.length} —
    посмотрите, все ли они за стеклом. Если да, одна кнопка снимает все ${list.length}.`;
  $('#sweepGrid').innerHTML = sample.map(t => card(t)).join('');
  $('#sweepGo').textContent = `Отметить прохожими: ${list.length}`;
  $('#sweepGo').onclick = () => {
    const before = {};
    for (const track of list) {
      before[track.key] = store.tracks[track.key] ? JSON.parse(JSON.stringify(store.tracks[track.key])) : null;
      const item = store.tracks[track.key] || (store.tracks[track.key] = {cuts: [], who: [null], fp: track.fp});
      item.who[0] = '_passer';
      item.rules = {...(item.rules || {}), 0: 'gallery_not_on_floor'};
    }
    send({body: () => ({action: 'sweep_gallery', seconds: SHORT}),
          revert: () => { for (const track of list) { if (before[track.key]) store.tracks[track.key] = before[track.key]; else delete store.tracks[track.key]; } }});
    recount();
    $('#sweepBox').close();
    notice(`Галерея снята: ${list.length} треков отмечены прохожими`);
    paintAll();
  };
  $('#sweepBox').showModal();
}

/** After naming somebody: every other track of the day that could be them, in one grid.
 *  The ticks are the machine's guess at the distance it was calibrated to; each tick is a
 *  photograph the owner looks at, and the impossible ones cannot be ticked at all. */
async function grid(person) {
  if (!store.people[person]) return;
  notice('Ищу остальные треки этого человека…');
  try {
    const found = await api(`/api/dayplayer/${day}/similar?anchor=${encodeURIComponent(person)}`);
    notice('');
    if (!found.ready) { notice('Похожесть для этого дня ещё не посчитана.', true); return; }
    const list = found.candidates.filter(c => c.distance <= 0.55 || c.suggested).slice(0, 80);
    if (!list.length) { notice('Других треков этого человека не нашлось.'); return; }
    $('#gridWho').textContent = `Это тоже ${store.people[person].name}?`;
    $('#gridGrid').innerHTML = list.map(c => card(c, (c.suggested ? 'keep ' : '') + (c.impossible ? 'impossible' : ''))).join('');
    $('#gridGrid').querySelectorAll('button').forEach(b => b.onclick = () => {
      if (!b.classList.contains('impossible')) b.classList.toggle('keep');
    });
    $('#gridGo').onclick = () => {
      const pairs = [...$('#gridGrid').querySelectorAll('button.keep')].map(b => [b.dataset.key, 0]);
      $('#gridBox').close();
      if (!pairs.length) return;
      const before = {};
      for (const [key] of pairs) {
        before[key] = store.tracks[key] ? JSON.parse(JSON.stringify(store.tracks[key])) : null;
        const item = store.tracks[key] || (store.tracks[key] = {cuts: [], who: [null], fp: trackOf(key).fp});
        item.who[0] = person;
      }
      send({body: () => ({action: 'assign_many', parts: pairs, who: person, rule: 'looks_the_same'}),
            revert: () => { for (const [key] of pairs) { if (before[key]) store.tracks[key] = before[key]; else delete store.tracks[key]; } }});
      recount();
      notice(`${store.people[person].name}: записано ещё ${pairs.length} треков`);
      paintAll();
      jump(1);
    };
    $('#gridBox').showModal();
  } catch (error) {
    notice(error.message, true);
  }
}

/* ---------- drawing ---------- */

function paintSelected() {
  const box = $('#selected');
  if (!sel || !trackOf(sel.key)) {
    box.innerHTML = 'Нажмите <kbd>Tab</kbd> — перейду к первому неразобранному человеку.';
    return;
  }
  const track = trackOf(sel.key), item = record(sel.key), [from, to] = partRange(sel.key, sel.part);
  box.innerHTML = `<b>${track.cam === 'cam1' ? 'Камера 1' : 'Камера 2'} · ${esc(clock(from))}–${esc(clock(to))}</b>
    · ${Math.round(to - from)} с${item.cuts.length ? ` · часть ${sel.part + 1} из ${item.who.length}` : ''}
    · ${track.floor == null ? '' : track.floor > 0 ? `в магазине ${Math.round(track.floor * 100)}% кадров` : '<b>за стеклом</b>'}<br>
    ${item.who[sel.part] ? `сейчас: <b>${esc(nameOf(item.who[sel.part]))}</b>` : 'кто это?'} · трек ${esc(track.key)}`;
}

function paintRoster() {
  const list = roster();
  $('#roster').innerHTML = list.map((id, i) => {
    const seen = (spans.get(id) || []).map(s => s[1]);
    const last = seen.length ? clock(Math.max(...seen)) : '—';
    const person = store.people[id];
    return `<li data-who="${id}" class="${nearby(id) ? 'near' : ''} ${sel && record(sel.key).who[sel.part] === id ? 'current' : ''}">
      <span class="key">${i < 9 ? i + 1 : '·'}</span>
      <span class="swatch" style="background:${colourOf(id)}"></span>
      <span class="who">${esc(person.name)}${person.kind === 'staff' ? ' · сотрудник' : ''}</span>
      <span class="when">${esc(last)}</span></li>`;
  }).join('') || '<li class="muted" style="cursor:default">Пока никого. Выберите человека на видео и нажмите <kbd>N</kbd>.</li>';
  $('#roster').querySelectorAll('[data-who]').forEach(li => li.onclick = () => answer(li.dataset.who));
}

function paintProgress() {
  let total = 0, done = 0, longTotal = 0, longDone = 0;
  for (const track of data.tracks) {
    const item = record(track.key);
    const big = track.last - track.first >= SHORT;
    item.who.forEach(who => {
      total += 1; if (who) done += 1;
      if (big) { longTotal += 1; if (who) longDone += 1; }
    });
  }
  const people = Object.values(store.people);
  $('#progress').textContent =
    `разобрано ${done} из ${total} треков · от 10 с: ${longDone} из ${longTotal} · людей ${people.filter(p => p.kind === 'customer').length}`
    + (people.some(p => p.kind === 'staff') ? ` · сотрудников ${people.filter(p => p.kind === 'staff').length}` : '')
    + (outbox.length ? ` · сохраняю ${outbox.length}` : '');
}

function paintPlay() {
  $('#play').textContent = playing ? '❚❚ Пауза · Space' : '▶ Пуск · Space';
  $('#rate').textContent = rate + '×';
}

function paintTimeline() {
  const canvas = $('#timeline'), width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; tapeKey = ''; }
  const from = data.windows[0].start, to = data.windows[data.windows.length - 1].end;
  const x = t => (t - from) / (to - from) * width;
  const key = `${width}x${height}|${store.revision}`;
  if (tapeKey !== key) {
    tape = document.createElement('canvas');
    tape.width = width; tape.height = height;
    const paint = tape.getContext('2d');
    paint.fillStyle = '#0e1419'; paint.fillRect(0, 0, width, height);
    paint.font = '11px system-ui,sans-serif'; paint.fillStyle = '#7f8c99';
    for (let t = Math.ceil(from / 3600) * 3600; t < to; t += 3600) {
      paint.fillRect(x(t), 0, 1, height);
      paint.fillText(hhmm.format(new Date(t * 1000)), x(t) + 4, 12);
    }
    for (const [cam, top] of [['cam1', 26], ['cam2', 58]]) {
      paint.fillStyle = cam === 'cam1' ? '#2f7de1' : '#e67a1c';
      paint.fillRect(0, top, 3, 26);
      for (const track of data.tracks) {
        if (track.cam !== cam) continue;
        const item = record(track.key);
        item.who.forEach((who, part) => {
          const [a, b] = partRange(track.key, part);
          paint.fillStyle = who ? colourOf(who) : '#39495a';
          paint.fillRect(x(a), top + (who ? 0 : 9), Math.max(1, x(b) - x(a)), who ? 26 : 8);
        });
      }
    }
    tapeKey = key;
  }
  const paint = canvas.getContext('2d');
  paint.clearRect(0, 0, width, height);
  paint.drawImage(tape, 0, 0);
  paint.fillStyle = '#ffffff';
  paint.fillRect(x(T) - 1, 0, 2, height);
}

function paintAll() {
  paintSelected();
  paintRoster();
  paintProgress();
  paintTimeline();
}

/* ---------- the loop ---------- */

let previous = performance.now();
function tick(now) {
  const step = Math.min(0.25, (now - previous) / 1000);
  previous = now;
  if (data) {
    if (playing) {
      const cam = driving();
      const segment = cam && segmentAt(cam, T + shift(cam));
      if (cam && segment && videos[cam].dataset.seg === segment.name) {
        T = segment.start + segment.origin + videos[cam].currentTime - shift(cam);
      } else {
        T += step * rate;
      }
    }
    ensure('cam1'); ensure('cam2');
    overlay('cam1'); overlay('cam2');
    paintTimeline();
    $('#clock').textContent = clock(T);
  }
  requestAnimationFrame(tick);
}

/* ---------- wiring ---------- */

for (const cam of ['cam1', 'cam2']) overlays[cam].onclick = event => pick(cam, event);
$('#play').onclick = () => { playing = !playing; paintPlay(); };
$('#next').onclick = () => jump(1);
$('#newPerson').onclick = newPerson;
$('#passer').onclick = () => answer('_passer');
$('#falsePos').onclick = () => answer('_false');
$('#cut').onclick = cutHere;
$('#sweep').onclick = sweep;
$('#undo').onclick = undo;
$('#help').onclick = () => $('#helpBox').showModal();
$('#longOnly').onchange = event => { longOnly = event.target.checked; };
document.querySelectorAll('dialog [data-close]').forEach(b => b.onclick = () => b.closest('dialog').close());
$('#timeline').onclick = event => {
  const rect = $('#timeline').getBoundingClientRect();
  const from = data.windows[0].start, to = data.windows[data.windows.length - 1].end;
  seek(from + (event.clientX - rect.left) / rect.width * (to - from));
};

document.addEventListener('keydown', event => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName)) return;
  const open = document.querySelector('dialog[open]');
  if (open) {
    if (event.key === 'Escape') { open.close(); event.preventDefault(); }
    if (event.key === 'Enter' && open.id === 'gridBox') { $('#gridGo').click(); event.preventDefault(); }
    if (event.key === 'Enter' && open.id === 'sweepBox') { $('#sweepGo').click(); event.preventDefault(); }
    return;
  }
  if (!data) return;
  const key = event.key;
  const stop = () => event.preventDefault();
  if (key === '?' || (key === '/' && event.shiftKey)) { $('#helpBox').showModal(); stop(); }
  else if (key === 'Tab') { jump(event.shiftKey ? -1 : 1); stop(); }
  else if (key === ' ') { playing = !playing; paintPlay(); stop(); }
  else if (key === 'ArrowLeft') { playing = false; seek(T - (event.shiftKey ? 10 : 2)); paintPlay(); stop(); }
  else if (key === 'ArrowRight') { playing = false; seek(T + (event.shiftKey ? 10 : 2)); paintPlay(); stop(); }
  else if (key === ',') { playing = false; seek(T - 0.2); paintPlay(); stop(); }
  else if (key === '.') { playing = false; seek(T + 0.2); paintPlay(); stop(); }
  else if (key === '[' || key === ']') {
    const at = RATES.indexOf(rate);
    rate = RATES[Math.min(RATES.length - 1, Math.max(0, at + (key === ']' ? 1 : -1)))];
    paintPlay(); stop();
  }
  else if (/^[1-9]$/.test(key)) { const who = roster()[Number(key) - 1]; if (who) answer(who); stop(); }
  else if (key === 'n' || key === 'N' || key === 'т' || key === 'Т') { newPerson(); stop(); }
  else if (key === 'p' || key === 'P' || key === 'з' || key === 'З') { answer('_passer'); stop(); }
  else if (key === 'x' || key === 'X' || key === 'ч' || key === 'Ч') { answer('_false'); stop(); }
  else if (key === 'c' || key === 'C' || key === 'с' || key === 'С') { cutHere(); stop(); }
  else if (key === 'g' || key === 'G' || key === 'п' || key === 'П') { sweep(); stop(); }
  else if (key === 's' || key === 'S' || key === 'ы' || key === 'Ы') {
    const who = sel && record(sel.key).who[sel.part];
    if (who && !SPECIAL[who]) grid(who); else notice('Сначала назовите человека на этом треке.');
    stop();
  }
  else if (key === 'z' || key === 'Z' || key === 'я' || key === 'Я') { undo(); stop(); }
  else if (key === 'Backspace') { answer(null); stop(); }
  else if (key === 'Escape') { sel = null; paintSelected(); stop(); }
});

window.addEventListener('beforeunload', event => { if (outbox.length) { event.preventDefault(); event.returnValue = ''; } });

async function load(which) {
  loading = true;
  notice('Собираю день…');
  try {
    day = which;
    const fresh = await api(`/api/dayplayer/${day}`);
    data = fresh;
    store = fresh.store;
    byKey = Object.fromEntries(fresh.tracks.map(t => [t.key, t]));
    stale = new Set(fresh.stale || []);
    $('#undo').disabled = !fresh.can_undo;
    sel = null;
    for (const clip of Object.keys(boxes)) delete boxes[clip];
    asked.clear();
    recount();
    T = fresh.windows.length ? fresh.windows[0].start : 0;
    seek(T);
    loading = false;
    notice(fresh.tracks.length ? '' : 'За этот день треков нет', !fresh.tracks.length);
    paintAll();
    paintPlay();
  } catch (error) {
    loading = false;
    notice(error.message, true);
  }
}

(async () => {
  try {
    const clips = await api('/api/clips');
    const days = [...new Set(clips.map(c => c.start.slice(0, 10).replaceAll('-', '')))];
    $('#day').innerHTML = days.map(d => `<option>${d}</option>`).join('');
    $('#day').onchange = () => load($('#day').value);
    requestAnimationFrame(tick);
    if (days.length) await load(days[0]); else notice('Нет готовых записей', true);
  } catch (error) {
    notice(error.message, true);
  }
})();
