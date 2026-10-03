// The door, labelled: one crossing of the live counter per screen, one key per answer.
'use strict';
const $ = s => document.querySelector(s);
const NAMES = {in: 'вошёл', out: 'вышел', none: 'не проход', staff: 'сотрудник (старый ответ, без направления)', dup: 'повтор', unsure: '?'};
// Everybody is a person with entries and exits, staff included: a role belongs to the person, said once.
const KEYS = [['1', 'in', 'Покупатель вошёл'], ['2', 'out', 'Покупатель вышел'], ['3', 'none', 'Никто не прошёл'],
              ['4', 'staffdir', 'Сотрудник'], ['5', 'dup', 'Повтор'], ['6', 'unsure', 'Не разобрать']];
const BEFORE = 4, AFTER = 3, ZOOM = 2.4;
let day = null, E = [], L = {}, R = {}, i = 0, pairing = null, film = false, undo = [], busy = false;
// pairing: null | 'staffdir' | {kind: 'in'|'out', staff: bool}
const RETURN_WITHIN = 3600, LATE_EXIT = 180;   // s: who may have come back / who may be going out again

function note(msg, bad) { const n = $('#notice'); n.textContent = msg || ''; n.className = bad ? 'error' : ''; }
function snap(e, what) { return `/api/door/${day}/snap/${e.id}/${what}`; }

async function api(url, body) {
  const r = await fetch(url, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ('ошибка ' + r.status));
  return j;
}

async function loadDays() {
  const {days} = await api('/api/door');
  const want = new URLSearchParams(location.search).get('day');
  $('#day').innerHTML = days.slice().reverse().map(d => `<option value="${d}">${d.slice(6)}.${d.slice(4, 6)}.${d.slice(0, 4)}</option>`).join('');
  $('#day').value = days.includes(want) ? want : (days.includes('20260918') ? '20260918' : days[days.length - 1]);
  $('#day').onchange = () => openDay($('#day').value);
  await openDay($('#day').value);
}

async function openDay(d) {
  day = d; history.replaceState(null, '', '/door?day=' + d);
  const j = await api('/api/door/' + d);
  E = j.events; L = j.review.labels || {}; R = j.review.roles || {}; film = j.film; undo = [];
  const v = $('#video');
  if (film) { v.src = `/movie-file/${d}/cam1.mp4`; $('#nofilm').hidden = true; v.hidden = false; }
  else { v.removeAttribute('src'); v.hidden = true; $('#nofilm').hidden = false; }
  i = Math.max(0, E.findIndex(e => !L[e.id]));
  if (E.every(e => L[e.id])) i = 0;
  show();
}

function progress() {
  const n = E.filter(e => L[e.id]).length;
  const c = k => Object.values(L).filter(x => x.kind === k).length;
  const V = visitList(), cust = V.filter(v => !v.staff), whole = cust.filter(v => v.entry && v.exit).length;
  $('#progress').textContent = `размечено ${n} из ${E.length} · визитов покупателей ${cust.length}, с входом и выходом ${whole} · сотрудников: ${V.length - cust.length}` +
    (n === E.length && E.length ? ' · день готов' : '');
  $('#undo').disabled = !undo.length;
}

function focus(e) {
  // the film is 16:9 like the box: scale it up around the person the counter boxed
  const [x1, y1, x2, y2] = e.box.map(Number);
  const fx = ((x1 + x2) / 2) / 2560, fy = ((y1 + y2) / 2) / 1440;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const v = $('#video');
  v.style.width = v.style.height = (ZOOM * 100) + '%';
  v.style.left = clamp(50 - fx * ZOOM * 100, (1 - ZOOM) * 100, 0) + '%';
  v.style.top = clamp(50 - fy * ZOOM * 100, (1 - ZOOM) * 100, 0) + '%';
}

function replay() {
  const e = E[i]; if (!film || !e || e.film == null) return;
  const v = $('#video'); v.currentTime = Math.max(0, e.film - BEFORE); v.play().catch(() => {});
}

$('#video').ontimeupdate = () => {
  const e = E[i], v = $('#video'); if (!e || e.film == null) return;
  if (v.currentTime > e.film + AFTER || v.currentTime < e.film - BEFORE - 1) v.currentTime = Math.max(0, e.film - BEFORE);
  const dt = v.currentTime - e.film, t = $('#tick');
  t.textContent = Math.abs(dt) < 0.4 ? 'момент события' : (dt < 0 ? `до события ${(-dt).toFixed(1)} с` : `после ${dt.toFixed(1)} с`);
  t.className = 'door-tick' + (Math.abs(dt) < 0.4 ? ' now' : '');
};
document.addEventListener('visibilitychange', () => { if (!document.hidden) replay(); });

function show() {
  pairing = null; $('#pairing').hidden = true;
  const e = E[i];
  progress();
  if (!e) { $('#ask').textContent = 'За этот день у счётчика нет событий.'; keys(); return; }
  const lab = L[e.id];
  $('#ask').innerHTML = `<span class="num-time">${e.time}</span> · счётчик: <b>${e.event === 'entry' ? 'ВХОД' : 'ВЫХОД'}</b> (${e.role || '—'}, номер ${e.gid}).
    Что на самом деле сделал человек в рамке?` + (lab ? ` <span class="badge">Ваш ответ: ${NAMES[lab.kind]}${pairText(lab)}</span>` : '');
  $('#zoom').src = snap(e, 'zoom'); $('#crop').src = snap(e, 'crop');
  focus(e); replay(); near(); keys();
  for (const k of [1, 2]) { const n = E[i + k]; if (n) { new Image().src = snap(n, 'zoom'); new Image().src = snap(n, 'crop'); } }
}

function pairText(lab) {
  if (lab.kind !== 'in' && lab.kind !== 'out') return '';
  if (lab.person === 'unseen') return (lab.role === 'staff' ? ' · сотрудник' : '') + ' · входа не видно';
  const v = visitList().find(v => v.entry === E[i].id || v.exit === E[i].id);
  if (!v) return lab.kind === 'in' ? '' : ' · этот выход перекрыт более поздним';
  return `${v.staff ? ' · сотрудник' : ''} · визит ${v.entryTime || '?'} → ${v.exitTime || 'ещё внутри'}`;
}

function near() {
  const out = [];
  for (let k = Math.max(0, i - 4); k <= Math.min(E.length - 1, i + 4); k++) {
    const e = E[k], lab = L[e.id];
    out.push(`<figure class="${k === i ? 'cur' : ''} ${lab ? 'done' : ''}" onclick="go(${k})"><img loading="lazy" src="${snap(e, 'crop')}">
      <figcaption>${e.time} ${e.event === 'entry' ? 'вх' : 'вых'}${lab ? `<span class="lab">${NAMES[lab.kind]}</span>` : ''}</figcaption></figure>`);
  }
  $('#near').innerHTML = out.join('');
}

const byId = id => E.find(x => x.id === id);
const personOf = id => { const l = L[id]; return !l ? null : (l.kind === 'in' ? (l.person || id) : l.person); };

// The same rule as door_review.visits on the server: per person, in time order, an entry
// opens a visit unless one is open, and an exit ends it -- the latest exit wins.
function visitList() {
  const by = {};
  for (const e of E) {
    const l = L[e.id]; if (!l || (l.kind !== 'in' && l.kind !== 'out')) continue;
    let who = personOf(e.id);
    if (!who || who === 'unseen' || !L[who]) who = 'unseen:' + e.id;
    (by[who] = by[who] || []).push(e);
  }
  const out = [];
  for (const [who, rows] of Object.entries(by)) {
    rows.sort((a, b) => a.unix - b.unix);
    let cur = null;
    for (const e of rows) {
      if (L[e.id].kind === 'in') { if (cur && !cur.exit) continue; cur = {person: who, entry: e.id, exit: null}; out.push(cur); }
      else if (!cur) { cur = {person: who, entry: null, exit: e.id}; out.push(cur); }
      else cur.exit = e.id;
    }
  }
  for (const v of out) {
    v.entryTime = v.entry && byId(v.entry).time; v.exitTime = v.exit && byId(v.exit).time;
    const one = L[v.entry || v.exit];
    v.staff = R[v.person] === 'staff' || (one && one.role === 'staff');
  }
  return out;
}

// Everybody's state just before this event, from the answers about earlier events only.
function people() {
  const e = E[i], P = {};
  for (const x of E) {
    if (x.unix >= e.unix || x.id === e.id) continue;
    const l = L[x.id]; if (!l || (l.kind !== 'in' && l.kind !== 'out')) continue;
    const who = personOf(x.id); if (!who || who === 'unseen' || !L[who]) continue;
    const p = P[who] = P[who] || {id: who, face: null, inside: false, lastIn: 0, lastOut: 0, outTime: '', staff: R[who] === 'staff'};
    if (l.kind === 'in') { p.inside = true; p.face = x; p.lastIn = x.unix; }
    else { p.inside = false; p.lastOut = x.unix; p.outTime = x.time; }
  }
  return Object.values(P);
}

function candidates() {
  if (!pairing || pairing === 'staffdir') return [];
  // Everybody inside is offered whichever key was pressed: on 23.09 a woman entered as staff,
  // her exit was pressed as a customer's, and she vanished from the list. The role belongs to
  // the person chosen; the key only says what a *new* person would be. Same role first.
  const e = E[i], P = people().sort((a, b) => (a.staff !== pairing.staff) - (b.staff !== pairing.staff));
  if (pairing.kind === 'out') {
    // who is inside, newest first; then who went out a moment ago (this is a later exit of theirs)
    const mine = p => (p.staff === pairing.staff ? 0 : 1);
    const inside = P.filter(p => p.inside).sort((a, b) => mine(a) - mine(b) || b.lastIn - a.lastIn);
    const late = P.filter(p => !p.inside && e.unix - p.lastOut < LATE_EXIT).sort((a, b) => b.lastOut - a.lastOut);
    return inside.concat(late).slice(0, 9);
  }
  // coming in: somebody who went out within the hour may be coming back; staff come back after hours
  return P.filter(p => !p.inside && (p.staff || e.unix - p.lastOut < RETURN_WITHIN))
          .sort((a, b) => (a.staff !== pairing.staff) - (b.staff !== pairing.staff) || b.lastOut - a.lastOut).slice(0, 9);
}

function caption(p) {
  return (p.staff ? 'сотр. · ' : '') + (p.inside ? `вошёл ${p.face.time}` : `вышел ${p.outTime}`);
}

function staffDirection() {
  pairing = 'staffdir';
  $('#pairing').hidden = false;
  $('#pairing .ask').textContent = 'Сотрудник вошёл или вышел?';
  $('#cands').innerHTML = '';
  keys();
}

function startPairing(kind, staff) {
  pairing = {kind, staff: !!staff};
  const C = candidates(), who = staff ? 'сотрудник' : 'покупатель';
  if (kind === 'in' && !staff && !C.length) { save('in', null, 'customer'); return; }   // nobody could be coming back
  $('#pairing').hidden = false;
  $('#pairing .ask').textContent = kind === 'out'
    ? `Кто вышел? Все, кто внутри (сотрудники помечены «сотр.»); последние — кто вышел только что (тогда это его более поздний выход).`
    : (staff ? 'Какой сотрудник вернулся? Enter — новый сотрудник (первый раз за день).'
             : 'Это кто-то вернулся? Enter — нет, новый покупатель.');
  $('#cands').innerHTML = C.length ? C.map((p, k) => `<figure onclick="pick(${k})"><img src="${snap(p.face, 'crop')}">
      <figcaption><span class="num">${k + 1}</span>${caption(p)}</figcaption></figure>`).join('')
    : `<p class="muted">${kind === 'out' ? 'Внутри никого подходящего не отмечено — нажмите 0.' : 'Некого предложить — нажмите Enter.'}</p>`;
  keys();
  $('#pairing').scrollIntoView({block: 'nearest'});
}

async function save(kind, person, role) {
  const e = E[i]; if (!e) return;
  const before = L[e.id] || null;
  busy = true;  // a second key before the answer is stored would land on the same event
  try {
    const body = {event: e.id, kind, person: person === undefined ? null : person};
    if (role) body.role = role;
    const r = await api('/api/door/' + day, body);
    if (r.label) L[e.id] = r.label; else delete L[e.id];
    if (role && r.label && r.label.person !== 'unseen') R[r.label.person] = role;
    undo.push({index: i, id: e.id, before});
    note('');
  } catch (err) { note('Не сохранилось: ' + err.message, true); return; }
  finally { busy = false; }
  next(1);
}

function next(dir) {
  let k = i + dir;
  while (k >= 0 && k < E.length && L[E[k].id]) k += dir;
  if (k < 0 || k >= E.length) { k = E.findIndex(e => !L[e.id]); if (k < 0) { note('Все события дня размечены. Спасибо!'); k = Math.min(E.length - 1, Math.max(0, i + dir)); } }
  i = k; show();
}

window.go = k => { i = k; show(); };
window.pick = k => { const c = candidates()[k]; if (c) save(pairing.kind, c.id); };
const roleNow = () => (pairing && pairing.staff ? 'staff' : 'customer');

async function undoLast() {
  const u = undo.pop(); if (!u) return;
  try {
    const b = u.before;
    const r = await api('/api/door/' + day, {event: u.id, kind: b ? b.kind : null, person: b ? b.person : null,
                                             ...(b && b.role ? {role: b.role} : {})});
    if (r.label) L[u.id] = r.label; else delete L[u.id];
    i = u.index; show(); note('Отменено');
  } catch (err) { note('Не отменилось: ' + err.message, true); }
}

function keys() {
  const list = pairing === 'staffdir'
    ? [['1', () => startPairing('in', true), 'Сотрудник вошёл'], ['2', () => startPairing('out', true), 'Сотрудник вышел'], ['Esc', show, 'Назад']]
    : pairing
    ? candidates().map((c, k) => [String(k + 1), () => pick(k), caption(c)]).concat(pairing.kind === 'out'
        ? [['0', () => save('out', 'unseen', roleNow()), 'Его входа нет в списке'], ['Esc', show, 'Назад']]
        : [['Enter', () => save('in', null, roleNow()), pairing.staff ? 'Новый сотрудник' : 'Новый покупатель'], ['Esc', show, 'Назад']])
    : KEYS.map(([k, kind, label]) => [k, () => kind === 'staffdir' ? staffDirection()
        : (kind === 'out' || kind === 'in') ? startPairing(kind, false) : save(kind), label])
        .concat([['пробел', replay, 'Видео заново'], ['←', () => next(-1), 'Назад'], ['→', () => next(1), 'Дальше']]);
  const bar = $('#keys'); bar.innerHTML = '';
  for (const [k, fn, label] of list) {
    const b = document.createElement('button'); b.innerHTML = `<kbd>${k}</kbd>${label}`; b.onclick = fn; bar.appendChild(b);
  }
  keys.map = Object.fromEntries(list.map(([k, fn]) => [k, fn]));
}

document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'SELECT' || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  if (busy || !E.length) { ev.preventDefault(); return; }
  const code = ev.code || '';
  let k = /^Digit\d$/.test(code) ? code.slice(5) : /^Numpad\d$/.test(code) ? code.slice(6) : ev.key;
  if (code === 'Space' || ev.key === ' ') k = 'пробел';
  if (ev.key === 'Escape') k = 'Esc';
  if (ev.key === 'ArrowLeft') k = '←';
  if (ev.key === 'ArrowRight') k = '→';
  if (ev.key === 'Enter') k = 'Enter';
  if (code === 'KeyZ' || /^[zя]$/i.test(ev.key)) { undoLast(); ev.preventDefault(); return; }
  if (ev.key === '?' || code === 'Slash') { $('#helpbox').showModal(); ev.preventDefault(); return; }
  const fn = keys.map && keys.map[k];
  if (fn) { fn(); ev.preventDefault(); }
});
$('#undo').onclick = undoLast;
$('#help').onclick = () => $('#helpbox').showModal();
loadDays().catch(err => note('Не загрузилось: ' + err.message, true));
