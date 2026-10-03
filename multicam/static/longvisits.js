'use strict';
/* The owner's labelling queue.
 *
 * Two things were making it unusable. Wholeness ("is this visit complete?") cannot be
 * answered by looking at what is on the screen -- to answer it honestly you would have to
 * search the whole day -- so it is no longer asked: the page shows everyone who appeared
 * just after the visit ended and just before it started, and the person answers one local
 * question about each. When nothing is left, the page states what was checked and offers
 * that as the answer. And every answer used to be a form to fill with the mouse and a wait
 * for the whole day to come back; now three keys mean the same three things at every step,
 * saving happens behind the screen, and the next visits' pictures are already loaded.
 */
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clock = s => new Intl.DateTimeFormat('ru-RU', {hour:'2-digit',minute:'2-digit',second:'2-digit',timeZone:'Asia/Bangkok'}).format(new Date(s * 1000));
// Rounded once, before splitting: 1199.6 s is 20 min 0 s, not "19 min 60 s".
const length = s => { const t = Math.round(s); return t >= 60 ? `${Math.floor(t / 60)} мин ${t % 60} с` : `${Math.round(s * 10) / 10} с`; };
const soon = minutes => minutes >= 90 ? `${Math.round(minutes / 6) / 10} ч` : `${Math.round(minutes)} мин`;
const cams = list => (list || []).map(c => c.replace('cam', '')).join(', ') || '—';
const many = (n, one, few, rest) => {
  const tail = Math.abs(n) % 100, last = tail % 10;
  return `${n} ${tail > 4 && tail < 21 ? rest : last === 1 ? one : last > 1 && last < 5 ? few : rest}`;
};

let data = null, day = null, index = 0, focus = null, showRest = false;
let loading = false, blocked = false, ticket = 0;
const drafts = new Map();          // visit id -> the answer being composed
const local = new Map();           // node key -> edge answer given here, before the server confirms
const warmed = new Set();
const session = {answered: 0, started: Date.now()};
const outbox = [];
let sending = false;
let actions = [];                  // the keybar and the keyboard are dispatched from this one list
let shown = null;                  // what is on screen now, so the page scrolls back up only on a new question
let touched = null;                // the visit the last saved answer was about, so undo comes back to it
let merging = false;               // a «same» is in flight and the visit is about to be rebuilt

const minimum = () => Number($('#minimum').value);
const visit = () => data && data.visits[index];

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

/* ---------- pictures: a crop of this person, never the whole 2560x1440 frame ---------- */

function crop(clip, shot) {
  return `/api/video/${encodeURIComponent(clip)}/image?cam=${encodeURIComponent(shot.cam)}`
       + `&frame=${shot.frame}&box=${shot.box.join(',')}&pad=0.45`;
}

function pictures(item, lead, limit) {
  const shots = (item.shots || []).slice();
  if (lead === 'last') shots.reverse();
  const out = shots.slice(0, limit).map(shot => ({
    src: crop(item.clip, shot), shot,
    open: `/video?clip=${encodeURIComponent(item.clip)}&cam=${encodeURIComponent(shot.cam)}&frame=${shot.frame}`}));
  if (!out.length && item.portrait != null)
    out.push({src: `/portrait/${encodeURIComponent(item.clip)}/${item.portrait}`, shot: null,
              open: `/video?clip=${encodeURIComponent(item.clip)}`});
  return out;
}

/** Every picture says which camera took it. The two cameras look at the hall from opposite
 *  sides, so the same person can look like two people -- the owner's first session went
 *  wrong exactly there, because the camera was only written in small grey text. */
const CAMERA = {cam1: 'КАМЕРА 1', cam2: 'КАМЕРА 2'};
function framed(picture, alt, lazy) {
  const cam = picture.shot && picture.shot.cam;
  return `<span class="shot-wrap${cam ? ' ' + cam : ''}">${cam ? `<b class="cam-badge">${CAMERA[cam] || esc(cam)}</b>` : ''}`
    + `<img src="${esc(picture.src)}" alt="${esc(alt)}"${lazy ? ' loading="lazy"' : ''} decoding="async"></span>`;
}
const camsOf = list => new Set(list.map(p => p.shot && p.shot.cam).filter(Boolean));

function warm(url) {
  if (warmed.has(url)) return;
  warmed.add(url);
  const image = new Image();
  image.decoding = 'async';
  image.src = url;
}

/** The next visits' pictures, and this visit's edge, are fetched before they are needed:
 *  a frame costs a seek into the recording the first time anybody asks for it. */
function prefetch() {
  const here = visit();
  if (here) for (const c of [...here.after.ask, ...here.before.ask].slice(0, 3))
    for (const p of pictures(c, c.side === 'after' ? 'first' : 'last', 2)) warm(p.src);
  for (const v of data.visits.slice(index + 1, index + 4)) {
    for (const f of v.fragments.slice(0, 4)) for (const p of pictures(f, 'first', 1)) warm(p.src);
    for (const c of [...v.after.ask, ...v.before.ask].slice(0, 2))
      for (const p of pictures(c, c.side === 'after' ? 'first' : 'last', 1)) warm(p.src);
  }
}

/* ---------- what is answered, and what is still open ---------- */

function draft() {
  const v = visit();
  if (!drafts.has(v.id)) {
    const known = v.judgement;
    drafts.set(v.id, {purity: known ? known.purity : null,
                      foreign: new Set(known ? known.foreign || [] : []),
                      foreignDone: !!known});
  }
  return drafts.get(v.id);
}

const decisionOf = c => local.has(c.node) ? local.get(c.node) : c.decision;
const asked = () => [...visit().after.ask, ...visit().before.ask];
const pending = () => asked().filter(c => decisionOf(c) == null);
const everything = () => ['after', 'before'].flatMap(s => [...visit()[s].ask, ...visit()[s].rest]);

/** How one edge stands. Open counts only what was actually put to the person: the
 *  candidates asked about, plus any that did not fit on screen this time. Someone the
 *  appearance model set aside is counted apart and said out loud, never folded in --
 *  and an answer of "cannot tell" leaves the edge open, because it is not an answer. */
function edge(side) {
  const box = visit()[side];
  let open = box.over, unsure = 0, checked = 0;
  for (const c of box.ask) if (decisionOf(c) == null) open += 1;
  for (const c of [...box.ask, ...box.rest]) {
    const said = decisionOf(c);
    if (said === 'unsure') unsure += 1; else if (said != null) checked += 1;
  }
  return {open, unsure, checked, needed: checked + unsure + open, far: box.far,
          total: box.total, closed: open === 0 && unsure === 0};
}

function edges() {
  const after = edge('after'), before = edge('before');
  return {open: after.open + before.open, unsure: after.unsure + before.unsure,
          over: visit().after.over + visit().before.over,
          far: after.far + before.far, total: after.total + before.total,
          closed: after.closed && before.closed};
}

function step() {
  const v = visit();
  if (!v) return 'none';
  if (focus) return 'edge';
  const d = draft();
  if (!d.purity) return 'purity';
  if (d.purity === 'mixed' && !d.foreignDone) return 'foreign';
  if (pending().length) return 'edge';
  return 'confirm';
}

const current = () => focus
  ? everything().find(c => c.node === focus) || pending()[0] || null
  : pending()[0] || null;

/* ---------- saving: serialised, behind the screen, never silently lost ---------- */

function send(job) {
  outbox.push(job);
  pump();
}

async function pump() {
  if (sending || blocked || !outbox.length) return;
  sending = true;
  const job = outbox[0];
  try {
    const result = await api(`/api/day/${day}/visits`, {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...job.body(), revision: data.revision, minimum: minimum()})});
    outbox.shift();
    sending = false;
    if (result.revision !== undefined) data.revision = result.revision;
    if (result.can_undo !== undefined) data.can_undo = result.can_undo;
    if (result.counts) data.counts = result.counts;
    if (job.done) job.done(result);
    paint();
    pump();
  } catch (error) {
    sending = false;
    blocked = true;
    // A refusal and a broken connection need opposite treatment. The server refuses when
    // the day has moved on -- the night pass relabelled a clip, another tab answered
    // first -- and repeating the same answer can only be refused again, so it is taken
    // back off the screen instead of being shown as saved. Anything else may well work on
    // the second try, so the answer is kept and can be sent again.
    const refused = error.status === 409 || error.status === 400;
    if (refused) {
      const lost = outbox.shift();
      if (lost && lost.revert) lost.revert();
    }
    const said = /[.!?]$/.test(error.message) ? error.message : error.message + '.';
    notice('Не сохранено: ' + said
           + (refused ? ' Этот ответ снят с экрана — обновите список и ответьте заново.'
                      : ' Ответ не потерян, можно повторить.'), true);
    draw();
  }
}

function retry() {
  blocked = false;
  notice('Пробую сохранить снова…');
  pump();
}

/** The other way out of a refused save. Retrying is right when the server was merely
 *  unreachable; when it refused because the day moved on -- the night pass relabelled a
 *  clip, another tab answered first -- retrying can only be refused again, and the answer
 *  has to be given against what is there now. Nothing is thrown away silently: the page
 *  says which answer is being dropped. */
function reload() {
  const lost = outbox.length;
  outbox.length = 0;
  blocked = false;
  load(visit() && visit().nodes[0]).then(() =>
    notice(lost ? `Список обновлён. Несохранённых ответов: ${lost} — дайте их заново.` : 'Список обновлён.', !!lost));
}

async function load(anchor) {
  const mine = ++ticket;
  loading = true;
  notice('Собираю визиты дня…');
  draw();
  try {
    const fresh = await api(`/api/day/${day}/visits?seconds=${encodeURIComponent(minimum())}`);
    if (mine !== ticket) return;
    loading = false;
    apply(fresh, anchor);
    notice('');
  } catch (error) {
    if (mine !== ticket) return;
    loading = false; data = null;
    notice(error.message, true);
    draw();
  }
}

function apply(fresh, anchor) {
  data = fresh;
  local.clear();
  focus = null;
  showRest = false;
  if (anchor) {
    const at = data.visits.findIndex(v => v.nodes.includes(anchor));
    if (at >= 0) index = at;
  }
  index = Math.max(0, Math.min(index, data.visits.length - 1));
  paint();
  draw();
}

/* ---------- answering ---------- */

function answerPurity(value) {
  const v = visit(), d = draft();
  d.purity = value;
  d.assisted = d.assisted || !!(v.suggestion && !v.judgement);
  d.foreignDone = value !== 'mixed';
  d.foreign.clear();
  // Agreeing that someone else is mixed in starts from the fragments the proposal named.
  if (value === 'mixed' && v.suggestion && v.suggestion.purity === 'mixed')
    for (const key of v.suggestion.foreign || []) d.foreign.add(key);
  draw();
}

/** An employee is not a shopper: one key, no boundary questions, and out of the acceptance
 *  number. On 17.09 he was more than half of the long visits. */
function saveStaff() {
  const v = visit(), was = v.judgement, wasRetired = v.retired, from = index;
  const assisted = !!(v.suggestion && !v.judgement);
  v.judgement = {kind: 'staff', purity: null, wholeness: null, foreign: [], evidence: v.evidence};
  v.retired = false;
  touched = v.nodes[0];
  session.answered += 1;
  send({body: () => ({nodes: v.nodes, evidence: v.evidence, kind: 'staff', brief: true,
                      ...(assisted ? {assisted: 'claude'} : {})}),
        revert: () => { v.judgement = was; v.retired = wasRetired; session.answered -= 1; index = from; }});
  move(1);
}

function toggleForeign(key) {
  const d = draft();
  if (d.foreign.has(key)) d.foreign.delete(key); else d.foreign.add(key);
  draw();
}

function answerEdge(decision) {
  const c = current();
  if (!c) return;
  const anchor = touched = visit().nodes[0];
  local.set(c.node, decision);
  focus = null;
  const body = {action: 'link', decision, a: c.a, b: c.b,
                evidence_a: c.evidence_a, evidence_b: c.evidence_b, brief: decision !== 'same',
                ...(c.suggested ? {assisted: 'claude'} : {})};
  if (decision === 'same') merging = true;
  send({body: () => body,
        revert: () => { merging = false; local.delete(c.node); },
        done: result => {
          if (decision !== 'same') return;
          merging = false;
          apply(result, anchor);
          notice('Связано — визит пересобран, границу спрашиваю заново');
        }});
  if (decision === 'same') notice('Связываю…');
  draw();
}

function save(wholeness) {
  const v = visit(), d = draft(), was = v.judgement, wasRetired = v.retired, from = index;
  const body = {nodes: v.nodes, evidence: v.evidence, purity: d.purity, wholeness,
                foreign: d.purity === 'mixed' ? [...d.foreign] : [], brief: true,
                ...(d.assisted ? {assisted: 'claude'} : {})};
  // Shown as answered at once, and taken back just as plainly if the server refuses it.
  v.judgement = {purity: d.purity, wholeness, foreign: body.foreign, evidence: v.evidence};
  v.retired = false;
  touched = v.nodes[0];
  session.answered += 1;
  send({body: () => body, revert: () => {
    v.judgement = was;
    v.retired = wasRetired;
    session.answered -= 1;
    index = from;
    drafts.delete(v.id);
  }});
  move(1);
}

function move(by) {
  if (!data || !data.visits.length) return;
  index = Math.max(0, Math.min(index + by, data.visits.length - 1));
  focus = null;
  showRest = false;
  notice('');
  paint();
  draw();
}

function restart() {
  const d = draft();
  d.purity = null;
  d.foreignDone = false;
  focus = null;
  draw();
}

function undo() {
  const back = touched || (visit() && visit().nodes[0]);
  send({body: () => ({action: 'undo'}), done: () => { touched = null; load(back); }});
  notice('Отменяю…');
}

/* ---------- video and the exact frame ---------- */

async function video() {
  const c = current(), v = visit();
  if (!c) { notice('Видео показывается для сравнения на границе визита.'); return; }
  const tail = v.fragments[v.fragments.length - 1];
  const wanted = [[tail, 'last', 'Конец визита'], [c, c.side === 'after' ? 'first' : 'last', 'Кандидат']];
  $('#videos').innerHTML = wanted.map(([, , title]) =>
    `<div class="video"><p>${esc(title)}</p><p class="muted">готовлю видео…</p></div>`).join('');
  $('#player').showModal();
  const boxes = document.querySelectorAll('#videos .video');
  await Promise.all(wanted.map(async ([item, lead, title], i) => {
    const shot = (pictures(item, lead, 1)[0] || {}).shot;
    if (!shot) { boxes[i].innerHTML = `<p>${esc(title)}</p><p class="muted">нет точного кадра</p>`; return; }
    try {
      const made = await api(`/api/video/${encodeURIComponent(item.clip)}/window?cam=${encodeURIComponent(shot.cam)}&frame=${shot.frame}`);
      boxes[i].innerHTML = `<p>${esc(title)}</p><video controls playsinline src="${esc(made.url)}"></video>`;
    } catch (error) {
      boxes[i].innerHTML = `<p>${esc(title)}</p><p class="muted">${esc(error.message)}</p>`;
    }
  }));
}

function openFrame() {
  const v = visit();
  const item = current() || v.fragments[0];
  const place = pictures(item, 'first', 1)[0];
  if (place) window.open(place.open, '_blank', 'noopener');
}

/* ---------- drawing ---------- */

function paint() {
  const c = data && data.counts;
  $('#undo').disabled = !data || !data.can_undo || loading || blocked;
  if (!c) { $('#progress').textContent = ''; return; }
  const left = c.long_visits - c.answered;
  const minutes = (Date.now() - session.started) / 60000;
  const rate = session.answered >= 2 && minutes > 0.5 ? session.answered / minutes : 0;
  $('#progress').textContent =
    `отвечено ${c.answered} из ${c.long_visits} · осталось ${left}`
    + (rate ? ` · ${rate.toFixed(1)} в минуту · ещё ≈${soon(left / rate)}` : '')
    + (c.staff ? ` · сотрудник ${c.staff}` : '')
    + (c.suggested ? ` · подсказок Claude ${c.suggested}` : '')
    + (c.retired ? ` · ${c.retired} устарели после правок` : '');
}

function fragmentCard(fragment, i, marking, foreign, compact) {
  const picture = pictures(fragment, 'first', 1)[0];
  const caption = compact
    ? `<span class="num">${i + 1}</span><span class="num-time">${esc(clock(fragment.first))}</span>`
    : `<span class="num">${i + 1}</span><span class="num-time">${esc(clock(fragment.first))}–${esc(clock(fragment.last))}</span><br>
       ${esc(length(fragment.seconds))} · ${fragment.cams.length > 1 ? 'камеры' : 'камера'} ${esc(cams(fragment.cams))}<br>
       ${esc(fragment.label)}${fragment.reviewed ? ' · проверен' : ''}${marking && foreign ? ' · чужой' : ''}`;
  return `<figure class="${foreign ? 'foreign' : ''}" data-fragment="${esc(fragment.key)}">
    ${picture ? framed(picture, 'Фрагмент ' + (i + 1), true) : '<p class="muted">нет кадра</p>'}
    <figcaption>${caption}</figcaption>
  </figure>`;
}

/** Another labeller's answer, shown as a proposal: Enter accepts it, any digit overrides it.
 *  Without one, Enter does nothing on a question -- it used to mean "same" and "one person"
 *  by default, which is how a tired Enter turns into a wrong merge. */
function suggestionBox(s) {
  if (!s) return '';
  const trust = {high: 'уверенно', medium: 'средняя уверенность', low: 'низкая уверенность'}[s.confidence] || '';
  return `<div class="proposal"><b>Claude предлагает: ${esc(s.text)}</b>${trust ? ` · ${trust}` : ''} · <kbd>Enter</kbd> — согласиться
    ${s.notes ? `<div class="hint">${esc(s.notes)}</div>` : ''}</div>`;
}

function candidateColumns(c) {
  const v = visit();
  const tail = c.side === 'after' ? v.fragments[v.fragments.length - 1] : v.fragments[0];
  const ours = pictures(tail, c.side === 'after' ? 'last' : 'first', 2);
  const theirs = pictures(c, c.side === 'after' ? 'first' : 'last', 2);
  const shows = list => list.map(p => framed(p, '', false)).join('') || '<p class="muted">нет кадра</p>';
  const left = camsOf(ours), right = camsOf(theirs);
  const crossed = left.size && right.size && ![...right].some(cam => left.has(cam));
  const when = c.side === 'after'
    ? `Визит кончается в <span class="num-time">${esc(clock(v.last))}</span>`
    : `Визит начинается в <span class="num-time">${esc(clock(v.first))}</span>`;
  const theirWhen = c.side === 'after'
    ? `Появился через <span class="gap">${c.gap_s} с</span>, в <span class="num-time">${esc(clock(c.first))}</span>`
    : `Пропал за <span class="gap">${c.gap_s} с</span> до этого, в <span class="num-time">${esc(clock(c.last))}</span>`;
  return `${crossed ? `<p class="warn">Слева и справа — <b>разные камеры</b>. Они смотрят на зал с противоположных сторон: один и тот же человек выглядит иначе. Сравнивайте одежду, сумку, волосы — не ракурс и не фон.</p>` : ''}
    ${suggestionBox(c.suggested && {...c.suggested, text: {same: 'тот же', different: 'другой', unsure: 'не разобрать'}[c.suggested.decision]})}
    <div class="pair">
    <div class="col"><h3>${when}</h3><div class="frames">${shows(ours)}</div>
      <p class="hint">${esc(tail.clip)} · ${esc(tail.label)} · камера ${esc(cams(tail.cams))}</p></div>
    <div class="col"><h3>${theirWhen}</h3><div class="frames">${shows(theirs)}</div>
      <p class="hint">${esc(c.clip)} · ${esc(c.label)} · ${esc(length(c.seconds))} · камера ${esc(cams(c.cams))}
      ${c.same_clip ? ' · <b>та же запись</b>' : ''}
      ${c.distance == null ? ' · без описания внешности' : ` · непохожесть ${c.distance.toFixed(3)}`}</p>
      ${c.joins ? `<p class="hint">Это уже целый визит из ${c.joins.fragments} фрагментов (${esc(length(c.joins.seconds))}); ответ «тот же» соединит их.</p>` : ''}
      ${decisionOf(c) ? `<p class="hint">Уже отвечено: ${esc({same:'тот же',different:'другой',unsure:'не разобрать'}[decisionOf(c)])}. Ответьте заново, чтобы изменить.</p>` : ''}
    </div></div>`;
}

function restList() {
  const v = visit();
  const rest = ['after', 'before'].flatMap(s => v[s].rest);
  const hidden = v.after.more + v.before.more;
  if (!rest.length) return '';
  const label = {same: 'тот же', different: 'другой', unsure: 'не разобрать'};
  if (!showRest) return `<p class="hint">Ещё ${rest.length + hidden} рядом: отвеченные и те, кого модель считает явно другими. Показать — <kbd>A</kbd>.</p>`;
  return `<div class="rest">${rest.map(c => {
    const picture = pictures(c, c.side === 'after' ? 'first' : 'last', 1)[0];
    const answer = decisionOf(c);
    return `<button data-focus="${esc(c.node)}">
      ${picture ? framed(picture, '', true) : ''}
      ${c.side === 'after' ? '+' : '−'}${c.gap_s} с · ${c.distance == null ? 'без описания' : c.distance.toFixed(2)}<br>
      ${answer ? esc(label[answer]) : 'не спрашивали'}</button>`;
  }).join('')}</div>${hidden ? `<p class="hint">Показаны ближайшие по времени; ещё ${hidden} дальше в окне.</p>` : ''}`;
}

function verdict() {
  const v = visit(), state = edges(), window = Math.round(data.window / 60);
  const said = side => {
    const box = edge(side), when = clock(side === 'after' ? v.last : v.first);
    const word = side === 'after' ? 'После ухода' : 'До появления';
    if (!box.total) return `${word} в ${when} рядом никого не было — за ${window} мин ни одного фрагмента.`;
    if (box.open) return `${word} в ${when}: без ответа ${box.open} из ${box.needed}.`;
    if (box.unsure) return `${word} в ${when}: ${many(box.unsure, 'кандидат', 'кандидата', 'кандидатов')} с ответом «не разобрать» — граница не закрыта.`;
    if (box.checked) return `${word} в ${when}: проверено ${box.checked}, все — другие люди.`;
    return `${word} в ${when}: рядом ${many(box.total, 'человек', 'человека', 'человек')}, но по внешности это явно не он — вас не спрашивали.`;
  };
  return `<div class="verdict ${state.closed ? 'ok' : 'open'}">
    <p>${esc(said('after'))}</p><p>${esc(said('before'))}</p>
    ${state.far ? `<p class="hint">Не спрашивали про ${state.far}: модель считает их явно другими людьми. Посмотреть — <kbd>A</kbd>.</p>` : ''}
    <p class="hint">Проверено окно в ${window} мин с каждой стороны. Человек, вернувшийся позже, в него не попадает.</p>
  </div>`;
}

function draw() {
  const stage = $('#stage'), bar = $('#keybar');
  actions = [];
  document.body.classList.remove('marking');
  if (loading || !data) {
    stage.innerHTML = '<div class="panel"><p>Загрузка…</p></div>';
    bar.innerHTML = '';
    return;
  }
  if (!data.visits.length) {
    stage.innerHTML = '<div class="panel"><h1>Длинных визитов нет</h1><p>За этот день никто не задержался дольше выбранного времени, либо окна дня ещё размечаются.</p></div>';
    bar.innerHTML = '';
    return;
  }
  const v = visit(), d = draft(), now = step(), state = edges();
  const head = `<div class="panel">
    <h1>${esc(length(v.seconds))} в зале · <span class="num-time">${esc(clock(v.first))}–${esc(clock(v.last))}</span></h1>
    <p class="muted">Визит ${index + 1} из ${data.visits.length} · ${esc(many(v.fragments.length, 'фрагмент', 'фрагмента', 'фрагментов'))} · ${v.cams.length > 1 ? 'камеры' : 'камера'} ${esc(cams(v.cams))}
      ${v.judgement ? ` · ваш ответ: ${v.judgement.kind === 'staff' ? 'сотрудник' : `${esc({clean:'без чужих',mixed:'есть чужой',unsure:'не разобрать'}[v.judgement.purity])}, ${esc({whole:'целиком',partial:'неполный',unsure:'не разобрать'}[v.judgement.wholeness])}`}` : ''}
      ${v.retired ? ' · прежний ответ устарел после правок' : ''}</p>`;
  // Where the answer is about the whole visit, its fragments come first; where it is about
  // a boundary or what was checked there, the evidence for that comes first and the
  // fragments stay below as small context. Nothing worth reading needs scrolling to.
  const first = now === 'purity' || now === 'foreign';
  const strip = `<div class="strip${first ? '' : ' small'}">${
    v.fragments.map((f, i) => fragmentCard(f, i, now === 'foreign', d.foreign.has(f.key), !first)).join('')}</div>`;

  let body = '';
  if (now === 'purity') {
    body = `<p class="ask">${v.fragments.length === 1
        ? 'В этом фрагменте один человек, без чужих?'
        : `Это один и тот же человек во всех ${esc(many(v.fragments.length, 'фрагменте', 'фрагментах', 'фрагментах'))}?`}</p>
      <p class="hint">Вопрос только про подмешанных чужих. Полнота визита — следующий шаг, и её не надо угадывать.
      Сотрудник за столом или у стендов — клавиша <kbd>4</kbd>, больше ничего спрашивать не буду.</p>`
      + (v.cams.length > 1 ? `<p class="warn">Фрагменты сняты <b>обеими камерами</b> — синяя и оранжевая рамка. Камеры смотрят с разных сторон, поэтому один человек выглядит по-разному: сравнивайте одежду.</p>` : '');
    const proposal = !v.judgement && v.suggestion;
    const proposed = proposal ? (proposal.kind === 'staff' ? '4' : {clean: '1', mixed: '2', unsure: '3'}[proposal.purity]) : null;
    if (proposal) {
      const named = (proposal.foreign || []).map(key => v.fragments.findIndex(f => f.key === key) + 1).filter(n => n > 0);
      body += suggestionBox({...proposal, text: proposal.kind === 'staff' ? 'это сотрудник'
        : {clean: 'один человек, чужих нет', mixed: 'есть чужой' + (named.length ? ` (фрагменты ${named.join(', ')})` : ''),
           unsure: 'не разобрать'}[proposal.purity] || '—'});
    }
    actions = [{key: '1', label: 'Да, один человек', run: () => answerPurity('clean')},
               {key: '2', label: 'Нет, есть чужой', run: () => answerPurity('mixed')},
               {key: '3', label: 'Не разобрать', run: () => answerPurity('unsure')},
               {key: '4', label: 'Это сотрудник', run: saveStaff}];
    for (const a of actions) if (a.key === proposed) { a.enter = a.primary = true; a.label += ' · Enter'; }
  } else if (now === 'foreign') {
    document.body.classList.add('marking');
    body = `<p class="ask">Какие фрагменты — чужие?</p>
      <p class="hint">Нажмите номер кадра или щёлкните по нему. Можно ничего не отмечать — ответ «есть чужой» сохранится и так.</p>`;
    actions = [{key: 'Enter', label: d.foreign.size ? `Готово · отмечено ${d.foreign.size}` : 'Готово, без уточнения',
                primary: true, run: () => { draft().foreignDone = true; draw(); }},
               {key: 'Esc', label: 'Назад к вопросу', run: restart}];
  } else if (now === 'edge') {
    const c = current();
    body = `<p class="ask">${c.side === 'after' ? 'Визит продолжается этим человеком?' : 'Визит начинается с этого человека?'}</p>`
      + candidateColumns(c)
      + `<p class="hint">Осталось спросить на границах: ${pending().length}${state.over ? ` (+${state.over} дальше в очереди)` : ''}. Видео — <kbd>V</kbd>, точный кадр — <kbd>F</kbd>.</p>`
      + restList();
    actions = [{key: '1', label: 'Тот же', decision: 'same', run: () => answerEdge('same')},
               {key: '2', label: 'Другой', decision: 'different', run: () => answerEdge('different')},
               {key: '3', label: 'Не разобрать', decision: 'unsure', run: () => answerEdge('unsure')},
               {key: 'V', label: 'Видео', run: video}];
    const edgeProposal = c.suggested && !decisionOf(c) ? c.suggested.decision : null;
    for (const a of actions) if (a.decision && a.decision === edgeProposal) { a.enter = a.primary = true; a.label += ' · Enter'; }
  } else {
    const suggestion = state.closed ? 'whole' : 'unsure';
    const purity = {clean: 'один человек', mixed: 'есть чужой', unsure: 'чистота не разобрана'}[d.purity];
    body = `<p class="ask">Визит собран целиком?</p>` + verdict()
      + `<p class="hint">Сохранится как: ${esc(purity)}${d.foreign.size ? ` (чужие: ${d.foreign.size})` : ''} + ваш ответ о полноте.</p>`
      + restList();
    const choices = [['1', 'Да, целиком', 'whole'], ['2', 'Нет, часть потеряна', 'partial'], ['3', 'Не разобрать', 'unsure']];
    actions = choices.map(([key, label, value]) => ({
      key, label: label + (value === suggestion ? ' · Enter' : ''), primary: value === suggestion,
      enter: value === suggestion, run: () => save(value)}));
    actions.push({key: 'Esc', label: 'Переспросить сначала', run: restart});
  }

  stage.innerHTML = head + (first ? strip + body : body + strip) + '</div>';
  const view = `${index}|${now}|${(current() || {}).node || ''}`;
  if (view !== shown) { shown = view; window.scrollTo(0, 0); }
  bar.innerHTML = actions.map((a, i) =>
    `<button data-act="${i}" class="${a.primary ? 'primary' : ''}"${blocked ? ' disabled' : ''}>
      <kbd>${esc(a.key)}</kbd>${esc(a.label)}</button>`).join('')
    + (blocked ? (outbox.length ? '<button data-retry class="primary">Повторить сохранение</button>' : '')
               + `<button data-reload class="${outbox.length ? '' : 'primary'}">Обновить список</button>` : '');

  bar.querySelectorAll('[data-act]').forEach(b => b.onclick = () => run(actions[+b.dataset.act]));
  const again = bar.querySelector('[data-retry]');
  if (again) again.onclick = retry;
  const afresh = bar.querySelector('[data-reload]');
  if (afresh) afresh.onclick = reload;
  stage.querySelectorAll('[data-fragment]').forEach(f => {
    if (now === 'foreign') f.onclick = () => toggleForeign(f.dataset.fragment);
  });
  stage.querySelectorAll('[data-focus]').forEach(b => b.onclick = () => { focus = b.dataset.focus; draw(); });
  paint();
  prefetch();
}

function run(action) {
  if (!action) return;
  if (blocked && action.run !== retry) { notice('Сначала нужно сохранить предыдущий ответ.', true); return; }
  // The screen still shows the old boundary while the join is being made; an answer given
  // now would be about a visit that no longer exists by the time it arrives.
  if (merging) { notice('Связываю визит — секунду.'); return; }
  action.run();
}

/* ---------- keyboard: dispatched from the very list the keybar draws ---------- */

document.addEventListener('keydown', event => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  const inField = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName);
  const open = document.querySelector('dialog[open]');
  if (open) {
    if (event.key === 'Escape') { open.close(); event.preventDefault(); }
    return;
  }
  if (inField && event.key !== 'Escape') return;
  const key = event.key;
  if (key === '?' || (key === '/' && event.shiftKey)) { $('#helpBox').showModal(); event.preventDefault(); return; }
  if (!data || !data.visits.length) return;
  if (key === 'ArrowRight') { move(1); event.preventDefault(); return; }
  if (key === 'ArrowLeft') { move(-1); event.preventDefault(); return; }
  if (key === 'z' || key === 'Z' || key === 'я' || key === 'Я') {
    if (!$('#undo').disabled) undo();
    event.preventDefault(); return;
  }
  if (step() === 'foreign' && /^[1-9]$/.test(key)) {
    const fragment = visit().fragments[+key - 1];
    if (fragment) toggleForeign(fragment.key);
    event.preventDefault(); return;
  }
  if (key === 'a' || key === 'A' || key === 'ф' || key === 'Ф') { showRest = !showRest; draw(); event.preventDefault(); return; }
  if (key === 'f' || key === 'F' || key === 'а' || key === 'А') { openFrame(); event.preventDefault(); return; }
  if ((key === 'Enter' || key === ' ') && !actions.some(a => a.enter || a.key === 'Enter')) {
    notice('Enter подтверждает только предложенный ответ — здесь его нет. Нажмите цифру.');
    event.preventDefault(); return;
  }
  const wanted = key === 'Enter' || key === ' '
    ? actions.find(a => a.enter) || actions.find(a => a.key === 'Enter')
    : actions.find(a => a.key.toLowerCase() === key.toLowerCase()
        || (a.key === 'V' && (key === 'м' || key === 'М'))
        || (a.key === 'Esc' && key === 'Escape'));
  if (wanted) { run(wanted); event.preventDefault(); }
});

/* ---------- wiring ---------- */

$('#undo').onclick = () => { if (!$('#undo').disabled) undo(); };
$('#help').onclick = () => $('#helpBox').showModal();
document.querySelectorAll('dialog [data-close]').forEach(b => b.onclick = () => b.closest('dialog').close());
$('#minimum').onchange = () => { drafts.clear(); index = 0; load(); };
window.addEventListener('beforeunload', event => {
  if (outbox.length) { event.preventDefault(); event.returnValue = ''; }
});

(async () => {
  try {
    const clips = await api('/api/clips');
    const days = [...new Set(clips.map(c => c.start.slice(0, 10).replaceAll('-', '')))];
    $('#day').innerHTML = days.map(d => `<option>${d}</option>`).join('');
    day = days[0];
    $('#day').onchange = () => { day = $('#day').value; drafts.clear(); index = 0; load(); };
    if (day) await load(); else notice('Нет готовых записей', true);
  } catch (error) {
    notice(error.message, true);
  }
})();
