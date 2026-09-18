"""Manual identity labelling of tracker pieces: who is this?

Each piece is one continuous track fragment from one camera, shown as a strip of
8 crops cut by the person's mask. The answer is a person label; the labels
already used in this clip are offered as buttons, so 'the same person as before'
is one click, and a new person is one key."""
import os, json, glob, secrets
from flask import Flask, jsonify, request, send_file, abort, redirect, make_response

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(ROOT, 'data', 'raw_clips')
KEY_FILE = os.path.join(os.path.expanduser('~'), '_labelers_key.txt')
KEY = open(KEY_FILE).read().strip() if os.path.exists(KEY_FILE) else 'cctv'
app = Flask(__name__)


@app.before_request
def gate():
    if request.args.get('key') and secrets.compare_digest(request.args['key'], KEY):
        args = {k: v for k, v in request.args.items() if k != 'key'}
        target = request.path + (('?' + '&'.join('%s=%s' % kv for kv in args.items())) if args else '')
        r = make_response(redirect(target))
        r.set_cookie('labeler_key', KEY, max_age=30 * 24 * 3600, httponly=True, samesite='Lax')
        return r
    if secrets.compare_digest(request.cookies.get('labeler_key', ''), KEY):
        return None
    return ('<meta charset="utf-8"><body style="font:16px system-ui;padding:40px">Нужен ключ: откройте ссылку с ?key=…</body>', 401)


def labels_path(clip):
    return os.path.join(CLIPS, clip, 'gt_manual.json')


@app.get('/api/clips')
def clips():
    out = []
    for d in sorted(glob.glob(os.path.join(CLIPS, '*'))):
        c = os.path.basename(d)
        pieces = glob.glob(os.path.join(d, 'pieces_*.json'))
        if not pieces:
            continue
        done = len(json.load(open(labels_path(c)))) if os.path.exists(labels_path(c)) else 0
        meta = json.load(open(glob.glob(os.path.join(d, 'meta_*.json'))[0]))
        out.append({'clip': c, 'pieces': len(json.load(open(pieces[0]))), 'done': done,
                    'start': meta['start'], 'seconds': meta['seconds']})
    return jsonify(out)


@app.get('/api/pieces/<clip>')
def pieces(clip):
    p = glob.glob(os.path.join(CLIPS, clip, 'pieces_*.json'))
    if not p:
        abort(404)
    data = json.load(open(p[0]))
    labels = json.load(open(labels_path(clip))) if os.path.exists(labels_path(clip)) else {}
    return jsonify({'pieces': data, 'labels': labels})


@app.post('/api/label/<clip>')
def label(clip):
    body = request.get_json(force=True)
    cur = json.load(open(labels_path(clip))) if os.path.exists(labels_path(clip)) else {}
    if body.get('label'):
        cur[str(body['piece'])] = body['label']
    else:
        cur.pop(str(body['piece']), None)
    tmp = labels_path(clip) + '.tmp'
    json.dump(cur, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.replace(tmp, labels_path(clip))
    return jsonify({'ok': True, 'done': len(cur)})


@app.get('/ref/<clip>/<label>')
def ref(clip, label):
    """A sample picture of a person already labelled, so 'the same one' is recognisable."""
    labels = json.load(open(labels_path(clip))) if os.path.exists(labels_path(clip)) else {}
    for piece, lab in sorted(labels.items(), key=lambda kv: int(kv[0])):
        if lab == label:
            p = os.path.join(CLIPS, clip, 'crops', '%04d.jpg' % int(piece))
            if os.path.exists(p):
                return send_file(p)
    abort(404)


@app.post('/api/merge/<clip>')
def merge(clip):
    """Two labels turn out to be one person: relabel every piece of `from` as `to`."""
    body = request.get_json(force=True)
    src, dst = body['from'], body['to']
    cur = json.load(open(labels_path(clip))) if os.path.exists(labels_path(clip)) else {}
    n = 0
    for k, v in list(cur.items()):
        if v == src:
            cur[k] = dst; n += 1
    tmp = labels_path(clip) + '.tmp'
    json.dump(cur, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.replace(tmp, labels_path(clip))
    return jsonify({'ok': True, 'moved': n})


@app.get('/api/groups/<clip>')
def groups(clip):
    """What the system thinks is one person, so a whole person can be confirmed at once."""
    g = glob.glob(os.path.join(CLIPS, clip, 'groups_*.json'))
    p = glob.glob(os.path.join(CLIPS, clip, 'pieces_*.json'))
    if not p:
        abort(404)
    pieces = json.load(open(p[0]))
    data = json.load(open(g[0])) if g else [{'person': None, 'pieces': [x['piece']], 't0': x['t0'],
                                             't1': x['t1'], 'cams': [x['cam']], 'dets': x['n_dets']} for x in pieces]
    labels = json.load(open(labels_path(clip))) if os.path.exists(labels_path(clip)) else {}
    by_id = {x['piece']: x for x in pieces}
    for grp in data:
        grp['detail'] = [{'piece': i, 'cam': by_id[i]['cam'], 't0': by_id[i]['t0'], 't1': by_id[i]['t1'],
                          'n_dets': by_id[i]['n_dets']} for i in grp['pieces'] if i in by_id]
    return jsonify({'groups': data, 'labels': labels, 'pieces': len(pieces)})


@app.post('/api/label_many/<clip>')
def label_many(clip):
    """One answer for every piece of one person."""
    body = request.get_json(force=True)
    cur = json.load(open(labels_path(clip))) if os.path.exists(labels_path(clip)) else {}
    for i in body['pieces']:
        if body.get('label'):
            cur[str(i)] = body['label']
        else:
            cur.pop(str(i), None)
    tmp = labels_path(clip) + '.tmp'
    json.dump(cur, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.replace(tmp, labels_path(clip))
    return jsonify({'ok': True, 'done': len(cur)})


@app.get('/crop/<clip>/<int:piece>')
def crop(clip, piece):
    p = os.path.join(CLIPS, clip, 'crops', '%04d.jpg' % piece)
    if not os.path.exists(p):
        abort(404)
    return send_file(p)


PAGE = r'''<!doctype html><meta charset="utf-8"><title>Кто это?</title>
<style>
body{font:15px system-ui;margin:0;background:#15171a;color:#eee}
header{position:sticky;top:0;background:#1e2126;padding:10px 14px;z-index:5;border-bottom:1px solid #2c3138}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
select,button{font:15px system-ui;padding:7px 11px;border-radius:8px;border:1px solid #3a3f47;background:#262a31;color:#eee;cursor:pointer}
button:hover{background:#30353d}
.who{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 6px}
.who .card{display:flex;flex-direction:column;align-items:center;gap:4px;background:#22262c;border:2px solid #333a44;border-radius:10px;padding:6px 8px;cursor:pointer;min-width:96px}
.who .card.cur{border-color:#49a24b;background:#243024}
.who .card img{height:74px;border-radius:5px;background:#000;object-fit:cover}
.who .card b{font-weight:600}
.who .card span{color:#9aa4b2;font-size:12px}
.special{background:#2b3240}
#strip{display:block;max-width:100%;border-radius:8px;background:#000;margin:8px 0}
.meta{color:#9aa4b2}
.bar{height:6px;background:#2a2f36;border-radius:4px;overflow:hidden;width:220px}
.bar i{display:block;height:100%;background:#49a24b}
h2{margin:6px 0 0;font-size:17px;font-weight:600}
kbd{background:#333;border-radius:4px;padding:1px 6px;font:13px system-ui}
</style>
<header>
  <div class="row">
    <select id="clip"></select>
    <span class="bar"><i id="fill"></i></span>
    <span id="pos"></span>
    <button id="jump">Перейти к первому неразмеченному</button>
    <button id="prev">← Назад</button>
    <button id="next">Пропустить →</button>
  </div>
  <h2>Кто на этих кадрах?</h2>
  <div class="meta"><a href="/people" style="color:#8ab4f8">Быстрее: подтверждать человека целиком →</a></div>
  <div class="meta">Только личность: этот человек уже был или новый. Заходил ли он в магазин и сотрудник ли он — считаем мы по траектории, отмечать это не нужно.</div>
  <div class="meta" id="meta"></div>
</header>
<div style="padding:4px 14px 40px">
  <img id="strip">
  <div class="who" id="who"></div>
  <div class="row" style="margin:10px 0 4px">
    <span class="meta">Ошиблись? Объединить</span>
    <select id="mfrom"></select><span class="meta">→ в</span><select id="mto"></select>
    <button id="merge">Объединить</button>
    <span class="meta" id="mmsg"></span>
  </div>
  <div class="meta">Клавиши: <kbd>1</kbd>…<kbd>9</kbd> — выбрать человека из списка · <kbd>N</kbd> — новый человек ·
    <kbd>X</kbd> — в рамке двое или не разобрать ·
    <kbd>⌫</kbd> — назад · <kbd>→</kbd> — пропустить. Ответ сохраняется сразу.</div>
</div>
<script>
let clip=null,P=[],L={},i=0;
const q=s=>document.querySelector(s);
const nameOf=l=>l==='?'?'Не разобрать':'Человек '+String(l).replace('P','');
async function loadClips(){
  const cs=await (await fetch('/api/clips')).json();
  q('#clip').innerHTML=cs.map(c=>`<option value="${c.clip}">${c.start.slice(11,16)} · ${Math.round(c.seconds/60)} мин · ${c.pieces} отрезков</option>`).join('');
  q('#clip').onchange=()=>openClip(q('#clip').value);
  if(cs.length) openClip(cs[0].clip);
}
async function openClip(c){
  clip=c; const d=await (await fetch('/api/pieces/'+c)).json();
  P=d.pieces; L=d.labels; i=P.findIndex(p=>!L[p.piece]); if(i<0)i=0; show();
}
function people(){
  return [...new Set(Object.values(L))].filter(v=>v&&v!=='?').sort((a,b)=>{
    const na=+String(a).replace(/\D/g,'')||0, nb=+String(b).replace(/\D/g,'')||0; return na-nb;});
}
function card(label,key,cur,special){
  const img=special?'':`<img src="/ref/${clip}/${encodeURIComponent(label)}" onerror="this.style.display='none'">`;
  return `<div class="card ${cur?'cur':''} ${special?'special':''}" onclick="put('${label}')">${img}
    <b>${nameOf(label)}</b><span>${key?'клавиша '+key:''}</span></div>`;
}
function show(){
  if(!P.length)return;
  i=Math.max(0,Math.min(P.length-1,i));
  const p=P[i], done=Object.keys(L).length;
  q('#pos').textContent=`отрезок ${i+1} из ${P.length} · размечено ${done}`;
  q('#fill').style.width=(100*done/P.length)+'%';
  q('#meta').textContent=`камера ${p.cam.replace('cam','')} · с ${p.t0} по ${p.t1} секунду клипа · ${p.n_dets} кадров подряд`
    +(L[p.piece]?` · сейчас отмечено: ${nameOf(L[p.piece])}`:'');
  q('#strip').src=`/crop/${clip}/${p.piece}?v=${Date.now()}`;
  const cur=L[p.piece]||'';
  const ppl=people();
  const opts=ppl.map(n=>`<option value="${n}">${nameOf(n)}</option>`).join('');
  q('#mfrom').innerHTML=opts; q('#mto').innerHTML=opts;
  if(ppl.length>1){q('#mfrom').value=ppl[ppl.length-1]; q('#mto').value=ppl[ppl.length-2];}
  q('#who').innerHTML=ppl.map((n,k)=>card(n,k<9?(k+1):'',cur===n,false)).join('')
    +card('NEW','N',false,true).replace('>Человек NEW<','>Новый человек<')
    +card('?','X',cur==='?',true);
}
function newName(){
  const n=people().filter(x=>/^P\d+$/.test(x)).map(x=>+x.slice(1));
  return 'P'+((n.length?Math.max(...n):0)+1);
}
async function put(lab){
  if(lab==='NEW')lab=newName();
  const p=P[i];
  await fetch('/api/label/'+clip,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({piece:p.piece,label:lab})});
  if(lab)L[p.piece]=lab; else delete L[p.piece];
  i++; show();
}
document.onkeydown=e=>{
  if(e.target.tagName==='SELECT')return;
  const ppl=people(), k=e.key.toLowerCase();
  if(e.key>='1'&&e.key<='9'){const n=ppl[+e.key-1]; if(n)put(n);}
  else if(k==='n')put('NEW');
  else if(k==='x')put('?');
  else if(e.key==='Backspace'){i--;show();e.preventDefault();}
  else if(e.key==='ArrowRight'){i++;show();}
  else if(e.key==='ArrowLeft'){i--;show();}
};
q('#merge').onclick=async()=>{
  const a=q('#mfrom').value,b=q('#mto').value;
  if(!a||!b||a===b){q('#mmsg').textContent='выберите двух разных';return;}
  const r=await (await fetch('/api/merge/'+clip,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({from:a,to:b})})).json();
  const d=await (await fetch('/api/pieces/'+clip)).json(); L=d.labels;
  q('#mmsg').textContent=`${nameOf(a)} → ${nameOf(b)}: перенесено ${r.moved} отрезков`;
  show();
};
q('#jump').onclick=()=>{const k=P.findIndex(p=>!L[p.piece]); if(k>=0){i=k;show();}};
q('#prev').onclick=()=>{i--;show();};
q('#next').onclick=()=>{i++;show();};
loadClips();
</script>'''


PEOPLE_PAGE = r"""<!doctype html><meta charset="utf-8"><title>Подтвердить человека</title>
<style>
body{font:15px system-ui;margin:0;background:#15171a;color:#eee}
header{position:sticky;top:0;background:#1e2126;padding:10px 14px;z-index:5;border-bottom:1px solid #2c3138}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
select,button,a.btn{font:15px system-ui;padding:7px 11px;border-radius:8px;border:1px solid #3a3f47;background:#262a31;color:#eee;cursor:pointer;text-decoration:none}
button:hover,a.btn:hover{background:#30353d}
.bar{height:6px;background:#2a2f36;border-radius:4px;overflow:hidden;width:220px}
.bar i{display:block;height:100%;background:#49a24b}
.grp{border:1px solid #2c3138;border-radius:12px;padding:10px 12px;margin:12px 0;background:#1b1e23}
.grp.ok{border-color:#49a24b}
.grp h3{margin:0 0 6px;font-size:15px;font-weight:600}
.meta{color:#9aa4b2;font-size:13px}
.strips img{display:block;max-width:100%;border-radius:6px;background:#000;margin:4px 0}
.who{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 0}
.who .card{display:flex;flex-direction:column;align-items:center;gap:3px;background:#22262c;border:2px solid #333a44;border-radius:10px;padding:5px 7px;cursor:pointer;min-width:84px}
.who .card.cur{border-color:#49a24b}
.who .card img{height:44px;border-radius:4px}
.who .card b{font-weight:600;font-size:13px}
.special{background:#2b3240}
.piece{border-top:1px solid #2c3138;margin-top:8px;padding-top:8px}
h2{margin:6px 0 0;font-size:17px;font-weight:600}
</style>
<header>
  <div class="row">
    <select id="clip"></select>
    <span class="bar"><i id="fill"></i></span>
    <span id="pos"></span>
    <button id="onlynew">Только неподтверждённые</button>
    <a class="btn" href="/">По одному отрезку</a>
  </div>
  <h2>Это один человек?</h2>
  <div class="meta">Система сама собрала отрезки в людей. Если на всех кадрах внутри карточки один и тот же
    человек — нажмите «Да, один человек». Если склеились разные — «Разобрать по одному» и отметьте каждый отрезок.</div>
</header>
<div id="list" style="padding:4px 14px 60px"></div>
<script>
let clip=null,G=[],L={},NP=0,onlyNew=false,open_={};
const q=s=>document.querySelector(s);
const nameOf=l=>l==='?'?'Не разобрать':'Человек '+String(l).replace('P','');
function people(){
  return [...new Set(Object.values(L))].filter(v=>v&&v!=='?').sort((a,b)=>
    (+String(a).replace(/\D/g,'')||0)-(+String(b).replace(/\D/g,'')||0));
}
function newName(){
  const n=people().filter(x=>/^P\d+$/.test(x)).map(x=>+x.slice(1));
  return 'P'+((n.length?Math.max(...n):0)+1);
}
async function loadClips(){
  const cs=await (await fetch('/api/clips')).json();
  q('#clip').innerHTML=cs.map(c=>`<option value="${c.clip}">${c.start.slice(11,16)} · ${Math.round(c.seconds/60)} мин · ${c.pieces} отрезков</option>`).join('');
  q('#clip').onchange=()=>openClip(q('#clip').value);
  if(cs.length) openClip(cs[0].clip);
}
async function openClip(c){
  clip=c; const d=await (await fetch('/api/groups/'+c)).json();
  G=d.groups; L=d.labels; NP=d.pieces; open_={}; draw();
}
function state(g){
  const labs=g.pieces.map(i=>L[i]).filter(Boolean);
  if(labs.length!==g.pieces.length) return {done:false,label:null};
  const uniq=[...new Set(labs)];
  return {done:uniq.length===1,label:uniq.length===1?uniq[0]:null,mixed:uniq.length>1};
}
function card(g,label,text,special){
  const st=state(g);
  const img=special?'':`<img src="/ref/${clip}/${encodeURIComponent(label)}" onerror="this.style.display='none'">`;
  return `<div class="card ${st.label===label?'cur':''} ${special?'special':''}"
    onclick="putAll(${g.person===null?'null':g.person},${JSON.stringify(g.pieces)},'${label}')">${img}
    <b>${text||nameOf(label)}</b></div>`;
}
function pieceCard(g,pi,label,text,special){
  return `<div class="card ${L[pi]===label?'cur':''} ${special?'special':''}"
    onclick="putOne(${pi},'${label}')"><b>${text||nameOf(label)}</b></div>`;
}
function draw(){
  const doneG=G.filter(g=>state(g).done).length;
  q('#pos').textContent=`подтверждено ${doneG} из ${G.length} человек · ${Object.keys(L).length} отрезков из ${NP}`;
  q('#fill').style.width=(100*doneG/Math.max(1,G.length))+'%';
  const ppl=people();
  const shown=G.filter(g=>!onlyNew||!state(g).done);
  q('#list').innerHTML=shown.map((g,n)=>{
    const st=state(g), key=G.indexOf(g);
    const strips=g.detail.map(d=>`<img src="/crop/${clip}/${d.piece}" title="отрезок ${d.piece}, камера ${d.cam.replace('cam','')}, ${d.t0}-${d.t1} с">`).join('');
    const buttons=ppl.map(p=>card(g,p)).join('')+card(g,'NEW','Да, один человек (новый)',true)+card(g,'?','Не разобрать',true);
    const detail=open_[key]?g.detail.map(d=>`<div class="piece">
        <div class="meta">отрезок ${d.piece} · камера ${d.cam.replace('cam','')} · ${d.t0}–${d.t1} с · ${d.n_dets} кадров
          ${L[d.piece]?'· сейчас: '+nameOf(L[d.piece]):''}</div>
        <img src="/crop/${clip}/${d.piece}" style="max-width:100%;border-radius:6px;background:#000;margin:4px 0">
        <div class="who">${ppl.map(p=>pieceCard(g,d.piece,p)).join('')}
          ${pieceCard(g,d.piece,'NEW','Новый человек',true)}${pieceCard(g,d.piece,'?','Не разобрать',true)}</div>
      </div>`).join(''):'';
    return `<div class="grp ${st.done?'ok':''}">
      <h3>${st.done?'✓ '+nameOf(st.label)+' — ':''}${g.detail.length} отрезков · с ${g.t0} по ${g.t1} секунду · камеры ${g.cams.map(c=>c.replace('cam','')).join(' и ')} · ${g.dets} кадров</h3>
      <div class="strips">${strips}</div>
      <div class="who">${buttons}</div>
      <div class="row" style="margin-top:8px">
        <button onclick="toggle(${key})">${open_[key]?'Свернуть':'Разобрать по одному'}</button>
      </div>${detail}</div>`;
  }).join('') || '<div class="meta">Здесь пусто — всё подтверждено.</div>';
}
function toggle(k){open_[k]=!open_[k];draw();}
async function putAll(person,pieces,lab){
  if(lab==='NEW')lab=newName();
  await fetch('/api/label_many/'+clip,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({pieces:pieces,label:lab})});
  pieces.forEach(i=>L[i]=lab); draw();
}
async function putOne(piece,lab){
  if(lab==='NEW')lab=newName();
  await fetch('/api/label/'+clip,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({piece:piece,label:lab})});
  L[piece]=lab; draw();
}
q('#onlynew').onclick=()=>{onlyNew=!onlyNew;q('#onlynew').textContent=onlyNew?'Показать всех':'Только неподтверждённые';draw();};
loadClips();
</script>"""


@app.get('/people')
def people_page():
    return PEOPLE_PAGE


@app.get('/')
def index():
    return PAGE


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5070, threaded=True)
