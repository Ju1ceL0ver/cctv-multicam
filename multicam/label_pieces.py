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


@app.get('/')
def index():
    return PAGE


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5070, threaded=True)
