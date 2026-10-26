"""Appearance pair review, separate from owner identities and evaluation holdout."""
import hashlib
import json
from pathlib import Path
from storage import atomic_json, read_json, file_lock

LABELS={'same','similar_strong','similar_some','different','unclear'}


def build_queue(day,root):
    """Only train events; never expose held-out images as training pairs."""
    import numpy as np
    from door_identity import event_vectors, similarity
    folder=Path(root)/'data/door_identity'/day
    index=read_json(folder/'index.json',{})
    allowed=set(index['split']['train_keys'])
    events=[e for e in index['events'] if e['key'] in allowed]
    with np.load(folder/'features.npz') as cache:
        vectors=event_vectors(events,cache,'raw',.5)
    def image(e):
        manifest=read_json(e['manifest'],{})
        views=manifest.get('views',[])
        view=next((v for v in views if v.get('variant','raw')=='raw'),None)
        if not view:return None
        return dict(event=e['key'],clock=e['clock'],kind=e['kind'],sha256=view['sha256'],
                    path=str(Path(e['manifest']).parent/view['path']))
    entries=[];pairs=[]
    for e in events:
        if e['kind']=='out' and e['key'] in vectors:
            ranked=sorted([(similarity(vectors[e['key']],vectors[a['key']]),a) for a in entries
                           if a['key'] in vectors],key=lambda a:a[0],reverse=True)
            # Close impostors and an easy contrast, with no owner-answer prefill.
            selected=ranked[:3]+(ranked[-1:] if len(ranked)>3 else [])
            for _,a in selected:
                left,right=image(a),image(e)
                if not left or not right:continue
                digest=hashlib.sha256(json.dumps([a['key'],e['key'],left['sha256'],right['sha256']]).encode()).hexdigest()[:24]
                pairs.append(dict(id=digest,left=left,right=right))
        if e['kind']=='in':entries.append(e)
    value=dict(day=day,policy='train_only',pairs=pairs)
    atomic_json(folder/'rank_queue.json',value)
    return value


PAGE='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Сходство людей</title><style>
body{font:17px system-ui;background:#151515;color:#eee;margin:0;padding:20px;max-width:1000px;margin:auto}h1{font-size:24px}p{color:#bbb}.photos{display:grid;grid-template-columns:1fr 1fr;gap:16px}.photos img{width:100%;height:360px;object-fit:contain;background:#222}.actions{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0}button{font:inherit;padding:14px;border:1px solid #777;border-radius:8px;background:#292929;color:white;cursor:pointer}button:disabled{opacity:.4}button:focus-visible{outline:3px solid #7fc6ff}#status{min-height:26px}#same{background:#246846}@media(max-width:600px){body{padding:12px}.photos img{height:260px}.actions button{flex:1 1 45%}}
</style><h1>Один человек или разные?</h1><p>«Тот же» — одна личность. Остальные оценки — разные люди с разной внешней похожестью. Если не видно — «Не могу определить».</p>
<div id="progress"></div><div class="photos"><div><p id="lc"></p><img id="left" alt="Человек на входе"></div><div><p id="rc"></p><img id="right" alt="Человек на выходе"></div></div>
<div class="actions" id="actions"><button id="same" data-label="same">1 · Тот же человек</button><button data-label="similar_strong">2 · Другой, сильно похож</button><button data-label="similar_some">3 · Другой, немного похож</button><button data-label="different">4 · Не похож</button><button data-label="unclear">5 · Не могу определить</button></div>
<div id="status" role="status" aria-live="polite"></div><button id="back">← Предыдущая пара</button><script>
const params=new URLSearchParams(location.search),day=params.get('day')||'20261014',key=params.get('key')||'';
const url=(path)=>path+(path.includes('?')?'&':'?')+new URLSearchParams({day,key});
let data=null,index=0,busy=false;
const el=id=>document.getElementById(id);
function show(){const pair=data.pairs[index];el('progress').textContent=`Пара ${Math.min(index+1,data.pairs.length)} из ${data.pairs.length} · оценено ${Object.keys(data.answers).length}`;
 el('actions').hidden=!pair;el('back').disabled=index===0;
 if(!pair){el('status').textContent='Готово. Все пары оценены.';el('left').removeAttribute('src');el('right').removeAttribute('src');return}
 for(const side of ['left','right'])el(side).src=url(`/api/doorrank/image?id=${encodeURIComponent(pair.id)}&side=${side}`);
 el('lc').textContent='Вход · '+pair.left.clock;el('rc').textContent='Выход · '+pair.right.clock;
 el('status').textContent=data.answers[pair.id]?'Оценка сохранена. Можно изменить.':'';}
async function answer(label){if(busy||!data?.pairs[index])return;busy=true;document.querySelectorAll('[data-label]').forEach(b=>b.disabled=true);
 try{const pair=data.pairs[index],r=await fetch(url('/api/doorrank'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:pair.id,label})});if(!r.ok)throw Error('Сохранение не удалось');data.answers[pair.id]=label;index++;show()}
 catch(e){el('status').textContent=e.message+' — повторите';}finally{busy=false;document.querySelectorAll('[data-label]').forEach(b=>b.disabled=false)}}
 document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>answer(b.dataset.label));el('back').onclick=()=>{if(!busy){index=Math.max(0,index-1);show()}};
 document.addEventListener('keydown',e=>{if(e.repeat)return;const labels=['same','similar_strong','similar_some','different','unclear'];if(e.key>='1'&&e.key<='5')answer(labels[+e.key-1]);});
 fetch(url('/api/doorrank')).then(r=>{if(!r.ok)throw Error('Нет подготовленных пар');return r.json()}).then(d=>{data=d;const next=d.pairs.findIndex(p=>!Object.hasOwn(d.answers,p.id));index=next<0?d.pairs.length:next;show()}).catch(e=>el('status').textContent=e.message);
</script></html>'''


def register(app,root):
    from flask import request,jsonify,abort,send_file
    def folder():
        day=request.args.get('day','20261014')
        if len(day)!=8 or not day.isdigit():abort(400)
        return Path(root())/'data/door_identity'/day
    def queue():
        value=read_json(folder()/'rank_queue.json',{})
        if value.get('policy')!='train_only':abort(404)
        return value
    @app.get('/doorrank')
    def doorrank_page():return PAGE
    @app.get('/api/doorrank')
    def doorrank_get():
        value=queue();answers=read_json(folder()/'rank_answers.json',{})
        pairs=[dict(id=p['id'],left={k:v for k,v in p['left'].items() if k!='path'},
                    right={k:v for k,v in p['right'].items() if k!='path'}) for p in value['pairs']]
        return jsonify(pairs=pairs,answers={k:v['label'] for k,v in answers.items() if k in {p['id'] for p in pairs}})
    @app.post('/api/doorrank')
    def doorrank_post():
        b=request.get_json(force=True);pair=next((p for p in queue()['pairs'] if p['id']==b.get('id')),None)
        if pair is None or b.get('label') not in LABELS:abort(400)
        import time
        with file_lock(folder()/'rank_answers.lock'):
            path=folder()/'rank_answers.json';answers=read_json(path,{})
            answers[pair['id']]=dict(label=b['label'],at=time.time(),left=pair['left']['sha256'],right=pair['right']['sha256'])
            atomic_json(path,answers)
        return jsonify(ok=True)
    @app.get('/api/doorrank/image')
    def doorrank_image():
        pair=next((p for p in queue()['pairs'] if p['id']==request.args.get('id')),None);side=request.args.get('side')
        if pair is None or side not in ('left','right'):abort(400)
        view=pair[side];path=Path(view['path']).resolve()
        if not path.is_relative_to((folder()/'views').resolve()) or not path.is_file():abort(404)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=view['sha256']:abort(409)
        return send_file(str(path),mimetype='image/jpeg',max_age=86400)
