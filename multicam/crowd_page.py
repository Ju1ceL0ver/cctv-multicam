"""/crowdmasks: the owner's masks of crowds at the door, by clicks (15.10.2026).

Crowd events of the live door keep their raw frames and the model's masks (door_clips_seg.keep_crowd). Per event three
frames are asked: 1 s before the crossing, the crossing, 1 s after. On a frame the model's masks are drawn; the owner
clicks a wrong one (merged, not a person) to drop it, clicks a person nobody outlined to have the click segmenter
(SAM 2.1 tiny on the processor, day_masks.Segmenter) outline him, shift-click adds a part to the last one, alt-click
takes a part away; Enter keeps the frame. -> data/live/crowd_labels/<day>/<key>/<i>.json (+ .png, instance ids).
These are the truth that the door model and the SAM 3.1 teacher are measured against, and the crowd training data."""
import json, threading, time
from pathlib import Path

import numpy as np

LOCK = threading.Lock()
SEG = {'model': None}
AROUND = (-1.0, 0.0, 1.0)

PAGE = r'''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Маски толпы</title>
<style>
:root{--bg:#111;--fg:#eee;--mut:#999}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui}
.top{padding:10px 16px;border-bottom:1px solid #333}.q{font-size:20px;font-weight:600}.hint{color:var(--mut);margin-top:4px}
.wrap{padding:10px 16px;display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
canvas{max-width:100%;border-radius:6px;cursor:crosshair;background:#000}
.side{min-width:220px;max-width:300px}.side p{margin:6px 0}.k{display:inline-block;min-width:22px;text-align:center;background:#333;border-radius:4px;padding:0 5px;margin-right:4px}
button{padding:9px 12px;background:#2a2a2a;color:var(--fg);border:1px solid #444;border-radius:8px;cursor:pointer;margin:3px 0;width:100%;text-align:left}
#busy{color:#e7b416}
</style></head><body>
<div class="top"><div class="q">Отметь каждого человека у двери</div>
<div class="hint">Зелёные — найденные моделью. Нажми на <b>неверную</b> маску (два человека в одной, не человек) — она станет красной и уйдёт.
Нажми на <b>человека без маски</b> — он обведётся синим. Потом <span class="k">Enter</span>. <span id="cnt"></span></div></div>
<div class="wrap"><canvas id="c"></canvas><div class="side">
<p id="where"></p><p id="busy"></p>
<p><span class="k">клик</span>по человеку без маски — обвести</p>
<p><span class="k">клик</span>по зелёной — убрать (ещё раз — вернуть)</p>
<p><span class="k">Shift+клик</span>добавить часть к последнему синему</p>
<p><span class="k">Alt+клик</span>убрать лишнее у последнего синего</p>
<p><span class="k">Z</span>отменить последнее</p>
<button onclick="save()"><b>Enter</b> — кадр готов, дальше</button>
<button onclick="skip()"><b>S</b> — пропустить кадр (не понять)</button>
<button onclick="go(-1)">← назад</button>
</div></div>
<script>
const q=new URLSearchParams(location.search);let day=q.get('day')||'';let L=null,j=0,F=null,img=new Image(),drop=new Set(),added=[],hist=[];
const cv=document.getElementById('c'),cx=cv.getContext('2d');
async function load(){L=await (await fetch('/api/crowd/list?day='+day)).json();day=L.day;j=L.items.findIndex(x=>!x.done);if(j<0)j=L.items.length;open_()}
async function open_(){cnt.textContent=`Готово ${L.items.filter(x=>x.done).length} из ${L.items.length} кадров.`;
 if(j>=L.items.length){where.textContent='Всё размечено, спасибо!';cx.clearRect(0,0,cv.width,cv.height);return}
 const it=L.items[j];where.textContent=`${it.kind==='in'?'Вход':'Выход'} ${it.clock.slice(11)} · кадр ${it.at>0?'+':''}${it.at} с · ${j+1}/${L.items.length}`;
 F=await (await fetch(`/api/crowd/model?day=${day}&key=${it.key}&i=${it.i}`)).json();drop=new Set();added=[];hist=[];
 img.onload=()=>{cv.width=img.width;cv.height=img.height;draw()};img.src=`/api/crowd/frame?day=${day}&key=${it.key}&i=${it.i}`}
function path(poly){cx.beginPath();poly.forEach((p,k)=>k?cx.lineTo(p[0],p[1]):cx.moveTo(p[0],p[1]));cx.closePath()}
function inside(poly,x,y){let c=false;for(let a=0,b=poly.length-1;a<poly.length;b=a++){const[xi,yi]=poly[a],[xj,yj]=poly[b];if((yi>y)!=(yj>y)&&x<(xj-xi)*(y-yi)/(yj-yi)+xi)c=!c}return c}
function draw(){cx.drawImage(img,0,0);cx.lineWidth=3;
 F.masks.forEach((m,k)=>{const bad=drop.has(k);cx.fillStyle=bad?'rgba(230,50,50,.35)':'rgba(60,200,60,.30)';cx.strokeStyle=bad?'#e33':'#3c3';
  m.polys.forEach(p=>{path(p);cx.fill();cx.stroke()})});
 added.forEach(a=>{cx.fillStyle='rgba(60,130,255,.40)';cx.strokeStyle='#48f';(a.polys||[]).forEach(p=>{path(p);cx.fill();cx.stroke()});
  a.points.forEach((p,k)=>{cx.fillStyle=a.labels[k]?'#48f':'#f44';cx.beginPath();cx.arc(p[0],p[1],6,0,7);cx.fill()})})}
async function seg(a){busy.textContent='обвожу…';const it=L.items[j];
 const r=await (await fetch('/api/crowd/click',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({day,key:it.key,i:it.i,points:a.points,labels:a.labels})})).json();
 a.polys=r.polys||[];busy.textContent=r.error?('ошибка: '+r.error):'';draw()}
cv.onclick=async(e)=>{const r=cv.getBoundingClientRect(),x=(e.clientX-r.left)*cv.width/r.width,y=(e.clientY-r.top)*cv.height/r.height;
 if((e.shiftKey||e.altKey)&&added.length){const a=added[added.length-1];a.points.push([x,y]);a.labels.push(e.altKey?0:1);hist.push(['pt']);await seg(a);return}
 const k=F.masks.findIndex(m=>m.polys.some(p=>inside(p,x,y)));
 if(k>=0&&!added.some(a=>(a.polys||[]).some(p=>inside(p,x,y)))){drop.has(k)?drop.delete(k):drop.add(k);hist.push(['drop',k]);draw();return}
 const a={points:[[x,y]],labels:[1]};added.push(a);hist.push(['add']);await seg(a)}
function undo(){const h=hist.pop();if(!h)return;if(h[0]=='drop'){drop.has(h[1])?drop.delete(h[1]):drop.add(h[1])}
 else if(h[0]=='add')added.pop();else{const a=added[added.length-1];a.points.pop();a.labels.pop();if(a.points.length)return seg(a)}draw()}
async function save(skipped){const it=L.items[j];
 await fetch('/api/crowd/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({day,key:it.key,i:it.i,skip:!!skipped,
  keep:F.masks.map((m,k)=>k).filter(k=>!drop.has(k)),added:added.map(a=>({points:a.points,labels:a.labels}))})});
 it.done=true;j++;open_()}
function skip(){save(true)}function go(d){j=Math.max(0,Math.min(L.items.length,j+d));open_()}
document.onkeydown=(e)=>{if(e.key==='Enter')save();else if(e.key==='z'||e.key==='Z'||e.key==='я')undo();else if(e.key==='s'||e.key==='S'||e.key==='ы')skip();
 else if(e.key==='ArrowLeft')go(-1);else if(e.key==='ArrowRight')go(1)};
load();
</script></body></html>'''


def _polys(mask, scale=1.0):
    import cv2
    cn, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cn:
        if cv2.contourArea(c) < 30:
            continue
        c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2) * scale
        out.append([[round(float(x), 1), round(float(y), 1)] for x, y in c])
    return out


def model_masks(folder, i):
    meta = json.loads((folder / 'meta.json').read_text(encoding='utf-8'))
    z = np.load(folder / 'masks.npz')
    shape = meta.get('shape', [180, 320])
    out = []
    for j, pid in enumerate(meta['frames'][i]['pids']):
        k = 'f%04d_%d' % (i, j)
        if k in z.files:
            m = np.unpackbits(z[k])[:shape[0] * shape[1]].reshape(shape).astype(bool)
            out.append((pid, m))
    return out


def register(app, root):
    from flask import jsonify, request, send_file, abort

    def crowd(day):
        return Path(root()) / 'data' / 'live' / 'crowd' / day

    def labels(day):
        return Path(root()) / 'data' / 'live' / 'crowd_labels' / day

    def ok_day(day):
        if not day.isdigit():
            abort(400)
        return day

    def ok_key(key):
        if not key or not all(c.isalnum() or c in '_.' for c in key):
            abort(400)
        return key

    @app.get('/crowdmasks')
    def crowd_page():
        return PAGE

    @app.get('/api/crowd/list')
    def crowd_list():
        day = request.args.get('day') or ''
        if not day:
            days = sorted(p.name for p in (Path(root()) / 'data' / 'live' / 'crowd').glob('2*'))
            day = days[-1] if days else time.strftime('%Y%m%d')
        ok_day(day)
        items = []
        for d in sorted(crowd(day).glob('*')) if crowd(day).exists() else []:
            try:
                meta = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            ev = meta['event']
            ts = np.array([f['t'] for f in meta['frames']])
            for at in AROUND:
                i = int(np.argmin(np.abs(ts - (ev['t'] + at))))
                items.append({'key': d.name, 'i': i, 'at': at, 'kind': ev['kind'], 'clock': ev.get('clock', ''),
                              'done': (labels(day) / d.name / ('%04d.json' % i)).exists()})
        items.sort(key=lambda x: (x['clock'], x['at']))
        return jsonify({'day': day, 'items': items})

    @app.get('/api/crowd/frame')
    def crowd_frame():
        day, key = ok_day(request.args.get('day', '')), ok_key(request.args.get('key', ''))
        p = crowd(day) / key / ('%04d.jpg' % int(request.args.get('i', 0)))
        if not p.exists():
            abort(404)
        return send_file(str(p), mimetype='image/jpeg', max_age=86400)

    @app.get('/api/crowd/model')
    def crowd_model():
        day, key = ok_day(request.args.get('day', '')), ok_key(request.args.get('key', ''))
        i = int(request.args.get('i', 0))
        return jsonify({'masks': [{'pid': pid, 'polys': _polys(m, 4.0)} for pid, m in model_masks(crowd(day) / key, i)]})

    def segmenter():
        if SEG['model'] is None:
            import day_masks
            SEG['model'] = day_masks.Segmenter()
        return SEG['model']

    def sam_mask(day, key, i, points, labels_):
        import cv2
        img = cv2.imread(str(crowd(day) / key / ('%04d.jpg' % i)))
        if img is None:
            abort(404)
        s = segmenter()
        s.encode('crowd', (day, key, i), img)
        return s.mask('crowd', points, labels_)

    @app.post('/api/crowd/click')
    def crowd_click():
        b = request.get_json(force=True)
        day, key = ok_day(str(b.get('day', ''))), ok_key(str(b.get('key', '')))
        try:
            with LOCK:
                m = sam_mask(day, key, int(b['i']), b['points'], b['labels'])
        except Exception as exc:
            return jsonify({'error': str(exc)[:200]})
        return jsonify({'polys': _polys(m) if m is not None else []})

    @app.post('/api/crowd/save')
    def crowd_save():
        import cv2
        from storage import atomic_json
        b = request.get_json(force=True)
        day, key = ok_day(str(b.get('day', ''))), ok_key(str(b.get('key', '')))
        i = int(b['i'])
        out = labels(day) / key
        out.mkdir(parents=True, exist_ok=True)
        rec = {'day': day, 'key': key, 'i': i, 'skip': bool(b.get('skip')), 'keep': b.get('keep', []),
               'added': b.get('added', []), 'at': time.strftime('%Y-%m-%d %H:%M:%S')}
        if not rec['skip']:                        # instance map 1280x720: kept model masks first, then the clicked ones
            model = model_masks(crowd(day) / key, i)
            ids = np.zeros((720, 1280), np.uint8)
            n = 0
            for k in rec['keep']:
                if 0 <= k < len(model):
                    n += 1
                    big = cv2.resize(model[k][1].astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST) > 0
                    ids[big & (ids == 0)] = n
            with LOCK:
                for a in rec['added']:
                    m = sam_mask(day, key, i, a['points'], a['labels'])
                    if m is not None:
                        n += 1
                        ids[m & (ids == 0)] = n
            cv2.imwrite(str(out / ('%04d.png' % i)), ids)
            rec['people'] = n
        atomic_json(out / ('%04d.json' % i), rec)
        return jsonify({'ok': True, 'people': rec.get('people')})
