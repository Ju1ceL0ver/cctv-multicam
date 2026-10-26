"""/visitpairs: the owner names the people behind the live door's events -- the ground truth of visits.

Every entry and exit of the day in time order. For each event the owner picks who it is: one of the people
already named (those inside now and most alike first) or a new person. A person's events in order are their
visits (in -> out, in -> out ...), with returns and missed events visible on the day's ribbon.
Answers -> data/visits/<day>/people_answers.json {event key: person number | "x" (nobody passed), "_staff": [person numbers]}."""
import json, time, threading
from pathlib import Path

PAGE = r'''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Кто есть кто</title>
<style>
:root{--bg:#121212;--card:#1d1d1d;--fg:#eee;--mut:#9a9a9a;--acc:#2f9e6b;--line:#333}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px system-ui;padding-bottom:150px}
.top{padding:12px 20px;border-bottom:1px solid var(--line)}
.q{font-size:22px;font-weight:600}.hint{color:var(--mut);margin-top:4px}
.wrap{display:flex;flex-direction:column;gap:16px;padding:12px 16px;max-width:1400px;margin:0 auto}
.ev{background:var(--card);border-radius:12px;padding:12px;text-align:center}
.ev h3{margin:0 0 8px;font-size:26px}.in{color:#5fd38d}.out{color:#f0a35e}
.ev .main{display:block;margin:0 auto;width:100%;max-width:1280px;max-height:62vh;object-fit:contain;border-radius:8px;background:#000}
.zoom{position:relative;overflow:hidden;aspect-ratio:16/9;width:100%;max-width:1100px;max-height:64vh;margin:0 auto;border-radius:8px;background:#000;cursor:zoom-out}
.zoom>.main{position:absolute;max-width:none;max-height:none;border-radius:0}
.zoom.whole{aspect-ratio:16/9;cursor:zoom-in}
.vb{padding:8px 14px;background:#2a2a2a;color:var(--fg);border:2px solid #444;border-radius:8px;font-size:14px;cursor:pointer}.zoom.whole>.main{width:100%!important;left:0!important;top:0!important}
.strip{display:flex;gap:6px;margin-top:8px;justify-content:center;flex-wrap:wrap}.strip img{height:200px;max-width:120px;object-fit:contain;border-radius:6px;background:#000}
@media(max-width:800px){.strip img{height:130px}}
.row{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:10px}.row .acts{flex-direction:row;margin:0}.row .nav{margin:0;flex:0 0 260px}
.nav{display:flex;gap:8px;margin-top:10px}.nav button{flex:1;padding:9px;background:#2a2a2a;color:var(--fg);border:0;border-radius:8px;font-size:14px;cursor:pointer}
.sec{color:var(--mut);margin:4px 0 8px;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:14px}
.p{background:var(--card);border:3px solid transparent;border-radius:12px;padding:7px;cursor:pointer;position:relative}
.p:hover{border-color:#555}.p.on{border-color:#fff}
.p img{width:100%;height:300px;object-fit:contain;background:#000;border-radius:8px}
.id{position:absolute;top:10px;left:10px;font-size:20px;font-weight:800;min-width:40px;height:40px;padding:0 8px;border-radius:20px;display:flex;align-items:center;justify-content:center;color:#000}
.k{position:absolute;top:10px;right:10px;background:#000c;border-radius:6px;padding:2px 8px;font-size:15px}
.st{font-size:14px;margin-top:5px}.st span{color:var(--mut)}
.new{background:var(--card);border:3px dashed #666;border-radius:12px;padding:16px;text-align:center;font-size:18px;cursor:pointer;margin-bottom:14px}
.new.on{border-style:solid;border-color:#fff}
.acts{display:flex;flex-direction:column;gap:8px;margin-top:10px}.acts button{padding:10px;background:#2a2a2a;color:var(--fg);border:2px solid #444;border-radius:8px;font-size:15px;cursor:pointer;text-align:left}.acts button.on{border-color:#fff}
.rib{position:fixed;left:0;right:0;bottom:0;background:#0b0b0bf0;border-top:1px solid var(--line);padding:8px 12px;overflow-x:auto;white-space:nowrap}
.rib .e{display:inline-block;width:46px;margin-right:3px;text-align:center;cursor:pointer;font-size:11px;color:var(--mut);vertical-align:top}
.rib .e img{width:46px;height:62px;object-fit:cover;border-radius:4px;border:3px solid #333;display:block}
.rib .e.cur img{outline:2px solid #fff}.rib .e b{display:block;font-size:13px;color:#000;border-radius:3px;margin-top:2px}
</style></head><body>
<div class="top"><div class="q">Кто это? Выбери человека или «Новый».</div>
<div class="hint">Все входы и выходы по порядку. Нажми на того же человека (цифра на карточке — клавиша), <b>N</b> — новый человек, <b>X</b> — никто не прошёл (ложное), <b>S</b> — отметить, что это сотрудник.
Внизу лента дня: цвет — кто это; нажми на кадр, чтобы исправить. <span id="cnt"></span></div></div>
<div id="m" class="wrap">загрузка…</div><div class="rib" id="rib"></div>
<script>
const q=new URLSearchParams(location.search);let day=q.get('day')||'';let D=null,A={},i=0,keys=[],STAFF=new Set(),ZW=false,VID=null;
const pic=(e)=>e.ev_views&&e.ev_views.length?`/api/visitpairs/evview?day=${D.day}&n=${e.name}&f=${e.ev_views[0]}`:e.crop?`/api/visitpairs/file?day=${D.day}&f=${encodeURIComponent(e.crop)}`:'';
const full=(e)=>e.photo?`/api/visitpairs/file?day=${D.day}&f=${encodeURIComponent(e.photo)}`:'';
const ctx=(e)=>e.crop?`/api/visitpairs/file?day=${D.day}&f=${encodeURIComponent(e.crop)}`:'';
const view=(v)=>`/api/visitpairs/view?day=${D.day}&f=${encodeURIComponent(v)}`;
const vid=(e)=>`/api/visitpairs/clip?day=${D.day}&n=${e.name}`;
// 21.10: zoom on the person -- a 16:9 window ~1.7 heights of the event box, centred on it, kept inside the frame
const fz=(e)=>{const b=e.focus;if(!b)return 'width:100%;left:0;top:0';const bh=b[3]-b[1],cx=(b[0]+b[2])/2,cy=(b[1]+b[3])/2;
 let h=Math.min(1,Math.max(.3,1.7*bh)),w=h;const x=Math.min(1-w,Math.max(0,cx-w/2)),y=Math.min(1-h,Math.max(0,cy-h/2));
 return `width:${100/w}%;left:${-x/w*100}%;top:${-y/h*100}%`};
const col=(p)=>`hsl(${(p*137.5)%360},70%,62%)`;const hm=(c)=>c.slice(11,16);
async function load(){D=await (await fetch('/api/visitpairs?day='+day)).json();A=D.answers||{};STAFF=new Set((A._staff||[]).map(Number));delete A._staff;
 i=D.timeline.findIndex(e=>!(e.key in A));if(i<0)i=D.timeline.length;draw();}
function people(upto){const P={};D.timeline.forEach((e,k)=>{const p=A[e.key];if(p==null||p==='x'||k>=upto)return;
 (P[p]=P[p]||{id:+p,ev:[]}).ev.push(k)});return Object.values(P)}
function draw(){const T=D.timeline,N=T.length;cnt.textContent=` Размечено ${Object.keys(A).length} из ${N}.`;drawRib();
 if(i>=N){m.innerHTML=`<div style="padding:40px;font-size:22px">Готово — все ${N} событий размечены. Можно исправлять по ленте внизу.</div>`;return}
 const e=T[i],a=A[e.key];const P=people(i).concat(people(N).filter(p=>!people(i).some(q=>q.id===p.id)&&p.ev.some(k=>k>i)));
 P.forEach(p=>{const last=T[p.ev.filter(k=>k<i).slice(-1)[0]??p.ev[0]];p.last=last;
  p.inside=last.kind==='in'&&p.ev.some(k=>k<i);p.sim=Math.max(...p.ev.map(k=>D.sim[i][k]??-1))});
 // an exit looks for people inside, an entry for people outside; then by likeness
 P.sort((x,y)=>((y.inside===(e.kind==='out'))-(x.inside===(e.kind==='out')))||(y.sim-x.sim));
 keys=P.slice(0,9);
 let h=`<div class="ev"><h3 class="${e.kind}">${e.kind==='in'?'Вошёл':'Вышел'} в ${hm(e.clock)}</h3>${(()=>{const vid=e.clip&&(VID??!e.photo);   // 21.10 owner: the photo first wherever there is one; the clip on the button
 return `<div class="zoom${ZW?' whole':''}" title="щелчок — человек крупно / весь кадр" onclick="ZW=!ZW;this.classList.toggle('whole',ZW)">${vid?`<video class="main" style="${fz(e)}" src="/api/visitpairs/clip?day=${D.day}&n=${e.name}" autoplay loop muted playsinline></video>`:`<img class="main" style="${fz(e)}" src="${e.photo?full(e):pic(e)}">`}</div>
 ${!e.clip?`<div class="sec" style="margin-top:6px;color:#f0a35e">видео этого события нет; кадр может быть не из того момента — смотри кадры человека ниже</div>`:''}
 ${e.clip&&e.photo?`<div class="row"><button class="vb" onclick="VID=${!vid};draw()">${vid?'показать фото':'показать видео (если в рамке пусто)'}</button></div>`:''}`})()}
 ${e.ev_views&&e.ev_views.length?`<div class="strip">${e.ev_views.map(f=>`<img src="/api/visitpairs/evview?day=${D.day}&n=${e.name}&f=${f}">`).join('')}</div>`:''}
 <div class="row"><div class="acts"><button class="${a==='x'?'on':''}" onclick="say('x')"><b>X</b> — никто не прошёл</button>
 <button class="${a!=null&&a!=='x'&&STAFF.has(+a)?'on':''}" onclick="staff()"><b>S</b> — это сотрудник</button></div>
 <div class="nav"><button onclick="go(-1)">← назад</button><button onclick="go(1)">пропустить →</button></div></div></div><div>`;
 h+=`<div class="new ${a!=null&&!P.some(p=>p.id==a)?'on':''}" onclick="say('new')"><b>N</b> — Новый человек (раньше не встречался)</div>`;
 const grp=(title,list)=>{if(!list.length)return'';let s=`<div class="sec">${title}</div><div class="grid">`;
  list.forEach(p=>{const n=keys.indexOf(p);s+=`<div class="p ${a==p.id?'on':''}" onclick="say(${p.id})"><div class="id" style="background:${col(p.id)}">${p.id}</div>
  ${n>=0?`<div class="k">${n+1}</div>`:''}<img src="${pic(p.last)}"><div class="st">${p.inside?'внутри с '+hm(p.last.clock):(p.last.kind==='out'?'вышел в '+hm(p.last.clock):'вошёл в '+hm(p.last.clock))}
  <span>· событий ${p.ev.length}</span>${STAFF.has(p.id)?' <b style="color:#7cc4ff">сотрудник</b>':''}</div></div>`});return s+'</div>'};
 const want=P.filter(p=>p.inside===(e.kind==='out')),rest=P.filter(p=>p.inside!==(e.kind==='out'));
 h+=grp(e.kind==='out'?'Сейчас внутри — скорее всего кто-то из них':'Уже выходили — может, вернулся',want);
 h+=grp(e.kind==='out'?'Остальные':'Остальные',rest);m.innerHTML=h+'</div>';scrollTo(0,0);}
function drawRib(){rib.innerHTML=D.timeline.map((e,k)=>{const p=A[e.key];return `<div class="e ${k===i?'cur':''}" onclick="i=${k};draw()">
 <img src="${pic(e)}" style="border-color:${p==='x'?'#555':p!=null?col(+p):'#333'}"><b style="background:${p!=null&&p!=='x'?col(+p):'transparent'};color:${p!=null&&p!=='x'?'#000':'#777'}">${p==='x'?'✕':p!=null?p+(STAFF.has(+p)?'с':''):'?'}</b>${e.kind==='in'?'↓':'↑'}${hm(e.clock)}</div>`}).join('');
 const c=rib.children[i];if(c)c.scrollIntoView({inline:'center',block:'nearest'});}
function go(d){i=Math.max(0,Math.min(D.timeline.length,i+d));draw()}
function staff(){const e=D.timeline[i];if(!e)return;let p=A[e.key];let fresh=false;if(p==null||p==='x'){say('new');p=A[e.key];fresh=true}
 p=+p;fresh?STAFF.add(p):(STAFF.has(p)?STAFF.delete(p):STAFF.add(p));
 fetch('/api/visitpairs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({day:D.day,staff:[...STAFF]})});draw()}
function say(v){const e=D.timeline[i];if(v==='new'){const used=Object.values(A).filter(x=>x!=='x').map(Number);v=used.length?Math.max(...used)+1:1;
 if(A[e.key]!=null&&!Object.entries(A).some(([k,x])=>k!==e.key&&x==A[e.key]))v=A[e.key]}
 A[e.key]=v;fetch('/api/visitpairs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({day:D.day,event:e.key,person:v})});
 i++;while(i<D.timeline.length&&(D.timeline[i].key in A))i++;draw();}
document.onkeydown=(ev)=>{if(!D||i>=D.timeline.length)return;
 if(ev.key>='1'&&ev.key<='9'){const p=keys[+ev.key-1];if(p)say(p.id)}
 else if(ev.key==='n'||ev.key==='N'||ev.key==='т'||ev.key==='Т')say('new');
 else if(ev.key==='x'||ev.key==='X'||ev.key==='ч'||ev.key==='Ч')say('x');
 else if(ev.key==='s'||ev.key==='S'||ev.key==='ы'||ev.key==='Ы')staff();
 else if(ev.key==='ArrowLeft')go(-1);else if(ev.key==='ArrowRight')go(1)};
load();
</script></body></html>'''


def visits_from(timeline, answers):
    """Owner's people -> visits (in..out in order) and the share of system pairs that agree."""
    per = {}
    for e in timeline:
        p = answers.get(e['key'])
        if p is not None:
            per.setdefault(p, []).append(e)
    return per


LOCK = threading.Lock()          # answers come fast, one read-modify-write at a time


def register(app, root):
    from door_rank_page import register as register_door_rank
    register_door_rank(app, root)
    from flask import jsonify, request, send_file, abort

    def folder(day):
        return Path(root()) / 'data' / 'visits' / day

    def latest_day():
        days = sorted(p.name for p in (Path(root()) / 'data' / 'visits').glob('2*') if (p / 'pairs.json').exists())
        return days[-1] if days else time.strftime('%Y%m%d')

    def read(p, default):
        return json.load(open(p, encoding='utf-8')) if p.exists() else default

    def live_timeline(day):
        """every event of the live door that day, in time order, straight from its events file"""
        out = []
        f = Path(root()) / 'data' / 'live' / 'events_cam1.jsonl'
        for line in open(f, encoding='utf-8'):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get('engine') != 'doorseg_c+zones' or not e['clock'].replace('-', '').startswith(day):
                continue
            out.append({'key': '%.1f' % e['t'], 'kind': e['kind'], 'clock': e['clock'], 't': e['t'],
                        'crop': e.get('crop'), 'photo': e.get('photo'), 'views': [],
                        'name': '%s_%s_%.1f' % (e['kind'], e['clock'][11:19].replace(':', ''), float(e['t']))})
        return sorted(out, key=lambda e: e['t'])

    def photo_boxes(day, timeline):
        """21.10: the green box the live door drew on the event's photo, as fractions of the frame -- the page zooms
        on it. Found once per photo, cached in data/visits/<day>/photo_boxes.json."""
        import cv2, numpy as np
        from storage import atomic_json
        cp = folder(day) / 'photo_boxes.json'
        cache = read(cp, {})
        new = False
        for e in timeline:
            if e['key'] in cache or not e.get('photo') or not Path(e['photo']).exists():
                continue
            img = cv2.imread(e['photo'])
            box = None
            if img is not None:
                m = (np.abs(img.astype(np.int16) - (60, 220, 60)).max(axis=2) < 40).astype(np.uint8)
                cn, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cn:
                    x, y, w, h = cv2.boundingRect(max(cn, key=cv2.contourArea))
                    if w > 10 and h > 30:
                        H, W = img.shape[:2]
                        box = [round(x / W, 4), round(y / H, 4), round((x + w) / W, 4), round((y + h) / H, 4)]
            cache[e['key']] = box
            new = True
        if new:
            folder(day).mkdir(parents=True, exist_ok=True)
            atomic_json(cp, cache)
        return cache

    def event_view_sim(day, timeline):
        import numpy as np
        f = Path(root()) / 'data' / 'door_identity' / day / 'live_features.npz'
        if not f.exists():
            return None
        try:
            F = dict(np.load(f))
        except (OSError, ValueError):
            return None
        evv = Path(root()) / 'data' / 'live' / 'event_views' / day
        V = []
        for e in timeline:
            m = read(evv / e['name'] / 'manifest.json', {})
            rows = [np.r_[np.sqrt(.5) * F['c_' + v['sha256']], np.sqrt(.5) * F['s_' + v['sha256']]]
                    for v in m.get('views', []) if 'c_' + v['sha256'] in F and 's_' + v['sha256'] in F]
            V.append(np.asarray(rows) if rows else None)
        from door_identity import similarity
        return [[round(similarity(a, b), 3) if a is not None and b is not None else None for b in V] for a in V]

    @app.get('/api/visitpairs/evview')
    def visitpairs_evview():
        day = request.args.get('day', ''); name = request.args.get('n', ''); f = request.args.get('f', '')
        if not day.isdigit() or not all(c.isalnum() or c in '_.' for c in name + f) or not f.endswith('.jpg') or '..' in name + f:
            abort(400)
        p = Path(root()) / 'data' / 'live' / 'event_views' / day / name / f
        if not p.exists():
            abort(404)
        return send_file(str(p), mimetype='image/jpeg', max_age=86400)

    @app.get('/api/visitpairs/clip')
    def visitpairs_clip():
        day = request.args.get('day', ''); name = request.args.get('n', '')
        if not day.isdigit() or not all(c.isalnum() or c in '_.' for c in name):
            abort(400)
        p = Path(root()) / 'data' / 'live' / 'clips_seg' / day / (name + '.mp4')
        if not p.exists():
            abort(404)
        return send_file(str(p), mimetype='video/mp4', conditional=True, max_age=3600)

    @app.get('/visitpairs')
    def visitpairs_page():
        return PAGE

    @app.get('/api/visitpairs')
    def visitpairs_get():
        day = request.args.get('day') or latest_day()
        if not day.isdigit():
            abort(400)
        pairs = read(folder(day) / 'pairs.json', {})
        timeline = live_timeline(day)
        old = {e['key']: k for k, e in enumerate(pairs.get('timeline', []))}
        S = pairs.get('sim', [])
        sim = [[(S[old[a['key']]][old[b['key']]] if a['key'] in old and b['key'] in old else None) for b in timeline] for a in timeline]
        clips = Path(root()) / 'data' / 'live' / 'clips_seg' / day
        evv = Path(root()) / 'data' / 'live' / 'event_views' / day
        for e in timeline:
            e['clip'] = (clips / (e['name'] + '.mp4')).exists()
            m = read(evv / e['name'] / 'manifest.json', {})
            e['ev_views'] = [v['path'] for v in m.get('views', [])]
        focus = photo_boxes(day, timeline)
        for e in timeline:
            e['focus'] = focus.get(e['key'])
        live_sim = event_view_sim(day, timeline)        # 20.10: likeness from the event's own 8 views, when computed
        if live_sim is not None:
            sim = [[live_sim[a][b] if live_sim[a][b] is not None else sim[a][b] for b in range(len(timeline))] for a in range(len(timeline))]
        return jsonify({'day': day, 'timeline': timeline, 'sim': sim,
                        'answers': read(folder(day) / 'people_answers.json', {})})

    @app.post('/api/visitpairs')
    def visitpairs_post():
        from storage import atomic_json
        b = request.get_json(force=True)
        day = str(b.get('day', ''))
        if not day.isdigit():
            abort(400)
        p = folder(day) / 'people_answers.json'
        with LOCK:
            return _save(p, b)

    def _save(p, b):
        from storage import atomic_json
        ans = read(p, {})
        if 'staff' in b:
            ans['_staff'] = sorted({int(x) for x in b['staff']})
        elif b.get('person') is None:
            ans.pop(str(b['event']), None)
        elif b['person'] == 'x':
            ans[str(b['event'])] = 'x'
        else:
            ans[str(b['event'])] = int(b['person'])
        atomic_json(p, ans)
        return jsonify({'ok': True, 'n': len(ans)})

    @app.get('/api/visitpairs/file')
    def visitpairs_file():
        f = Path(request.args.get('f', ''))
        snaps = (Path(root()) / 'data' / 'live' / 'snapshots').resolve()
        try:
            p = f.resolve()
            p.relative_to(snaps)
        except (ValueError, OSError):
            abort(400)
        if p.suffix != '.jpg' or not p.exists():
            abort(404)
        return send_file(str(p), mimetype='image/jpeg', max_age=86400)

    @app.get('/api/visitpairs/view')
    def visitpairs_view():
        day = request.args.get('day', ''); name = request.args.get('f', '')
        if not day.isdigit() or not name.endswith('.jpg') or '/' in name or '\\' in name or '..' in name:
            abort(400)
        p = Path(root()) / 'data' / 'live' / 'views_seg' / day / name
        if not p.exists():
            abort(404)
        return send_file(str(p), mimetype='image/jpeg', max_age=86400)
