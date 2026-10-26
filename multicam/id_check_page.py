"""/idcheck: only the door events the visit IDs are unsure about, one question each, one key (26.10.2026 owner).

Built from the day's presence_live.json (door_presence candidates) and identity_<day>.json:
  link    an exit tied to an entry with a low score or margin -> same person?        1 same, 2 different, 3 cannot tell
  orphan  an exit tied to nobody -> which of the people inside, if any                1-4 that one, 0 entry missed, x nobody went out
  repeat  a crossing not counted (a repeat of the same person)                         1 repeat, 2 a new real crossing, x nobody crossed
Answers -> data/door_identity/<day>/idcheck.json {item id: {answer, at}}; door_count_compare and the scores read them
next to /visitpairs. The pictures come from the /visitpairs endpoints (event views, photo, clip)."""
import json
import threading
import time
from pathlib import Path

LOW_SCORE, LOW_MARGIN = .75, .10
LOCK = threading.Lock()

PAGE = r'''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Проверка ID</title>
<style>
:root{--bg:#121212;--card:#1d1d1d;--fg:#eee;--mut:#9a9a9a;--line:#333;--in:#5fd38d;--out:#f0a35e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px system-ui}
.top{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
.top b{font-size:20px}.mut{color:var(--mut)}
.wrap{max-width:1400px;margin:0 auto;padding:12px 16px}
.q{font-size:24px;font-weight:600;margin:6px 0 12px}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
.ev{background:var(--card);border-radius:12px;padding:10px;flex:1 1 320px;min-width:0}
.ev h3{margin:0 0 6px;font-size:18px}.in{color:var(--in)}.out{color:var(--out)}
.ev .main{width:100%;max-height:44vh;object-fit:contain;background:#000;border-radius:8px;display:block}
.strip{display:flex;gap:4px;margin-top:6px;flex-wrap:wrap}.strip img{height:150px;max-width:90px;object-fit:contain;background:#000;border-radius:6px}
.k{display:inline-block;background:#fff;color:#000;border-radius:6px;padding:0 8px;font-weight:800;margin-right:6px}
.acts{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}
.acts button{padding:12px 16px;background:#2a2a2a;color:var(--fg);border:2px solid #444;border-radius:10px;font-size:16px;cursor:pointer}
.acts button.on{border-color:#fff}
.nav{display:flex;gap:8px}.nav button{padding:8px 14px;background:#2a2a2a;color:var(--fg);border:0;border-radius:8px;cursor:pointer}
video.main{cursor:pointer}
@media(max-width:700px){.strip img{height:100px}}
</style></head><body>
<div class="top"><b>Проверка ID</b><span id="cnt" class="mut"></span><span class="mut">клавиши — цифры на кнопках, ← → листать</span></div>
<div class="wrap" id="m">загрузка…</div>
<script>
const q=new URLSearchParams(location.search);const day=q.get('day')||'';let D=null,i=0;
const ev=(d,e)=>e.views&&e.views.length?`/api/visitpairs/evview?day=${d}&n=${e.name}&f=${e.views[0]}`:'';
const strip=(d,e)=>(e.views||[]).map(f=>`<img src="/api/visitpairs/evview?day=${d}&n=${e.name}&f=${f}">`).join('');
const card=(e,title)=>`<div class="ev"><h3 class="${e.kind}">${title} — ${e.kind==='in'?'вошёл':'вышел'} в ${e.clock.slice(11,19)}</h3>
 ${e.clip?`<video class="main" src="/api/visitpairs/clip?day=${D.day}&n=${e.name}" autoplay loop muted playsinline title="щелчок — пауза" onclick="this.paused?this.play():this.pause()"></video>`:`<img class="main" src="${ev(D.day,e)}">`}
 <div class="strip">${strip(D.day,e)}</div></div>`;
async function load(){D=await (await fetch('/api/idcheck?day='+day)).json();i=D.items.findIndex(x=>!(x.id in D.answers));if(i<0)i=D.items.length;draw()}
function draw(){const N=D.items.length,done=Object.keys(D.answers).length;cnt.textContent=`${D.day}: отвечено ${done} из ${N}`;
 if(!N){m.innerHTML='<div class="q">Сомнительных событий нет.</div>';return}
 if(i>=N){m.innerHTML=`<div class="q">Готово — все ${N} вопросов отвечены. Можно листать назад ←.</div>`;return}
 const x=D.items[i],a=(D.answers[x.id]||{}).answer;let h='',opts=[];
 if(x.type==='link'){h=`<div class="q">Это один и тот же человек? <span class="mut">(система связала, похожесть ${x.score})</span></div><div class="row">${card(x.event,'Выход')}${card(x.entry,'Вход')}</div>`;
  opts=[['1','Да, тот же'],['2','Нет, разные'],['3','Не понять']]}
 else if(x.type==='orphan'){h=`<div class="q">Кто это вышел? <span class="mut">(система не нашла пару)</span></div><div class="row">${card(x.event,'Выход')}${x.cands.map((c,j)=>card(c,'Внутри '+(j+1))).join('')}</div>`;
  opts=x.cands.map((c,j)=>[String(j+1),'Это '+(j+1)]).concat([['0','Никто из них (вход пропущен)'],['x','Никто не выходил']])}
 else {h=`<div class="q">${x.event.kind==='in'?'Вход':'Выход'} не засчитан как повтор. Верно? <span class="mut">(${x.why})</span></div><div class="row">${card(x.event,'Это событие')}${x.prev?card(x.prev,'Его прошлое событие'):''}</div>`;
  opts=[['1','Да, повтор того же прохода'],['2','Нет, это новый настоящий проход'],['x','Никто не проходил']]}
 m.innerHTML=h+`<div class="acts">${opts.map(([k,t])=>`<button class="${a===k?'on':''}" onclick="say('${k}')"><span class="k">${k.toUpperCase()}</span>${t}</button>`).join('')}</div>
 <div class="nav"><button onclick="go(-1)">← назад</button><button onclick="go(1)">пропустить →</button></div>`;scrollTo(0,0)}
function go(d){i=Math.max(0,Math.min(D.items.length,i+d));draw()}
function say(k){const x=D.items[i];if(!x)return;D.answers[x.id]={answer:k};
 fetch('/api/idcheck',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({day:D.day,item:x.id,answer:k})});
 i++;while(i<D.items.length&&(D.items[i].id in D.answers))i++;draw()}
document.onkeydown=(e)=>{if(!D)return;const x=D.items[i];if(e.key==='ArrowLeft')return go(-1);if(e.key==='ArrowRight')return go(1);
 if(!x)return;let k=e.key.toLowerCase();if(k==='ч')k='x';
 const ok=x.type==='link'?['1','2','3']:x.type==='orphan'?x.cands.map((c,j)=>String(j+1)).concat(['0','x']):['1','2','x'];
 if(ok.includes(k))say(k)};
load();
</script></body></html>'''


def items(root, day):
    """The doubtful events of the day, in time order."""
    root = Path(root)
    read = lambda p, d: json.load(open(p, encoding='utf-8')) if p.exists() else d
    pres = {x['event']: x for x in read(root / 'data' / 'door_identity' / day / 'presence_live.json', [])}
    ident = read(root / 'data' / 'live' / 'records' / ('identity_%s.json' % day), [])
    views = root / 'data' / 'live' / 'event_views' / day
    clips = root / 'data' / 'live' / 'clips_seg' / day

    def info(key):
        m = read(views / key / 'manifest.json', {})
        e = m.get('event', {})
        return dict(name=key, kind=e.get('kind', key.split('_')[0]), clock=e.get('clock', ''), t=e.get('t'),
                    views=[v['path'] for v in m.get('views', [])], clip=(clips / (key + '.mp4')).exists())
    last = {}                                       # visit -> its previous event, for the repeat questions
    out = []
    for x in ident:
        k, p = x['key'], pres.get(x['key'], {})
        cands = p.get('candidates') or []
        if x['kind'] == 'out' and x.get('entry') and x['status'] == 'matched':
            score = cands[0]['score'] if cands else None
            gap = p.get('margin')
            if score is not None and (score < LOW_SCORE or (gap is not None and gap < LOW_MARGIN)):
                out.append(dict(id='link:' + k, type='link', event=info(k), entry=info(x['entry']), score=round(score, 2)))
        elif x['kind'] == 'out' and not x.get('visit'):
            if cands:
                out.append(dict(id='orphan:' + k, type='orphan', event=info(k),
                                cands=[info(c['entry_event']) for c in cands[:4]]))
        elif x.get('count') is False and x.get('role') != 'staff':
            why = {'repeat_entry': 'тот же человек уже внутри', 'return': 'вернулся меньше чем через 5 минут',
                   'repeat_exit': 'тот же выход дважды', 'matched': 'у визита будет выход позже'}.get(x['status'], x['status'])
            out.append(dict(id='repeat:' + k, type='repeat', event=info(k), why=why,
                            prev=info(last[x['visit']]) if x.get('visit') in last else None))
        if x.get('visit'):
            last[x['visit']] = k
    return out


def register(app, root):
    from flask import jsonify, request, abort

    def path(day):
        return Path(root()) / 'data' / 'door_identity' / day / 'idcheck.json'

    @app.get('/idcheck')
    def idcheck_page():
        return PAGE

    @app.get('/api/idcheck')
    def idcheck_get():
        day = request.args.get('day') or time.strftime('%Y%m%d')
        if not day.isdigit():
            abort(400)
        p = path(day)
        return jsonify({'day': day, 'items': items(root(), day),
                        'answers': json.load(open(p, encoding='utf-8')) if p.exists() else {}})

    @app.post('/api/idcheck')
    def idcheck_post():
        from storage import atomic_json
        b = request.get_json(force=True)
        day, item, ans = str(b.get('day', '')), str(b.get('item', '')), str(b.get('answer', ''))
        if not day.isdigit() or ':' not in item or ans not in ('0', '1', '2', '3', '4', 'x'):
            abort(400)
        p = path(day)
        with LOCK:
            a = json.load(open(p, encoding='utf-8')) if p.exists() else {}
            a[item] = {'answer': ans, 'at': time.strftime('%Y-%m-%d %H:%M:%S')}
            p.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(p, a)
        return jsonify({'ok': True, 'n': len(a)})
