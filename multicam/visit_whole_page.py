"""Read candidates and record owner judgements without changing people_answers.json."""
import re
import hashlib
import json
from pathlib import Path
from storage import atomic_json, file_lock, read_json

PAGE = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Визит целиком</title><link rel="icon" href="data:,"><style>body{font:16px system-ui;background:#151515;color:#eee;margin:24px;max-width:1000px}input,button,select{font:inherit;padding:8px;margin:4px;background:#333;color:#eee;border:1px solid #666;border-radius:5px}video{width:360px;max-width:100%}svg{width:360px;max-width:100%;height:auto;aspect-ratio:1;background:#242424}article{border-top:1px solid #555;padding:20px 0}.muted{color:#bbb}pre{white-space:pre-wrap}</style>
<h1>Визит целиком</h1><input id="day" aria-label="День" type="date"><button id="load">Показать</button><p id="status" role="status"></p><div id="list"></div>
<script>
const day=document.querySelector('#day'),list=document.querySelector('#list'),status=document.querySelector('#status');
day.value=new URLSearchParams(location.search).get('day')?.replace(/^(....)(..)(..)$/,'$1-$2-$3')||new Date().toLocaleDateString('en-CA');
const hms=s=>new Date(s*1000).toISOString().slice(11,19);
const el=(tag,text)=>{let e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e};
async function load(){list.replaceChildren();status.textContent='Загрузка…';try{let r=await fetch('/api/wholevisits?day='+day.value.replaceAll('-',''));if(!r.ok)throw Error('Нет готовых визитов за этот день');let d=await r.json();status.textContent='Кандидатов: '+d.visits.length+'. Проверено: '+d.quality.verified_visits+'. Склейка и взаимодействия требуют проверки владельца.';
let visitNumber=0;for(let v of d.visits){let a=el('article');a.append(el('h2','Визит '+(++visitNumber)+' · '+hms(v.start||0)+'–'+hms(v.end||0)),el('p','Роль: '+({customer:'покупатель',staff:'сотрудник',unknown:'не определена'}[v.role]||'не определена')+' · '+(v.door_bounded?'Есть вход и выход':'Не хватает входа или выхода')+' · разрывов: '+v.coverage_gaps.length));
if(v.media.status==='ready'){let vid=el('video');vid.controls=true;vid.preload='metadata';vid.src='/api/wholevisits/media?day='+d.day+'&id='+v.id+'&ext=mp4';a.append(vid)}else a.append(el('p','Видео: '+({missing_detection_provenance:'нет исходных рамок человека',role_not_customer:'роль покупателя не подтверждена',not_requested:'ещё не собрано',failed:'ошибка сборки',expired:'срок хранения 30 дней истёк'}[v.media.status]||'ещё не готово')));
let pts=v.trajectory;if(pts.length){let xs=pts.map(p=>p[1]),ys=pts.map(p=>p[2]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x||1,h=Math.max(...ys)-y||1;let svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox',`${x-1} ${y-1} ${w+2} ${h+2}`);svg.setAttribute('aria-label','Траектория в метрах; разрывы не соединены');let runs=[],run=[];pts.forEach((p,i)=>{if(i&&p[0]-pts[i-1][0]>3){runs.push(run);run=[]}run.push(p)});runs.push(run);for(let run of runs){let line=document.createElementNS(svg.namespaceURI,'polyline');line.setAttribute('points',run.map(p=>p[1]+','+p[2]).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke','#69d9a1');line.setAttribute('stroke-width',Math.max(w,h)/150);svg.append(line)}a.append(svg)}
a.append(el('p','Зоны: '+(v.zones.length?v.zones.map(z=>z.name+': '+z.seconds+' с').join('; '):'не заданы на плане')),el('p',v.interactions.length?'Взаимодействия: '+v.interactions.map(x=>hms(x.start)+'–'+hms(x.end)+' · '+x.seconds+' с · сотрудник '+(d.visits.findIndex(r=>r.id===x.staff)+1)).join('; '):'Взаимодействия не найдены. Нужна проверка.'));
let saved=d.review[v.id]||{};for(let [field,label,choices] of [['purity','Чистота',[['','Выбрать'],['clean','Один человек'],['mixed','Есть чужой человек'],['unsure','Не понять']]],['wholeness','Полнота',[['','Выбрать'],['whole','Визит целиком'],['partial','Есть пропуски'],['unsure','Не понять']]],['interactions','Взаимодействия',[['','Выбрать'],['correct','Верно'],['wrong','Есть ошибка'],['unsure','Не понять']]]]){let l=el('label',label),s=el('select');for(let [value,text]of choices){let o=el('option',text);o.value=value;s.append(o)}s.value=saved[field]||'';s.onchange=async()=>{saved[field]=s.value;let r=await fetch('/api/wholevisits/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({day:d.day,id:v.id,...saved,evidence:v.evidence})});status.textContent=r.ok?'Ответ сохранён':'Ответ не сохранён'};l.append(s);a.append(l)}list.append(a)}}catch(e){status.textContent=e.message}}
document.querySelector('#load').onclick=load;load();</script></html>'''


def evidence(visit):
    fields = {k:visit.get(k) for k in ('fragments','trajectory','entry','exit','role','interactions')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()


def register(app, root):
    from flask import request, jsonify, abort, send_file
    def folder(day):
        if not re.fullmatch(r'\d{8}', day): abort(400)
        return Path(root()) / 'data/visits' / day
    def visit_id(value):
        if not re.fullmatch(r'v_[A-Za-z0-9_]+', value): abort(400)
        return value
    @app.get('/wholevisits')
    def page(): return PAGE
    @app.get('/api/wholevisits')
    def data():
        base = folder(request.args.get('day',''))
        data = read_json(base/'index.json',None)
        if data is None: abort(404)
        reviews = read_json(base/'whole_review.json',{})
        current_review = {}
        for visit in data.get('visits',[]):
            visit['evidence'] = evidence(visit)
            if reviews.get(visit['id'],{}).get('evidence') == visit['evidence']:
                current_review[visit['id']] = reviews[visit['id']]
            media = visit.get('media',{})
            if media.get('status') == 'ready' and not (base / (visit['id']+'.mp4')).exists():
                visit['media'] = dict(media,status='expired')
        judged = [r for r in current_review.values() if r.get('purity') in ('clean','mixed') and r.get('wholeness') in ('whole','partial')]
        correct = sum(r['purity']=='clean' and r['wholeness']=='whole' for r in judged)
        data['quality'] = dict(data.get('quality',{}), verified_visits=len(judged), fully_correct=correct,
                               fully_correct_share=correct/len(judged) if judged else None)
        return jsonify(dict(data,review=current_review))
    @app.get('/api/wholevisits/media')
    def media():
        base=folder(request.args.get('day','')); ident=visit_id(request.args.get('id',''))
        ext=request.args.get('ext')
        if ext not in ('mp4','gif'): abort(400)
        path=base/(ident+'.'+ext)
        if not path.exists(): abort(404)
        return send_file(str(path),conditional=True)
    @app.post('/api/wholevisits/review')
    def review():
        body=request.get_json();base=folder(body.get('day','')); ident=visit_id(body.get('id',''))
        current = next((v for v in read_json(base/'index.json',{}).get('visits',[]) if v['id']==ident),None)
        if current is None: abort(404)
        if body.get('evidence') != evidence(current): abort(409)
        values={'evidence':evidence(current)}
        for key,allowed in [('purity',('','clean','mixed','unsure')),('wholeness',('','whole','partial','unsure')),('interactions',('','correct','wrong','unsure'))]:
            value=body.get(key,'')
            if value not in allowed: abort(400)
            values[key]=value
        with file_lock(base/'whole_review.lock'):
            data=read_json(base/'whole_review.json',{});data[ident]=values;atomic_json(base/'whole_review.json',data)
        return jsonify(ok=True)
