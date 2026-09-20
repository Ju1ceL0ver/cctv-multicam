'use strict';
const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data,day,index=0,busy=false,loading=false,ticket=0;
async function api(url,opts){const r=await fetch(url,opts),j=await r.json();if(!r.ok)throw Error(j.error||'Ошибка сохранения');return j;}
function notice(t){$('#notice').textContent=t;}
const clock=seconds=>new Intl.DateTimeFormat('ru-RU',{hour:'2-digit',minute:'2-digit',second:'2-digit',timeZone:'Asia/Bangkok'}).format(new Date(seconds*1000));
const length=s=>s>=60?`${Math.floor(s/60)} мин ${Math.round(s%60)} с`:`${Math.round(s)} с`;
const minimum=()=>$('#minimum').value;

async function load(keep){
 const mine=++ticket;loading=true;$('#undo').disabled=true;
 document.querySelectorAll('main button,header select').forEach(el=>el.disabled=true);
 notice('Собираю визиты дня…');
 try{
  const fresh=await api(`/api/day/${day}/visits?seconds=${encodeURIComponent(minimum())}`);
  if(mine!==ticket)return;
  data=fresh;if(!keep)index=0;loading=false;draw();notice('');
 }catch(e){if(mine===ticket){loading=false;data=null;notice(e.message);}}
}

/** A picture of this fragment: the exact frame when we have one, its portrait otherwise. */
function face(n,side){
 const clip=encodeURIComponent(n.clip);
 const ranges=(n.ranges||[]).slice().sort((a,b)=>side?a.start_frame-b.start_frame:b.end_frame-a.end_frame);
 const frame=ranges.length?{cam:ranges[0].cam,frame:side?ranges[0].start_frame:ranges[0].end_frame}:n.representative;
 if(frame)return{src:`/api/video/${clip}/image?cam=${encodeURIComponent(frame.cam)}&frame=${frame.frame}`,
  open:`/video?clip=${clip}&cam=${encodeURIComponent(frame.cam)}&frame=${frame.frame}`};
 const piece=n.pieces[side?0:n.pieces.length-1];
 return piece===undefined?null:{src:`/portrait/${clip}/${piece}`,open:`/video?clip=${clip}`};
}

function fragmentCard(n,i,total){
 const shot=face(n,i===0?1:0);
 const cams=(n.cams||[]).map(c=>c.replace('cam','')).join(', ')||'—';
 return `<figure style="margin:0;min-width:0">
 ${shot?`<img src="${shot.src}" alt="Фрагмент ${i+1} из ${total}" loading="lazy" style="display:block;height:180px;max-width:100%;object-fit:contain;background:#000">`:'<p class="muted">нет кадра</p>'}
 <figcaption class="muted" style="font-size:13px">${i+1}. ${esc(clock(n.first))}–${esc(clock(n.last))} · ${esc(length(n.seconds))} · камера ${esc(cams)}<br>${esc(n.clip)} · ${esc(n.label)}${n.reviewed?' · проверен':''}${shot?` · <a href="${shot.open}" target="_blank" rel="noopener">открыть ↗</a>`:''}</figcaption>
 </figure>`;
}

function continuationCard(c){
 const n=data.visits[index].byNode[c.node];
 const shot=n?face(n,1):null;
 return `<div class="video" style="min-width:0">
 ${shot?`<img src="${shot.src}" alt="Возможное продолжение" loading="lazy" style="display:block;height:180px;max-width:100%;object-fit:contain;background:#000">`:''}
 <p class="muted" style="font-size:13px">${n?`${esc(clock(n.first))}–${esc(clock(n.last))} · ${esc(n.clip)} · ${esc(n.label)}`:esc(c.node)}<br>промежуток ${c.gap_s.toFixed(1)} с · непохожесть ${c.distance.toFixed(3)}</p>
 <div class="actions"><button data-link="same" data-node="${esc(c.node)}" class="primary">Тот же</button><button data-link="different" data-node="${esc(c.node)}">Другой</button><button data-link="unsure" data-node="${esc(c.node)}">Не уверен</button></div>
 </div>`;
}

function draw(){
 const c=data&&data.counts;
 $('#summary').textContent=data?`${c.long_visits} длинных визитов · отвечено ${c.answered} · собран верно целиком ${c.clean_and_whole} · с чужим человеком ${c.mixed} · неполных ${c.partial}${c.retired?` · ${c.retired} ответов устарели после правок`:''}`:'';
 $('#undo').disabled=!data||!data.can_undo||busy;
 document.querySelectorAll('header select').forEach(el=>el.disabled=busy||loading);
 if(!data)return;
 if(!data.visits.length){$('#content').innerHTML='<div class="panel"><h1>Длинных визитов нет</h1><p>За этот день никто не задержался дольше выбранного времени, либо окна дня ещё размечаются.</p></div>';return;}
 index=Math.min(index,data.visits.length-1);
 const v=data.visits[index];
 v.byNode=Object.fromEntries(v.fragments.map(f=>[f.key,f]));
 for(const cont of v.continuations)if(!v.byNode[cont.node])v.byNode[cont.node]=null;
 const answered=v.judgement;
 $('#content').innerHTML=`<div class="panel">
 <h1>${esc(length(v.seconds))} в зале · ${esc(clock(v.first))}–${esc(clock(v.last))}</h1>
 <p>Визит ${index+1} из ${data.visits.length} · собран из ${v.fragments.length} ${v.fragments.length===1?'фрагмента':'фрагментов'} · камеры ${esc(v.cams.map(x=>x.replace('cam','')).join(', ')||'—')}${answered?` · ваш ответ: ${answered.purity==='clean'?'без чужих':answered.purity==='mixed'?'есть чужой':'не уверены'}, ${answered.wholeness==='whole'?'целиком':answered.wholeness==='partial'?'неполный':'не уверены'}`:''}${v.retired?' · прежний ответ устарел после правок':''}</p>
 <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,220px),1fr));gap:12px;margin:12px 0">${v.fragments.map((f,i)=>fragmentCard(f,i,v.fragments.length)).join('')}</div>
 ${v.continuations.length?`<h2 style="font-size:17px">Возможно, визит продолжается</h2><div id="videos">${v.continuations.map(continuationCard).join('')}</div>`:''}
 <h2 style="font-size:17px">Что здесь на самом деле?</h2>
 <div class="actions" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));align-items:start;gap:12px">
  <fieldset style="border:1px solid #333;border-radius:8px;min-width:0"><legend>Все фрагменты — один человек?</legend>
   <label style="display:block"><input type="radio" name="purity" value="clean"${answered&&answered.purity==='clean'?' checked':''}> да, чужих нет</label>
   <label style="display:block"><input type="radio" name="purity" value="mixed"${answered&&answered.purity==='mixed'?' checked':''}> нет, подмешан другой</label>
   <label style="display:block"><input type="radio" name="purity" value="unsure"${answered&&answered.purity==='unsure'?' checked':''}> не разобрать</label>
  </fieldset>
  <fieldset style="border:1px solid #333;border-radius:8px;min-width:0"><legend>Визит собран целиком?</legend>
   <label style="display:block"><input type="radio" name="wholeness" value="whole"${answered&&answered.wholeness==='whole'?' checked':''}> да, от появления до ухода</label>
   <label style="display:block"><input type="radio" name="wholeness" value="partial"${answered&&answered.wholeness==='partial'?' checked':''}> нет, часть потеряна</label>
   <label style="display:block"><input type="radio" name="wholeness" value="unsure"${answered&&answered.wholeness==='unsure'?' checked':''}> не разобрать</label>
  </fieldset>
 </div>
 <div class="actions"><button id="save" class="primary">Сохранить ответ</button><button id="prev">← Назад</button><button id="next">Дальше →</button></div>
 <p class="muted">Ответ описывает этот визит в его нынешнем виде. Если потом исправить его фрагменты, ответ помечается устаревшим, а не переносится на другой визит.</p>
 </div>`;
 $('#save').onclick=save;
 $('#prev').onclick=()=>{index=(index-1+data.visits.length)%data.visits.length;draw();};
 $('#next').onclick=()=>{index=(index+1)%data.visits.length;draw();};
 document.querySelectorAll('[data-link]').forEach(b=>b.onclick=()=>link(b.dataset.node,b.dataset.link));
 document.querySelectorAll('main button').forEach(b=>b.disabled=busy||loading);
}

async function save(){
 if(busy||loading||!data)return;
 const v=data.visits[index];
 const purity=document.querySelector('input[name=purity]:checked'),whole=document.querySelector('input[name=wholeness]:checked');
 if(!purity||!whole){notice('Ответьте на оба вопроса.');return;}
 busy=true;draw();
 try{
  data=await api(`/api/day/${day}/visits`,{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({revision:data.revision,nodes:v.nodes,evidence:v.evidence,
    purity:purity.value,wholeness:whole.value,minimum:Number(minimum())})});
  notice('Сохранено');index=Math.min(index+1,data.visits.length-1);
 }catch(e){notice(e.message);}
 finally{busy=false;draw();}
}

/** Answering a continuation goes to the same store the pairwise page uses. */
async function link(node,decision){
 if(busy||loading||!data)return;
 const cont=data.visits[index].continuations.find(c=>c.node===node);
 if(!cont)return;
 busy=true;draw();
 try{
  await api('/api/day/'+day,{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({a:cont.a,b:cont.b,decision,evidence_a:cont.evidence_a,evidence_b:cont.evidence_b,revision:data.revision})});
  notice(decision==='same'?'Связано; визит пересобран':'Сохранено');
  busy=false;await load(true);
 }catch(e){notice(e.message);busy=false;draw();}
}

async function undo(){
 if(busy||loading||!data)return;
 busy=true;draw();
 try{await api('/api/day/'+day,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'undo',revision:data.revision})});notice('Отменено');busy=false;await load(true);}
 catch(e){notice(e.message);busy=false;draw();}
}

$('#undo').onclick=undo;
$('#minimum').onchange=()=>load().catch(e=>notice(e.message));
(async()=>{
 try{
  const clips=await api('/api/clips'),days=[...new Set(clips.map(c=>c.start.slice(0,10).replaceAll('-','')))];
  $('#day').innerHTML=days.map(d=>`<option>${d}</option>`).join('');
  day=days[0];
  $('#day').onchange=()=>{day=$('#day').value;load().catch(e=>notice(e.message));};
  if(day)await load();else notice('Нет готовых записей');
 }catch(e){notice(e.message);}
})();
