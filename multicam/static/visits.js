'use strict';
const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data,day,index=0,busy=false,manualPair=null,loading=false,loadTicket=0,renderTicket=0;
async function api(url,opts){const r=await fetch(url,opts),j=await r.json();if(!r.ok)throw Error(j.error||'Ошибка сохранения');return j;}
function notice(t){$('#notice').textContent=t;}
async function load(){
 const ticket=++loadTicket;loading=true;manualPair=null;$('#undo').disabled=true;
 document.querySelectorAll('main button,main select').forEach(el=>el.disabled=true);
 notice('Собираю связи по записям дня…');
 try{const fresh=await api('/api/day/'+day);if(ticket!==loadTicket)return;data=fresh;index=0;loading=false;draw();notice('');}
 catch(e){if(ticket===loadTicket){loading=false;data=null;$('#undo').disabled=true;notice(e.message);}throw e;}
}
const timeText=seconds=>new Intl.DateTimeFormat('ru-RU',{hour:'2-digit',minute:'2-digit',second:'2-digit',timeZone:'Asia/Bangkok'}).format(new Date(seconds*1000));
function fillNodeChoices(){
 const nodes=data.nodes.slice().sort((a,b)=>a.first-b.first||a.label.localeCompare(b.label));
 for(const selector of ['#nodeA','#nodeB']){
  const select=$(selector),old=select.value;
  select.innerHTML='<option value="">Выберите фрагмент…</option>'+nodes.map(n=>{
   const cameras=(n.cams||n.ranges?.map(r=>r.cam)||[n.representative?.cam]).filter(Boolean);
   const cameraText=[...new Set(cameras)].map(cam=>cam.replace('cam','')).join(', ')||'в записи';
   const text=`${n.label} · ${timeText(n.first)}–${timeText(n.last)} · камера ${cameraText} · ${n.clip}${n.pieces.length?'':' · добавлен вручную'}`;
   return `<option value="${esc(n.key)}">${esc(text)}</option>`;
  }).join('');
  if(nodes.some(n=>n.key===old))select.value=old;
  select.disabled=busy||loading||nodes.length<2;
 }
 refreshCompareButton();
}
function refreshCompareButton(){
 $('#compareNodes').disabled=busy||loading||!$('#nodeA').value||!$('#nodeB').value||$('#nodeA').value===$('#nodeB').value;
}
$('#nodeA').onchange=refreshCompareButton;$('#nodeB').onchange=refreshCompareButton;
$('#compareNodes').onclick=()=>{
 if(busy||loading)return;
 const a=data.nodes.find(n=>n.key===$('#nodeA').value),b=data.nodes.find(n=>n.key===$('#nodeB').value);
 if(!a||!b||a.key===b.key){notice('Выберите два разных фрагмента.');return;}
 manualPair={a:a.key,b:b.key,evidence_a:a.evidence,evidence_b:b.evidence,
  gap_s:Math.max(0,Math.max(a.first,b.first)-Math.min(a.last,b.last))};
 draw();notice('Пара выбрана вручную. Связь появится только после вашего ответа.');
 $('#content').style.scrollMarginTop=(document.querySelector('header').getBoundingClientRect().height+16)+'px';
 $('#content').scrollIntoView({behavior:'smooth',block:'start'});
};
function nodeFrame(n,side){
 if(!n.representative)return null;
 const ranges=(n.ranges||[]).slice().sort((a,b)=>side?a.start_frame-b.start_frame:b.end_frame-a.end_frame);
 return ranges.length?{cam:ranges[0].cam,frame:side?ranges[0].start_frame:ranges[0].end_frame}:n.representative;
}
function nodeCard(n,side){
 const frame=nodeFrame(n,side),clip=encodeURIComponent(n.clip);
 const src=frame?`/api/video/${clip}/image?cam=${encodeURIComponent(frame.cam)}&frame=${frame.frame}`:
   `/portrait/${clip}/${n.pieces[side?0:n.pieces.length-1]}`;
 const editor=frame?`<p><a href="/video?clip=${clip}&cam=${encodeURIComponent(frame.cam)}&frame=${frame.frame}" target="_blank" rel="noopener">Открыть этот кадр и исправить на видео ↗</a></p>`:'';
 return `<div class="video"><p>${esc(n.clip)} · ${esc(n.label)} · ${n.reviewed?'личность проверена':'машинная группа'}</p><div data-node-image="${side}" style="position:relative;display:inline-block;max-width:100%;line-height:0"><img src="${src}" alt="Человек в записи ${side+1}" style="display:block;height:240px;max-width:100%;object-fit:contain">${frame?'<svg aria-hidden="true" preserveAspectRatio="xMidYMid meet" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none"></svg>':''}</div>${editor}<div id="preview${side}"><button data-video="${side}">Посмотреть ${side?'начало':'конец'} ▶</button></div></div>`;
}
function draw(){
 const ticket=++renderTicket;
 const link=manualPair||data.proposals[index];
 fillNodeChoices();
 $('#summary').textContent=`${data.nodes.length} фрагментов визитов · ${data.links.length} связей · ${data.proposals.length} предложений`;
 $('#undo').disabled=!data.can_undo||busy;
 if(!link){$('#content').innerHTML='<div class="panel"><h1>Нет непроверенных предложений</h1><p>Вы можете выбрать любые два фрагмента выше, в том числе добавленного вручную человека. Связи по общим исходным кадрам уже учтены.</p></div>';return;}
 const nodes=[link.a,link.b].map(key=>data.nodes.find(n=>n.key===key));
 $('#content').innerHTML=`<div class="panel"><h1>Это продолжение одного визита?</h1><p>${manualPair?'Пара выбрана вручную':`Предложение ${index+1} из ${data.proposals.length}`} · промежуток ${link.gap_s.toFixed(1)} с</p><div id="videos">${nodes.map(nodeCard).join('')}</div><div class="actions"><button data-answer="same" class="primary">Тот же человек</button><button data-answer="different">Разные люди</button><button data-answer="unsure">Не уверен</button><button id="skip">${manualPair?'К предложениям':'Пропустить →'}</button></div><p class="muted">Ответ связывает визиты и не выдаёт машинные маски за проверенную разметку.</p></div>`;
 nodes.forEach(async(n,i)=>{
  const frame=nodeFrame(n,i);if(!frame)return;
  try{
   const current=await api(`/api/video/${encodeURIComponent(n.clip)}/frame?cam=${encodeURIComponent(frame.cam)}&frame=${frame.frame}`);
   if(ticket!==renderTicket)return;
   const svg=document.querySelector(`[data-node-image="${i}"] svg`);if(!svg)return;
   svg.setAttribute('viewBox',`0 0 ${current.width} ${current.height}`);
   const ids=new Set(n.observations||[]);
   svg.innerHTML=current.observations.filter(o=>ids.has(o.id)).map(o=>{
    const [x1,y1,x2,y2]=o.box.map(Number);
    return [x1,y1,x2,y2].every(Number.isFinite)?`<rect x="${x1}" y="${y1}" width="${x2-x1}" height="${y2-y1}" fill="none" stroke="#87ffb2" stroke-width="4"/>`:'';
   }).join('');
  }catch(e){if(ticket===renderTicket)notice('Выделение человека недоступно. Откройте точный кадр в видеоредакторе перед ответом.');}
 });
 document.querySelectorAll('[data-answer]').forEach(b=>b.onclick=()=>answer({...link,decision:b.dataset.answer}));
 $('#skip').onclick=()=>{if(manualPair)manualPair=null;else index=(index+1)%Math.max(1,data.proposals.length);draw();};
 document.querySelectorAll('[data-video]').forEach(b=>b.onclick=async()=>{
  const i=+b.dataset.video,n=nodes[i];b.disabled=true;
  try{
   const frame=nodeFrame(n,i);let r;
   if(frame)r=await api(`/api/video/${encodeURIComponent(n.clip)}/window?cam=${encodeURIComponent(frame.cam)}&frame=${frame.frame}`);
   else{
    const state=await api('/api/pieces/'+n.clip);
    const ps=state.pieces.filter(p=>n.pieces.includes(p.piece)).sort((a,b)=>a.raw_t0-b.raw_t0);
    const p=i?ps[0]:ps[ps.length-1];
    if(!p)throw Error('Нет доступного отрезка; откройте запись в видеоредакторе.');
    r=await api(`/api/preview/${n.clip}/${p.piece}?at=${i?p.raw_t0:p.raw_t1}`);
   }
   if(ticket!==renderTicket)return;
   $('#preview'+i).innerHTML=`<video controls playsinline src="${esc(r.url)}"></video>`;
  }catch(e){if(ticket===renderTicket){notice(e.message);b.disabled=false;}}
 });
}
async function answer(body){if(busy||loading||!data)return;busy=true;$('#day').disabled=true;document.querySelectorAll('button,main select').forEach(b=>b.disabled=true);try{data=await api('/api/day/'+day,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,revision:data.revision})});index=Math.min(index,Math.max(0,data.proposals.length-1));manualPair=null;notice('Сохранено');}catch(e){notice(e.message);}finally{busy=false;$('#day').disabled=false;draw();}}
$('#undo').onclick=()=>answer({action:'undo'});
(async()=>{try{const clips=await api('/api/clips'),days=[...new Set(clips.map(c=>c.start.slice(0,10).replaceAll('-','')))];$('#day').innerHTML=days.map(d=>`<option>${d}</option>`).join('');day=days[0];$('#day').onchange=()=>{day=$('#day').value;load().catch(e=>notice(e.message));};if(day)await load();else notice('Нет готовых записей');}catch(e){notice(e.message);}})();
