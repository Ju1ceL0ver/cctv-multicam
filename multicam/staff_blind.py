"""A separate, prediction-free held-day role review. Answers never enter data/staff/labels.json.
Run: staff_blind.py build [ROOT]; frozen days are intentionally not configurable after publish.
"""
import hashlib,json,re,shutil,sys,time
from pathlib import Path
from storage import atomic_json,file_lock
DAYS=('20260911','20260912')
NAME='staff_blind_20260911_20260912'


def folder(root):return Path(root)/'data'/NAME


def build(root):
 root=Path(root);out=folder(root);out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest.json').exists():return json.loads((out/'manifest.json').read_text())
 old=json.loads((root/'data'/'staff'/'people.json').read_text())
 assert not set(DAYS)&{v['day'] for v in old.values()},'Held days already in staff feature table'
 report=json.loads((root/'data'/'staff'/'improve_20261008'/'report.json').read_text())
 assert not set(DAYS)&set(report['train_days']+report['test_days']), 'Days previously used by selected model'
 source=root.parent/'retail_analytics'/'runs'/'live'/'entrance_events.jsonl'
 imgs=out/'images';imgs.mkdir(exist_ok=True)
 items={};missing=0;counts={d:0 for d in DAYS}
 for line in source.open(encoding='utf-8'):
  if not line.strip():continue
  e=json.loads(line);d=time.strftime('%Y%m%d',time.localtime(e['unix_ms']/1000))
  if d not in DAYS:continue
  counts[d]+=1;p=Path(e.get('snapshot_crop',''))
  if not p.is_file():missing+=1;continue
  ident=hashlib.sha256((d+':'+e['event_id']).encode()).hexdigest()[:32]
  shutil.copyfile(p,imgs/(ident+'.jpg'))
  items[ident]={'day':d,'time':time.strftime('%H:%M:%S',time.localtime(e['unix_ms']/1000)),
                'event_id':e['event_id'],'cam':'cam1'}
 model=root/'data'/'staff'/'improve_20261008'/'selected.pkl'
 shutil.copyfile(model,out/'frozen_model.pkl')
 # Preserve the predictor code too, so later source edits cannot change the tested implementation.
 shutil.copyfile(root/'staff_model.py',out/'frozen_staff_model.py')
 data={'days':list(DAYS),'items':items,'source_counts':counts,'missing_crops':missing,
       'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'selected':report['selected'],
       'created_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'protocol':'All available events of two untrained days; no predictions, no model-based ordering, separate labels; frozen weights.'}
 atomic_json(out/'manifest.json',data)
 print('READY',len(items),'missing',missing,'days',counts,flush=True)
 return data


def register(app,root):
 from flask import abort,jsonify,render_template,request,send_file
 here=root if callable(root) else lambda:root
 def manifest():
  p=folder(here())/'manifest.json'
  if not p.exists():abort(503)
  return json.loads(p.read_text(encoding='utf-8'))
 def read_labels():
  p=folder(here())/'labels.json'
  return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
 @app.get('/staff-test')
 def staff_test_page():
  return render_template('staff_test.html')
 @app.get('/api/staff-test')
 def staff_test_list():
  m=manifest();labels=read_labels();items=m['items']
  order=sorted(items,key=lambda i:hashlib.sha256(i.encode()).hexdigest())
  todo=[i for i in order if i not in labels]
  recent=sorted(labels,key=lambda i:labels[i]['at'])[-60:]
  return jsonify(todo=todo[:60],recent=recent,labels={i:labels[i]['label'] for i in recent},
                 meta={i:items[i] for i in todo[:60]+recent},total=len(items),done=len(labels),days=m['days'],missing_crops=m['missing_crops'])
 @app.post('/api/staff-test')
 def staff_test_answer():
  b=request.get_json(silent=True) or {};i=b.get('id');label=b.get('label')
  if i not in manifest()['items']:abort(404)
  if label is not None and (isinstance(label,bool) or not isinstance(label,int) or label not in (0,1,2)):
   return jsonify(error='Ответ — 1, 2 или 0'),400
  out=folder(here())
  try:
   with file_lock(str(out/'labels.json')+'.lock'):
    labels=read_labels();at=time.strftime('%Y-%m-%dT%H:%M:%S')
    if label is None:labels.pop(i,None)
    else:labels[i]={'label':label,'at':at}
    atomic_json(out/'labels.json',labels)
    with (out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'id':i,'label':label,'at':at})+'\n')
  except TimeoutError as exc:return jsonify(error=str(exc)),503
  return jsonify(ok=True,done=len(labels))
 @app.get('/staff-test-img/<ident>')
 def staff_test_img(ident):
  if not re.fullmatch('[a-f0-9]{32}',ident) or ident not in manifest()['items']:abort(404)
  response=send_file(folder(here())/'images'/(ident+'.jpg'),mimetype='image/jpeg')
  response.headers['Cache-Control']='private, max-age=3600'
  return response

if __name__=='__main__':
 if sys.argv[1:] and sys.argv[1]=='build':build(sys.argv[2] if len(sys.argv)>2 else Path(__file__).resolve().parent)
