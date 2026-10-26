"""One fixed evaluation of the separately annotated held days. No fitting or threshold selection."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'4')
import hashlib,importlib.util,json,pickle,sys,time,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from staff_blind import folder
from storage import atomic_json


def wait_gpu():
 import psutil
 t=time.time()
 while True:
  busy=[]
  for p in psutil.process_iter(['name','cmdline','status']):
   try:
    cmd=' '.join(p.info['cmdline'] or [])
    if 'python' in (p.info['name'] or '').lower() and 'micro_sam.py' in cmd and (' train ' in cmd or ' eval ' in cmd):busy.append(p.pid)
   except psutil.Error:pass
  q=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
  free=int(q.stdout.strip().splitlines()[0]) if q.returncode==0 else 0
  if not busy and free>=9000:print('GPU_READY',free,flush=True);return
  if time.time()-t>1800:raise RuntimeError('GPU did not become free within 30 min; no model result produced')
  print('WAIT_GPU',busy,free,flush=True);time.sleep(20)


def extract(out,manifest):
 import cv2,torch
 from ultralytics import SAM
 import track_emb as T
 from staff_masked import crop_masked
 wait_gpu();sam=SAM(str(ROOT/'sam2.1_l.pt'));embed=T.teachers()
 events={}
 source=ROOT.parent/'retail_analytics'/'runs'/'live'/'entrance_events.jsonl'
 for line in source.open(encoding='utf-8'):
  if line.strip():
   e=json.loads(line)
   events[e['event_id']]=e
 ids=[];cloth=[];shape=[];boxes=[];flags=[];failures=[];batch=[];keys=[]
 masked=out/'masked';masked.mkdir(exist_ok=True)
 def flush():
  if not batch:return
  a,b=embed(batch)
  for entry,v1,v2 in zip(keys,a,b):
   i,box,flag=entry;ids.append(i);cloth.append(v1);shape.append(v2);boxes.append(box);flags.append(flag)
  batch.clear();keys.clear()
 t=time.time()
 for n,(i,item) in enumerate(sorted(manifest['items'].items())):
  e=events[item['event_id']];full=cv2.imread(e.get('snapshot_full',''))
  if full is None:failures.append({'id':i,'reason':'missing_full_frame'});continue
  bx=[float(e[k]) for k in ('box_x1','box_y1','box_x2','box_y2')]
  px,py=(bx[0]+bx[2])/2,bx[1]+.35*(bx[3]-bx[1]);mk=None
  with torch.autocast('cuda',dtype=torch.float16):
   r=sam.predict(full,bboxes=[bx],points=[[px,py]],labels=[1],verbose=False)[0]
  if r.masks is not None:
   mk=r.masks.data[0].cpu().numpy()>.5
   if mk.shape!=full.shape[:2]:mk=cv2.resize(mk.astype(np.uint8),(full.shape[1],full.shape[0]),interpolation=cv2.INTER_NEAREST)>0
   n_,lab,st,_=cv2.connectedComponentsWithStats(mk.astype(np.uint8))
   if n_>2:
    hit=lab[min(int(py),lab.shape[0]-1),min(int(px),lab.shape[1]-1)]
    mk=lab==(hit if hit else 1+int(np.argmax(st[1:,cv2.CC_STAT_AREA])))
  crop=crop_masked(full,mk,bx)
  if crop is None:failures.append({'id':i,'reason':'invalid_box'});continue
  # Same preprocessing as training counter snapshots; fallback is recorded, never label-driven.
  cv2.imwrite(str(masked/(i+'.jpg')),crop,[cv2.IMWRITE_JPEG_QUALITY,90])
  h,w=full.shape[:2];box=(np.array(bx)*[1280/w,720/h,1280/w,720/h]).astype(np.int32)
  batch.append(crop);keys.append((i,box,mk is not None))
  if len(batch)>=32:flush()
  if (n+1)%40==0:print('EXTRACT',n+1,'/',len(manifest['items']),'seconds',round(time.time()-t),flush=True)
 flush()
 np.savez_compressed(out/'features.npz',ids=np.array(ids),cloth=np.array(cloth,np.float32),shape=np.array(shape,np.float32),boxes=np.array(boxes,np.float32),masked=np.array(flags))
 atomic_json(out/'feature_audit.json',{'n':len(ids),'masked':sum(flags),'failures':failures,'seconds':time.time()-t,'preprocessing':'SAM2.1 L, original counter full frame and box, torso .35, connected component; staff_masked.crop_masked margin .05 +1 grey128; frozen TransReID MSMT17 and CSCI LTCC.'})
 print('FEATURES_READY',len(ids),'failures',len(failures),flush=True)


def metrics(y,q):
 n=len(y);right=int((y==q).sum())
 return {'n':n,'correct':right,'errors':n-right,'accuracy':right/n if n else None,
  'staff_as_customer':int((y&~q).sum()),'customer_as_staff':int((~y&q).sum()),
  'staff_n':int(y.sum()),'customer_n':int((~y).sum()),
  'staff_recall':float(q[y].mean()) if y.any() else None,'customer_recall':float((~q[~y]).mean()) if (~y).any() else None}


def main():
 out=folder(ROOT);manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
 if (out/'evaluation.json').exists():raise RuntimeError('Completed evaluation exists; refusing to overwrite')
 model_hash=hashlib.sha256((out/'frozen_model.pkl').read_bytes()).hexdigest()
 assert model_hash==manifest['model_sha256'],'Frozen model changed'
 labels=json.loads((out/'labels.json').read_text(encoding='utf-8'))
 assert set(labels)==set(manifest['items']),'Annotation is incomplete or has unexpected IDs'
 if not (out/'evaluation_labels.json').exists():atomic_json(out/'evaluation_labels.json',labels)
 frozen_labels=json.loads((out/'evaluation_labels.json').read_text(encoding='utf-8'))
 assert labels==frozen_labels,'Labels changed since evaluation started'
 if not (out/'features.npz').exists():extract(out,manifest)
 spec=importlib.util.spec_from_file_location('staff_model',out/'frozen_staff_model.py');mod=importlib.util.module_from_spec(spec);sys.modules['staff_model']=mod;spec.loader.exec_module(mod)
 with (out/'frozen_model.pkl').open('rb') as f:model=pickle.load(f)
 z=np.load(out/'features.npz');ids=[str(i) for i in z['ids']]
 p=model.predict_proba(z['cloth'],z['shape'],z['boxes'],np.array(['cam1']*len(ids)))[:,1]
 q=p>=model.threshold;y=np.array([frozen_labels[i]['label']==2 for i in ids]);defined=np.array([frozen_labels[i]['label'] in (1,2) for i in ids]);days=np.array([manifest['items'][i]['day'] for i in ids])
 report={'days':manifest['days'],'model_sha256':model_hash,'threshold':model.threshold,'total':len(labels),'feature_n':len(ids),
 'unknown':sum(v['label']==0 for v in labels.values()),'metrics':metrics(y[defined],q[defined]),
 'by_day':{d:metrics(y[defined&(days==d)],q[defined&(days==d)]) for d in manifest['days']},
 'feature_audit':json.loads((out/'feature_audit.json').read_text()),
 'labels_sha256':hashlib.sha256((out/'evaluation_labels.json').read_bytes()).hexdigest(),
 'protocol':'Frozen classifier and threshold .45; no fitting, no PCA updates; two previously unseen historical days annotated without guesses; counter events, not person-unique or future-only benchmark.',
 'limitations':['Snapshot full frames contain baked-in old counter overlays. Same preprocessing as training; performance on clean frames and future days not established.','Unknowns reported separately, not counted as correct.']}
 report['errors']=[{'id':i,'day':manifest['items'][i]['day'],'time':manifest['items'][i]['time'],'label':frozen_labels[i]['label'],'prediction':2 if q[k] else 1,'score':float(p[k])} for k,i in enumerate(ids) if defined[k] and q[k]!=y[k]]
 report['unprocessed_defined']=[i for i,v in labels.items() if v['label'] in (1,2) and i not in set(ids)]
 # Denominator with preprocessing failures counted as errors, so omissions cannot flatter accuracy.
 report['accuracy_all_defined']=report['metrics']['correct']/sum(v['label'] in (1,2) for v in labels.values())
 np.savez_compressed(out/'predictions.npz',ids=np.array(ids),p=p,q=q,y=y,defined=defined,days=days)
 assert hashlib.sha256((out/'frozen_model.pkl').read_bytes()).hexdigest()==model_hash
 atomic_json(out/'evaluation.json',report)
 print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ('errors',)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
