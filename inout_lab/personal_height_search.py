"""Label-free per-track height/camera-height estimation from visible moments.
Relative units; no metric scale and no hard side override. All rows/classes kept.
"""
from pathlib import Path
import json,gzip,time
import numpy as np,cv2
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from track_context_features import geometry
from owner_boundary_search import metrics
R=Path(__file__).parent

def build():
 z=np.load(R/'io_cam1.npz');A=np.load(R/'diagnostics/recovered_features.npy');heads=A[:,234:236];matches=json.loads((R/'track_context_match_report.json').read_text())['matches'];cal=json.loads((R/'diagnostics/calib_final_source.json').read_text())['cam1'];K=np.array(cal['K'],float);K[:2]*=.5;dist=np.array(cal['dist']);n=cv2.Rodrigues(np.array(cal['rvec']))[0][:,2]
 def rays(p):
  u=cv2.undistortPoints((np.asarray(p)*[1280,720]).astype(float).reshape(-1,1,2),K,dist).reshape(-1,2);return np.c_[u,np.ones(len(u))]
 ends=np.array(next(o['points'] for o in json.loads((R/'owner_seams.json').read_text())['objects'] if o['id']=='threshold1'))/[1280,720];line=np.cross(*rays(ends));line/=np.linalg.norm(line[:2])
 if rays([[.5,.9]])[0]@line<0:line=-line
 tracks={}
 for day in ['20260917','20260918','20260919']:
  with gzip.open(R/f'diagnostics/temporal/{day}_sam31.jsonl.gz','rt') as f:
   next(f)
   for text in f:
    tick=json.loads(text)
    for p in tick['p']:
     if p['foot'] is not None:tracks.setdefault((day,tick['s'],p['w']),[]).append((tick['t'],geometry(p)))
 data={}
 for key,rows in tracks.items():
  t=np.array([r[0] for r in rows]);b=np.array([r[1] for r in rows]);head=np.c_[b[:,0],b[:,1]-.475*b[:,3]];hq=rays(head);fq=rays(b[:,4:6]);a=np.cross(fq,hq);delta=np.cross(fq,(hq@n)[:,None]*n-hq);ratio=-(a*delta).sum(1)/np.maximum((delta*delta).sum(1),1e-10);res=np.linalg.norm(a+ratio[:,None]*delta,axis=1)
  good=(ratio>.05)&(ratio<.9)&(abs(b[:,5]-(b[:,1]+b[:,3]/2))<.03)&(b[:,2]/np.maximum(b[:,3],.001)<.8)&(b[:,3]>.035)&(res<.015)
  data[key]=(t,ratio,good)
 Q=rays(heads);out=np.zeros((len(heads),37),np.float32)
 for ix,m in matches.items():
  i=int(ix);key=(m['day'],m['span'],m['track'])
  if key not in data:continue
  t,ratio,good=data[key];vec=[1.]
  for past,future in [(10,0),(10,10),(30,30)]:
   use=good&(t>=m['time']-past)&(t<=m['time']+future+1e-5);h=ratio[use]
   if not len(h):vec.extend([0.]*12);continue
   values=np.quantile(h,[.5,.75,.9]);vec.extend([min(len(h)/100,1),float(h.std()),*values])
   for value in values:
    p=(1-value)*Q[i]+value*(Q[i]@n)*n;p=p/max(p[2],.01);vec.extend([float(np.clip(p@line,-3,3)),float(np.clip(p[1],-3,3))])
   vec.append(float(np.min(abs(t[use]-m['time']))/30))
  assert len(vec)==37;out[i]=vec
 np.save(R/'diagnostics/personal_height_features.npy',out);return out

if __name__=='__main__':
 start=time.perf_counter();E=build();z=np.load(R/'io_cam1.npz');y=z['y'];days=z['day'];v=np.load(R/'diagnostics/original_labels.npy')!=0;A=np.load(R/'diagnostics/recovered_features.npy')[:,:1584];G=np.load(R/'diagnostics/aligned_geometry.npy');sets={'context':np.c_[A,E],'geometry':np.c_[G,A[:,:39],A[:,234:360],E]};P={k:np.full((len(y),3),np.nan) for k in sets}
 with threadpool_limits(limits=2):
  for d in np.unique(days):
   tr=v&(days!=d);te=days==d
   for name,X in sets.items():
    model=ExtraTreesClassifier(n_estimators=160,max_features=.7,random_state=19,n_jobs=2) if name=='context' else HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19);P[name][te]=model.fit(X[tr],y[tr]).predict_proba(X[te])
   print('personal height',d,'elapsed',round(time.perf_counter()-start),flush=True)
 report={}
 for name,p in P.items():
  report[name]=metrics(p,y,v,days);np.savez(R/f'diagnostics/personal_height_{name}.npz',probabilities=p,completed_days=np.unique(days));print(name,report[name]['wrong'],report[name]['gross'],report[name]['gross_indices'],flush=True)
 (R/'personal_height_report.json').write_text(json.dumps({'models':report,'seconds':time.perf_counter()-start,'future_seconds_max':30,'height_in_metres':False,'label_free_track_features':True},indent=2))
