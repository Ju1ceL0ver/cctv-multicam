"""Semantic door floor-plane learned from visible feet; not the free-floor mask.
Plane fitting uses definite side labels on reliable visible feet from TRAIN days.
Doorway is retained in every final classifier/evaluation. Train auxiliary plane
features cross-fit by day; outer evaluated day excluded from every fit.
"""
from pathlib import Path
import numpy as np,json,gzip
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits
root=Path(__file__).parent

def fit_plane(g,y,valid,visible,days,excluded):
 use=valid&visible&(y<2)&~np.isin(days,excluded)
 assert len(np.unique(y[use]))==2
 model=LogisticRegression(C=100,max_iter=500,random_state=19).fit(g[use,6:8],y[use]);coef=model.coef_[0];bias=model.intercept_[0];norm=np.linalg.norm(coef/[1280,720]);return coef,bias,norm,int(use.sum())

def extras(g,plane,matches,tracks,indices,buffered):
 coef,bias,norm,_=plane;result=np.zeros((len(indices),25),np.float32)
 for j,i in enumerate(indices):
  result[j,0]=(g[i,6:8]@coef+bias)/norm/720
  match=matches.get(str(int(i)))
  if match is None:continue
  times,feet=tracks[(match['day'],match['span'],match['track'])];t=match['time'];signed=(feet@coef+bias)/norm/720;vec=[result[j,0],1.]
  for radius in [1.,3.,10.]:
   use=(times>=t-radius)&(times<=t+(min(3.,radius) if buffered else 0.));values=signed[use]
   if len(values):vec.extend([values.mean(),values.std(),values.min(),values.max(),np.mean(values>0),values[0],values[-1]])
   else:vec.extend([0.]*7)
  past=times[times<=t];future=times[(times>=t)&(times<=t+3)] if buffered else np.array([])
  vec.extend([min(10.,t-past.min())/10 if len(past) else 0.,(future.max()-t)/3 if len(future) else 0.]);assert len(vec)==25;result[j]=vec
 return result

if __name__=='__main__':
 z=np.load(root/'io_cam1.npz');g=z['G'][:,1:];y=z['y'];days=z['day'];v=np.load(root/'diagnostics/original_labels.npy')!=0;p=np.load(root/'diagnostics/pose_features.npy');a,b=p[:,45:48],p[:,48:51];feet=(a[:,:2]+b[:,:2])/2;visible=(a[:,2]>.7)&(b[:,2]>.7)&(np.abs(feet[:,1]-g[:,7])<.03)&(np.abs(feet[:,0]-g[:,6])<.05);A=np.column_stack([np.load(root/('diagnostics/'+name+'_features.npy')) for name in ['rich','head','context']]);matches=json.loads((root/'track_context_match_report.json').read_text())['matches'];tracks={}
 for day in ['20260917','20260918','20260919']:
  with gzip.open(root/('diagnostics/temporal/'+day+'_sam31.jsonl.gz'),'rt') as f:
   next(f)
   for line in f:
    tick=json.loads(line)
    for q in tick['p']:
     if q['foot'] is None:continue
     tracks.setdefault((day,tick['s'],q['w']),[]).append((tick['t'],np.asarray(q['foot'])/[2176,1224]))
 tracks={k:(np.array([x[0] for x in rows]),np.stack([x[1] for x in rows])) for k,rows in tracks.items()};P={name:np.zeros((len(y),3)) for name in ['causal_plane','buffered_plane']};planes={}
 with threadpool_limits(limits=2):
  for d in np.unique(days):
   tr=np.where(v&(days!=d))[0];te=np.where(days==d)[0];plane=fit_plane(g,y,v,visible,days,[d]);planes[str(d)]={'coef':plane[0].tolist(),'bias':float(plane[1]),'training_visible_sides':plane[3]}
   for name in P:
    buffered=name.startswith('buffered');E=np.zeros((len(y),25),np.float32);E[te]=extras(g,plane,matches,tracks,te,buffered)
    for cal in np.unique(days[tr]):
     ix=tr[days[tr]==cal];local=fit_plane(g,y,v,visible,days,[d,cal]);E[ix]=extras(g,local,matches,tracks,ix,buffered)
    B=np.column_stack([A,E]);model=ExtraTreesClassifier(n_estimators=200,max_features=.7,random_state=19,n_jobs=2).fit(B[tr],y[tr]);P[name][te]=model.predict_proba(B[te])
   print('door plane',d,'visible training',plane[3],flush=True)
 report={}
 for name,prob in P.items():
  pred=prob.argmax(1);gross=v&(y<2)&(pred<2)&(pred!=y);report[name]={'accuracy':float((pred[v]==y[v]).mean()),'wrong':int(((pred!=y)&v).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean()),'future_seconds':3 if name.startswith('buffered') else 0};np.savez(root/('diagnostics/'+name+'_predictions.npz'),probabilities=prob);print('PLANE',name,report[name],flush=True)
 (root/'door_plane_report.json').write_text(json.dumps({'results':report,'planes':planes},indent=2))
