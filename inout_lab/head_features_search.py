from pathlib import Path
import numpy as np,cv2,json,pickle
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from scipy.ndimage import distance_transform_edt
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');F=z['F'];X=z['X'];G=z['G'];y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;rich=np.load(root/'diagnostics/rich_features.npy');rows=[]
for f in F:
 m=f[:,:,3]/255.;r,c=np.where(m>.25);top,bottom=r.min(),r.max();v=[]
 for q in [.05,.1,.15,.25,.4,.6,.8,1.]:
  cut=top+max(1,int((bottom-top+1)*q));rr,cc=np.where((m>.25)&(np.indices(m.shape)[0]<cut));weights=m[rr,cc];a=max(1,weights.sum())
  v += [(cc*weights).sum()/a/320,(rr*weights).sum()/a/180,len(rr)/m.size]
  v+=list(np.percentile(cc,[0,10,50,90,100])/320) if len(cc) else [0]*5
 for q in [0,10,25,50,75,90,100]:v += [np.percentile(r,q)/180,np.percentile(c,q)/320]
 for row in np.array_split(np.arange(top,bottom+1),12):
  rr,cc=np.where(m[row]>.25);v += [len(cc)/m.size,float(cc.mean()/320) if len(cc) else 0,float(cc.min()/320) if len(cc) else 0,float(cc.max()/320) if len(cc) else 0]
 v+=cv2.resize(m,(32,18),interpolation=cv2.INTER_AREA).ravel().tolist();rows.append(v)
head=np.asarray(rows,np.float32);np.save(root/'diagnostics/head_features.npy',head);sets={'rich_head':np.column_stack([rich,head]),'head_geometry':np.column_stack([G[:,1:9],head])}
configs={'extra':lambda:ExtraTreesClassifier(n_estimators=200,min_samples_leaf=1,max_features=1.,random_state=7,n_jobs=4),'forest':lambda:RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_features=.7,random_state=7,n_jobs=4)}
report={}
for name,A in sets.items():
 for algo,make in configs.items():
  key=name+'_'+algo;P=np.zeros((len(y),3))
  for day in np.unique(days):
   tr=(days!=day)&valid;te=days==day;P[te]=make().fit(A[tr],y[tr]).predict_proba(A[te])
  pred=P.argmax(1);cross=valid&(y<2)&(pred<2)&(pred!=y);report[key]={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int((pred[valid]!=y[valid]).sum()),'gross':int(cross.sum()),'days':{str(d):{'wrong':int(((pred!=y)&valid&(days==d)).sum()),'gross':int((cross&(days==d)).sum())} for d in np.unique(days)}}
  np.savez(root/('diagnostics/'+key+'.npz'),probabilities=P);print(key,report[key],flush=True)
(root/'head_features_report.json').write_text(json.dumps(report,indent=2))
