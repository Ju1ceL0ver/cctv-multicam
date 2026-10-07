from pathlib import Path
import json,pickle,time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from inside_outside import features,InsideOutside
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];A=np.load(root/'diagnostics/rich_features.npy');raw=np.load(root/'diagnostics/original_labels.npy');valid=(raw!=0)&(y<2)
counts=[16,32,64,128,300];pred={n:np.zeros((len(y),2)) for n in counts}
for day in np.unique(days):
 tr=(days!=day)&valid;te=(days==day)&valid
 m=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=1.,random_state=0,n_jobs=4).fit(A[tr],y[tr])
 for n in counts:
  pred[n][te]=sum(t.predict_proba(A[te]) for t in m.estimators_[:n])/n
 print('validated',day,flush=True)
report={str(n):{'accuracy':float((p[valid].argmax(1)==y[valid]).mean()),'wrong':int((p[valid].argmax(1)!=y[valid]).sum())} for n,p in pred.items()}
best=min(counts,key=lambda n:(report[str(n)]['wrong'],n))
m=ExtraTreesClassifier(n_estimators=best,min_samples_leaf=2,max_features=1.,random_state=0,n_jobs=1).fit(A[valid],y[valid])
path=root/'inside_outside.pkl';path.write_bytes(pickle.dumps(m,protocol=5))
p=pred[best];report['selected_trees']=best;report['bytes']=path.stat().st_size;report['defined_count']=int(valid.sum());report['excluded_doorway']=int((y==2).sum());report['excluded_unknown']=int((raw==0).sum())
report['days']={str(day):{'n':int(((days==day)&valid).sum()),'wrong':int((p[(days==day)&valid].argmax(1)!=y[(days==day)&valid]).sum())} for day in np.unique(days)}
report['thresholds']={str(t):{'coverage':float((p[valid].max(1)>=t).mean()),'accuracy':float((p[valid][p[valid].max(1)>=t].argmax(1)==y[valid][p[valid].max(1)>=t]).mean())} for t in [.7,.8,.9,.95]}
F=z['F'];X=z['X'];G=z['G'];diff=max(float(np.max(np.abs(features(F[i],X[i,:,:,3],G[i,1:])-A[i]))) for i in range(0,len(y),17));assert diff<1e-6,diff
model=InsideOutside(); model.predict_prepared(F[0],X[0,:,:,3],G[0,1:]);t=time.perf_counter()
for i in range(100): model.predict_prepared(F[i],X[i,:,:,3],G[i,1:])
report['prepared_ms']=(time.perf_counter()-t)*10;report['feature_parity_max_difference']=diff
frame=np.asarray(F[0,:,:,:3]);mask=F[0,:,:,3]>.25*255
model.predict_rgb(frame,mask);t=time.perf_counter()
for _ in range(100):model.predict_rgb(frame,mask)
report['rgb_mask_pipeline_ms']=(time.perf_counter()-t)*10
(root/'inside_outside_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
