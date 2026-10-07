"""Diagnose direct inside/outside errors, separately from doorway classification.
Every tested day's labels excluded from fitting; unknown and doorway excluded
from binary target. This is a different metric from three-class accuracy.
"""
from pathlib import Path
import json,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
root=Path(__file__).parent;out=root/'diagnostics';z=np.load(root/'io_cam1.npz');y=z['y'];d=z['day'];g=z['G'][:,1:];raw=np.load(out/'original_labels.npy');rich=np.load(out/'rich_features.npy');valid=(raw!=0)&(y<2)
sets={'geometry':g,'geometry_no_floor':g[:,:8],'head_box_only':g[:,:6],'rich':rich}
r={}
for name,A in sets.items():
 P=np.zeros((len(y),2));per={}
 for day in np.unique(d):
  tr=(d!=day)&valid;te=(d==day)&valid;m=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=1.,random_state=0,n_jobs=4).fit(A[tr],y[tr]);P[te]=m.predict_proba(A[te]);per[str(day)]={'n':int(te.sum()),'wrong':int((P[te].argmax(1)!=y[te]).sum())}
 r[name]={'accuracy':float((P[valid].argmax(1)==y[valid]).mean()),'wrong':int((P[valid].argmax(1)!=y[valid]).sum()),'n':int(valid.sum()),'days':per};print(name,r[name],flush=True)
 np.savez(out/('binary_'+name+'.npz'),probabilities=P,valid=valid)
(out/'binary_report.json').write_text(json.dumps(r,indent=2))
