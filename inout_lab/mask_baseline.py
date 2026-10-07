"""Explicit fast mask/floor geometry; no test labels used in fitting."""
from pathlib import Path
import json,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');f=z['F'];g=z['G'][:,1:];y=z['y'];days=z['day'];m=f[:,:,:,3]/255.;floor=f[:,:,:,4]/255.
a=m.sum((1,2));features=[]
for i in range(len(y)):
 mm=m[i];ff=floor[i];rows=np.where(mm.sum(1)>0)[0];bottom=np.zeros_like(mm);cut=rows[-1]-max(1,int((rows[-1]-rows[0]+1)*.2));bottom[cut:]=mm[cut:]
 overlap=(mm*ff).sum()/a[i];bo=(bottom*ff).sum()/max(1,bottom.sum())
 xx,yy=np.meshgrid(np.linspace(0,1,320),np.linspace(0,1,180))
 features.append([overlap,bo,a[i]/(180*320),(mm*xx).sum()/a[i],(mm*yy).sum()/a[i]])
A=np.column_stack([g,features]);np.save(root/'diagnostics/mask_features.npy',A)
tr=~np.isin(days,['20260918','20260919']);model=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=2,max_features=1.,random_state=0,n_jobs=4).fit(A[tr],y[tr]);report={};preds=dict(np.load(root/"diagnostics/predictions.npz"))
for split,day in [('development18','20260918'),('validation19','20260919')]:
 ix=days==day;p=model.predict_proba(A[ix]);preds["mask_extra_trees_"+split]=p;report[split]={'accuracy':float((p.argmax(1)==y[ix]).mean()),'confusion':confusion_matrix(y[ix],p.argmax(1),labels=[0,1,2]).tolist()}
import pickle
pickle.dump(model,open(root/"diagnostics/mask_extra_trees.pkl","wb"))
np.savez(root/"diagnostics/predictions.npz",**preds)
combined=json.loads((root/"diagnostics/report.json").read_text());combined["mask_extra_trees"]=report;(root/"diagnostics/report.json").write_text(json.dumps(combined,indent=2))
print(report)
(root/'diagnostics/mask_report.json').write_text(json.dumps(report,indent=2))
