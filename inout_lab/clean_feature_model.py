"""Exclude undefined original labels from training; report original and defined-label scores separately."""
import json,pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
root=Path(__file__).parent;out=root/'diagnostics';z=np.load(root/'io_cam1.npz');y=z['y'];day=z['day'];original=np.load(out/'original_labels.npy');A=np.load(out/'rich_features.npy');tr=~np.isin(day,['20260918','20260919'])&(original!=0)
m=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=1,max_features=1.,random_state=1,n_jobs=4).fit(A[tr],y[tr]);r={'train_count':int(tr.sum())};ps={}
for d,split in [('20260918','development18'),('20260919','validation19')]:
 ix=day==d;p=m.predict_proba(A[ix]);valid=original[ix]!=0;pred=p.argmax(1);ps['rich_clean_'+split]=p
 r[split]={'all_legacy_accuracy':float((pred==y[ix]).mean()),'defined_accuracy':float((pred[valid]==y[ix][valid]).mean()),'defined_count':int(valid.sum()),'all_count':int(ix.sum()),'confusion':confusion_matrix(y[ix],pred,labels=[0,1,2]).tolist()}
np.savez(out/'clean_predictions.npz',**ps);pickle.dump(m,open(out/'rich_clean.pkl','wb'));(out/'clean_report.json').write_text(json.dumps(r,indent=2));print(r)
