"""Leave ONE day out; every prediction made without labels from its own day."""
from pathlib import Path
import json,pickle,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
root=Path(__file__).parent;out=root/'diagnostics';z=np.load(root/'io_cam1.npz');y=z['y'];day=z['day'];door=z['door'];original=np.load(out/'original_labels.npy');A=np.load(out/'rich_features.npy');P=np.zeros((len(y),3));r={}
for d in np.unique(day):
 tr=(day!=d)&(original!=0);te=day==d
 m=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=1,max_features=1.,random_state=1,n_jobs=4).fit(A[tr],y[tr]);P[te]=m.predict_proba(A[te]);pred=P[te].argmax(1);valid=original[te]!=0;near=door[te]&valid
 r[str(d)]={'all_legacy':float((pred==y[te]).mean()),'defined':float((pred[valid]==y[te][valid]).mean()),'defined_n':int(valid.sum()),'door_defined':float((pred[near]==y[te][near]).mean()) if near.any() else None,'door_n':int(near.sum()),'confusion':confusion_matrix(y[te][valid],pred[valid],labels=[0,1,2]).tolist()}
 print(d,r[str(d)],flush=True)
pred=P.argmax(1);valid=original!=0;near=door&valid;r['overall']={'all_defined':float((pred[valid]==y[valid]).mean()),'door_defined':float((pred[near]==y[near]).mean()),'door_n':int(near.sum()),'all_n':int(valid.sum()),'wrong':int((pred[valid]!=y[valid]).sum()),'door_wrong':int((pred[near]!=y[near]).sum())}
np.savez(out/'lodo_predictions.npz',probabilities=P,valid=valid);(out/'lodo_report.json').write_text(json.dumps(r,indent=2))
m=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=1,max_features=1.,random_state=1,n_jobs=4).fit(A[valid],y[valid]);pickle.dump(m,open(out/'rich_final.pkl','wb'));print('OVERALL',r['overall'],flush=True)
