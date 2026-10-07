from pathlib import Path
import json,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];raw=np.load(root/'diagnostics/original_labels.npy');A=np.load(root/'diagnostics/rich_features.npy');valid=(raw!=0)&(y<2);P=np.zeros((len(y),2))
for day in np.unique(days):
 tr=(days!=day)&valid;te=days==day
 model=ExtraTreesClassifier(n_estimators=32,min_samples_leaf=2,max_features=1.,random_state=0,n_jobs=4).fit(A[tr],y[tr]);P[te]=model.predict_proba(A[te])
ix=y==2;r={'doorway_count':int(ix.sum()),'thresholds':{str(t):{'uncertain':int((P[ix].max(1)<t).sum()),'confident':int((P[ix].max(1)>=t).sum())} for t in [.8,.9,.95]}}
np.savez(root/'diagnostics/binary_all_predictions.npz',probabilities=P);(root/'binary_doorway_report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
