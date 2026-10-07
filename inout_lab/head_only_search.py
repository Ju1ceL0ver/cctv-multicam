"""Expert deliberately independent of visible feet: position + upper mask shape."""
from pathlib import Path
import numpy as np,json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;h=np.load(root/'diagnostics/head_features.npy');r=np.load(root/'diagnostics/rich_features.npy');virtual=np.load(root/'diagnostics/virtual_foot_features.npy');sets={'upper_only':h[:,:32],'upper_geometry':np.column_stack([h[:,:32],r[:,:4],virtual])};report={}
for name,A in sets.items():
 P=np.zeros((len(y),3))
 for d in np.unique(days):
  tr=(days!=d)&valid;te=days==d;m=make_pipeline(StandardScaler(),SVC(C=5,gamma='scale',probability=True,random_state=19)).fit(A[tr],y[tr]);P[te]=m.predict_proba(A[te])
 pred=P.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);report[name]={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int(((pred!=y)&valid).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist()};np.savez(root/('diagnostics/'+name+'_predictions.npz'),probabilities=P);print(name,report[name],flush=True)
(root/'head_only_report.json').write_text(json.dumps(report,indent=2))
