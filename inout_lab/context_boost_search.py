from pathlib import Path
import numpy as np,json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;A=np.column_stack([np.load(root/('diagnostics/'+n+'_features.npy')) for n in ['rich','head','context']]);report={}
for name,make in [('boost',lambda:HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19)),('svm',lambda:make_pipeline(StandardScaler(),SVC(C=10,probability=True,random_state=19)))]:
 P=np.zeros((len(y),3))
 for day in np.unique(days):
  tr=(days!=day)&valid;te=days==day;P[te]=make().fit(A[tr],y[tr]).predict_proba(A[te]);print(name,day,flush=True)
 pred=P.argmax(1);cross=valid&(y<2)&(pred<2)&(pred!=y);r={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int((pred[valid]!=y[valid]).sum()),'gross':int(cross.sum())};report[name]=r;np.savez(root/('diagnostics/context_'+name+'.npz'),probabilities=P);print(name,r,flush=True)
(root/'context_boost_report.json').write_text(json.dumps(report,indent=2))
