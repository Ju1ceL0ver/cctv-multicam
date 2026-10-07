"""Visibility-dependent fusion of detailed body/context and head-only experts.
Outer evaluation excludes an entire day. Gate training sees OUT-OF-BAG body
predictions and day-cross-fitted head predictions, not self-label lookups.
"""
from pathlib import Path
import numpy as np,json
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
root=Path(__file__).parent

def gate_features(A,probabilities,virtual):
 r=A[:,:234];h=A[:,234:936];body_height=(r[:,13]-r[:,12])*180;head_width=(h[:,7]-h[:,3])*320
 ratio=np.clip(body_height/(head_width+.1),0,50)
 bbox_width=(r[:,15]-r[:,14])*320
 bottom_coverage=np.clip(h[:,122]*57600/(np.maximum(1,body_height/12)*np.maximum(1,bbox_width)),0,1.5)
 return np.column_stack([*probabilities,r[:,:39],h[:,:126],virtual,ratio,bottom_coverage]).astype(np.float32)

if __name__=='__main__':
 z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;A=np.column_stack([np.load(root/('diagnostics/'+n+'_features.npy')) for n in ['rich','head','context']]+[np.load(root/'diagnostics/crop_rgb_features.npy')]);h=np.load(root/'diagnostics/head_features.npy');virtual=np.load(root/'diagnostics/virtual_foot_features.npy');head_sets=[h[:,:32],np.column_stack([h[:,:32],A[:,:4],virtual])];P=np.zeros((len(y),3));body_prob=np.zeros_like(P)
 with threadpool_limits(limits=3):
  for d in np.unique(days):
   tr=(days!=d)&valid;te=days==d;idx=np.where(tr)[0];forest=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=3,bootstrap=True,max_samples=.8,oob_score=True).fit(A[tr],y[tr]);train_probs=[forest.oob_decision_function_];test_probs=[forest.predict_proba(A[te])];body_prob[te]=test_probs[0]
   assert np.all(np.isfinite(train_probs[0])) and np.all(train_probs[0].sum(1)>.99)
   for features in head_sets:
    train_p=np.zeros((len(idx),3))
    for inner in np.unique(days[tr]):
     fit=tr&(days!=inner);cal=days[idx]==inner;m=make_pipeline(StandardScaler(),SVC(C=5,gamma='scale',probability=True,random_state=19)).fit(features[fit],y[fit]);train_p[cal]=m.predict_proba(features[idx[cal]])
    m=make_pipeline(StandardScaler(),SVC(C=5,gamma='scale',probability=True,random_state=19)).fit(features[tr],y[tr]);train_probs.append(train_p);test_probs.append(m.predict_proba(features[te]))
   gate=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=7,min_samples_leaf=10,l2_regularization=5,random_state=19).fit(gate_features(A[tr],train_probs,virtual[tr]),y[tr]);P[te]=gate.predict_proba(gate_features(A[te],test_probs,virtual[te]));print('visibility gate',d,flush=True)
 pred=P.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);r={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int(((pred!=y)&valid).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean()),'doorway_precision':float((y[valid&(pred==2)]==2).mean())};np.savez(root/'diagnostics/visibility_gate_predictions.npz',probabilities=P,body_probabilities=body_prob);(root/'visibility_gate_report.json').write_text(json.dumps(r,indent=2));print('VISIBILITY GATE',r,flush=True)
