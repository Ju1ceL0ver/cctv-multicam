"""All-class evaluation of actual track geometry with complete day exclusion.
Causal window10s; buffered additionally uses at most3s future (user allows delay).
Missing history remains an explicit flag; all1581 defined rows retained.
These repeatedly studied days are diagnostic, not a fresh blind benchmark.
"""
from pathlib import Path
import numpy as np,json
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
root=Path(__file__).parent
if __name__=='__main__':
 z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];v=np.load(root/'diagnostics/original_labels.npy')!=0;T=np.load(root/'diagnostics/track_context_features.npy');A=np.column_stack([np.load(root/('diagnostics/'+name+'_features.npy')) for name in ['rich','head','context']]);sets={'causal_context':np.column_stack([A,T[:,:133]]),'buffered_context':np.column_stack([A,T]),'buffered_geometry':np.column_stack([z['G'][:,1:],T])};P={k:np.zeros((len(y),3)) for k in sets};report={}
 with threadpool_limits(limits=2):
  for d in np.unique(days):
   tr=(days!=d)&v;te=days==d
   for name,B in sets.items():
    model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19) if name=='buffered_geometry' else ExtraTreesClassifier(n_estimators=200,max_features=.7,random_state=19,n_jobs=2)
    model.fit(B[tr],y[tr]);P[name][te]=model.predict_proba(B[te])
   print('track context evaluated',d,flush=True)
 for name,p in P.items():
  pred=p.argmax(1);gross=v&(y<2)&(pred<2)&(pred!=y);covered=v&(T[:,0]>0);report[name]={'defined':int(v.sum()),'accuracy':float((pred[v]==y[v]).mean()),'wrong':int(((pred!=y)&v).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'track_covered':int(covered.sum()),'track_covered_accuracy':float((pred[covered]==y[covered]).mean()),'track_covered_wrong':int(((pred!=y)&covered).sum()),'doorway_recall':float((pred[y==2]==2).mean()),'future_seconds':0 if name.startswith('causal') else 3};np.savez(root/('diagnostics/'+name+'_predictions.npz'),probabilities=p);print('TRACK_CONTEXT',name,report[name],flush=True)
 (root/'track_context_report.json').write_text(json.dumps(report,indent=2))
