"""Hierarchical door-vs-defined and side heads, evaluated on ALL three classes.
Side head's undefined doorway target is excluded from its loss, not evaluation.
"""
from pathlib import Path
import numpy as np,json
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];door=z['door'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;A=np.column_stack([np.load(root/('diagnostics/'+n+'_features.npy')) for n in ['rich','head','context']]);P={name:np.zeros((len(y),3)) for name in ['hier_forest','hier_boost','side_forest','side_boost']};heads={name:np.zeros((len(y),2)) for name in ['forest','boost']};gates={name:np.zeros(len(y)) for name in ['forest','boost']}
with threadpool_limits(limits=4):
 for d in np.unique(days):
  tr=(days!=d)&valid;te=days==d;side_tr=tr&(y<2)
  for name,make in [('forest',lambda:ExtraTreesClassifier(n_estimators=160,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=4)),('boost',lambda:HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19))]:
   side=make().fit(A[side_tr],y[side_tr]).predict_proba(A[te]);gate=make().fit(A[tr],(y[tr]==2).astype(int)).predict_proba(A[te])[:,1];heads[name][te]=side;gates[name][te]=gate;P['hier_'+name][te]=np.column_stack([side*(1-gate[:,None]),gate]);P['side_'+name][te]=np.column_stack([side,np.zeros(len(side))])
  print('hierarchy',d,flush=True)
report={}
for name,p in P.items():
 pred=p.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);report[name]={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int(((pred!=y)&valid).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean())};np.savez(root/('diagnostics/'+name+'.npz'),probabilities=p);print(name,report[name],flush=True)
(root/'hierarchical_report.json').write_text(json.dumps(report,indent=2));np.savez(root/'diagnostics/hierarchical_heads.npz',side_forest=heads['forest'],side_boost=heads['boost'],door_forest=gates['forest'],door_boost=gates['boost'])
