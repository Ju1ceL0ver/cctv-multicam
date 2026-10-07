from pathlib import Path
import numpy as np,json
from sklearn.ensemble import ExtraTreesClassifier
from spatial_prior import SpatialPrior
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];door=z['door'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;A=np.column_stack([np.load(root/('diagnostics/'+n+'_features.npy')) for n in ['rich','head','context']]);unique=np.unique(days);cache={};pred={name:np.zeros((len(y),3)) for name in ['means','means_knn','knn_only']}

def cross_features(excluded,indices):
 key=tuple(sorted(excluded))
 if key not in cache:
  tr=valid&~np.isin(days,list(excluded));prior=SpatialPrior().fit(A[tr],y[tr]);cache[key]=prior
 return cache[key].transform(A[indices])

for day in unique:
 tr=valid&(days!=day);te=days==day;train_idx=np.where(tr)[0];test_idx=np.where(te)[0]
 test=cross_features({day},test_idx);train=np.empty((len(train_idx),test.shape[1]),np.float32)
 for cal in unique:
  ix=np.where(days[train_idx]==cal)[0]
  if len(ix):train[ix]=cross_features({day,cal},train_idx[ix])
 for name,limit in [('means',36),('means_knn',test.shape[1])]:
  model=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=4).fit(np.column_stack([A[tr],train[:,:limit]]),y[tr])
  pred[name][te]=model.predict_proba(np.column_stack([A[te],test[:,:limit]]))
 # Direct coordinate kNN: joint head+centroid+foot, k=25 weighted votes.
 pred['knn_only'][te]=test[:,129:132]
 print('spatial validated',day,flush=True)
report={}
for name,P in pred.items():
 p=P.argmax(1);gross=valid&(y<2)&(p<2)&(p!=y);r={'accuracy':float((p[valid]==y[valid]).mean()),'wrong':int(((p!=y)&valid).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'days':{str(d):{'wrong':int(((p!=y)&valid&(days==d)).sum()),'gross':int((gross&(days==d)).sum())} for d in unique}}
 report[name]=r;np.savez(root/('diagnostics/spatial_'+name+'.npz'),probabilities=P);print(name,r,flush=True)
prior=SpatialPrior().fit(A[valid],y[valid]);(root/'spatial_means.json').write_text(json.dumps(prior.summary(),indent=2));np.save(root/'spatial_channels.npy',prior.channels());(root/'spatial_report.json').write_text(json.dumps(report,indent=2))
