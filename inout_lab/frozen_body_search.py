"""Held-day small classifiers on existing pretrained body features, no fine-tune."""
from pathlib import Path
import json,time,argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from owner_boundary_search import metrics
R=Path(__file__).parent
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--tag',default='');ap.add_argument('--features-source');args=ap.parse_args();suffix=('_'+args.tag) if args.tag else '';start=time.perf_counter();z=np.load(R/'io_cam1.npz');y=z['y'];days=z['day'];v=np.load(R/'diagnostics/original_labels.npy')!=0;body=np.load(R/f'diagnostics/frozen_body{suffix}_features.npy');A=np.load(args.features_source or R/'diagnostics/aligned_features.npy');geo=np.column_stack([np.load(R/'diagnostics/aligned_geometry.npy'),A[:,:39],A[:,234:360],np.load(R/'diagnostics/owner_boundary_features.npy')]);P={k:np.full((len(y),3),np.nan) for k in ['svm10','svm100','boost','forest']};folds=[]
 with threadpool_limits(limits=2):
  for d in np.unique(days):
   tr=v&(days!=d);te=days==d;gs=StandardScaler().fit(geo[tr]);bs=StandardScaler().fit(body[tr]);G=gs.transform(geo)/np.sqrt(geo.shape[1]);scaled=bs.transform(body);pc=PCA(n_components=64,svd_solver='randomized',iterated_power=3,random_state=19).fit(scaled[tr]);B=pc.transform(scaled)/np.sqrt(body.shape[1]);X=np.column_stack([G,B])
   for C in [10,100]:
    m=SVC(C=C,gamma='scale',probability=False,break_ties=True,random_state=19).fit(X[tr],y[tr]);score=m.decision_function(X[te]);score-=score.max(1,keepdims=True);p=np.exp(score);P[f'svm{C}'][te]=p/p.sum(1,keepdims=True)
   X=np.column_stack([geo,B]);m=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19);P['boost'][te]=m.fit(X[tr],y[tr]).predict_proba(X[te]);X=np.column_stack([A[:,:1584],body]);P['forest'][te]=ExtraTreesClassifier(n_estimators=160,max_features=.7,n_jobs=2,random_state=19).fit(X[tr],y[tr]).predict_proba(X[te]);print('frozen evaluated',d,'elapsed',round(time.perf_counter()-start),flush=True)
   folds.append(str(d))
   for name,p in P.items():np.savez(R/f'diagnostics/frozen_body{suffix}_{name}.npz',probabilities=p,completed_days=np.array(folds))
 report={}
 for name,p in P.items():
  report[name]=metrics(p,y,v,days);np.savez(R/f'diagnostics/frozen_body{suffix}_{name}.npz',probabilities=p,completed_days=np.unique(days));print(name,report[name]['wrong'],report[name]['gross'],report[name]['gross_indices'],flush=True)
 (R/f'frozen_body{suffix}_report.json').write_text(json.dumps({'models':report,'seconds':time.perf_counter()-start,'svm_scores':'Normalized OVR decision scores, not calibrated posterior probabilities','protocol':'Whole-day exclusion of model/scalers/PCA, frozen COCO pose backbone never trained on this dataset'},indent=2))
