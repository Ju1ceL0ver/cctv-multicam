"""Recovered-pixel appearance, physical boundary and real-track context.
Single fixed configuration; every transform/model excludes the whole test day.
"""
from pathlib import Path
import json,time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
from cost_boost_search import loss
from owner_boundary_search import metrics
R=Path(__file__).parent
if __name__=='__main__':
 start=time.perf_counter();z=np.load(R/'io_cam1.npz');y=z['y'];days=z['day'];v=np.load(R/'diagnostics/original_labels.npy')!=0;A=np.load(R/'diagnostics/recovered_features.npy');body=np.load(R/'diagnostics/frozen_body_recovered_features.npy');E=np.load(R/'diagnostics/owner_boundary_features.npy');T=np.load(R/'diagnostics/owner_track_30_features.npy');P=np.full((len(y),3),np.nan);folds=[]
 with threadpool_limits(limits=2):
  for d in np.unique(days):
   tr=v&(days!=d);te=days==d;scaler=StandardScaler().fit(body[tr]);B=scaler.transform(body);pc=PCA(n_components=64,svd_solver='randomized',iterated_power=3,random_state=19).fit(B[tr]);X=np.column_stack([A,E,T,pc.transform(B)])
   model=XGBClassifier(n_estimators=220,max_depth=3,learning_rate=.07,subsample=.9,colsample_bytree=.8,min_child_weight=3,reg_lambda=3,n_jobs=2,random_state=19,objective=loss(3),num_class=3);model.fit(X[tr],y[tr]);P[te]=model.predict_proba(X[te]);folds.append(str(d));np.savez(R/'diagnostics/recovered_joint_boost.npz',probabilities=P,completed_days=np.array(folds));print('joint boost',d,'elapsed',round(time.perf_counter()-start),flush=True)
 report=metrics(P,y,v,days);report.update(seconds=time.perf_counter()-start,protocol='LODO all9days, diagnostic not fresh blind; no abstention',future_seconds=30,new_model_configuration_count=1);(R/'recovered_joint_boost_report.json').write_text(json.dumps(report,indent=2));print('JOINT BOOST',report,flush=True)
