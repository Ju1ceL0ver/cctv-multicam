"""kNN with visual context and a learned metric; every held day excluded."""
from pathlib import Path
import numpy as np,json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis
from threadpoolctl import threadpool_limits
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;r=np.load(root/'diagnostics/rich_features.npy');h=np.load(root/'diagnostics/head_features.npy');c=np.load(root/'diagnostics/context_features.npy')
# Explicit anchors + shape/occlusion + context color means/std at each scale.
A=np.column_stack([r[:,:36],h[:,:78],c.reshape(len(c),12,54)[:,:,48:54].reshape(len(c),-1)])
P={name:np.zeros((len(y),3)) for name in ['pca','learned_metric']}
with threadpool_limits(limits=3):
 for day in np.unique(days):
  tr=(days!=day)&valid;te=days==day;scaler=StandardScaler().fit(A[tr]);pca=PCA(n_components=16,random_state=11).fit(scaler.transform(A[tr]));train=pca.transform(scaler.transform(A[tr]));test=pca.transform(scaler.transform(A[te]))
  knn=KNeighborsClassifier(n_neighbors=15,weights='distance').fit(train,y[tr]);P['pca'][te]=knn.predict_proba(test)
  nca=NeighborhoodComponentsAnalysis(n_components=10,max_iter=35,random_state=11,init='pca').fit(train,y[tr]);knn=KNeighborsClassifier(n_neighbors=15,weights='distance').fit(nca.transform(train),y[tr]);P['learned_metric'][te]=knn.predict_proba(nca.transform(test));print('metric',day,flush=True)
report={}
for name,p in P.items():
 pred=p.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);report[name]={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int(((pred!=y)&valid).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist()};np.savez(root/('diagnostics/metric_knn_'+name+'.npz'),probabilities=p);print(name,report[name],flush=True)
(root/'metric_knn_report.json').write_text(json.dumps(report,indent=2))
