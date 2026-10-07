import json,numpy as np
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
z=np.load('io_cam1.npz');d=z['day'];o=np.load('diagnostics/original_labels.npy');y3=z['y']
D='diagnostics/';rich=np.load(D+'rich_features.npy');ctx=np.load(D+'context_features.npy')
lab=json.load(open('relabel2.json'));y=np.full(len(d),-1)
y[(o==1)]=1;y[(o==2)]=0
for i,v in lab.items(): y[int(i)]=1 if v=='inside' else 0
old=(o==3)|(o==0);v=y>=0
print('n',v.sum(),'inside',(y[v]==1).sum(),'rel',old.sum())
for name,A in {'rich':rich,'rich+ctx':np.hstack([rich,ctx])}.items():
 for mn,mk in {'ET':lambda:ExtraTreesClassifier(300,min_samples_leaf=2,max_features=1.,random_state=0,n_jobs=4),'HGB':lambda:HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,random_state=0)}.items():
  P=np.zeros(len(y))
  for day in np.unique(d):
   tr=(d!=day)&v;te=(d==day)&v;P[te]=mk().fit(A[tr],y[tr]).predict_proba(A[te])[:,1]
  w=(P>.5)!=(y==1)
  print(name,mn,'wrong',w[v].sum(),'/',v.sum(),'| old-doorway',w[old].sum(),'/',old.sum(),'| clear',w[v&~old].sum(),'/',(v&~old).sum(),flush=True)
  if name=='rich+ctx' and mn=='ET': np.save(D+'binary2_P.npy',P)
