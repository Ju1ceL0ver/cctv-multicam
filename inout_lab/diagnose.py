"""Compare fixed-scene CNN without mirroring and geometric baselines.
18 September is development; 19 September stays excluded for validation.
"""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import train as T
out=Path(__file__).parent/'diagnostics';out.mkdir(exist_ok=True)
tr=np.where(~np.isin(T.day,['20260918','20260919']))[0]
dev=np.where(T.day=='20260918')[0];val=np.where(T.day=='20260919')[0]
report={};predictions={}
def record(name,model,predict):
 report[name]={}
 for split,ix in [('development18',dev),('validation19',val)]:
  p=predict(ix);predictions[name+'_'+split]=p
  report[name][split]={'accuracy':float((p.argmax(1)==T.y[ix]).mean()),'confusion':confusion_matrix(T.y[ix],p.argmax(1),labels=[0,1,2]).tolist()}
 print(name,report[name],flush=True)
 (out/'report.json').write_text(json.dumps(report,indent=2))
 np.savez(out/'predictions.npz',train=tr,development18=dev,validation19=val,**predictions)
for name,m in [('boost',T.boost()),('extra_trees',ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=1.0,random_state=0,n_jobs=4))]:
 m.fit(T.G[tr],T.y[tr]);record(name,m,lambda ix:m.predict_proba(T.G[ix]))
for name,flip in [('cnn_no_flip',0.0),('cnn_flip',0.5)]:
 m=T.fit_net(tr,epochs=30,flip_prob=flip)
 print('clean train accuracy',name,float((T.predict_net(m,tr).argmax(1)==T.y[tr]).mean()),flush=True)
 torch.save({'state':m[0].state_dict(),'gmean':m[1],'gstd':m[2]},out/(name+'.pt'))
 record(name,m,lambda ix:T.predict_net(m,ix))
