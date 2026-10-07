import json
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import confusion_matrix
import train as T
out=Path(__file__).parent/'diagnostics'
T.G=np.load(out/'mask_features.npy').astype(np.float32)
tr=np.where(~np.isin(T.day,['20260918','20260919']))[0]
m=T.fit_net(tr,epochs=30,flip_prob=0,balanced=False)
report=json.loads((out/'report.json').read_text());saved=dict(np.load(out/'predictions.npz'))
name='cnn_mask_unweighted';report[name]={}
for split,d in [('development18','20260918'),('validation19','20260919')]:
 ix=np.where(T.day==d)[0];p=T.predict_net(m,ix);saved[name+'_'+split]=p
 report[name][split]={'accuracy':float((p.argmax(1)==T.y[ix]).mean()),'confusion':confusion_matrix(T.y[ix],p.argmax(1),labels=[0,1,2]).tolist()}
print(name,report[name],flush=True)
np.savez(out/'predictions.npz',**saved);(out/'report.json').write_text(json.dumps(report,indent=2))
torch.save({'state':m[0].state_dict(),'gmean':m[1],'gstd':m[2]},out/(name+'.pt'))
