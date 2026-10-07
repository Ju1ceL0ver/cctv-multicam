"""Choose a small ensemble/doorway bias on development18 only."""
import json,numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
root=Path(__file__).parent;out=root/'diagnostics';z=np.load(root/'io_cam1.npz');y=z['y'];day=z['day'];rich=np.load(out/'rich_predictions.npz');cnn=np.load(out/'predictions.npz');base='extra_leaf1';dev=day=='20260918';val=day=='20260919'
best=None
for source in ['cnn_no_flip','cnn_mask_unweighted','mask_extra_trees']:
 for alpha in np.linspace(0,1,11):
  p=(1-alpha)*rich[base+'_development18']+alpha*cnn[source+'_development18']
  for door_bias in [.6,.8,1.,1.2,1.4,1.6,1.8,2.]:
   weighted=p*np.array([1.,1.,door_bias]);acc=float((weighted.argmax(1)==y[dev]).mean())
   if best is None or acc>best['development18']['accuracy']:
    best={'source':source,'alpha':float(alpha),'door_bias':door_bias,'development18':{'accuracy':acc,'confusion':confusion_matrix(y[dev],weighted.argmax(1),labels=[0,1,2]).tolist()}}
p=(1-best['alpha'])*rich[base+'_validation19']+best['alpha']*cnn[best['source']+'_validation19'];p*=np.array([1,1,best['door_bias']]);pred=p.argmax(1)
best['validation19']={'accuracy':float((pred==y[val]).mean()),'confusion':confusion_matrix(y[val],pred,labels=[0,1,2]).tolist()}
(out/'mix_report.json').write_text(json.dumps(best,indent=2));print(best)
