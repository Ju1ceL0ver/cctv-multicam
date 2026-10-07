"""Same lightweight CNN, but explicitly pools features of the selected person."""
import json
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
import train as T
out=Path(__file__).parent/'diagnostics';T.G=np.load(out/'mask_features.npy').astype('float32')
class FocusNet(T.Net):
 def __init__(self):
  super().__init__();self.fc1=nn.Linear(256*6+T.G.shape[1],256)
 def forward(self,f,g):
  x=self.blocks(f/127.5-1)
  mask=F.avg_pool2d(f[:,3:4]/255,8,8)
  def weighted(w):return (x*w).sum((2,3))/w.sum((2,3)).clamp(min=1e-4)
  yy=torch.arange(mask.shape[2],device=x.device,dtype=x.dtype).view(1,1,-1,1)
  center=(mask*yy).sum((2,3),keepdim=True)/mask.sum((2,3),keepdim=True).clamp(min=1e-4)
  lower=mask*(yy>=center)
  grid=F.adaptive_avg_pool2d(F.interpolate(x,size=(10,20),mode='bilinear',align_corners=False),(2,2)).flatten(1)
  a=torch.cat([grid,weighted(mask),weighted(lower),g],1)
  return self.fc3(F.relu(self.fc2(F.relu(self.fc1(a)))))
T.Net=FocusNet
tr=np.where(~np.isin(T.day,['20260918','20260919']))[0]
m=T.fit_net(tr,epochs=30,flip_prob=0,balanced=False)
r={};ps=dict(np.load(out/'predictions.npz'))
for split,d in [('development18','20260918'),('validation19','20260919')]:
 ix=np.where(T.day==d)[0];p=T.predict_net(m,ix);ps['cnn_focus_'+split]=p;r[split]={'accuracy':float((p.argmax(1)==T.y[ix]).mean()),'confusion':confusion_matrix(T.y[ix],p.argmax(1),labels=[0,1,2]).tolist()}
torch.save({'state':m[0].state_dict(),'gmean':m[1],'gstd':m[2]},out/'cnn_focus.pt');np.savez(out/'predictions.npz',**ps)
rep=json.loads((out/'report.json').read_text());rep['cnn_focus']=r;(out/'report.json').write_text(json.dumps(rep,indent=2));print(r,flush=True)
