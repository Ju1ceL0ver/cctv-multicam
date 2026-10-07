"""Three shared conv blocks for full scene AND detailed target crop on MPS.
3 FC layers256/64/3. All classes retained; held day excluded entirely.
No pretrained encoder and no dependency installation. Cache full image to avoid
8GB RAM paging; keep crop uint8, transfer minibatches only.
"""
from pathlib import Path
import argparse,json,time
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from spatial_cnn import cached_frames
root=Path(__file__).parent
class DualViewNet(nn.Module):
 def __init__(self):
  super().__init__();self.blocks=nn.Sequential(nn.Conv2d(7,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(2));self.head=nn.Sequential(nn.Linear(256*12+9,256),nn.ReLU(),nn.Dropout(.15),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,3))
 def encode(self,image):
  x=self.blocks(image/127.5-1);grid=F.adaptive_avg_pool2d(F.interpolate(x,size=(10,20),mode='bilinear',align_corners=False),(2,2)).flatten(1);mask=F.interpolate(image[:,3:4]/255.,size=x.shape[-2:],mode='area');pool=(x*mask).sum((2,3))/mask.sum((2,3)).clamp_min(.01);mass=mask.sum(3,keepdim=True);upper=(mass.cumsum(2)<=mass.sum(2,keepdim=True)*.3).to(mask.dtype)*mask;top=(x*upper).sum((2,3))/upper.sum((2,3)).clamp_min(.01);return torch.cat([grid,pool,top],1)
 def forward(self,frame,crop,geometry):
  # Joint batch means identical batch-normalization statistics for both views.
  both=self.encode(torch.cat([frame,crop],0));a,b=both.chunk(2);return self.head(torch.cat([a,b,geometry],1))

def run(day,epochs=60):
 z=np.load(root/'io_cam1.npz');frames=cached_frames(z);crops=z['X'];y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;g=z['G'][:,1:];tr=np.where(valid&(days!=day))[0];te=np.where(valid&(days==day))[0];gm=g[tr].mean(0);gs=np.maximum(g[tr].std(0),.01);G=torch.from_numpy(((g-gm)/gs).astype(np.float32));device='mps' if torch.backends.mps.is_available() else 'cpu';torch.manual_seed(19);model=DualViewNet().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs);weights=torch.tensor(np.sqrt(len(tr)/(3*np.bincount(y[tr],minlength=3))),dtype=torch.float32,device=device);start=time.perf_counter()
 def inputs(ix,augment=False):
  f=torch.from_numpy(np.array(frames[ix],copy=True)).to(device);c=torch.from_numpy(crops[ix].transpose(0,3,1,2).copy()).float().to(device);c=F.interpolate(c,size=(88,160),mode='bilinear',align_corners=False)
  if augment:
   # Fixed-camera location is meaningful: do not mirror the camera geometry.
   # Paired photometric augmentation preserves RGB/mask/position alignment.
   gain=torch.empty(len(ix),1,1,1,device=device).uniform_(.85,1.15);offset=torch.empty(len(ix),1,1,1,device=device).uniform_(-8,8)
   for image in [f,c]:image[:,:3]=(image[:,:3]*gain+offset+torch.randn_like(image[:,:3])*1.5).clamp(0,255)
  return f,c,G[ix].to(device)
 for epoch in range(epochs):
  model.train();correct=0;losses=[]
  for batch in torch.randperm(len(tr)).split(24):
   ix=tr[batch.numpy()];target=torch.from_numpy(y[ix]).long().to(device);logits=model(*inputs(ix,True));loss=F.cross_entropy(logits,target,weight=weights);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();correct+=int((logits.detach().argmax(1)==target).sum());losses.append(float(loss.detach()))
  schedule.step()
  (root/('diagnostics/dualview_'+day+'_progress.json')).write_text(json.dumps({'epoch':epoch+1,'epochs':epochs,'train_accuracy':correct/len(tr),'loss':float(np.mean(losses)),'seconds':time.perf_counter()-start,'held_day':day}))
  if epoch==0 or (epoch+1)%5==0:print('dualview',day,'epoch',epoch+1,'train',round(correct/len(tr),3),'loss',round(np.mean(losses),4),'seconds',round(time.perf_counter()-start),flush=True)
 model.eval();P=[]
 with torch.inference_mode():
  for ix in np.array_split(te,max(1,int(np.ceil(len(te)/24)))):P.append(model(*inputs(ix)).softmax(1).cpu().numpy())
 p=np.concatenate(P);pred=p.argmax(1);gross=(y[te]<2)&(pred<2)&(pred!=y[te]);report={'day':day,'epochs':epochs,'defined':len(te),'accuracy':float((pred==y[te]).mean()),'wrong':int((pred!=y[te]).sum()),'gross':int(gross.sum()),'gross_indices':te[gross].tolist(),'doorway_recall':float((pred[y[te]==2]==2).mean()),'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-start,'device':device};path=root/('diagnostics/dualview_'+day);torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'geometry_mean':gm,'geometry_std':gs,'held_day':day,'epochs':epochs},str(path)+'.pt');np.savez(str(path)+'_predictions.npz',indices=te,probabilities=p);Path(str(path)+'_report.json').write_text(json.dumps(report,indent=2));print('DUALVIEW',report,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--day',default='20260919');parser.add_argument('--epochs',type=int,default=60);a=parser.parse_args();run(a.day,a.epochs)
