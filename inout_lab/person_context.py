"""Object-centric shared CNN: local person crop + full-frame context.
No floor mask/signed floor distance; those represent visible free floor, not
shop membership under furniture occlusion. One held-out day, never fitted.
"""
from pathlib import Path
import argparse,json,time
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
import train as T
class PersonContext(nn.Module):
 def __init__(self):
  super().__init__()
  self.body=nn.Sequential(nn.Conv2d(4,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(128,256,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
  self.fc1=nn.Linear(256*8+8,256);self.fc2=nn.Linear(256,64);self.fc3=nn.Linear(64,3);self.binary=nn.Linear(64,1);self.drop=nn.Dropout(.2)
 def branch(self,x):
  x=self.body(x/127.5-1);x=F.interpolate(x,size=(10,12),mode='bilinear',align_corners=False);return F.avg_pool2d(x,(5,6)).flatten(1)
 def forward(self,frame,crop,g):
  x=torch.cat([self.branch(frame),self.branch(crop),g],1);x=self.drop(F.relu(self.fc1(x)));x=F.relu(self.fc2(x));return self.fc3(x),self.binary(x).squeeze(1)
def main(day='20260918',epochs=25):
 torch.manual_seed(1);root=Path(__file__).parent;out=root/'diagnostics';original=np.load(out/'original_labels.npy');tr=np.where((T.day!=day)&(original!=0))[0];te=np.where((T.day==day)&(original!=0))[0]
 frame=T.Fr[:,:4].to(T.DEV);crop=F.interpolate(torch.from_numpy(T.X[:,:,:,:4].transpose(0,3,1,2).copy()).float(),size=(80,48),mode='bilinear',align_corners=False).to(T.DEV)
 raw=T.G[:,:8];gm,gs=raw[tr].mean(0),np.maximum(raw[tr].std(0),.01);g=torch.tensor((raw-gm)/gs,device=T.DEV);y=torch.tensor(T.y,device=T.DEV)
 m=PersonContext().to(T.DEV);opt=torch.optim.AdamW(m.parameters(),lr=.0007,weight_decay=.001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00005)
 counts=np.bincount(T.y[tr],minlength=3);w=np.sqrt(len(tr)/(3*counts));w=torch.tensor(w/w.mean(),dtype=torch.float32,device=T.DEV)
 print('held day',day,'train',len(tr),'test',len(te),'params',sum(p.numel() for p in m.parameters()),flush=True);start=time.time()
 for epoch in range(epochs):
  m.train();losses=[];correct=0
  for b in torch.tensor(tr)[torch.randperm(len(tr))].split(64):
   f,c=frame[b].clone(),crop[b].clone();gain=torch.empty(len(b),1,1,1,device=T.DEV).uniform_(.9,1.1)
   f[:,:3]=(f[:,:3]*gain).clamp(0,255);c[:,:3]=(c[:,:3]*gain).clamp(0,255)
   logits,binary=m(f,c,g[b]);loss=F.cross_entropy(logits,y[b],weight=w);defined=y[b]!=2
   if defined.any():loss=loss+.25*F.binary_cross_entropy_with_logits(binary[defined],y[b][defined].float())
   opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(loss.detach());correct+=(logits.argmax(1)==y[b]).sum().item()
  scheduler.step();print(epoch+1,torch.stack(losses).mean().item(),correct/len(tr),round(time.time()-start,1),flush=True)
 m.eval();ps=[];bs=[]
 with torch.inference_mode():
  for b in np.array_split(te,max(1,int(np.ceil(len(te)/64)))):
   logits,binary=m(frame[b],crop[b],g[b]);ps.append(logits.softmax(1).cpu().numpy());bs.append(binary.sigmoid().cpu().numpy())
 p=np.concatenate(ps);binary=np.concatenate(bs);pred=p.argmax(1);truth=T.y[te];r={'day':day,'epochs':epochs,'train_n':len(tr),'test_n':len(te),'accuracy':float((pred==truth).mean()),'confusion':confusion_matrix(truth,pred,labels=[0,1,2]).tolist(),'direct_cross_side':int(((truth<2)&(pred<2)&(pred!=truth)).sum()),'binary_accuracy_on_defined_sides':float(((binary[truth<2]>.5)==truth[truth<2]).mean())}
 print('RESULT',r,flush=True);(out/('person_context_'+day+'.json')).write_text(json.dumps(r,indent=2));np.savez(out/('person_context_'+day+'.npz'),indices=te,probabilities=p,binary=binary);torch.save({'state':m.state_dict(),'gmean':gm,'gstd':gs},out/('person_context_'+day+'.pt'))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--day',default='20260918');parser.add_argument('--epochs',type=int,default=25);args=parser.parse_args();main(args.day,args.epochs)
