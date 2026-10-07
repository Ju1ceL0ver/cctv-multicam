"""Small geometry / shape / RGB experts; auxiliary tasks + confusion penalty.
Complete day exclusion; all three target classes evaluated, no abstention.
"""
from pathlib import Path
import numpy as np,json,time
import torch
from torch import nn
from torch.nn import functional as F
root=Path(__file__).parent
class Experts(nn.Module):
 def __init__(self,dims):
  super().__init__();self.branches=nn.ModuleList([nn.Sequential(nn.Linear(n,128),nn.ReLU(),nn.Linear(128,64),nn.LayerNorm(64),nn.ReLU()) for n in dims]);self.fusion=nn.Sequential(nn.Linear(192,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU());self.classifier=nn.Linear(64,3);self.side=nn.Linear(64,2);self.door=nn.Linear(64,1)
 def forward(self,*parts):
  x=self.fusion(torch.cat([m(p) for m,p in zip(self.branches,parts)],1));return self.classifier(x),self.side(x),self.door(x).squeeze(1)
if __name__=='__main__':
 z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;r=np.load(root/'diagnostics/rich_features.npy');h=np.load(root/'diagnostics/head_features.npy');c=np.load(root/'diagnostics/context_features.npy');crop=np.load(root/'diagnostics/crop_rgb_features.npy');virtual=np.load(root/'diagnostics/virtual_foot_features.npy');parts=[np.column_stack([r[:,:39],h[:,:126],virtual]),np.column_stack([r[:,39:231],h[:,126:]]),np.column_stack([r[:,231:],c,crop])];device='mps' if torch.backends.mps.is_available() else 'cpu';report={};start=time.perf_counter()
 for alpha in [0.,3.]:
  P=np.zeros((len(y),3))
  for d in np.unique(days):
   tr=np.where((days!=d)&valid)[0];te=np.where(days==d)[0];torch.manual_seed(19);gm=[p[tr].mean(0) for p in parts];gs=[np.maximum(p[tr].std(0),.01) for p in parts];train=[torch.from_numpy(((p[tr]-m)/s).astype(np.float32)).to(device) for p,m,s in zip(parts,gm,gs)];test=[torch.from_numpy(((p[te]-m)/s).astype(np.float32)).to(device) for p,m,s in zip(parts,gm,gs)];target=torch.from_numpy(y[tr]).long().to(device);model=Experts([p.shape[1] for p in parts]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=70);w=torch.tensor(np.sqrt(len(tr)/(3*np.bincount(y[tr],minlength=3))),dtype=torch.float32,device=device)
   for epoch in range(70):
    model.train()
    for idx in torch.randperm(len(tr)).split(128):
     inputs=[p[idx] for p in train];logit,side,door=model(*inputs);t=target[idx];prob=logit.softmax(1);defined=t<2;loss=F.cross_entropy(logit,t,weight=w)+.25*F.cross_entropy(side[defined],t[defined])+.25*F.binary_cross_entropy_with_logits(door,(t==2).float(),pos_weight=torch.tensor(2.,device=device))
     bad=prob[torch.arange(len(idx),device=device),1-t.clamp(max=1)];loss=loss+alpha*(bad*defined).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    scheduler.step()
   model.eval()
   with torch.inference_mode():P[te]=model(*test)[0].softmax(1).cpu().numpy()
   print('experts',alpha,d,'elapsed',round(time.perf_counter()-start),flush=True);del model,opt,train,test
   if device=='mps':torch.mps.empty_cache()
  pred=P.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);report[str(alpha)]={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int(((pred!=y)&valid).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist()};np.savez(root/('diagnostics/expert_mlp_'+str(alpha)+'.npz'),probabilities=P);print('EXPERTS',alpha,report[str(alpha)],flush=True)
 (root/'expert_mlp_report.json').write_text(json.dumps(report,indent=2))
