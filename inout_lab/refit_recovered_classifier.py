"""All-data inference refit only. Accuracy comes exclusively from held-day report."""
from pathlib import Path
import time,json
import numpy as np,torch
from torch.nn import functional as F
from head_scale_mlp_search import ScaleExperts
from owner_boundary_mlp import make_parts
R=Path(__file__).parent
if __name__=='__main__':
 torch.set_num_threads(1);torch.manual_seed(19);z=np.load(R/'io_cam1.npz');v=np.load(R/'diagnostics/original_labels.npy')!=0;y=z['y'][v];parts=make_parts(R/'diagnostics/recovered_features.npy');means=[p[v].mean(0) for p in parts];stds=[np.maximum(p[v].std(0),.01) for p in parts];train=[torch.from_numpy(((p[v]-m)/s).astype('float32')) for p,m,s in zip(parts,means,stds)];target=torch.from_numpy(y).long();model=ScaleExperts([p.shape[1] for p in parts]);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=70);weight=torch.tensor(np.sqrt(len(y)/(3*np.bincount(y,minlength=3))),dtype=torch.float32);start=time.perf_counter()
 for epoch in range(70):
  model.train()
  for idx in torch.randperm(len(y)).split(128):
   logits,side,door,geo=model(*[p[idx] for p in train]);t=target[idx];defined=t<2;prob=logits.softmax(1);loss=F.cross_entropy(logits,t,weight=weight)+.25*F.cross_entropy(geo,t,weight=weight)+.25*F.cross_entropy(side[defined],t[defined])+.25*F.binary_cross_entropy_with_logits(door,(t==2).float(),pos_weight=torch.tensor(2.));opposite=prob[torch.arange(len(idx)),1-t.clamp(max=1)];loss+=3*(opposite*defined).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  sched.step()
 model.eval();out={'state_dict':model.state_dict(),'means':means,'stds':stds,'dims':[p.shape[1] for p in parts],'seed':19,'epochs':70,'training_days':np.unique(z['day'][v]).tolist(),'training_rows':int(v.sum()),'purpose':'all-data inference refit, not an independent test','held_day_report':'owner_boundary_mlp_recovered_report.json'};torch.save(out,R/'recovered_classifier.pt');(R/'recovered_classifier_refit.json').write_text(json.dumps({k:a for k,a in out.items() if k not in ['state_dict','means','stds']}|{'seconds':time.perf_counter()-start,'bytes':(R/'recovered_classifier.pt').stat().st_size},indent=2));print('refit complete',time.perf_counter()-start,flush=True)
