"""Tiny jointly trained experts with cheap mask-only angular-size features.
No pose/depth inference needed by this experiment. Complete held-day exclusion.
"""
from pathlib import Path
import json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from expert_mlp_search import Experts

ROOT=Path(__file__).parent

class ScaleExperts(Experts):
    def __init__(self,dims):
        super().__init__(dims)
        self.geometry_class=nn.Linear(64,3)
    def forward(self,*parts):
        embeddings=[m(p) for m,p in zip(self.branches,parts)]
        x=self.fusion(torch.cat(embeddings,1))
        return self.classifier(x),self.side(x),self.door(x).squeeze(1),self.geometry_class(embeddings[0])

if __name__=='__main__':
    z=np.load(ROOT/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(ROOT/'diagnostics/original_labels.npy')!=0
    r=np.load(ROOT/'diagnostics/rich_features.npy');h=np.load(ROOT/'diagnostics/head_features.npy');c=np.load(ROOT/'diagnostics/context_features.npy');crop=np.load(ROOT/'diagnostics/crop_rgb_features.npy');virtual=np.load(ROOT/'diagnostics/virtual_foot_features.npy');scale=np.load(ROOT/'diagnostics/head_scale_features.npy')[:,11:107]
    parts=[np.column_stack([r[:,:39],h[:,:126],virtual,scale]),np.column_stack([r[:,39:231],h[:,126:]]),np.column_stack([r[:,231:],c,crop])]
    device='mps' if torch.backends.mps.is_available() else 'cpu';P=np.zeros((len(y),3));start=time.perf_counter();folds={}
    for d in np.unique(days):
        tr=np.where(valid&(days!=d))[0];te=np.where(days==d)[0];torch.manual_seed(19)
        means=[p[tr].mean(0) for p in parts];stds=[np.maximum(p[tr].std(0),.01) for p in parts]
        train=[torch.from_numpy(((p[tr]-m)/s).astype(np.float32)).to(device) for p,m,s in zip(parts,means,stds)]
        test=[torch.from_numpy(((p[te]-m)/s).astype(np.float32)).to(device) for p,m,s in zip(parts,means,stds)]
        target=torch.from_numpy(y[tr]).long().to(device);model=ScaleExperts([p.shape[1] for p in parts]).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=70)
        weight=torch.tensor(np.sqrt(len(tr)/(3*np.bincount(y[tr],minlength=3))),dtype=torch.float32,device=device)
        for epoch in range(70):
            model.train()
            for idx in torch.randperm(len(tr)).split(128):
                logits,side,door,geometry_logits=model(*[p[idx] for p in train]);t=target[idx];prob=logits.softmax(1);defined=t<2
                loss=F.cross_entropy(logits,t,weight=weight)+.25*F.cross_entropy(geometry_logits,t,weight=weight)+.25*F.cross_entropy(side[defined],t[defined])+.25*F.binary_cross_entropy_with_logits(door,(t==2).float(),pos_weight=torch.tensor(2.,device=device))
                opposite=prob[torch.arange(len(idx),device=device),1-t.clamp(max=1)]
                loss=loss+3*(opposite*defined).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            schedule.step()
        model.eval()
        with torch.inference_mode():P[te]=model(*test)[0].softmax(1).cpu().numpy()
        pred=P[te].argmax(1);v=valid[te];gross=v&(y[te]<2)&(pred<2)&(pred!=y[te])
        folds[str(d)]={'wrong':int((v&(pred!=y[te])).sum()),'gross':int(gross.sum()),'gross_indices':te[gross].tolist()}
        np.savez(ROOT/'diagnostics/head_scale_mlp_predictions.npz',probabilities=P,completed_days=np.array(list(folds)))
        print('scale MLP',d,folds[str(d)],'seconds',round(time.perf_counter()-start),flush=True)
        del model,opt,train,test
        if device=='mps':torch.mps.empty_cache()
    pred=P.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y)
    report={'defined':int(valid.sum()),'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int((valid&(pred!=y)).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean()),'folds':folds,'seconds':time.perf_counter()-start,'pose_inference_required':False,'depth_inference_required':False}
    (ROOT/'head_scale_mlp_report.json').write_text(json.dumps(report,indent=2));print('SCALE MLP RESULT',json.dumps(report),flush=True)
