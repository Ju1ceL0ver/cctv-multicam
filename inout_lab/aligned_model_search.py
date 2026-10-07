"""Visible-mask bbox/crop alignment ablation; three classes and day exclusion."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from expert_mlp_search import Experts
from head_scale_mlp_search import ScaleExperts
from head_scale_search import mask_head_scale_features
from mask_density_features import density_features
from virtual_foot_search import virtual_features

ROOT=Path(__file__).parent

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['mlp','forest'],default='mlp');args=parser.parse_args()
    z=np.load(ROOT/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(ROOT/'diagnostics/original_labels.npy')!=0
    A=np.load(ROOT/'diagnostics/aligned_features.npy');G=np.load(ROOT/'diagnostics/aligned_geometry.npy');r=A[:,:234];h=A[:,234:936];c=A[:,936:1584];crop=A[:,1584:]
    cal=json.loads((ROOT/'diagnostics/calib_final_source.json').read_text())['cam1'];V=virtual_features(A[:,:1584],cal);S=mask_head_scale_features(h,G,cal);D=density_features(r,h,G)
    np.save(ROOT/'diagnostics/aligned_density_features.npy',D);parts=[np.column_stack([r[:,:39],h[:,:126],V]),np.column_stack([r[:,39:231],h[:,126:]]),np.column_stack([r[:,231:],c,crop])]
    names=['aligned','aligned_density'] if args.model=='mlp' else ['aligned_context','aligned_context_density']
    P={name:np.zeros((len(y),3)) for name in names};start=time.perf_counter();report={};device='mps' if torch.backends.mps.is_available() else 'cpu'
    with threadpool_limits(limits=2):
        for d in np.unique(days):
            tr=np.where(valid&(days!=d))[0];te=np.where(days==d)[0]
            for name in names:
                dense=name.endswith('density')
                if args.model=='forest':
                    B=np.column_stack([A[:,:1584],S,D]) if dense else A[:,:1584]
                    P[name][te]=ExtraTreesClassifier(n_estimators=160,max_features=.7,random_state=19,n_jobs=2).fit(B[tr],y[tr]).predict_proba(B[te])
                else:
                    data=[np.column_stack([parts[0],S,D]),parts[1],parts[2]] if dense else parts
                    torch.manual_seed(19);means=[p[tr].mean(0) for p in data];stds=[np.maximum(p[tr].std(0),.01) for p in data]
                    train=[torch.from_numpy(((p[tr]-m)/s).astype(np.float32)).to(device) for p,m,s in zip(data,means,stds)];test=[torch.from_numpy(((p[te]-m)/s).astype(np.float32)).to(device) for p,m,s in zip(data,means,stds)];target=torch.from_numpy(y[tr]).long().to(device)
                    model=(ScaleExperts if dense else Experts)([p.shape[1] for p in data]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=70)
                    weight=torch.tensor(np.sqrt(len(tr)/(3*np.bincount(y[tr],minlength=3))),dtype=torch.float32,device=device)
                    for epoch in range(70):
                        model.train()
                        for idx in torch.randperm(len(tr)).split(128):
                            out=model(*[p[idx] for p in train]);logits,side,door=out[:3];t=target[idx];prob=logits.softmax(1);defined=t<2
                            loss=F.cross_entropy(logits,t,weight=weight)+.25*F.cross_entropy(side[defined],t[defined])+.25*F.binary_cross_entropy_with_logits(door,(t==2).float(),pos_weight=torch.tensor(2.,device=device))
                            if dense:loss=loss+.25*F.cross_entropy(out[3],t,weight=weight)
                            opposite=prob[torch.arange(len(idx),device=device),1-t.clamp(max=1)];loss=loss+3*(opposite*defined).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                        scheduler.step()
                    model.eval()
                    with torch.inference_mode():P[name][te]=model(*test)[0].softmax(1).cpu().numpy()
                    del model,opt,train,test
                    if device=='mps':torch.mps.empty_cache()
                pred=P[name][te].argmax(1);v=valid[te];g=v&(y[te]<2)&(pred<2)&(pred!=y[te]);print(args.model,name,d,'wrong',int((v&(pred!=y[te])).sum()),'gross',te[g].tolist(),'seconds',round(time.perf_counter()-start),flush=True)
                np.savez(ROOT/f'diagnostics/{name}_{args.model}_predictions.npz',probabilities=P[name],latest_day=str(d))
    for name,prob in P.items():
        pred=prob.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);matrix=np.zeros((3,3),int)
        for a,b in zip(y[valid],pred[valid]):matrix[a,b]+=1
        report[name]={'defined':int(valid.sum()),'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int((valid&(pred!=y)).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean()),'confusion':matrix.tolist(),'hard_cases':{str(i):prob[i].tolist() for i in [483,625,789,970,1149]}}
    (ROOT/f'aligned_{args.model}_report.json').write_text(json.dumps(report,indent=2));print('ALIGNMENT RESULT',json.dumps(report),flush=True)
