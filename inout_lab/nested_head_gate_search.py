"""Nested day-group cross-fitting for RGB/shape MLP vs mask-head geometry.
Outer test day never appears in experts or gate training, even indirectly.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from expert_mlp_search import Experts

ROOT=Path(__file__).parent

def neural_predict(parts,y,tr,te):
    torch.manual_seed(19);device='mps' if torch.backends.mps.is_available() else 'cpu'
    means=[p[tr].mean(0) for p in parts];stds=[np.maximum(p[tr].std(0),.01) for p in parts]
    train=[torch.from_numpy(((p[tr]-m)/s).astype(np.float32)).to(device) for p,m,s in zip(parts,means,stds)]
    test=[torch.from_numpy(((p[te]-m)/s).astype(np.float32)).to(device) for p,m,s in zip(parts,means,stds)]
    target=torch.from_numpy(y[tr]).long().to(device);model=Experts([p.shape[1] for p in parts]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=70)
    w=torch.tensor(np.sqrt(len(tr)/(3*np.bincount(y[tr],minlength=3))),dtype=torch.float32,device=device)
    for epoch in range(70):
        model.train()
        for idx in torch.randperm(len(tr)).split(128):
            logits,side,door=model(*[p[idx] for p in train]);t=target[idx];prob=logits.softmax(1);defined=t<2
            loss=F.cross_entropy(logits,t,weight=w)+.25*F.cross_entropy(side[defined],t[defined])+.25*F.binary_cross_entropy_with_logits(door,(t==2).float(),pos_weight=torch.tensor(2.,device=device))
            opposite=prob[torch.arange(len(idx),device=device),1-t.clamp(max=1)];loss=loss+3*(opposite*defined).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        scheduler.step()
    model.eval()
    with torch.inference_mode():P=model(*test)[0].softmax(1).cpu().numpy()
    del model,opt,train,test
    if device=='mps':torch.mps.empty_cache()
    return P

def geometry_model():
    return HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19)

def gate_features(M,H,reliability):
    entropy=lambda p:-(p*np.log(np.maximum(p,1e-8))).sum(1,keepdims=True)
    return np.column_stack([M,H,M-H,entropy(M),entropy(H),(M.argmax(1)==H.argmax(1)).astype(float),reliability])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--day',default='20260919');args=parser.parse_args()
    z=np.load(ROOT/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(ROOT/'diagnostics/original_labels.npy')!=0
    r=np.load(ROOT/'diagnostics/rich_features.npy');h=np.load(ROOT/'diagnostics/head_features.npy');c=np.load(ROOT/'diagnostics/context_features.npy');crop=np.load(ROOT/'diagnostics/crop_rgb_features.npy');v=np.load(ROOT/'diagnostics/virtual_foot_features.npy');S=np.load(ROOT/'diagnostics/head_scale_features.npy')[:,11:107]
    parts=[np.column_stack([r[:,:39],h[:,:126],v]),np.column_stack([r[:,39:231],h[:,126:]]),np.column_stack([r[:,231:],c,crop])]
    geometry=np.column_stack([z['G'][:,1:],S]);reliability=np.column_stack([r[:,:39],h[:,:126],S]);start=time.perf_counter()
    for day in (np.unique(days) if args.day=='all' else [args.day]):
        tr=np.where(valid&(days!=day))[0];te=np.where(days==day)[0]
        innerM=np.zeros((len(tr),3));innerH=np.zeros((len(tr),3));audit=[]
        with threadpool_limits(limits=2):
            for fold,(itr,ival) in enumerate(GroupKFold(n_splits=3).split(tr,groups=days[tr])):
                train_idx=tr[itr];val_idx=tr[ival]
                assert not np.any(days[train_idx]==day) and not np.any(days[val_idx]==day)
                assert not set(days[train_idx])&set(days[val_idx])
                innerM[ival]=neural_predict(parts,y,train_idx,val_idx)
                innerH[ival]=geometry_model().fit(geometry[train_idx],y[train_idx]).predict_proba(geometry[val_idx])
                audit.append({'train_days':np.unique(days[train_idx]).tolist(),'val_days':np.unique(days[val_idx]).tolist()})
                print('nested',day,'inner',fold+1,'seconds',round(time.perf_counter()-start),flush=True)
            testM=neural_predict(parts,y,tr,te);testH=geometry_model().fit(geometry[tr],y[tr]).predict_proba(geometry[te])
            trainA=gate_features(innerM,innerH,reliability[tr]);testA=gate_features(testM,testH,reliability[te])
            models={'boost':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=15,l2_regularization=5,random_state=19),'forest':ExtraTreesClassifier(n_estimators=160,min_samples_leaf=3,max_features=.7,random_state=19,n_jobs=2)}
            report={'held_day':str(day),'inner_audit':audit,'experts':{},'gates':{}}
            for name,P in [('visual',testM),('head_geometry',testH)]:
                p=P.argmax(1);vv=valid[te];g=vv&(y[te]<2)&(p<2)&(p!=y[te]);report['experts'][name]={'wrong':int((vv&(p!=y[te])).sum()),'gross':int(g.sum()),'gross_indices':te[g].tolist()}
            for name,model in models.items():
                P=model.fit(trainA,y[tr]).predict_proba(testA);p=P.argmax(1);vv=valid[te];g=vv&(y[te]<2)&(p<2)&(p!=y[te]);report['gates'][name]={'accuracy':float((p[vv]==y[te][vv]).mean()),'wrong':int((vv&(p!=y[te])).sum()),'gross':int(g.sum()),'gross_indices':te[g].tolist(),'hard_cases':{str(i):P[np.where(te==i)[0][0]].tolist() for i in [970,1149] if i in te}}
                np.savez(ROOT/f'diagnostics/nested_head_{day}_{name}.npz',indices=te,probabilities=P)
        report['seconds']=time.perf_counter()-start;(ROOT/f'nested_head_{day}_report.json').write_text(json.dumps(report,indent=2));print('NESTED RESULT',json.dumps(report),flush=True)
