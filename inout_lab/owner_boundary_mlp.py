"""Small mask/geometry/RGB experts with owner's threshold and two bbox views.
Every outer held day is excluded from feature normalization and model training.
No CNN resume, no depth/pose at inference, no changes to production.
"""
from pathlib import Path
import argparse,json,time
import numpy as np,torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from head_scale_mlp_search import ScaleExperts
from head_scale_search import mask_head_scale_features
from mask_density_features import density_features
from owner_boundary_search import boundary_features,metrics
ROOT=Path(__file__).parent

def make_parts(features_path=None,body_features_path=None,pose_features_path=None):
    z=np.load(ROOT/'io_cam1.npz');A=np.load(features_path or ROOT/'diagnostics/aligned_features.npy');g=np.load(ROOT/'diagnostics/aligned_geometry.npy');r=A[:,:234];h=A[:,234:936];c=A[:,936:1584];crop=A[:,1584:]
    E=boundary_features(h,g);cal=json.loads((ROOT/'diagnostics/calib_final_source.json').read_text())['cam1'];S=mask_head_scale_features(h,g,cal);D=density_features(r,h,g) if features_path else np.load(ROOT/'diagnostics/aligned_density_features.npy')
    # Original metadata is retained as evidence of bbox/mask disagreement, not truth.
    original=z['G'][:,1:];geo=np.column_stack([r[:,:39],h[:,:126],S,D,E,original,original-g])
    if pose_features_path:geo=np.column_stack([geo,np.load(pose_features_path)])
    appearance=np.load(body_features_path) if body_features_path else crop
    return [geo,np.column_stack([r[:,39:231],h[:,126:]]),np.column_stack([r[:,231:],c,appearance])]

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[19,71]);ap.add_argument('--epochs',type=int,default=70);ap.add_argument('--features-source');ap.add_argument('--body-features');ap.add_argument('--pose-features');ap.add_argument('--tag',default='');args=ap.parse_args();suffix=('_'+args.tag) if args.tag else ''
    torch.set_num_threads(2);start=time.perf_counter();z=np.load(ROOT/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(ROOT/'diagnostics/original_labels.npy')!=0;parts=make_parts(args.features_source,args.body_features,args.pose_features);device='mps' if torch.backends.mps.is_available() else 'cpu';P={s:np.full((len(y),3),np.nan) for s in args.seeds};folds=[]
    with threadpool_limits(limits=2):
        for d in np.unique(days):
            tr=np.flatnonzero(valid&(days!=d));te=np.flatnonzero(days==d);assert not set(days[tr])&set(days[te])
            means=[p[tr].mean(0) for p in parts];stds=[np.maximum(p[tr].std(0),.01) for p in parts];train=[torch.from_numpy(((p[tr]-m)/s).astype('float32')).to(device) for p,m,s in zip(parts,means,stds)];test=[torch.from_numpy(((p[te]-m)/s).astype('float32')).to(device) for p,m,s in zip(parts,means,stds)];target=torch.from_numpy(y[tr]).long().to(device)
            for seed in args.seeds:
                torch.manual_seed(seed);model=ScaleExperts([p.shape[1] for p in parts]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs);weight=torch.tensor(np.sqrt(len(tr)/(3*np.bincount(y[tr],minlength=3))),dtype=torch.float32,device=device)
                for epoch in range(args.epochs):
                    model.train()
                    for idx in torch.randperm(len(tr)).split(128):
                        logits,side,door,geo=model(*[p[idx] for p in train]);t=target[idx];defined=t<2;prob=logits.softmax(1)
                        loss=F.cross_entropy(logits,t,weight=weight)+.25*F.cross_entropy(geo,t,weight=weight)+.25*F.cross_entropy(side[defined],t[defined])+.25*F.binary_cross_entropy_with_logits(door,(t==2).float(),pos_weight=torch.tensor(2.,device=device))
                        opposite=prob[torch.arange(len(idx),device=device),1-t.clamp(max=1)];loss+=3*(opposite*defined).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                    schedule.step()
                model.eval()
                with torch.inference_mode():P[seed][te]=model(*test)[0].softmax(1).cpu().numpy()
                path=ROOT/f'diagnostics/owner_boundary_mlp{suffix}_{seed}_{d}.pt';torch.save({'state_dict':{k:t.cpu() for k,t in model.state_dict().items()},'means':means,'stds':stds,'dims':[p.shape[1] for p in parts],'excluded_day':str(d),'training_days':np.unique(days[tr]).tolist(),'epochs':args.epochs,'seed':seed,'features_source':args.features_source,'body_features_source':args.body_features,'pose_features_source':args.pose_features},path)
                pp=P[seed][te].argmax(1);v=valid[te];gr=v&(y[te]<2)&(pp<2)&(y[te]!=pp)
                print('boundary mlp',d,'seed',seed,'wrong',int((v&(pp!=y[te])).sum()),'gross',te[gr].tolist(),'elapsed',round(time.perf_counter()-start),flush=True)
                del model,opt
            folds.append(str(d))
            for seed in args.seeds:np.savez(ROOT/f'diagnostics/owner_boundary_mlp{suffix}_{seed}.npz',probabilities=P[seed],completed_days=np.array(folds))
            (ROOT/f'owner_boundary_mlp{suffix}_progress.json').write_text(json.dumps({'completed_days':folds,'seeds':args.seeds,'seconds':time.perf_counter()-start},indent=2))
            del train,test
            if device=='mps':torch.mps.empty_cache()
    report={str(s):metrics(p,y,valid,days) for s,p in P.items()};avg=np.mean(list(P.values()),axis=0);report['equal_seeds']=metrics(avg,y,valid,days);np.savez(ROOT/f'diagnostics/owner_boundary_mlp{suffix}_equal.npz',probabilities=avg,completed_days=np.unique(days))
    (ROOT/f'owner_boundary_mlp{suffix}_report.json').write_text(json.dumps({'models':report,'seconds':time.perf_counter()-start,'parameters_per_model':sum(t.numel() for t in ScaleExperts([p.shape[1] for p in parts]).parameters()),'protocol':'All 9 whole days held out; diagnostic, not fresh blind test'},indent=2));print('FINAL',json.dumps(report),flush=True)
