"""Evaluate frozen visibility checkpoint and benchmark synchronized MPS latency."""
import json,time,argparse
from pathlib import Path
import torch,numpy as np
from train import ROOT,Tiny,Crops
parser=argparse.ArgumentParser();parser.add_argument('--run-name',default='visibility');args=parser.parse_args();out=ROOT/'runs'/args.run_name;ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True);cfg=ck['config'];labels=json.loads((out/'labels_snapshot.json').read_text());items=json.loads((ROOT/'data/manifest.json').read_text())['items'];byid={r['id']:dict(r,label=labels[r['id']]) for r in items if r['id'] in labels};device='mps';m=Tiny('visibility',1 if cfg.get('input_mode')=='mask' else 4,cfg.get('spatial',False)).to(device);m.load_state_dict(ck['model']);m.eval()
def evaluate(ids):
 ds=Crops([byid[i] for i in ids],'visibility',False,input_mode=cfg.get('input_mode','rgb_mask'));truth=[];prob=[]
 with torch.no_grad():
  for x,y in torch.utils.data.DataLoader(ds,batch_size=8):truth.extend(y.numpy().tolist());prob.extend(m(x.to(device)).sigmoid().cpu().numpy().tolist())
 y=np.array(truth).astype(int);pred=(np.array(prob)>.5).astype(int);cm=np.array([[( (y==i)&(pred==j)).sum() for j in range(2)] for i in range(2)])
 return dict(n=len(y),confusion_matrix=cm.tolist(),accuracy=float((pred==y).mean()),balanced_accuracy=float(np.mean([cm[i,i]/cm[i].sum() for i in range(2)])),full_recall=float(cm[0,0]/cm[0].sum()),occluded_recall=float(cm[1,1]/cm[1].sum()),predictions=[dict(id=i,truth=int(t),occluded_probability=float(p)) for i,t,p in zip(ids,y,prob)])
train=evaluate(cfg['train_ids']);val=evaluate(cfg['val_ids']);ds=Crops([byid[cfg['val_ids'][0]]],'visibility',False,input_mode=cfg.get('input_mode','rgb_mask'));x=ds[0][0][None].to(device)
with torch.no_grad():
 for _ in range(15):m(x)
 torch.mps.synchronize();times=[]
 for _ in range(100):
  start=time.perf_counter();m(x);torch.mps.synchronize();times.append((time.perf_counter()-start)*1000)
report=dict(device=device,parameters=sum(p.numel() for p in m.parameters()),checkpoint_epoch=ck['report']['epoch'],train=train,validation=val,latency_ms_median=float(np.median(times)),latency_ms_p90=float(np.percentile(times,90)),latency_scope='batch 1 network only, input already on MPS, synchronized; excludes preprocessing',limitation='validation day used to select best epoch; no independent final test, identities across days unverified')
(out/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:({a:b for a,b in v.items() if a!='predictions'} if isinstance(v,dict) else v) for k,v in report.items()},indent=2))
