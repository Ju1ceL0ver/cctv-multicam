"""Frozen synthetic completion validation, synchronized MPS timing and contact sheet."""
import json,time,argparse
import numpy as np
import torch
from PIL import Image,ImageDraw
from train import ROOT,Tiny,Crops
parser=argparse.ArgumentParser();parser.add_argument('--run-name',default='completion');args=parser.parse_args();out=ROOT/'runs'/args.run_name;ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True);cfg=ck['config'];labels=json.loads((out/'labels_snapshot.json').read_text());items=json.loads((ROOT/'data/manifest.json').read_text())['items'];byid={r['id']:dict(r,label=labels[r['id']]) for r in items if r['id'] in labels};ds=Crops([byid[i] for i in cfg['val_ids']],'completion',False,input_mode=cfg.get('input_mode','rgb_mask'));m=Tiny('completion',1 if cfg.get('input_mode')=='mask' else 4).to('mps');m.load_state_dict(ck['model']);m.eval();records=[];examples=[];si=su=fi=fu=bi=bu=0
with torch.no_grad():
 for i in range(len(ds)):
  x,y,occ,v=ds[i];prob=m(x[None].to('mps')).sigmoid().cpu()[0];pred=(prob>.5)|v.bool();truth=y.bool();o=occ.bool();inter=(pred&truth&o).sum().item();union=((pred|truth)&o).sum().item();full_i=(pred&truth).sum().item();full_u=(pred|truth).sum().item();si+=inter;su+=union;fi+=full_i;fu+=full_u;bi+=(v.bool()&truth).sum().item();bu+=(v.bool()|truth).sum().item()
  records.append(dict(id=cfg['val_ids'][i],hidden_IoU=inter/max(1,union),full_IoU=full_i/max(1,full_u)))
  rgb=(x[:3].permute(1,2,0).numpy()*255).astype('uint8');rgb=np.repeat(rgb,3,axis=2) if rgb.shape[2]==1 else rgb;examples.append((rgb,v[0].numpy(),pred[0].numpy(),y[0].numpy()))
 x=ds[0][0][None].to('mps')
 for _ in range(15):m(x)
 torch.mps.synchronize();times=[]
 for _ in range(100):
  t=time.perf_counter();m(x);torch.mps.synchronize();times.append((time.perf_counter()-t)*1000)
report=dict(device='mps',parameters=sum(p.numel() for p in m.parameters()),train=len(cfg['train_ids']),validation=len(ds),best_epoch=ck['report']['epoch'],hidden_region_IoU=si/max(1,su),full_mask_IoU=fi/max(1,fu),visible_only_full_IoU=bi/max(1,bu),visible_only_hidden_IoU=0,latency_ms_median=float(np.median(times)),latency_ms_p90=float(np.percentile(times,90)),latency_scope='network only, batch1 input on MPS, synchronized',limitation='synthetic occlusions on approved full masks; day used to select checkpoint, no real hidden ground truth; cross-day identities unverified',examples=records)
(out/'evaluation.json').write_text(json.dumps(report,indent=2))
# All validation examples, no cherry picking. Columns: occluded RGB, visible mask, completed, true full mask.
sheet=Image.new('RGB',(384,len(examples)*184+25),'#172033');draw=ImageDraw.Draw(sheet)
for j,t in enumerate(['Occluded RGB','Visible','Completed','Full target']):draw.text((j*96+3,5),t,fill='white')
for i,(rgb,v,p,y) in enumerate(examples):
 for j,a in enumerate([rgb,(v*255).astype('uint8'),(p*255).astype('uint8'),(y*255).astype('uint8')]):sheet.paste(Image.fromarray(a).convert('RGB'),(j*96,25+i*184))
 draw.text((3,185+i*184),f'{i+1} hidden IoU {records[i]["hidden_IoU"]:.2f}',fill='white')
sheet.save(out/'validation_sheet.jpg');print(json.dumps({k:v for k,v in report.items() if k!='examples'},indent=2))
