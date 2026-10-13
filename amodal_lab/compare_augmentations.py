"""Compare both models on exactly the same held-out masks and 13 fixed corruption families."""
import json,hashlib,argparse
import numpy as np,torch
from PIL import Image
from train import ROOT,Tiny
from augment import KINDS,rich_occlude
parser=argparse.ArgumentParser();parser.add_argument('--run-name',default='completion_rich');parser.add_argument('--baseline',default='completion');args=parser.parse_args();out=ROOT/'runs'/args.run_name;base=torch.load(ROOT/'runs'/args.baseline/'best.pt',map_location='cpu',weights_only=True);new=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
assert base['config']['train_ids']==new['config']['train_ids'];assert base['config']['val_ids']==new['config']['val_ids'];assert base['config']['mask_sources']==new['config']['mask_sources']
manifest={r['id']:r for r in json.loads((ROOT/'data/manifest.json').read_text())['items']};models={}
for name,ck in [('legacy',base),('rich',new)]:
 m=Tiny('completion',1 if ck['config'].get('input_mode')=='mask' else 4).to('mps');m.load_state_dict(ck['model']);m.eval();models[name]=(m,ck['config'].get('input_mode','rgb_mask'))
results={}
with torch.no_grad():
 for kind in KINDS:
  scores={name:[0,0] for name in models}
  for ident in base['config']['val_ids']:
   row=manifest[ident];rgb=np.asarray(Image.open(ROOT/'data'/row['image']).convert('RGB').resize((96,160)),np.float32)/255
   path=base['config']['mask_sources'][ident]['path'];mask=(np.asarray(Image.open(ROOT/'data'/path).convert('L').resize((96,160),Image.Resampling.NEAREST))>127).astype(np.float32)
   seed=int(hashlib.sha256((ident+kind).encode()).hexdigest()[:8],16);rgb,v,o=rich_occlude(rgb,mask,np.random.default_rng(seed),kind);x=torch.from_numpy(np.concatenate([rgb,v[:,:,None]],2).transpose(2,0,1))[None].to('mps')
   for name,(m,mode) in models.items():
    model_input=torch.from_numpy(v[None,None]).to('mps') if mode=='mask' else x
    pred=(m(model_input).sigmoid().cpu().numpy()[0,0]>.5)|(v>0);truth=mask>0;scores[name][0]+=(pred&truth&o).sum();scores[name][1]+=((pred|truth)&o).sum()
  results[kind]={name:float(a/max(1,b)) for name,(a,b) in scores.items()}
report=dict(per_type_hidden_IoU=results,macro_hidden_IoU={name:float(np.mean([r[name] for r in results.values()])) for name in models},examples_per_type=len(base['config']['val_ids']),limitation='same validation day used for epoch selection; synthetic only, no final blind test; stand/bottom and top/top_edge overlap in geometry')
(out/'augmentation_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
