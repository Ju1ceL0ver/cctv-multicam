"""After training, export original/visible/predicted masks; guesses are separate files."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from train import ROOT,Tiny
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'runs/completion/best.pt'));p.add_argument('--id',required=True);a=p.parse_args()
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);task=ck['config']['task'];m=Tiny(task,1 if ck['config'].get('input_mode')=='mask' else 4,ck['config'].get('spatial',False));m.load_state_dict(ck['model']);m.eval()
r=next(i for i in json.loads((ROOT/'data/manifest.json').read_text())['items'] if i['id']==a.id)
rgb=np.asarray(Image.open(ROOT/'data'/r['image']).convert('RGB').resize((96,160)),dtype=np.float32)/255
labels=json.loads((ROOT/'data/labels.json').read_text()) if (ROOT/'data/labels.json').exists() else {}
mask=np.asarray(Image.open(ROOT/'data'/labels.get(a.id,{}).get('mask_path',r['mask'])).convert('L').resize((96,160),Image.Resampling.NEAREST))>127
a=mask[None].astype(np.float32) if ck['config'].get('input_mode')=='mask' else np.concatenate([rgb,mask[:,:,None]],2).astype(np.float32).transpose(2,0,1)
x=torch.from_numpy(a)[None]
with torch.no_grad():prob=m(x).sigmoid().numpy()
if task=='visibility':print(json.dumps({'id':a.id,'occluded_probability':float(prob[0])}))
else:
 out=ROOT/'runs/completion/predictions'/a.id;out.mkdir(parents=True,exist_ok=True)
 Image.fromarray((prob[0,0]*255).astype(np.uint8)).save(out/'probability.png')
 Image.fromarray(((prob[0,0]>.5)|mask).astype(np.uint8)*255).save(out/'completed.png')
 Image.fromarray(mask.astype(np.uint8)*255).save(out/'observed.png')
 Image.fromarray(((prob[0,0]>.5)&~mask).astype(np.uint8)*255).save(out/'inferred_only.png')
 print(out)
