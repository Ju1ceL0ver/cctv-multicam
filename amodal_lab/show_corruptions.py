"""Show actual rich_occlude masks, no RGB, fixed random examples."""
import json
import numpy as np
from PIL import Image,ImageDraw
from train import ROOT
from augment import KINDS,rich_occlude
labels=json.loads((ROOT/'runs/completion/labels_snapshot.json').read_text());items=json.loads((ROOT/'data/manifest.json').read_text())['items'];rows=[r for r in items if labels.get(r['id'],{}).get('visibility')=='full' and labels[r['id']].get('quality')=='good']
r=rows[0];mask=(np.asarray(Image.open(ROOT/'data'/labels[r['id']].get('mask_path',r['mask'])).convert('L').resize((96,160),Image.Resampling.NEAREST))>127).astype(np.float32)
cols=4;cw=304;rh=215;sheet=Image.new('RGB',(cols*cw,4*rh+45),'#172033');draw=ImageDraw.Draw(sheet);draw.text((12,10),'CURRENT TRAINING: full target | removed area (red) | visible input',fill='white')
for i,kind in enumerate(KINDS):
 _,v,o=rich_occlude(np.zeros((160,96,3),np.float32),mask,np.random.default_rng(100+i),kind)
 removed=np.zeros((160,96,3),np.uint8);removed[mask>0]=[65,65,65];removed[(mask>0)&o]=[255,75,65]
 x=(i%cols)*cw;y=45+(i//cols)*rh;draw.text((x+5,y),kind,fill='white')
 for j,a in enumerate([(mask*255).astype(np.uint8),removed,(v*255).astype(np.uint8)]):sheet.paste(Image.fromarray(a).convert('RGB'),(x+j*100+3,y+23))
 draw.text((x+5,y+186),f'removed {(mask-v).sum()/mask.sum():.0%}',fill='white')
out=ROOT/'runs/completion_mask/current_corruptions.png';sheet.save(out);print(out)
