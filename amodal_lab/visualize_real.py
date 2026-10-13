"""Display real owner-labeled occlusions; inferred area is not ground truth."""
import json,hashlib,argparse
import torch,numpy as np
from PIL import Image,ImageDraw
from train import ROOT,Tiny
parser=argparse.ArgumentParser();parser.add_argument('--run-name',default='completion_rich');args=parser.parse_args();out=ROOT/'runs'/args.run_name;ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True);m=Tiny('completion',1 if ck['config'].get('input_mode')=='mask' else 4);m.load_state_dict(ck['model']);m.eval()
labels=json.loads((ROOT/'data/labels.json').read_text());items=json.loads((ROOT/'data/manifest.json').read_text())['items']
rows=[r for r in items if labels.get(r['id'],{}).get('visibility')=='occluded' and labels[r['id']].get('quality')=='good']
# Fixed selection, no selection based on model output; prefer held-out day.
rows.sort(key=lambda r:(r['day']!='20260918',hashlib.sha256(r['id'].encode()).hexdigest()));rows=rows[:8]
W,H=192,320;sheet=Image.new('RGB',(W*3,len(rows)*(H+25)+50),'#172033');draw=ImageDraw.Draw(sheet)
for j,s in enumerate(['Real occlusion','Observed mask','Completion guess']):draw.text((j*W+8,12),s,fill='white')
with torch.no_grad():
 for i,r in enumerate(rows):
  rgb=np.asarray(Image.open(ROOT/'data'/r['image']).convert('RGB').resize((96,160)),np.float32)/255
  mask=np.asarray(Image.open(ROOT/'data'/labels[r['id']].get('mask_path',r['mask'])).convert('L').resize((96,160),Image.Resampling.NEAREST))>127
  a=mask[None].astype(np.float32) if ck['config'].get('input_mode')=='mask' else np.concatenate([rgb,mask[:,:,None]],2).astype(np.float32).transpose(2,0,1);x=torch.from_numpy(a)[None];pred=m(x).sigmoid().numpy()[0,0]>.5
  observed=rgb.copy();observed[mask]=observed[mask]*.5+np.array([0,.5,.12])
  completed=observed.copy();guess=pred&~mask;completed[guess]=completed[guess]*.35+np.array([.65,.25,0])
  for j,a in enumerate([rgb,observed,completed]):sheet.paste(Image.fromarray((np.clip(a,0,1)*255).astype('uint8')).resize((W,H)),(j*W,50+i*(H+25)))
  draw.text((8,50+i*(H+25)+H+4),r['id'],fill='white')
sheet.save(out/'real_occlusions.jpg');(out/'real_occlusions_ids.json').write_text(json.dumps([r['id'] for r in rows],indent=2));print(out/'real_occlusions.jpg')
