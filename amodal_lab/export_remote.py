"""CPU export of 500 source person crops; run on Windows via jctl runfile."""
from pathlib import Path
import json, hashlib, zipfile
import cv2
import numpy as np
import sys
root=Path(r'C:\Users\ArykovAA\cctv_ai\multicam')
sys.path.insert(0,str(root))
import rate
samples=json.loads((root/'data/inout/samples.json').read_text(encoding='utf-8'))
frames=rate.index(root)
out=root/'data/amodal_seed_v1';out.mkdir(exist_ok=True)
# Equal round-robin sampling across days, one person per source frame.
buckets={}
for sid,s in samples.items():
 if s['frame'] in frames:
  buckets.setdefault(sid[:8],[]).append((sid,s))
for b in buckets.values():b.sort(key=lambda x:hashlib.sha256(x[0].encode()).hexdigest())
selected=[];seen=set()
while len(selected)<500 and any(buckets.values()):
 for day in sorted(buckets):
  if not buckets[day]:continue
  sid,s=buckets[day].pop()
  if s['frame'] in seen:continue
  seen.add(s['frame']); selected.append((sid,s))
  if len(selected)==500:break
items=[]
for sid,s in selected:
 ip,mp=frames[s['frame']];img=cv2.imread(str(ip));lab=cv2.imread(str(mp),cv2.IMREAD_UNCHANGED)
 if img is None or lab is None:continue
 if lab.ndim==3:lab=lab[:,:,0]
 lab=cv2.resize(lab,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
 m=lab==s['value'];ys,xs=np.where(m)
 if not len(xs):continue
 h,w=img.shape[:2];x1,x2=int(xs.min()),int(xs.max()+1);y1,y2=int(ys.min()),int(ys.max()+1)
 pw=max(24,int((x2-x1)*.5));ph=max(24,int((y2-y1)*.25))
 a,b,c,d=max(0,x1-pw),max(0,y1-ph),min(w,x2+pw),min(h,y2+ph)
 rgb=img[b:d,a:c];mask=m[b:d,a:c].astype('uint8')*255
 cv2.imwrite(str(out/(sid+'.jpg')),rgb,[cv2.IMWRITE_JPEG_QUALITY,94]);cv2.imwrite(str(out/(sid+'_mask.png')),mask)
 # Context makes the target and real occluder inspectable.
 context=img.copy();cv2.rectangle(context,(x1,y1),(x2,y2),(0,255,255),3)
 cv2.imwrite(str(out/(sid+'_context.jpg')),context,[cv2.IMWRITE_JPEG_QUALITY,85])
 items.append(dict(id=sid,day=sid[:8],frame=s['frame'],person_value=int(s['value']),image=sid+'.jpg',mask=sid+'_mask.png',context=sid+'_context.jpg',bbox=[a,b,c,d],source_image=str(ip),source_mask=str(mp),touches_edge=x1<=1 or y1<=1 or x2>=w-1 or y2>=h-1))
(out/'manifest.json').write_text(json.dumps(dict(version=1,source='inout samples; source RGB + teacher label maps; no visibility labels yet',split_policy='hold out whole days; identities across days not verified',items=items),ensure_ascii=False,indent=2),encoding='utf-8')
archive=root/'data/amodal_seed_v1.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in out.iterdir():z.write(p,p.name)
print(json.dumps(dict(count=len(items),archive=str(archive),bytes=archive.stat().st_size)))
