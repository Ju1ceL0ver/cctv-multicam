"""Match annotated detections to actual SAM tracks, never to draft pN indices.
History/future geometries contain no human side labels. Future <=3 s is allowed
by requested 10-30 s delay; this is a buffered, not instantaneous prediction.
"""
from pathlib import Path
import gzip,json,re
import numpy as np
root=Path(__file__).parent

def geometry(q):
 box=np.asarray(q['box'])*[1,1248/1224,1,1248/1224]
 foot=np.asarray(q['foot'])/[2176,1224] if q['foot'] is not None else np.r_[box[0],box[1]+box[3]/2]
 return np.r_[box,foot]

if __name__=='__main__':
 z=np.load(root/'io_cam1.npz');ids=z['ids'];g=z['G'][:,1:];out=np.zeros((len(ids),265),np.float32);matches={};coverage={}
 for day in ['20260917','20260918','20260919']:
  with gzip.open(root/('diagnostics/temporal/'+day+'_sam31.jsonl.gz'),'rt') as f:
   meta=json.loads(next(f));ticks=[json.loads(line) for line in f]
  tracks={};at={}
  for tick in ticks:
   for q in tick['p']:
    tracks.setdefault((tick['s'],q['w']),[]).append((tick['t'],geometry(q)))
   at[(tick['s'],round(tick['t'],2))]=tick['p']
  tracks={k:(np.array([x[0] for x in rows]),np.stack([x[1] for x in rows])) for k,rows in tracks.items()}
  for i,ident in enumerate(ids):
   m=re.fullmatch(day+r'_door(\d+)_k(\d+)_cam1_p\d+',str(ident))
   if not m:continue
   start,k=map(int,m.groups());span=[j for j,s in enumerate(meta['spans']) if int(round(s[0]))==start]
   if len(span)!=1:continue
   s=span[0];t=round(meta['spans'][s][0]+k*meta.get('tick',.08),2);qs=at.get((s,t),[])
   if not qs:continue
   target=np.r_[(g[i,:2]+g[i,2:4])/2,g[i,2:4]-g[i,:2]]
   errors=np.array([np.max(np.abs(geometry(q)[:4]-target)) for q in qs]);order=np.argsort(errors);j=int(order[0])
   if errors[j]>.004 or (len(order)>1 and errors[order[1]]-errors[j]<.001):continue
   q=qs[j];times,B=tracks[(s,q['w'])];vec=[1.]
   for direction in [-1,1]:
    for radius in ([1.,3.,10.] if direction<0 else [.5,1.,3.]):
     # Future deliberately capped at3s, including the 10s summary slot.
     r=radius if direction<0 else min(radius,3.)
     use=(times>=t-r)&(times<=t) if direction<0 else (times>=t)&(times<=t+r)
     b=B[use]
     summary=np.r_[b.mean(0),b.std(0),b.min(0),b.max(0),b[0],b[-1],b[-1]-b[0]] if len(b) else np.zeros(42)
     vec.extend([min(1.,len(b)/max(1,r*12.5)),float((times[use].max()-times[use].min())/r) if len(b) else 0.,*summary])
   # 1+6*(2+42)=265 (six windows); shape checked below.
   assert len(vec)==265,len(vec)
   out[i]=vec
   matches[str(i)]={'day':day,'span':s,'track':int(q['w']),'piece':q.get('piece'),'time':t,'box_error':float(errors[j])}
  coverage[day]=sum(v['day']==day for v in matches.values())
 np.save(root/'diagnostics/track_context_features.npy',out);(root/'track_context_match_report.json').write_text(json.dumps({'coverage':coverage,'matches':matches},indent=2));print('track coverage',coverage,flush=True)
