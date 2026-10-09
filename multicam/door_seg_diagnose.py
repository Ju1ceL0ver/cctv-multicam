"""Profile and cache predictions exclusively on owner-labelled train door clips.

Run on the Windows experiment GPU: door_seg_diagnose.py [--ticks 400].
Saved predictions can be replayed without GPU for association diagnostics.
"""
import os
os.environ['RA_DS_GLUED']='1'
import argparse,json,time,datetime
from pathlib import Path
import cv2,numpy as np,torch
import door_seg as S
from door_seg_run import contexts
from door_seg_track import Tracker
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--ticks",type=int,default=400);args=parser.parse_args()
r=S.ROOT
st=json.loads((r/'data/door_side/20260918.json').read_text())
tag=next(k for k,v in st.items() if v.get('done') and v.get('role')=='train')
out=r/'data/quality_backups/doorseg_tracker_diag';out.mkdir(parents=True,exist_ok=True)
src=S.Source(r/'data/sam31_door'/tag/'cam1')
net=S.DoorSeg('cuda'); net.load(torch.load(r/'runs/doorseg_b/step8604.pt',map_location='cpu',weights_only=False)['model']);net.yolo.eval();net.off.eval()
cap=cv2.VideoCapture(str(src.d/'video.mp4'));prev=np.zeros((S.H,S.W),np.uint8);bg=S.background('20260918','cam1');tr=Tracker();saved=[];timing=[];offsets=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for tick,frames in contexts(cap):
  if tick>=args.ticks or datetime.datetime.now().time()>=datetime.time(9,35):break
  if tick%50==0 and list(__import__('bg').running('door_live.py')): print('live door present, stopping',flush=True);break
  t=time.perf_counter();frames={o:cv2.resize(f,(S.W,S.H),interpolation=cv2.INTER_AREA) for o,f in frames.items()};img=torch.from_numpy(S.channels(frames,bg,prev))[None].cuda().float()/255;torch.cuda.synchronize();a=time.perf_counter()
  p=S.predict(net,img);torch.cuda.synchronize();b=time.perf_counter();linked=tr.update(tick,p);c=time.perf_counter();prev.fill(0)
  for _,pp in linked:prev|=cv2.resize(pp[0][:180].astype(np.uint8),(S.W,S.H),interpolation=cv2.INTER_NEAREST)
  truth=src.people(None,tick);before=src.people(None,tick-1);after=src.people(None,tick+1);gt=[(cv2.copyMakeBorder(cv2.resize(m,(320,180),interpolation=cv2.INTER_NEAREST),0,4,0,0,cv2.BORDER_CONSTANT,value=0),box,who) for who,(m,box) in truth.items()]
  # D._match consumes mask/box/score predictions and mask/box truth.
  matches=S._match([pp[0] for pp in p],[m for m,box,who in gt])
  saved.append((tick,p,gt,[(pid,j) for j,(pid,_) in enumerate(linked)]))
  timing.append([a-t,b-a,c-b,time.perf_counter()-c])
  for i,j,iou in matches:
   who=gt[j][2];box=gt[j][1];center=(box[:2]+box[2:])/2
   for direction,other in enumerate((before,after)):
    if who in other:
     ob=other[who][1];target=(ob[:2]+ob[2:])/2-center
     offsets.append(float(np.linalg.norm(p[i][3+direction]-target)))
  if tick%50==0:print(tick,tr.stats,flush=True)
cap.release();src.close();torch.save(saved,out/'train_predictions.pt')
summary={'tag':tag,'ticks':len(saved),'stats':tr.stats,'mean_ms':(np.mean(timing[10:],axis=0)*1000).tolist(),'timing_order':['context','predict','tracker','prev_and_gt'],'offset_error_quantiles':np.quantile(offsets,[.5,.9,.99]).tolist() if offsets else []}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
