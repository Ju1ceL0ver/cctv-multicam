"""Bounded SAM 2.1 pilot: 4 seconds, one reviewed identity, no training or GT edits.

Outputs side-by-side video for human judgement. Teacher-mask agreement is not
independent accuracy. Runs only in the night GPU window and caps this process's VRAM.
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime,timedelta
from pathlib import Path
import cv2
import numpy as np
from rawsource import Stream,FPS
from review_store import read_state
from storage import atomic_json
ROOT=Path(__file__).resolve().parent


def encoder(path,width,height,fps):
 import imageio_ffmpeg
 return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)


def finish(proc):
 proc.stdin.close();err=proc.stderr.read().decode(errors='replace');code=proc.wait(timeout=30)
 if code:raise RuntimeError(err[-500:])


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('clip');p.add_argument('piece',type=int)
 p.add_argument('--model',default='sam2.1_s.pt');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 folder=ROOT/'data/raw_clips'/a.clip;state=read_state(folder)
 piece=next(p for p in state['pieces'] if p['piece']==a.piece)
 label=state['labels'].get(str(a.piece))
 if not label or label=='?':raise ValueError('Pilot must start from a reviewed identity')
 cam=piece['cam'];meta=json.loads((folder/'meta_yolo26x-seg.json').read_text(encoding='utf-8'))
 with np.load(folder/'dets_yolo26x-seg.npz') as dz:
  D=dz[cam].copy()
 ids=sorted({i for p in state['pieces'] if p['cam']==cam and state['labels'].get(str(p['piece']))==label for i in p['dets']})
 rows=D[ids];seed=D[piece['dets'][0]];start=float(seed[0])
 out=ROOT/'data/sam_pilot'/('%s_%04d'%(a.clip,a.piece));out.mkdir(parents=True,exist_ok=True)
 video=out/'input.mp4';frames=[];stream=Stream(cam,meta['day']);stream.seek(datetime.fromisoformat(meta['start'])+timedelta(seconds=start))
 try:
  for k in range(min(100,int((meta['seconds']-start)*FPS))):
   _,fr=stream.read()
   if fr is None:break
   if k%3==0:frames.append(cv2.resize(fr,(1280,720)))
 finally:
  if stream.cap is not None:stream.cap.release()
 if not frames:raise RuntimeError('No source frames')
 enc=encoder(video,1280,720,FPS/3)
 for fr in frames:enc.stdin.write(fr.tobytes())
 finish(enc)
 report={'clip':a.clip,'piece':a.piece,'label':label,'camera':cam,'start':start,'frames':len(frames),'model':a.model,'status':'prepared','note':'Side-by-side review only; no independently verified mask ground truth'}
 atomic_json(out/'report.json',report)
 if a.prepare_only:print(json.dumps(report));return
 now=datetime.now().time()
 if datetime.strptime('09:45','%H:%M').time()<=now<datetime.strptime('21:00','%H:%M').time():raise RuntimeError('Daytime GPU belongs to the live counter')
 import torch
 free,total=torch.cuda.mem_get_info()
 if free<3.5*1024**3:raise RuntimeError('Less than 3.5 GiB free VRAM; run pilot when teachers free memory')
 torch.cuda.set_per_process_memory_fraction(.22,0)
 from ultralytics.models.sam import SAM2VideoPredictor
 predictor=SAM2VideoPredictor(overrides={'conf':.25,'task':'segment','mode':'predict','imgsz':1024,'model':a.model,'device':0,'save':False,'verbose':False})
 with np.load(folder/'polys_yolo26x-seg.npz') as pz:
  pts=pz[cam+'_pts'].copy();offsets=pz[cam+'_off'].copy()
 enc=encoder(out/'comparison.mp4',1280,360,FPS/3);t0=time.perf_counter();count=0
 try:
  for k,result in enumerate(predictor(source=str(video),bboxes=(seed[1:5]*.5).tolist(),stream=True)):
   if k>=len(frames):break
   target=frames[k].copy();t=start+k*3/FPS;j=int(np.argmin(abs(rows[:,0]-t)))
   if abs(rows[j,0]-t)<=.15:
    di=ids[j];poly=np.rint(pts[offsets[di]:offsets[di+1]]*.5).astype(np.int32)
    if len(poly)>=3:cv2.polylines(target,[poly],True,(80,255,120),3)
   sam=result.plot(labels=False,boxes=False)
   left=cv2.resize(target,(640,360));right=cv2.resize(sam,(640,360))
   cv2.putText(left,'YOLO teacher',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
   cv2.putText(right,'SAM 2.1 small',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
   enc.stdin.write(np.hstack((left,right)).tobytes());count+=1
  finish(enc)
  report.update(status='ready_for_review',processed=count,elapsed_s=round(time.perf_counter()-t0,2),peak_vram_mb=round(torch.cuda.max_memory_allocated()/1024**2))
 except Exception as exc:
  try:finish(enc)
  except Exception:pass
  report.update(status='failed',error=str(exc)[-1200:]);atomic_json(out/'report.json',report);raise
 atomic_json(out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
