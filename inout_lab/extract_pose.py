"""Checkpointed pretrained pose extraction. No manual side labels are consumed."""
from pathlib import Path
import os,time,json
import numpy as np
root=Path(__file__).parent
(root/'diagnostics/yolo_config').mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(root/'diagnostics/yolo_config')
from pose_features import select_target

def pose_vector(pose,score,g):
 if pose is None or score<.3:return np.zeros(132,np.float32)
 p=pose.copy().astype(float);x1,y1,x2,y2=g[:4]*[1280,720,1280,720];w,h=x2-x1,y2-y1
 left=max(0,int(x1-w/2));right=min(1280,int(np.ceil(x2+w/2)));top=max(0,int(y1-h/8));bottom=min(720,int(np.ceil(y2+h/8)))
 xy=p[:,:2].copy();p[:,0]=(left+xy[:,0]/96*(right-left))/1280;p[:,1]=(top+xy[:,1]/160*(bottom-top))/720
 pairs=[(1,2),(3,4),(5,6),(11,12),(0,5),(0,6),(5,11),(6,12),(11,15),(12,16),(13,15),(14,16)]
 v=[p.ravel()]
 for a,b in pairs:
  delta=p[a,:2]-p[b,:2];v.append(np.array([*delta,np.linalg.norm(delta),min(p[a,2],p[b,2])]))
 # 51 + 48 + 17 confidence-weighted validity + 6 landmark confidence groups +
 # 4 observed target geometry offsets + 6 association/validity summaries =132.
 v.append((p[:,2]>.5).astype(float))
 v.append(np.array([p[:5,2].mean(),p[5:7,2].mean(),p[11:13,2].mean(),p[13:15,2].mean(),p[15:17,2].mean(),p[:,2].mean()]))
 v.append(np.array([p[0,0]-g[6],p[0,1]-g[7],p[15,0]-g[6],p[15,1]-g[7]]))
 v.append(np.array([score,1.,float((p[:5,2]>.5).sum()),float((p[15:17,2]>.5).sum()),float((p[:,2]>.5).sum()),float(p[:,2].max())]))
 out=np.concatenate(v).astype(np.float32);assert len(out)==132,len(out);return out

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--imgsz',type=int,default=256);args=parser.parse_args()
 assert args.imgsz in (256,512,640), 'Supported image sizes:256,512,640'
 suffix='' if args.imgsz==256 else '_'+str(args.imgsz)
 import torch
 from ultralytics import YOLO
 torch.set_num_threads(3);model=YOLO(str(root/'weights/yolo26n-pose.pt'));z=np.load(root/'io_cam1.npz');X=z['X'];G=z['G'][:,1:];path=root/('diagnostics/pose'+suffix+'_features.npy');done_path=root/('diagnostics/pose'+suffix+'_done.npy');raw_path=root/('diagnostics/pose'+suffix+'_landmarks.npy');score_path=root/('diagnostics/pose'+suffix+'_association.npy')
 if path.exists() and done_path.exists():vectors=np.load(path);done=np.load(done_path);landmarks=np.load(raw_path);scores=np.load(score_path)
 else:vectors=np.zeros((len(X),132),np.float32);done=np.zeros(len(X),bool);landmarks=np.zeros((len(X),17,3),np.float32);scores=np.zeros(len(X),np.float32)
 start=time.perf_counter()
 todo=np.where(~done)[0]
 for indices in np.array_split(todo,max(1,int(np.ceil(len(todo)/16)))):
  if not len(indices):continue
  results=model.predict([X[i,:,:,:3][:,:,::-1].copy() for i in indices],imgsz=args.imgsz,device='cpu',conf=.1,verbose=False)
  for i,result in zip(indices,results):
   pose,score=select_target(result,X[i,:,:,3]);scores[i]=score
   if pose is not None and score>=.3:landmarks[i]=pose
   vectors[i]=pose_vector(pose,score,G[i]);done[i]=True
  np.save(path,vectors);np.save(done_path,done);np.save(raw_path,landmarks);np.save(score_path,scores)
  if done.sum()%128==0 or done.all():print('pose extracted',int(done.sum()),'/',len(X),'target found',int((vectors[:,126]>0).sum()),'seconds',round(time.perf_counter()-start),flush=True)
 (root/('pose'+suffix+'_extraction_report.json')).write_text(json.dumps({'n':len(X),'selected_targets':int((vectors[:,127]>0).sum()),'seconds_this_run':time.perf_counter()-start,'imgsz':args.imgsz,'model':'yolo26n-pose','association_min':.3},indent=2))
