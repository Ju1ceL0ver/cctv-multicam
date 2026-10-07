"""COCO pose landmarks matched to the TARGET mask; no inside/outside labels.
Low-confidence/occluded landmarks remain uncertainty features, not truth feet.
"""
from pathlib import Path
import numpy as np,cv2,json,time
root=Path(__file__).parent

def select_target(result,mask):
 if result.keypoints is None or not len(result.keypoints.data):return None,0.
 points=result.keypoints.data.cpu().numpy();scores=[];h,w=mask.shape
 for pose in points:
  x=np.clip(np.rint(pose[:,0]).astype(int),0,w-1);y=np.clip(np.rint(pose[:,1]).astype(int),0,h-1);inside=mask[y,x]/255.;conf=pose[:,2]
  score=float((inside[:5]*conf[:5]).sum()/max(.1,conf[:5].sum())+.15*(inside*conf).sum()/max(.1,conf.sum()))
  scores.append(score)
 index=int(np.argmax(scores));return points[index],scores[index]

if __name__=='__main__':
 import torch
 from ultralytics import YOLO
 torch.set_num_threads(3);model=YOLO(str(root/'weights/yolo26n-pose.pt'));z=np.load(root/'diagnostics/spatial_predictor_samples.npz');crops=z['X'];images=[x[:,:,:3][:,:,::-1].copy() for x in crops];start=time.perf_counter();results=model.predict(images,imgsz=256,device='cpu',conf=.1,verbose=False)
 out=[]
 for i,x,result in zip(z['indices'],crops,results):
  pose,score=select_target(result,x[:,:,3]);record={'index':int(i),'detections':len(result.boxes),'association':score,'keypoints':pose.tolist() if pose is not None else None};out.append(record);print('pose',i,'detections',len(result.boxes),'association',round(score,3),'headconf',pose[:5,2].round(2).tolist() if pose is not None else [],'ankleconf',pose[15:,2].round(2).tolist() if pose is not None else [],flush=True)
 (root/'pose_probe.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'records':out},indent=2))
