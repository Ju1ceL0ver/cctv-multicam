"""Pose from recovered original pixels, fixed pretrained weights, all targets."""
from pathlib import Path
import os,time,json
R=Path(__file__).parent
os.environ['YOLO_CONFIG_DIR']=str(R/'diagnostics/yolo_config')
import numpy as np,torch
from ultralytics import YOLO
from pose_features import select_target
from extract_pose import pose_vector
if __name__=='__main__':
 torch.set_num_threads(2);q=np.load(R/'diagnostics/recovered_crops.npz');X=q['X'];g=q['G'][:,1:];model=YOLO(str(R/'weights/yolo26n-pose.pt'));rows=np.zeros((1628,132),np.float32);done=np.zeros(1628,bool);start=time.perf_counter()
 for a in range(0,len(X),16):
  ix=np.arange(a,min(a+16,len(X)));out=model.predict([X[i,:,:,:3][:,:,::-1].copy() for i in ix],imgsz=256,device='cpu',conf=.1,verbose=False)
  for i,p in zip(ix,out):
   pose,score=select_target(p,X[i,:,:,3]);rows[i]=pose_vector(pose,score,g[i]);done[i]=True
  if a%256==0:print('source pose',int(done.sum()),'elapsed',round(time.perf_counter()-start),flush=True)
 assert done.all() and np.isfinite(rows).all();np.save(R/'diagnostics/recovered_pose_features.npy',rows);np.save(R/'diagnostics/recovered_pose_done.npy',done);report={'n':1628,'selected':int((rows[:,127]>0).sum()),'seconds':time.perf_counter()-start,'labels_used':False,'imgsz':256,'pretrained_weights':'existing yolo26n-pose.pt','missing_rows_retained':True};(R/'recovered_pose_features_report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
