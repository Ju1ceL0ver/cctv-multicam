"""Occlusion invariance: preserve physical side label while hiding lower body.
Fold-specific backgrounds/occluders use training frames only. No held-day labels
or frames enter augmentation; every original defined test sample is scored.
"""
from pathlib import Path
import numpy as np,cv2,json,time
from sklearn.ensemble import ExtraTreesClassifier
from door_features import expanded_features
from crop_feature_search import crop_features
from inside_outside import POLYGON
root=Path(__file__).parent

def floor_distance():
 original=np.zeros((1440,2560),np.uint8);cv2.fillPoly(original,[(np.array(POLYGON,np.float32)*1.6).astype(np.int32)],255);floor=cv2.resize(original,(1280,720),interpolation=cv2.INTER_NEAREST)
 return cv2.distanceTransform(floor,cv2.DIST_L2,3)-cv2.distanceTransform(255-floor,cv2.DIST_L2,3)

def augment(full,crop,g,bg,rng,dist):
 f=full.copy();x=crop.copy();rr,cc=np.where(f[:,:,3]>63)
 if len(rr)<20 or rr.max()-rr.min()<8:return None
 top,bottom,left,right=rr.min(),rr.max(),cc.min(),cc.max();cut=top+max(3,int((bottom-top+1)*rng.uniform(.35,.75)))
 f[cut:bottom+1,left:right+1,:3]=bg[cut:bottom+1,left:right+1];f[cut:,:,3]=0
 # Preserve high-resolution crop detail above the occluder.
 gx=x[:,:,5]/255.;gy=x[:,:,6]/255.;hidden=(gy>=cut/180)&(gy<=(bottom+1)/180)&(gx>=left/320)&(gx<=(right+1)/320)
 u=np.clip((gx*320).astype(int),0,319);v=np.clip((gy*180).astype(int),0,179);x[hidden,:3]=f[v[hidden],u[hidden],:3];x[gy>=cut/180,3]=0
 rr,cc=np.where(f[:,:,3]>63)
 if not len(rr):return None
 x1,y1,x2,y2=cc.min()/320,rr.min()/180,(cc.max()+1)/320,(rr.max()+1)/180;fx=cc[rr>=rr.max()-2].mean()/320;fy=rr.max()/180
 d=float(dist[int(np.clip(fy*720,0,719)),int(np.clip(fx*1280,0,1279))])/100
 geometry=np.array([x1,y1,x2,y2,x2-x1,y2-y1,fx,fy,d],np.float32)
 # Re-crop the original detailed patch to match its NEW visible bounding box.
 X1,X2=max(0,x1-(x2-x1)/2),min(1,x2+(x2-x1)/2);Y1,Y2=max(0,y1-(y2-y1)/8),min(1,y2+(y2-y1)/8)
 cols=np.where((gx[0]>=X1)&(gx[0]<=X2))[0];rows=np.where((gy[:,0]>=Y1)&(gy[:,0]<=Y2))[0]
 if not len(cols) or not len(rows):return None
 x=cv2.resize(x[rows.min():rows.max()+1,cols.min():cols.max()+1,:4].astype(np.float32),(96,160),interpolation=cv2.INTER_AREA).astype(np.uint8)
 # Crop floor channel is needed too; resize independently (OpenCV max 4 chans).
 crop_floor=cv2.resize(crop[rows.min():rows.max()+1,cols.min():cols.max()+1,4].astype(np.float32),(96,160),interpolation=cv2.INTER_AREA).astype(np.uint8)
 x=np.dstack([x,crop_floor])
 return f,x,geometry

if __name__=='__main__':
 z=np.load(root/'io_cam1.npz');F=z['F'];X=z['X'];G=z['G'][:,1:];y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;A=np.column_stack([np.load(root/('diagnostics/'+n+'_features.npy')) for n in ['rich','head','context']]+[np.load(root/'diagnostics/crop_rgb_features.npy')]);P=np.zeros((len(y),3));dist=floor_distance();start=time.perf_counter()
 for day in np.unique(days):
  tr=(days!=day)&valid;te=days==day;indices=np.where(tr)[0];rng=np.random.default_rng(71);background=np.median(F[rng.choice(indices,min(24,len(indices)),replace=False),:,:,:3],axis=0).astype(np.uint8);aug=[];labels=[]
  for i in indices:
   sample=augment(F[i],X[i],G[i],background,rng,dist)
   if sample is None:continue
   full,crop,geometry=sample;aug.append(np.concatenate([expanded_features(full,crop[:,:,3],geometry),crop_features(crop)]));labels.append(y[i])
  train=np.vstack([A[tr],aug]);target=np.concatenate([y[tr],labels]);weights=np.concatenate([np.ones(tr.sum()),np.full(len(aug),.5)])
  model=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=3).fit(train,target,sample_weight=weights);P[te]=model.predict_proba(A[te]);print('occlusion',day,'augmented',len(aug),'elapsed',round(time.perf_counter()-start),flush=True)
 pred=P.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);r={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int(((pred!=y)&valid).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'days':{str(d):{'wrong':int(((pred!=y)&valid&(days==d)).sum()),'gross':int((gross&(days==d)).sum())} for d in np.unique(days)}};np.savez(root/'diagnostics/occlusion_predictions.npz',probabilities=P);(root/'occlusion_report.json').write_text(json.dumps(r,indent=2));print('OCCLUSION',r,flush=True)
