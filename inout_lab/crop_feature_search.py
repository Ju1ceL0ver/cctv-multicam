"""Recover high-resolution crop appearance/context missing from previous trees."""
from pathlib import Path
import numpy as np,json,cv2
from sklearn.ensemble import ExtraTreesClassifier
root=Path(__file__).parent

def crop_features(crop):
 rgb=crop[:,:,:3]/255.;mask=crop[:,:,3]/255.;floor=crop[:,:,4]/255.;v=[]
 for w,h in [(4,6),(8,12)]:
  mass=cv2.resize(mask,(w,h),interpolation=cv2.INTER_AREA);background=1-mask
  bgmass=cv2.resize(background,(w,h),interpolation=cv2.INTER_AREA)
  fg=cv2.resize(rgb*mask[:,:,None],(w,h),interpolation=cv2.INTER_AREA)/np.maximum(mass[:,:,None],.01)
  bg=cv2.resize(rgb*background[:,:,None],(w,h),interpolation=cv2.INTER_AREA)/np.maximum(bgmass[:,:,None],.01)
  v.extend([fg.ravel(),bg.ravel(),(fg-bg).ravel(),mass.ravel(),cv2.resize(floor,(w,h),interpolation=cv2.INTER_AREA).ravel()])
 gray=cv2.cvtColor((rgb*255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(float)/255
 gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3);mag=np.hypot(gx,gy)
 for a in [mag*mask,mag*(1-mask),np.abs(gx)*mask,np.abs(gy)*mask]:v.append(cv2.resize(a,(8,12),interpolation=cv2.INTER_AREA).ravel())
 return np.concatenate(v).astype(np.float32)

if __name__=='__main__':
 z=np.load(root/'io_cam1.npz');X=z['X'];features=np.stack([crop_features(x) for x in X]);np.save(root/'diagnostics/crop_rgb_features.npy',features);del X;y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;A=np.column_stack([np.load(root/('diagnostics/'+n+'_features.npy')) for n in ['rich','head','context']]+[features]);P=np.zeros((len(y),3))
 for d in np.unique(days):
  tr=(days!=d)&valid;te=days==d;model=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=3);P[te]=model.fit(A[tr],y[tr]).predict_proba(A[te]);print('crop',d,flush=True)
 pred=P.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);r={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int(((pred!=y)&valid).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'days':{str(d):{'wrong':int(((pred!=y)&valid&(days==d)).sum()),'gross':int((gross&(days==d)).sum())} for d in np.unique(days)}};np.savez(root/'diagnostics/crop_rgb_predictions.npz',probabilities=P);(root/'crop_feature_report.json').write_text(json.dumps(r,indent=2));print('CROP RESULT',r,flush=True)
