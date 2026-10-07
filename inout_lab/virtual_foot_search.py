"""Camera-ray virtual feet at several body-height/camera-height ratios.
Uses camera calibration as an optional feature, never as a hard truth rule.
"""
from pathlib import Path
import numpy as np,cv2,json
from sklearn.ensemble import ExtraTreesClassifier
root=Path(__file__).parent

def virtual_features(A,cal):
 K=np.array(cal['K']);dist=np.array(cal['dist']);rv=np.array(cal['rvec']);tv=np.array(cal['tvec']);R=cv2.Rodrigues(rv)[0];C=-R.T@tv;H=C[2];head=A[:,234:234+2]*[2560,1440];n=cv2.undistortPoints(head.reshape(-1,1,2).astype(float),K,dist).reshape(-1,2);rays=np.column_stack([n,np.ones(len(n))])@R;parts=[]
 for fraction in [.2,.3,.4,.5,.6]:
  z=H*fraction;lam=(z-H)/rays[:,2];xyz=C+lam[:,None]*rays;xyz[:,2]=0;pixel=cv2.projectPoints(xyz,rv,tv,K,dist)[0].reshape(-1,2)/[2560,1440];pixel=np.clip(np.nan_to_num(pixel),-5,5);parts.extend([pixel,pixel-A[:,6:8],np.linalg.norm(pixel-A[:,6:8],axis=1,keepdims=True)])
 return np.column_stack(parts).astype(np.float32)

if __name__=='__main__':
 z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;A=np.column_stack([np.load(root/('diagnostics/'+n+'_features.npy')) for n in ['rich','head','context']]);cal=json.loads((root/'diagnostics/calib_final_source.json').read_text())['cam1'];virtual=virtual_features(A,cal);np.save(root/'diagnostics/virtual_foot_features.npy',virtual);P=np.zeros((len(y),3))
 for d in np.unique(days):
  tr=(days!=d)&valid;te=days==d;model=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=3).fit(np.column_stack([A[tr],virtual[tr]]),y[tr]);P[te]=model.predict_proba(np.column_stack([A[te],virtual[te]]));print('virtual foot',d,flush=True)
 pred=P.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);r={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int(((pred!=y)&valid).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist()};np.savez(root/'diagnostics/virtual_foot_predictions.npz',probabilities=P);(root/'virtual_foot_report.json').write_text(json.dumps(r,indent=2));print('VIRTUAL',r,flush=True)
