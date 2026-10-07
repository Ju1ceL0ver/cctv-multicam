"""Inference implementation parity using original source pixels, no refit scores."""
from pathlib import Path
import json,time
import numpy as np,cv2,torch
from recovered_classifier import RecoveredClassifier,feature_parts
from door_features import expanded_features
from crop_feature_search import crop_features
R=Path(__file__).parent
if __name__=='__main__':
 torch.set_num_threads(1);z=np.load(R/'io_cam1.npz');G=z['G'][:,1:];A=np.load(R/'diagnostics/recovered_features.npy');aligned=np.load(R/'diagnostics/aligned_geometry.npy');results=[];m=None
 for i in [393,970,1149,1394]:
  d=z['day'][i];m=RecoveredClassifier(R/f'diagnostics/owner_boundary_mlp_recovered_19_{d}.pt');rgb=cv2.imread(str(R/f'diagnostics/api_{i}_rgb.png'))[:,:,::-1].copy();mask=cv2.imread(str(R/f'diagnostics/api_{i}_mask.png'),0);g=G[i];kwargs={'box':g[:4]*[1280,720,1280,720],'foot':g[6:8]*[1280,720],'floor_px':float(g[8]*100)}
  full,crop,ag,orig=m.prepare_rgb(rgb,mask,**kwargs);a=np.concatenate([expanded_features(full,crop[:,:,3],ag,m.helper.signed),crop_features(crop)]);err=float(abs(a-A[i]).max());p=m.predict_features(a,ag,orig)[0];expected=np.load(R/'diagnostics/owner_boundary_mlp_recovered_19.npz')['probabilities'][i];perr=float(abs(p-expected).max());assert err<1e-4,(i,err);assert perr<1e-4,(i,perr)
  results.append({'index':i,'feature_error':err,'probability_error':perr,'response':m.predict_rgb(rgb,mask,**kwargs)})
 # This benchmark is inference only on CPU, excludes segmentation and source IO.
 for _ in range(10):m.predict_rgb(rgb,mask,**kwargs)
 times=[]
 for _ in range(50):
  t=time.perf_counter();m.predict_rgb(rgb,mask,**kwargs);times.append((time.perf_counter()-t)*1000)
 out={'original_rgb_mask_parity':results,'cpu_rgb_plus_mask_ms':{'median':float(np.median(times)),'p90':float(np.percentile(times,90))},'segmentation_included':False,'fit_on_all_data':False,'prediction_count_per_call':1,'device':'CPU,1torch thread','concurrent_training':True};(R/'recovered_classifier_api_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
