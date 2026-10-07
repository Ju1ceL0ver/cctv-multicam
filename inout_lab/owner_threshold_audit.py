"""Owner's doorway boundary: diagnostic foot-side audit, no training or deployment."""
import json,cv2,numpy as np
from pathlib import Path
r=Path(__file__).parent
annotation=json.loads((r/'owner_seams.json').read_text());ends=np.asarray(next(o['points'] for o in annotation['objects'] if o['id']=='threshold1'),float)
cal=json.loads((r/'diagnostics/calib_final_source.json').read_text())['cam1'];K=np.array(cal['K'],float);K[:2]*=.5;dist=np.array(cal['dist'],float)
z=np.load(r/'io_cam1.npz');G=z['G'];y=z['y'];day=z['day'];valid=np.load(r/'diagnostics/original_labels.npy')!=0
undistort=lambda x:cv2.undistortPoints(np.asarray(x,float).reshape(-1,1,2),K,dist,P=K).reshape(-1,2)
a,b=undistort(ends);v=b-a;normal=np.array([-v[1],v[0]])/np.linalg.norm(v)
inside=undistort([[640,600]])[0]
if (inside-a)@normal<0:normal=-normal
foot=G[:,7:9]*[1280,720];signed=(undistort(foot)-a)@normal;pred=(signed>=0).astype(int)
side=valid&(y!=2);wrong=side&(pred!=y)
result={'boundary_source':'owner hand-marked threshold1','image_size_px':[1280,720],'endpoints_px':ends.tolist(),'lens_distortion_used':True,'units':'signed perpendicular pixels in undistorted 1280x720 grid','training_run':False,'three_class_model_evaluated':False,'binary_foot_diagnostic_only':{'count':int(side.sum()),'wrong':int(wrong.sum()),'accuracy':float((pred[side]==y[side]).mean()),'wrong_indices':np.flatnonzero(wrong).tolist()},'class_signed_distance_quantiles':{str(c):np.percentile(signed[valid&(y==c)],[5,25,50,75,95]).tolist() for c in [0,1,2]},'by_day':{str(d):{'count':int((side&(day==d)).sum()),'wrong':int((wrong&(day==d)).sum())} for d in np.unique(day)},'previous_gross_cases':{str(i):{'true_label':int(y[i]),'signed_px':float(signed[i]),'foot_side':int(pred[i])} for i in [483,625,789,970,1149]},'production_changed':False}
(r/'owner_threshold_audit.json').write_text(json.dumps(result,indent=2));np.save(r/'diagnostics/owner_threshold_signed_px.npy',signed)
print(json.dumps({k:v for k,v in result.items() if k!='binary_foot_diagnostic_only'},indent=2));print('binary diagnostic wrong',wrong.sum(),'of',side.sum())
