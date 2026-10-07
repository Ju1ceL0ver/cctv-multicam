"""Diagnostic vanishing directions from owner's corridor seams; fixed intrinsics."""
import json,cv2,numpy as np
from pathlib import Path
r=Path(__file__).parent
annotations=json.loads((r/'owner_seams.json').read_text());cal=json.loads((r/'diagnostics/calib_final_source.json').read_text())['cam1'];K=np.array(cal['K'],float);K[:2]*=.5;dist=np.array(cal['dist'],float)
rows=[];lines={}
for o in annotations['objects']:
 p=np.asarray(o['points'],float);u=cv2.undistortPoints(p.reshape(-1,1,2),K,dist,P=K).reshape(-1,2)
 mid=u.mean(0);_,_,v=np.linalg.svd(u-mid);normal=v[-1];line=np.r_[normal,-normal@mid];lines[o['id']]=line
 rows.append({'id':o['id'],'n':len(p),'max_line_error_px':float(np.max(np.abs((u-mid)@normal))),'rms_line_error_px':float(np.sqrt(np.mean(((u-mid)@normal)**2)))})
def direction(ids):
 # normalize in ray coordinates, fit common vanishing ray using equal seam weights
 L=np.array([K.T@lines[i] for i in ids]);L/=np.linalg.norm(L,axis=1)[:,None];_,sing,v=np.linalg.svd(L);ray=v[-1];ray/=np.linalg.norm(ray)
 return ray,{'ids':ids,'ray':ray.tolist(),'line_ray_angle_residual_deg':np.degrees(np.arcsin(np.clip(np.abs(L@ray),0,1))).tolist(),'singular_values':sing.tolist()}
a,A=direction(['seam3','seam4','seam5']);b,B=direction(['seam6','seam7','seam8']);angle=np.degrees(np.arccos(np.clip(abs(a@b),0,1)));normal=np.cross(a,b);normal/=np.linalg.norm(normal)
oldnormal=cv2.Rodrigues(np.asarray(cal['rvec']))[0][:,2];difference=np.degrees(np.arccos(np.clip(abs(normal@oldnormal),0,1)))
# Leave-one-seam-out stability within each direction (2 lines remain)
stability=[]
for ids in [A['ids'],B['ids']]:
 full=a if ids==A['ids'] else b
 for leave in ids:
  ray,_=direction([i for i in ids if i!=leave]);stability.append({'omit':leave,'direction_change_deg':float(np.degrees(np.arccos(np.clip(abs(ray@full),0,1))))})
out={'fixed_intrinsics':True,'metric_distance_estimated':False,'tile_aspect_ratio_estimated':False,'line_audit':rows,'corridor_direction_1':A,'corridor_direction_2':B,'acute_angle_between_directions_deg':float(angle),'floor_normal_camera_coordinates':normal.tolist(),'normal_difference_from_old_floor_deg':float(difference),'leave_one_seam_out':stability,'new_calibration_accepted':False}
(r/'owner_seams_geometry.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
