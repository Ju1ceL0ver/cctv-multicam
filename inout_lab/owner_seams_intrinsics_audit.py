"""Sensitivity audit only: rectangular seam directions, no tile lengths/layout."""
from pathlib import Path
import json
import numpy as np
import cv2
from scipy.optimize import minimize_scalar
R=Path(__file__).parent

def solve(objects, focal):
    cal=json.loads((R/'diagnostics/calib_final_source.json').read_text())['cam1']
    K=np.array(cal['K'],float);K[:2]*=.5;K[0,0]=K[1,1]=focal
    dist=np.array(cal['dist'],float)
    lines={};errors=[]
    for o in objects:
        if not o['id'].startswith('seam'):continue
        p=cv2.undistortPoints(np.array(o['points'],float).reshape(-1,1,2),K,dist).reshape(-1,2)
        mid=p.mean(0);_,_,V=np.linalg.svd(p-mid);n=V[-1];l=np.r_[n,-n@mid];l/=np.linalg.norm(l);lines[o['id']]=l
        errors.append(float(np.max(np.abs((p-mid)@n))*focal))
    rays=[]
    for ids in [['seam3','seam4','seam5'],['seam6','seam7','seam8']]:
        _,_,V=np.linalg.svd([lines[i] for i in ids if i in lines]);rays.append(V[-1])
    a,b=rays
    # Remove arbitrary SVD signs for continuous root finding.
    if a[0]<0:a=-a
    if b[0]<0:b=-b
    return float(a@b),rays,max(errors)

def fit(objects):
    fs=np.linspace(500,1700,121);scores=[solve(objects,f)[0]**2 for f in fs];roots=[]
    for i in range(1,len(fs)-1):
        if scores[i]<=scores[i-1] and scores[i]<=scores[i+1]:
            opt=minimize_scalar(lambda f:solve(objects,f)[0]**2,bounds=(fs[i-1],fs[i+1]),method='bounded')
            f=float(opt.x);d,r,e=solve(objects,f);n=np.cross(*r);n/=np.linalg.norm(n)
            roots.append({'focal_px_1280':f,'orthogonality_error_deg':float(np.degrees(np.arcsin(abs(d)))),'normal':n.tolist(),'max_line_residual_px':e})
    return roots

if __name__=='__main__':
    objects=json.loads((R/'owner_seams.json').read_text())['objects'];full=fit(objects)
    loo={o['id']:fit([q for q in objects if q['id']!=o['id']]) for o in objects if o['id'] in ['seam3','seam4','seam5','seam6','seam7','seam8']}
    out={'assumptions':['corridor seam families parallel and mutually perpendicular','principal point fixed at image center','fx=fy','old normalized distortion held fixed'], 'tile_square_assumed':False,'tile_spacing_assumed':False,'metric_distance_estimated':False,'full_fit':full,'leave_one_seam_out':loo,'new_calibration_accepted':False,'reason':'One orthogonality constraint cannot independently identify focal length, distortion and principal point. Sensitivity only.'}
    (R/'owner_seams_intrinsics_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
