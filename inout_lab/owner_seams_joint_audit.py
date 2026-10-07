"""Joint straight/parallel/orthogonal seam fit; only sensitivity, no deployment.
Rectangular directions do not require known tile ratio, equal spacing or metres.
"""
from pathlib import Path
import json,cv2,numpy as np
from scipy.optimize import least_squares
R=Path(__file__).parent
objects={o['id']:np.array(o['points'],float) for o in json.loads((R/'owner_seams.json').read_text())['objects']}
cal=json.loads((R/'diagnostics/calib_final_source.json').read_text())['cam1'];oldf=cal['K'][0][0]/2;oldk=cal['dist'][0]
audit=json.loads((R/'owner_seams_geometry.json').read_text());a=np.array(audit['corridor_direction_1']['ray']);b=np.array(audit['corridor_direction_2']['ray']);b-=a*(a@b);b/=np.linalg.norm(b);basis=np.column_stack([a,b,np.cross(a,b)]);rv=cv2.Rodrigues(basis)[0].ravel();initial=np.r_[np.log(oldf),oldk,rv]
groups=[['seam3','seam4','seam5'],['seam6','seam7','seam8']]

def residual(x,omit=None):
    f=np.exp(x[0]);K=np.array([[f,0,640],[0,f,360],[0,0,1.]])
    rot=cv2.Rodrigues(x[2:])[0];errors=[]
    for j,ids in enumerate(groups):
        vp=rot[:,j]
        for key in ids:
            if key==omit:continue
            p=cv2.undistortPoints(objects[key].reshape(-1,1,2),K,np.array([x[1],0,0,0,0.])).reshape(-1,2);center=np.r_[p.mean(0),1];line=np.cross(vp,center);line/=max(np.linalg.norm(line[:2]),1e-9);errors.extend((np.c_[p,np.ones(len(p))]@line*f).tolist())
    return np.array(errors)

def fit(omit=None):
    fit=least_squares(residual,initial,args=(omit,),bounds=([np.log(500),-.8,-np.inf,-np.inf,-np.inf],[np.log(2000),.3,np.inf,np.inf,np.inf]),max_nfev=1000,xtol=1e-10,ftol=1e-10,gtol=1e-10)
    x=fit.x;res=residual(x,omit);sing=np.linalg.svd(fit.jac,compute_uv=False);normal=cv2.Rodrigues(x[2:])[0][:,2]
    return {'focal_px_1280':float(np.exp(x[0])),'k1':float(x[1]),'fit_rms_px':float(np.sqrt(np.mean(res**2))),'fit_max_px':float(abs(res).max()),'normal':normal.tolist(),'jacobian_condition':float(sing[0]/sing[-1]),'bound_active':fit.active_mask.tolist(),'success':bool(fit.success),'radial_max_observed_radius_px':float(2*np.exp(x[0])/(3*np.sqrt(-3*x[1]))) if x[1]<0 else None,'image_corner_radius_px':float(np.hypot(640,360))}

if __name__=='__main__':
    out={'full':fit(),'leave_one_seam_out':{key:fit(key) for group in groups for key in group},'new_calibration_accepted':False,'fixed_principal_point':[640,360],'square_tiles_assumed':False,'spacing_assumed':False,'metric_scale_estimated':False,'warning':'Fit imposes rectangular corridor directions; uncertainty excludes annotation/systematic plane assumptions.'}
    (R/'owner_seams_joint_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
