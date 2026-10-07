"""Synthetic sensitivity only; never estimates the real shop distance."""
import json
from pathlib import Path
import numpy as np
import cv2
from plane_reference import fit_plane
ROOT=Path(__file__).parent
cal=json.loads((ROOT/'diagnostics/calib_final_source.json').read_text())['cam1']
K=np.asarray(cal['K'],float);K[:2]*=.5;dist=np.asarray(cal['dist'],float)
rng=np.random.default_rng(20261007);report={'synthetic_only':True,'real_camera_distance_measured':False,'cases':[]}
for name,rv,tv in [('wall', [.1,.25,-.05],[-1,-.5,5]),('oblique_floor',[1.2,.05,.1],[-1,-.3,4])]:
    rv=np.asarray(rv,float);tv=np.asarray(tv,float)
    obj=np.array([[0,0,0],[2,0,0],[2,1.5,0],[0,1.5,0]],float)
    corners=cv2.projectPoints(obj,rv,tv,K,dist)[0].reshape(4,2)
    true_distance=abs((-cv2.Rodrigues(rv)[0].T@tv)[2])
    case={'name':name,'true_perpendicular_distance_m':float(true_distance),'corner_noise':[],'common_size_scale':[]}
    for sigma in [.5,1,2,4]:
        errors=[];ambiguous=0;rejected=0
        for _ in range(200):
            p={'name':name,'size_m':[2,1.5],'image_corners_px':(corners+rng.normal(0,sigma,(4,2))).tolist()}
            try:fit=fit_plane(p,[1280,720],K,dist)
            except ValueError:rejected+=1;continue
            errors.append(abs(fit['primary']['perpendicular_camera_to_plane_m']/true_distance-1)*100)
            ambiguous+=fit['planar_ambiguity_unresolved']
        case['corner_noise'].append({'sigma_px':sigma,'median_absolute_distance_error_percent':float(np.median(errors)),'p95_absolute_distance_error_percent':float(np.percentile(errors,95)),'ambiguous':ambiguous,'rejected':rejected})
    for scale in [.9,1,1.1]:
        fit=fit_plane({'name':name,'size_m':[2*scale,1.5*scale],'image_corners_px':corners.tolist()},[1280,720],K,dist)
        case['common_size_scale'].append({'scale':scale,'estimated_distance_m':fit['primary']['perpendicular_camera_to_plane_m']})
    report['cases'].append(case)
(ROOT/'plane_reference_sensitivity.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
