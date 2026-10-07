"""Metric pose and pixel-area density of measured static rectangles.
No person-depth or classifier accuracy claims follow from a plane fit alone.
Uses already installed OpenCV; existing intrinsics/distortion remain a prior.
"""
from pathlib import Path
import argparse,json
import cv2
import numpy as np

ROOT=Path(__file__).parent

def validate_plane(plane,image_size):
    try:
        size=np.asarray(plane['size_m'],dtype=float)
        corners=np.asarray(plane['image_corners_px'],dtype=float)
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError('Enter real dimensions in metres and four image corners') from error
    if size.shape!=(2,) or not np.isfinite(size).all() or np.any(size<=0):
        raise ValueError('size_m must contain two known positive lengths in metres')
    if corners.shape!=(4,2) or not np.isfinite(corners).all():
        raise ValueError('image_corners_px must contain four real [x,y] pairs')
    if np.any(corners<0) or np.any(corners>=np.asarray(image_size)):
        raise ValueError('Corner coordinates lie outside the stated source image')
    if not cv2.isContourConvex(corners.astype(np.float32)) or abs(cv2.contourArea(corners.astype(np.float32)))<16:
        raise ValueError('Mark adjacent corners around a non-degenerate rectangle, not crossed diagonals')
    return size,corners

def pixel_area_density(points_xy_m,rvec,tvec,K,dist):
    """Local projected px²/m² on this known plane, including lens distortion."""
    xy=np.asarray(points_xy_m,float).reshape(-1,2)
    points=np.column_stack([xy,np.zeros(len(xy))]);step=1e-4
    offsetx=np.array([step,0,0]);offsety=np.array([0,step,0])
    project=lambda p:cv2.projectPoints(p,rvec,tvec,K,dist)[0].reshape(-1,2)
    dx=(project(points+offsetx)-project(points-offsetx))/(2*step)
    dy=(project(points+offsety)-project(points-offsety))/(2*step)
    return np.abs(dx[:,0]*dy[:,1]-dx[:,1]*dy[:,0])

def fit_plane(plane,image_size,K,dist):
    size,corners=validate_plane(plane,image_size);w,h=size
    obj=np.array([[0,0,0],[w,0,0],[w,h,0],[0,h,0]],float)
    result=cv2.solvePnPGeneric(obj,corners,K,dist,flags=cv2.SOLVEPNP_IPPE)
    if not result[0]:raise ValueError('No planar pose solution')
    candidates=[]
    for rv,tv in zip(result[1],result[2]):
        R=cv2.Rodrigues(rv)[0];camera_points=obj@R.T+tv.reshape(1,3)
        if np.any(camera_points[:,2]<=0):continue
        projected=cv2.projectPoints(obj,rv,tv,K,dist)[0].reshape(-1,2)
        errors=np.linalg.norm(projected-corners,axis=1);C=-R.T@tv.reshape(3)
        center=R@np.array([w/2,h/2,0])+tv.reshape(3)
        checks=[]
        for point in plane.get('check_points',[]):
            local=np.asarray(point['plane_xy_m'],float);observed=np.asarray(point['image_px'],float)
            if local.shape!=(2,) or observed.shape!=(2,) or not np.isfinite(local).all() or not np.isfinite(observed).all():
                raise ValueError('Each check point needs finite plane_xy_m and image_px pairs')
            predicted=cv2.projectPoints(np.array([[*local,0]],float),rv,tv,K,dist)[0].reshape(2)
            checks.append({'observed_px':observed.tolist(),'predicted_px':predicted.tolist(),'error_px':float(np.linalg.norm(predicted-observed))})
        samples=np.array([[w/2,h/2],[w*.1,h*.1],[w*.9,h*.1],[w*.9,h*.9],[w*.1,h*.9]])
        candidates.append({'rvec':rv.reshape(3).tolist(),'tvec_m':tv.reshape(3).tolist(),'camera_in_plane_coordinates_m':C.tolist(),'perpendicular_camera_to_plane_m':float(abs(C[2])),'camera_to_rectangle_center_m':float(np.linalg.norm(center)),'corner_rmse_px':float(np.sqrt(np.mean(errors**2))),'corner_errors_px':errors.tolist(),'independent_checks':checks,'density_samples':{'plane_xy_m':samples.tolist(),'pixels_squared_per_metre_squared':pixel_area_density(samples,rv,tv,K,dist).tolist()}})
    if not candidates:raise ValueError('All pose candidates put rectangle corners behind camera')
    candidates.sort(key=lambda p:p['corner_rmse_px'])
    return {'name':plane['name'],'size_m':size.tolist(),'primary':candidates[0],'alternatives':candidates[1:],'planar_ambiguity_unresolved':len(candidates)>1 and candidates[1]['corner_rmse_px']-candidates[0]['corner_rmse_px']<.5,'independently_checked':bool(plane.get('check_points'))}

def solve_document(document,calibration):
    size=np.asarray(document['image_size_px'],float);original=np.asarray(document['calibration_image_size_px'],float)
    if size.shape!=(2,) or original.shape!=(2,) or not np.isfinite(size).all() or not np.isfinite(original).all() or np.any(size<=0) or np.any(original<=0):raise ValueError('Invalid image dimensions')
    cal=calibration[document.get('camera','cam1')];K=np.asarray(cal['K'],float).copy();K[0,:]*=size[0]/original[0];K[1,:]*=size[1]/original[1];dist=np.asarray(cal['dist'],float)
    if not document.get('planes'):raise ValueError('No measured rectangles provided')
    return {'camera':document.get('camera','cam1'),'image_size_px':size.tolist(),'K_scaled':K.tolist(),'distortion_prior':dist.tolist(),'units':'metres; plane density px²/m²','intrinsics_reestimated':False,'person_depth_estimated':False,'planes':[fit_plane(p,size,K,dist) for p in document['planes']]}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('annotation',type=Path);parser.add_argument('--calibration',type=Path,default=ROOT/'diagnostics/calib_final_source.json');parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try:result=solve_document(json.loads(args.annotation.read_text()),json.loads(args.calibration.read_text()))
    except (ValueError,KeyError) as error:parser.error(str(error))
    args.out.write_text(json.dumps(result,indent=2));print('Saved',args.out)
