"""Make tracker metadata/crop agree with the visible draft target mask.
Draft construction overlays nearer masks; saved tracker bbox precedes that
overlay. Correction uses no labels and retains every input row.
"""
from pathlib import Path
import json,time
import cv2
import numpy as np
from door_features import expanded_features
from crop_feature_search import crop_features

ROOT=Path(__file__).parent

def crop_bounds(g):
    x1,y1,x2,y2=g[:4]*[1280,720,1280,720];w=x2-x1;h=y2-y1
    return max(0,int(x1-w/2)),max(0,int(y1-h/8)),min(1280,int(np.ceil(x2+w/2))),min(720,int(np.ceil(y2+h/8)))

def align_sample(full,crop,g):
    rr,cc=np.where(full[:,:,3]>63)
    if not len(rr):return crop,g.copy(),False
    geometry=g.copy();geometry[:4]=[cc.min()/320,rr.min()/180,(cc.max()+1)/320,(rr.max()+1)/180]
    geometry[4:6]=geometry[2:4]-geometry[:2]
    old=crop_bounds(g);new=crop_bounds(geometry)
    a,b,c,d=old;A,B,C,D=new
    if c<=a or d<=b or C<=A or D<=B:return crop,g.copy(),False
    xx=(np.arange(96,dtype=np.float32)+.5)*(C-A)/96+A
    yy=(np.arange(160,dtype=np.float32)+.5)*(D-B)/160+B
    u=(xx-a)/(c-a)*96-.5;v=(yy-b)/(d-b)*160-.5
    uu,vv=np.meshgrid(u,v)
    # Quantized coordinate channels are regenerated, not interpolated twice.
    result=np.stack([cv2.remap(crop[:,:,j],uu,vv,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE) for j in range(5)],2)
    xcoords=cv2.resize(np.linspace(0,255,1280,dtype=np.float32)[None,A:C],(96,1),interpolation=cv2.INTER_AREA).astype(np.uint8)
    ycoords=cv2.resize(np.linspace(0,255,720,dtype=np.float32)[B:D,None],(1,160),interpolation=cv2.INTER_AREA).astype(np.uint8)
    result=np.dstack([result,np.broadcast_to(xcoords,(160,96)),np.broadcast_to(ycoords,(160,96))])
    return result,geometry,True

if __name__=='__main__':
    z=np.load(ROOT/'io_cam1.npz');F=z['F'];X=z['X'];G=z['G'][:,1:];rows=[];gs=[];changed=[];start=time.perf_counter()
    for i,(f,x,g) in enumerate(zip(F,X,G)):
        crop,geometry,ok=align_sample(f,x,g)
        rows.append(np.concatenate([expanded_features(f,crop[:,:,3],geometry),crop_features(crop)]));gs.append(geometry)
        if np.max(np.abs((geometry[:4]-g[:4])*[1280,720,1280,720]))>30:changed.append(i)
        if i in [970,1149]:
            cv2.imwrite(str(ROOT/f'diagnostics/aligned_{i}_crop.jpg'),crop[:,:,:3][:,:,::-1])
            cv2.imwrite(str(ROOT/f'diagnostics/before_{i}_crop.jpg'),x[:,:,:3][:,:,::-1])
        if (i+1)%256==0:print('bbox alignment',i+1,'seconds',round(time.perf_counter()-start),flush=True)
    A=np.asarray(rows,np.float32);np.save(ROOT/'diagnostics/aligned_features.npy',A);np.save(ROOT/'diagnostics/aligned_geometry.npy',np.asarray(gs,np.float32))
    assert A.shape==(1628,3288) and np.isfinite(A).all()
    report={'rows':len(A),'changed_over30px':len(changed),'changed_indices':changed,'seconds':time.perf_counter()-start,'labels_used':False,'original_files_modified':False}
    (ROOT/'visible_bbox_alignment_report.json').write_text(json.dumps(report,indent=2));print('ALIGNED',report,flush=True)
