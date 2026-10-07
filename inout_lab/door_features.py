"""Three-class fixed-camera features, including RGB context under occlusion."""
import numpy as np
import cv2
from inside_outside import features

def expanded_features(f,crop_mask,g,signed=None):
    m=f[:,:,3]/255.;r,c=np.where(m>.25)
    if not len(r):raise ValueError('Empty person mask')
    top,bottom=r.min(),r.max();v=[]
    for q in [.05,.1,.15,.25,.4,.6,.8,1.]:
        cut=top+max(1,int((bottom-top+1)*q));rr,cc=np.where((m>.25)&(np.indices(m.shape)[0]<cut));weights=m[rr,cc];a=max(1,weights.sum())
        v += [(cc*weights).sum()/a/320,(rr*weights).sum()/a/180,len(rr)/m.size]
        v+=list(np.percentile(cc,[0,10,50,90,100])/320) if len(cc) else [0]*5
    for q in [0,10,25,50,75,90,100]:v += [np.percentile(r,q)/180,np.percentile(c,q)/320]
    for row in np.array_split(np.arange(top,bottom+1),12):
        rr,cc=np.where(m[row]>.25);v += [len(cc)/m.size,float(cc.mean()/320) if len(cc) else 0,float(cc.min()/320) if len(cc) else 0,float(cc.max()/320) if len(cc) else 0]
    v+=cv2.resize(m,(32,18),interpolation=cv2.INTER_AREA).ravel().tolist()
    rgb=f[:,:,:3]/255.;centers=[(float(c[r<=top+max(1,int((bottom-top)*.2))].mean()),float(top)),(float(c[r>=bottom-2].mean()),float(bottom)),(g[6]*320,g[7]*180)]
    for cx,cy in centers:
        for radius in [4,10,20,40]:
            a,b=max(0,int(cx-radius)),max(0,int(cy-radius));e,d=min(320,int(cx+radius+1)),min(180,int(cy+radius+1));patch=rgb[b:d,a:e];mask=m[b:d,a:e]
            v+=cv2.resize(patch,(4,4),interpolation=cv2.INTER_AREA).ravel().tolist()
            bg=mask<.1;v+=patch[bg].mean(0).tolist() if bg.any() else [0]*3
            v+=patch[bg].std(0).tolist() if bg.any() else [0]*3
    return np.concatenate([features(f,crop_mask,g,signed),np.asarray(v,np.float32)])
