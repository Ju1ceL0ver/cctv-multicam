"""Camera 1: inside/outside, two classes (doorway relabelled by outcome). Same RGB+mask API as InsideOutside.
Features = rich (mask/floor/geometry, 234) + RGB context around head/visible-bottom/foot (648)."""
from pathlib import Path
import numpy as np
from inside_outside import InsideOutside
import cv2
from inside_outside import features

def context_features(f,g):
    m=f[:,:,3]/255.;r,c=np.where(m>.25)
    if not len(r):raise ValueError('Empty person mask')
    top,bottom=r.min(),r.max();v=[];rgb=f[:,:,:3]/255.
    centers=[(float(c[r<=top+max(1,int((bottom-top)*.2))].mean()),float(top)),(float(c[r>=bottom-2].mean()),float(bottom)),(g[6]*320,g[7]*180)]
    for cx,cy in centers:
        for radius in [4,10,20,40]:
            a,b=max(0,int(cx-radius)),max(0,int(cy-radius));e,d=min(320,int(cx+radius+1)),min(180,int(cy+radius+1));patch=rgb[b:d,a:e];mask=m[b:d,a:e]
            v+=cv2.resize(patch,(4,4),interpolation=cv2.INTER_AREA).ravel().tolist()
            bg=mask<.1;v+=patch[bg].mean(0).tolist() if bg.any() else [0]*3
            v+=patch[bg].std(0).tolist() if bg.any() else [0]*3
    return np.asarray(v,np.float32)

def binary_features(full,crop_mask,geometry,signed=None):
    return np.concatenate([features(full,crop_mask,geometry,signed),context_features(full,geometry)]).astype(np.float32)

class BinaryDoorClassifier(InsideOutside):
    def __init__(self,model_path=None,threshold=.9):
        super().__init__(model_path or Path(__file__).with_name('binary_door.pkl'),threshold)

    def _features(self,full,crop_mask,geometry,signed=None):
        return binary_features(full,crop_mask,geometry,signed)
