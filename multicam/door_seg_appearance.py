"""Cheap masked colour descriptor for same-camera association experiments.

No recognition across long gaps: original geometry and maximum gap remain.
"""
import cv2
import numpy as np
from door_seg_track import Tracker


def descriptors(frame, predictions):
    hsv=cv2.cvtColor(cv2.resize(frame,(320,180),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2HSV)
    code=(hsv[:,:,0].astype(np.int32)*8//180)*16+(hsv[:,:,1].astype(np.int32)//64)*4+hsv[:,:,2].astype(np.int32)//64
    result=[]
    for p in predictions:
        mask=p[0][:180];ys,xs=np.nonzero(mask)
        vector=np.zeros(384,np.float32)
        if len(ys):
            bands=np.minimum(2,(ys-ys.min())*3//max(1,int(ys.max()-ys.min()+1)))
            h=np.bincount(bands*128+code[ys,xs],minlength=384).astype(np.float32).reshape(3,128)
            h/=np.maximum(h.sum(1,keepdims=True),1)
            vector=np.sqrt(h).flatten();vector/=max(np.linalg.norm(vector),1e-6)
        result.append((*p,vector))
    return result


class AppearanceTracker(Tracker):
    def __init__(self,config=None,weight=.4):
        super().__init__(config);self.weight=weight
    def cost(self,tr,p,tick):
        base=super().cost(tr,p,tick)
        if not np.isfinite(base) or len(p)<6 or len(tr.pred)<6:return base
        a,b=tr.pred[5],p[5]
        if not a.any() or not b.any():return base
        distance=float(np.clip(1-np.dot(a,b),0,1))
        return (1-self.weight)*base+self.weight*distance
