"""Camera 1 binary classifier. RGB uint8 + one person's binary mask; CPU only."""
from pathlib import Path
import pickle
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

POLYGON = [(310,505),(365,490),(625,352),(690,340),(640,430),(632,600),(720,615),(860,740),(1000,770),(1110,795),(1245,690),(1255,600),(1250,450),(1350,430),(1365,580),(1560,600),(1560,690),(1260,735),(1175,900),(660,900)]

def features(full, crop_mask, geometry, signed=None):
    """Prepared export format: full 180x320 RGB+mask+floor, crop mask 160x96."""
    m=full[:,:,3]/255.; floor=full[:,:,4]/255.; binary=m>.25
    rr,cc=np.where(binary)
    if not len(rr):
        raise ValueError('Person mask is empty or too small after resizing')
    area=max(1,m.sum()); yy,xx=np.indices(m.shape)
    cx=(m*xx).sum()/area; cy=(m*yy).sum()/area
    if signed is None:
        signed=distance_transform_edt(floor>.5)-distance_transform_edt(floor<=.5)
    row0,row1=rr.min(),rr.max(); col0,col1=cc.min(),cc.max()
    v=[area/m.size,cx/320,cy/180,row0/180,row1/180,col0/320,col1/320,(m*floor).sum()/area]
    v+=list(np.percentile(signed[binary],[0,10,25,50,75,90,100])/180)
    for fraction in [.05,.1,.2,.35,.5]:
        cut=row1-max(1,int((row1-row0+1)*fraction)); low=m.copy(); low[:cut]=0; a=max(1,low.sum())
        v += [(low*floor).sum()/a,(low*signed).sum()/a/180,(low*xx).sum()/a/320]
    v+=cv2.resize(m,(16,8),interpolation=cv2.INTER_AREA).ravel().tolist()
    v+=cv2.resize(crop_mask/255.,(8,8),interpolation=cv2.INTER_AREA).ravel().tolist()
    rgb=full[:,:,:3]/255.;v+=((rgb*m[:,:,None]).sum((0,1))/area).tolist()
    return np.asarray(list(geometry)+v,dtype=np.float32)

class InsideOutside:
    def __init__(self, model_path=None, threshold=.8):
        # Load only trusted local model files (pickle).
        with open(model_path or Path(__file__).with_name('inside_outside.pkl'),'rb') as f:
            self.model=pickle.load(f)
        self.model.n_jobs=1
        self.threshold=threshold
        self._prepare_floor()

    def _prepare_floor(self):
        original=np.zeros((1440,2560),np.uint8)
        cv2.fillPoly(original,[np.asarray(POLYGON,dtype=float).__mul__(1.6).astype(np.int32)],255)
        self.floor=cv2.resize(original,(1280,720),interpolation=cv2.INTER_NEAREST)
        self.small_floor=cv2.resize(self.floor.astype(np.float32),(320,180),interpolation=cv2.INTER_AREA).astype(np.uint8)
        inside=self.small_floor/255.>.5
        self.signed=distance_transform_edt(inside)-distance_transform_edt(~inside)
        self.floor_distance=cv2.distanceTransform(self.floor,cv2.DIST_L2,3)-cv2.distanceTransform(255-self.floor,cv2.DIST_L2,3)

    def predict_prepared(self, full, crop_mask, geometry):
        return self._predict_inputs(full,crop_mask,geometry)

    def _predict_inputs(self,full,crop_mask,geometry,signed=None,crop_rgb=None,crop_floor=None):
        return self._predict(self._features(full,crop_mask,geometry,signed))

    def _features(self,full,crop_mask,geometry,signed=None):
        return features(full,crop_mask,geometry,signed)

    def _predict(self,a):
        p=self.model.predict_proba(a[None])[0]; side=int(p.argmax())
        return dict(label=('outside','inside')[side], confidence=float(p[side]),
                    uncertain=bool(p[side]<self.threshold), p_inside=float(p[1]))

    def predict_rgb(self, frame, person_mask, *, box=None, foot=None, floor_px=None):
        """box/foot in source-frame pixels. Optional original tracker geometry preferred.
        Mask must isolate ONE person; nonzero values are foreground. Camera 1 only.
        uncertain is not a learned doorway class; confidence is a forest score.
        """
        if frame.ndim!=3 or frame.shape[2]!=3 or frame.dtype!=np.uint8:
            raise ValueError('frame must be HxWx3 RGB uint8')
        if person_mask.shape!=frame.shape[:2]:
            raise ValueError('mask shape must equal frame shape')
        h,w=person_mask.shape; sx,sy=1280/w,720/h
        rgb=cv2.resize(frame,(1280,720),interpolation=cv2.INTER_LINEAR)
        mask=cv2.resize((person_mask!=0).astype(np.uint8)*255,(1280,720),interpolation=cv2.INTER_NEAREST)
        rr,cc=np.where(mask)
        if not len(rr): raise ValueError('Empty person mask')
        if box is None: x1,y1,x2,y2=cc.min(),rr.min(),cc.max()+1,rr.max()+1
        else: x1,y1,x2,y2=np.asarray(box)*[sx,sy,sx,sy]
        if foot is None:
            _,lab,stats,_=cv2.connectedComponentsWithStats(mask)
            blob=lab==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA])); br,bc=np.where(blob)
            fy=float(br.max());fx=float(bc[br>=fy-2].mean())
        else: fx,fy=np.asarray(foot)*[sx,sy]
        if floor_px is None: floor_px=round(float(self.floor_distance[int(np.clip(fy,0,719)),int(np.clip(fx,0,1279))]),1)
        g=[x1/1280,y1/720,x2/1280,y2/720,(x2-x1)/1280,(y2-y1)/720,fx/1280,fy/720,floor_px/100]
        a,b=max(0,int(x1-(x2-x1)/2)),max(0,int(y1-(y2-y1)*.125))
        c,d=min(1280,int(np.ceil(x2+(x2-x1)/2))),min(720,int(np.ceil(y2+(y2-y1)*.125)))
        if c<=a or d<=b: raise ValueError('Invalid box')
        small_rgb=cv2.resize(rgb.astype(np.float32),(320,180),interpolation=cv2.INTER_AREA).astype(np.uint8)
        small_mask=cv2.resize(mask.astype(np.float32),(320,180),interpolation=cv2.INTER_AREA).astype(np.uint8)
        crop=cv2.resize(mask[b:d,a:c].astype(np.float32),(96,160),interpolation=cv2.INTER_AREA).astype(np.uint8)
        full=np.dstack([small_rgb,small_mask,self.small_floor])
        crop_rgb=crop_floor=None
        if getattr(self,'requires_crop_rgb',False):
            crop_rgb=cv2.resize(rgb[b:d,a:c].astype(np.float32),(96,160),interpolation=cv2.INTER_AREA).astype(np.uint8)
            crop_floor=cv2.resize(self.floor[b:d,a:c].astype(np.float32),(96,160),interpolation=cv2.INTER_AREA).astype(np.uint8)
        return self._predict_inputs(full,crop,g,self.signed,crop_rgb,crop_floor)
