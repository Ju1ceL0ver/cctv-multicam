"""Trainable association on cached owner TRAIN identities, with physical gates."""
import numpy as np
from door_seg_track import Config,Tracker,centre,shifted_iou


def features(a,b,gap):
    ca,cb=centre(a[1]),centre(b[1]);h=max(16.,min(a[1][3]-a[1][1],b[1][3]-b[1][1]))
    d=cb-ca;forward=np.asarray(a[4]);backward=np.asarray(b[3])
    iou_f=shifted_iou(a[0],b[0],forward)
    iou_b=shifted_iou(b[0],a[0],backward)
    ar=max(1,int(a[0].sum()));br=max(1,int(b[0].sum()))
    app=1-float(np.dot(a[5],b[5])) if len(a)>5 and len(b)>5 else .5
    return np.array([gap,np.linalg.norm(d)/h,np.linalg.norm(d-forward)/h,np.linalg.norm(d+backward)/h,
                     iou_f,iou_b,abs(np.log(br/ar)),a[2],b[2],app,
                     abs((b[1][3]-b[1][1])-(a[1][3]-a[1][1]))/h],np.float32)


class LearnedTracker(Tracker):
    def __init__(self,model,config=None,threshold=.8):
        from dataclasses import replace
        base = config or Config()
        super().__init__(replace(base, max_gap=min(base.max_gap, 6), max_cost=1-threshold, ambiguity=.05))
        self.model=model
    def cost_matrix(self,old,predictions,tick):
        result=np.full((len(old),len(predictions)),np.inf);items=[];xs=[]
        for i,tr in enumerate(old):
            gap=tick-tr.tick
            if gap > 6:
                continue
            for j,p in enumerate(predictions):
                if np.linalg.norm(centre(tr.pred[1])-centre(p[1]))>self.config.max_step*gap:continue
                f=features(tr.pred,p,gap)
                if f[1]>.5 and max(f[4],f[5])<.2:continue
                xs.append(f);items.append((i,j))
        if xs:
            prob=self.model.predict_proba(np.array(xs))[:,1]
            for (i,j),v in zip(items,prob):result[i,j]=1-v
        return result
