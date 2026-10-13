"""Synthetic visibility corruptions; target mask stays intact. No hidden GT invented."""
import numpy as np
import cv2
KINDS=['pole','double_pole','stand','top','left_edge','right_edge','bottom_edge','top_edge','diagonal','rectangle','person','multiple','irregular']

def augment_pair(rgb,mask,rng):
    h,w=mask.shape
    if rng.random()<.5:rgb=rgb[:,::-1].copy();mask=mask[:,::-1].copy()
    # Shrink/shift jointly, retaining the full silhouette inside the canvas.
    scale=rng.uniform(.72,1.0);nw,nh=max(1,int(w*scale)),max(1,int(h*scale));dx=int(rng.integers(0,w-nw+1));dy=int(rng.integers(0,h-nh+1))
    canvas=np.broadcast_to(np.median(rgb.reshape(-1,3),axis=0),(h,w,3)).copy();target=np.zeros_like(mask)
    canvas[dy:dy+nh,dx:dx+nw]=cv2.resize(rgb,(nw,nh));target[dy:dy+nh,dx:dx+nw]=cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_NEAREST)
    rgb=np.clip(canvas*rng.uniform(.7,1.3)*rng.uniform(.85,1.15,(1,1,3))+rng.uniform(-.08,.08),0,1)
    if rng.random()<.25:rgb=cv2.GaussianBlur(rgb,(3,3),rng.uniform(.3,1.0))
    if rng.random()<.4:rgb=np.clip(rgb+rng.normal(0,rng.uniform(.005,.025),rgb.shape),0,1)
    return rgb.astype(np.float32),target

def rich_occlude(rgb,mask,rng,kind=None):
    h,w=mask.shape;yy,xx=np.mgrid[:h,:w];ys,xs=np.where(mask>0)
    if not len(xs):raise ValueError('Empty mask')
    name=kind or KINDS[int(rng.integers(len(KINDS)))];lo,hi=float(rng.uniform(.06,.18)),float(rng.uniform(.65,.9))
    def shape(name):
        cx=int(rng.choice(xs));cy=int(rng.choice(ys));bw=float(rng.uniform(.06,.4)*w)
        if name=='pole':return abs(xx-cx)<bw/2
        if name=='double_pole':return (abs(xx-cx)<bw/3)|(abs(xx-int(rng.choice(xs)))<bw/3)
        if name in ['stand','bottom_edge']:return yy>=cy
        if name in ['top','top_edge']:return yy<=cy
        if name=='left_edge':return xx<=cx
        if name=='right_edge':return xx>=cx
        if name=='diagonal':return abs(xx-cx-rng.uniform(-1,1)*(yy-cy))<bw
        if name=='rectangle':return (abs(xx-cx)<bw)&(abs(yy-cy)<rng.uniform(.1,.5)*h)
        if name=='person':
            bh=rng.uniform(.3,.7)*h;body=((xx-cx)/(bw*.8+1))**2+((yy-cy)/(bh*.5+1))**2<1
            head=(xx-cx)**2+(yy-(cy-bh*.65))**2<(bw*.42)**2
            legs=((abs(xx-(cx-bw*.3))<bw*.18)|(abs(xx-(cx+bw*.3))<bw*.18))&(yy>cy)&(yy<cy+bh)
            return body|head|legs
        if name=='multiple':return shape('pole')|shape('rectangle')
        points=np.array([[int(cx+rng.uniform(-.4,.4)*w),int(cy+rng.uniform(-.4,.4)*h)] for _ in range(7)],np.int32)
        a=np.zeros((h,w),np.uint8);cv2.fillPoly(a,[cv2.convexHull(points)],1);return a.astype(bool)
    for _ in range(80):
        occ=shape(name);fraction=float((occ&(mask>0)).sum()/mask.sum())
        if lo<=fraction<=hi:break
    else:
        occ=xx>=int(np.median(xs))
    # Edges represent missing image pixels; occluders get varied textures instead of one flat colour.
    style=int(rng.integers(4));fill=np.broadcast_to(rng.uniform(.05,.95,3),rgb.shape).copy()
    if 'edge' in name:fill[:]=rng.uniform(0,.15)
    elif style==1:fill=np.clip(fill+rng.normal(0,.12,rgb.shape),0,1)
    elif style==2:fill=np.clip(fill+np.sin(xx/rng.uniform(2,8))[...,None]*rng.uniform(.05,.25),0,1)
    elif style==3:fill=np.roll(rgb,(int(rng.integers(1,h)),int(rng.integers(1,w))),(0,1))
    result=rgb.copy();result[occ]=fill[occ]
    return result.astype(np.float32),(mask*(~occ)).astype(np.float32),occ
