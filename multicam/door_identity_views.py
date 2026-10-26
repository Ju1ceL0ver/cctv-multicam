"""Immutable, event-scoped appearance views. Never use a track-number folder as a person's identity."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from storage import atomic_json, read_json


def legacy_person(frame, kind):
    """Recover the explicitly coloured subject of an old event video (640x360).

    Reject frames without a clear person outline. The colour overlay is inverted
    inside the recovered mask; title and tracker text are excluded.
    """
    col = np.array((60,220,60) if kind=='in' else (30,150,255),np.float32)
    delta = np.max(np.abs(frame.astype(np.float32)-col),axis=2)
    line = (delta<48).astype(np.uint8)
    line[:28]=0
    line=cv2.morphologyEx(line,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(line,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour);area=cv2.contourArea(contour)
        if area<120 or h<42 or w<8 or w>1.3*h or area/max(1,w*h)<.12:
            continue
        candidates.append((area,contour,(x,y,x+w,y+h)))
    if not candidates:
        return None
    area,contour,box=max(candidates,key=lambda q:q[0])
    mask=np.zeros(frame.shape[:2],np.uint8);cv2.drawContours(mask,[contour],-1,1,-1)
    mask=cv2.erode(mask,np.ones((3,3),np.uint8))>0
    restored=np.clip((frame.astype(np.float32)-.45*col)/.55,0,255).astype(np.uint8)
    clean=np.full_like(frame,128);clean[mask]=restored[mask]
    x1,y1,x2,y2=box;pad=2
    crop=clean[max(0,y1-pad):min(frame.shape[0],y2+pad),max(0,x1-pad):min(frame.shape[1],x2+pad)]
    # The old clips can be too small to identify: do not upscale them into fake evidence.
    return crop,box,float(area)


def save_views(folder, event, items, source):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for i,item in enumerate(items):
        crop=item['image'];path=folder/('view_%02d.jpg'%i)
        ok,buf=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok:continue
        payload=buf.tobytes();path.write_bytes(payload)
        records.append(dict(path=path.name,sha256=hashlib.sha256(payload).hexdigest(),
                            box=[float(x) for x in item['box']],at=float(item['at']),score=float(item['score'])))
    manifest=dict(event={k:event.get(k) for k in ('kind','t','clock','w')},source=source,views=records)
    atomic_json(folder/'manifest.json',manifest)
    return manifest


def from_legacy(path, folder, event, keep=4):
    manifest=read_json(Path(folder)/'manifest.json',{})
    if manifest.get('source')=='legacy_clip_inverse_v1' and manifest.get('views'):
        return manifest
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates=[]
    try:
        if fps<=0:return save_views(folder,event,[],'legacy_clip_inverse_v1')
        for index in range(0,n,max(1,int(round(fps/2)))):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
            if not ok:continue
            got=legacy_person(frame,event['kind'])
            if got is not None:
                crop,box,area=got
                sharp=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_32F).var())
                candidates.append(dict(image=crop,box=box,at=index/fps,score=area*min(2,max(.2,sharp/100))))
    finally:cap.release()
    selected=[]
    if candidates:
        # One clear view per temporal quarter, instead of four adjacent copies.
        for block in np.array_split(np.arange(len(candidates)),keep):
            if len(block): selected.append(max((candidates[k] for k in block),key=lambda q:q['score']))
    return save_views(folder,event,selected,'legacy_clip_inverse_v1')


def from_tape(folder,event,part,keep=8):
    candidates=[];pid=str(event['w'])
    for t,jpg,masks in part:
        if abs(t-event['t'])>8:continue
        match=next(((packed,shape) for p,packed,shape in masks if str(p)==pid),None)
        if match is None:continue
        packed,shape=match;m=np.unpackbits(packed)[:shape[0]*shape[1]].reshape(shape).astype(np.uint8)
        ys,xs=np.nonzero(m)
        if not len(xs):continue
        h,w=shape
        full=cv2.imdecode(np.frombuffer(jpg,np.uint8),cv2.IMREAD_COLOR)
        mask=cv2.resize(m,(full.shape[1],full.shape[0]),interpolation=cv2.INTER_NEAREST)>0
        sy,sx=full.shape[0]/h,full.shape[1]/w
        box=[xs.min()*sx,ys.min()*sy,(xs.max()+1)*sx,(ys.max()+1)*sy]
        if box[3]-box[1]<75:continue
        area=float(mask.sum());cut=box[1]<=2 or box[3]>=full.shape[0]-2
        overlap=False
        for p,pb,ps in masks:
            if str(p)==pid:continue
            other=np.unpackbits(pb)[:ps[0]*ps[1]].reshape(ps)>0
            oy,ox=np.nonzero(other)
            if len(ox) and min(xs.max(),ox.max())>max(xs.min(),ox.min()) and min(ys.max(),oy.max())>max(ys.min(),oy.min()):overlap=True
        score=area*(.2 if cut else 1)*(.15 if overlap else 1)
        full[~mask]=128
        x1,y1,x2,y2=map(int,box);crop=full[max(0,y1-3):min(full.shape[0],y2+3),max(0,x1-3):min(full.shape[1],x2+3)]
        candidates.append(dict(image=crop,box=box,at=t,score=score))
    selected=[]
    if candidates:
        for block in np.array_split(np.arange(len(candidates)),keep):
            if len(block):selected.append(max((candidates[k] for k in block),key=lambda q:q['score']))
    return save_views(folder,event,selected,'tape_mask_v1')


def snapshot_views(event, folder, segmenter=None):
    """Historical events without video: preserve the event's own photograph, never views_seg."""
    existing=read_json(Path(folder)/'manifest.json',{})
    if existing.get('source')=='event_snapshot_v1' and (segmenter is None or existing.get('mask_attempted')):return existing
    path=Path(event.get('crop') or '')
    crop=cv2.imread(str(path)) if path.is_file() else None
    if crop is None or min(crop.shape[:2])<12:return save_views(folder,event,[],'event_snapshot_v1')
    items=[dict(image=crop,box=[0,0,crop.shape[1],crop.shape[0]],at=event['t'],score=1.)]
    variants=['raw']
    if segmenter is not None:
        result=segmenter.predict(crop,imgsz=640,classes=[0],conf=.35,device=0,verbose=False,retina_masks=True)[0]
        if result.masks is not None and len(result.masks.data):
            masks=result.masks.data.cpu().numpy();j=max(range(len(masks)),key=lambda k:float(masks[k].sum()))
            mask=cv2.resize(masks[j],(crop.shape[1],crop.shape[0]),interpolation=cv2.INTER_NEAREST)>.5
            if mask.mean()>=.08:
                image=crop.copy();image[~mask]=128
                items.append(dict(image=image,box=items[0]['box'],at=event['t'],score=float(mask.mean())))
                variants.append('masked')
    result=save_views(folder,event,items,'event_snapshot_v1')
    result['mask_attempted']=segmenter is not None
    for view,variant in zip(result['views'],variants):view['variant']=variant
    atomic_json(Path(folder)/'manifest.json',result)
    return result
