"""Fixed 400-moment teacher validation with four ordered recurrent frames.

The already viewed 23.09 cohort is a proxy, not a new blind test. Thresholds
are unchanged; fingerprint retrieval is reported, never plugged into tracking.
"""
import random
import time

import cv2
import numpy as np
import torch

import door_seg as S
from door_seg_sequence import sequence_starts
from door_seg_temporal import pool_prediction_masks


def metrics(tot, times=()):
    f = tot['hit']/max(1, tot['truth']); p = tot['hit']/max(1, tot['pred'])
    return {'found': f, 'precision': p, 'f1': 2*f*p/max(1e-9, f+p),
            'mask_iou': tot['iou']/max(1, tot['hit']),
            'links': tot['links_ok']/max(1, tot['links']), 'n_links': tot['links'],
            'truth': tot['truth'], 'pred': tot['pred'], 'hit': tot['hit'],
            'fingerprint_top1': tot['finger_ok']/max(1, tot['finger_n']),
            'fingerprint_pairs': tot['finger_n'],
            'ms_per_frame': 1000*float(np.median(times)) if times else None}


def evaluate(net, dirs, *, n=400, guard=lambda: None, seed=7, progress=lambda x: None):
    srcs = []; times = []; rows = []
    keys = ('truth', 'pred', 'hit', 'iou', 'links', 'links_ok', 'finger_n', 'finger_ok')
    tot = dict.fromkeys(keys, 0)
    try:
        for d in dirs:
            guard(); src = S.Source(d); srcs.append(src)
        starts = [sequence_starts(src, 4) for src in srcs]
        if any(not x for x in starts):
            raise ValueError('frozen VAL source has no four-frame sequence')
        rng = random.Random(seed)
        picks = [(i, rng.choice(starts[i])) for i in rng.choices(range(len(srcs)), k=n)]
        net.yolo.eval(); net.off.eval(); net.temporal.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for index, (i, (session, first)) in enumerate(picks):
                guard(); src = srcs[i]; net.reset_memory()
                cache = src.read(first-max(S.PAST), first+4+max(S.FUT))
                if any(k not in cache for k in range(first-max(S.PAST), first+4+max(S.FUT))):
                    raise RuntimeError('incomplete frozen VAL context')
                before = src.people(session, first-1)
                prev = np.bitwise_or.reduce([m for m, _ in before.values()]) if before else np.zeros((S.H,S.W), np.uint8)
                pair = []
                for j in range(4):
                    guard(); tick = first+j
                    image = S.channels(src.frames(tick, cache), S.background(src.day,src.cam), prev)
                    net.prepare_frame([(str(src.d),session)], [tick])
                    torch.cuda.synchronize(); begin=time.monotonic()
                    predictions = S.predict(net, torch.from_numpy(image)[None].cuda().float()/255)
                    torch.cuda.synchronize()
                    if j == 3: times.append(time.monotonic()-begin)
                    if j >= 2:
                        masks = torch.from_numpy(np.stack([p[0] for p in predictions])).cuda() if predictions else torch.empty((0,S.MH,S.MW),device='cuda')
                        vec, valid = pool_prediction_masks(net.embedding_map, masks)
                        pair.append((predictions, vec.cpu().numpy(), valid.cpu().numpy()))
                    prev = np.zeros((S.H,S.W),np.uint8)
                    for mask, *_ in predictions:
                        prev |= cv2.resize(mask[:S.H//4].astype(np.uint8),(S.W,S.H),interpolation=cv2.INTER_NEAREST)
                (p0,v0,ok0),(p1,v1,ok1)=pair; k=first+3
                truth1,truth0=src.people(session,k),src.people(session,k-1)
                ids1,ids0=list(truth1),list(truth0)
                def small(m):
                    return np.pad(cv2.resize(m,(S.MW,S.H//4),interpolation=cv2.INTER_NEAREST)>0,((0,S.MH-S.H//4),(0,0)))
                t1=[small(truth1[key][0]) for key in ids1];t0=[small(truth0[key][0]) for key in ids0]
                m1=S._match([p[0] for p in p1],t1);m0=S._match([p[0] for p in p0],t0)
                row=dict.fromkeys(keys,0);row.update(truth=len(t1),pred=len(p1),hit=len(m1),iou=sum(p[2] for p in m1))
                who0={a:ids0[b] for a,b,_ in m0}
                centers=np.array([[(p[1][0]+p[1][2])/2,(p[1][1]+p[1][3])/2] for p in p0])
                for a,b,_ in m1:
                    if ids1[b] not in ids0 or not p0: continue
                    box=p1[a][1];center=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])+p1[a][3]
                    nearest=int(np.argmin(((centers-center)**2).sum(1)))
                    row['links']+=1;row['links_ok']+=int(who0.get(nearest)==ids1[b])
                    if ok1[a] and ok0.any():
                        candidates=np.flatnonzero(ok0)
                        chosen=int(candidates[np.argmax(v0[candidates]@v1[a])])
                        row['finger_n']+=1;row['finger_ok']+=int(who0.get(chosen)==ids1[b])
                for key in keys: tot[key]+=row[key]
                rows.append({'source':str(src.d),'session':session,'tick':k,**row})
                if (index+1)%50==0: progress({'moments':index+1,**metrics(tot,times)})
        return {'protocol':'fixed seed7, four ordered frames per moment, own previous masks, BF16 AMP, teacher23.09 proxy',
                'moments':len(rows),**metrics(tot,times),'per_moment':rows}
    finally:
        net.reset_memory()
        for src in srcs: src.close()
