"""Read-only CPU benchmark against reviewed identities; no video/model inference."""
import argparse
import json
import os
import sys
import types
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
from scipy.optimize import linear_sum_assignment


def score(root, clip, sources=None):
    root=Path(root); os.chdir(root); sys.path.insert(0,str(root))
    if sources:
        for name in ('imtrack','clipdata','fusion2'):
            module=types.ModuleType(name); module.__file__=str(root/(name+'.py'))
            sys.modules[name]=module
            exec(compile((Path(sources)/(name+'.py')).read_text(encoding='utf-8'), module.__file__, 'exec'),module.__dict__)
    from person3d import Camera
    from fusion2 import build, fuse
    from clipdata import load
    dets, feats, meta, embs=load(clip)
    cams={c:Camera(c,json.loads((root/'data/calib_final.json').read_text())) for c in dets}
    per_cam,_=build(cams,dets,feats,embs,meta.get('clean'))
    segs,people,_=fuse(per_cam);items=per_cam['cam1']+per_cam['cam2']
    labels={'cam1':{},'cam2':{}}
    for pid,p in enumerate(people):
        for si in p['segments']:
            for m in segs[si]['members']:
                item=items[m]
                for di in item['det']: labels[item['cam']][str(int(di))]=pid
    gt=json.loads((root/'data/raw_clips'/clip/'gt_identity_yolo26x-seg.json').read_text())
    per_person=defaultdict(Counter); per_id=defaultdict(Counter); switches=0
    for cam in gt:
        last={}
        for di,person in sorted(gt[cam].items(),key=lambda kv:float(dets[cam][int(kv[0]),0])):
            pid=labels[cam].get(di)
            per_person[person][pid]+=1
            if pid is not None:
                per_id[pid][person]+=1
                if person in last and last[person]!=pid: switches+=1
                last[person]=pid
    total=sum(sum(c.values()) for c in per_person.values())
    main=sum(max((n for k,n in c.items() if k is not None),default=0) for c in per_person.values())
    wrong=sum(sum(c.values())-max(c.values()) for c in per_id.values())
    old_wrong=sum(sum(c.values())-max(c.values()) for c in per_id.values() if (sum(c.values())-max(c.values()))/sum(c.values())>.05)
    gs=list(per_person); ps=list(per_id)
    mat=np.array([[per_id[p][g] for p in ps] for g in gs],float).reshape(len(gs),len(ps))
    r,c=linear_sum_assignment(-mat); tp=float(mat[r,c].sum())
    assigned=sum(sum(c.values()) for c in per_id.values())
    return {'clip':clip,'one_id':main/max(1,total),'wrong_all':wrong/max(1,total),
            'wrong_legacy':old_wrong/max(1,total),'idf1_on_teacher_detections':2*tp/max(1,total+assigned),
            'unassigned':(total-assigned)/max(1,total),'id_switches':switches,'pieces':len(items),
            'people':len(people),'sync_s':meta.get('cam1_offset_s',0),'gt_detections':total}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('clip');p.add_argument('--root',default=str(Path(__file__).resolve().parent));p.add_argument('--sources');a=p.parse_args()
    print(json.dumps(score(a.root,a.clip,a.sources)))
