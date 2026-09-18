"""Who exactly got mixed together, and where they were when it happened."""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera
from fusion2 import build, fuse
from clipdata import load
from eval_ids import evaluate

clip = sys.argv[1]
dets, feats, meta, embs = load(clip)
cams = {c: Camera(c, json.load(open('data/calib_final.json'))) for c in ('cam1', 'cam2')}
per_cam, _ = build(cams, dets, feats, embs, meta.get('clean'))
items = per_cam['cam1'] + per_cam['cam2']
segs, people, _ = fuse(per_cam)
label = {'cam1': {}, 'cam2': {}}
for pid, p in enumerate(people):
    for si in p['segments']:
        for m in segs[si]['members']:
            it = items[m]
            for di in it['det']:
                label[it['cam']][int(di)] = pid
gt = json.load(open(os.path.join('data', 'raw_clips', clip, 'gt_identity_yolo26x-seg.json')))
rows, errors = evaluate(label, os.path.join('data', 'raw_clips', clip, 'gt_identity_yolo26x-seg.json'), verbose=True)
print()
for pid, c in sorted(errors, key=lambda e: -sum(e[1].values())):
    tot = sum(c.values()); minor = tot - max(c.values())
    print('id %s mixes %s (%d detections, %d of them wrong)' % (pid, c, tot, minor))
    times = {}
    for cam in ('cam1', 'cam2'):
        for di, p in label[cam].items():
            if p != pid:
                continue
            g = gt[cam].get(str(di))
            if g is None:
                continue
            t = float(dets[cam][di, 0])
            times.setdefault(g, []).append(t)
    for g, ts in sorted(times.items()):
        print('    %s: %.0f-%.0f s, %d dets' % (g, min(ts), max(ts), len(ts)))
