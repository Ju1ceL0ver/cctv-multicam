"""One number per configuration, on the labelled clip: how much of each real person's
time carries a single id, and how many detections sit under an id shared with someone else."""
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
gt_path = os.path.join('data', 'raw_clips', clip, 'gt_identity_yolo26x-seg.json')
rows, errors = evaluate(label, gt_path, verbose=False)
tot = sum(r['detections'] for r in rows.values())
weighted = sum(r['main_id_share'] * r['detections'] for r in rows.values()) / tot
wrong = sum(sum(c.values()) - max(c.values()) for _, c in errors)
main = {g: r for g, r in rows.items() if r['detections'] > 2000}
print('CONFIG %s | pieces %d | people %d | weighted one-id %.3f | wrong dets %d (%.1f%%) | main: %s'
      % (os.environ.get('RA_TAG', ''), len(items), len(people), weighted, wrong, 100 * wrong / tot,
         ', '.join('%s %.2f/%d' % (g, r['main_id_share'], r['ids']) for g, r in sorted(main.items()))))
