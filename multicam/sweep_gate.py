"""How sensitive is the result to the clustering threshold? (and the full per-person table)"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import fusion2
from person3d import Camera
from clipdata import load
from eval_ids import evaluate
clip = sys.argv[1]
dets, feats, meta, embs = load(clip)
cams = {c: Camera(c, json.load(open('data/calib_final.json'))) for c in ('cam1', 'cam2')}
per_cam, _ = fusion2.build(cams, dets, feats, embs, meta.get('clean'))
items = per_cam['cam1'] + per_cam['cam2']
gt_path = os.path.join('data', 'raw_clips', clip, 'gt_identity_yolo26x-seg.json')
gt = json.load(open(gt_path))
big = {p for p in set(list(gt['cam1'].values()) + list(gt['cam2'].values()))
       if sum(v == p for v in list(gt['cam1'].values()) + list(gt['cam2'].values())) > 2000}
for gate in [float(x) for x in (sys.argv[2:] or ['0.18', '0.22', '0.26', '0.30', '0.34', '0.40'])]:
    fusion2.GATE_WITHIN = gate
    segs, people, _ = fusion2.fuse(per_cam)
    label = {'cam1': {}, 'cam2': {}}
    for pid, p in enumerate(people):
        for si in p['segments']:
            for m in segs[si]['members']:
                it = items[m]
                for di in it['det']:
                    label[it['cam']][int(di)] = pid
    rows, errors = evaluate(label, gt_path, verbose=False)
    main = [r for g, r in rows.items() if g in big]
    small = [r for g, r in rows.items() if g not in big]
    err_dets = sum(sum(c.values()) - max(c.values()) for _, c in errors)
    print('gate %.2f | people %3d | main: one-id %.2f, ids %.1f | others: one-id %.2f | mixed ids %d, wrong dets %d'
          % (gate, len(people), np.mean([r['main_id_share'] for r in main]), np.mean([r['ids'] for r in main]),
             np.mean([r['main_id_share'] for r in small]), len(errors), err_dets), flush=True)
    if abs(gate - 0.26) < 1e-6:
        for g, r in sorted(rows.items()):
            print('     %-4s %6d dets | one-id %.2f | ids %d' % (g, r['detections'], r['main_id_share'], r['ids']))
