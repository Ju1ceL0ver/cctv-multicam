"""One person, frame by frame: which piece each of their detections landed in."""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from clipdata import load, FPS
clip, who, cam = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'cam2'
d = os.path.join('data', 'raw_clips', clip)
gt = json.load(open(os.path.join(d, 'gt_identity_yolo26x-seg.json')))
pieces = json.load(open(os.path.join(d, 'pieces_yolo26x-seg.json')))
piece_of = {}
for p in pieces:
    if p['cam'] == cam:
        for i in p['dets']:
            piece_of[int(i)] = p['piece']
dets, feats, meta, embs = load(clip, 'yolo26x-seg')
mine = sorted((float(dets[cam][int(k), 0]), int(k)) for k, v in gt[cam].items() if v == who and int(k) < len(dets[cam]))
print('%s in %s: %d detections' % (who, cam, len(mine)))
seq = [(round(t, 2), piece_of.get(i)) for t, i in mine]
print('first 45:', seq[:45])
from collections import Counter
runs, cur, n = [], None, 0
for _, pc in seq:
    if pc == cur:
        n += 1
    else:
        if cur is not None: runs.append((cur, n))
        cur, n = pc, 1
runs.append((cur, n))
print('pieces in order (piece, how many frames in a row):', runs[:40])
print('distinct pieces:', sorted(set(p for p, _ in runs)))
print('runs shorter than 3 frames: %d of %d' % (sum(1 for _, n in runs if n < 3), len(runs)))
