"""Carry detection-level ground truth from one detector's boxes to another's (same clip, same frames)
by per-frame IoU matching. Also reports how well the frames line up (IoU distribution)."""
import sys, json, numpy as np
from scipy.optimize import linear_sum_assignment
from imtrack import iou_matrix

src_npz, src_gt, dst_npz, out = sys.argv[1:5]
OFF = {'cam1': int(sys.argv[5]) if len(sys.argv) > 5 else 0, 'cam2': int(sys.argv[6]) if len(sys.argv) > 6 else 0}
S = dict(np.load(src_npz)); D = dict(np.load(dst_npz)); G = json.load(open(src_gt))
res = {}
for cam in ('cam1', 'cam2'):
    s, d = S[cam], D[cam]
    fs, fd = np.round(s[:, 0] * 25).astype(int), np.round(d[:, 0] * 25).astype(int) + OFF[cam]
    lab = {}
    ious = []
    unmatched_dst = 0
    for f in np.unique(fd):
        di = np.nonzero(fd == f)[0]; si = np.nonzero(fs == f)[0]
        si = np.array([i for i in si if str(int(i)) in G[cam]], int)
        if len(si) == 0:
            unmatched_dst += len(di)
            continue
        M = iou_matrix(d[di, 1:5], s[si, 1:5])
        r, c = linear_sum_assignment(-M)
        for i, j in zip(r, c):
            if M[i, j] > 0.5:
                lab[str(int(di[i]))] = G[cam][str(int(si[j]))]
                ious.append(M[i, j])
    res[cam] = lab
    print('%s: %d teacher detections labelled from %d GT detections; matched IoU median %.2f; frames w/o GT dets: %d dets'
          % (cam, len(lab), len(G[cam]), np.median(ious) if ious else 0, unmatched_dst))
json.dump(res, open(out, 'w'))
