"""Tracker-level appearance gates from ground truth.

Inside one camera the tracker must pick, among the detections of the NEXT frames, the
one belonging to its track; its competitors are the other people visible at that
moment. So the gate is set against same-person distances over short time gaps versus
different-person distances in the same frame."""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from clipdata import load

rng = np.random.RandomState(0)
FPR = float(os.environ.get('RA_FPR_TRACK', '0.01'))
same, diff, splits = [], [], []
for clip in (sys.argv[1:] or ['c103700', 'c155236']):
    dets, feats, meta, embs = load(clip)
    gt = json.load(open(os.path.join('data', 'raw_clips', clip, 'gt_identity_yolo26x-seg.json')))
    for cam in ('cam1', 'cam2'):
        L = {int(k): v for k, v in gt[cam].items()}
        idx = np.array(sorted(L)); lab = np.array([L[i] for i in idx])
        t = dets[cam][idx, 0]
        E = embs[cam][idx]
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        order = np.argsort(t)
        idx, lab, t, E = idx[order], lab[order], t[order], E[order]
        # same person, 0.04-1.5 s apart (what the tracker bridges)
        for _ in range(60000):
            i = rng.randint(len(idx))
            j = i + rng.randint(1, 60)
            if j >= len(idx) or not (0.02 < t[j] - t[i] < 1.5):
                continue
            (same if lab[i] == lab[j] else None) is not None and same.append(1 - float(E[i] @ E[j]))
        # different people in the same frame
        for _ in range(60000):
            i = rng.randint(len(idx))
            j = i + rng.randint(1, 8)
            if j >= len(idx) or abs(t[j] - t[i]) > 1e-6 or lab[i] == lab[j]:
                continue
            diff.append(1 - float(E[i] @ E[j]))
        # the same person over 1 s windows, as the splitter measures it
        for _ in range(20000):
            i = rng.randint(len(idx) - 60)
            w = 25
            if i + 2 * w >= len(idx) or t[i + 2 * w] - t[i] > 3:
                continue
            a, b = E[i:i + w], E[i + w:i + 2 * w]
            if len(set(lab[i:i + 2 * w])) == 1:
                splits.append(float(np.linalg.norm(np.median(a, 0) - np.median(b, 0))))
same, diff, splits = np.array(same), np.array(diff), np.array(splits)
gate = float(np.quantile(diff, FPR))
out = {'high': round(gate, 3), 'low': round(gate * 0.8, 3), 'split': round(float(np.quantile(splits, 0.995)), 3),
       'same_median': round(float(np.median(same)), 3), 'diff_median': round(float(np.median(diff)), 3),
       'kept_same': round(float((same <= gate).mean()), 3), 'n_same': len(same), 'n_diff': len(diff), 'n_split': len(splits)}
print('tracker gates:', out)
p = 'data/appearance_thresholds.json'
t = json.load(open(p)) if os.path.exists(p) else {}
t['tracker'] = out
json.dump(t, open(p, 'w'), indent=1)
