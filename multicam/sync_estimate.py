"""Time offset between the two cameras' raw recordings, from people alone.

The recorders name segments by when ffmpeg was launched, but each RTSP stream
starts after its own connection delay, so the two cameras' timelines differ by
seconds -- and by a different amount after every recorder restart. For each
candidate shift, count moments where a WALKING person on cam2 has a cam1
detection on the same floor spot (standing people agree at any shift, so they
carry no timing information). The peak is the offset.

usage: sync_estimate.py CLIP [TAG]   -> prints the peak; cam1 time + offset = cam2 time"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera
from fusion2 import build
from clipdata import load
from scipy.spatial import cKDTree


def estimate(dets, feats, span=8.0, step=0.04, radius=0.4, min_speed=0.5):
    calib = json.load(open('data/calib_final.json'))
    cams = {c: Camera(c, calib) for c in ('cam1', 'cam2')}
    per_cam, _ = build(cams, dets, feats, embs, meta.get('clean'))
    c1 = [(it['t'], it['xy']) for it in per_cam['cam1']]
    moving = []
    for it in per_cam['cam2']:
        t, xy = it['t'], it['xy']
        if len(t) < 10:
            continue
        v = np.gradient(xy, axis=0) / np.maximum(np.gradient(t), 1e-3)[:, None]
        sp = np.linalg.norm(v, axis=1)
        k = sp > min_speed
        moving.append(np.c_[t[k], xy[k]])
    M = np.concatenate(moving) if moving else np.zeros((0, 3))
    T1 = np.concatenate([np.c_[t, xy] for t, xy in c1]) if c1 else np.zeros((0, 3))
    shifts = np.arange(-span, span + 1e-9, step)
    scores = []
    q = np.round(M[:, 0] / step).astype(int)
    for s in shifts:
        k1 = np.round((T1[:, 0] + s) / step).astype(int)
        by = {}
        for i, kk in enumerate(k1):
            by.setdefault(kk, []).append(i)
        hit = 0
        for j, kk in enumerate(q):
            idx = by.get(kk)
            if idx and np.min(np.linalg.norm(T1[idx, 1:] - M[j, 1:], axis=1)) < radius:
                hit += 1
        scores.append(hit)
    scores = np.array(scores, float)
    b = int(np.argmax(scores))
    others = np.abs(shifts - shifts[b]) > 0.6
    return shifts[b], scores[b], float(np.median(scores[others])), len(M), shifts, scores


if __name__ == '__main__':
    clip = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'yolo26x-seg'
    dets, feats, meta, _emb = load(clip, tag, apply_sync=False)
    s, peak, base, n, shifts, scores = estimate(dets, feats)
    top = np.argsort(scores)[::-1][:5]
    print('%s: offset %+.2f s (cam1 time + offset = cam2 time); peak %d vs typical %d over %d walking cam2 points; top %s'
          % (clip, s, peak, base, n, [(round(float(shifts[i]), 2), int(scores[i])) for i in top]))
