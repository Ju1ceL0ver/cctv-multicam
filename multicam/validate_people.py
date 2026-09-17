"""Independent check of a calibration with people (never used in fitting).

If both cameras are right, the same person at the same instant lands on the same
floor spot up to a pure rigid shift between the two floor frames: a fitted affine
map should have singular values ~1 and a translation-only fit should leave the
same small residual near and far. A person's height should not depend on where
they stand."""
import sys, json, cv2, numpy as np
import person3d
from person3d import Camera, place

def run(path):
    calib = json.load(open(path))
    cams = {c: Camera(c, calib) for c in ('cam1', 'cam2')}
    z = dict(np.load('data/clip/dets25.npz'))
    dets = {c: z[c] for c in ('cam1', 'cam2')}
    gt = json.load(open('data/clip/gt_identity.json'))
    P = place(cams, {'rotation_deg': 0, 'T_m': [0, 0]}, dets)
    idx = {}
    for cam in ('cam1', 'cam2'):
        for k, g in gt[cam].items():
            if g != 'D':
                idx.setdefault(cam, {}).setdefault((round(float(dets[cam][int(k), 0]), 2), g), []).append(int(k))
    pairs = [(key, a[0], idx['cam2'][key][0]) for key, a in idx['cam1'].items()
             if key in idx['cam2'] and len(a) == 1 and len(idx['cam2'][key]) == 1]
    A = np.array([P['cam1']['xy'][a] for _, a, b in pairs]); B = np.array([P['cam2']['xy'][b] for _, a, b in pairs])
    ok = np.all(np.isfinite(A), 1) & np.all(np.isfinite(B), 1)
    A, B = A[ok], B[ok]
    t = np.array([k[0] for k, a, b in pairs])[ok]
    tr = (np.floor(t) % 2) == 0
    # rotation-0 translation (the physically expected model: both frames on one tile lattice)
    T = np.median(B[tr] - A[tr], 0)
    d_t = np.linalg.norm(A[~tr] + T - B[~tr], axis=1)
    M, _ = cv2.estimateAffine2D(A[tr].astype(np.float32), B[tr].astype(np.float32), method=cv2.LMEDS)
    d_a = np.linalg.norm(A[~tr] @ M[:, :2].T + M[:, 2] - B[~tr], axis=1)
    sv = np.linalg.svd(M[:, :2])[1]
    far = B[~tr][:, 1] > 3.5
    print('%s' % path)
    print('   translation-only: T=(%.2f, %.2f)  held-out median %.2f p90 %.2f m | near %.2f far %.2f'
          % (*T, np.median(d_t), np.percentile(d_t, 90), np.median(d_t[~far]), np.median(d_t[far])))
    print('   affine          : singular values %.2f %.2f   held-out median %.2f p90 %.2f m' % (sv[0], sv[1], np.median(d_a), np.percentile(d_a, 90)))
    for cam in ('cam1', 'cam2'):
        c = cams[cam]
        rows = []
        for k, g in gt[cam].items():
            k = int(k)
            if g == 'D' or not P[cam]['vis'][k] or not np.isfinite(P[cam]['h'][k]):
                continue
            dist = np.linalg.norm(P[cam]['xy'][k] - c.C[:2] * 0.26)
            rows.append((g, dist, P[cam]['h'][k]))
        s = []
        for g in 'ABC':
            r = np.array([(d, h) for gg, d, h in rows if gg == g])
            if len(r) >= 15:
                slope = np.polyfit(r[:, 0], r[:, 1], 1)[0] if np.ptp(r[:, 0]) > 1.0 else np.nan
                s.append('%s %.2f m (n=%d, %.1f-%.1f m away, slope %+.3f m/m)' % (g, np.median(r[:, 1]), len(r), r[:, 0].min(), r[:, 0].max(), slope))
        print('   %s heights: %s' % (cam, ' | '.join(s)))

for p in sys.argv[1:]:
    run(p)
