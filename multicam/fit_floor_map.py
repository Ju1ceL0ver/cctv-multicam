"""cam1 floor -> cam2 floor mapping from ground-truth identity correspondences.

Pairs: the same real person at the same instant on both cameras (gt_identity),
both feet visible for accuracy. Models of increasing freedom are compared by
cross-validated residual (fit on even seconds, test on odd), so a model only
wins by generalising, not by having more parameters."""
import json, cv2, numpy as np
from person3d import Camera, place

calib = json.load(open('data/calib_final.json'))
cams = {c: Camera(c, calib) for c in ('cam1', 'cam2')}
z = dict(np.load('data/clip/dets25.npz'))
dets = {c: z[c] for c in ('cam1', 'cam2')}
gt = json.load(open('data/clip/gt_identity.json'))
P = place(cams, {'rotation_deg': 0, 'T_m': [0, 0]}, dets)          # own frames, metres

def index(cam):
    out = {}
    for k, g in gt[cam].items():
        k = int(k)
        if g == 'D':
            continue
        out.setdefault((round(float(dets[cam][k, 0]), 2), g), []).append(k)
    return out
i1, i2 = index('cam1'), index('cam2')
pairs = []
for key, ks1 in i1.items():
    ks2 = i2.get(key)
    if not ks2 or len(ks1) != 1 or len(ks2) != 1:
        continue
    a, b = ks1[0], ks2[0]
    pairs.append((key[0], key[1], a, b, P['cam1']['vis'][a], P['cam2']['vis'][b]))
pairs = np.array(pairs, dtype=object)
t = pairs[:, 0].astype(float)
vis = pairs[:, 4].astype(bool) & pairs[:, 5].astype(bool)
X1 = P['cam1']['xy'][pairs[:, 2].astype(int)]; X2 = P['cam2']['xy'][pairs[:, 3].astype(int)]
fin = np.all(np.isfinite(X1), 1) & np.all(np.isfinite(X2), 1)
print('GT simultaneous pairs %d, both feet visible %d' % (len(pairs), (vis & fin).sum()))
from collections import Counter
print('  by person (both feet):', Counter(pairs[vis & fin, 1]))

def fit(model, A, B):
    if model == 'translation':
        T = np.median(B - A, 0); return lambda X: X + T
    if model == 'similarity':
        M, _ = cv2.estimateAffinePartial2D(A.astype(np.float32), B.astype(np.float32), method=cv2.LMEDS)
    elif model == 'affine':
        M, _ = cv2.estimateAffine2D(A.astype(np.float32), B.astype(np.float32), method=cv2.LMEDS)
    elif model == 'homography':
        H, _ = cv2.findHomography(A.astype(np.float32), B.astype(np.float32), cv2.LMEDS)
        return lambda X: cv2.perspectiveTransform(X.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    return lambda X: X @ M[:, :2].T + M[:, 2]

for subset_name, S in (('both feet visible', vis & fin), ('all', fin)):
    A, B, tt = X1[S], X2[S], t[S]
    tr = (np.floor(tt) % 2) == 0
    print('--- %s: %d pairs, cam2 y range %.1f..%.1f' % (subset_name, len(A), B[:, 1].min(), B[:, 1].max()))
    for model in ('translation', 'similarity', 'affine', 'homography'):
        f = fit(model, A[tr], B[tr])
        d = np.linalg.norm(f(A[~tr]) - B[~tr], axis=1)
        far = B[~tr][:, 1] > 4.0
        print('   %-12s test median %.2f p90 %.2f m | near (y<=4) %.2f  far (y>4) %.2f'
              % (model, np.median(d), np.percentile(d, 90), np.median(d[~far]) if (~far).any() else np.nan,
                 np.median(d[far]) if far.any() else np.nan))
A, B = X1[vis & fin], X2[vis & fin]
H, inl = cv2.findHomography(A.astype(np.float32), B.astype(np.float32), cv2.LMEDS)
M, _ = cv2.estimateAffinePartial2D(A.astype(np.float32), B.astype(np.float32), method=cv2.LMEDS)
sc = np.sqrt(abs(np.linalg.det(M[:, :2]))); ang = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
print('similarity: scale %.3f rotation %+.1f deg T (%.2f, %.2f)' % (sc, ang, *M[:, 2]))
json.dump({'H_cam1_to_cam2': H.tolist(), 'similarity': M.tolist(), 'pairs_both_feet': int(len(A)),
           'source': 'gt_identity clip 15:52:36'}, open('data/floor_map_gt.json', 'w'), indent=1)
