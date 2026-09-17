"""cam1 -> cam2 translation from appearance-verified simultaneous detections.

No position gating at all (the previous registration was wrong by metres, so it
cannot be trusted to gate): at each moment, detections on the two cameras are
matched by clothing colour (mutual best, clear margin to the second best), and
each match votes for the offset between the two floor frames (rotation 0). The
mode of the votes is the translation."""
import json, numpy as np
from person3d import Camera, place
from topview import UNIT

calib = json.load(open('data/calib_final.json'))
cams = {c: Camera(c, calib) for c in ('cam1', 'cam2')}
z = dict(np.load('data/clip/dets25.npz'))
dets = {c: z[c] for c in ('cam1', 'cam2')}
feat = {c: z[c + '_feat'] for c in ('cam1', 'cam2')}
P = place(cams, {'rotation_deg': 0, 'T_m': [0.0, 0.0]}, dets)       # own frames, metres

def desc(f):
    # Lab means of torso and legs, lightly weighting L (lighting differs per camera)
    return np.r_[f[6] * 0.5, f[7], f[8], f[12] * 0.5, f[13], f[14]]

votes = []
for t in np.unique(np.round(dets['cam1'][:, 0], 2))[::5]:
    i1 = np.nonzero(np.abs(dets['cam1'][:, 0] - t) < 1e-3)[0]
    i2 = np.nonzero(np.abs(dets['cam2'][:, 0] - t) < 1e-3)[0]
    if len(i1) == 0 or len(i2) == 0:
        continue
    D = np.array([[np.linalg.norm(desc(feat['cam1'][a]) - desc(feat['cam2'][b])) for b in i2] for a in i1])
    for ra, a in enumerate(i1):
        rb = int(np.argmin(D[ra]))
        if int(np.argmin(D[:, rb])) != ra:
            continue
        srt = np.sort(D[ra]); srt2 = np.sort(D[:, rb])
        margin = min(srt[1] if len(srt) > 1 else 99, srt2[1] if len(srt2) > 1 else 99) - D[ra, rb]
        if D[ra, rb] > 18 or margin < 6:
            continue
        b = i2[rb]
        p1, p2 = P['cam1']['xy'][ra if False else np.nonzero(i1 == a)[0][0]], None
        k1 = np.nonzero(np.abs(dets['cam1'][:, 0] - t) < 1e-3)[0].tolist().index(a)
        xy1 = P['cam1']['xy'][a]; xy2 = P['cam2']['xy'][b]
        if not (np.all(np.isfinite(xy1)) and np.all(np.isfinite(xy2))):
            continue
        votes.append((t, *(xy2 - xy1), D[ra, rb], margin, P['cam1']['vis'][a], P['cam2']['vis'][b], a, b))
V = np.array(votes, dtype=float)
print('appearance-matched simultaneous pairs:', len(V))
off = V[:, 1:3]
# mode by density
best = None
for o in off:
    n = (np.linalg.norm(off - o, axis=1) < 0.6).sum()
    if best is None or n > best[0]:
        best = (n, o)
near = np.linalg.norm(off - best[1], axis=1) < 0.6
T = np.median(off[near], axis=0)
print('mode offset T = (%.2f, %.2f) m supported by %d of %d votes; spread p50 %.2f m' %
      (T[0], T[1], near.sum(), len(V), np.median(np.linalg.norm(off[near] - T, axis=1))))
both_vis = near & (V[:, 6] > 0) & (V[:, 7] > 0)
if both_vis.sum():
    print('  with feet visible on both cameras: %d votes, median T=(%.2f, %.2f)' % (both_vis.sum(), *np.median(off[both_vis], 0)))
H, xe, ye = np.histogram2d(off[:, 0], off[:, 1], bins=[np.arange(-4, 4.01, 0.5), np.arange(-2, 10.01, 0.5)])
k = np.unravel_index(np.argsort(H.ravel())[::-1][:5], H.shape)
print('top offset cells (x, y, votes):', [(xe[i], ye[j], int(H[i, j])) for i, j in zip(*k)])
json.dump({'rotation_deg': 0, 'T_m': [float(T[0]), float(T[1])], 'votes': int(near.sum()),
           'method': 'appearance-matched simultaneous detections, clip 15:52:36'},
          open('data/cam1_to_cam2_appearance.json', 'w'), indent=1)
