"""doorseg_c tracks (cached predictions) -> tracker variants -> zone rows (+ feet from the head) -> rules -> F1"""
import gzip, json, os, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # door_seg_track lives in multicam/
import rules as R
from door_seg_track import Tracker, Config
from sklearn.neighbors import KNeighborsRegressor

H4, W4 = 184, 320
ZF = R.load('zone_feats.json.gz')
ZONES = ZF['zones']; NAMES = ['band', 'shop', 'door', 'gallery']
ZM = {}; POLY = {}
for n, z in zip(NAMES, ZONES):
    m = np.zeros((180, 320), np.uint8); cv2.fillPoly(m, [(np.array(z['pts']) / 4).astype(np.int32)], 1)
    ZM[n] = m.astype(bool); POLY[n] = np.array(z['pts'], np.float32)


def zone_of(x, y):
    return next((n for n in ('door', 'shop', 'band', 'gallery') if cv2.pointPolygonTest(POLY[n], (float(x), float(y)), False) >= 0), 'none')


# head -> feet, from the teacher's unoccluded people on 'train' stretches (top-centre of the box -> lowest pixel)
X, Y = [], []
for t, d in ZF['tags'].items():
    if d['role'] != 'train':
        continue
    for r in d['rows']:
        if r[R.LEGS] < .02 and r[R.FOOT] != 'none' and r[R.Y2] - r[R.Y1] > 60:
            X.append([(r[R.X1] + r[R.X2]) / 2, r[R.Y1]]); Y.append([r[R.FX], r[R.FY]])
HEAD = KNeighborsRegressor(15).fit(np.array(X), np.array(Y))


def load(tag):
    z = np.load(os.path.join(os.environ.get('ZL_CACHE', 'dsc_cache'), '%s.npz' % tag))
    rows, bits = z['rows'], z['bits']
    by = {}
    for r in rows:
        k = int(r[0]); x0, y0, w, h, off = (int(v) for v in r[10:15])
        m = np.zeros((H4, W4), bool)
        m[y0:y0 + h, x0:x0 + w] = np.unpackbits(bits[off:off + (w * h + 7) // 8])[:w * h].reshape(h, w).astype(bool)
        by.setdefault(k, []).append((m, r[2:6].astype(float), float(r[1]), r[6:8].astype(float), r[8:10].astype(float)))
    return by


def track(by, cfg, birth=.4, cont=.4):
    """keep detections >= cont; a track exists only once one of its detections reached `birth` (weak ones only continue)"""
    tr = Tracker(cfg); out = {}; ok = set()
    for k in range(max(by) + 1 if by else 0):
        res = tr.update(k, [p for p in by.get(k, []) if p[2] >= cont])
        keep = []
        for pid, p in res:
            if pid not in ok and p[2] < birth:
                tr.tracks.pop(pid, None); continue
            ok.add(pid); keep.append((pid, p))
        out[k] = keep
    return out


def rows_of(tracks):
    rows = []
    for k, ps in tracks.items():
        if k % 3:
            continue
        people = [(pid, p[0][:180]) for pid, p in ps if p[0][:180].any()]
        boxes = []
        for pid, m in people:
            ys, xs = np.nonzero(m); boxes.append((xs.min() * 4, ys.min() * 4, xs.max() * 4 + 4, ys.max() * 4 + 4))
        heads = np.array([[(b[0] + b[2]) / 2, b[1]] for b in boxes]) if boxes else np.zeros((0, 2))
        vf = HEAD.predict(heads) if len(heads) else []
        for i, (pid, m) in enumerate(people):
            x1, y1, x2, y2 = boxes[i]; ys, xs = np.nonzero(m); j = np.argmax(ys)
            fx, fy = int(xs[j] * 4 + 2), int(ys[j] * 4 + 3)
            zc = [int((m & ZM[z]).sum()) * 16 for z in NAMES]
            h = y2 - y1; L = (x1 - 5, y2 - .2 * h, x2 + 5, y2 + .5 * h); cov = 0.
            for jj, (q, mm) in enumerate(people):
                if jj != i:
                    a = mm[max(0, int(L[1] / 4)):max(0, int(L[3] / 4)) + 1, max(0, int(L[0] / 4)):max(0, int(L[2] / 4)) + 1]
                    if a.size:
                        cov = max(cov, float(a.mean()))
            rows.append([k, str(pid), None, None, x1, y1, x2, y2, fx, fy, int(m.sum()) * 16] + zc + [round(cov, 3), zone_of(fx, fy), zone_of(*vf[i])])
    return rows


if __name__ == '__main__':
    pass
