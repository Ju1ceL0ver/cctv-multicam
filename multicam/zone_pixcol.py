"""13.10.2026, the owner's idea: every mask pixel knows where it is (x, y) and how much of the person is above and below
it in its column -> p(inside) per pixel from the owner's labels ('train' stretches), averaged over the person.
Also variants: only the upper half of the body (robust to hidden legs). -> data/door_v2/zone_pixcol.json.gz"""
import gzip, json, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import door_side as S
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

F = json.load(gzip.open(ROOT / 'data/door_v2/zone_feats.json.gz', 'rt'))
rng = np.random.default_rng(0)


def pix(poly):
    m = np.zeros((180, 320), np.uint8)
    cv2.fillPoly(m, [(np.array(poly) / 4).astype(np.int32)], 1)
    m = m.astype(bool)
    if m.sum() < 8:
        return None
    up = np.cumsum(m, 0) - 1                    # person pixels above in the column
    down = np.cumsum(m[::-1], 0)[::-1] - 1      # and below
    ys, xs = np.nonzero(m)
    return np.c_[xs / 320, ys / 180, up[ys, xs] / 180, down[ys, xs] / 180]


Xtr, ytr, obs = [], [], {}
for tag, d in F['tags'].items():
    ol = S.outlines(tag)
    for r in d['rows']:
        k, piece, tru = r[0], r[1], r[3]
        if tru not in ('in', 'out'):
            continue
        poly = next((p for pc, _, p in ol.get(str(k), []) if str(pc) == piece), None)
        if poly is None:
            continue
        P = pix(poly)
        if P is None:
            continue
        obs[(tag, k, piece)] = P
        if d['role'] == 'train':
            i = rng.choice(len(P), min(len(P), 15), replace=False)
            Xtr.append(P[i]); ytr += [tru == 'in'] * len(i)
    print(tag, flush=True)
Xtr = np.concatenate(Xtr); ytr = np.array(ytr)
print('pixels', len(Xtr), flush=True)
m4 = HGB(max_iter=300).fit(Xtr, ytr)
m2 = HGB(max_iter=300).fit(Xtr[:, :2], ytr)                       # only (x, y): the plain pixel map
out = {}
for (tag, k, piece), P in obs.items():
    p4 = m4.predict_proba(P)[:, 1]; p2 = m2.predict_proba(P[:, :2])[:, 1]
    upper = P[:, 3] > P[:, 2]                                    # pixels with more person below than above: upper body
    out.setdefault(tag, {})['%d:%s' % (k, piece)] = [round(float(p4.mean()), 3), round(float(np.median(p4)), 3),
        round(float(p4[upper].mean()) if upper.any() else float(p4.mean()), 3), round(float(p2.mean()), 3),
        round(float(p4.max()), 3)]
json.dump(out, gzip.open(ROOT / 'data/door_v2/zone_pixcol.json.gz', 'wt'))
print('done')
