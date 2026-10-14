"""13.10.2026, the owner: a pixel kNN prior -- every mask pixel of labelled people (inside / outside), k=32 on (x, y):
the share of inside neighbours = a map p_in(x, y). For each day the map is built from the other two days' 'train'
stretches only. Per observation: mean / max / share>0.5 of p_in over the mask, and the mean over its lowest 20 %.
-> data/door_v2/zone_pix.json.gz {tag: {"k:piece": [mean, max, frac, low]}}"""
import gzip, json, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import door_side as S
from sklearn.neighbors import KNeighborsClassifier

F = json.load(gzip.open(ROOT / 'data/door_v2/zone_feats.json.gz', 'rt'))
rng = np.random.default_rng(0)
masks = {}                                          # tag -> {(k, piece): (ys, xs)} at 1/4
samples = {}                                        # day -> (X, y)
for tag, d in F['tags'].items():
    ol = S.outlines(tag)
    tru = {(r[0], r[1]): r[3] for r in d['rows']}
    mm = {}
    for (k, piece), t in tru.items():
        poly = next((p for pc, _, p in ol.get(str(k), []) if str(pc) == piece), None)
        if poly is None:
            continue
        m = np.zeros((180, 320), np.uint8)
        cv2.fillPoly(m, [(np.array(poly) / 4).astype(np.int32)], 1)
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        mm[(k, piece)] = (ys, xs)
        if d['role'] == 'train' and t in ('in', 'out'):
            i = rng.choice(len(xs), min(len(xs), 20), replace=False)
            X, y = samples.setdefault(tag[5:13], ([], []))
            X.extend(np.c_[xs[i], ys[i]].tolist()); y.extend([t == 'in'] * len(i))
    masks[tag] = mm
    print(tag, len(mm), flush=True)
gx, gy = np.meshgrid(np.arange(320), np.arange(180))
grid = np.c_[gx.ravel(), gy.ravel()]
maps = {}
for day in sorted({t[5:13] for t in F['tags']}):
    X = [x for d_, (xx, _) in samples.items() if d_ != day for x in xx]
    y = [v for d_, (_, yy) in samples.items() if d_ != day for v in yy]
    knn = KNeighborsClassifier(32).fit(np.array(X), np.array(y))
    maps[day] = knn.predict_proba(grid)[:, 1].reshape(180, 320)
    cv2.imwrite(str(ROOT / 'data/door_v2' / ('zone_pixmap_%s.png' % day)), (maps[day] * 255).astype(np.uint8))
    print('map', day, len(X), flush=True)
out = {}
for tag, mm in masks.items():
    M = maps[tag[5:13]]
    o = {}
    for (k, piece), (ys, xs) in mm.items():
        p = M[ys, xs]
        low = ys >= ys.min() + .8 * (ys.max() - ys.min())
        o['%d:%s' % (k, piece)] = [round(float(p.mean()), 3), round(float(p.max()), 3), round(float((p > .5).mean()), 3), round(float(p[low].mean()), 3)]
    out[tag] = o
json.dump(out, gzip.open(ROOT / 'data/door_v2/zone_pix.json.gz', 'wt'))
print('done')
