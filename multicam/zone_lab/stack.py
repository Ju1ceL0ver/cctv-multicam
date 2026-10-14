"""13.10.2026, the owner: rules and pixel-set features combined -- AND/OR of 2-3 binary rules, products of numeric
features, region-by-zone fractions (top/mid/bottom of the mask x 4 owner zones) -> L1 logistic regression and boosting.
Teacher rows (zone_feats), train stretches fit, test stretches scored; 'hard' = legs covered or no shop pixel with the
lowest pixel in the band/door. usage (machine, cwd zone_lab): python stack.py"""
import gzip, itertools, json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rules as R, hyp as Hy, lab2 as L
from lab3 import SIZE
import door_side as S
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = Hy.d
pix = json.load(gzip.open(os.path.join(ROOT, 'data/door_v2/zone_pixcol.json.gz'), 'rt'))
zm = {}
for n, z in zip(['band', 'shop', 'door', 'gallery'], d['zones']):
    m = np.zeros((180, 320), np.uint8); cv2.fillPoly(m, [(np.array(z['pts']) / 4).astype(np.int32)], 1); zm[n] = m.astype(bool)
REG = {}
for tag in d['tags']:
    ol = S.outlines(tag)
    for ks, items in ol.items():
        for piece, _, poly in items:
            m = np.zeros((180, 320), np.uint8); cv2.fillPoly(m, [(np.array(poly) / 4).astype(np.int32)], 1); m = m.astype(bool)
            ys = np.nonzero(m)[0]
            if len(ys) < 4:
                continue
            y0, y1 = ys.min(), ys.max(); h = max(1, y1 - y0)
            f = []
            for a, b in ((0, .2), (.2, .8), (.8, 1.01)):
                part = m.copy(); part[:int(y0 + a * h)] = False; part[int(y0 + b * h) + 1:] = False
                n = max(1, part.sum())
                f += [float((part & zm[z]).sum()) / n for z in ('band', 'shop', 'door', 'gallery')]
            REG[(tag, int(ks), str(piece))] = f
    print('regions', tag, flush=True)


def binary(t, r):
    s = Hy.seq[(t, r[R.PIECE])]; k = r[R.K]; h = r[R.Y2] - r[R.Y1]
    med = np.median([q[R.Y2] - q[R.Y1] for q in s if q[R.LEGS] < .2] or [h])
    p = pix.get(t, {}).get('%d:%s' % (k, r[R.PIECE]), [.5] * 5)
    past = [q for q in s if k - 37 <= q[R.K] < k]; fut = [q for q in s if k < q[R.K] <= k + 37]
    return {'shop1': r[R.SHOP] >= 1, 'shop400': r[R.SHOP] >= 400, 'foot_gal': r[R.FOOT] == 'gallery', 'foot_band': r[R.FOOT] == 'band',
            'foot_door': r[R.FOOT] == 'door', 'legs': r[R.LEGS] > .3, 'cut': h < .75 * med, 'pix_in': p[0] > .9, 'pix_out': p[0] < .1,
            'shop_before': any(q[R.SHOP] > 0 for q in past), 'shop_after': any(q[R.SHOP] > 0 for q in fut),
            'gal_before': any(q[R.FOOT] == 'gallery' for q in past), 'gal_after': any(q[R.FOOT] == 'gallery' for q in fut)}


def features(t, r):
    b = binary(t, r); names = sorted(b); v = [float(b[n]) for n in names]
    combos = [1 - x for x in v]                                    # NOT
    for a, c in itertools.combinations(range(len(names)), 2):
        x, y = v[a], v[c]
        combos += [x * y, max(x, y), float(x != y), float(x == y),  # AND, OR, XOR, XNOR
                   max(1 - x, y), max(1 - y, x),                  # a -> b, b -> a
                   1 - x * y, 1 - max(x, y),                      # NAND, NOR
                   x * (1 - y), y * (1 - x)]                       # a AND NOT b, b AND NOT a
    for a, c, e in itertools.combinations(range(len(names)), 3):
        combos += [v[a] * v[c] * v[e], float((v[a] + v[c] + v[e]) % 2), float(v[a] + v[c] + v[e] >= 2)]   # AND, XOR, majority
    num = Hy.base(r) + (pix.get(t, {}).get('%d:%s' % (r[R.K], r[R.PIECE])) or [.5] * 5)
    prods = [num[i] * num[j] for i, j in itertools.combinations(range(len(num)), 2)]
    reg = REG.get((t, r[R.K], r[R.PIECE]), [0.] * 12)
    h = r[R.Y2] - r[R.Y1]; w = r[R.X2] - r[R.X1]; cx = (r[R.X1] + r[R.X2]) / 2
    s_ = Hy.seq[(t, r[R.PIECE])]
    med = np.median([q[R.Y2] - q[R.Y1] for q in s_ if q[R.LEGS] < .2] or [h])
    vf = L.HEAD.predict([[cx, r[R.Y1]]])[0]
    vfz = L.zone_of(*vf)
    psize = float(SIZE.predict_proba([[cx / 1280, r[R.Y1] / 720, w / 300, r[R.N] / max(1, h) / 300]])[0][1])
    raw = [r[R.SHOP], r[R.DOOR], r[R.BAND], r[R.GAL], r[R.N], r[R.X1], r[R.Y1], r[R.X2], r[R.Y2], r[R.FX], r[R.FY], h, w,
           h / max(1, med), r[R.LEGS], vf[0], vf[1]] + [float(r[R.FOOT] == z) for z in ('shop', 'door', 'band', 'gallery', 'none')] + \
          [float(vfz == z) for z in ('shop', 'door', 'band', 'gallery', 'none')] + [psize]
    return v + combos + num + prods + reg + raw


rows = {role: [(features(t, r), y, h) for t, r, y, h in Hy.S[role]] for role in ('train', 'test')}
Xtr = np.array([x for x, _, _ in rows['train']]); ytr = np.array([y for _, y, _ in rows['train']])
Xt = np.array([x for x, _, _ in rows['test']]); yt = np.array([y for _, y, _ in rows['test']]); H = np.array([h for *_, h in rows['test']])
print('features', Xtr.shape, flush=True)
res = {}
sc = StandardScaler().fit(Xtr)
models = {'logreg_L1': LogisticRegression(penalty='l1', C=.1, solver='saga', max_iter=3000),
          'logreg_L2': LogisticRegression(C=1., max_iter=3000), 'boost': HGB(max_iter=400, random_state=0)}
for name, m in models.items():
    fitX = sc.transform(Xtr) if 'logreg' in name else Xtr
    m.fit(fitX, ytr)
    p = m.predict_proba(sc.transform(Xt) if 'logreg' in name else Xt)[:, 1]
    for thr in (.5, .9, .97):
        dec = (p >= thr) | (p <= 1 - thr); pr = p >= .5
        line = '%-10s thr %.2f: all ans %3.0f%% acc %.4f | hard ans %3.0f%% acc %.4f err %d' % (
            name, thr, 100 * dec.mean(), (pr == yt)[dec].mean(), 100 * dec[H].mean(), (pr == yt)[dec & H].mean(), ((pr != yt) & dec & H).sum())
        print(line, flush=True); res.setdefault(name, []).append(line)
    if name == 'logreg_L1':
        nz = int((m.coef_ != 0).sum()); res['l1_nonzero'] = nz
        print('L1 kept', nz, 'of', Xtr.shape[1], flush=True)
json.dump(res, open('stack_result.json', 'w'), indent=1)
print('done')
