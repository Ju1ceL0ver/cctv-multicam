import numpy as np, rules as R, lab2 as L, cv2
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
d = L.ZF
seq = {}
for t, v in d['tags'].items():
    for r in v['rows']:
        seq.setdefault((t, r[R.PIECE]), []).append(r)
for k in seq:
    seq[k].sort(key=lambda r: r[R.K])
byk = {}
for t, v in d['tags'].items():
    for r in v['rows']:
        byk.setdefault((t, r[R.K]), []).append(r)
masks = {n: None for n in ('band', 'door', 'shop')}
for n, z in zip(['band', 'shop', 'door', 'gallery'], d['zones']):
    m = np.zeros((720, 1280), np.uint8); cv2.fillPoly(m, [np.array(z['pts'], np.int32)], 1)
    masks[n] = m
dist = {n: cv2.distanceTransform(1 - masks[n], cv2.DIST_L2, 3) for n in ('band', 'door', 'shop')}
def g(n, x, y):
    return float(dist[n][int(min(719, max(0, y))), int(min(1279, max(0, x)))]) / 100

def base(r):
    x1, y1, x2, y2 = r[R.X1], r[R.Y1], r[R.X2], r[R.Y2]; w = x2 - x1; h = max(1, y2 - y1); cx = (x1 + x2) / 2; n = max(1, r[R.N])
    return [r[R.SHOP] / n, r[R.DOOR] / n, r[R.BAND] / n, r[R.GAL] / n, y2 / 720, h / 300, w / h, r[R.LEGS], cx / 1280, y1 / 720, w / 300, n / h / 300]

def extra(t, r, kind):
    s = seq[(t, r[R.PIECE])]; k = r[R.K]
    f = []
    if 'mem' in kind:
        for W in (1, 3, 6):
            past = [q for q in s if k - W * 12.5 <= q[R.K] < k]; fut = [q for q in s if k < q[R.K] <= k + W * 12.5]
            f += [float(any(q[R.SHOP] > 0 for q in past)), float(any(q[R.SHOP] > 0 for q in fut)),
                  float(any(q[R.FOOT] == 'gallery' for q in past)), float(any(q[R.FOOT] == 'gallery' for q in fut))]
    if 'nb' in kind:
        others = [q for q in byk[(t, k)] if q is not r]
        cx = (r[R.X1] + r[R.X2]) / 2
        if others:
            o = min(others, key=lambda q: abs((q[R.X1] + q[R.X2]) / 2 - cx) + abs(q[R.Y2] - r[R.Y2]))
            f += [o[R.SHOP] / max(1, o[R.N]), (o[R.Y2] - r[R.Y2]) / 300, abs((o[R.X1] + o[R.X2]) / 2 - cx) / 300, len(others)]
        else:
            f += [0, 0, 5, 0]
    if 'dist' in kind:
        cx = (r[R.X1] + r[R.X2]) / 2
        f += [g('band', r[R.FX], r[R.FY]), g('door', r[R.FX], r[R.FY]), g('shop', r[R.FX], r[R.FY]), g('band', cx, r[R.Y1]), g('door', cx, r[R.Y1])]
    return f

S = {}
for t, v in d['tags'].items():
    for r in v['rows']:
        if r[R.TRU] in ('in', 'out'):
            hard = r[R.LEGS] > .3 or (r[R.SHOP] == 0 and r[R.FOOT] in ('band', 'door', 'none'))
            S.setdefault(v['role'], []).append((t, r, r[R.TRU] == 'in', hard))

def run(kind, smooth=0):
    Xtr = np.array([base(r) + extra(t, r, kind) for t, r, _, _ in S['train']]); ytr = np.array([b for _, _, b, _ in S['train']])
    m = HGB(max_iter=300).fit(Xtr, ytr)
    rows = S['test']
    p = m.predict_proba(np.array([base(r) + extra(t, r, kind) for t, r, _, _ in rows]))[:, 1]
    if smooth:
        pk = {(t, r[R.PIECE], r[R.K]): pi for (t, r, _, _), pi in zip(rows, p)}
        p2 = []
        for (t, r, _, _), pi in zip(rows, p):
            vals = [pk[(t, q[R.PIECE], q[R.K])] for q in seq[(t, r[R.PIECE])] if abs(q[R.K] - r[R.K]) <= smooth * 12.5 and (t, q[R.PIECE], q[R.K]) in pk]
            p2.append(np.mean(vals))
        p = np.array(p2)
    yt = np.array([b for _, _, b, _ in rows]); H = np.array([h for *_, h in rows])
    for thr in (.5, .9, .97):
        dec = (p >= thr) | (p <= 1 - thr); pr = p >= .5
        print('%-12s smooth %ds thr %.2f: all ans %3.0f%% acc %.4f | hard ans %3.0f%% acc %.4f err %d' % (kind or 'geometry', smooth, thr, 100 * dec.mean(), (pr == yt)[dec].mean(), 100 * dec[H].mean(), (pr == yt)[dec & H].mean(), ((pr != yt) & dec & H).sum()))

if __name__ == '__main__':
    for kind in ('', 'mem', 'nb', 'dist', 'mem+nb+dist'):
        run(kind)
    for sm in (1, 2, 3):
        run('', sm)
    run('mem', 2)
