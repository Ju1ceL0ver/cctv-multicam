import numpy as np, rules as R
from sklearn.neighbors import KNeighborsClassifier

def feat(r):
    n = max(1, r[R.N]); h = r[R.Y2] - r[R.Y1]
    return [r[R.FX] / 1280, r[R.FY] / 720, h / 300, r[R.SHOP] / n, r[R.DOOR] / n, r[R.BAND] / n, r[R.GAL] / n, r[R.LEGS]]

def make(data, k=15, hi=.85, roles=('train',)):
    rows = {t: [(feat(r), r[R.TRU] == 'in') for r in d['rows'] if r[R.TRU] in ('in', 'out')]
            for t, d in data['tags'].items() if d['role'] in roles}
    def fn(tag):
        X = [x for t, v in rows.items() if t != tag for x, _ in v]          # leave this stretch out
        y = [yy for t, v in rows.items() if t != tag for _, yy in v]
        m = KNeighborsClassifier(k).fit(np.array(X), np.array(y))
        cache = {}
        def vote(r):
            p = m.predict_proba([feat(r)])[0][1]
            return 1 if p >= hi else (0 if p <= 1 - hi else None)
        return vote
    return fn
