"""08.10: staff vs customer on the owner's /staff answers, every day scored by a model of the other days:
logistic (the page's), kNN on the ReID vectors (cosine), ExtraTrees, MLP; also how sure the best one is."""
import json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import staff
F, r = staff.feats(str(R)), staff.labels(str(R))
ans = [(i, v['label']) for i, v in r.items() if v['label'] in (1, 2) and i in F['row']]
X = F['X'][[F['row'][i] for i, _ in ans]]
y = np.array([a == 2 for _, a in ans])
days = np.array([F['meta'][i]['day'] for i, _ in ans])
cam = np.array([F['meta'][i]['cam'] for i, _ in ans])
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
emb = normalize(X[:, :192])                      # ReID part (cloth 128 + shape 64), cosine
models = {
    'logistic (now)': lambda: (LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'), X),
    'kNN k=5': lambda: (KNeighborsClassifier(5, weights='distance'), emb),
    'kNN k=15': lambda: (KNeighborsClassifier(15, weights='distance'), emb),
    'ExtraTrees': lambda: (ExtraTreesClassifier(400, min_samples_leaf=2, class_weight='balanced', n_jobs=8, random_state=0), X),
    'MLP': lambda: (MLPClassifier((256, 64), max_iter=400, early_stopping=True, random_state=0), X),
}
for name, mk in models.items():
    P = np.zeros(len(y))
    for d in sorted(set(days)):
        te = days == d
        m, Z = mk()
        m.fit(Z[~te], y[~te])
        P[te] = m.predict_proba(Z[te])[:, 1]
    q = P >= 0.5
    c1 = cam == 'cam1'
    sure = np.maximum(P, 1 - P) >= 0.9
    print('%-15s acc %.3f | cam1 %.3f | staff->cust %d cust->staff %d | sure %.2f acc %.3f' % (
        name, (q == y).mean(), (q[c1] == y[c1]).mean(), int((~q & y).sum()), int((q & ~y).sum()), sure.mean(), (q[sure] == y[sure]).mean()), flush=True)
