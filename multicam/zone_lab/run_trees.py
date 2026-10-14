import gzip, json, sys
import numpy as np
import rules as R, trees as T
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
d = R.load('zone_feats.json.gz'); pix = json.load(gzip.open('zone_pix.json.gz', 'rt'))
F = T.features(d, pix)
base = dict(R.BASE, shop=1, first=0, occl_legs=.6, occl_h=.6, conf=1, pair_s=3)
print('rules', R.score(d, base))
makers = {'tree3': lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, class_weight='balanced', random_state=0),
          'tree5': lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, class_weight='balanced', random_state=0),
          'tree7': lambda: DecisionTreeClassifier(max_depth=7, min_samples_leaf=50, class_weight='balanced', random_state=0),
          'rf3x5': lambda: RandomForestClassifier(3, max_depth=5, min_samples_leaf=50, class_weight='balanced', random_state=0),
          'rf5x7': lambda: RandomForestClassifier(5, max_depth=7, min_samples_leaf=50, class_weight='balanced', random_state=0)}
res = {}
for name, mk in makers.items():
    M = T.fit_lodo(d, F, mk)
    cache = {}
    def clf(tag, r, M=M, cache=cache):
        key = (tag, r[R.K], r[R.PIECE])
        if key not in cache:
            cache[key] = M[tag[5:13]].predict_proba([F[key]])[0][1]
        return cache[key]
    for hi in (.6, .8, .9):
        for sw in (False, True):
            c = dict(base, clf=clf, clf_hi=hi, clf_shop_wins=sw)
            s = R.score(d, c)
            res[(name, hi, sw)] = s
            print(name, hi, 'shop_wins' if sw else '', s, flush=True)
    if name == 'tree3':
        print(export_text(M['20260918'], feature_names=T.NAMES))
