"""The learned door rule, fitted once and kept, for the live door (06.10.2026). door_learn.py scores it day against
day; here it is fitted on every given day and saved, and applied to any run's ticks.

fit: the candidates and labels of each day (door_learn.load_day), the side map's features of each day seen through a
map of the other days (as in door_learn.evaluate), one boosting on all of them; the threshold from the days'
out-of-fold scores (each day scored by a model of the others), the best mean F1 of entries and exits; the side map
for new ticks fitted on all days.
apply: ticks (door_v2's run format, io added) -> [{'kind': 'in'|'out', 't': the moment of crossing (film or clock
seconds, as the ticks), 'w': the person, 'p': the rule's probability}].

usage: door_rule.py fit NAME DAY=RUN.jsonl.gz ...  -> data/door_v2/rule_NAME.pkl (+ its out-of-fold scores)"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'door_v2'


def fit(name, args):
    import door_learn as L
    import door_state as S
    import door_v2 as D
    days = []
    for a in args:
        day, path = a.split('=', 1)
        days.append(L.load_day(day, path))
    for d in days:
        d['cands_raw'] = d['cands']
    fitmap = lambda ds: S.fit_map([{'learn': {'cands': e['cands_raw'], 'y': e['y']}, 'by': e['by_world']} for e in ds])
    feats = {d['day']: L.map_features(dict(d, cands=d['cands_raw']), fitmap([e for e in days if e is not d])) for d in days}
    names = sorted(feats[days[0]['day']][0][0])
    XY = {d['day']: (L.matrix(feats[d['day']], names)[0], d['y']) for d in days}
    # out-of-fold scores: each day by a model of the others -> the threshold
    oof = {}
    for d in days:
        others = [e for e in days if e is not d]
        X = np.concatenate([XY[e['day']][0][XY[e['day']][1] >= 0] for e in others])
        y = np.concatenate([XY[e['day']][1][XY[e['day']][1] >= 0] for e in others])
        oof[d['day']] = L.model().fit(X, y).predict_proba(XY[d['day']][0])[:, 1]
    best, rep = (0.5, -1), {}
    for thr in np.arange(0.02, 0.951, 0.02):
        f = []
        for d in days:
            k = L.nms(feats[d['day']], oof[d['day']], thr)
            m = D.match([{'kind': feats[d['day']][i][1], 't': feats[d['day']][i][2], 'w': feats[d['day']][i][3]} for i in k], d['truth'], L.TOL)
            f.append((m['in']['f1'] + m['out']['f1']) / 2)
        if np.mean(f) > best[1]:
            best = (float(thr), float(np.mean(f)))
    thr = best[0]
    for d in days:
        k = L.nms(feats[d['day']], oof[d['day']], thr)
        m = D.match([{'kind': feats[d['day']][i][1], 't': feats[d['day']][i][2], 'w': feats[d['day']][i][3]} for i in k], d['truth'], L.TOL)
        rep[d['day']] = {kind: [round(m[kind]['precision'], 3), round(m[kind]['recall'], 3)] for kind in ('in', 'out')}
    X = np.concatenate([XY[d['day']][0][XY[d['day']][1] >= 0] for d in days])
    y = np.concatenate([XY[d['day']][1][XY[d['day']][1] >= 0] for d in days])
    rule = {'clf': L.model().fit(X, y), 'names': names, 'thr': thr, 'kmap': fitmap(days), 'shift': L.SHIFT,
            'days': [d['day'] for d in days], 'oof': rep}
    pickle.dump(rule, open(OUT / ('rule_%s.pkl' % name), 'wb'))
    print(json.dumps({'threshold': thr, 'mean_f1_oof': round(best[1], 3), 'oof_precision_recall': rep}, indent=1))
    return rule


def load(name):
    return pickle.load(open(OUT / ('rule_%s.pkl' % name), 'rb'))


def apply(ticks, rule):
    """ticks with io already added (door_v2.add_io)."""
    import door_learn as L
    import door_state as S
    cands = L.cluster(L.candidates(ticks))
    if not cands:
        return []
    cands = L.map_features({'cands': cands, 'by_world': S.tracks(ticks, split=False)}, rule['kmap'])
    p = rule['clf'].predict_proba(L.matrix(cands, rule['names'])[0])[:, 1]
    keep = L.nms(cands, p, rule['thr'])
    return sorted(({'kind': cands[i][1], 't': round(cands[i][2] - rule['shift'], 2), 'w': cands[i][3], 'p': round(float(p[i]), 3)}
                   for i in keep), key=lambda e: e['t'])


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        fit(sys.argv[2], sys.argv[3:])
