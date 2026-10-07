"""The inside/outside classifier (place boosting, inout_train.geometry) before and after the owner's door batch
(06.10.2026): every day scored by a model of the other days; accuracy on the door samples and on all.
Then (with --refit) data/door_v2/io_geom.pkl is refitted on every answer, the door batch included."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(refit=False, review=False):
    import inout
    import inout_train as IT
    S, L = inout.samples(str(ROOT)), inout.labels(str(ROOT))
    rows = []
    for sid, a in L.items():
        s = S.get(sid)
        k = a.get('label') if isinstance(a, dict) else a
        if s is None or k not in IT.CLASS:
            continue
        rows.append((sid, s['day'], bool(s.get('door')), IT.geometry(s), IT.CLASS[k]))
    days = sorted({r[1] for r in rows})
    X = np.array([r[3] for r in rows], np.float32); y = np.array([r[4] for r in rows]); door = np.array([r[2] for r in rows])
    day = np.array([r[1] for r in rows])
    print('answers', len(rows), 'door batch', int(door.sum()), 'classes', np.bincount(y).tolist(), 'door classes', np.bincount(y[door], minlength=3).tolist())
    prob = np.zeros(len(y))
    for name, use_door in (('without the door batch', False), ('with the door batch', True)):
        pred = np.full(len(y), -1)
        for d in days:
            tr = (day != d) & (use_door | ~door)
            te = day == d
            if te.sum() == 0 or len(set(y[tr])) < 2:
                continue
            m_ = IT.boost().fit(X[tr], y[tr])
            pred[te] = m_.predict(X[te])
            if use_door:
                prob[te] = m_.predict_proba(X[te]).max(1)
        ok = pred >= 0
        acc = lambda m: float((pred[m & ok] == y[m & ok]).mean()) if (m & ok).any() else float('nan')
        io_err = lambda m: int(((pred != y) & (y < 2) & (pred < 2) & m & ok).sum())
        print('%-24s all %.3f | door batch %.3f (%d of %d wrong, inside<->outside %d) | old samples %.3f' % (
            name, acc(np.ones(len(y), bool)), acc(door), int(((pred != y) & door & ok).sum()), int((door & ok).sum()), io_err(door), acc(~door)))
    if review:      # 07.10: answers the model of the other days disagrees with, the surest disagreement first
        import json, time
        L = inout.labels(str(ROOT))
        bad = [i for i in range(len(y)) if pred[i] >= 0 and pred[i] != y[i]]
        bad.sort(key=lambda i: -prob[i])
        out = {'made': time.strftime('%Y-%m-%dT%H:%M:%S'),
               'items': [{'id': rows[i][0], 'old': L[rows[i][0]]['label'] if isinstance(L[rows[i][0]], dict) else L[rows[i][0]],
                          'model': IT.NAMES[pred[i]], 'p': round(float(prob[i]), 3), 'door': bool(door[i])} for i in bad]}
        json.dump(out, open(inout.folder(str(ROOT)) / 'review.json', 'w'), ensure_ascii=False, indent=0)
        print('review: %d answers the model disagrees with (%d at the door)' % (len(bad), sum(door[i] for i in bad)))
    if refit:
        import pickle
        m = IT.boost().fit(X, y)
        m.days_ = days
        p = ROOT / 'data' / 'door_v2' / 'io_geom.pkl'
        if p.exists():
            p.replace(p.with_suffix('.before_%s.pkl' % __import__('time').strftime('%Y%m%d_%H%M')))
        pickle.dump(m, open(p, 'wb'))
        print('refitted', p)


if __name__ == '__main__':
    main('--refit' in sys.argv, '--review' in sys.argv)
