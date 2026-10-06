"""The inside/outside classifier (place boosting, inout_train.geometry) before and after the owner's door batch
(06.10.2026): every day scored by a model of the other days; accuracy on the door samples and on all.
Then (with --refit) data/door_v2/io_geom.pkl is refitted on every answer, the door batch included."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(refit=False):
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
    for name, use_door in (('without the door batch', False), ('with the door batch', True)):
        pred = np.full(len(y), -1)
        for d in days:
            tr = (day != d) & (use_door | ~door)
            te = day == d
            if te.sum() == 0 or len(set(y[tr])) < 2:
                continue
            pred[te] = IT.boost().fit(X[tr], y[tr]).predict(X[te])
        ok = pred >= 0
        acc = lambda m: float((pred[m & ok] == y[m & ok]).mean()) if (m & ok).any() else float('nan')
        io_err = lambda m: int(((pred != y) & (y < 2) & (pred < 2) & m & ok).sum())
        print('%-24s all %.3f | door batch %.3f (%d of %d wrong, inside<->outside %d) | old samples %.3f' % (
            name, acc(np.ones(len(y), bool)), acc(door), int(((pred != y) & door & ok).sum()), int((door & ok).sum()), io_err(door), acc(~door)))
    if refit:
        import pickle
        m = IT.boost().fit(X, y)
        m.days_ = days
        p = ROOT / 'data' / 'door_v2' / 'io_geom.pkl'
        if p.exists():
            p.replace(p.with_suffix('.before_door_batch.pkl'))
        pickle.dump(m, open(p, 'wb'))
        print('refitted', p)


if __name__ == '__main__':
    main('--refit' in sys.argv)
