"""A learned check of every candidate crossing (08.10.2026): candidates generously (every side flip of a stitched track,
one vote enough, no gate), each described by what happened around it (the mask's depth past the line and the model's p
before/after, the feet's move along the door axis, the distance to the drawn door, how much the track flickered, its
length...), and a small boosting model, learned on the owner's /doorside crossings, says which are real. Leave one day
out: fitted on two days (its threshold too), scored on the third. -> data/door_v2/event_clf_<src>.json (+ .pkl, all days)

usage: RA_BIN_SRC=micro1s6 door_event_clf.py"""
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault('RA_BIN_SRC', 'micro1s6')
os.environ.setdefault('RA_BIN_VER', 'ldpc2')
os.environ.setdefault('RA_POS_VER', 'pc')
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'door_v2'
FEATS = ['kind', 'd_at', 'd_before', 'd_after', 'd_min_after', 'd_max_before', 'p_before', 'p_after', 'n_new', 'n_old',
         'move', 'seg_px', 'zone', 'len', 'since_start', 'to_end', 'flips_near', 'x', 'y', 'h', 'top', 'birthlike']


def candidates(tracks, line, u, hz):
    """tracks: {name: [(t, vote, xy, d, p, h, top)]} -> [(feature dict, event)]"""
    import door_line
    out = []
    for name, xs in tracks.items():
        ts = np.array([x[0] for x in xs])
        side, flips = None, []
        for i, x in enumerate(xs):
            v = x[1]
            if v not in (0.0, 1.0):
                continue
            if side is not None and v != side:
                flips.append(i)
            side = v
        for i in flips:
            t, v, xy, d, p, h, top = xs[i]
            bef = [x for x in xs if t - 2.0 <= x[0] < t]
            aft = [x for x in xs if t <= x[0] <= t + 2.0]
            dv = lambda rs: [r[3] for r in rs if r[3] is not None]
            pv = lambda rs: [r[4] for r in rs if r[4] is not None]
            b_xy = [r[2] for r in xs if t - 1.5 <= r[0] < t and r[2]]
            a_xy = [r[2] for r in xs if t < r[0] <= t + 1.5 and r[2]]
            move = float((np.median(a_xy, 0) - np.median(b_xy, 0)) @ u) if a_xy and b_xy else 0.0
            kind = 'in' if v == 1.0 else 'out'
            f = {'kind': 1.0 if kind == 'in' else 0.0,
                 'd_at': d if d is not None else -999.0,
                 'd_before': float(np.mean(dv(bef))) if dv(bef) else -999.0,
                 'd_after': float(np.mean(dv(aft))) if dv(aft) else -999.0,
                 'd_min_after': float(np.min(dv(aft))) if dv(aft) else -999.0,
                 'd_max_before': float(np.max(dv(bef))) if dv(bef) else -999.0,
                 'p_before': float(np.mean(pv(bef))) if pv(bef) else -1.0,
                 'p_after': float(np.mean(pv(aft))) if pv(aft) else -1.0,
                 'n_new': float(sum(1 for r in aft if r[1] == v)),
                 'n_old': float(sum(1 for r in bef if r[1] == 1.0 - v)),
                 'move': move * (1 if kind == 'in' else -1),
                 'seg_px': door_line.seg_dist(line, xy[0] * 1280, xy[1] * 720) if xy else 999.0,
                 'zone': float(np.min(np.hypot(*(hz - np.asarray(xy)).T))) if xy and len(hz) else 9.0,
                 'len': float(ts[-1] - ts[0]), 'since_start': float(t - ts[0]), 'to_end': float(ts[-1] - t),
                 'flips_near': float(sum(1 for j in flips if abs(xs[j][0] - t) <= 10.0)),
                 'x': xy[0] if xy else -1.0, 'y': xy[1] if xy else -1.0, 'h': h or 0.0, 'top': top or 0.0,
                 'birthlike': 0.0}
            out.append((f, {'kind': kind, 't': t, 'w': name, 'xy': xy}))
    return out


def main():
    import door_combo as C
    import door_learn as L
    import door_line
    import door_side_tune as ST
    import door_v2 as D
    from sklearn.ensemble import HistGradientBoostingClassifier
    days, line, table = ST.load()
    f_ = json.load(open(OUT / 'combo_final.json'))
    u, hz = np.array(f_['u']), np.array(f_['zone'])
    sd = json.load(open(OUT / 'side_final.json'))
    stride = int(os.environ.get('RA_ZONE_STRIDE', '6'))
    data = {}
    for day, v in days.items():
        T = {}
        for tag, xs in v['obs'].items():
            per = {}
            for o in xs:
                if o['k'] % stride:
                    continue
                xy = v['dd'].P.get('%.2f_%d' % (o['t'], o['pid']))
                vt = door_line.side_vote(sd, table, line, o['d'], o['top'], o['h'], o['bx'], o['p'])
                per.setdefault(o['pid'], []).append((o['t'], vt, xy, o['d'], o['p'], o['h'], o['top']))
            per = {k: sorted(x) for k, x in per.items()}
            per = C.stitch_tracks({k: [(a[0], a[1], a[2]) + a[3:] for a in x] for k, x in per.items()}, tuple(f_['stitch']))
            for r, lst in per.items():
                T['%s:%s' % (tag, r)] = lst
        cands = candidates(T, line, u, hz)
        # labels: the owner's crossings, one to one, same kind, within TOL (door_learn.label)
        sp = [(x, y) for x, y, r in v['spans']]
        inside = lambda t: any(x - 1 <= t <= y + 1 for x, y in sp)
        cands = [(f, e) for f, e in cands if inside(e['t'] + L.SHIFT)]
        y = L.label([(None, e['kind'], e['t'] + L.SHIFT, e['w']) for _, e in cands], v['truth'])
        data[day] = (np.array([[f[k] for k in FEATS] for f, _ in cands], np.float32), np.array(y), [e for _, e in cands])
        print(day, len(cands), 'candidates,', int(np.sum(y)), 'real; truth', len(v['truth']), flush=True)

    def f1_of(day, keep):
        v = days[day]
        tot = [0] * 6
        for role in ('train', 'test'):
            sp = [(x, y) for x, y, r in v['spans'] if r == role]
            if not sp:
                continue
            inside = lambda t: any(x - 1 <= t <= y + 1 for x, y in sp)
            m = D.match([dict(e, t=e['t'] + L.SHIFT) for e in keep if inside(e['t'] + L.SHIFT)],
                        [t for t in v['truth'] if t['role'] == role], L.TOL)
            tot = [a + b for a, b in zip(tot, [m['in']['hit'], m['in']['pred'], m['in']['true'], m['out']['hit'], m['out']['pred'], m['out']['true']])]
        return tot
    f1 = lambda c: ((2 * c[0] / max(1, c[1] + c[2])) + (2 * c[3] / max(1, c[4] + c[5]))) / 2
    ds = sorted(data)
    rep = {'lodo': {}}
    TOT = [0] * 6
    for d in ds:
        tr = [x for x in ds if x != d]
        X = np.concatenate([data[x][0] for x in tr]); Y = np.concatenate([data[x][1] for x in tr])
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
                                             l2_regularization=1.0, random_state=0).fit(X, Y)
        # the threshold: best F1 on the training days (each scored by the model of the other training day would be
        # stricter; with two days we take the in-sample one)
        best = (0, 0.5)
        for thr in np.arange(0.1, 0.91, 0.05):
            c = [0] * 6
            for x in tr:
                pr = clf.predict_proba(data[x][0])[:, 1]
                keep = C.alternate([e for e, p in zip(data[x][2], pr) if p >= thr], 'first')
                c = [a + b for a, b in zip(c, f1_of(x, keep))]
            if f1(c) > best[0]:
                best = (f1(c), float(thr))
        pr = clf.predict_proba(data[d][0])[:, 1]
        keep = C.alternate([e for e, p in zip(data[d][2], pr) if p >= best[1]], 'first')
        c = f1_of(d, keep)
        TOT = [a + b for a, b in zip(TOT, c)]
        rep['lodo'][d] = {'thr': best[1], 'counts': c, 'f1': round(f1(c), 3)}
        print('LODO', d, json.dumps(rep['lodo'][d]), flush=True)
    rep['lodo_total'] = {'counts': TOT, 'f1': round(f1(TOT), 3)}
    print('LODO total', json.dumps(rep['lodo_total']), flush=True)
    X = np.concatenate([data[x][0] for x in ds]); Y = np.concatenate([data[x][1] for x in ds])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
                                         l2_regularization=1.0, random_state=0).fit(X, Y)
    thr = float(np.median([rep['lodo'][d]['thr'] for d in ds]))
    pickle.dump({'clf': clf, 'feats': FEATS, 'thr': thr, 'stride': stride}, open(OUT / ('event_clf_%s.pkl' % os.environ['RA_BIN_SRC']), 'wb'))
    json.dump(rep, open(OUT / ('event_clf_%s.json' % os.environ['RA_BIN_SRC']), 'w'), indent=1)
    print('done', flush=True)


if __name__ == '__main__':
    main()
