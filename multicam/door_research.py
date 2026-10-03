"""Hypotheses about the door, tested on the runs that exist (03.10.2026). Read-only, CPU.

H1 why a real crossing is not among the candidates: a candidate at a wider tolerance (timing), a track that ends
   on one side and another that starts on the other side near then (a broken track), nobody tracked near the door
   then (detection), or something else
H2 the time of a right candidate minus the owner's event time: is there a constant shift
H3 a door line learned from the owner's answers: where the feet of the right candidates were at the event's time;
   a crossing = the feet go from beyond one line to beyond the other (two lines +-margin around it); fitted on two
   days, scored on the third
H4 the learned rule's precision and recall on the held-out day for every threshold (is the picked one far off)

usage: door_research.py DAY=RUN.jsonl.gz ...  -> data/door_v2/research.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def tracks(ticks):
    by = {}
    for r in ticks:
        for q in r['p']:
            by.setdefault(q['w'], []).append((r['t'], q))
    return by


def foot_at(seq, t, win=0.6):
    f = [q['foot'] for s, q in seq if abs(s - t) <= win and q.get('foot')]
    return np.median(f, 0) if f else None


def side_of(q):
    return int(np.argmax(q.get('io_mix', q['io'])))


def h1_h2(d, by):
    import door_v2 as D
    cand = [{'kind': c[1], 't': c[2], 'w': c[3]} for c in d['cands']]
    out = {'missed': [], 'offsets': {'in': [], 'out': []}}
    for kind in ('in', 'out'):
        T = [t for t in d['truth'] if t['kind'] == kind]
        P = [c for c in cand if c['kind'] == kind]
        from scipy.optimize import linear_sum_assignment
        if not T or not P:
            continue
        C = np.array([[abs(p['t'] - t['t']) for t in T] for p in P])
        r, c = linear_sum_assignment(np.where(C <= 6.0, C, 1e6))
        hitT = {j for i, j in zip(r, c) if C[i, j] <= 6.0}
        out['offsets'][kind] += [float(P[i]['t'] - T[j]['t']) for i, j in zip(r, c) if C[i, j] <= 6.0]
        for j, t in enumerate(T):
            if j in hitT:
                continue
            wide = min((abs(p['t'] - t['t']) for p in P), default=99)
            ends = [(w, seq[-1][0] - t['t'], side_of(seq[-1][1])) for w, seq in by.items() if abs(seq[-1][0] - t['t']) <= 8]
            starts = [(w, seq[0][0] - t['t'], side_of(seq[0][1])) for w, seq in by.items() if abs(seq[0][0] - t['t']) <= 8]
            near = [w for w, seq in by.items() if any(abs(s - t['t']) <= 2 for s, _ in seq)]
            want_from, want_to = (0, 1) if kind == 'in' else (1, 0)
            broken = any(e[2] != want_to for e in ends) and any(s[2] == want_to for s in starts)
            why = ('timing' if wide <= 12 else 'broken_track' if broken else 'nobody_tracked' if not near else 'other')
            out['missed'].append({'day': d['day'], 'kind': kind, 't': round(t['t'], 1), 'why': why, 'nearest_candidate_s': round(wide, 1),
                                  'ends': [(e[0] % 10000, round(e[1], 1), 'oid'[e[2]]) for e in ends],
                                  'starts': [(s[0] % 10000, round(s[1], 1), 'oid'[s[2]]) for s in starts], 'people_near': len(near)})
    return out


def line_points(d, by):
    """Feet of the right candidates' tracks at the owner's event time, with the direction."""
    import door_learn as L
    pts = []
    for (f, kind, t, w), y in zip(d['cands'], d['y']):
        if y != 1:
            continue
        gt = min((x for x in d['truth'] if x['kind'] == kind), key=lambda x: abs(x['t'] - t))
        p = foot_at(by[w], gt['t'])
        pre, post = foot_at(by[w], gt['t'] - 1.5, 0.5), foot_at(by[w], gt['t'] + 1.5, 0.5)
        if p is not None and pre is not None and post is not None:
            pts.append((p, post - pre if kind == 'in' else pre - post))     # the inward direction
    return pts


def fit_line(pts):
    P = np.array([p for p, _ in pts])
    c = P.mean(0)
    u = np.linalg.svd(P - c)[2][0]                     # along the line
    n = np.array([-u[1], u[0]])
    inward = np.mean([v for _, v in pts], 0)
    if n @ inward < 0:
        n = -n                                         # the normal points into the shop
    return c, u, n


def line_crossings(by, c, n, margin, half_len, u, min_hold=0.3):
    out = []
    for w, seq in by.items():
        sd, ts = [], []
        for t, q in seq:
            if not q.get('foot'):
                continue
            f = np.array(q['foot'])
            if abs((f - c) @ u) > half_len:                # far along the line: not the door
                sd.append(np.nan)
            else:
                sd.append((f - c) @ n)
            ts.append(t)
        side = [None if not np.isfinite(v) else (1 if v > margin else 0 if v < -margin else None) for v in sd]
        last, last_t, run0 = None, None, None
        for t, s in zip(ts, side):
            if s is None:
                continue
            if last is None:
                last, last_t, run0 = s, t, t
                continue
            if s != last:
                out.append({'w': w, 'kind': 'in' if s == 1 else 'out', 't': round((last_t + t) / 2, 2)})
                last, run0 = s, t
            last_t = t
    return out


def h4(rep_days, days):
    """precision / recall of the learned rule on each held-out day at every threshold."""
    import door_learn as L
    import door_v2 as D
    out = {}
    names = None
    for test in days:
        train = [d for d in days if d is not test]
        X = np.concatenate([L.matrix(d['cands'], names)[0][d['y'] >= 0] for d in train])
        names = L.matrix(train[0]['cands'])[1]
        y = np.concatenate([d['y'][d['y'] >= 0] for d in train])
        clf = L.model().fit(X, y)
        p = clf.predict_proba(L.matrix(test['cands'], names)[0])[:, 1]
        rows = []
        for thr in np.arange(0.05, 0.96, 0.05):
            k = L.nms(test['cands'], p, thr)
            m = D.match([{'kind': test['cands'][i][1], 't': test['cands'][i][2], 'w': test['cands'][i][3]} for i in k], test['truth'], L.TOL)
            rows.append([round(float(thr), 2), m['in']['precision'], m['in']['recall'], m['out']['precision'], m['out']['recall']])
        out[test['day']] = rows
    return out


def main(args):
    import door_learn as L
    import door_v2 as D
    days = []
    for a in args:
        day, path = a.split('=', 1)
        d = L.load_day(day, path)
        ticks, _ = D._read(path)
        D.add_io(ticks, path)
        d['by'] = tracks(ticks)
        days.append(d)
    rep = {'h1': [], 'h2': {'in': [], 'out': []}}
    for d in days:
        r = h1_h2(d, d['by'])
        rep['h1'] += r['missed']
        for k in ('in', 'out'):
            rep['h2'][k] += r['offsets'][k]
    from collections import Counter
    rep['h1_counts'] = dict(Counter(m['why'] for m in rep['h1']))
    rep['h2'] = {k: {'n': len(v), 'median_s': round(float(np.median(v)), 2), 'p10': round(float(np.percentile(v, 10)), 2),
                     'p90': round(float(np.percentile(v, 90)), 2)} for k, v in rep['h2'].items() if v}
    # H3: the line from two days, scored on the third, several margins
    rep['h3'] = {}
    for test in days:
        pts = sum((line_points(d, d['by']) for d in days if d is not test), [])
        c, u, n = fit_line(pts)
        P = np.array([p for p, _ in pts])
        along = (P - c) @ u
        half = float(np.percentile(np.abs(along), 98)) + 60
        across = (P - c) @ n
        res = {'points': len(pts), 'centre': c.round(1).tolist(), 'along': u.round(3).tolist(), 'half_len_px': round(half),
               'feet_across_px': [round(float(v), 1) for v in np.percentile(across, [5, 25, 50, 75, 95])]}
        for margin in (20, 40, 60, 90, 120):
            cs = line_crossings(test['by'], c, n, margin, half, u)
            m = D.match(cs, test['truth'], 6.0)
            res['margin_%d' % margin] = {k: [m[k]['precision'], m[k]['recall'], m[k]['false']] for k in ('in', 'out')}
            mu = D.match(D.undither(cs), test['visits_truth'], 6.0)
            res['margin_%d_visits' % margin] = {k: [mu[k]['precision'], mu[k]['recall'], mu[k]['false']] for k in ('in', 'out')}
        rep['h3'][test['day']] = res
    rep['h4'] = h4(rep, days)
    json.dump(rep, open(ROOT / 'data' / 'door_v2' / 'research.json', 'w'), indent=1)
    print(json.dumps({k: rep[k] for k in ('h1_counts', 'h2', 'h3')}, indent=1))
    for d, rows in rep['h4'].items():
        print(d, 'thr, in P R, out P R:', rows)


if __name__ == '__main__':
    main(sys.argv[1:])
