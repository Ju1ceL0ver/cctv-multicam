"""Entry and exit: the plain walk-throughs by geometry, the rest by the learned rule (03.10.2026).

A person who goes from deep in the gallery to deep in the shop (or back) within TRANSIT seconds -- the side map of
door_state.py plainly outside, then plainly inside -- crossed; this is how a two-zone counter works and it does not
need learning. The learned rule (door_learn.py, its held-out day predictions learn_preds_<day>.json at the threshold
it picked) adds what geometry cannot settle: lingering on the threshold, groups. A learned event within DUP_S of a
geometric one of the same direction is the same crossing.

usage: door_hybrid.py DAY=RUN.jsonl.gz ... (after door_learn.py on the same days)"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DUP_S = 4.0


def merge(geo, learned, dup=DUP_S):
    out = [dict(g, src='geo') for g in geo]
    for e in learned:
        if any(g['kind'] == e['kind'] and abs(g['t'] - e['t']) <= dup for g in geo):
            continue
        out.append(dict(e, src='learned'))
    return sorted(out, key=lambda e: e['t'])


def main(args):
    import door_learn as L
    import door_state as S
    import door_v2 as D
    days = []
    for a in args:
        day, path = a.split('=', 1)
        ticks, spans = D._read(path)
        skip = L.skipped(day)
        out_of = lambda t, sk=skip: any(x - 10 <= t <= y + 10 for x, y in sk)
        inside = lambda t, sp=spans, oo=out_of: any(x <= t <= y for x, y in sp) and not oo(t)
        days.append({'day': day, 'learn': L.load_day(day, path), 'by': S.tracks(ticks, split=False), 'out_of': out_of,
                     'truth': [t for t in D.truth(day, True) if inside(t['t'])],
                     'counter': [x for x in D.counter(day) if inside(x['t'])]})
    rep = json.load(open(ROOT / 'data' / 'door_v2' / 'learn.json'))
    res = {}
    grid = list(itertools.product((0.05, 0.1, 0.2), (0.8, 0.9, 0.95), (2.0, 3.0, 5.0, 8.0), (True, False)))
    for test in days:
        train = [d for d in days if d is not test]
        # geometry settings picked on the training days (map of the other training day), with their learned events
        sc = {}
        for d in train:
            others = [e for e in train if e is not d] or train
            m = S.fit_map(others)
            lp = [p for p in json.load(open(ROOT / 'data' / 'door_v2' / ('learn_preds_%s.json' % d['day'])))
                  if p['p'] >= rep['days'][d['day']]['threshold'] and not d['out_of'](p['t'])]
            for lo, hi, tr, jn in grid:
                geo = [dict(e, t=e['t'] + L.SHIFT) for e in S.events(d['by'], m, lo, hi, joined=jn) if e['transit'] <= tr and not d['out_of'](e['t'] + L.SHIFT)]
                r = D.match(merge(geo, lp), d['truth'], L.TOL)
                sc.setdefault((lo, hi, tr, jn), []).append((r['in']['f1'] + r['out']['f1']) / 2)
        best = max(sc, key=lambda k: np.mean(sc[k]))
        m = S.fit_map(train)
        lo, hi, tr, jn = best
        geo = [dict(e, t=e['t'] + L.SHIFT) for e in S.events(test['by'], m, lo, hi, joined=jn) if e['transit'] <= tr and not test['out_of'](e['t'] + L.SHIFT)]
        lp = [p for p in json.load(open(ROOT / 'data' / 'door_v2' / ('learn_preds_%s.json' % test['day'])))
              if p['p'] >= rep['days'][test['day']]['threshold'] and not test['out_of'](p['t'])]
        for name, ev in (('geometry only', geo), ('learned only', lp), ('hybrid', merge(geo, lp))):
            r = D.match(ev, test['truth'], L.TOL)
            n_in_t = sum(t['kind'] == 'in' for t in test['truth']); n_in_p = sum(e['kind'] == 'in' for e in ev)
            print('%s %-13s in P%.2f R%.2f (%d false, %d missed) | out P%.2f R%.2f (%d false, %d missed) | entries true %d counted %d' % (
                test['day'], name, r['in']['precision'], r['in']['recall'], r['in']['false'], r['in']['true'] - r['in']['hit'],
                r['out']['precision'], r['out']['recall'], r['out']['false'], r['out']['true'] - r['out']['hit'], n_in_t, n_in_p), flush=True)
            res.setdefault(test['day'], {})[name] = r
        res[test['day']]['chosen'] = list(best)
        json.dump([dict(e, day=test['day']) for e in merge(geo, lp)], open(ROOT / 'data' / 'door_v2' / ('hybrid_events_%s.json' % test['day']), 'w'))
        c = D.match(test['counter'], test['truth'], 1.0)
        print('%s counter        in P%.2f out P%.2f | chosen lo %.2f hi %.2f transit<=%.0f s join %s' % (test['day'], c['in']['precision'], c['out']['precision'], *best), flush=True)
    json.dump(res, open(ROOT / 'data' / 'door_v2' / 'hybrid.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
