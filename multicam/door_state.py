"""Entry and exit as a two-zone counter with a learned side map (03.10.2026).

Camera 1 sees the door in the top middle of its frame: the gallery is the strip behind the glass, the vestibule
(with the sample stand where people linger) is below the door, the shop is further down and to the right. A crossing
is a person going from plainly-gallery to plainly-shop (an entry) or back (an exit); the vestibule in between never
changes anyone's state, so lingering at the stand counts nothing.

Side map: where in the frame is 'shop' and where 'gallery', learned from the people who really crossed (the owner's
/door answers, their matched tracks): their feet 1.5-6 s before an entry are outside, after it inside (an exit the
other way round). A boosting on the foot position and the box height gives p(inside) for any person anywhere.

State machine per person (world number; a track that starts at the door right after another ended there nearby is
joined to it): the state is set only where p(inside) > HI (shop) or < LO (gallery); a change of state is the event,
timed between the last tick of the old zone and the first of the new one.

usage: door_state.py DAY=RUN.jsonl.gz ...   (side map fitted on the other days, each day scored held out)"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
NEAR, FAR = 1.5, 6.0           # s from a crossing: the feet are plainly on one side between these
JOIN_S, JOIN_PX = 2.5, 220.0   # a track born this soon and this near (2176 px) after another one ended: the same person


def tracks(ticks, split=None):
    """world -> [(t, person)]; with split (RA_STATE_SPLIT=1, the default) a world is cut back into the tracker's own
    pieces -- at every tick where its slot is a new proposal (the world number was given by appearance, SameHead) and
    at every gap over 1 s -- so that two people the appearance join mixed up are two again."""
    split = os.environ.get('RA_STATE_SPLIT', '0') == '1' if split is None else split
    by, piece, last = {}, {}, {}
    for r in ticks:
        for q in r['p']:
            if not q.get('foot'):
                continue
            w = q['w']
            if split:
                if q.get('new') or (w in last and r['t'] - last[w] > 1.0) or w not in piece:
                    piece[w] = piece.get(w, -1) + 1
                last[w] = r['t']
                key = (w, piece[w])
            else:
                key = w
            by.setdefault(key, []).append((r['t'], q))
    return by


def xfeat(q):
    cx, cy, w, h = q['box']
    return [q['foot'][0], q['foot'][1], h * 1248, w * 2176]


def side_samples(day_data):
    """(X, y) from the really crossing people of a day: y 1 inside, 0 outside."""
    X, y = [], []
    d, by = day_data['learn'], day_data['by']
    import door_learn as L
    for (f, kind, t, w), lab in zip(d['cands'], d['y']):
        if lab != 1:
            continue
        tc = t - L.SHIFT
        seq = by.get(w) or sorted((x for k, v in by.items() if isinstance(k, tuple) and k[0] == w for x in v), key=lambda x: x[0])
        for s, q in seq:
            dt = s - tc
            if NEAR <= abs(dt) <= FAR:
                after = dt > 0
                inside = after if kind == 'in' else not after
                X.append(xfeat(q)); y.append(int(inside))
    return np.array(X, np.float32), np.array(y)


class KernelMap:
    """p(inside) at a foot position and person height: Gaussian-smoothed counts of inside and outside samples on a
    grid (8 px, HBIN px of height), pulled to 0.5 where there are few (PRIOR samples' worth) -- unknown places never
    change anybody's state. The height tells a person standing inside right at the door (near, big) from one beyond
    the glass at the same image point (far, small)."""
    CELL, SIGMA, PRIOR = 8, float(os.environ.get('RA_MAP_SIGMA', '40')), float(os.environ.get('RA_MAP_PRIOR', '3'))
    HBIN, HN, HSIG = 50.0, 24, float(os.environ.get('RA_MAP_HSIG', '1.5'))     # height bins of 50 px up to 1200, smoothed 1.5 bins

    def __init__(self, X, y):
        from scipy.ndimage import gaussian_filter
        self.use_h = os.environ.get('RA_MAP_H', '1') == '1'
        H, W = 1224 // self.CELL + 1, 2176 // self.CELL + 1
        n_h = self.HN if self.use_h else 1
        cin, cout = np.zeros((n_h, H, W), np.float32), np.zeros((n_h, H, W), np.float32)
        gx, gy, gh = self._idx(X, W, H)
        np.add.at(cin, (gh[y == 1], gy[y == 1], gx[y == 1]), 1)
        np.add.at(cout, (gh[y == 0], gy[y == 0], gx[y == 0]), 1)
        k = self.SIGMA / self.CELL
        sig = (self.HSIG if self.use_h else 0, k, k)
        norm = 2 * np.pi * k * k * (np.sqrt(2 * np.pi) * self.HSIG if self.use_h else 1)
        cin = gaussian_filter(cin, sig) * norm
        cout = gaussian_filter(cout, sig) * norm
        self.p = (cin + self.PRIOR / 2) / (cin + cout + self.PRIOR)
        self.W, self.H = W, H

    def _idx(self, X, W, H):
        gx = np.clip((X[:, 0] / self.CELL).astype(int), 0, W - 1)
        gy = np.clip((X[:, 1] / self.CELL).astype(int), 0, H - 1)
        gh = np.clip((X[:, 2] / self.HBIN).astype(int), 0, self.HN - 1) if self.use_h else np.zeros(len(X), int)
        return gx, gy, gh

    def predict_proba(self, X):
        gx, gy, gh = self._idx(X, self.W, self.H)
        p = self.p[gh, gy, gx]
        return np.stack([1 - p, p], 1)


def fit_map(days):
    X = np.concatenate([side_samples(d)[0] for d in days])
    y = np.concatenate([side_samples(d)[1] for d in days])
    return KernelMap(X, y)


def join(by, p_in, lo, hi):
    """world -> joined world: a track that starts within JOIN_S / JOIN_PX of where another ended (not plainly inside
    at its end), is the same person going on."""
    starts = sorted((seq[0][0], w) for w, seq in by.items())
    ends = {w: (seq[-1][0], np.array(seq[-1][1]['foot'])) for w, seq in by.items()}
    parent = {w: w for w in by}

    def root(w):
        while parent[w] != w:
            w = parent[w]
        return w
    for t0, w in starts:
        f0 = np.array(by[w][0][1]['foot'])
        best = None
        for v, (t1, f1) in ends.items():
            if v == w or not (0 <= t0 - t1 <= JOIN_S):
                continue
            d = float(np.linalg.norm(f0 - f1))
            if d <= JOIN_PX and (best is None or d < best[0]):
                best = (d, v)
        if best is not None and root(best[1]) != w:
            parent[w] = root(best[1])
    out = {}
    for w, seq in by.items():
        out.setdefault(root(w), []).extend(seq)
    return {w: sorted(seq, key=lambda x: x[0]) for w, seq in out.items()}


def events(by, model, lo, hi, smooth=2, joined=True):
    allq = [q for seq in by.values() for _, q in seq]
    if not allq:
        return []
    P = model.predict_proba(np.array([xfeat(q) for q in allq], np.float32))[:, 1]
    for q, p in zip(allq, P):
        q['p_in'] = float(p)
    people = join(by, None, lo, hi) if joined else by
    out = []
    for w, seq in people.items():
        p = np.array([q['p_in'] for _, q in seq])
        if smooth:
            p = np.array([np.median(p[max(0, i - smooth):i + smooth + 1]) for i in range(len(p))])
        state, last_t, prev_q = None, None, None
        for (t, q), v in zip(seq, p):
            z = 1 if v > hi else 0 if v < lo else None
            if z is None:
                continue
            if state is not None and z != state:
                out.append({'w': w, 'kind': 'in' if z == 1 else 'out', 't': round((last_t + t) / 2, 2), 'transit': round(t - last_t, 2),
                            'foot_from': list(prev_q['foot']), 'foot_to': list(q['foot'])})
            state, last_t, prev_q = z, t, q
    return out


def main(args):
    import door_learn as L
    import door_v2 as D
    days = []
    for a in args:
        day, path = a.split('=', 1)
        ticks, spans = D._read(path)
        inside = lambda t, sp=spans: any(x <= t <= y for x, y in sp)
        days.append({'day': day, 'learn': L.load_day(day, path), 'by': tracks(ticks),
                     'truth': [t for t in D.truth(day, True) if inside(t['t'])],
                     'visits': [t for t in D.truth(day, False) if inside(t['t'])],
                     'counter': [x for x in D.counter(day) if inside(x['t'])]})
    rep = {}
    grid = list(itertools.product((0.05, 0.1, 0.2, 0.3), (0.7, 0.8, 0.9, 0.95), (True, False)))
    for test in days:
        train = [d for d in days if d is not test]
        # hysteresis picked on the training days (each scored by a map fitted on the remaining one)
        scores = {}
        for d in train:
            others = [e for e in train if e is not d] or train
            m = fit_map(others)
            for lo, hi, jn in grid:
                ev = [dict(e, t=e['t'] + L.SHIFT) for e in events(d['by'], m, lo, hi, joined=jn)]
                r = D.match(ev, d['truth'], L.TOL)
                scores.setdefault((lo, hi, jn), []).append((r['in']['f1'] + r['out']['f1']) / 2)
        best = max(scores, key=lambda k: np.mean(scores[k]))
        m = fit_map(train)
        ev = [dict(e, t=e['t'] + L.SHIFT) for e in events(test['by'], m, *best)]
        r = D.match(ev, test['truth'], L.TOL)
        rv = D.match(D.undither(ev), test['visits'], L.TOL)
        c = D.match(test['counter'], test['truth'], 1.0)
        rep[test['day']] = {'chosen': list(best), 'model': r, 'model_visits': rv, 'counter': c}
        print('%s lo %.2f hi %.2f join %s | in P%.2f R%.2f (%d false, %d missed) | out P%.2f R%.2f (%d false, %d missed) | counter in P%.2f out P%.2f' % (
            test['day'], best[0], best[1], best[2], r['in']['precision'], r['in']['recall'], r['in']['false'], r['in']['true'] - r['in']['hit'],
            r['out']['precision'], r['out']['recall'], r['out']['false'], r['out']['true'] - r['out']['hit'], c['in']['precision'], c['out']['precision']), flush=True)
        json.dump([dict(e, day=test['day']) for e in ev], open(ROOT / 'data' / 'door_v2' / ('state_events_%s.json' % test['day']), 'w'))
    json.dump(rep, open(ROOT / 'data' / 'door_v2' / 'state.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
