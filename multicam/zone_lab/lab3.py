"""13.10.2026, the owner: buffered side per track on doorseg_c tracks.
Per observation a vote with a reason:
  IN  'shop'  -- a mask pixel in the shop zone (visible -> certain)
  IN  'head'  -- mask clearly shorter than the person's expected height there, and the feet from the head are in the shop
  IN  'front' -- mask cut at the bottom by a person in front (lower feet) who is in the shop
  OUT 'full'  -- no shop pixel, the whole body visible (height ~ expected, legs not covered), feet out of the shop
  ?           -- otherwise (cut, covered, unsure)
Track side: majority of the reliable votes in a centred window of W s (looks W/2 ahead: the delay); unknown stretches
take the side of the nearest reliable vote within FILL s on both sides (decided after the occlusion ends).
Event: side change on a track (or a track inheriting its side from a vanished neighbour), through the door zone
within DOOR s; out/in pairs of one track within PAIR s cancel."""
import numpy as np, cv2
import lab2 as L
import rules as R
from sklearn.neighbors import KNeighborsClassifier

# 13.10, the owner's idea: head position + width (+ area per height) say how far the person is -- robust to hidden legs
_X, _Y = [], []
for _t, _d in L.ZF['tags'].items():
    if _d['role'] != 'train':
        continue
    for _r in _d['rows']:
        if _r[R.TRU] in ('in', 'out'):
            _h = max(1, _r[R.Y2] - _r[R.Y1])
            _X.append([(_r[R.X1] + _r[R.X2]) / 2 / 1280, _r[R.Y1] / 720, (_r[R.X2] - _r[R.X1]) / 300, _r[R.N] / _h / 300])
            _Y.append(_r[R.TRU] == 'in')
SIZE = KNeighborsClassifier(32).fit(np.array(_X), np.array(_Y))

FPS = 12.5


def obs_rows(tracks):
    """per track: list of dicts every 3rd tick"""
    out = {}
    for k, ps in tracks.items():
        if k % 3:
            continue
        people = [(pid, p[0][:180]) for pid, p in ps if p[0][:180].any()]
        if not people:
            continue
        info = []
        for pid, m in people:
            ys, xs = np.nonzero(m)
            x1, y1, x2, y2 = xs.min() * 4, ys.min() * 4, xs.max() * 4 + 4, ys.max() * 4 + 4
            j = np.argmax(ys)
            info.append(dict(pid=pid, m=m, x1=x1, y1=y1, x2=x2, y2=y2, fx=xs[j] * 4 + 2, fy=ys[j] * 4 + 3,
                             shop=int((m & L.ZM['shop']).sum()), door=int((m & L.ZM['door']).sum())))
        heads = np.array([[(d['x1'] + d['x2']) / 2, d['y1']] for d in info])
        vf = L.HEAD.predict(heads)
        for i, d in enumerate(info):
            d['vfx'], d['vfy'] = vf[i]
            d['exp_h'] = max(1.0, vf[i][1] - d['y1'])
            d['hr'] = (d['y2'] - d['y1']) / d['exp_h']
            d['foot'] = L.zone_of(d['fx'], d['fy']); d['vfoot'] = L.zone_of(*vf[i])
            hh = max(1, d['y2'] - d['y1'])
            d['p_size'] = float(SIZE.predict_proba([[(d['x1'] + d['x2']) / 2 / 1280, d['y1'] / 720, (d['x2'] - d['x1']) / 300,
                                                     int(d['m'].sum()) * 16 / hh / 300]])[0][1])
            # who covers the legs: a mask under this one's bottom (within the box columns), lower feet = in front
            h = d['y2'] - d['y1']; a, b = int(d['y2'] / 4), int(min(180, (d['y2'] + .6 * d['exp_h'] * (1 - min(1, d['hr']))+ .3 * h) / 4))
            cx0, cx1 = int(d['x1'] / 4), int(d['x2'] / 4) + 1
            d['cov'], d['front_shop'] = 0.0, False
            for e in info:
                if e is d or e['y2'] <= d['y2']:
                    continue
                c = e['m'][max(0, a - 2):b, cx0:cx1]
                if c.size and c.mean() > d['cov']:
                    d['cov'] = float(c.mean()); d['front_shop'] = e['shop'] > 0
            out.setdefault(d['pid'], []).append((k, d))
    return out


def vote(d, c):
    if d['shop'] >= 1:
        return 1, 'shop'
    cut = d['hr'] < c['cut']
    covered = d['cov'] > c['cov']
    if (cut or covered) and c['head'] and d['vfoot'] == 'shop':
        return 1, 'head'
    if covered and c['front'] and d['front_shop']:
        return 1, 'front'
    if not cut and not covered and d['foot'] != 'shop':
        if c.get('size') and d['p_size'] >= c['size']:
            return None, '?'                       # looks whole and outside by the zones, but too big for outside
        return 0, 'full'
    if c.get('size'):
        if d['p_size'] >= c['size']:
            return 1, 'size'
        if d['p_size'] <= 1 - c['size']:
            return 0, 'size'
    return None, '?'


def track_sides(seq, c):
    """seq: [(k, d)] -> per tick side (1/0/None) with the buffered rules"""
    ks = np.array([k for k, _ in seq]); v = [vote(d, c)[0] for _, d in seq]
    t = ks / FPS
    rel = [(t[i], v[i]) for i in range(len(v)) if v[i] is not None]
    side = []
    W, FILL = c['W'], c['FILL']
    for i in range(len(seq)):
        win = [s for tt, s in rel if abs(tt - t[i]) <= W / 2]
        if win:
            m = np.mean(win)
            s = 1 if m > .5 else 0 if m < .5 else None
        else:
            s = None
        if s is None and FILL:
            near = [(abs(tt - t[i]), s_) for tt, s_ in rel if abs(tt - t[i]) <= FILL]
            if near:
                s = min(near)[1]
        side.append(s)
    return ks, side


def events(tracks, c):
    rows = obs_rows(tracks)
    first = {p: s[0][0] for p, s in rows.items()}; last = {p: s[-1][0] for p, s in rows.items()}
    ev, sides, end_state = [], {}, {}
    for p in sorted(rows, key=lambda p: first[p]):
        seq = rows[p]
        ks, side = track_sides(seq, c)
        # inherit the side (never the number) from exactly one neighbour that vanished just before, close by
        d0 = seq[0][1]; h0 = max(1, d0['y2'] - d0['y1'])
        cand = [q for q in end_state if 0 <= (first[p] - last[q]) / FPS <= c['inh_s'] and
                np.hypot(end_state[q][1] - d0['fx'], end_state[q][2] - d0['fy']) / h0 <= c['inh_px']]
        prev = end_state[cand[0]][0] if len(cand) == 1 else None
        cur, door_t = prev, -1e9
        for i, (k, d) in enumerate(seq):
            tt = k / FPS
            if d['door'] >= 2:
                door_t = tt
            s = side[i]
            sides[(p, k)] = s if s is not None else cur
            if s is None or s == cur:
                continue
            if cur is not None and (tt - door_t) <= c['door_s'] + c['W'] / 2:
                ev.append({'kind': 'in' if s == 1 else 'out', 't': tt, 'w': p})
            cur = s
        end_state[p] = (cur, seq[-1][1]['fx'], seq[-1][1]['fy'])
    if c['pair_s']:
        import rules as R
        ev = R.cancel_pairs(ev, c['pair_s'])
    return ev, sides


BASE3 = dict(cut=.75, cov=.3, head=True, front=True, W=3.0, FILL=4.0, inh_s=2.0, inh_px=1.0, door_s=3.0, pair_s=3.0)
