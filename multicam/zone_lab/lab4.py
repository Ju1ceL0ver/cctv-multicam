"""side per observation by a small boosting model (geometry + two-sided shop memory), trained on the teacher rows of
'train' stretches; smoothed over +-SM s of the same track; side held between confident answers; events through the door
zone; side (never the number) inherited from exactly one track that vanished just before nearby."""
import numpy as np
import rules as R, lab2 as L, hyp as Hy
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

FPS = 12.5
_Y = np.array([b for _, _, b, _ in Hy.S['train']])
MODEL_MEM = None                                  # built on first use (mem=True): only the geometry model is needed by default


def _mem_model():
    global MODEL_MEM
    if MODEL_MEM is None:
        _X = [Hy.base(r) + Hy.extra(t, r, 'mem') for t, r, _, _ in Hy.S['train']]
        MODEL_MEM = HGB(max_iter=300, random_state=0).fit(np.array(_X), _Y)
    return MODEL_MEM
MODEL_GEO = HGB(max_iter=300, random_state=0).fit(np.array([Hy.base(r) for _, r, _, _ in Hy.S['train']]), _Y)


def mem(s, i, W):
    k = s[i][R.K]
    past = [q for q in s[:i] if q[R.K] >= k - W * FPS]; fut = [q for q in s[i + 1:] if q[R.K] <= k + W * FPS]
    return [float(any(q[R.SHOP] > 0 for q in past)), float(any(q[R.SHOP] > 0 for q in fut)),
            float(any(q[R.FOOT] == 'gallery' for q in past)), float(any(q[R.FOOT] == 'gallery' for q in fut))]


def events(rows, c):
    seq = {}
    for r in rows:
        seq.setdefault(r[R.PIECE], []).append(r)
    for s in seq.values():
        s.sort(key=lambda r: r[R.K])
    first = {p: s[0][R.K] for p, s in seq.items()}; last = {p: s[-1][R.K] for p, s in seq.items()}
    P = {}
    for p, s in seq.items():
        if c.get('mem'):
            X = [Hy.base(r) + sum((mem(s, i, W) for W in (1, 3, 6)), []) for i, r in enumerate(s)]
            P[p] = _mem_model().predict_proba(np.array(X))[:, 1]
        else:
            memo = c.get('memo')                       # 14.10 live: one prediction per observation, ever
            if memo is None:
                P[p] = MODEL_GEO.predict_proba(np.array([Hy.base(r) for r in s]))[:, 1]
            else:
                new = [r for r in s if (p, r[R.K]) not in memo]
                if new:
                    for r, pr in zip(new, MODEL_GEO.predict_proba(np.array([Hy.base(r) for r in new]))[:, 1]):
                        memo[(p, r[R.K])] = float(pr)
                P[p] = np.array([memo[(p, r[R.K])] for r in s])
    ev, sides, end = [], {}, {}
    for p in sorted(seq, key=lambda p: first[p]):
        s, pr = seq[p], P[p]
        ks = np.array([r[R.K] for r in s])
        sm = np.array([pr[np.abs(ks - k) <= c['SM'] * FPS].mean() for k in ks]) if c['SM'] else pr
        r0 = s[0]; h0 = max(1, r0[R.Y2] - r0[R.Y1])
        cand = [q for q in end if 0 <= (first[p] - last[q]) / FPS <= c['inh_s'] and
                np.hypot(end[q][1] - r0[R.FX], end[q][2] - r0[R.FY]) / h0 <= c['inh_px']]
        if c.get('inh_overlap'):
            def ov(q):
                a = seq[q][-1]; ix = max(0, min(a[R.X2], r0[R.X2]) - max(a[R.X1], r0[R.X1])); iy = max(0, min(a[R.Y2], r0[R.Y2]) - max(a[R.Y1], r0[R.Y1]))
                u = (a[R.X2] - a[R.X1]) * (a[R.Y2] - a[R.Y1]) + (r0[R.X2] - r0[R.X1]) * (r0[R.Y2] - r0[R.Y1]) - ix * iy
                return u > 0 and ix * iy / u >= c['inh_overlap']
            cand = [q for q in cand if ov(q)]
        cur = end[cand[0]][0] if len(cand) == 1 else None
        door_t, pend, n, t0 = -1e9, None, 0, None
        for r, q in zip(s, sm):
            t = r[R.K] / FPS
            if r[R.DOOR] >= c['door_px']:
                door_t = t
            v = 1 if q >= c['thr'] else (0 if q <= 1 - c['thr'] else None)
            if c.get('rules_first'):     # 14.10 owner: his rules above the classifier, the model only where they are silent
                if r[R.SHOP] >= c['rules_first']:
                    v = 1
                elif r[R.SHOP] == 0 and r[R.FOOT] == 'gallery':
                    v = 0
                elif r[R.SHOP] == 0 and v == 1:
                    v = None
            if v == 1 and c.get('shop_veto') and r[R.SHOP] == 0:     # 14.10 owner: inside only with a pixel in the shop
                v = 0 if c['shop_veto'] == 'out' else None
            if v is None or v == cur:
                pend, n = None, 0
            else:
                if pend != v:
                    pend, n, t0 = v, 0, t
                n += 1
                if n >= c['conf']:
                    if cur is not None and (t0 - door_t) <= c['door_s'] + c['SM']:
                        ev.append({'kind': 'in' if v == 1 else 'out', 't': t0, 'w': p,
                                   'x': r[R.FX], 'y': r[R.FY], 'h': max(1, r[R.Y2] - r[R.Y1])})   # 21.10: where, for burst_s
                        if c.get('group_w') and R.WIDTH is not None:
                            exp = R.WIDTH(r)
                            if exp and (r[R.X2] - r[R.X1]) > c['group_w'] * exp:
                                ev.append({'kind': 'in' if v == 1 else 'out', 't': t0 + .01, 'w': p + '#2'})
                    cur, pend, n = v, None, 0
            sides[(p, r[R.K])] = cur
        end[p] = (cur, s[-1][R.FX], s[-1][R.FY])
    if c.get('door_near') is not None:   # 15.10 owner: an event far from the door is false -- touch the doorway at the flip itself
        ok = []
        for e in ev:
            s = seq.get(e['w'], [])
            near = [r for r in s if abs(r[R.K] / FPS - e['t']) <= c['door_near']]
            if any(r[R.DOOR] >= c['door_px'] for r in near):
                # 22.10 owner: a passer-by behind the glass touches the doorway only with his picture -- the feet
                # (or, hidden, the feet from the head) must stand in one of the door_foot zones near the flip
                zf = c.get('door_foot')
                tol = c.get('door_foot_tol', 0)      # 22.10: the doorway is a thin strip; rows come every 0.24 s
                def in_door(r):
                    if not tol:
                        return False
                    import cv2
                    return cv2.pointPolygonTest(L.POLY['door'], (float(r[R.FX]), float(r[R.FY])), True) >= -tol
                if not zf or any(r[R.FOOT] in zf or in_door(r) or
                                 (c.get('door_vfoot') and len(r) > R.VFOOT and r[R.VFOOT] in zf) for r in near):
                    ok.append(e)
        ev = ok
    if c.get('foot_s'):          # 14.10 owner: passers-by behind the glass -- an entry must stand in the shop after, an exit before
        ok = []
        for e in ev:
            s = seq.get(e['w'], [])
            lo, hi = (e['t'], e['t'] + c['foot_s']) if e['kind'] == 'in' else (e['t'] - c['foot_s'], e['t'])
            win = [r for r in s if lo <= r[R.K] / FPS <= hi]
            shop = sum(r[R.FOOT] == 'shop' for r in win)
            gal = sum(r[R.FOOT] == 'gallery' for r in win)
            if shop >= c.get('foot_n', 1) and not (c.get('foot_gal') and gal > shop):
                ok.append(e)
        ev = ok
    if c['pair_s']:
        ev = R.cancel_pairs(ev, c['pair_s'])
    if c.get('burst_s'):         # 21.10 owner: someone lingering in the doorway -- new track numbers, no new crossings
        ev = R.cancel_bursts(ev, c['burst_s'], c.get('burst_px', 1.0), c.get('burst_dup', 0))
    return ev, sides


BASE4 = dict(mem=False, SM=2.0, thr=.9, conf=1, door_px=20, door_s=3.0, pair_s=3.0, inh_s=2.0, inh_px=1.0)
