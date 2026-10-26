"""13.10.2026: deterministic zone rules + track consistency -> door events, scored against the owner's /doorside events.
usage: python3 rules.py zone_feats.json.gz [grid]"""
import gzip, itertools, json, sys
from collections import defaultdict
import numpy as np

K, PIECE, ROOT, TRU, X1, Y1, X2, Y2, FX, FY, N, BAND, SHOP, DOOR, GAL, LEGS, FOOT, VFOOT = range(18)
FPS = 12.5
BASE = dict(shop=20, first=390, conf=2, door=20, door_win=3.0, inherit_s=2.0, inherit_px=1.0, gal_out=True,
            need_door=True, silent=True, alt=True, birth_entry=False, knn=0, knn_k=15)


def votes(rows, c, prior=None):
    """per observation: 1 inside / 0 outside / None no word, with the track memory rule built in later"""
    out = []
    for r in rows:
        if r[SHOP] >= c['shop']:
            v = 1
        elif c['gal_out'] and r[SHOP] == 0 and r[FOOT] == 'gallery':
            v = 0
        else:
            v = None
        out.append(v)
    return out


def _foot(r):
    return r[-1] if isinstance(r[-1], str) else 'gal' if r[GAL] > 0 and r[SHOP] == 0 else 'x'


def run_tag(tag, c, knn=None, tag_name=None):
    rows = sorted(tag['rows'], key=lambda r: r[K])
    by = defaultdict(list)
    for r in rows:
        by[r[PIECE]].append(r)
    first = {p: xs[0][K] for p, xs in by.items()}
    last = {p: xs[-1][K] for p, xs in by.items()}
    order = sorted(by, key=lambda p: first[p])
    state_end, person, end_xy = {}, {}, {}
    events = []
    side_obs = {}
    for p in order:
        xs = by[p]
        h0 = max(1, xs[0][Y2] - xs[0][Y1])
        # inheritance: a track that ended just before, close to where this one starts
        best, ncand = None, 0
        for q in state_end:
            gap = (first[p] - last[q]) / FPS
            if 0 <= gap <= c['inherit_s']:
                d = np.hypot(end_xy[q][0] - xs[0][FX], end_xy[q][1] - xs[0][FY]) / h0
                if d <= c['inherit_px']:
                    ncand += 1
                    if best is None or d < best[0]:
                        best = (d, q)
        if c.get('inherit_unique') and ncand > 1:
            best = None
        if best is not None and c.get('inherit_overlap'):
            q = best[1]; a = by[q][-1]; b0 = xs[0]
            ix = max(0, min(a[X2], b0[X2]) - max(a[X1], b0[X1])); iy = max(0, min(a[Y2], b0[Y2]) - max(a[Y1], b0[Y1]))
            u = (a[X2] - a[X1]) * (a[Y2] - a[Y1]) + (b0[X2] - b0[X1]) * (b0[Y2] - b0[Y1]) - ix * iy
            if u <= 0 or ix * iy / u < c['inherit_overlap']:
                best = None
        side, seen_shop, door_t = None, False, -1e9
        person[p] = p
        if best is not None:
            q = best[1]
            side, seen_shop = state_end[q], state_end[q] == 1
            if c.get('inherit_person', True):
                person[p] = person[q]
        if side is None and c.get('vfoot_birth') and len(xs[0]) > VFOOT and xs[0][VFOOT] == 'shop' and xs[0][SHOP] == 0:
            side, seen_shop = 1, True                     # born with hidden legs, the head says the feet are in the shop
        pend, n, t0 = None, 0, None
        door_after_shop = False
        hist_h = []
        for r in xs:
            t = r[K] / FPS
            if r[DOOR] >= c['door']:
                door_t = t
                if seen_shop:
                    door_after_shop = True
            if r[SHOP] >= c['shop'] and (seen_shop or r[SHOP] >= c['first'] or side == 1):
                v = 1
            elif seen_shop and not door_after_shop:
                v = None                                  # tree rule 2: legs hidden, still inside
            elif knn is not None and r[FOOT] in ('band', 'door', 'none'):
                v = knn(r)
            else:
                v = 0                                     # tree rule 3
                if c.get('vfoot_hold') and len(r) > VFOOT and r[VFOOT] == 'shop' and (r[LEGS] > .3 or side == 1):
                    v = None                              # legs hidden, the head says the feet are in the shop
            if c.get('clf') is not None:
                p_ = c['clf'](tag_name, r)
                v = 1 if p_ >= c['clf_hi'] else (0 if p_ <= 1 - c['clf_hi'] else None)
                if c.get('clf_shop_wins') and r[SHOP] >= c['shop']:
                    v = 1
            if v == 0 and c.get('out_off_door') and side == 1 and r[FOOT] == 'door':
                v = None                                  # standing in the doorway: not out yet
            if v == 0 and c.get('out_px') and side == 1 and r[SHOP] > 0:
                v = None
            hh = r[Y2] - r[Y1]
            if v is not None and c.get('occl_legs') is not None and pend != side and r[LEGS] > c['occl_legs']:
                v = None                                  # someone covers where the legs are: no word on the side
            if v is not None and c.get('occl_h') and hist_h and hh < c['occl_h'] * np.median(hist_h[-20:]):
                v = None                                  # the mask got much shorter: legs hidden
            if r[LEGS] < 0.2:
                hist_h.append(hh)
            if v is None:
                side_obs[(p, r[K])] = side
                pend, n = None, 0
                continue
            if v == 1:
                seen_shop, door_after_shop = True, False
            if v == side:
                pend, n = None, 0
            else:
                if pend != v:
                    pend, n, t0 = v, 0, t
                n += 1
                if n >= c['conf']:
                    if side is not None:
                        through = (t0 - door_t) <= c['door_win'] or not c['need_door']
                        if through or not c['silent']:
                            events.append({'kind': 'in' if v == 1 else 'out', 't': t0, 'w': person[p]})
                            if c.get('group_w') and WIDTH is not None:
                                exp = WIDTH(r)
                                if exp and (r[X2] - r[X1]) > c['group_w'] * exp:
                                    events.append({'kind': 'in' if v == 1 else 'out', 't': t0 + .01, 'w': person[p] + '#2'})
                    side, pend, n = v, None, 0
            side_obs[(p, r[K])] = side
        state_end[p] = side
        end_xy[p] = (xs[-1][FX], xs[-1][FY])
    global LAST_PERSON
    LAST_PERSON = person
    if c.get('pair_s'):
        events = cancel_pairs(events, c['pair_s'])
    if c['alt']:
        events = alternate(events, c.get('alt_s'))
    return events, side_obs


def cancel_pairs(ev, T):
    """out and back in (or in and back out) of the same person within T s: never really crossed"""
    ev = sorted(ev, key=lambda e: e['t'])
    drop = set()
    for i, e in enumerate(ev):
        if i in drop:
            continue
        for j in range(i + 1, len(ev)):
            f = ev[j]
            if f['t'] - e['t'] > T:
                break
            if j not in drop and f['w'] == e['w'] and f['kind'] != e['kind']:
                drop |= {i, j}
                break
    return [e for i, e in enumerate(ev) if i not in drop]


def cancel_bursts(ev, T, near, dup=0):
    """21.10 owner: one person lingering in the doorway keeps getting new track numbers, and each piece flips a side.
    An in and an out (any tracks) within T s and within `near` body heights of each other never crossed: both go.
    dup > 0: a second event of the same kind within dup s at the same place is the same crossing counted twice."""
    ev = sorted(ev, key=lambda e: e['t'])
    def close(a, b):
        if 'x' not in a or 'x' not in b:
            return a['w'] == b['w']
        return ((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2) ** .5 <= near * max(a['h'], b['h'])
    drop = set()
    for i, e in enumerate(ev):
        if i in drop:
            continue
        for j in range(i + 1, len(ev)):
            f = ev[j]
            if f['t'] - e['t'] > max(T, dup):
                break
            if j in drop or not close(e, f):
                continue
            if f['kind'] != e['kind'] and f['t'] - e['t'] <= T:
                drop |= {i, j}
                break
            if dup and f['kind'] == e['kind'] and f['t'] - e['t'] <= dup:
                drop.add(j)
    return [e for i, e in enumerate(ev) if i not in drop]


def alternate(ev, window=None):
    out, lastk = [], {}
    for e in sorted(ev, key=lambda e: e['t']):
        prev = lastk.get(e['w'])
        if prev and prev[0] == e['kind'] and (window is None or e['t'] - prev[1] <= window):
            continue
        lastk[e['w']] = (e['kind'], e['t'])
        out.append(e)
    return out


WIDTH = None


def match(pred, truth, tol=6.0):
    res = {}
    for kind in ('in', 'out'):
        P = sorted([e['t'] for e in pred if e['kind'] == kind])
        T = sorted([e['t'] for e in truth if e['kind'] == kind])
        used, hit = set(), 0
        pairs = sorted((abs(p - t), i, j) for i, p in enumerate(P) for j, t in enumerate(T) if abs(p - t) <= tol)
        ui = set()
        for d, i, j in pairs:
            if i in ui or j in used:
                continue
            ui.add(i); used.add(j); hit += 1
        res[kind] = (hit, len(P), len(T))
    return res


def score(data, c, roles=('train',), knn_fn=None):
    tot = {'in': [0, 0, 0], 'out': [0, 0, 0]}
    obs = [0, 0]
    for tag, d in data['tags'].items():
        if d['role'] not in roles:
            continue
        ev, so = run_tag(d, c, knn_fn(tag) if knn_fn else None, tag)
        m = match(ev, d['events'])
        for k in tot:
            for i in range(3):
                tot[k][i] += m[k][i]
        for r in d['rows']:
            s = so.get((r[PIECE], r[K]))
            if r[TRU] in ('in', 'out') and s is not None:
                obs[0] += int(s == (r[TRU] == 'in')); obs[1] += 1
    f = []
    for k in tot:
        h, p, t = tot[k]
        f.append(2 * h / max(1, p + t))
    return dict(f1=round(sum(f) / 2, 4), inn=tot['in'], out=tot['out'], obs_acc=round(obs[0] / max(1, obs[1]), 4))


def load(path):
    import cv2
    data = json.load(gzip.open(path, 'rt'))
    polys = {n: np.array(z['pts'], np.float32) for n, z in zip(['band', 'shop', 'door', 'gallery'], data['zones'])}
    for d in data['tags'].values():
        for r in d['rows']:
            r.append(next((n for n in ('door', 'shop', 'band', 'gallery') if cv2.pointPolygonTest(polys[n], (float(r[FX]), float(r[FY])), False) >= 0), 'none'))
    return data


if __name__ == '__main__':
    data = load(sys.argv[1])
    print('base train', score(data, BASE))
