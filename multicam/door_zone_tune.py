"""Count by the whole track, the way production people counters do (08.10.2026): fragments of one person are stitched
into one track, then the track's START side (its first trusted votes) and END side (its last) decide -- outside ->
inside = an entry, inside -> outside = an exit, the same side = nothing, however it flickered in the doorway. The track
must pass by the door (its feet within `near` px of the drawn segment at the crossing). Compared with the flip logic
(side_final.json) on the same tracks: the owner's /doorside stretches, leave-one-day-out selection.

usage: RA_BIN_SRC=micro1s6 door_zone_tune.py   (needs the ldpc2/mpc runs and pc positions of door_side_tune)"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('RA_BIN_SRC', 'micro1s6')
os.environ.setdefault('RA_BIN_VER', 'ldpc2')
os.environ.setdefault('RA_POS_VER', 'pc')
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'door_v2'


def zone_events(tracks, ns, near_px, line, birth=None):
    """tracks: {name: [(t, vote, xy)]} -> events by start/end side. birth: (u, gain, horizon, hz, rad) -- a track that
    is inside from its start, starts at the door and walks in, counts as an entry too."""
    import door_line
    ev = []
    for name, xs in tracks.items():
        tv = [(t, v, xy) for t, v, xy in xs if v in (0.0, 1.0)]
        if len(tv) < 2:
            continue
        head = [v for _, v, _ in tv[:ns]]
        tail = [v for _, v, _ in tv[-ns:]]
        s0 = 1.0 if sum(head) * 2 > len(head) else (0.0 if sum(head) * 2 < len(head) else head[0])
        s1 = 1.0 if sum(tail) * 2 > len(tail) else (0.0 if sum(tail) * 2 < len(tail) else tail[-1])
        if s0 != s1:
            # the crossing: the first vote of the final side after the last vote of the first side
            last0 = max(i for i, (_, v, _) in enumerate(tv) if v == s0)
            nxt = next((i for i in range(last0 + 1, len(tv)) if tv[i][1] == s1), None)
            if nxt is None:
                continue
            t, _, xy = tv[nxt]
            xy = xy or next((x for _, _, x in tv[nxt:] if x), None) or next((x for _, _, x in reversed(tv[:nxt]) if x), None)
            if near_px and (xy is None or door_line.seg_dist(line, xy[0] * 1280, xy[1] * 720) > near_px):
                continue
            ev.append({'kind': 'in' if s1 == 1.0 else 'out', 't': t, 'w': name, 'xy': xy})
        elif birth is not None and s0 == 1.0:
            u, gain, horizon, hz, rad = birth
            first = next((xy for _, _, xy in xs if xy is not None), None)
            if first is None or not len(hz) or float(np.min(np.hypot(*(hz - np.asarray(first)).T))) > rad:
                continue
            if near_px and door_line.seg_dist(line, first[0] * 1280, first[1] * 720) > near_px:
                continue
            pr = [float(np.asarray(xy) @ u) for t, _, xy in xs if xy is not None and t - xs[0][0] <= horizon]
            if len(pr) >= 2 and max(pr) - pr[0] >= gain and xs[-1][0] - xs[0][0] >= 2.0:
                ev.append({'kind': 'in', 't': xs[0][0], 'w': name, 'xy': first, 'birth': True})
    return ev


def main():
    import door_combo as C
    import door_learn as L
    import door_line
    import door_side_tune as ST
    import door_v2 as D
    t0 = time.time()
    days, line, table = ST.load()
    f_ = json.load(open(OUT / 'combo_final.json'))
    u, hz = np.array(f_['u']), np.array(f_['zone'])
    sd = json.load(open(OUT / 'side_final.json'))
    stride = int(os.environ.get('RA_ZONE_STRIDE', '6'))
    print('days', list(days), '%.0f s' % (time.time() - t0), flush=True)

    def tracks_of(day, stitch, vote_cfg):
        v = days[day]
        out = {}
        for tag, xs in v['obs'].items():
            per = {}
            for o in xs:
                if o['k'] % stride:
                    continue
                xy = v['dd'].P.get('%.2f_%d' % (o['t'], o['pid']))
                vt = door_line.side_vote(vote_cfg, table, line, o['d'], o['top'], o['h'], o['bx'], o['p'])
                per.setdefault(o['pid'], []).append((o['t'], vt, xy))
            per = {k: sorted(x) for k, x in per.items()}
            if stitch:
                per = C.stitch_tracks(per, stitch)
            for r, lst in per.items():
                out['%s:%s' % (tag, r)] = lst
        return out

    def score(day, ev):
        v = days[day]
        c = [0] * 6
        for role in ('train', 'test'):
            sp = [(x, y) for x, y, r in v['spans'] if r == role]
            if not sp:
                continue
            inside = lambda t: any(x - 1 <= t <= y + 1 for x, y in sp)
            m = D.match([dict(e, t=e['t'] + L.SHIFT) for e in ev if inside(e['t'] + L.SHIFT)],
                        [t for t in v['truth'] if t['role'] == role], L.TOL)
            c = [a + b for a, b in zip(c, [m['in']['hit'], m['in']['pred'], m['in']['true'], m['out']['hit'], m['out']['pred'], m['out']['true']])]
        return c

    f1 = lambda c: ((2 * c[0] / max(1, c[1] + c[2])) + (2 * c[3] / max(1, c[4] + c[5]))) / 2
    res = []
    stitches = [(1.5, 0.04), (3.0, 0.06), (5.0, 0.08), (8.0, 0.12), (12.0, 0.15)]
    for mode, band in (('either_in+both_out', 60), ('either_in+both_out', None), ('model', None), ('line_in+both_out', 60)):
        cfg = dict(sd, mode=mode, band=band)
        for stitch in stitches:
            T = {d: tracks_of(d, stitch, cfg) for d in days}
            # the flip logic on the same tracks (today's live)
            for conf, near in itertools.product((1, 2), (None, 150)):
                per = {}
                for d in days:
                    ev = C.gate(C.flip_events(T[d], conf, 0.95), u, hz, f_['move'], f_['rad'])
                    ev = C.near_line(C.alternate(ev, 'first'), line, near)
                    per[d] = score(d, ev)
                res.append({'logic': 'flip', 'mode': mode, 'band': band, 'stitch': stitch, 'conf': conf, 'near': near, 'per': per})
            # the whole-track logic
            for ns, near, birth in itertools.product((1, 2, 3, 5), (None, 150, 250), (False, True)):
                per = {}
                for d in days:
                    ev = zone_events(T[d], ns, near, line, (u, 0.05, 4.0, hz, f_['rad']) if birth else None)
                    per[d] = score(d, ev)
                res.append({'logic': 'zone', 'mode': mode, 'band': band, 'stitch': stitch, 'ns': ns, 'near': near, 'birth': birth, 'per': per})
        print(mode, band, 'done %.0f s' % (time.time() - t0), flush=True)
    ds = sorted(days)
    tot = lambda r, dd: [sum(r['per'][d][i] for d in dd) for i in range(6)]
    key = lambda r: {k: r[k] for k in r if k != 'per'}
    for logic in ('flip', 'zone'):
        rs = [r for r in res if r['logic'] == logic]
        best = max(rs, key=lambda r: f1(tot(r, ds)))
        T_ = [0] * 6
        picks = []
        for d in ds:
            o = [x for x in ds if x != d]
            b = max(rs, key=lambda r: round(f1(tot(r, o)), 4))
            picks.append(key(b))
            T_ = [a + x for a, x in zip(T_, tot(b, [d]))]
        print('== %s: best on all days %.3f %s %s' % (logic, f1(tot(best, ds)), tot(best, ds), json.dumps(key(best))), flush=True)
        print('   LODO %.3f %s' % (f1(T_), T_), flush=True)
        for p in picks:
            print('   pick', json.dumps(p), flush=True)
        for r in sorted(rs, key=lambda r: -f1(tot(r, ds)))[:8]:
            print('   %.3f %s %s' % (f1(tot(r, ds)), tot(r, ds), json.dumps(key(r))), flush=True)
    json.dump([dict(key(r), per=r['per']) for r in res], open(OUT / ('zone_tune_%s.json' % os.environ['RA_BIN_SRC']), 'w'))
    print('done %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
