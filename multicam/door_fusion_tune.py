"""One score for the side instead of two yes/no votes (08.10.2026): s = logit(p_model) + w * depth / 50 (the line's
depth past the drawn line, clipped), smoothed over the last `win` observations of the track; inside when s > T, outside
when s < -T, unsure between. Then the usual flips + gate + births + near-the-door + alternation, on the owner's
/doorside stretches, leave-one-day-out. usage: RA_BIN_SRC=micro1s6 door_fusion_tune.py"""
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
    stride = int(os.environ.get('RA_ZONE_STRIDE', '6'))
    base = {}
    for day, v in days.items():
        tr = {}
        for tag, xs in v['obs'].items():
            per = {}
            for o in xs:
                if o['k'] % stride:
                    continue
                xy = v['dd'].P.get('%.2f_%d' % (o['t'], o['pid']))
                per.setdefault(o['pid'], []).append((o['t'], o, xy))
            per = {k: sorted(x, key=lambda a: a[0]) for k, x in per.items()}
            st = C.stitch_tracks({k: [(a[0], a[1], a[2]) for a in x] for k, x in per.items()}, tuple(f_['stitch']))
            for r, lst in st.items():
                tr['%s:%s' % (tag, r)] = lst
        base[day] = tr

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
    lg = lambda p: float(np.log(max(1e-4, min(1 - 1e-4, p)) / (1 - max(1e-4, min(1 - 1e-4, p)))))
    res = []
    for w, T, win, conf, near, birth in itertools.product((0.0, 0.5, 1.0, 2.0, 4.0, 99.0), (1.0, 2.0, 3.0, 4.0), (1, 3, 5), (1, 2),
                                                          (None, 150), (False, True)):
        per = {}
        for day in days:
            tracks = {}
            for name, lst in base[day].items():
                s = []
                for t, o, xy in lst:
                    p = o['p'] if o['p'] is not None else 0.5
                    d = 0.0 if o['d'] is None else max(-200.0, min(200.0, o['d']))
                    s.append(lg(p) * (0.0 if w == 99.0 else 1.0) + (w if w != 99.0 else 1.0) * d / 50.0)
                sm = np.convolve(s, np.ones(win) / win, mode='full')[:len(s)] if win > 1 else np.array(s)
                if win > 1:                                      # the first few: mean of what there is
                    for i in range(min(win - 1, len(s))):
                        sm[i] = float(np.mean(s[:i + 1]))
                tracks[name] = [(t, 1.0 if x > T else (0.0 if x < -T else 0.5), xy) for (t, o, xy), x in zip(lst, sm)]
            ev = C.gate(C.flip_events(tracks, conf, 0.95), u, hz, f_['move'], f_['rad'])
            if birth:
                ev = ev + C.birth_death(tracks, conf, 0.95, hz, f_['rad'], True, False, u=u, move=f_['move'], min_len=2.0,
                                        starts=[st['a'] for st in days[day]['dd'].S.values()], gain=0.05)
            ev = C.alternate(C.near_line(ev, line, near), 'first')
            per[day] = score(day, ev)
        res.append({'w': w, 'T': T, 'win': win, 'conf': conf, 'near': near, 'birth': birth, 'per': per})
    ds = sorted(days)
    tot = lambda r, dd: [sum(r['per'][d][i] for d in dd) for i in range(6)]
    key = lambda r: {k: r[k] for k in r if k != 'per'}
    best = max(res, key=lambda r: f1(tot(r, ds)))
    T_ = [0] * 6
    for d in ds:
        o = [x for x in ds if x != d]
        b = max(res, key=lambda r: (round(f1(tot(r, o)), 4), -r['win'], r['near'] is not None))
        print('pick for', d, json.dumps(key(b)), tot(b, [d]), flush=True)
        T_ = [a + x for a, x in zip(T_, tot(b, [d]))]
    print('best on all days %.3f %s %s' % (f1(tot(best, ds)), tot(best, ds), json.dumps(key(best))), flush=True)
    print('LODO %.3f %s' % (f1(T_), T_), flush=True)
    for r in sorted(res, key=lambda r: -f1(tot(r, ds)))[:10]:
        print('  %.3f %s %s' % (f1(tot(r, ds)), tot(r, ds), json.dumps(key(r))), flush=True)
    print('done %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
