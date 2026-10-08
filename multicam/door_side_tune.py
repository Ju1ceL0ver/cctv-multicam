"""Which side is a person on -- the owner's model, the shop line, or both -- and when does it count as a crossing
(08.10.2026). Per observation of every SAM track (pieces, no ReID) at the /doorside stretches: the model's p_inside
(binary_micro1s3_mpc_<day>.json, RA_DOOR_LINE=0) and the line's depth / mask top / height / bottom x
(binary_micro1s3_ldpc2_<day>.json). A vote per observation by one of MODES, a side change after `conf` votes in a
row, then the door-zone/movement gate and alternation (a person's counted events alternate).

Selection without peeking: for every day, the setting best on the other two days (micro F1 over their crossings) is
scored on it; next to it, the owner's split (17.09 train -> 18/19.09 test). -> data/door_v2/side_tune.json

usage: door_side_tune.py [fast]  (CPU)"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('RA_BIN_SRC', 'micro1s3')
os.environ.setdefault('RA_BIN_VER', 'ldpc2')
os.environ.setdefault('RA_POS_VER', 'pc')
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'door_v2'

MODES = ('line', 'model', 'model+line_in', 'agree', 'line_in+model_out', 'line_in+both_out', 'either_in+both_out')


def mvote(p, thr):
    return 1.0 if p >= thr else (0.0 if p <= 1 - thr else 0.5)


def combine_vote(mode, lv, mv, line_in):
    if mode == 'line':
        return lv
    if mode == 'model':
        return mv
    if mode == 'model+line_in':                 # the model, a piece past the line overrides to inside (today's live)
        return 1.0 if line_in else mv
    if mode == 'agree':
        return lv if lv == mv else 0.5
    if mode == 'line_in+model_out':
        return 1.0 if lv == 1 else (0.0 if mv == 0 else 0.5)
    if mode == 'line_in+both_out':
        return 1.0 if lv == 1 else (0.0 if mv == 0 and lv == 0 else 0.5)
    if mode == 'either_in+both_out':
        return 1.0 if (lv == 1 or mv == 1) else (0.0 if lv == 0 and mv == 0 else 0.5)
    raise ValueError(mode)


def load():
    import door_combo as C
    import door_line
    import door_line_tune as T
    line = door_line.load()
    days = {}
    for day in C.DAYS:
        f_ld = OUT / ('binary_micro1s3_%s%s.json' % (os.environ['RA_BIN_VER'], day))
        f_mo = OUT / ('binary_micro1s3_%s%s.json' % (os.environ.get('RA_TUNE_MODEL', 'mpc'), day))
        if not (f_ld.exists() and f_mo.exists()):
            print('skip', day, f_ld.exists(), f_mo.exists(), flush=True)
            continue
        dd = C.Day(day)                                   # S = the depth run, P = positions (pieces)
        mo = json.load(open(f_mo))
        pm = {(tag, round(t, 2), pid): p for tag, st in mo.items() for t, pid, p in st['obs']}
        obs = {}
        for tag, st in dd.S.items():
            a = st['a']
            for o in st['obs']:
                t, pid = round(o[0], 2), o[1]
                k = int(round((t - a) / 0.08))
                obs.setdefault(tag, []).append({'t': t, 'pid': pid, 'k': k, 'd': o[2], 'top': o[3], 'h': o[4],
                                                'bx': o[5] if len(o) > 5 else None, 'p': pm.get((tag, t, pid))})
        spans, truth = T.stretches(day)
        days[day] = {'dd': dd, 'obs': obs, 'spans': spans, 'truth': truth}
    table = door_line.full_height_table([(o['top'], o['h']) for v in days.values() for xs in v['obs'].values() for o in xs])
    return days, line, table


def main(fast=False):
    import door_combo as C
    import door_learn as L
    import door_line
    import door_v2 as D
    t0 = time.time()
    days, line, table = load()
    f_ = json.load(open(OUT / 'combo_final.json'))
    u, hz = np.array(f_['u']), np.array(f_['zone'])
    stitch = tuple(f_['stitch'])
    print('days', list(days), 'loaded %.0f s' % (time.time() - t0), flush=True)
    # stitched tracks once per (day, stride): {name: [obs index...]} with positions
    tracks = {}
    for day, v in days.items():
        for stride in (3, 6):
            tr = {}
            for tag, xs in v['obs'].items():
                per = {}
                for i, o in enumerate(xs):
                    if o['k'] % stride:
                        continue
                    xy = v['dd'].P.get('%.2f_%d' % (o['t'], o['pid']))
                    per.setdefault(o['pid'], []).append((o['t'], i, xy))
                for pid in per:
                    per[pid].sort()
                roots = {k: k for k in per}
                st = C.stitch_tracks(per, stitch, roots)
                for r, lst in st.items():
                    tr['%s:%s' % (tag, r)] = [(t, (tag, i), xy) for t, i, xy in lst]
            tracks[(day, stride)] = tr

    fcache = {}

    def counts(day, stride, vote_of, conf, gate, alt, vkey=None):
        v = days[day]
        fk = (vkey, stride, conf)
        if vkey is None or fk not in fcache:
            tr = {name: [(t, vote_of((tag, i)), xy) for t, (tag, i), xy in lst] for name, lst in tracks[(day, stride)].items()}
            if len(fcache) > 256:
                fcache.clear()
            fcache[fk] = C.flip_events(tr, conf, 0.95, L.SHIFT)
        ev = fcache[fk]
        if gate:
            ev = C.gate(ev, u, hz, f_['move'], f_['rad'])
        if alt != 'none':
            ev = C.alternate(ev, alt)
        out = {}
        for role in ('train', 'test'):
            sp = [(x, y) for x, y, r in v['spans'] if r == role]
            if not sp:
                continue
            inside = lambda t: any(x - 1 <= t <= y + 1 for x, y in sp)
            m = D.match([e for e in ev if inside(e['t'])], [t for t in v['truth'] if t['role'] == role], L.TOL)
            out[role] = [m['in']['hit'], m['in']['pred'], m['in']['true'], m['out']['hit'], m['out']['pred'], m['out']['true']]
        return out

    def f1(c):
        return ((2 * c[0] / max(1, c[1] + c[2])) + (2 * c[3] / max(1, c[4] + c[5]))) / 2

    legs_opts = ((0.0, 'none'), (0.7, 'unsure'), (0.7, 'virtual'), (0.8, 'virtual'), (0.6, 'virtual'))
    hs = ((0, 0), (10, 15)) if fast else ((0, 0), (5, 10), (10, 15), (20, 30))
    grid = []
    for mode in MODES:
        for thr in ((None,) if mode == 'line' else (0.8, 0.9, 0.95)):
            for (h_in, h_out), (trunc, legs) in itertools.product(hs, legs_opts):
                if mode == 'model' and (h_in, h_out, trunc) != hs[0] + (0.0,):
                    continue                              # the model alone has no line settings
                for conf, gate, alt, stride in itertools.product((1, 2, 3), (True, False), ('none', 'first', 'last'), (3, 6)):
                    grid.append(dict(mode=mode, thr=thr, h_in=h_in, h_out=h_out, trunc=trunc, legs=legs, conf=conf,
                                     gate=gate, alt=alt, stride=stride))
    if os.environ.get('RA_TUNE_SMOKE'):
        grid = grid[::max(1, len(grid) // int(os.environ['RA_TUNE_SMOKE']))]
    print('settings', len(grid), flush=True)
    # per day: flat arrays of every observation, the index of (tag, i) into them
    arr, at = {}, {}
    full = np.array(table['full'], float)
    for d, v in days.items():
        flat = [(tag, i, o) for tag, xs in v['obs'].items() for i, o in enumerate(xs)]
        at[d] = {(tag, i): j for j, (tag, i, o) in enumerate(flat)}
        g_ = lambda key, dflt=np.nan: np.array([o[key] if o[key] is not None else dflt for _, _, o in flat], float)
        D_, top, hh, bx, pp = g_('d'), g_('top'), g_('h'), g_('bx'), g_('p')
        fh = full[np.clip((np.nan_to_num(top) // table['bin_px']).astype(int), 0, len(full) - 1)]
        (x1, y1), (x2, y2) = line['p1'], line['p2']
        nrm = float(np.hypot(x2 - x1, y2 - y1)) / door_line.side_sign(line, *line['inside'])
        dv = ((x2 - x1) * (top + fh - y1) - (y2 - y1) * (bx - x1)) / nrm     # depth of the virtual feet
        arr[d] = dict(d=D_, top=top, h=hh, fh=fh, dv=dv, p=pp)

    def votes(day, g):
        A = arr[day]
        d, h_in, h_out = A['d'], g['h_in'], g['h_out']
        lv = np.full(len(d), 0.5)
        trunc = (A['h'] < g['trunc'] * A['fh']) if g['trunc'] > 0 else np.zeros(len(d), bool)
        lv[(~trunc) & (d <= -h_out)] = 0.0
        if g['legs'] == 'virtual':
            vv = trunc & (d < h_in)
            lv[vv & (A['dv'] >= h_in)] = 1.0
            lv[vv & (A['dv'] <= -h_out)] = 0.0
        lv[d >= h_in] = 1.0
        lv[np.isnan(d)] = 0.5
        if g['thr']:
            mv = np.where(A['p'] >= g['thr'], 1.0, np.where(A['p'] <= 1 - g['thr'], 0.0, 0.5))
            mv[np.isnan(A['p'])] = 0.5
        else:
            mv = np.full(len(d), 0.5)
        m = g['mode']
        if m == 'line':
            return lv
        if m == 'model':
            return mv
        if m == 'model+line_in':
            return np.where(d >= h_in, 1.0, mv)
        if m == 'agree':
            return np.where(lv == mv, lv, 0.5)
        if m == 'line_in+model_out':
            return np.where(lv == 1, 1.0, np.where(mv == 0, 0.0, 0.5))
        if m == 'line_in+both_out':
            return np.where(lv == 1, 1.0, np.where((mv == 0) & (lv == 0), 0.0, 0.5))
        if m == 'either_in+both_out':
            return np.where((lv == 1) | (mv == 1), 1.0, np.where((lv == 0) & (mv == 0), 0.0, 0.5))
        raise ValueError(m)

    res = []
    vcache = {}
    for n, g in enumerate(grid):
        per_day = {}
        for d in days:
            key = (d, g['mode'], g['thr'], g['h_in'], g['h_out'], g['trunc'], g['legs'])
            if key not in vcache:
                vcache.clear() if len(vcache) > 64 else None
                vcache[key] = votes(d, g)
            V = vcache[key]
            per_day[d] = counts(d, g['stride'], lambda o, V=V, d=d: V[at[d][o]], g['conf'], g['gate'], g['alt'], key)
        res.append(dict(g, per_day=per_day))
        if n % 1000 == 0:
            print(n, 'of', len(grid), '%.0f s' % (time.time() - t0), flush=True)

    def total(r, ds, role=None):
        c = [0] * 6
        for d in ds:
            for rl, v in r['per_day'][d].items():
                if role is None or rl == role:
                    c = [a + b for a, b in zip(c, v)]
        return c

    simple = lambda r: (r['h_in'] + r['h_out']) / 100 + r['trunc'] + (r['alt'] != 'none') * .05 + (not r['gate']) * .05 + \
        (r['mode'] != 'line') * .05 + (r['stride'] == 3) * .02
    rep = {'lodo': {}, 'split': None}
    ds = sorted(days)
    tot_lodo = [0] * 6
    for d in ds:
        others = [x for x in ds if x != d]
        best = max(res, key=lambda r: (round(f1(total(r, others)), 4), -simple(r)))
        c = total(best, [d])
        tot_lodo = [a + b for a, b in zip(tot_lodo, c)]
        rep['lodo'][d] = {'setting': {k: best[k] for k in grid[0]}, 'counts': c, 'f1': round(f1(c), 3)}
        print('LODO', d, json.dumps(rep['lodo'][d]), flush=True)
    rep['lodo_total'] = {'counts': tot_lodo, 'f1': round(f1(tot_lodo), 3)}
    print('LODO total', json.dumps(rep['lodo_total']), flush=True)
    best = max(res, key=lambda r: (round(f1(total(r, ds, 'train')), 4), -simple(r)))
    rep['split'] = {'setting': {k: best[k] for k in grid[0]}, 'train': total(best, ds, 'train'), 'test': total(best, ds, 'test'),
                    'f1_test': round(f1(total(best, ds, 'test')), 3)}
    print('17.09 -> 18/19:', json.dumps(rep['split']), flush=True)
    # reference rows: today's settings
    for name, want in (('model alone (old live)', dict(mode='model', thr=0.95, conf=2, gate=True, alt='none', stride=3)),
                       ('model+line now', dict(mode='model+line_in', thr=0.95, h_in=0, h_out=0, trunc=0.0, conf=2, gate=True, alt='none', stride=3)),
                       ('line only now', dict(mode='line', h_in=0, h_out=0, trunc=0.0, conf=2, gate=True, alt='none', stride=3))):
        r = next((x for x in res if all(x[k] == v for k, v in want.items())), None)
        if r is None:
            continue
        c = total(r, ds)
        rep[name] = {'counts': c, 'f1': round(f1(c), 3)}
        print('%-24s all days %s F1 %.3f' % (name, c, f1(c)), flush=True)
    # the best of each mode on all three days (optimistic: chosen on what it is scored on)
    rep['best_by_mode_all_days'] = {}
    for mode in MODES:
        for stride in (3, 6):
            cand = [x for x in res if x['mode'] == mode and x['stride'] == stride]
            if not cand:
                continue
            r = max(cand, key=lambda x: (f1(total(x, ds)), -simple(x)))
            rep['best_by_mode_all_days']['%s/%d' % (mode, stride)] = {'setting': {k: r[k] for k in grid[0]}, 'counts': total(r, ds),
                                                                      'f1': round(f1(total(r, ds)), 3)}
            print('best %-20s stride %d F1 %.3f %s' % (mode, stride, f1(total(r, ds)), json.dumps({k: r[k] for k in grid[0]})), flush=True)
    # the final setting: the most common LODO choice re-picked on all three days among the same mode/stride
    fin = max(res, key=lambda r: (round(f1(total(r, ds)), 4), -simple(r)))
    rep['final_all_days'] = {'setting': {k: fin[k] for k in grid[0]}, 'counts': total(fin, ds), 'f1': round(f1(total(fin, ds)), 3)}
    rep['table'] = table
    json.dump(rep, open(OUT / 'side_tune.json', 'w'), indent=1)
    json.dump([dict({k: r[k] for k in grid[0]}, per_day=r['per_day']) for r in res], open(OUT / 'side_tune_all.json', 'w'))
    print('done %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main(len(sys.argv) > 1 and sys.argv[1] == 'fast')
