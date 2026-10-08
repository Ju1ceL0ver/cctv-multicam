"""Entries and exits from the owner's two-class inside/outside model (inout_lab/binary_door.py), 07.10.2026.

predict DAY: SAM 3.1's people of every door stretch of the day (data/sam31_door, the same people as the door rule's
  sam31 tracks), every STRIDE-th tick, the model's p_inside per person -> data/door_v2/binary_<day>.json (CPU).
eval: a track's side changes only after `conf` trusted votes in a row (p >= thr or <= 1-thr; the side_state recipe,
  no door zone); outside->inside = entry, back = exit, the first side of a track is no event. conf/thr chosen on two
  days, scored on the third (as door_learn); next to it the learned door rule on the same days (door_learn.evaluate's
  numbers in compare/fuse outputs are on the same truth). -> data/door_v2/binary_eval.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inout_lab'))
OUT = ROOT / 'data' / 'door_v2'
STRIDE = 3
import os
SRC = os.environ.get('RA_BIN_SRC', 'sam31')        # 07.10: or a small-SAM run under data/micro_door/<name>
SUF = ('' if SRC == 'sam31' else SRC + '_') + os.environ.get('RA_BIN_VER', '')   # 07.10: v2 = the refitted model
POS_SUF = '' if SRC == 'sam31' else SRC + '_'


def src_base(tag):
    """(the people's chunks dir, the stride of its ticks against the stretch video's frames)"""
    if SRC == 'sam31':
        return ROOT / 'data' / 'sam31_door' / tag / 'cam1', 1
    b = ROOT / 'data' / 'micro_door' / SRC / tag / 'cam1'
    return b, int(json.load(open(b / 'info.json')).get('stride', 1))
DAYS = ('20260917', '20260918', '20260919')


def predict(day):
    import cv2
    import door_compare_video as V
    import door_v2 as D
    from binary_door import BinaryDoorClassifier
    clf = BinaryDoorClassifier(os.environ.get('RA_BIN_MODEL') or None)
    import door_line
    line = door_line.load() if os.environ.get('RA_DOOR_LINE', '1') == '1' else None
    print('shop line', line, flush=True)
    res = {}
    tags = sorted(p.parent.name for p in (ROOT / 'data' / 'sam31_door').glob('door_%s_*/cam1' % day))
    if os.environ.get('RA_BIN_TAGS'):
        tags = [t for t in tags if t in os.environ['RA_BIN_TAGS'].split(',')]
    for i, tag in enumerate(tags):
        base, sstride = src_base(tag)
        if not (base / 'report.json').exists():
            continue
        M, by_tick, info = V.people(base)
        if sstride > 1:                                   # the small SAM's ticks are every sstride-th frame
            by_tick = {kk * sstride: v for kk, v in by_tick.items()}
        a = float(tag.split('_')[2])
        a = next((x for x, y in D.stretches(day) if int(round(x)) == int(a)), a)
        cap = cv2.VideoCapture(str(ROOT / 'data' / 'sam31_door' / tag / 'cam1' / 'video.mp4'))
        H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        obs = []
        k = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if k % STRIDE == 0 and by_tick.get(k):
                rgb = f[:, :, ::-1].copy()
                for r, pid in by_tick[k]:
                    x1, y1 = int(M.rows[r, 4]), int(M.rows[r, 5])
                    m = np.zeros((H, W), np.uint8)
                    c = M.crop(r)
                    m[y1:y1 + c.shape[0], x1:x1 + c.shape[1]] = c[:H - y1, :W - x1]
                    try:
                        p = clf.predict_rgb(rgb, m)['p_inside']
                    except ValueError:
                        continue
                    if line is not None and door_line.inside(line, m > 0):
                        p = 1.0                           # the owner's shop line: a piece of mask past it = inside
                    obs.append([round(a + k * D.TICK, 2), int(pid), round(p, 4)])
            k += 1
        cap.release()
        res[tag] = {'a': a, 'end': a + k * D.TICK, 'obs': obs}
        print(day, i + 1, len(tags), tag, len(obs), flush=True)
    json.dump(res, open(OUT / ('binary_%s%s.json' % (SUF, day)), 'w'))


def events(stretch, conf, thr):
    tracks = {}
    for t, pid, p in stretch['obs']:
        tracks.setdefault(pid, []).append((t, p))
    ev = []
    for pid, xs in tracks.items():
        xs.sort()
        side, pend, n = None, None, 0
        for t, p in xs:
            v = 1 if p >= thr else (0 if p <= 1 - thr else None)
            if v is None:
                continue
            if v == side:
                pend, n = None, 0
                continue
            if pend != v:
                pend, n, t0 = v, 0, t
            n += 1
            if n >= conf:
                if side is not None:
                    ev.append({'kind': 'in' if v == 1 else 'out', 't': t0, 'w': pid})
                side, pend, n = v, None, 0
    return ev


def day_events(day, conf, thr):
    S = json.load(open(OUT / ('binary_%s%s.json' % (SUF, day))))
    ev, spans = [], []
    for st in S.values():
        ev += events(st, conf, thr)
        spans.append((st['a'], st['end']))
    return ev, spans


def score(day, conf, thr):
    import door_learn as L
    import door_v2 as D
    ev, spans = day_events(day, conf, thr)
    inside = lambda t: any(a <= t <= b for a, b in spans)
    truth = [t for t in D.truth(day, True) if inside(t['t'])]
    skip = L.skipped(day)
    if skip:
        out_of = lambda t: any(a - 10 <= t <= b + 10 for a, b in skip)
        truth = [t for t in truth if not out_of(t['t'])]
        ev = [e for e in ev if not out_of(e['t'] + L.SHIFT)]
    m = D.match([dict(e, t=e['t'] + L.SHIFT) for e in ev], truth, L.TOL)
    return m, (m['in']['f1'] + m['out']['f1']) / 2


def evaluate():
    grid = [(c, th) for c in (1, 2, 3, 4, 6, 8) for th in (0.6, 0.7, 0.8, 0.9, 0.95)]
    have = [d for d in DAYS if (OUT / ('binary_%s.json' % d)).exists()]
    f1 = {(d, g): score(d, *g)[1] for d in have for g in grid}
    rep = {'days': {}}
    for test in have:
        train = [d for d in have if d != test]
        best = max(grid, key=lambda g: np.mean([f1[(d, g)] for d in train]))
        m, f = score(test, *best)
        rep['days'][test] = {'conf': best[0], 'thr': best[1], 'f1': round(f, 3),
                             'in': {k: m['in'][k] for k in ('true', 'pred', 'hit', 'precision', 'recall', 'f1')},
                             'out': {k: m['out'][k] for k in ('true', 'pred', 'hit', 'precision', 'recall', 'f1')}}
        print(test, json.dumps(rep['days'][test]), flush=True)
    rep['grid_mean_f1'] = {'%d_%.2f' % g: round(float(np.mean([f1[(d, g)] for d in have])), 3) for g in grid}
    json.dump(rep, open(OUT / 'binary_eval.json', 'w'), indent=1)


if __name__ == '__main__' and sys.argv[1] in ('predict', 'eval'):
    if sys.argv[1] == 'predict':
        predict(sys.argv[2])
    else:
        evaluate()


# ------------------------------------------------------------------ 07.10: the door zone gate

def positions(day):
    """Where each observed person stands: the bottom of the mask (door_sam.foot_of), 0..1 of the frame, per (t, pid)."""
    import door_compare_video as V
    import door_sam as DS
    import door_v2 as D
    S = json.load(open(OUT / ('binary_%s%s.json' % (SUF, day))))
    pos = {}
    for tag, st in S.items():
        base, sstride = src_base(tag)
        M, by_tick, info = V.people(base)
        if sstride > 1:
            by_tick = {kk * sstride: v for kk, v in by_tick.items()}
        W, H = 2176.0, 1224.0
        for k, items in by_tick.items():
            if k % STRIDE:
                continue
            for r, pid in items:
                f = DS.foot_of(M.crop(r))
                x1, y1, x2, y2 = M.rows[r, 4:8]
                fx, fy = (x1 + f[0], y1 + f[1]) if f else ((x1 + x2) / 2, y2)
                pos['%.2f_%d' % (round(st['a'] + k * D.TICK, 2), pid)] = [round(fx / W, 4), round(fy / H, 4)]
        print(day, tag, flush=True)
    json.dump(pos, open(OUT / ('binary_pos_%s%s.json' % (POS_SUF, day)), 'w'))


def gated_eval():
    """Side changes count only where real crossings happen: the place of each binary event (where the person stands at
    the first vote of the new side) must lie within `r` of a place of a right event of the training days."""
    import door_learn as L
    import door_v2 as D
    P = {d: json.load(open(OUT / ('binary_pos_%s%s.json' % (POS_SUF, d)))) for d in DAYS}

    def place(d, e):
        return P[d].get('%.2f_%d' % (round(e['t'], 2), e['w']))

    def scored(d, conf, thr):
        ev, spans = day_events(d, conf, thr)
        inside = lambda t: any(a <= t <= b for a, b in spans)
        truth = [t for t in D.truth(d, True) if inside(t['t'])]
        E = [dict(e, t=e['t'] + L.SHIFT, xy=place(d, e)) for e in ev]
        y = L.label([(None, e['kind'], e['t'], e['w']) for e in E], truth)
        return E, y, truth

    grid = [(c, th) for c in (2, 3, 4, 6) for th in (0.8, 0.9, 0.95)]
    cache = {(d, g): scored(d, *g) for d in DAYS for g in grid}
    rep = {}
    for test in DAYS:
        train = [d for d in DAYS if d != test]
        best = None
        for g in grid:
            hits = np.array([e['xy'] for d in train for e, yy in zip(cache[(d, g)][0], cache[(d, g)][1]) if yy == 1 and e['xy']])
            for rad in (0.02, 0.04, 0.06, 0.09, 0.13, 1.0):
                fs = []
                for d in train:
                    E, y, truth = cache[(d, g)]
                    # the zone from the OTHER training day only, so the radius is not chosen on its own hits
                    oth = [o for o in train if o != d][0]
                    hz = np.array([e['xy'] for e, yy in zip(cache[(oth, g)][0], cache[(oth, g)][1]) if yy == 1 and e['xy']])
                    keep = [e for e in E if e['xy'] and len(hz) and np.min(np.hypot(*(hz - e['xy']).T)) <= rad]
                    m = D.match(keep, truth, L.TOL)
                    fs.append((m['in']['f1'] + m['out']['f1']) / 2)
                if best is None or np.mean(fs) > best[0]:
                    best = (np.mean(fs), g, rad, hits)
        _, g, rad, hits = best
        E, y, truth = cache[(test, g)]
        keep = [e for e in E if e['xy'] and len(hits) and np.min(np.hypot(*(hits - e['xy']).T)) <= rad]
        m = D.match(keep, truth, L.TOL)
        m0 = D.match(E, truth, L.TOL)
        rep[test] = {'conf': g[0], 'thr': g[1], 'radius': rad, 'f1_gated': round((m['in']['f1'] + m['out']['f1']) / 2, 3),
                     'f1_ungated': round((m0['in']['f1'] + m0['out']['f1']) / 2, 3),
                     'in': [m['in'][k] for k in ('precision', 'recall', 'f1')], 'out': [m['out'][k] for k in ('precision', 'recall', 'f1')]}
        print(test, json.dumps(rep[test]), flush=True)
    json.dump(rep, open(OUT / ('binary_gated%s.json' % ('_' + SRC if SUF else '')), 'w'), indent=1)


if __name__ == '__main__' and sys.argv[1] in ('positions', 'gated'):
    positions(sys.argv[2]) if sys.argv[1] == 'positions' else gated_eval()


# ------------------------------------------------------------------ 07.10: side change + movement + door zone

def motion_eval():
    """An entry/exit counts when (1) the model's side of the track flips and holds `conf` trusted votes, (2) the
    person's feet moved the right way around the flip (median place 1.5 s after minus 1.5 s before, projected on
    the 'into the shop' direction learned from the training days' right entries; an exit must go the other way by
    at least `move`), (3) the flip is where crossings happen (zone of radius `rad` around training hits).
    conf/thr/move/rad from two days, scored on the third."""
    import door_learn as L
    import door_v2 as D
    P = {d: json.load(open(OUT / ('binary_pos_%s%s.json' % (POS_SUF, d)))) for d in DAYS}
    track_pos = {}
    for d in DAYS:
        tp = {}
        for key, xy in P[d].items():
            t, w = key.split('_')
            tp.setdefault(int(w), []).append((float(t), xy))
        track_pos[d] = {w: (np.array([x[0] for x in sorted(v)]), np.array([x[1] for x in sorted(v)])) for w, v in tp.items()}

    def disp(d, e):
        tt = track_pos[d].get(e['w'])
        if tt is None:
            return None
        ts, xy = tt
        t0 = e['t0']
        b, a = xy[(ts >= t0 - 1.5) & (ts < t0)], xy[(ts > t0) & (ts <= t0 + 1.5)]
        if not len(b) or not len(a):
            return None
        return np.median(a, 0) - np.median(b, 0)

    def scored(d, conf, thr):
        ev, spans = day_events(d, conf, thr)
        inside = lambda t: any(a <= t <= b for a, b in spans)
        truth = [t for t in D.truth(d, True) if inside(t['t'])]
        E = []
        for e in ev:
            x = dict(e, t0=e['t'], t=e['t'] + L.SHIFT, xy=P[d].get('%.2f_%d' % (round(e['t'], 2), e['w'])))
            x['disp'] = disp(d, x)
            E.append(x)
        y = L.label([(None, e['kind'], e['t'], e['w']) for e in E], truth)
        return E, y, truth

    grid = [(c, th) for c in (2, 3, 4) for th in (0.8, 0.9, 0.95)]
    cache = {(d, g): scored(d, *g) for d in DAYS for g in grid}

    def direction(days, g):
        v = [e['disp'] for d in days for e, yy in zip(*cache[(d, g)][:2]) if yy == 1 and e['disp'] is not None and e['kind'] == 'in']
        v += [-e['disp'] for d in days for e, yy in zip(*cache[(d, g)][:2]) if yy == 1 and e['disp'] is not None and e['kind'] == 'out']
        m = np.mean(v, 0) if v else np.array([0.0, 1.0])
        return m / (np.linalg.norm(m) + 1e-9)

    def zone(days, g):
        return np.array([e['xy'] for d in days for e, yy in zip(*cache[(d, g)][:2]) if yy == 1 and e['xy']])

    def keep(E, u, hz, move, rad):
        out = []
        for e in E:
            if rad < 1 and (not e['xy'] or not len(hz) or np.min(np.hypot(*(hz - e['xy']).T)) > rad):
                continue
            if move > -1:
                if e['disp'] is None:
                    continue
                pr = float(e['disp'] @ u) * (1 if e['kind'] == 'in' else -1)
                if pr < move:
                    continue
            out.append(e)
        return out

    rep = {}
    for test in DAYS:
        train = [d for d in DAYS if d != test]
        best = None
        for g in grid:
            for move in (-9, 0.0, 0.005, 0.01, 0.02, 0.04):
                for rad in (0.02, 0.04, 0.06, 0.09, 1.0):
                    fs = []
                    for d in train:
                        oth = [o for o in train if o != d]          # direction and zone from the other training day
                        E, y, truth = cache[(d, g)]
                        m = D.match(keep(E, direction(oth, g), zone(oth, g), move, rad), truth, L.TOL)
                        fs.append((m['in']['f1'] + m['out']['f1']) / 2)
                    if best is None or np.mean(fs) > best[0]:
                        best = (np.mean(fs), g, move, rad)
        _, g, move, rad = best
        E, y, truth = cache[(test, g)]
        m = D.match(keep(E, direction(train, g), zone(train, g), move, rad), truth, L.TOL)
        rep[test] = {'conf': g[0], 'thr': g[1], 'move': move, 'radius': rad, 'f1': round((m['in']['f1'] + m['out']['f1']) / 2, 3),
                     'in': [m['in'][k] for k in ('precision', 'recall', 'f1')], 'out': [m['out'][k] for k in ('precision', 'recall', 'f1')]}
        print(test, json.dumps(rep[test]), flush=True)
    json.dump(rep, open(OUT / ('binary_motion%s.json' % ('_' + SRC if SUF else '')), 'w'), indent=1)


if __name__ == '__main__' and sys.argv[1] == 'motion':
    motion_eval()
