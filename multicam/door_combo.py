"""Entries and exits: the owner's binary inside/outside model + movement + door zone, with track stitching, and its
combination with the learned door rule (07.10.2026). Every choice is made on two days and scored on the third.

usage: door_combo.py rules           -> learn_preds_<src>_<day>.json (the rule's out-of-day p per candidate)
       door_combo.py eval            -> data/door_v2/combo_<src>.json
RA_BIN_SRC = sam31 | micro1s3 (the tracks)."""
import itertools
import json
import os
import shutil
import sys
from pathlib import Path

for k in ('RA_DOOR_NOCNN', 'RA_DOOR_NODEPTH', 'RA_DOOR_NOLK', 'RA_DOOR_NOTAP', 'RA_DOOR_STITCH'):
    os.environ.setdefault(k, '1')
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'door_v2'
DAYS = ('20260917', '20260918', '20260919')
SRC = os.environ.get('RA_BIN_SRC', 'micro1s3')
SUF = ('' if SRC == 'sam31' else SRC + '_') + os.environ.get('RA_BIN_VER', '')
POS_SUF = ('' if SRC == 'sam31' else SRC + '_') + os.environ.get('RA_POS_VER', '')   # 08.10: pc = SAM pieces


def rules():
    import door_learn as L
    days = [L.load_day(d, OUT / ('%s_%s.jsonl.gz' % (d, SRC))) for d in DAYS]
    L.evaluate(days)
    for d in DAYS:
        shutil.copy(OUT / ('learn_preds_%s.json' % d), OUT / ('learn_preds_%s_%s.json' % (SRC, d)))



def stitch_tracks(tr, stitch, roots=None):
    """{pid: [(t, p, xy)]} -> the same with pieces joined: one that starts where and when another ended (<= gap s,
    feet <= dist apart, the earlier one never seen after) takes the earlier one's number."""
    gap, dist = stitch
    parent = {}
    root = lambda k: root(parent[k]) if k in parent else k
    order = sorted(tr, key=lambda k: tr[k][0][0])
    for b in order:
        tb, xyb = tr[b][0][0], tr[b][0][2]
        best = None
        for a in order:
            if a == b or tr[a][0][0] >= tb or root(a) == root(b):
                continue
            ta, xya = tr[a][-1][0], tr[a][-1][2]
            if not (tb - gap <= ta <= tb + 0.25) or any(x[0] > tb + 0.25 for x in tr[a]):
                continue
            if xya is None or xyb is None:
                continue
            dd = float(np.hypot(xya[0] - xyb[0], xya[1] - xyb[1]))
            if dd <= dist and (best is None or dd < best[0]):
                best = (dd, a)
        if best:
            parent[b] = best[1]
    merged = {}
    for k, v in tr.items():
        merged.setdefault(root(k), []).extend(v)
        if roots is not None:
            roots[k] = root(k)
    return {k: sorted(v) for k, v in merged.items()}


def flip_events(tracks, conf, thr, shift=0.0):
    """{name: [(t, p_inside, xy)]} -> side flips held for `conf` trusted votes (p >= thr or <= 1 - thr); the first side
    of a track is no event; each with the place at the flip and the feet's move (median 1.5 s after - before)."""
    ev = []
    for name, xs in tracks.items():
        ts = np.array([x[0] for x in xs])
        xy = [x[2] for x in xs]
        side, pend, n, t0, i0 = None, None, 0, None, None
        for i, (t, p, _) in enumerate(xs):
            v = 1 if p >= thr else (0 if p <= 1 - thr else None)
            if v is None:
                continue
            if v == side:
                pend, n = None, 0
                continue
            if pend != v:
                pend, n, t0, i0 = v, 0, t, i
            n += 1
            if n >= conf:
                if side is not None:
                    b = [xy[j] for j in range(len(xs)) if t0 - 1.5 <= ts[j] < t0 and xy[j]]
                    a = [xy[j] for j in range(len(xs)) if t0 < ts[j] <= t0 + 1.5 and xy[j]]
                    disp = (np.median(a, 0) - np.median(b, 0)) if a and b else None
                    ev.append({'kind': 'in' if v == 1 else 'out', 't': t0 + shift, 'w': name, 'xy': xy[i0], 'disp': disp})
                side, pend, n = v, None, 0
    return ev


# ------------------------------------------------------------------ data of a day

class Day:
    def __init__(self, d):
        import door_learn as L
        import door_v2 as D
        self.d = d
        self.S = json.load(open(OUT / ('binary_%s%s.json' % (SUF, d))))
        self.P = json.load(open(OUT / ('binary_pos_%s%s.json' % (POS_SUF, d))))
        spans = [(st['a'], st['end']) for st in self.S.values()]
        inside = lambda t: any(a <= t <= b for a, b in spans)
        self.truth = [t for t in D.truth(d, True) if inside(t['t'] - L.SHIFT)]
        rp = OUT / ('learn_preds_%s_%s.json' % (SRC, d))
        self.rule = json.load(open(rp)) if rp.exists() else []          # t already on the truth's clock
        self.cache = {}

    def tracks(self, stitch):
        """{(tag, pid): [(t, p_inside, xy)]} -- with stitching, a piece that starts where and when another ended
        (<= GAP s, feet <= DIST apart, never seen together after) takes that one's number."""
        key = ('tr', stitch)
        if key in self.cache:
            return self.cache[key]
        out = {}
        for tag, st in self.S.items():
            tr = {}
            for t, pid, p in st['obs']:
                tr.setdefault(pid, []).append((t, p, self.P.get('%.2f_%d' % (round(t, 2), pid))))
            for v in tr.values():
                v.sort()
            if stitch:
                tr = stitch_tracks(tr, stitch)
            for pid, v in tr.items():
                out[(tag, pid)] = v
        self.cache[key] = out
        return out

    def binary_events(self, conf, thr, stitch):
        """side flips held for `conf` trusted votes; each with its place and the feet's move around it."""
        import door_learn as L
        key = ('ev', conf, thr, stitch)
        if key in self.cache:
            return self.cache[key]
        ev = flip_events({'%s:%s' % k: v for k, v in self.tracks(stitch).items()}, conf, thr, L.SHIFT)
        self.cache[key] = ev
        return ev


# ------------------------------------------------------------------ the decision

def learned(days, ev_of):
    """'into the shop' direction and the places of right events, from the given days."""
    import door_learn as L
    v, z = [], []
    for D_, E in ((D_, ev_of(D_)) for D_ in days):
        y = L.label([(None, e['kind'], e['t'], e['w']) for e in E], D_.truth)
        for e, yy in zip(E, y):
            if yy != 1:
                continue
            if e['disp'] is not None:
                v.append(e['disp'] if e['kind'] == 'in' else -e['disp'])
            if e['xy']:
                z.append(e['xy'])
    u = np.mean(v, 0) if v else np.array([0.0, 1.0])
    return u / (np.linalg.norm(u) + 1e-9), np.array(z)


def gate(E, u, hz, move, rad):
    out = []
    for e in E:
        if rad < 1 and (not e['xy'] or not len(hz) or np.min(np.hypot(*(hz - e['xy']).T)) > rad):
            continue
        if move > -1:
            if e['disp'] is None or float(e['disp'] @ u) * (1 if e['kind'] == 'in' else -1) < move:
                continue
        out.append(e)
    return out


def combine(B, R, lo, hi, join=4.0):
    """B: binary events kept by the gate; R: the rule's candidates with p (already one per person and direction within
    NMS). A rule candidate counts at p >= hi, or at p >= lo when a binary event of the same direction is within join s.
    Rule and binary events are paired one to one (same direction, nearest within join): a binary event paired with a
    counted rule event is the same crossing; an unpaired binary event counts on its own (hi <= 1) -- two people
    entering together stay two events."""
    if lo > 1:                                              # the binary logic alone
        return list(B)
    pairs = {}
    used = set()
    for r_i, r in sorted(enumerate(R), key=lambda x: -x[1]['p']):
        cand = [(abs(b['t'] - r['t']), b_i) for b_i, b in enumerate(B)
                if b_i not in used and b['kind'] == r['kind'] and abs(b['t'] - r['t']) <= join]
        if cand:
            b_i = min(cand)[1]
            pairs[r_i] = b_i
            used.add(b_i)
    out, absorbed = [], set()
    for r_i, r in enumerate(R):
        if r['p'] >= hi or (r['p'] >= lo and r_i in pairs):
            out.append(dict(r, bw=B[pairs[r_i]]['w']) if r_i in pairs else dict(r))   # bw: the side track (role)
            if r_i in pairs:
                absorbed.add(pairs[r_i])
    if hi <= 1:
        out += [dict(b) for b_i, b in enumerate(B) if b_i not in absorbed]
    return out


def f1(D_, events):
    import door_learn as L
    import door_v2 as Dv
    m = Dv.match(events, D_.truth, L.TOL)
    return m, (m['in']['f1'] + m['out']['f1']) / 2


def evaluate():
    days = {d: Day(d) for d in DAYS}
    have_rule = all(days[d].rule for d in DAYS)
    stitches = (None, (1.5, 0.04), (2.5, 0.06), (4.0, 0.08))
    bins = [(c, th) for c in (2, 3, 4) for th in (0.9, 0.95)]
    gates = [(mv, rd) for mv in (-9, 0.0, 0.01, 0.02) for rd in (0.04, 0.06, 0.09, 1.0)]
    mixes = [(9, 9)] + ([(lo, hi) for lo in (0.1, 0.2, 0.3) for hi in (0.5, 0.6, 0.7, 0.8, 0.9, 2.0)] if have_rule else [])
    rep = {'src': SRC, 'days': {}}
    for test in DAYS:
        train = [d for d in DAYS if d != test]
        best = None
        for st, bp in itertools.product(stitches, bins):
            ev_of = lambda D_: D_.binary_events(bp[0], bp[1], st)
            for mv, rd in gates:
                gated = {}
                for d in train:
                    oth = [days[o] for o in train if o != d]
                    u, hz = learned(oth, ev_of)
                    gated[d] = gate(ev_of(days[d]), u, hz, mv, rd)
                for lo, hi in mixes:
                    sc = np.mean([f1(days[d], combine(gated[d], days[d].rule, lo, hi))[1] for d in train])
                    if best is None or sc > best[0]:
                        best = (sc, st, bp, mv, rd, lo, hi)
        _, st, bp, mv, rd, lo, hi = best
        ev_of = lambda D_: D_.binary_events(bp[0], bp[1], st)
        u, hz = learned([days[d] for d in train], ev_of)
        E = combine(gate(ev_of(days[test]), u, hz, mv, rd), days[test].rule, lo, hi)
        m, f = f1(days[test], E)
        # the rule alone at its best threshold on the training days, for the same table
        rule_only = None
        if have_rule:
            th = max(np.arange(0.1, 0.95, 0.05), key=lambda h: np.mean([f1(days[d], combine([], days[d].rule, h, h))[1] for d in train]))
            rule_only = round(f1(days[test], combine([], days[test].rule, th, th))[1], 3)
        rep['days'][test] = {'f1': round(f, 3), 'rule_only': rule_only, 'stitch': st, 'conf': bp[0], 'thr': bp[1], 'move': mv,
                             'radius': rd, 'mix': [lo, hi], 'in': [m['in'][k] for k in ('precision', 'recall', 'f1')],
                             'out': [m['out'][k] for k in ('precision', 'recall', 'f1')]}
        print(test, json.dumps(rep['days'][test]), flush=True)
    rep['mean_f1'] = round(float(np.mean([r['f1'] for r in rep['days'].values()])), 3)
    if have_rule:
        rep['mean_rule_only'] = round(float(np.mean([r['rule_only'] for r in rep['days'].values()])), 3)
    print('mean', rep['mean_f1'], 'rule only', rep.get('mean_rule_only'), flush=True)
    json.dump(rep, open(OUT / ('combo_%s.json' % SRC), 'w'), indent=1)


FINAL = dict(stitch=(1.5, 0.04), conf=2, thr=0.95, move=0.01, rad=0.09, lo=0.1, hi=2.0)


def fit_final():
    """The live setting: the parameters the out-of-day choice took on the small SAM's tracks, the direction into the
    shop and the zone of right crossings from all three days -> data/door_v2/combo_final.json."""
    days = [Day(d) for d in DAYS]
    f = FINAL
    u, hz = learned(days, lambda D_: D_.binary_events(f['conf'], f['thr'], f['stitch']))
    sc = [f1(D_, combine(gate(D_.binary_events(f['conf'], f['thr'], f['stitch']), u, hz, f['move'], f['rad']), D_.rule, f['lo'], f['hi']))[1]
          for D_ in days]
    json.dump(dict(f, src=SRC, u=u.tolist(), zone=hz.tolist(), in_sample_f1=[round(x, 3) for x in sc]),
              open(OUT / 'combo_final.json', 'w'), indent=1)
    print('final', json.dumps({'in_sample_f1': sc, 'zone_points': len(hz)}), flush=True)


class Live:
    """The combination on a live window: call add(t, w, frame_rgb, mask, foot_xy) for every person of every tick
    (once each), then events(rule_cands) with the rule's candidates at p >= lo (door_rule.apply with thr lo)."""

    def __init__(self):
        sys.path.insert(0, str(ROOT / 'inout_lab'))
        from binary_door import BinaryDoorClassifier
        self.clf = BinaryDoorClassifier()
        self.cfg = json.load(open(OUT / 'combo_final.json'))
        self.u, self.hz = np.array(self.cfg['u']), np.array(self.cfg['zone'])
        self.obs = {}                                       # row key -> (t, p_inside, foot)
        import door_line
        self.line = door_line.load() if os.environ.get('RA_DOOR_LINE', '1') == '1' else None
        lf = OUT / 'line_final.json'
        self.line_cfg = json.load(open(lf)) if lf.exists() and os.environ.get('RA_LINE_ONLY') == '1' else None

    def add(self, key, t, frame_rgb, mask, foot_xy):
        if self.line is not None and os.environ.get('RA_LINE_ONLY') == '1':   # 08.10: only the owner's line
            import door_line
            m = np.asarray(mask) > 0
            if m.any():
                lf = self.line_cfg
                if lf:                                      # with hysteresis (door_line_tune.py -> line_final.json)
                    d = door_line.depth(self.line, m)
                    p = 1.0 if d >= lf['h_in'] else (0.0 if d <= -lf['h_out'] else 0.5)
                else:
                    p = 1.0 if door_line.inside(self.line, m) else 0.0
                self.obs[key] = (t, p, foot_xy)
            return
        try:
            p = self.clf.predict_rgb(frame_rgb, mask)['p_inside']
        except ValueError:
            return
        if self.line is not None:
            import door_line
            if door_line.inside(self.line, np.asarray(mask) > 0):
                p = 1.0                                     # the owner's shop line: a piece of mask past it = inside
        self.obs[key] = (t, p, foot_xy)

    def events(self, rule_cands, person_of):
        """person_of: row key -> the person's current number (ReID numbers change as the window grows)."""
        c = self.cfg
        tr = {}
        for key, w in person_of.items():
            if key in self.obs:
                tr.setdefault(w, []).append(self.obs[key])
        tr = {k: sorted(v) for k, v in tr.items()}
        roots = {k: k for k in tr}
        if c['stitch']:
            tr = stitch_tracks(tr, tuple(c['stitch']), roots)
        self.members = {}                                   # 08.10: the person -> its stitched keys (for the role)
        for k, r in roots.items():
            self.members.setdefault(str(r), []).append(k)
        lf = self.line_cfg or {}
        B = flip_events({str(k): v for k, v in tr.items()}, lf.get('conf', c['conf']), c['thr'])
        if lf.get('gate', True):
            B = gate(B, self.u, self.hz, c['move'], c['rad'])
        if os.environ.get('RA_COMBO_ALONE') == '1' or rule_cands is None:   # the side model (or line) alone
            return B
        return combine(B, rule_cands, c['lo'], c['hi'])


if __name__ == '__main__':
    {'rules': rules, 'eval': evaluate, 'final': fit_final}[sys.argv[1]]()
