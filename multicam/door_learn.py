"""The door rule learned from the owner's /door answers instead of tuned by hand (03.10.2026).

Candidates: every change of side (outside <-> inside, the doorway skipped) of every tracked person, with low hold
times so that nearly every real crossing is among them (door_v2.crossings at 0.24 s). Each candidate is described
by numbers (where the feet were before and after, how sure the two /inout classifiers are on each side, how long
each side held, how much the side flickers, the track's start and end, the advertising stand, the crowd) and
labelled by the owner's answers: a candidate one-to-one matched to an answered crossing of the same direction
within TOL is right, the rest wrong; candidates at a staff crossing without a direction are left out.
A boosting is trained on the other days and scores the held-out day; the threshold is picked on the training days.

usage: door_learn.py DAY=RUN.jsonl.gz [DAY=RUN.jsonl.gz ...]  -> data/door_v2/learn.json (+ learn_errors.json)"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
TOL = float(__import__('os').environ.get('RA_DOOR_TOL', '6.0'))
HOLD = 0.24
POSTER = np.array([902.0, 244.0])       # the stand's feet in the 2176 frame (seen at the same pixel all day)
NMS_S = 4.0                             # one person, one direction: one crossing within this
SHIFT = float(__import__('os').environ.get('RA_DOOR_SHIFT', '2.85'))   # the counter writes its event ~2.85 s after the
                                        # model sees the side change (door_research H2: 210 right pairs, p10 -4.2, p90 +1.6 s)


def sides(seq, key):
    return np.array([int(np.argmax(q[key])) if key in q else -1 for _, q in seq])


_GEO = {}


def floor_geo(cam='cam1'):
    """foot pixels (2176 frame) -> (signed distance to the door line, m, + inside; position along the door, 0..1
    between its posts) by the calibrated camera, as v2_prep.geometry does for the targets."""
    if cam not in _GEO:
        import person3d
        calib = json.load(open(ROOT / 'data' / 'calib_final.json'))
        reg = json.load(open(ROOT / 'data' / 'cam1_to_cam2_affine.json'))
        d = json.load(open(ROOT / 'data' / 'plan_door.json'))
        a, b, inside = (np.array(d[k], float) for k in ('door_a', 'door_b', 'inside'))
        c = person3d.Camera(cam, calib)

        def geo(feet):
            feet = np.asarray(feet, float).reshape(-1, 2) * (2560 / 2176)
            xy = c.on_plane(feet) * person3d.UNIT
            if cam == 'cam1':
                xy = person3d.map_cam1(xy, reg)
            ab = b - a
            L = np.linalg.norm(ab)
            sd = (ab[0] * (xy[:, 1] - a[1]) - ab[1] * (xy[:, 0] - a[0])) / L
            sgn = np.sign((ab[0] * (inside[1] - a[1]) - ab[1] * (inside[0] - a[0])))
            along = ((xy - a) @ ab) / (L * L)
            return sd * sgn, along
        _GEO[cam] = geo
    return _GEO[cam]


def add_floor(by):
    """q['sd'], q['al'] for every person tick with feet."""
    geo = floor_geo()
    refs = [q for seq in by.values() for _, q in seq if q.get('foot')]
    if refs:
        sd, al = geo([q['foot'] for q in refs])
        for q, x, y in zip(refs, sd, al):
            q['sd'], q['al'] = float(x), float(y)


def floor_crossings(seq, margin=0.3, slack=0.25):
    """Times the feet go from beyond -margin to beyond +margin of the door line (or back) while along the door
    (between its posts, +-slack of its width): [(t, 'in'|'out')]."""
    out, last, last_t = [], None, None
    for t, q in seq:
        if 'sd' not in q:
            continue
        if not (-slack <= q['al'] <= 1 + slack):
            continue
        side = 1 if q['sd'] > margin else 0 if q['sd'] < -margin else None
        if side is None:
            continue
        if last is not None and side != last:
            out.append(((last_t + t) / 2, 'in' if side == 1 else 'out'))
        last, last_t = side, t
    return out


def candidates(ticks):
    """[(features dict, kind, t, world)]: every side change of a tracked person (by the crop + place mix when the crop
    net ran, else the place), and the ends of broken tracks near the door -- a track born already inside (its outside
    part lost: an entry?) or lost while inside (an exit?), when the first / last second is not plainly inside."""
    by, crowd = {}, {}
    for r in ticks:
        crowd[r['t']] = len(r['p'])
        for q in r['p']:
            by.setdefault(q['w'], []).append((r['t'], q))
    add_floor(by)
    ends = []                                          # (world, t, 'start'|'end', side, foot)
    out = []
    for w, seq in by.items():
        key = 'io_mix' if any('io_mix' in q for _, q in seq) else 'io'
        seq = [(t, q) for t, q in seq if key in q]
        if len(seq) < 3:
            continue
        by[w] = seq
        ts = np.array([t for t, _ in seq])
        z = sides(seq, key)
        tri = os.environ.get('RA_DOOR_TRI', '1') == '1'
        if tri:                                        # the door is at the frame's edge: an entrant is often seen
            z = np.array([{0: 0, 2: 1, 1: 2}[int(v)] for v in z])   # only doorway -> inside; rank o 0, d 1, i 2
        runs = []
        for t, v in zip(ts, z):
            if v == 2 and not tri:
                continue
            if runs and runs[-1][0] == v and t - runs[-1][2] < 3.0:
                runs[-1][2] = t
            else:
                runs.append([int(v), t, t])
        runs = [r_ for r_ in runs if r_[2] - r_[1] >= HOLD]
        merged = []
        for r_ in runs:
            if merged and merged[-1][0] == r_[0]:
                merged[-1][2] = r_[2]
            else:
                merged.append(list(r_))
        first_foot = _mean(seq, ts, ts[0], ts[0] + 0.5, lambda q: q.get('foot'))
        last_foot = _mean(seq, ts, ts[-1] - 0.5, ts[-1], lambda q: q.get('foot'))
        INS = 2 if tri else 1
        ends.append((w, ts[0], 'start', (1 if merged[0][0] == INS else 0) if merged else 2, first_foot))
        ends.append((w, ts[-1], 'end', (1 if merged[-1][0] == INS else 0) if merged else 2, last_foot))
        changes = [((a[2] + b[1]) / 2) for a, b in zip(merged, merged[1:])]
        INSIDE = 2 if tri else 1
        for a, b in zip(merged, merged[1:]):
            t = (a[2] + b[1]) / 2
            kind = 'in' if b[0] > a[0] else 'out'
            aa, bb = [int(a[0] == INSIDE), a[1], a[2]], [int(b[0] == INSIDE), b[1], b[2]]
            if kind == 'in':
                aa[0], bb[0] = 0, 1
            else:
                aa[0], bb[0] = 1, 0
            f = features(seq, ts, t, aa, bb, key, changes, crowd)
            f['src'] = 0.0
            f['step_from'], f['step_to'] = float(a[0]), float(b[0])
            out.append((f, kind, round(t + SHIFT, 2), w))
        if os.environ.get('RA_DOOR_FLOOR', '1') == '1':
            for t, kind in floor_crossings(seq):
                aa, bb = ([0, t, t], [1, t, t]) if kind == 'in' else ([1, t, t], [0, t, t])
                f = features(seq, ts, t, aa, bb, key, changes, crowd)
                f['src'] = 3.0
                f['step_from'], f['step_to'] = np.nan, np.nan
                out.append((f, kind, round(t + SHIFT, 2), w))
        if tri:
            merged = [[int(m[0] == INSIDE), m[1], m[2]] for m in merged]
        if not merged or os.environ.get('RA_DOOR_FRAG', '1') == '0':
            continue
        not_in = lambda lo, hi: _mean(seq, ts, lo, hi, lambda q: 1.0 - q[key][1])
        if merged[0][0] == 1 and (not_in(ts[0], ts[0] + 1.0) or 0) >= 0.1:          # born inside, at the door
            t = ts[0]
            f = features(seq, ts, t, [0, t, t], merged[0], key, changes, crowd)
            f['src'] = 1.0
            f['step_from'], f['step_to'] = np.nan, np.nan
            out.append((f, 'in', round(t + SHIFT, 2), w))
        if merged[-1][0] == 1 and (not_in(ts[-1] - 1.0, ts[-1]) or 0) >= 0.1:        # lost inside, at the door
            t = ts[-1]
            f = features(seq, ts, t, merged[-1], [0, t, t], key, changes, crowd)
            f['src'] = 2.0
            f['step_from'], f['step_to'] = np.nan, np.nan
            out.append((f, 'out', round(t + SHIFT, 2), w))
    # the other half of a broken crossing: for an entry, somebody else's track that ended just before outside; for
    # an exit, somebody else's track that started just after outside -- and another candidate of the same kind near
    for f, kind, t, w in out:
        t0 = t - SHIFT
        foot = np.array([f['post_fx'], f['post_fy']]) if kind == 'in' else np.array([f['pre_fx'], f['pre_fy']])
        if kind == 'in':
            other = [(t0 - et, ef) for ew, et, what, side, ef in ends if ew != w and what == 'end' and side != 1 and -1.0 <= t0 - et <= 4.0]
        else:
            other = [(st - t0, sf) for sw, st, what, side, sf in ends if sw != w and what == 'start' and side != 1 and -1.0 <= st - t0 <= 4.0]
        if other:
            dt, ef = min(other, key=lambda x: abs(x[0]))
            f['frag_dt'] = float(dt)
            f['frag_px'] = float(np.linalg.norm(ef - foot)) if ef is not None and np.isfinite(foot).all() else np.nan
        else:
            f['frag_dt'], f['frag_px'] = np.nan, np.nan
        same = [abs(t2 - t) for f2, k2, t2, w2 in out if k2 == kind and w2 != w and abs(t2 - t) <= 6]
        f['same_kind_near_s'] = float(min(same)) if same else np.nan
        f['same_kind_n'] = float(len(same))
    return out


def _mean(seq, ts, lo, hi, f):
    v = [f(q) for t, q in seq if lo <= t <= hi]
    v = [x for x in v if x is not None]
    return np.mean(v, 0) if v else None


def features(seq, ts, t, a, b, key, changes, crowd):
    f = {}
    f['in'] = 1.0 if a[0] == 0 else 0.0
    f['before_s'] = min(30.0, a[2] - a[1])
    f['after_s'] = min(30.0, b[2] - b[1])
    f['from_start_s'] = min(60.0, t - ts[0])
    f['to_end_s'] = min(60.0, ts[-1] - t)
    f['track_s'] = min(600.0, ts[-1] - ts[0])
    f['flips_20s'] = sum(abs(c - t) <= 10 for c in changes)
    f['flips_60s'] = sum(abs(c - t) <= 30 for c in changes)
    for name, lo, hi in (('pre', t - 2.0, t - 0.3), ('post', t + 0.3, t + 2.0), ('pre5', t - 5, t - 2), ('post5', t + 2, t + 5)):
        for k in ('io', 'io_cnn'):
            m = _mean(seq, ts, lo, hi, lambda q: q.get(k))
            for j, cls in enumerate(('out', 'in', 'door')):
                f['%s_%s_%s' % (name, k, cls)] = float(m[j]) if m is not None else np.nan
        ft = _mean(seq, ts, lo, hi, lambda q: q.get('foot'))
        f['%s_fx' % name], f['%s_fy' % name] = (float(ft[0]), float(ft[1])) if ft is not None else (np.nan, np.nan)
        bx = _mean(seq, ts, lo, hi, lambda q: q['box'])
        f['%s_h' % name] = float(bx[3]) if bx is not None else np.nan
        sc = _mean(seq, ts, lo, hi, lambda q: q['s'])
        f['%s_score' % name] = float(sc) if sc is not None else np.nan
    for x in ('fx', 'fy'):
        f['d_' + x] = f['post_' + x] - f['pre_' + x]
        f['d5_' + x] = f['post5_' + x] - f['pre5_' + x]
    f['move'] = float(np.hypot(f['d_fx'], f['d_fy'])) if np.isfinite(f['d_fx']) else np.nan
    f['move5'] = float(np.hypot(f['d5_fx'], f['d5_fy'])) if np.isfinite(f['d5_fx']) else np.nan
    for name, lo, hi in (('pre', t - 2.0, t - 0.3), ('post', t + 0.3, t + 2.0)):
        v = [(q['sd'], q['al']) for t_, q in seq if lo <= t_ <= hi and 'sd' in q]
        f['%s_sd' % name] = float(np.median([x for x, _ in v])) if v else np.nan
        f['%s_al' % name] = float(np.median([y for _, y in v])) if v else np.nan
    f['d_sd'] = f['post_sd'] - f['pre_sd']
    v = [(abs(q['sd']), q['al']) for t_, q in seq if abs(t_ - t) <= 2.0 and 'sd' in q]
    f['min_abs_sd'] = float(min(x for x, _ in v)) if v else np.nan
    f['al_at_min'] = float(min(v)[1]) if v else np.nan
    allv = [q['sd'] for _, q in seq if 'sd' in q]
    f['track_sd_min'], f['track_sd_max'] = (float(min(allv)), float(max(allv))) if allv else (np.nan, np.nan)
    # distance from the camera (door_depth.py; nearer = inside): the person against the hall behind them and the floor
    # under their feet, before and after; for an exit the sign is turned so that 'outside -> inside' reads the same
    sg = 1.0 if a[0] == 0 else -1.0
    for name, lo, hi in (('pre', t - 2.5, t - 0.3), ('post', t + 0.3, t + 2.5)):
        v = [q['dep'] for t_, q in seq if lo <= t_ <= hi and 'dep' in q]
        if v:
            V = np.array([[x if x is not None else np.nan for x in row] for row in v], float)
            f['%s_dep' % name] = float(np.nanmedian(V[:, 0])) * sg
            f['%s_dep_behind' % name] = float(np.nanmedian(V[:, 0] - V[:, 2])) * sg
            f['%s_dep_floor' % name] = float(np.nanmedian(V[:, 0] - V[:, 3])) * sg if np.isfinite(V[:, 3]).any() else np.nan
        else:
            f['%s_dep' % name] = f['%s_dep_behind' % name] = f['%s_dep_floor' % name] = np.nan
    f['d_dep'] = f['post_dep'] - f['pre_dep'] if np.isfinite(f['post_dep']) and np.isfinite(f['pre_dep']) else np.nan
    # a track that jumps from one person to the next (two walking together): the feet and the box leap
    win = [(t_, q) for t_, q in seq if abs(t_ - t) <= 1.5 and q.get('foot')]
    jumps = [np.hypot(b_[1]['foot'][0] - a_[1]['foot'][0], b_[1]['foot'][1] - a_[1]['foot'][1]) / max(0.08, b_[0] - a_[0])
             for a_, b_ in zip(win, win[1:])]
    f['max_jump_px_s'] = float(max(jumps)) if jumps else np.nan
    hs = [q['box'][3] for _, q in win]
    f['h_ratio'] = float(max(hs) / max(1e-6, min(hs))) if hs else np.nan
    ws = [q['box'][2] for _, q in win]
    f['w_ratio'] = float(max(ws) / max(1e-6, min(ws))) if ws else np.nan
    allf = np.array([q['foot'] for _, q in seq if q.get('foot')])
    f['poster_px'] = float(np.min(np.linalg.norm(allf - POSTER, axis=1))) if len(allf) else np.nan
    f['track_spread_px'] = float(np.linalg.norm(allf.max(0) - allf.min(0))) if len(allf) else np.nan
    near = [crowd[k] for k in crowd if abs(k - t) <= 1.0]
    f['crowd'] = float(np.mean(near)) if near else 0.0
    return f


def cluster(cands, gap=NMS_S):
    """One person's candidates of one direction within `gap` of each other are one event: its features are the
    mean and the max of the members' (each source's presence counted), its time the floor crossing's when there is
    one, else the members' median."""
    if os.environ.get('RA_DOOR_CLUSTER', '1') != '1':
        return cands
    groups = {}
    for c in sorted(cands, key=lambda c: c[2]):
        g = groups.setdefault((c[3], c[1]), [])
        if g and c[2] - g[-1][-1][2] <= gap:
            g[-1].append(c)
        else:
            g.append([c])
    names = sorted({k for c in cands for k in c[0]})
    out = []
    for (w, kind), gs in groups.items():
        for g in gs:
            M = np.array([[c[0].get(n, np.nan) for n in names] for c in g], np.float32)
            with np.errstate(all='ignore'):
                import warnings
                warnings.simplefilter('ignore')
                mean, mx = np.nanmean(M, 0), np.nanmax(M, 0)
            f = {('m_' + n): float(v) for n, v in zip(names, mean)}
            f.update({('x_' + n): float(v) for n, v in zip(names, mx)})
            srcs = [c[0].get('src', 0) for c in g]
            for k in range(4):
                f['n_src%d' % k] = float(sum(sv == k for sv in srcs))
            f['n_members'] = float(len(g))
            f['span_s'] = float(g[-1][2] - g[0][2])
            fl = [c[2] for c in g if c[0].get('src') == 3]
            t = float(np.median(fl)) if fl else float(np.median([c[2] for c in g]))
            out.append((f, kind, round(t, 2), w))
    return out


def label(cands, truth):
    """1 right, 0 wrong, -1 at a staff crossing with no direction (left out)."""
    from scipy.optimize import linear_sum_assignment
    y = np.zeros(len(cands), int)
    mode = os.environ.get('RA_DOOR_LABEL', 'one')
    if mode == 'any':
        # every candidate of the right direction near a real crossing is right: one crossing seen as doorway -> inside
        # and outside -> doorway, or by two pieces of a broken track, is still one crossing (NMS keeps one at the end)
        for i, c in enumerate(cands):
            if any(t['kind'] == c[1] and abs(t['t'] - c[2]) <= TOL for t in truth):
                y[i] = 1
    for kind in ('in', 'out') if mode != 'any' else ():
        idx = [i for i, c in enumerate(cands) if c[1] == kind]
        T = [t for t in truth if t['kind'] == kind]
        if idx and T:
            C = np.array([[abs(cands[i][2] - t['t']) for t in T] for i in idx])
            r, c = linear_sum_assignment(np.where(C <= TOL, C, 1e6))   # most pairs within tol first, then the closest
            for i, j in zip(r, c):
                if C[i, j] <= TOL:
                    y[idx[i]] = 1
                    if mode == 'world':        # the same person's other steps of this crossing (o->d and d->i)
                        for k in idx:
                            if cands[k][3] == cands[idx[i]][3] and abs(cands[k][2] - T[j]['t']) <= TOL:
                                y[k] = 1
    neutral = [t['t'] for t in truth if t['kind'] == 'any']
    for i, c in enumerate(cands):
        if y[i] == 0 and any(abs(c[2] - n) <= TOL for n in neutral):
            y[i] = -1
    return y


def nms(cands, p, thr):
    """Kept candidates: p >= thr, and per person and direction only the surest within NMS_S."""
    keep = []
    order = np.argsort(-p)
    for i in order:
        if p[i] < thr:
            break
        c = cands[i]
        if any(cands[j][3] == c[3] and cands[j][1] == c[1] and abs(cands[j][2] - c[2]) <= NMS_S for j in keep):
            continue
        keep.append(i)
    return keep


def skipped(day):
    """RA_DOOR_SKIP='20260918 15:06-15:29;...': stretches left out of training and scoring (film seconds)."""
    import time as _t
    import day_movie
    out = []
    for part in os.environ.get('RA_DOOR_SKIP', '').split(';'):
        part = part.strip()
        if not part.startswith(day):
            continue
        a, b = part.split()[1].split('-')
        start, _ = day_movie.clock(day, str(ROOT))
        lt = _t.localtime(start)
        sod0 = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
        sec = lambda hm: int(hm[:2]) * 3600 + int(hm[3:5]) * 60 - sod0
        out.append((sec(a), sec(b)))
    return out


def load_day(day, path):
    import door_v2 as D
    ticks, spans = D._read(path)
    D.add_io(ticks, path)
    inside = lambda t: any(a <= t <= b for a, b in spans)
    truth = [t for t in D.truth(day, True) if inside(t['t'])]
    cands = cluster(candidates(ticks))
    skip = skipped(day)
    if skip:
        out_of = lambda t: any(a - 10 <= t <= b + 10 for a, b in skip)
        cands = [c for c in cands if not out_of(c[2])]
        truth = [t for t in truth if not out_of(t['t'])]
    keep = (lambda t: not out_of(t)) if skip else (lambda t: True)
    import door_state as S
    return {'day': day, 'cands': cands, 'by_world': S.tracks(ticks, split=False), 'truth': truth, 'y': label(cands, truth), 'spans': spans,
            'visits_truth': [t for t in D.truth(day, False) if inside(t['t']) and keep(t['t'])],
            'counter': [x for x in D.counter(day) if inside(x['t']) and keep(x['t'])]}


def map_features(d, kmap):
    """Candidates of a day with the side map's view added: how plainly outside the person was before and inside
    after (for an exit the other way round), whether they reached the deep zones within 8 s and how long they stayed."""
    by = d['by_world']
    allq = [q for seq in by.values() for _, q in seq]
    if allq and kmap is not None:
        import door_state as S
        P = kmap.predict_proba(np.array([S.xfeat(q) for q in allq], np.float32))[:, 1]
        for q, v in zip(allq, P):
            q['_pin'] = float(v)
    out = []
    for f, kind, t, w in d['cands']:
        f = dict(f)
        seq = by.get(w, [])
        t0 = t - SHIFT
        src = lambda lo, hi: [q.get('_pin', 0.5) for s_, q in seq if lo <= s_ - t0 <= hi]
        pre, post, before, after = src(-3, -0.5), src(0.5, 3), src(-8, 0), src(0, 8)
        flip = (lambda v: v) if kind == 'in' else (lambda v: 1 - v)        # 'outside before, inside after' as one scale
        f['map_pre'] = float(np.median([flip(v) for v in pre])) if pre else np.nan
        f['map_post'] = float(np.median([flip(v) for v in post])) if post else np.nan
        f['map_before_min'] = float(min(flip(v) for v in before)) if before else np.nan
        f['map_after_max'] = float(max(flip(v) for v in after)) if after else np.nan
        f['map_before_deep_s'] = float(sum(flip(v) < 0.1 for v in before) * 0.08)
        f['map_after_deep_s'] = float(sum(flip(v) > 0.9 for v in after) * 0.08)
        out.append((f, kind, t, w))
    return out


def with_maps(test, train):
    """Feature copies for one fold: the test day seen through a map of the training days, each training day through
    a map of the other training day(s) only -- the test day's answers never shape anything the model learns from."""
    import door_state as S
    fit = lambda days: S.fit_map([{'learn': {'cands': e['cands_raw'], 'y': e['y']}, 'by': e['by_world']} for e in days]) if days else None
    out = {test['day']: map_features(dict(test, cands=test['cands_raw']), fit(train))}
    for d in train:
        others = [e for e in train if e is not d]
        out[d['day']] = map_features(dict(d, cands=d['cands_raw']), fit(others))
    return out


def matrix(cands, names=None):
    names = names or sorted(cands[0][0])
    return np.array([[c[0][n] for n in names] for c in cands], np.float32), names


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          min_samples_leaf=10, random_state=0)


def evaluate(days):
    import door_v2 as D
    rep = {'tol_s': TOL, 'days': {}}
    errors = []
    for d in days:
        rep['days'][d['day']] = {'candidates': len(d['cands']), 'right': int((d['y'] == 1).sum()),
                                 'truth_in': sum(t['kind'] == 'in' for t in d['truth']), 'truth_out': sum(t['kind'] == 'out' for t in d['truth']),
                                 'candidate_recall': D.match([{'kind': c[1], 't': c[2], 'w': c[3]} for c in d['cands']], d['truth'], TOL)}
    names = None
    use_map = os.environ.get('RA_DOOR_MAP', '1') == '1'
    for d in days:
        d['cands_raw'] = d['cands']
    for test in days:
        train = [d for d in days if d is not test]
        if use_map:
            fold = with_maps(test, train)
            for d in days:
                d['cands'] = fold[d['day']]
            names = None
        Xs, ys = [], []
        for d in train:
            X, names = matrix(d['cands'], names)
            m_ = d['y'] >= 0
            Xs.append(X[m_]); ys.append(d['y'][m_])
        clf = model().fit(np.concatenate(Xs), np.concatenate(ys))
        # threshold on the training days (in-sample scores are optimistic; take it from their own cross-fit)
        best_thr, best_f = 0.5, -1
        oof = []
        for d in train:
            others = [e for e in train if e is not d]
            if others:
                Xo = np.concatenate([matrix(e['cands'], names)[0][e['y'] >= 0] for e in others])
                yo = np.concatenate([e['y'][e['y'] >= 0] for e in others])
                p = model().fit(Xo, yo).predict_proba(matrix(d['cands'], names)[0])[:, 1]
            else:
                p = clf.predict_proba(matrix(d['cands'], names)[0])[:, 1]
            oof.append((d, p))
        for thr in np.arange(0.02, 0.951, 0.02):
            f = []
            for d, p in oof:
                k = nms(d['cands'], p, thr)
                m = D.match([{'kind': d['cands'][i][1], 't': d['cands'][i][2], 'w': d['cands'][i][3]} for i in k], d['truth'], TOL)
                f.append((m['in']['f1'] + m['out']['f1']) / 2)
            if np.mean(f) > best_f:
                best_f, best_thr = float(np.mean(f)), float(thr)
        p = clf.predict_proba(matrix(test['cands'], names)[0])[:, 1]
        allk = nms(test['cands'], p, 0.0)              # every scored event, for the owner's check (door_check.py)
        preds_all = [{'day': test['day'], 'kind': test['cands'][i][1], 't': test['cands'][i][2], 'w': test['cands'][i][3],
                      'p': round(float(p[i]), 4)} for i in allk]
        json.dump(preds_all, open(ROOT / 'data' / 'door_v2' / ('learn_preds_%s.json' % test['day']), 'w'))
        k = nms(test['cands'], p, best_thr)
        pred = [{'kind': test['cands'][i][1], 't': test['cands'][i][2], 'w': test['cands'][i][3], 'p': float(p[i])} for i in k]
        r = rep['days'][test['day']]
        r['threshold'] = round(best_thr, 2)
        r['model'] = D.match(pred, test['truth'], TOL)
        r['model_tol3'] = D.match(pred, test['truth'], 3.0)
        r['model_visits'] = D.match(D.undither(pred), test['visits_truth'], TOL)
        r['counter'] = D.match(test['counter'], test['truth'], 1.0)
        r['counter_visits'] = D.match(test['counter'], test['visits_truth'], 1.0)
        # what is left wrong, for a look
        from scipy.optimize import linear_sum_assignment
        for kind in ('in', 'out'):
            P = [x for x in pred if x['kind'] == kind]
            T = [t for t in test['truth'] if t['kind'] == kind]
            hitP, hitT = set(), set()
            if P and T:
                C = np.array([[abs(a['t'] - b['t']) for b in T] for a in P])
                rr, cc = linear_sum_assignment(np.where(C <= TOL, C, 1e6))
                for i, j in zip(rr, cc):
                    if C[i, j] <= TOL:
                        hitP.add(i); hitT.add(j)
            neutral = [t['t'] for t in test['truth'] if t['kind'] == 'any']
            errors += [{'day': test['day'], 'what': 'false', 'kind': kind, 't': x['t'], 'w': x['w'], 'p': round(x['p'], 3)} for i, x in enumerate(P)
                       if i not in hitP and not any(abs(x['t'] - n) <= TOL for n in neutral)]
            errors += [{'day': test['day'], 'what': 'missed', 'kind': kind, 't': round(t['t'], 2), 'event': t['id'], 'staff': t['staff']} for j, t in enumerate(T) if j not in hitT]
    rep['features'] = names
    json.dump(rep, open(ROOT / 'data' / 'door_v2' / 'learn.json', 'w'), indent=1)
    json.dump(errors, open(ROOT / 'data' / 'door_v2' / 'learn_errors.json', 'w'), indent=1)
    return rep, errors


if __name__ == '__main__':
    days = []
    for a in sys.argv[1:]:
        day, path = a.split('=', 1)
        days.append(load_day(day, path))
    rep, errors = evaluate(days)
    for d, r in rep['days'].items():
        cr = r['candidate_recall']
        line = '%s candidates %d (right %d) candidate recall in %.2f out %.2f' % (d, r['candidates'], r['right'], cr['in']['recall'], cr['out']['recall'])
        if 'model' in r:
            m, c = r['model'], r['counter']
            line += ' | thr %.2f model in P%.2f R%.2f out P%.2f R%.2f | counter in P%.2f R%.2f out P%.2f R%.2f' % (
                r['threshold'], m['in']['precision'], m['in']['recall'], m['out']['precision'], m['out']['recall'],
                c['in']['precision'], c['in']['recall'], c['out']['precision'], c['out']['recall'])
        print(line)
    print(len(errors), 'errors')
