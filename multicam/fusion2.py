"""People on one floor plan from both cameras, 25 fps, height-aware.

Pipeline per clip:
  1. place every detection on the floor (person3d): foot when visible, else head
     at a prior height;
  2. drop non-people: box pixel height must match a 1.1-2.1 m person standing at
     that floor spot (kills a display stand boxed as a person); drop the mall
     gallery beyond the glass line;
  3. track per camera in metres with constant velocity, surviving 1.5 s behind a
     stand; re-place head-only detections with the track's own measured height;
  4. merge tracks of the two cameras that stand on the same spot at the same
     time; stitch gaps with walking reachability, clothing colour and height."""
import json, os, numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import medfilt
from person3d import Camera, place, map_cam1
from topview import UNIT

REG = json.load(open('data/cam1_to_cam2_affine.json'))
DOOR = json.load(open('data/plan_door.json'))
APP = json.load(open('data/appearance_thresholds.json')) if os.path.exists('data/appearance_thresholds.json') else {}
GATE_CROSS = APP.get('cross', {}).get('threshold', 1.6)
GATE_WITHIN = float(os.environ.get('RA_GATE', APP.get('within', {}).get('threshold', 1.3)))
RATIO = 0.8          # a link must be this much better than the runner-up
MIN_BOX_H = 110      # px: below this the person is too small for their embedding to identify them


def door_side(xy):
    """Signed distance (m) to the door line extended along the glass; positive = inside."""
    a, b, inside = (np.array(DOOR[k]) for k in ('door_a', 'door_b', 'inside'))
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]])
    if (inside - a) @ n < 0:
        n = -n
    return (xy - a) @ n


def inside_shop(xy, margin=0.6):
    # the glass front runs along the door line; people beyond it are in the mall gallery
    return door_side(xy) > -margin


def expected_px_height(cam, xy_common, h=1.7):
    A = np.eye(2)
    own = xy_common.copy()
    if cam.name == 'cam1':
        M = np.array(REG['M'])
        own = (own - M[:, 2]) @ np.linalg.inv(M[:, :2]).T
    own = own / UNIT
    import cv2
    base = np.c_[own, np.zeros(len(own))]
    top = np.c_[own, np.full(len(own), h / UNIT)]
    rv, _ = cv2.Rodrigues(cam.R)
    pb = cv2.projectPoints(base.astype(np.float64), rv, cam.t, cam.K, cam.dist)[0].reshape(-1, 2)
    pt = cv2.projectPoints(top.astype(np.float64), rv, cam.t, cam.K, cam.dist)[0].reshape(-1, 2)
    return np.linalg.norm(pb - pt, axis=1)


def filter_people(cams, dets, P):
    keep = {}
    for cam, d in dets.items():
        pl = P[cam]
        ok = np.all(np.isfinite(pl['xy']), axis=1)
        exp = np.full(len(d), np.nan)
        exp[ok] = expected_px_height(cams[cam], pl['xy'][ok])
        boxh = d[:, 4] - d[:, 2]
        ratio = boxh / np.maximum(exp, 1)
        plausible = ok & (ratio < 1.45) & (ratio > 0.25)
        hv = pl['vis'] & np.isfinite(pl['h'])
        plausible &= ~hv | ((pl['h'] > 1.05) & (pl['h'] < 2.15))
        plausible &= inside_shop(np.where(ok[:, None], pl['xy'], 0))
        keep[cam] = plausible
    return keep


def track(t, xy, vis, gate=0.5, max_miss=1.5):
    """Nearest-neighbour + constant velocity on the floor, Hungarian per frame."""
    tracks, active = [], []
    for tt in np.unique(t):
        idx = np.nonzero(t == tt)[0]
        P = xy[idx]
        pred = []
        for a in active:
            tr = tracks[a]
            lt = tr['t'][-1]
            pred.append(tr['p'][-1] + tr['v'] * (tt - lt))
        assigned, still = set(), []
        if active and len(P):
            D = np.linalg.norm(np.array(pred)[:, None] - P[None], axis=2)
            gaps = np.array([tt - tracks[a]['t'][-1] for a in active])
            G = gate + 0.9 * gaps
            D = np.where(D <= G[:, None], D, 1e6)
            r, c = linear_sum_assignment(D)
            for i, j in zip(r, c):
                if D[i, j] >= 1e6:
                    continue
                a = active[i]; tr = tracks[a]
                dt = max(tt - tr['t'][-1], 1e-3)
                v = (P[j] - tr['p'][-1]) / dt
                tr['v'] = 0.8 * tr['v'] + 0.2 * np.clip(v, -2.0, 2.0)
                tr['t'].append(tt); tr['p'].append(P[j]); tr['i'].append(idx[j])
                assigned.add(j); still.append(a)
        for a in active:
            if a not in still and tt - tracks[a]['t'][-1] <= max_miss:
                still.append(a)
        for j in range(len(P)):
            if j not in assigned:
                tracks.append({'t': [tt], 'p': [P[j]], 'v': np.zeros(2), 'i': [idx[j]]})
                still.append(len(tracks) - 1)
        active = still
    return [tr for tr in tracks if len(tr['t']) >= 8]


def build(cams, dets, feats, embs=None, clean=None):
    P = place(cams, REG, dets)
    keep = filter_people(cams, dets, P)
    per_cam = {}
    stats = {}
    for cam, d in dets.items():
        k = np.nonzero(keep[cam])[0]
        from imtrack import track_camera
        from imtrack import split_on_appearance_change, unit
        E = embs[cam][k] if embs is not None else None
        local = track_camera(d[k], feats[cam][k], emb=E)
        if os.environ.get('RA_SPLIT', '1') == '1':   # cutting a track where the clothing changes
            local = [piece for tr in local for piece in split_on_appearance_change(d[k], feats[cam][k], tr, emb=E)]
        trs = [{'i': list(tr)} for tr in local]
        out = []
        for tr in trs:
            ii = k[np.array(tr['i'])]
            hs = P[cam]['h'][ii]
            hs = hs[np.isfinite(hs)]
            h_own = float(np.median(hs)) if len(hs) >= 5 else np.nan
            xy = P[cam]['xy'][ii].copy()
            if np.isfinite(h_own):          # re-place head-only frames with this person's own height
                c = cams[cam]
                head = d[ii, 8:10]
                xyh = c.on_plane(head, z=h_own / UNIT) * UNIT
                if cam == 'cam1':
                    xyh = map_cam1(xyh, REG)
                vis = P[cam]['vis'][ii]
                xy = np.where(vis[:, None], xy, xyh)
            if len(xy) >= 5:
                xy = np.c_[medfilt(xy[:, 0], 5), medfilt(xy[:, 1], 5)]
            # How much appearance evidence this piece really carries: a person 12 m
            # away is 60 px tall and their embedding is close to everyone else's.
            box_h = float(np.median(d[ii, 4] - d[ii, 2]))
            item = {'cam': cam, 't': d[ii, 0], 'xy': xy, 'det': ii, 'h': h_own,
                    'app': feats[cam][ii].mean(0), 'box_h': box_h}
            if embs is not None:
                # Prototypes are built only from CLEAN frames -- where the box is mostly
                # this person's own silhouette and no one else overlaps it. A crop holding
                # half of the companion describes the pair, not the person.
                e = unit(embs[cam][ii].astype(np.float32))
                ok = clean[cam][ii] if clean is not None else np.ones(len(ii), bool)
                sel = np.nonzero(ok)[0]
                if len(sel) < 5:
                    sel = np.arange(len(e))
                    item['clean_share'] = float(ok.mean()) if clean is not None else 1.0
                    item['strong'] = bool(box_h >= MIN_BOX_H and item['clean_share'] >= 0.2)
                else:
                    item['clean_share'] = float(ok.mean())
                item['strong'] = bool(box_h >= MIN_BOX_H and item.get('clean_share', 1.0) >= 0.2)
                thirds = np.array_split(sel, 3)
                item['protos'] = unit(np.stack([e[t].mean(0) for t in thirds if len(t)]))
            out.append(item)
        per_cam[cam] = out
        stats[cam] = {'detections': len(d), 'kept': int(keep[cam].sum()), 'tracks': len(out),
                      'foot_visible_share': float(P[cam]['vis'][k].mean()) if len(k) else 0.0}
    return per_cam, stats


def overlap(a, b):
    ta, tb = np.round(a['t'], 2), np.round(b['t'], 2)
    common, ia, ib = np.intersect1d(ta, tb, return_indices=True)
    if len(common) < 12:
        return None
    return float(np.median(np.linalg.norm(a['xy'][ia] - b['xy'][ib], axis=1)))


def time_overlap(a, b):
    return max(0.0, min(a['t'][-1], b['t'][-1]) - max(a['t'][0], b['t'][0]))


def app_dist(a, b):
    """Lab-colour fallback distance (used only when no ReID embeddings are loaded)."""
    f = [0, 1, 2, 6, 7, 8, 12, 13, 14]
    w = np.array([0.5, 1, 1, 0.5, 1, 1, 0.5, 1, 1])
    return float(np.linalg.norm((a[f] - b[f]) * w)) / 30.0


def appearance(a, b):
    """Distance between two pieces: nearest pair of ReID prototypes, else colour."""
    if 'protos' in a and 'protos' in b:
        return float((1 - a['protos'] @ b['protos'].T).min())
    return app_dist(a['app'], b['app'])


def co_distances(a, b):
    ta, tb = np.round(a['t'], 2), np.round(b['t'], 2)
    common, ia, ib = np.intersect1d(ta, tb, return_indices=True)
    return np.linalg.norm(a['xy'][ia] - b['xy'][ib], axis=1)


def distinct_people(a, b, apart=0.8, frames=8, cross_apart=1.4):
    """Two pieces of ONE camera are two people if they were seen at the same time
    clearly apart (overlapping duplicates of one person are not). Across cameras this
    test is NOT used: measured on ground truth, the same person can land 1.4 m apart
    on the two floor maps, so the rule split real people and cost 20 points of
    one-id share when it was tried."""
    if a['cam'] != b['cam']:
        return False
    d = co_distances(a, b)
    return int((d > apart).sum()) >= frames


def pair_state(a, b):
    """(feasible, appearance distance or None, co-located across cameras)."""
    if a['cam'] == b['cam'] and time_overlap(a, b) > 0 and distinct_people(a, b):
        return False, None, False
    if b['t'][0] > a['t'][-1] or a['t'][0] > b['t'][-1]:
        first, second = (a, b) if b['t'][0] > a['t'][-1] else (b, a)
        gap = second['t'][0] - first['t'][-1]
        if np.linalg.norm(second['xy'][0] - first['xy'][-1]) > 1.6 * gap + 1.0:
            return False, None, False
    co = False
    if a['cam'] != b['cam']:
        d = co_distances(a, b)
        co = len(d) >= 12 and float(np.median(d)) < 0.8 and float((d < 1.0).mean()) > 0.7
    ad = None
    if a.get('protos') is not None and b.get('protos') is not None:
        ad = float((1 - a['protos'] @ b['protos'].T).min())
    return True, ad, co


GAP_FREE = float(os.environ.get('RA_GAPFREE', '30'))    # seconds of absence that cost nothing
GAP_PEN = float(os.environ.get('RA_GAPPEN', '0.0075'))  # added to the appearance distance per further minute apart.
#   Measured on four labelled clips (wrong detections, then one-id):
#     c112000 14.0% -> 6.9%   c155236 0.1% -> 0.0%   c183400 0.5% -> 0.4% (held out)
#     c103700 3.3% -> 4.0%, but its one-id rose 0.705 -> 0.736
#   Without it a single identity swallowed seven strangers of that clip.


def gap_between(a, b):
    """Seconds between two pieces, 0 if they overlap in time."""
    if b['t'][0] > a['t'][-1]:
        return b['t'][0] - a['t'][-1]
    if a['t'][0] > b['t'][-1]:
        return a['t'][0] - b['t'][-1]
    return 0.0


def gap_cost(a, b):
    """An old memory is weaker evidence: the longer two pieces are apart, the closer
    their appearance has to be before they may be called the same person. Without this
    one identity swallows every stranger of the day who happens to wear dark clothes."""
    if GAP_PEN <= 0:
        return 0.0
    return GAP_PEN * max(0.0, gap_between(a, b) - GAP_FREE) / 60.0


def cluster_distance(A, B):
    """Average linkage under hard constraints.

    Nearest-pair linkage chains strangers together (one dark jacket at the glass
    resembles the next, and a dozen passers-by become one person); farthest-pair
    linkage lets a single bad crop veto a whole person. The average of all pairs is
    the usual middle ground and behaves well for both."""
    strong = any(a.get('strong') for a in A) and any(b.get('strong') for b in B)
    vals = []
    for a in A:
        for b in B:
            ok, ad, co = pair_state(a, b)
            if not ok:
                return None
            if co:
                vals.append(0.5 * GATE_CROSS)   # same spot at the same instant
                continue
            if ad is None or not strong:
                continue                        # no usable appearance evidence from this pair
            vals.append(ad + gap_cost(a, b))
    if not vals:
        return None
    return float(np.mean(vals))


def identity_distance(piece, members):
    """Distance from one piece to a person built from several pieces: the average of
    its distances to that person's usable (close enough to the camera) pieces."""
    vals = []
    for m in members:
        if m['idx'] == piece['idx']:
            continue
        ok, ad, co = pair_state(piece, m)
        if not ok:
            return None
        if co:
            vals.append(0.5 * GATE_CROSS)
        elif ad is not None and m.get('strong') and piece.get('strong'):
            vals.append(ad + gap_cost(piece, m))
    return float(np.mean(vals)) if vals else None


def refine_identities(groups, rounds=4):
    """Re-assign every piece to the person it resembles most, then rebuild and repeat.

    Clustering commits to merges in a fixed order and cannot undo them; this pass lets
    a fragment that ended up alone -- or with the wrong neighbour -- move to the person
    whose whole set of pieces it matches, as long as physics allows it (no same-camera
    overlap at a distance, reachable on foot). A piece moves only when the best person
    is clearly better than the runner-up."""
    by_idx = {m['idx']: m for mem in groups.values() for m in mem}
    assign = {m['idx']: g for g, mem in groups.items() for m in mem}
    for _ in range(rounds):
        moved = 0
        for pid in list(assign):
            piece = by_idx[pid]
            cur = assign[pid]
            scores = []
            for g in set(assign.values()):
                mem = [by_idx[k] for k, v in assign.items() if v == g]
                d = identity_distance(piece, mem)
                if d is not None and d <= GATE_WITHIN:
                    scores.append((d, g))
            if not scores:
                continue
            scores.sort()
            best_d, best_g = scores[0]
            if best_g == cur:
                continue
            if len(scores) > 1 and best_d > RATIO * scores[1][0]:
                continue
            own = identity_distance(piece, [by_idx[k] for k, v in assign.items() if v == cur])
            if own is not None and own <= best_d:
                continue
            assign[pid] = best_g
            moved += 1
        if not moved:
            break
    out = {}
    for pid, g in assign.items():
        out.setdefault(g, []).append(by_idx[pid])
    return out


def fuse(per_cam, same=0.8, max_gap=10.0, speed=1.5, slack=0.8):
    """Group tracker pieces into people.

    Two stages. First, pieces that the two cameras saw in the same spot at the same
    moment are merged -- geometry is the strongest evidence available and it needs no
    appearance model. Then the rest is clustered by appearance under complete linkage
    with hard physical constraints (one camera cannot show one person in two places,
    nobody walks faster than 1.6 m/s), so a person cut into dozens of fragments can
    still come back together through whichever fragment they resemble most."""
    items = per_cam['cam1'] + per_cam['cam2']
    for n, it in enumerate(items):
        it['idx'] = n
    groups = {i: [it] for i, it in enumerate(items)}

    # stage 1: cross-camera co-location, mutual best with a margin over the runner-up
    cand = {}
    for i in groups:
        for j in groups:
            if j <= i or items[i]['cam'] == items[j]['cam']:
                continue
            ok, ad, co = pair_state(items[i], items[j])
            if not ok or not co:
                continue
            if ad is not None and ad > GATE_CROSS:
                continue
            d = co_distances(items[i], items[j])
            cand[(i, j)] = float(np.median(d)) / same + (ad / GATE_CROSS if ad is not None else 0.5)
    merged_pairs = []
    for (i, j), c in sorted(cand.items(), key=lambda kv: kv[1]):
        others = [v for k, v in cand.items() if (k[0] == i or k[1] == j) and k != (i, j)]
        if others and c > RATIO * min(others):
            continue
        gi = next((g for g, m in groups.items() if any(x['idx'] == i for x in m)), None)
        gj = next((g for g, m in groups.items() if any(x['idx'] == j for x in m)), None)
        if gi is None or gj is None or gi == gj:
            continue
        if cluster_distance(groups[gi], groups[gj]) is None and not all(
                pair_state(a, b)[0] for a in groups[gi] for b in groups[gj]):
            continue
        groups[gi] = groups[gi] + groups[gj]; del groups[gj]
        merged_pairs.append(c)

    # stage 2: appearance clustering, complete linkage
    D = {}
    keys = list(groups)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            i, j = keys[a], keys[b]
            d = cluster_distance(groups[i], groups[j])
            if d is not None and d <= GATE_WITHIN:
                D[(i, j)] = d
    while D:
        (i, j), d = min(D.items(), key=lambda kv: kv[1])
        groups[i] = groups[i] + groups[j]; del groups[j]
        for k in list(D):
            if i in k or j in k:
                del D[k]
        for k in groups:
            if k == i:
                continue
            dd = cluster_distance(groups[min(i, k)], groups[max(i, k)])
            if dd is not None and dd <= GATE_WITHIN:
                D[(min(i, k), max(i, k))] = dd

    if os.environ.get('RA_REFINE') == '1':      # measured worse on c103700; kept for experiments
        groups = refine_identities(groups)

    segs, people = [], []
    for g, mem in groups.items():
        t = np.concatenate([m['t'] for m in mem]); xy = np.concatenate([m['xy'] for m in mem])
        o = np.argsort(t, kind='stable'); t, xy = t[o], xy[o]
        rt = np.round(t, 2); ut = np.unique(rt)
        mxy = np.array([xy[rt == u].mean(0) for u in ut])
        hs = [m['h'] for m in mem if np.isfinite(m['h'])]
        segs.append({'t': ut, 'xy': mxy, 'members': [m['idx'] for m in mem],
                     'cams': sorted({m['cam'] for m in mem}), 'h': float(np.median(hs)) if hs else np.nan})
        people.append({'segments': [len(segs) - 1], 't': ut, 'xy': mxy, 'cams': sorted({m['cam'] for m in mem}),
                       'h': float(np.median(hs)) if hs else np.nan, 'pieces': len(mem)})
    order = np.argsort([p['t'][0] for p in people])
    segs = [segs[i] for i in order]; people = [people[i] for i in order]
    for k, p in enumerate(people):
        p['segments'] = [k]
    return segs, people, merged_pairs


def door_distance(xy):
    """Distance (m) to the door segment itself, not the infinite line."""
    a, b = np.array(DOOR['door_a']), np.array(DOOR['door_b'])
    ab = b - a
    t = np.clip(((xy - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(xy - (a + t[:, None] * ab), axis=1)


def visits(people, hysteresis=0.35, near_door=1.0, min_inside_s=2.0, fps=25.0):
    """Entry/exit per person. Outside the glass people are rarely detected, so a
    track usually BEGINS on the threshold rather than crossing it: an entry is a
    crossing out->in, or a track born within a metre of the door that goes on
    inside; an exit mirrors it (crossing in->out, or a track that ends at the
    door after having been inside)."""
    out = []
    for pid, p in enumerate(people):
        s = door_side(p['xy']); dd = door_distance(p['xy'])
        events, state = [], None
        for t, v in zip(p['t'], s):
            new = 'in' if v > hysteresis else ('out' if v < -hysteresis else state)
            if state is not None and new != state:
                events.append(('entry' if new == 'in' else 'exit', float(t), 'crossing'))
            state = new if new is not None else state
        inside_s = float(np.sum(s > hysteresis)) / fps
        head = slice(0, min(len(s), int(4 * fps)))
        if not any(e[0] == 'entry' for e in events) and dd[0] < near_door and s[head].max() > hysteresis:
            events.insert(0, ('entry', float(p['t'][0]), 'appeared at door'))
        tail = slice(max(0, len(s) - int(2 * fps)), len(s))
        if (not events or events[-1][0] != 'exit') and dd[-1] < near_door and inside_s >= min_inside_s \
                and s[tail].min() < hysteresis:
            events.append(('exit', float(p['t'][-1]), 'vanished at door'))
        out.append({'person': pid, 'first': float(p['t'][0]), 'last': float(p['t'][-1]), 'events': events,
                    'inside_s': inside_s, 'customer_like': inside_s >= min_inside_s, 'cams': p['cams']})
    return out


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/clip/dets25.npz'
    calib = json.load(open('data/calib_final.json'))
    cams = {c: Camera(c, calib) for c in ('cam1', 'cam2')}
    z = dict(np.load(path))
    dets = {c: z[c] for c in ('cam1', 'cam2')}
    feats = {c: z[c + '_feat'] for c in ('cam1', 'cam2')}
    per_cam, stats = build(cams, dets, feats)
    for c, s in stats.items():
        print(c, s)
        hs = [tr['h'] for tr in per_cam[c] if np.isfinite(tr['h'])]
        print('   measured heights (m):', np.round(sorted(hs), 2))
    segs, people, cross = fuse(per_cam)
    print('cross-camera merges:', len(cross))
    print('segments %d -> people %d' % (len(segs), len(people)))
    for v in visits(people):
        print('  visit P%d: %.1f-%.1f s, inside %.1f s, cams %s, events %s' % (v['person'], v['first'], v['last'], v['inside_s'], '+'.join(v['cams']), v['events']))
    for k, p in enumerate(sorted(people, key=lambda p: -(p['t'][-1] - p['t'][0]))[:12]):
        steps = np.linalg.norm(np.diff(p['xy'], axis=0), axis=1)
        print('  person %2d: %5.1f s  %s  pieces %d  height %s  path %.1f m  jitter p90 %.3f m/frame'
              % (k, p['t'][-1] - p['t'][0], '+'.join(p['cams']), len(p['segments']),
                 '%.2f' % p['h'] if np.isfinite(p['h']) else '  - ', steps.sum(), np.percentile(steps, 90)))
