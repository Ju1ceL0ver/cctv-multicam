"""Whole visits from the night passes' track pieces: the teacher of tracking, offline, the whole day at once.

The night tracker cuts a customer into pieces (median 2 per person of a minute or longer, p90 8 -- AGENTS
section 18) and joins some pieces of different people (15 % of its groups). Offline we see the whole day,
forward and backward, so pieces are joined into people here, globally:

  nodes     the day's shop tracks (day_movie.index: film clock, both cameras, passers-by left out)
  looks     per track the mean appearance vector: OSNet per detection (in the night files), or the heavy
            teachers' (TransReID clothes + CSCI body shape, track_emb.py) when they are there
  edges     same camera, one after the other within GAP_MAX s, the walk between them possible
            (floor metres, WALK m/s); the two cameras at the same time, the floor places agreeing
  cost      appearance distance + a price per minute apart (beyond GAP_FREE) + distance on the floor
  never     one person on one camera twice at the same time (cannot-link)
  joining   cheapest edge first, while the cost is below TAU and no never-rule breaks (greedy
            correlation clustering)

Measured against the owner's identity labels of 17.09 (six clips he labelled: 97 people), transferred to
the day's tracks by detection (same camera, same moment, box IoU >= 0.5): for every labelled person the
number of machine people he is cut into, and for every machine person the labelled people mixed in.
TAU and the gap price are chosen on three clips and checked on the other three.

usage: stitch.py DAY [--looks osnet|teachers]   -> data/stitch/DAY.json (+ the score if the day has labels)"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
GAP_MAX = 120.0          # s: pieces further apart are not joined here (a return visit is another question)
GAP_FREE = 30.0          # s apart that cost nothing
WALK = 2.0               # m/s: faster than this between two pieces is not one person
XCAM_OVERLAP = 1.0       # s both cameras must see the two pieces together
XCAM_FLOOR = 1.5         # m: the two cameras' floor places may disagree this much
OWNER_CLIPS = ('c100054', 'c103700', 'c112000', 'c155236', 'c171400', 'c183400')
TUNE, CHECK = OWNER_CLIPS[:3], OWNER_CLIPS[3:]
TAG = 'yolo26x-seg'
OSNET = 'emb_osnet_ain_x1_0_msmt17.npz'


def _film_times(path, meta, cam, rows_t, day, root, segs, offset, start):
    import day_player
    landed = day_player.start_frames(path, meta, segs, day)
    began = day_player._epoch(meta['start'])
    frames = np.rint(rows_t * day_player.FPS).astype(np.int64)
    return day_player.walls(segs[cam], began, frames, landed[cam]) - (offset if cam == 'cam2' else 0.0) - start


def tracks(day, root=ROOT, looks='osnet'):
    """The day's shop tracks with what joining needs."""
    import day_movie
    idx = day_movie.index(day, root)
    clips = {}
    out = []
    emb_t = None
    if looks == 'teachers':
        p = Path(root) / 'data' / 'stitch' / ('%s_teachers.npz' % day)
        if p.exists():
            z = np.load(p, allow_pickle=True)
            emb_t = {k: (c, s) for k, c, s in zip(z['keys'], z['cloth'], z['shape'])}
    for name, tr in idx['tracks'].items():
        if not tr['shop'] or len(tr['rows']) < 3:
            continue
        clip, cam = tr['clip'], tr['cam']
        if clip not in clips:
            base = Path(root) / 'data' / 'raw_clips' / clip
            with np.load(base / ('dets_%s.npz' % TAG)) as a:
                dets = {c: a[c] for c in ('cam1', 'cam2')}
            emb = None
            if (base / OSNET).exists():
                with np.load(base / OSNET) as a:
                    emb = {c: a[c] for c in ('cam1', 'cam2')}
            pieces = {}
            pf = base / ('pieces_%s.json' % TAG)
            if pf.exists():
                for p in json.load(open(pf)):
                    pieces['%s:%d' % (clip, p['piece'])] = p
            clips[clip] = (dets, emb, pieces)
        dets, emb, pieces = clips[clip]
        rows = np.asarray(tr['rows'])
        v = None
        if emb_t is not None and name in emb_t:
            c, s = emb_t[name]
            v = np.concatenate([c / (np.linalg.norm(c) + 1e-9), s / (np.linalg.norm(s) + 1e-9)]) / np.sqrt(2)
        elif emb is not None:
            e = emb[cam][rows[:: max(1, len(rows) // 12)]]
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
            v = e.mean(0); v = v / (np.linalg.norm(v) + 1e-9)
        piece = pieces.get(name)
        out.append({'key': name, 'cam': cam, 'clip': clip, 'first': float(tr['first']), 'last': float(tr['last']),
                    'look': v, 'xy0': piece['xy0'] if piece else None, 'xy1': piece['xy1'] if piece else None,
                    'foot0': dets[cam][rows[0], 6:8].tolist(), 'foot1': dets[cam][rows[-1], 6:8].tolist(),
                    'n': int(len(rows))})
    return out, idx


def edges(T, tau_look=1.0):
    """Candidate joins: (cost parts, i, j)."""
    out = []
    order = sorted(range(len(T)), key=lambda i: T[i]['first'])
    firsts = np.array([T[i]['first'] for i in order])
    for a in range(len(T)):
        A = T[a]
        if A['look'] is None:
            continue
        # everything that starts from A's start to GAP_MAX after its end
        lo = np.searchsorted(firsts, A['first'] - 0.0, 'left')
        hi = np.searchsorted(firsts, A['last'] + GAP_MAX, 'right')
        for k in range(lo, hi):
            b = order[k]
            if b == a:
                continue
            B = T[b]
            if B['look'] is None:
                continue
            look = 1.0 - float(A['look'] @ B['look'])
            if look > tau_look:
                continue
            if B['cam'] == A['cam']:
                gap = B['first'] - A['last']
                if gap < -0.5:
                    continue                                      # overlapping on one camera: two people
                walk = None
                if A['xy1'] and B['xy0']:
                    walk = float(np.hypot(A['xy1'][0] - B['xy0'][0], A['xy1'][1] - B['xy0'][1]))
                    if walk > 0.5 + WALK * max(0.0, gap):
                        continue
                out.append((look, max(0.0, gap), walk or 0.0, a, b, 'seq'))
            else:
                ov = min(A['last'], B['last']) - max(A['first'], B['first'])
                if ov < XCAM_OVERLAP or A['first'] > B['first']:
                    continue                                      # each pair once, and only when both are on screen together
                if A['xy0'] and B['xy0']:
                    fa = np.array([A['xy0'], A['xy1']]).mean(0); fb = np.array([B['xy0'], B['xy1']]).mean(0)
                    if np.hypot(*(fa - fb)) > XCAM_FLOOR * 2:
                        continue
                out.append((look, 0.0, 0.0, a, b, 'xcam'))
    return out


def join(T, E, tau=0.25, gap_price=0.0075, walk_price=0.02, xcam_bonus=0.0, prior=()):
    """Greedy correlation clustering with cannot-link: the night passes' own links first (`prior`: pairs of
    track indices -- their grouping inside a window and the window seams), then the cheapest edge first."""
    cost = [(-1.0, a, b) for a, b in prior]
    cost += sorted((l + gap_price * max(0.0, g - GAP_FREE) / 60.0 + walk_price * w - (xcam_bonus if kind == 'xcam' else 0.0), a, b)
                   for l, g, w, a, b, kind in E)
    parent = list(range(len(T)))
    spans = {i: {T[i]['cam']: [(T[i]['first'], T[i]['last'])]} for i in range(len(T))}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def clash(sa, sb):
        for cam, xs in sa.items():
            for (f1, l1) in xs:
                for (f2, l2) in sb.get(cam, []):
                    if min(l1, l2) - max(f1, f2) > 1.0:
                        return True
        return False
    for c, a, b in cost:
        if c > tau:
            break
        ra, rb = find(a), find(b)
        if ra == rb or clash(spans[ra], spans[rb]):
            continue
        parent[rb] = ra
        for cam, xs in spans.pop(rb).items():
            spans[ra].setdefault(cam, []).extend(xs)
    return [find(i) for i in range(len(T))]


def owner_truth(day, T, root=ROOT):
    """{track index: (clip, owner label)} from the owner's clip labels, by detection: same camera, the same
    film moment within one grid step (0.13 s: the two passes sampled frames with different phases), box IoU >= 0.4."""
    import day_movie
    import day_player
    from review_store import read_state
    idx = day_movie.index(day, root)
    segs = day_player.segments(day, root)
    offset = day_movie.offset_of(day, root)
    start, _ = day_movie.clock(day, root)
    by_key = {t['key']: i for i, t in enumerate(T)}
    # the day's detections of the tracks, by camera: time, box, track index
    rows = {'cam1': [], 'cam2': []}
    cache = {}
    for name, tr in idx['tracks'].items():
        if name not in by_key:
            continue
        clip = tr['clip']
        if clip not in cache:
            with np.load(Path(root) / 'data' / 'raw_clips' / clip / ('dets_%s.npz' % TAG)) as a:
                cache[clip] = {c: a[c] for c in ('cam1', 'cam2')}
        d = cache[clip][tr['cam']]
        for t, r in zip(tr['times'], tr['rows']):
            rows[tr['cam']].append((t, *d[r, 1:5], by_key[name]))
    rows = {c: np.array(v) for c, v in rows.items()}
    truth = {}
    for clip in OWNER_CLIPS:
        path = Path(root) / 'data' / 'raw_clips' / clip
        if not path.exists():
            continue
        st = read_state(path)
        meta = json.load(open(path / ('meta_%s.json' % TAG)))
        with np.load(path / ('dets_%s.npz' % TAG)) as a:
            dets = {c: a[c] for c in ('cam1', 'cam2')}
        votes = {}
        for p in st['pieces']:
            lab = st['labels'].get(str(p['piece']))
            if not lab or lab == '?':
                continue
            cam = p['cam']
            R = rows[cam]
            if not len(R):
                continue
            rr = np.asarray(p['dets'])
            tt = _film_times(path, meta, cam, dets[cam][rr, 0], day, root, segs, offset, start)
            order = np.argsort(R[:, 0])
            Rt = R[order, 0]
            for r, t in zip(rr, tt):
                if np.isnan(t):
                    continue
                k0, k1 = np.searchsorted(Rt, t - 0.13), np.searchsorted(Rt, t + 0.13)   # grids sampled 0.12 s apart, shifted
                b = dets[cam][r, 1:5]
                for k in order[k0:k1]:
                    q = R[k, 1:5]
                    ix = max(0.0, min(b[2], q[2]) - max(b[0], q[0])); iy = max(0.0, min(b[3], q[3]) - max(b[1], q[1]))
                    inter = ix * iy
                    iou = inter / ((b[2] - b[0]) * (b[3] - b[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter + 1e-9)
                    if iou >= 0.4:
                        votes.setdefault(int(R[k, 5]), {}).setdefault(lab, 0)
                        votes[int(R[k, 5])][lab] += 1
        for ti, v in votes.items():
            lab, n = max(v.items(), key=lambda kv: kv[1])
            if n >= 2 and n >= 0.6 * sum(v.values()):
                truth[ti] = (clip, lab)
    return truth


def score(T, people, truth, clips=OWNER_CLIPS, long_s=60.0):
    """For the owner's people of these clips: machine people per owner person, owner people per machine person."""
    gt = {}
    for ti, (clip, lab) in truth.items():
        if clip in clips:
            gt.setdefault((clip, lab), []).append(ti)
    per_person, long_counts = [], []
    for key, tis in gt.items():
        n = len({people[t] for t in tis})
        per_person.append(n)
        dur = max(T[t]['last'] for t in tis) - min(T[t]['first'] for t in tis)
        if dur >= long_s:
            long_counts.append(n)
    mixed = {}
    for ti, (clip, lab) in truth.items():
        if clip in clips:
            mixed.setdefault(people[ti], set()).add((clip, lab))
    groups = list(mixed.values())
    return {'owner_people': len(gt), 'cut_into_more_than_one': round(float(np.mean([n > 1 for n in per_person])), 3) if per_person else None,
            'long_people': len(long_counts),
            'long_pieces_median': float(np.median(long_counts)) if long_counts else None,
            'long_pieces_p90': float(np.percentile(long_counts, 90)) if long_counts else None,
            'long_whole': round(float(np.mean([n == 1 for n in long_counts])), 3) if long_counts else None,
            'machine_people': len(groups), 'mixing_two_or_more': round(float(np.mean([len(g) > 1 for g in groups])), 3) if groups else None}


def machine_people(day, T, root=ROOT):
    """The night passes' own people (their groups and window seams) for the same tracks: the baseline."""
    import day_movie
    crowd = day_movie.people(day, root)
    part_person = {}
    for pid, part in crowd['parts'].items():
        part_person.setdefault(part['track'], part['person'])
    return [part_person.get(t['key'], 'solo:%d' % i) for i, t in enumerate(T)]


def run(day, looks='osnet', root=ROOT, log=print):
    t0 = time.time()
    T, idx = tracks(day, root, looks)
    E = edges(T)
    at = {t['key']: i for i, t in enumerate(T)}
    prior = [(at[a], at[b]) for _kind, a, b in idx['links'] if a in at and b in at]
    log('%s: %d shop tracks, %d candidate joins, %.0f s' % (day, len(T), len(E), time.time() - t0))
    rep = {'day': day, 'looks': looks, 'tracks': len(T), 'edges': len(E)}
    truth = owner_truth(day, T, root) if day == '20260917' else {}
    if truth:
        base = machine_people(day, T, root)
        rep['baseline'] = {'tune': score(T, base, truth, TUNE), 'check': score(T, base, truth, CHECK)}
        grid = []
        for tau in (-0.5, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6):     # -0.5: the night passes' links alone
            for gp in (0.0, 0.0075, 0.02, 0.05):
                ppl = join(T, E, tau, gp, prior=prior)
                s = score(T, ppl, truth, TUNE)
                # one number: whole long visits, minus mixing
                grid.append(((s['long_whole'] or 0) - 2 * (s['mixing_two_or_more'] or 0) - 0.01 * (s['long_pieces_median'] or 0), tau, gp, s))
        grid.sort(key=lambda g: -g[0])
        _, tau, gp, s_tune = grid[0]
        ppl = join(T, E, tau, gp, prior=prior)
        rep['links_only'] = {'tune': score(T, join(T, E, -0.5, 0, prior=prior), truth, TUNE),
                             'check': score(T, join(T, E, -0.5, 0, prior=prior), truth, CHECK)}
        rep['chosen'] = {'tau': tau, 'gap_price': gp, 'tune': s_tune, 'check': score(T, ppl, truth, CHECK)}
        rep['grid_top'] = [{'tau': g[1], 'gap_price': g[2], 'tune': g[3]} for g in grid[:5]]
        rep['truth_tracks'] = len(truth)
    else:
        pp = Path(root) / 'data' / 'stitch' / ('20260917%s.json' % ('' if looks == 'osnet' else '_' + looks))   # the labelled day's settings, same looks
        prev = json.load(open(pp)) if pp.exists() else {}
        tau, gp = prev.get('chosen', {}).get('tau', 0.24), prev.get('chosen', {}).get('gap_price', 0.0075)
        ppl = join(T, E, tau, gp, prior=prior)
        rep['chosen'] = {'tau': tau, 'gap_price': gp, 'from': '20260917'}
    persons = {}
    for i, p in enumerate(ppl):
        persons.setdefault(p, []).append(T[i]['key'])
    rep['people'] = len(persons)
    rep['seconds'] = round(time.time() - t0)
    out = Path(root) / 'data' / 'stitch'
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out / ('%s%s.json' % (day, '' if looks == 'osnet' else '_' + looks)), 'w'), indent=1)
    json.dump({str(k): v for k, v in persons.items()}, open(out / ('%s%s_people.json' % (day, '' if looks == 'osnet' else '_' + looks)), 'w'))
    log(json.dumps({k: v for k, v in rep.items() if k != 'grid_top'}, indent=1))
    return rep


if __name__ == '__main__':
    day = sys.argv[1]
    looks = sys.argv[sys.argv.index('--looks') + 1] if '--looks' in sys.argv else 'osnet'
    run(day, looks)
