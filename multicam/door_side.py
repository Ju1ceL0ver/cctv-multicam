"""Who is inside and who outside, and when it changes: /doorside (07.10.2026).

The owner marks, on a door stretch of camera 1 (data/sam31_door, the light copy of /doormark), each person's side
from a frame on: click a person (SAM 3.1's outlines, numbered as its tracks), 1 = inside from here, 2 = outside from
here. outside -> inside is an entry, inside -> outside an exit. A person SAM did not outline: click the empty spot,
the point becomes a person of its own. The marks are the ground truth both for entries/exits and for the per-frame
inside/outside model.

-> data/door_side/<day>.json {tag: {'sides': {person: [[t, 'in'|'out'], ...]}, 'points': {id: [x, y, t0]}, 'done': bool}}
   (t = seconds from the stretch start, person = SAM's track number or 'x<n>' for a point; x, y in 1280 x 720);
   every change also in <day>.history.jsonl."""
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'door_side'
DOOR = ROOT / 'data' / 'sam31_door'
FPS = 12.5


def load(day):
    p = OUT / ('%s.json' % day)
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def save(day, state, change):
    from storage import atomic_json
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / ('%s.json' % day), state)
    with open(OUT / ('%s.history.jsonl' % day), 'a', encoding='utf-8') as f:
        f.write(json.dumps(dict(change, at=time.strftime('%Y-%m-%dT%H:%M:%S')), ensure_ascii=False) + '\n')


def outlines(tag):
    """{k: [[piece, person, [[x, y], ...]], ...]} -- SAM 3.1's tracks (pieces linked over the session seams, one
    person each by construction) on every tick, outlines at 1280 x 720; person = the old ReID group (for moving
    marks made before 07.10 20:15). The poster stand is left out."""
    import cv2
    f = OUT / 'outlines' / (tag + '_v2.json.gz')
    if f.exists():
        return json.load(gzip.open(f, 'rt'))
    import door_sam as DS
    import sam31_reid as R
    base = DOOR / tag / 'cam1'
    info = json.load(open(base / 'info.json'))
    rep = json.load(open(base / 'report.json'))
    M = R.Masks(base / 'chunks.npz')
    owned, _ = R.link_seams(M, {int(s_): int(sh) for s_, e_, sh in info['sessions']})
    person = {int(p): v for p, v in rep['person_of_piece'].items()}
    static = DS.static_people(M, owned, person, info['ticks'])
    sx, sy = 1280 / 2176.0, 720 / 1224.0
    out = {}
    for piece, rs in owned.items():
        pid = int(person.get(int(piece)) or 0)
        if pid in static:
            continue
        for r in rs:
            k = int(M.rows[r, 1])
            x1, y1 = int(M.rows[r, 4]), int(M.rows[r, 5])
            cs, _ = cv2.findContours(M.crop(r).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            cnt = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 2.0, True)[:, 0, :]
            out.setdefault(str(k), []).append([int(piece), pid, [[round((x1 + float(x)) * sx, 1), round((y1 + float(y)) * sy, 1)]
                                                                 for x, y in cnt]])
    f.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, gzip.open(f, 'wt'))
    return out


JUMP_PX = 64.0                        # 1280-px of one tick (0.08 s): a person never steps that far; a track that does
                                      # has jumped to someone else (SAM's number handed over in a crowd)


def auto_cuts(tag):
    """09.10: {piece: [t, ...]} -- where a SAM track's centre jumps further than a person can walk in the time between
    its ticks (JUMP_PX per tick): another person from there on. Computed from the outlines (1280 x 720), cached."""
    f = OUT / 'outlines' / (tag + '_autocuts_v2.json')
    if f.exists():
        return json.load(open(f))
    by = {}
    for k, xs in outlines(tag).items():
        for piece, _, poly in xs:
            P = np.asarray(poly, float)
            if len(P):
                by.setdefault(str(piece), []).append((int(k), P[:, 0].mean(), P[:, 1].min(), P[:, 1].max()))
    out = {}
    W_ = 5                                              # ticks on each side: the jump must stay, not flicker
    for piece, xs in by.items():
        xs.sort()
        P = np.array([(x, t) for _, x, t, _ in xs])
        K = [k for k, _, _, _ in xs]
        last = -1e9
        for i in range(W_, len(xs) - W_ + 1):
            n = max(1, K[i] - K[i - 1])
            if n > 12:                                  # a gap of a second: SAM lost and found it, not a jump
                continue
            if np.hypot(*(P[i] - P[i - 1])) <= JUMP_PX * n:
                continue
            before, after = np.median(P[i - W_:i], 0), np.median(P[i:i + W_], 0)
            if np.hypot(*(after - before)) > JUMP_PX * n and K[i] / FPS - last > 1.5:
                out.setdefault(piece, []).append(round(K[i] / FPS, 2))
                last = K[i] / FPS
    f.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f, 'w'))
    return out


def _cut(s, piece, t):
    cuts = s.setdefault('cuts', {})
    cuts[piece] = sorted(set(cuts.get(piece, []) + [t]))
    for p in [p for p in list(s['sides']) if str(p).split('.')[0] == piece]:   # marks follow their part
        for m in s['sides'].pop(p):
            s['sides'].setdefault(eff(cuts, piece, m[0]), []).append(m)
    for p in list(s['sides']):
        s['sides'][p] = sorted(s['sides'][p])


def migrate(tag, s):
    """Marks made on ReID numbers (before 07.10 20:15) -> the SAM track that number covered at that moment."""
    if s.get('ids') == 'piece':
        return s
    ol = outlines(tag)
    new = {}
    for p, xs in s.get('sides', {}).items():
        if str(p).startswith('x'):
            new[p] = xs
            continue
        for t, side in xs:
            k = int(round(t * FPS))
            hit = None
            for dk in range(0, 25):
                for kk in (k - dk, k + dk):
                    for piece, person, _ in ol.get(str(kk), []):
                        if str(person) == str(p):
                            hit = str(piece)
                            break
                    if hit:
                        break
                if hit:
                    break
            new.setdefault(hit or 'old%s' % p, []).append([t, side])
    s['sides'] = {k: sorted(v) for k, v in new.items()}
    s['ids'] = 'piece'
    s.setdefault('merge', {})
    return s


def eff(cuts, piece, t):
    """A SAM track the owner cut (two people under one number): the part after the n-th cut is '<piece>.<n>'."""
    n = sum(1 for c in (cuts or {}).get(str(piece), []) if t >= c - 1e-6)
    return str(piece) if n == 0 else '%s.%d' % (piece, n)


def root_of(merge, p):
    seen = set()
    while str(p) in merge and p not in seen:
        seen.add(p)
        p = merge[str(p)]
    return str(p)


def events(sides, merge=None, noperson=()):
    """{person: [[t, side]]} -> [{person, t, kind}] (outside -> inside = in, inside -> outside = out); pieces
    merged by the owner (legs and torso of one person, a track broken behind a stand) are one timeline."""
    merge = merge or {}
    joined = {}
    for p, xs in sides.items():
        r = root_of(merge, p)
        if r in noperson:
            continue
        joined.setdefault(r, []).extend(xs)
    ev = []
    for p, xs in joined.items():
        prev = None
        for t, s in sorted(xs):
            if prev is not None and s != prev:
                ev.append({'person': p, 't': t, 'kind': 'in' if s == 'in' else 'out'})
            prev = s
    return sorted(ev, key=lambda e: e['t'])


def change(day, body):
    from storage import file_lock
    OUT.mkdir(parents=True, exist_ok=True)
    with file_lock(str(OUT / ('%s.json' % day)) + '.lock'):
        st = load(day)
        tag = str(body.get('tag', ''))
        if not re.fullmatch(r'door_\d{8}_\d{5}', tag):
            raise ValueError('bad stretch')
        s = migrate(tag, st.setdefault(tag, {'sides': {}, 'points': {}, 'done': False}))
        act = body.get('act')
        if act == 'side':
            p, t, side = str(body['person']), round(float(body['t']), 2), body['side']
            if side not in ('in', 'out'):
                raise ValueError('side is in or out')
            xs = [x for x in s['sides'].get(p, []) if abs(x[0] - t) > 0.04]
            s['sides'][p] = sorted(xs + [[t, side]])
        elif act == 'unside':
            p, t = str(body['person']), float(body['t'])
            xs = s['sides'].get(p, [])
            if xs:
                near = min(xs, key=lambda x: abs(x[0] - t))
                xs.remove(near)
                if xs:
                    s['sides'][p] = xs
                else:
                    s['sides'].pop(p, None)
        elif act == 'point':
            n = 1 + max([int(k[1:]) for k in s['points']] + [0])
            pid = 'x%d' % n
            s['points'][pid] = [round(float(body['x']), 1), round(float(body['y']), 1), round(float(body['t']), 2)]
            body = dict(body, person=pid)
        elif act == 'unpoint':
            pid = str(body['person'])
            s['points'].pop(pid, None)
            s['sides'].pop(pid, None)
        elif act == 'cut':                         # body: person (the SAM track), t -- another person from t on
            _cut(s, str(body['person']).split('.')[0], round(float(body['t']), 2))
        elif act == 'autocut':                     # 09.10: NOT used by the page -- checked by eye on 6 cuts of 18.09: most
                                                   # were SAM's piece hopping over one person's body (leg -> head) or a
                                                   # person behind the round sticker, not another person
            if not s.get('done') and not s.get('auto_applied'):
                n = 0
                for piece, ts in auto_cuts(tag).items():
                    for t in ts:
                        _cut(s, piece, t)
                        n += 1
                s['auto_applied'] = n
                s['auto_cuts'] = auto_cuts(tag)
        elif act == 'staff':                       # 09.10: this person is staff (toggle), the whole person over merges
            r = root_of(s.setdefault('merge', {}), body['person'])
            sl = s.setdefault('staff', [])
            if r in sl:
                sl.remove(r)
            else:
                sl.append(r)
        elif act == 'uncut':
            piece = str(body['person']).split('.')[0]
            s.setdefault('cuts', {}).pop(piece, None)
            for p in [p for p in list(s['sides']) if str(p).split('.')[0] == piece and p != piece]:
                s['sides'].setdefault(piece, []).extend(s['sides'].pop(p))
            if piece in s['sides']:
                s['sides'][piece] = sorted(s['sides'][piece])
        elif act == 'merge':                       # body: person (kept), other (joins it)
            a_, b_ = root_of(s['merge'], body['person']), root_of(s['merge'], body['other'])
            if a_ != b_:
                s['merge'][b_] = a_
        elif act == 'unmerge':
            p = str(body['person'])
            s['merge'].pop(p, None)
            for k_ in [k_ for k_, v_ in s['merge'].items() if v_ == p]:
                s['merge'].pop(k_)
        elif act == 'noperson':                    # the owner: this track is not a person (toggle)
            r = root_of(s.setdefault('merge', {}), body['person'])
            npl = s.setdefault('noperson', [])
            if r in npl:
                npl.remove(r)
            else:
                npl.append(r)
        elif act == 'role':                        # 'train' (frames for the inside/outside model) or 'test'
            if body.get('role') not in ('train', 'test'):
                raise ValueError('role is train or test')
            s['role'] = body['role']
        elif act == 'done':
            s['done'] = bool(body.get('done', True))
        else:
            raise ValueError('unknown action')
        save(day, st, dict(body, day=day))
        return {'stretch': s, 'events': events(s['sides'], s.get('merge'), s.get('noperson', [])), 'person': body.get('person')}


def register(app, root=None):
    from flask import abort, jsonify, render_template, request, send_file
    import door_mark as DM

    def valid(day):
        if not re.fullmatch(r'\d{8}', day):
            abort(400)

    @app.get('/doorside')
    def doorside_page():
        return render_template('doorside.html')

    @app.get('/api/doorside/<day>')
    def doorside_day(day):
        valid(day)
        st = load(day)
        out = []
        for x in DM.stretches(day):
            s = st.get(x['tag'])
            if s is None:
                s = {'sides': {}, 'points': {}, 'done': False, 'ids': 'piece', 'merge': {}}
            elif s.get('ids') != 'piece':
                s = migrate(x['tag'], s)
            out.append(dict(x, sides=s['sides'], points=s['points'], done=s['done'], merge=s.get('merge', {}),
                            noperson=s.get('noperson', []), role=s.get('role', 'test'), cuts=s.get('cuts', {}),
                            staff=s.get('staff', []), auto_applied=s.get('auto_applied'), auto_cuts=s.get('auto_cuts', {}),
                            events=events(s['sides'], s.get('merge'), s.get('noperson', []))))
        return jsonify({'day': day, 'stretches': out, 'fps': FPS})

    @app.get('/api/doorside/outlines/<tag>')
    def doorside_outlines(tag):
        if not re.fullmatch(r'door_\d{8}_\d{5}', tag):
            abort(400)
        return jsonify(outlines(tag))

    @app.get('/api/doorside/video/<tag>')
    def doorside_video(tag):
        if not re.fullmatch(r'door_\d{8}_\d{5}', tag):
            abort(400)
        return send_file(str(DM.video(tag)), mimetype='video/mp4', conditional=True)

    @app.get('/api/doorside/line')
    def doorside_line_get():
        import door_line
        return jsonify(door_line.load() or {})

    @app.post('/api/doorside/line')
    def doorside_line_set():
        import door_line
        from storage import atomic_json
        b = request.get_json(force=True) or {}
        try:
            line = {k: [float(b[k][0]), float(b[k][1])] for k in ('p1', 'p2', 'inside')}
        except (KeyError, TypeError, ValueError, IndexError):
            return jsonify({'error': 'p1, p2, inside'}), 400
        line['min_px'] = int(b.get('min_px', 20))
        line['at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        atomic_json(door_line.FILE, line)
        return jsonify(line)

    @app.post('/api/doorside/<day>')
    def doorside_change(day):
        valid(day)
        try:
            return jsonify(change(day, request.get_json(force=True) or {}))
        except (ValueError, KeyError) as exc:
            return jsonify({'error': str(exc)}), 400
