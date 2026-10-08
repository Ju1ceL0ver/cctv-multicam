"""The night teacher over the live door's clips (08.10.2026, the owner's plan: the day saves clips, the night labels them
with the strong model, the morning shows only where it disagrees).

For every 12.5 fps clip of the day (data/live/clips/<day>/, door_clips.py: events +-10 s, disputes, random minutes):
  1. SAM 3.1 itself (sam31_segment.build, the advertising stand suppressed as live) labels the clip -- its tracks
     (pieces linked over its 240-frame sessions) -> data/live/night/<day>/<clip>/cam1/chunks.npz
  2. per row, every 3rd frame (4.2 a second: the stride the door logic was tuned at): the shop line's depth, the mask's
     top/height, the owner's model p_inside, the feet -> the side vote of side_final.json (but conf/birth for stride 3)
     -> stitched tracks -> flips, gate, births, alternation: the teacher's entries/exits
  3. against the live system's events in the clip's span (same direction within TOL s): agree / live only / teacher only
-> data/live/night/<day>.json {clips: {name: {...}}, events: [{t, kind, by: live|teacher|both, live_key}], summary}
/liveevents shows it per event and lists the teacher-only crossings (what the live system missed).

usage (venv_sam3, the card): door_night.py [DAY] [--at HH:MM] [--until HH:MM]"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inout_lab'))
LIVE = ROOT / 'data' / 'live'
NIGHT = LIVE / 'night'
W, H, SAM_IN = 2176, 1224, 1008
TOL = 4.0
STRIDE = 3                       # rows judged every 3rd frame of 12.5 fps


def log(text):
    line = '%s %s' % (time.strftime('%m-%d %H:%M:%S'), text)
    NIGHT.mkdir(parents=True, exist_ok=True)
    with open(NIGHT / 'night.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode(), flush=True)


def clips_of(day):
    out = []
    for j in sorted((LIVE / 'clips' / day).glob('*.json')):
        c = json.load(open(j))
        if j.with_suffix('.mp4').exists() and (c.get('fps') or 0) >= 10:
            out.append((j.stem, c))
    return out


def prepare(name, c, day):
    import cv2
    out = NIGHT / day / name / 'cam1'
    if (out / 'chunks.npz').exists():
        return out, None
    shutil.rmtree(out, ignore_errors=True)
    (out / 'sam_in').mkdir(parents=True)
    cap = cv2.VideoCapture(str(LIVE / 'clips' / day / (name + '.mp4')))
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out / 'sam_in' / ('%05d.jpg' % k)), cv2.resize(f, (SAM_IN, SAM_IN), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 93])
        k += 1
    cap.release()
    import door_micro as DM
    sess = DM.sessions_of(k, 240, 8)
    json.dump({'day': day, 'cam': 'cam1', 'size': [W, H], 'tick': 0.08, 'ticks': k, 'sessions': [list(s) for s in sess],
               'clip': name, 'times': c['times'][:k]}, open(out / 'info.json', 'w'))
    return out, sess


def label_all(day, until):
    import torch
    import door_micro as DM
    import sam31_segment as SG
    todo = []
    for name, c in clips_of(day):
        out, sess = prepare(name, c, day)
        if sess is not None:
            todo.append((name, out, sess))
    log('%s: %d clips to label' % (day, len(todo)))
    if not todo:
        return
    pred = DM.no_poster(SG.build())
    for i, (name, out, sess) in enumerate(todo):
        if until and time.strftime('%H:%M') >= until and time.strftime('%H:%M') < '21:00':
            log('morning: stopped before %s' % name)
            break
        t0 = time.time()
        SG.label(pred, out, sess, None)
        shutil.rmtree(out / 'sam_in', ignore_errors=True)
        torch.cuda.empty_cache()
        log('teacher %d/%d %s: %.0f s' % (i + 1, len(todo), name, time.time() - t0))


def teacher_events(day, side):
    """The door logic on the teacher's tracks of every labelled clip."""
    import cv2
    import door_combo as C
    import door_line
    import door_sam as DS
    import sam31_reid as R
    from binary_door import BinaryDoorClassifier
    line = door_line.load()
    clf = BinaryDoorClassifier()
    f_ = json.load(open(ROOT / 'data' / 'door_v2' / 'combo_final.json'))
    u, hz = np.array(f_['u']), np.array(f_['zone'])
    sd = dict(side, conf=side.get('night_conf', 2))
    out = {}
    for d in sorted((NIGHT / day).iterdir()) if (NIGHT / day).exists() else []:
        base = d / 'cam1'
        if not (base / 'chunks.npz').exists():
            continue
        info = json.load(open(base / 'info.json'))
        times = info['times']
        M = R.Masks(base / 'chunks.npz')
        owned, _ = R.link_seams(M, {int(s): int(sh) for s, e, sh in info['sessions']})
        person = {int(p): 1000 + int(p) for p in owned}
        static = DS.static_people(M, owned, person, info['ticks'])
        cap = cv2.VideoCapture(str(LIVE / 'clips' / day / (d.name + '.mp4')))
        frames, k = {}, 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if k % STRIDE == 0:
                frames[k] = f[:, :, ::-1].copy()
            k += 1
        cap.release()
        tr = {}
        sx, sy = 1280.0 / W, 720.0 / H
        X, keys = [], []
        rows = []
        for p, rs in owned.items():
            if person[int(p)] in static:
                continue
            for r in rs:
                k = int(M.rows[r, 1])
                if k % STRIDE or k not in frames:
                    continue
                c_ = M.crop(r)
                x1, y1 = int(M.rows[r, 4]), int(M.rows[r, 5])
                cs = cv2.resize(c_.astype(np.uint8), (max(1, int(round(c_.shape[1] * sx))), max(1, int(round(c_.shape[0] * sy)))),
                                interpolation=cv2.INTER_NEAREST)
                ms = np.zeros((720, 1280), np.uint8)
                X1, Y1 = int(x1 * sx), int(y1 * sy)
                ms[Y1:Y1 + cs.shape[0], X1:X1 + cs.shape[1]] = cs[:720 - Y1, :1280 - X1]
                if not ms.any():
                    continue
                f = DS.foot_of(c_)
                fx, fy = (x1 + f[0], y1 + f[1]) if f else ((M.rows[r, 4] + M.rows[r, 6]) / 2, M.rows[r, 7])
                d_ = door_line.depth(line, ms > 0)
                top, hh = door_line.extent(ms > 0)
                t = times[k] if k < len(times) else times[-1] + (k - len(times) + 1) * 0.08
                rows.append([t, person[int(p)], d_, top, hh, door_line.bottom_x(ms > 0), None, [fx / W, fy / H]])
                need = sd['mode'] != 'line' and not (d_ is not None and d_ >= sd['h_in'] and sd['mode'] in
                                                     ('either_in+both_out', 'line_in+both_out', 'line_in+model_out', 'model+line_in'))
                if need:
                    try:
                        X.append(clf.features_rgb(frames[k], ms)); keys.append(len(rows) - 1)
                    except ValueError:
                        pass
        if X:
            for i, pv in zip(keys, clf.p_inside_batch(X)):
                rows[i][6] = float(pv)
        for t, pid, d_, top, hh, bx, pv, xy in rows:
            tr.setdefault(pid, []).append((t, door_line.side_vote(sd, sd.get('table'), line, d_, top, hh, bx, pv), xy))
        tr = {k_: sorted(v) for k_, v in tr.items()}
        tr = C.stitch_tracks(tr, tuple(f_['stitch'])) if f_.get('stitch') else tr
        tr = {'%s:%s' % (d.name, k_): v for k_, v in tr.items()}
        ev = C.flip_events(tr, sd['conf'], 0.95)
        if sd.get('gate', True):
            ev = C.gate(ev, u, hz, f_['move'], f_['rad'])
        if sd.get('birth'):
            ev = ev + C.birth_death(tr, sd['conf'], 0.95, hz, f_['rad'] * sd.get('brad', 1.0), True, False,
                                    u=u if sd.get('bmove') else None, move=f_['move'] if sd.get('bmove') else None,
                                    min_len=sd.get('blen', 0.0), dup=sd.get('bdup'), starts=[times[0]], gain=sd.get('bgain', 0.03))
        if sd.get('alt', 'none') != 'none':
            ev = C.alternate(ev, sd['alt'])
        out[d.name] = {'t0': times[0], 't1': times[-1], 'events': [{'t': round(e['t'], 2), 'kind': e['kind'], 'track': e['w']} for e in ev],
                       'people': len(tr)}
    return out


def compare(day, per_clip):
    import live_events as LE
    live = LE.events(day)
    res, used = [], set()
    for name, c in per_clip.items():
        lv = [(i, e) for i, e in enumerate(live) if c['t0'] <= e['t'] <= c['t1']]
        for te in c['events']:
            best = min(((abs(e['t'] - te['t']), i) for i, e in lv if e['kind'] == te['kind'] and i not in used
                        and abs(e['t'] - te['t']) <= TOL), default=None)
            if best:
                used.add(best[1])
                res.append({'t': te['t'], 'kind': te['kind'], 'by': 'both', 'live_key': LE.key_of(live[best[1]]), 'clip': name})
            else:
                res.append({'t': te['t'], 'kind': te['kind'], 'by': 'teacher', 'live_key': None, 'clip': name,
                            'clock': time.strftime('%H:%M:%S', time.localtime(te['t']))})
        for i, e in lv:
            if i not in used and not any(r.get('live_key') == LE.key_of(e) for r in res):
                used.add(i)
                res.append({'t': e['t'], 'kind': e['kind'], 'by': 'live', 'live_key': LE.key_of(e), 'clip': name})
    res.sort(key=lambda r: r['t'])
    s = {k: sum(r['by'] == k for r in res) for k in ('both', 'live', 'teacher')}
    s['live_events'] = len(live)
    s['live_events_in_clips'] = len({r['live_key'] for r in res if r['live_key']})
    return res, s


def main(day=None, at=None, until='09:30'):
    if at:
        while time.strftime('%H:%M') < at:
            time.sleep(30)
    day = day or time.strftime('%Y%m%d')
    log('night teacher for %s' % day)
    label_all(day, until)
    side = json.load(open(ROOT / 'data' / 'door_v2' / 'side_final.json'))
    per_clip = teacher_events(day, side)
    res, s = compare(day, per_clip)
    json.dump({'day': day, 'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'clips': per_clip, 'events': res, 'summary': s},
              open(NIGHT / ('%s.json' % day), 'w'), indent=1)
    log('%s: %s' % (day, json.dumps(s)))


if __name__ == '__main__':
    a = sys.argv[1:]
    kw = {}
    for flag in ('--at', '--until'):
        if flag in a:
            i = a.index(flag)
            kw[flag[2:]] = a[i + 1]
            del a[i:i + 2]
    main(*(a[:1] or [None]), **kw)
