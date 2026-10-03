"""Pictures of people for teaching the ReID model this shop, cut from the teachers' tracks.

Nobody labels anything here. A track of the night pass is one person for as long as the
tracker held it -- hundreds of looks, from the front and, once they turn in the hall, from
the back, which is exactly what the door needs (the camera sees an entering customer from
the front and a leaving one from the back). Two tracks seen at the same moment by the same
camera are two different people. That is all the training needs.

What would poison it is a track that changed person on the way. So a track is cut wherever
its look jumps (the same jump `day_movie.switches` finds), and only clean looks are kept:
the person is big enough to be recognisable, the detector is sure, and no other detection
touches their box in that frame.

The frames come from the light copies (`day_proxy`, 960 px, frame k = frame k of the
recording), read in order once per segment, on the CPU at low priority.

usage: reid_harvest.py DAY [DAY ...]   (env RA_LIMIT: only the first N pieces)
writes data/reid_harvest/<day>/<track>/<n>.jpg and data/reid_harvest/<day>/manifest.json"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MIN_H = 200          # px of the 2560x1440 frame: smaller people are a few dozen pixels in the copy
MIN_SCORE = 0.5
TOUCH = 0.05         # IoU with any other detection in the same frame that spoils a look
CUT = 0.20           # look jump inside a track that ends one identity and starts another
WINDOW = 8           # detections either side when measuring the jump
MIN_LOOKS = 6
MIN_SPAN = 1.0       # s
MOVED = 0.3          # of the person's height: pieces that never move this far are one pose repeated
PER_TRACK = 24
SIZE = (128, 256)    # w, h: what OSNet takes


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def jumps(looks, w=WINDOW, cut=CUT):
    """Positions where the look of a track changes for good, strongest first, 1 s apart."""
    if len(looks) < 3 * w:
        return []
    v = looks / np.maximum(np.linalg.norm(looks, axis=1, keepdims=True), 1e-8)
    total = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
    at = np.arange(w, len(v) - w + 1)
    before, after = total[at] - total[at - w], total[at + w] - total[at]
    cos = (before * after).sum(1) / np.maximum(np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1), 1e-8)
    out = []
    for j in np.argsort(-(1 - cos)):
        if 1 - cos[j] < cut:
            break
        if all(abs(at[j] - k) >= w for k in out):
            out.append(int(at[j]))
    return sorted(out)


def clean_rows(dets, rows):
    """The detections of a track that make a good picture of that person alone."""
    by_time = defaultdict(list)
    for i, t in enumerate(dets[:, 0]):
        by_time[round(float(t), 3)].append(i)
    keep = []
    for r in rows:
        x1, y1, x2, y2, score = dets[r, 1:6]
        if y2 - y1 < MIN_H or score < MIN_SCORE:
            continue
        others = [i for i in by_time[round(float(dets[r, 0]), 3)] if i != r]
        if others and iou(dets[r, 1:5], dets[others, 1:5]).max() > TOUCH:
            continue
        keep.append(r)
    return np.asarray(keep, np.int64)


def identities(day, root=ROOT):
    """Tracks of the day cut into single-person pieces, with their clean looks."""
    import day_movie
    data = day_movie.index(day, root)
    out = []
    cache = {}
    for key, tr in data['tracks'].items():
        if ':x' in key:                     # leftovers chained after the fact: not a tracker's identity
            continue
        clip = tr['clip']
        if clip not in cache:
            folder = Path(root) / 'data/raw_clips' / clip
            with np.load(folder / ('dets_%s.npz' % day_movie.TAG)) as z:
                dets = {c: z[c] for c in ('cam1', 'cam2')}
            emb = folder / day_movie.EMB
            looks = None
            if emb.exists():
                with np.load(emb) as z:
                    looks = {c: z[c] for c in ('cam1', 'cam2')}
            cache = {clip: (dets, looks)}
        dets, looks = cache[clip]
        cam, rows, times = tr['cam'], tr['rows'], tr['times']
        cuts = []
        if looks is not None and rows.max() < len(looks[cam]):
            cuts = jumps(looks[cam][rows].astype(np.float64))
        for n, (a, b) in enumerate(zip([0] + cuts, cuts + [len(rows)])):
            part, when = rows[a:b], times[a:b]
            good = clean_rows(dets[cam], part)
            if len(good) < MIN_LOOKS:
                continue
            at = {r: t for r, t in zip(part, when)}
            gt = np.array([at[r] for r in good])
            if gt[-1] - gt[0] < MIN_SPAN:
                continue
            pick = np.unique(np.linspace(0, len(good) - 1, min(PER_TRACK, len(good))).round().astype(int))
            # the saleswoman at her desk is one piece after another of the same sitting pose:
            # thousands of identical pictures that teach nothing about turning or walking
            b = dets[cam][good, 1:5]
            centre = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
            if np.ptp(centre, axis=0).max() < MOVED * np.median(b[:, 3] - b[:, 1]):
                continue
            out.append({'id': '%s#%d' % (key, n), 'cam': cam, 'clip': clip, 'shop': bool(tr['shop']),
                        'first': float(gt[0]), 'last': float(gt[-1]),
                        'looks': [(float(gt[k]), [float(v) for v in dets[cam][good[k], 1:5]]) for k in pick]})
    return out


def locate(day, root):
    """film time -> (segment path, frame index) per camera, the same rule as day_masks.Frames."""
    import day_movie
    import day_player
    start, _ = day_movie.clock(day, root)
    offset = day_movie.offset_of(day, root)
    segs = day_player.segments(day, root)

    def where(cam, t):
        when = start + t + (offset if cam == 'cam2' else 0.0)
        seg = None
        for s in segs[cam]:
            if s.ready and not s.broken and s.start + s.times()[0] <= when + 1e-6:
                seg = s
        if seg is None:
            return None
        times = seg.times()
        k = int(np.searchsorted(times, when - seg.start + 1e-6, 'right')) - 1
        if k < 0 or when - seg.start > times[-1] + 0.5:
            return None
        return str(Path(root) / 'data/day_proxy' / day / cam / seg.name), k
    return where


def harvest(day, root=ROOT, log=print, limit=None):
    import cv2
    out_dir = Path(root) / 'data/reid_harvest' / day
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    people = identities(day, root)
    if limit:
        people = people[:limit]
    log('%s: %d single-person pieces, %d looks to cut (%.0f s)' % (
        day, len(people), sum(len(p['looks']) for p in people), time.time() - t0))
    where = locate(day, root)
    jobs = defaultdict(list)            # segment -> [(frame, person, n, box)]
    for p in people:
        for n, (t, box) in enumerate(p['looks']):
            spot = where(p['cam'], t)
            if spot:
                jobs[spot[0]].append((spot[1], p['id'], n, box))
    saved = defaultdict(int)
    for s, (path, items) in enumerate(sorted(jobs.items())):
        items.sort()
        cap = cv2.VideoCapture(path)
        k, i = -1, 0
        while i < len(items):
            if not cap.grab():
                break
            k += 1
            if k < items[i][0]:
                continue
            ok, frame = cap.retrieve()
            while i < len(items) and items[i][0] == k:
                _, pid, n, box = items[i]
                i += 1
                if not ok:
                    continue
                sc = frame.shape[1] / 2560.0
                x1, y1, x2, y2 = [v * sc for v in box]
                px, py = 0.05 * (x2 - x1), 0.03 * (y2 - y1)
                a, b = int(max(0, x1 - px)), int(min(frame.shape[1], x2 + px))
                c, d = int(max(0, y1 - py)), int(min(frame.shape[0], y2 + py))
                if b - a < 8 or d - c < 16:
                    continue
                crop = cv2.resize(frame[c:d, a:b], SIZE, interpolation=cv2.INTER_AREA)
                folder = out_dir / pid.replace(':', '_').replace('#', '_')
                folder.mkdir(exist_ok=True)
                cv2.imwrite(str(folder / ('%02d.jpg' % n)), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved[pid] += 1
        cap.release()
        if s % 10 == 0:
            log('%s: segment %d/%d, %d pictures (%.0f s)' % (day, s + 1, len(jobs), sum(saved.values()), time.time() - t0))
    manifest = {'day': day, 'made': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'rules': {'min_h': MIN_H, 'min_score': MIN_SCORE, 'touch': TOUCH, 'cut': CUT, 'per_track': PER_TRACK},
                'people': [{k: p[k] for k in ('id', 'cam', 'clip', 'shop', 'first', 'last')}
                           | {'folder': p['id'].replace(':', '_').replace('#', '_'), 'n': saved[p['id']]}
                           for p in people if saved[p['id']] >= MIN_LOOKS // 2]}
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f)
    log('%s: done, %d people, %d pictures, %.0f s' % (day, len(manifest['people']), sum(saved.values()), time.time() - t0))
    return manifest


if __name__ == '__main__':
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except Exception:
        pass
    os.chdir(ROOT)
    limit = int(os.environ.get('RA_LIMIT', '0')) or None      # a few people only: a quick look before the whole day
    for day in sys.argv[1:]:
        harvest(day, ROOT, log=lambda m: print(time.strftime('%H:%M:%S'), m, flush=True), limit=limit)
