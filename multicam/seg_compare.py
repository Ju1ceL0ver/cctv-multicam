"""Is the heavy SAM worth labelling hours of video with? The owner looks and says.

A hundred hard frames of 17-18.09 -- a crowd, people overlapping, a child beside a grown-up --
each outlined three ways: the student (yolo26n-seg, what would run live), the teacher
(yolo26x-seg, what the student learned from) and SAM 2.1 Large given the teacher's boxes (what
the proposed relabelling would produce). The three pictures are shown side by side in an order
drawn at random per frame and hidden from the owner, who names the best one or says they are
the same. If SAM wins where the student fails, relabelling is worth nights; if not, it is not.

usage: seg_compare.py build [N]      -> data/segcompare/ (pictures + manifest.json)
page:  /segcompare on the review site, answers in data/segcompare/answers.json"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'segcompare'
STUDENT = ROOT / 'runs' / 'student_seg_326054ea1842c72029c2' / 'weights' / 'best.pt'
SAM_WEIGHTS = 'sam2.1_l.pt'
DAYS = ('20260917', '20260918')
UNIT = 960               # the student's input size, as it was trained
COLOURS = [(66, 135, 245), (245, 130, 48), (60, 180, 75), (230, 25, 75), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40)]
METHODS = ('student', 'teacher', 'sam')


def _iou(a, b):
    x1, y1 = np.maximum(a[0], b[0]), np.maximum(a[1], b[1])
    x2, y2 = np.minimum(a[2], b[2]), np.minimum(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def hardness(d):
    """How hard one frame's people are: overlapping pairs and a small figure beside a big one count most."""
    b = d[:, 1:5]
    h = b[:, 3] - b[:, 1]
    score = len(d)
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            if _iou(b[i], b[j]) > 0.1:
                score += 3
                if min(h[i], h[j]) < 0.7 * max(h[i], h[j]):
                    score += 3                 # a child beside a grown-up, or one half hidden
    return score


def candidates(day, root=ROOT):
    """(score, clip, cam, t, rows) for frames with at least three sure people, one per 60 s of film."""
    import day_movie
    out = []
    for folder, meta in day_movie._clips(day, Path(root)):
        dets_path = folder / ('dets_%s.npz' % day_movie.TAG)
        if not dets_path.exists():
            continue
        with np.load(dets_path) as z:
            dets = {c: z[c] for c in ('cam1', 'cam2')}
        for cam in ('cam1', 'cam2'):
            d = dets[cam]
            if not len(d):
                continue
            keep = (d[:, 5] >= 0.4) & (d[:, 4] - d[:, 2] >= 150)
            times = np.unique(d[keep, 0])
            for t in times:
                rows = np.nonzero(keep & (d[:, 0] == t))[0]
                if len(rows) >= 3:
                    out.append((hardness(d[rows]), folder.name, cam, float(t), rows, float(meta['seconds'])))
    out.sort(key=lambda x: -x[0])
    chosen, seen = [], []
    for c in out:
        key = (c[1], c[2], c[3] // 60)
        if key in seen:
            continue
        seen.append(key)
        chosen.append(c)
    return chosen


def film_time(day, clip, cam, t, root):
    import day_movie
    import day_player
    folder = Path(root) / 'data/raw_clips' / clip
    meta = json.load(open(folder / ('meta_%s.json' % day_movie.TAG)))
    segs = day_player.segments(day, root)
    landed = day_player.start_frames(folder, meta, segs, day)
    start, _ = day_movie.clock(day, root)
    began = day_player._epoch(meta['start'])
    when = day_player.walls(segs[cam], began, [int(round(t * day_player.FPS))], landed[cam])[0]
    return float(when - (day_movie.offset_of(day, root) if cam == 'cam2' else 0.0) - start)


def raw_at(day, cam, at, root):
    """The full 2560x1440 recorded frame the film shows at `at`, checked against the light copy.

    The student has to be judged on what it would see live -- the camera's own frame -- not on
    the 960 px re-encoded copy: on the first try it found 3 of 6, 1 of 4 and 5 of 9 people there."""
    import cv2
    import day_masks
    import day_player
    from rawsource import segments as raw_segments
    small, info = day_masks.FRAMES.at(day, cam, at, root)
    if small is None:
        return None, None
    seg = next(s for s in day_player.segments(day, root)[cam] if s.name == info['segment'])
    target = float(seg.times()[info['index']])
    path = next((p for p, _ in raw_segments(cam, day) if os.path.basename(p) == info['segment']), None)
    if path is None:
        return None, small
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (target - 3.0) * 1000))
    frame = None
    while cap.grab():
        if cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 >= target - 0.005:
            ok, frame = cap.retrieve()
            break
    cap.release()
    if frame is None:
        return None, small
    check = cv2.resize(frame, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    if float(np.abs(check.astype(np.int16) - small.astype(np.int16)).mean()) > 12:
        return None, small                     # not the same moment: better no frame than a wrong one
    return frame, small


def rings_from_masks(masks):
    import day_masks
    return [day_masks.rings_of(m.astype(bool), 1.0) for m in masks]


def draw(image, people, crop):
    import cv2
    over = image.copy()
    width = max(2, image.shape[1] // 480)
    for k, rings in enumerate(people):
        colour = COLOURS[k % len(COLOURS)]
        for r in rings:
            cv2.fillPoly(over, [np.asarray(r, np.int32)], colour)
    out = cv2.addWeighted(over, 0.42, image, 0.58, 0)
    for k, rings in enumerate(people):
        for r in rings:
            cv2.polylines(out, [np.asarray(r, np.int32)], True, COLOURS[k % len(COLOURS)], width, cv2.LINE_AA)
    x1, y1, x2, y2 = crop
    part = out[y1:y2, x1:x2]
    scale = 760 / max(1, part.shape[1])
    return cv2.resize(part, (760, int(part.shape[0] * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def build(n=100, root=ROOT, log=print):
    import cv2
    import day_masks
    import day_movie
    from ultralytics import SAM, YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    student = YOLO(str(STUDENT), task='segment')
    sam = SAM(SAM_WEIGHTS)
    rng = random.Random(23)
    per_day = n // len(DAYS)
    items = []
    t0 = time.time()
    for day in DAYS:
        pool = candidates(day, root)
        log('%s: %d hard frames to choose from' % (day, len(pool)))
        took = 0
        for score, clip, cam, t, rows, _ in pool:
            if took >= per_day:
                break
            try:
                at = film_time(day, clip, cam, t, root)
            except Exception:
                continue
            image, _ = raw_at(day, cam, at, root)
            if image is None:
                continue
            k = image.shape[1] / 2560.0
            folder = Path(root) / 'data/raw_clips' / clip
            with np.load(folder / ('dets_%s.npz' % day_movie.TAG)) as z:
                boxes = z[cam][rows, 1:5] * k
            with np.load(folder / ('polys_%s.npz' % day_movie.TAG)) as z:
                pts, off = z[cam + '_pts'], z[cam + '_off']
            teacher = [day_movie.clean_outline(pts[off[r]:off[r + 1]], unit=image.shape[1]) for r in rows]
            res = student.predict(image, imgsz=UNIT, conf=0.25, classes=[0], retina_masks=True, verbose=False)[0]
            stud = rings_from_masks(res.masks.data.cpu().numpy()) if res.masks is not None else []
            res = sam.predict(image, bboxes=boxes.tolist(), imgsz=1024, verbose=False)[0]
            samr = rings_from_masks(res.masks.data.cpu().numpy()) if res.masks is not None else []
            pad = 0.04 * image.shape[1]
            x1, y1 = boxes[:, :2].min(0) - pad
            x2, y2 = boxes[:, 2:].max(0) + pad
            crop = [int(max(0, x1)), int(max(0, y1)), int(min(image.shape[1], x2)), int(min(image.shape[0], y2))]
            order = list(METHODS)
            rng.shuffle(order)
            ident = '%03d' % len(items)
            drawn = {'student': stud, 'teacher': teacher, 'sam': samr}
            for letter, method in zip('abc', order):
                cv2.imwrite(str(OUT / ('%s_%s.jpg' % (ident, letter))), draw(image, drawn[method], crop),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
            cv2.imwrite(str(OUT / ('%s_raw.jpg' % ident)), cv2.resize(
                image[crop[1]:crop[3], crop[0]:crop[2]], (760, int((crop[3] - crop[1]) * 760 / max(1, crop[2] - crop[0]))),
                interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
            items.append({'id': ident, 'day': day, 'cam': cam, 'clip': clip, 't': t, 'film': round(at, 2),
                          'people': int(len(rows)), 'hardness': int(score), 'order': order,
                          'counts': {m: len(drawn[m]) for m in METHODS}})
            took += 1
            if len(items) % 10 == 0:
                log('%d frames (%.0f s)' % (len(items), time.time() - t0))
                json.dump({'items': items}, open(OUT / 'manifest.json', 'w'))
    json.dump({'items': items, 'made': time.strftime('%Y-%m-%dT%H:%M:%S')}, open(OUT / 'manifest.json', 'w'))
    log('done: %d frames, %.0f s' % (len(items), time.time() - t0))


# ---------------------------------------------------------------- the page

def answers(root=ROOT):
    p = Path(root) / 'data' / 'segcompare' / 'answers.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def tally(root=ROOT):
    """Which method the owner picked, unblinded only here."""
    m = json.load(open(Path(root) / 'data' / 'segcompare' / 'manifest.json'))
    order = {it['id']: it['order'] for it in m['items']}
    won = {k: 0 for k in METHODS}
    same = 0
    shared = {k: 0 for k in METHODS}       # picked together with another as equally good
    for ident, a in answers(root).items():
        best = a['best'] if isinstance(a['best'], list) else [a['best']]
        if best == ['same'] or sorted(best) == ['a', 'b', 'c']:
            same += 1
        elif ident in order:
            for letter in best:
                (won if len(best) == 1 else shared)[order[ident]['abc'.index(letter)]] += 1
    return {'frames': len(order), 'answered': len(answers(root)), 'best': won, 'best_shared': shared, 'same': same}


def register(app, root):
    from flask import abort, jsonify, render_template, request, send_file
    from storage import atomic_json, file_lock
    here = root if callable(root) else (lambda: root)

    @app.get('/segcompare')
    def segcompare_page():
        return render_template('segcompare.html')

    @app.get('/api/segcompare')
    def segcompare_list():
        p = Path(here()) / 'data' / 'segcompare' / 'manifest.json'
        if not p.exists():
            return jsonify({'items': [], 'answers': {}})
        items = [{'id': it['id']} for it in json.load(open(p))['items']]      # no order: the test is blind
        return jsonify({'items': items, 'answers': answers(here())})

    @app.post('/api/segcompare')
    def segcompare_answer():
        body = request.get_json(force=True) or {}
        ident, best = str(body.get('id', '')), body.get('best')
        if isinstance(best, list):             # several equally good: a set of letters
            best = sorted(set(best))
            if not best or any(b not in ('a', 'b', 'c') for b in best):
                abort(400)
        elif best not in ('a', 'b', 'c', 'same', None):
            abort(400)
        if not ident.isdigit():
            abort(400)
        p = Path(here()) / 'data' / 'segcompare' / 'answers.json'
        with file_lock(str(p) + '.lock'):
            a = answers(here())
            if best is None:
                a.pop(ident, None)
            else:
                a[ident] = {'best': best, 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
            atomic_json(p, a)
        return jsonify({'ok': True, 'answered': len(a)})

    @app.get('/segcompare-img/<name>')
    def segcompare_img(name):
        import re
        if not re.fullmatch(r'\d{3}_(a|b|c|raw)\.jpg', name):
            abort(404)
        p = Path(here()) / 'data' / 'segcompare' / name
        if not p.exists():
            abort(404)
        return send_file(p, mimetype='image/jpeg')


if __name__ == '__main__':
    os.chdir(ROOT)
    if sys.argv[1] == 'build':
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 100, log=lambda m: print(time.strftime('%H:%M:%S'), m, flush=True))
    elif sys.argv[1] == 'tally':
        print(json.dumps(tally(), indent=1))
