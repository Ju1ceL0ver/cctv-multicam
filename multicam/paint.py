"""Hard frames outlined by the owner himself, as in a paint program: the gold standard.

The machine draws first -- the teacher finds the people (it misses almost nobody), SAM 2.1
Large draws their outlines (the owner judged them the best of three on 23.09) -- and the owner
only corrects: an eyedropper picks a person, a brush extends them, an eraser takes away, a new
index adds somebody nobody found, delete removes a false one. One label per pixel, so painting a
person over another says who stands in front.

Frames are stored at 1280x720 (half of the camera's 2560x1440); a label map is an 8-bit PNG
where the value is the person's index (0 = nobody). These are the only masks the student may
learn from without doubt, and the fixed test it is measured on.

usage: paint.py prep [N]     -> data/paint/<id>.jpg, <id>_init.png, manifest.json
       paint.py days DAY...  -> more hard frames of other days, appended
page:  /paint, edits in data/paint/<id>_mask.png, state in data/paint/state.json"""
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'paint'
W, H = 1280, 720


def prep(n=100, root=ROOT, log=print):
    """Label maps for the hard frames of the blind comparison, SAM-drawn from the teacher's boxes."""
    import cv2
    import day_movie
    import seg_compare
    from ultralytics import SAM
    OUT.mkdir(parents=True, exist_ok=True)
    sam = SAM(seg_compare.SAM_WEIGHTS)
    source = json.load(open(Path(root) / 'data' / 'segcompare' / 'manifest.json'))['items'][:n]
    items = []
    t0 = time.time()
    for it in source:
        frame, _ = seg_compare.raw_at(it['day'], it['cam'], it['film'], root)
        if frame is None:
            continue
        with np.load(Path(root) / 'data/raw_clips' / it['clip'] / ('dets_%s.npz' % day_movie.TAG)) as z:
            d = z[it['cam']]
        rows = np.nonzero((d[:, 0] == it['t']) & (d[:, 5] >= 0.25))[0]      # far and unsure people too
        boxes = d[rows, 1:5]
        labels = np.zeros((H, W), np.uint8)
        if len(rows):
            res = sam.predict(frame, bboxes=boxes.tolist(), imgsz=1024, verbose=False)[0]
            masks = res.masks.data.cpu().numpy() if res.masks is not None else []
            # the nearer person (lower feet) is painted last: they stand in front
            for k in np.argsort(boxes[:, 3]):
                if k >= len(masks):
                    continue
                m = cv2.resize(masks[k].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                labels[m] = int(k) + 1
        ident = it['id']
        cv2.imwrite(str(OUT / ('%s.jpg' % ident)), cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(OUT / ('%s_init.png' % ident)), labels)
        items.append({'id': ident, 'day': it['day'], 'cam': it['cam'], 'film': it['film'], 'clip': it['clip'],
                      't': it['t'], 'machine_people': int(len(rows))})
        if len(items) % 10 == 0:
            log('%d frames (%.0f s)' % (len(items), time.time() - t0))
    json.dump({'items': items, 'size': [W, H], 'made': time.strftime('%Y-%m-%dT%H:%M:%S')},
              open(OUT / 'manifest.json', 'w'))
    log('done: %d frames, %.0f s' % (len(items), time.time() - t0))


def _raw_frame(path, second):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, frame = cap.read()
    at = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    cap.release()
    return (frame, at) if ok else (None, None)


def prep_days(days, per_day=25, hard=20, every=90, root=ROOT, log=print, batch=2, offset=5):
    """More frames to paint from days nobody has seen: sampled straight from the raw recording
    every `every` s, the teacher looks at each, the hardest `hard` and a few ordinary ones with
    people are drafted by SAM and appended to the manifest after the frames already there."""
    import random
    import cv2
    import seg_compare
    from rawsource import segments as raw_segments
    from ultralytics import SAM, YOLO
    teacher = YOLO(r'C:\Users\ArykovAA\cctv_ai\retail_analytics\models\yolo26x-seg.pt', task='segment')
    sam = SAM(seg_compare.SAM_WEIGHTS)
    man_path = OUT / 'manifest.json'
    man = json.load(open(man_path)) if man_path.exists() else {'items': [], 'size': [W, H]}
    next_id = max([int(it['id']) for it in man['items']] + [-1]) + 1
    rng = random.Random(24 + batch)
    taken = {(it.get('cam'), it.get('segment'), int(it.get('second', -99) // 60)) for it in man['items']}
    t0 = time.time()
    for day in days:
        seen = []
        for cam in ('cam1', 'cam2'):
            for path, start in raw_segments(cam, day):
                for second in range(offset, 900, every):
                    if (cam, os.path.basename(path), second // 60) in taken:
                        continue                 # a minute already in an earlier batch
                    frame, at = _raw_frame(path, second)
                    if frame is None:
                        break
                    r = teacher.predict(frame, imgsz=1536, conf=0.25, classes=[0], verbose=False)[0]
                    if r.boxes is None or not len(r.boxes):
                        continue
                    b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.conf.cpu().numpy()
                    sure = (c >= 0.4) & (b[:, 3] - b[:, 1] >= 150)
                    d = np.c_[np.zeros(len(b)), b, c][sure]
                    score = seg_compare.hardness(d) if len(d) else 0
                    seen.append((score, cam, path, second, b, rng.random()))
        log('%s: %d frames with people looked at (%.0f s)' % (day, len(seen), time.time() - t0))
        seen.sort(key=lambda x: -x[0])
        chosen, used = [], set()
        for x in seen:                                   # hardest, at most one per camera-minute
            key = (x[1], x[2], x[3] // 60)
            if key not in used and len(chosen) < hard:
                chosen.append(x); used.add(key)
        rest = sorted((x for x in seen if (x[1], x[2], x[3] // 60) not in used), key=lambda x: x[5])
        chosen += rest[:per_day - len(chosen)]           # and some ordinary ones
        for score, cam, path, second, boxes, _ in chosen:
            frame, at = _raw_frame(path, second)
            res = sam.predict(frame, bboxes=boxes.tolist(), imgsz=1024, verbose=False)[0]
            masks = res.masks.data.cpu().numpy() if res.masks is not None else []
            labels = np.zeros((H, W), np.uint8)
            for k in np.argsort(boxes[:, 3]):
                if k < len(masks):
                    m = cv2.resize(masks[k].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    labels[m] = int(k) + 1
            ident = '%03d' % next_id
            next_id += 1
            cv2.imwrite(str(OUT / ('%s.jpg' % ident)), cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(OUT / ('%s_init.png' % ident)), labels)
            man['items'].append({'id': ident, 'day': day, 'cam': cam, 'segment': os.path.basename(path),
                                 'second': round(float(at), 2), 'machine_people': int(len(boxes)),
                                 'hardness': int(score), 'batch': batch})
        from storage import atomic_json
        atomic_json(man_path, man)          # the page reads it meanwhile
        log('%s: %d frames added, manifest %d (%.0f s)' % (day, len(chosen), len(man['items']), time.time() - t0))


def raw_days():
    """Days both cameras recorded, today excluded (it is still being written)."""
    from rawsource import RAW
    today = time.strftime('%Y%m%d')
    days = set(os.listdir(os.path.join(RAW, 'cam1'))) & set(os.listdir(os.path.join(RAW, 'cam2')))
    return sorted(d for d in days if d.isdigit() and len(d) == 8 and d < today)


def more(n=100, root=ROOT, log=print):
    """The next batch of n frames: four days, the least painted ones first, the minutes not yet taken."""
    import random
    from storage import atomic_json
    man = json.load(open(OUT / 'manifest.json'))
    batch = max(int(it.get('batch', 1)) for it in man['items']) + 1
    used = {}
    for it in man['items']:
        used[it['day']] = used.get(it['day'], 0) + 1
    rng = random.Random(batch)
    days = sorted(raw_days(), key=lambda d: (used.get(d, 0), rng.random()))[:4]
    per = -(-n // len(days))
    job = OUT / 'job.json'
    atomic_json(job, {'running': True, 'batch': batch, 'days': days, 'started': time.strftime('%Y-%m-%dT%H:%M:%S')})
    try:
        prep_days(days, per_day=per, hard=int(per * 0.8), root=root, log=log, batch=batch, offset=rng.randint(5, 80))
        clean_drafts(log=log)
        atomic_json(job, {'running': False, 'batch': batch, 'days': days, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S')})
    except Exception as exc:
        atomic_json(job, {'running': False, 'batch': batch, 'days': days, 'error': str(exc)[-500:]})
        raise


def clean_drafts(min_px=60, root=ROOT, log=print):
    """Drop the specks SAM scatters from drafts nobody has edited yet. Of the 99 draft people the
    owner deleted on 24.09, 81 were pieces under 60 px -- work the machine can do itself."""
    import cv2
    from storage import atomic_json  # noqa: F401  (kept with the other writers)
    man = json.load(open(OUT / 'manifest.json'))
    changed = 0
    for it in man['items']:
        init, edited = OUT / ('%s_init.png' % it['id']), OUT / ('%s_mask.png' % it['id'])
        if edited.exists() or not init.exists():
            continue
        lab = cv2.imread(str(init), cv2.IMREAD_UNCHANGED)
        out = lab.copy()
        for v in np.unique(lab):
            if not v:
                continue
            n, comp, stats, _ = cv2.connectedComponentsWithStats((lab == v).astype(np.uint8), 8)
            for c in range(1, n):
                if stats[c, cv2.CC_STAT_AREA] < min_px:
                    out[comp == c] = 0
        if (out != lab).any():
            cv2.imwrite(str(init), out)
            changed += 1
    log('drafts cleaned: %d' % changed)
    return changed


def state(root=ROOT):
    p = Path(root) / 'data' / 'paint' / 'state.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def save(root, ident, png_b64, done):
    """Store one edited label map; the previous edit goes to history, never lost."""
    import cv2
    from storage import atomic_json, file_lock
    folder = Path(root) / 'data' / 'paint'
    raw = base64.b64decode(png_b64.split(',', 1)[-1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError('Не картинка')
    labels = img if img.ndim == 2 else img[:, :, 2 if img.shape[2] >= 3 else 0]      # BGR(A): red channel holds the index
    if labels.shape != (H, W):
        raise ValueError('Не тот размер: %s' % (labels.shape,))
    with file_lock(str(folder / 'state.json') + '.lock'):
        target = folder / ('%s_mask.png' % ident)
        if target.exists():
            hist = folder / 'history'
            hist.mkdir(exist_ok=True)
            target.replace(hist / ('%s_%s.png' % (ident, time.strftime('%Y%m%d_%H%M%S'))))
        cv2.imwrite(str(target), labels.astype(np.uint8))
        st = state(root)
        people = [int(v) for v in np.unique(labels) if v]
        st[ident] = {'done': bool(done), 'people': len(people), 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        atomic_json(folder / 'state.json', st)
    return st[ident]


def register(app, root):
    import re
    from flask import abort, jsonify, render_template, request, send_file
    here = root if callable(root) else (lambda: root)

    def valid(ident):
        if not re.fullmatch(r'\d{3}', ident):
            abort(404)

    @app.get('/paint')
    def paint_page():
        return render_template('paint.html')

    @app.get('/api/paint')
    def paint_list():
        p = Path(here()) / 'data' / 'paint' / 'manifest.json'
        m = json.load(open(p)) if p.exists() else {'items': []}
        job_path = Path(here()) / 'data' / 'paint' / 'job.json'
        job = json.load(open(job_path)) if job_path.exists() else None
        return jsonify({'items': [{'id': it['id'], 'machine_people': it['machine_people'], 'batch': int(it.get('batch', 1)),
                                   'day': it.get('day')} for it in m['items']],
                        'size': [W, H], 'state': state(here()), 'job': job})

    @app.post('/api/paint/more')
    def paint_more():
        import bg
        if bg.running('paint.py'):
            return jsonify({'error': 'Следующая партия уже набирается'}), 409
        bg.spawn('paintmore', ['paint.py', 'more', '100'], wait=1)
        return jsonify({'ok': True})

    @app.get('/paint-img/<ident>/<what>')
    def paint_img(ident, what):
        valid(ident)
        folder = Path(here()) / 'data' / 'paint'
        if what == 'frame':
            p, mime = folder / ('%s.jpg' % ident), 'image/jpeg'
        elif what == 'labels':      # the owner's edit if there is one, else the machine's draft
            p = folder / ('%s_mask.png' % ident)
            p, mime = (p if p.exists() else folder / ('%s_init.png' % ident)), 'image/png'
        elif what == 'init':
            p, mime = folder / ('%s_init.png' % ident), 'image/png'
        else:
            abort(404)
        if not p.exists():
            abort(404)
        r = send_file(p, mimetype=mime)
        r.headers['Cache-Control'] = 'no-store'
        return r

    @app.post('/api/paint/<ident>')
    def paint_save(ident):
        valid(ident)
        body = request.get_json(force=True) or {}
        try:
            return jsonify(save(here(), ident, body.get('png', ''), body.get('done', True)))
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        except TimeoutError as exc:
            return jsonify({'error': str(exc)}), 503


if __name__ == '__main__':
    os.chdir(ROOT)
    say = lambda m: print(time.strftime('%H:%M:%S'), m, flush=True)
    if sys.argv[1] == 'prep':
        prep(int(sys.argv[2]) if len(sys.argv) > 2 else 100, log=say)
    elif sys.argv[1] == 'days':
        prep_days(sys.argv[2:], log=say)
        clean_drafts(log=say)
    elif sys.argv[1] == 'clean':
        clean_drafts(log=say)
    elif sys.argv[1] == 'more':
        more(int(sys.argv[2]) if len(sys.argv) > 2 else 100, log=say)
