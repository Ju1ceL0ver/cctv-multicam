"""Click a person, get his outline: masks made by hand in the film, without drawing.

The owner clicks somebody the detector did not outline (or whose outline is wrong). A
promptable segmentation model (SAM family) turns the click into a mask within a second or
two on the CPU; further clicks add a part (left) or take one away (shift), each costing only
the light prompt decoder, because the heavy image encoding of that frame is kept. Saving it
creates the person, and a background job carries the mask forward and backward through the
film, frame to frame, prompting each next frame with the previous box, until the person is
gone or an ordinary track picks him up -- then the two are joined.

Everything runs on the CPU: the GPU belongs to the live counter by day. Interactive clicks
always come before the background job, which waits while a click is being served.

Frames come from the light copies (960 px), found by the recording's own timestamps, so a
mask is tied to one exact recorded frame (segment, index, time) and can be carried to the
original 2560 px frame for training. Rings are stored in pixels of the original frame.

Storage: data/day_movie/<day>/drawn.json, apart from the owner's edits.json, so a background
job writing a step never bumps the revision the owner's tab is editing against.
"""
import os
import threading
import time
from pathlib import Path

import numpy as np

from storage import atomic_json, file_lock, read_json

ROOT = Path(__file__).resolve().parent
RAW_W, RAW_H = 2560, 1440
STEP = 0.24              # seconds between carried frames: two looks of the detector
MAX_STEPS = 75           # 18 s each way at most; a longer stay is somebody the tracker holds
KEEP_AREA = (0.5, 2.0)   # a carried mask whose area jumps outside this is lost, not grown
JOIN_IOU = 0.5
_lock = threading.RLock()
_interactive = threading.Condition()
_waiting = [0]


# ---------------------------------------------------------------- frames

class Frames:
    """Exact recorded frames of the light copies, by shop-clock moment. Captures stay open so
    the background job reading forward frame after frame does not seek every time."""

    def __init__(self):
        self.open = {}

    def at(self, day, cam, t, root=ROOT):
        import cv2
        import day_movie
        import day_player
        start, _ = day_movie.clock(day, root)
        when = start + t + (day_movie.offset_of(day, root) if cam == 'cam2' else 0.0)
        segs = [s for s in day_player.segments(day, root)[cam] if s.ready and not s.broken]
        seg = None
        for s in segs:
            if s.start + s.times()[0] <= when + 1e-6:
                seg = s
        if seg is None:
            return None, None
        times = seg.times()
        k = int(np.searchsorted(times, when - seg.start + 1e-6, 'right')) - 1
        if k < 0 or when - seg.start > times[-1] + 0.5:
            return None, None
        side = read_json(Path(root) / 'data/day_proxy' / day / cam / (Path(seg.name).stem + '.json'), {})
        target = float(times[k]) + float(side.get('shift', 0.0))      # the copy's clock, same as the sidecar's
        path = str(Path(root) / 'data/day_proxy' / day / cam / seg.name)
        capture, last, image = self.open.get(path, (None, None, None))
        if capture is None or last is None or target < last - 0.001 or target > last + 3.0:
            if capture is not None:
                capture.release()
            capture = cv2.VideoCapture(path)
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (target - 3.0) * 1000))
            last, image = None, None
        if last is None or abs(last - target) >= 0.005 or image is None:
            image = None
            while capture.grab():
                last = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if last >= target - 0.005:
                    ok, image = capture.retrieve()
                    break
        self.open[path] = (capture, last, image)
        if len(self.open) > 6:
            self.open.pop(next(iter(self.open)))[0].release()
        if image is None:
            return None, None
        return image, {'cam': cam, 'segment': seg.name, 'index': k, 'stamp': round(seg.start + float(times[k]), 3),
                       'at': round(t, 3)}


FRAMES = Frames()


# ---------------------------------------------------------------- the model

def _weights(root=ROOT):
    """Which model outlines clicks. Chosen explicitly in data/day_movie/segmenter.json (written
    after measuring the machine's CPU), else the first checkpoint found where ultralytics looks
    for weights, else GrabCut -- which needs nothing and is only good enough for tests."""
    chosen = os.environ.get('RA_SEGMENTER') or read_json(Path(root) / 'data/day_movie/segmenter.json', {}).get('weights')
    if chosen:
        return chosen
    places = [Path(root), Path(root) / 'data' / 'weights']
    try:
        from ultralytics.utils import SETTINGS
        places.append(Path(SETTINGS['weights_dir']))
    except Exception:
        pass
    for name in ('sam2.1_t.pt', 'mobile_sam.pt', 'sam2.1_s.pt', 'sam2.1_b.pt'):
        for place in places:
            if (place / name).exists():
                return str(place / name)
    return 'grabcut'


class Segmenter:
    """One promptable model, one kept image encoding per camera slot."""

    def __init__(self, weights=None):
        self.weights = weights or _weights()
        self.slots = {}          # slot -> (key, predictor or grabcut image)
        self.shared = None

    @property
    def name(self):
        return Path(self.weights).name if self.weights != 'grabcut' else 'grabcut'

    def _predictor(self):
        from ultralytics.models.sam import Predictor, SAM2Predictor
        import torch
        torch.set_num_threads(int(os.environ.get('RA_SEG_THREADS', '6')))
        kind = SAM2Predictor if 'sam2' in self.name else Predictor
        predictor = kind(overrides=dict(conf=0.25, task='segment', mode='predict', imgsz=1024,
                                        model=self.weights, device='cpu', verbose=False, save=False))
        predictor.setup_model(model=self.shared)     # the weights are loaded once, for every slot
        self.shared = predictor.model
        return predictor

    def encode(self, slot, key, image):
        """Keep this frame's encoding in the slot; nothing to do if it is there already."""
        have = self.slots.get(slot)
        if have and have[0] == key:
            return False
        if self.name == 'grabcut':
            self.slots[slot] = (key, image)
            return True
        predictor = have[1] if have else self._predictor()
        predictor.reset_image()
        predictor.set_image(image)
        self.slots[slot] = (key, predictor)
        return True

    def mask(self, slot, points, labels, box=None):
        """Boolean mask the size of the encoded frame. Points and box in its pixels."""
        key, held = self.slots[slot]
        if self.name == 'grabcut':
            return _grabcut(held, points, labels, box)
        kw = {}
        if points:
            kw['points'] = [[list(map(float, p)) for p in points]]
            kw['labels'] = [[int(v) for v in labels]]
        if box is not None:
            kw['bboxes'] = [list(map(float, box))]
        result = held(**kw)[0]
        if result.masks is None or not len(result.masks.data):
            return None
        return result.masks.data[0].cpu().numpy() > 0.5


def _grabcut(image, points, labels, box):
    """Stand-in without weights (the Mac, the tests): a person-sized box around the clicks."""
    import cv2
    h, w = image.shape[:2]
    if box is None:
        pos = [p for p, l in zip(points, labels) if l == 1] or points
        x, y = np.mean(pos, 0)
        bw, bh = 0.07 * w, 0.42 * h
        box = [x - bw / 2, y - bh * 0.35, x + bw / 2, y + bh * 0.65]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
    for (px, py), label in zip(points, labels):
        cv2.circle(mask, (int(px), int(py)), 4, cv2.GC_FGD if label else cv2.GC_BGD, -1)
    back, fore = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, None, back, fore, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


SEGMENTER = None


def segmenter():
    global SEGMENTER
    with _lock:
        if SEGMENTER is None:
            SEGMENTER = Segmenter()
        return SEGMENTER


def rings_of(mask, scale, eps=0.8):
    """Outline rings of a boolean mask, scaled to the original frame's pixels."""
    import cv2
    if mask is None or not mask.any():
        return []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    biggest = max(cv2.contourArea(c) for c in contours)
    for c in contours:
        if cv2.contourArea(c) < max(12, 0.05 * biggest):      # specks the model scattered around
            continue
        ring = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ring) >= 3:
            out.append(np.rint(ring * scale).astype(int))
    return out


def box_of(rings):
    pts = np.vstack(rings) if rings else np.zeros((0, 2))
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())] if len(pts) else None


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class _Interactive:
    """Marks a click being served: the background job waits for it."""

    def __enter__(self):
        with _interactive:
            _waiting[0] += 1

    def __exit__(self, *exc):
        with _interactive:
            _waiting[0] -= 1
            _interactive.notify_all()


def _yield_to_clicks():
    with _interactive:
        while _waiting[0]:
            _interactive.wait(0.5)


def segment(day, cam, at, points, box=None, root=ROOT):
    """One interactive answer: the mask for these clicks on the frame shown at `at`.
    points: [[x, y, 1|0], ...] and box [x1, y1, x2, y2] in original-frame pixels."""
    began = time.time()
    with _Interactive(), _lock:
        image, info = FRAMES.at(day, cam, float(at), root)
        if image is None:
            raise ValueError('Здесь нет записи этой камеры')
        read = time.time() - began
        seg = segmenter()
        h, w = image.shape[:2]
        k = w / RAW_W
        fresh = seg.encode(cam, (info['segment'], info['index']), image)
        encoded = time.time() - began - read
        pts = [[p[0] * k, p[1] * k] for p in points]
        labels = [int(p[2]) for p in points]
        mask = seg.mask(cam, pts, labels, [v * k for v in box] if box else None)
        rings = rings_of(mask, 1 / k)
    return {'rings': [r.ravel().tolist() for r in rings], 'box': box_of(rings), 'frame': info,
            'model': seg.name, 'ms': {'read': round(read * 1000), 'encode': round(encoded * 1000) if fresh else 0,
                                      'total': round((time.time() - began) * 1000)}}


def warm(day, at, root=ROOT):
    """Encode both cameras' frames at `at` ahead of the first click (the video just stopped)."""
    def work():
        for cam in ('cam1', 'cam2'):
            try:
                with _Interactive(), _lock:
                    image, info = FRAMES.at(day, cam, float(at), root)
                    if image is not None:
                        segmenter().encode(cam, (info['segment'], info['index']), image)
            except Exception:
                pass
    threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------- what the owner drew

def _path(day, root):
    return Path(root) / 'data/day_movie' / day / 'drawn.json'


def drawn(day, root=ROOT):
    return read_json(_path(day, root), {'next': 1, 'items': {}})


def draw(day, body, root=ROOT):
    """Keep a mask the owner accepted. A new person starts carrying through the film at once;
    a fix replaces the machine's outline of that track at that frame only."""
    cam, at, rings = body.get('cam'), body.get('at'), body.get('rings')
    if cam not in ('cam1', 'cam2') or not isinstance(at, (int, float)) or not rings:
        raise ValueError('Нечего сохранять')
    rings = [[int(v) for v in r] for r in rings if isinstance(r, list) and len(r) >= 6]
    if not rings:
        raise ValueError('Нечего сохранять')
    replaces = body.get('replaces')
    path = _path(day, root)
    with file_lock(path.with_suffix('.lock')):
        store = drawn(day, root)
        ident = store['next']
        store['next'] += 1
        store['items'][str(ident)] = {
            'id': ident, 'cam': cam, 'kind': 'fix' if replaces else 'new', 'replaces': replaces,
            'samples': [[round(float(at), 3), rings, 'owner']], 'points': body.get('points') or [],
            'frame': body.get('frame'), 'deleted': False, 'joins': None,
            'carry': None if replaces else {'state': 'queued', 'forward': 0, 'backward': 0},
            'made': time.strftime('%Y-%m-%dT%H:%M:%S')}
        atomic_json(path, store)
    if not replaces:
        carry_later(day, ident, root)
    return {'id': ident, 'track': 'd%d' % ident}


def undraw(day, ident, root=ROOT):
    path = _path(day, root)
    with file_lock(path.with_suffix('.lock')):
        store = drawn(day, root)
        item = store['items'].get(str(ident))
        if not item:
            raise ValueError('Такой обводки нет')
        item['deleted'] = not item['deleted']
        atomic_json(path, store)
    return {'id': ident, 'deleted': item['deleted']}


def _update(day, ident, root, change):
    path = _path(day, root)
    with file_lock(path.with_suffix('.lock')):
        store = drawn(day, root)
        item = store['items'].get(str(ident))
        if item is None:
            return None
        change(item)
        atomic_json(path, store)
        return item


# ---------------------------------------------------------------- carrying through the film

_queue = []
_worker = [None]


def carry_later(day, ident, root=ROOT):
    with _lock:
        _queue.append((day, ident, str(root)))
        if _worker[0] is None or not _worker[0].is_alive():
            _worker[0] = threading.Thread(target=_run_queue, daemon=True)
            _worker[0].start()


def _run_queue():
    while True:
        with _lock:
            if not _queue:
                return
            day, ident, root = _queue.pop(0)
        try:
            carry(day, ident, Path(root))
        except Exception as exc:                       # noqa: BLE001 -- recorded, never raised into a request
            _update(day, ident, root, lambda it: it.__setitem__('carry', dict(it['carry'] or {}, state='failed',
                                                                               error=str(exc)[:200])))


def machine_boxes(day, cam, t, root, near=0.07):
    """Machine outlines of this camera around moment t: [(part id, box)]."""
    import day_movie
    data = day_movie.index(day, root)
    out = []
    for key, info in data['tracks'].items():
        if info['cam'] != cam or not info['first'] - near <= t <= info['last'] + near:
            continue
        j = int(np.argmin(np.abs(info['times'] - t)))
        if abs(info['times'][j] - t) > near:
            continue
        out.append((key, info, j))
    return out


def carry(day, ident, root=ROOT, segmenter_=None):
    """Carry one drawn person forward, then backward, frame by frame."""
    import day_movie
    store = drawn(day, root)
    item = store['items'].get(str(ident))
    if not item or item['deleted'] or item['kind'] != 'new':
        return
    seg = segmenter_ or segmenter()
    cam = item['cam']
    start, end = day_movie.clock(day, root)
    duration = end - start
    _update(day, ident, root, lambda it: it.__setitem__('carry', dict(it['carry'] or {}, state='running')))
    seed_at, seed_rings, _ = item['samples'][0]
    joins = []
    for direction, name in ((1, 'forward'), (-1, 'backward')):
        joined = None
        prev_rings = [np.asarray(r).reshape(-1, 2) for r in seed_rings]
        prev_box, prev_at = box_of(prev_rings), seed_at
        prev_area = _area(prev_rings)
        for step in range(1, MAX_STEPS + 1):
            _yield_to_clicks()
            if drawn(day, root)['items'][str(ident)]['deleted']:
                return
            at = prev_at + direction * STEP
            if not 0 <= at <= duration:
                break
            with _lock:
                image, info = FRAMES.at(day, cam, at, root)
                if image is None:
                    break
                k = image.shape[1] / RAW_W
                seg.encode('carry', (info['segment'], info['index']), image)
                x1, y1, x2, y2 = prev_box
                grow = 0.12 * max(x2 - x1, y2 - y1)
                box = [max(0, x1 - grow) * k, max(0, y1 - grow) * k, min(RAW_W, x2 + grow) * k, min(RAW_H, y2 + grow) * k]
                cx, cy = (x1 + x2) / 2 * k, (y1 + 0.45 * (y2 - y1)) * k
                mask = seg.mask('carry', [[cx, cy]], [1], box)
                rings = rings_of(mask, 1 / k)
            if not rings:
                break
            area, new_box = _area(rings), box_of(rings)
            if not KEEP_AREA[0] <= area / max(prev_area, 1) <= KEEP_AREA[1] or _iou(new_box, prev_box) < 0.3:
                break
            # an ordinary track has him here: stop carrying and join him to it
            for key, info_track, j in machine_boxes(day, cam, at, root):
                boxes = _machine_box(day, info_track, j, root)
                if boxes is not None and _iou(new_box, boxes) >= JOIN_IOU:
                    joined = key
                    break
            if joined:
                break
            sample = [round(at, 3), [r.ravel().tolist() for r in rings], seg.name]
            count = step

            def add(it, sample=sample, name=name, count=count):
                it['samples'].append(sample)
                it['samples'].sort(key=lambda s: s[0])
                it['carry'] = dict(it['carry'] or {}, **{name: count})
            _update(day, ident, root, add)
            prev_rings, prev_box, prev_at, prev_area = rings, new_box, at, area
        if joined:                          # this direction is the tracker's from here on
            joins.append(joined + '@')
    _update(day, ident, root, lambda it: (it.__setitem__('joins', joins or None),
                                          it.__setitem__('carry', dict(it['carry'] or {}, state='done'))))


def _area(rings):
    import cv2
    return float(sum(abs(cv2.contourArea(np.asarray(r, np.float32).reshape(-1, 1, 2))) for r in rings))


_box_cache = {}


def _machine_box(day, info, j, root):
    """Box of detection j of a machine track, in original pixels."""
    folder = Path(root) / 'data/raw_clips' / info['clip']
    key = (str(folder), info['cam'])
    dets = _box_cache.get(key)
    if dets is None:
        with np.load(folder / 'dets_yolo26x-seg.npz') as archive:
            dets = archive[info['cam']]
        _box_cache[key] = dets
        if len(_box_cache) > 16:
            _box_cache.pop(next(iter(_box_cache)))
    row = info['rows'][j]
    return [float(v) for v in dets[row, 1:5]]


# ---------------------------------------------------------------- the film's view of it

def tracks(day, root=ROOT):
    """Drawn people as tracks the film's people model can hold, and their joins."""
    import day_movie
    start, _ = day_movie.clock(day, root)
    out, joins = {}, []
    for item in drawn(day, root)['items'].values():
        if item['deleted'] or item['kind'] != 'new' or not item['samples']:
            continue
        samples = sorted(item['samples'], key=lambda s: s[0])
        times = np.array([s[0] for s in samples], np.float64)
        boxes = [box_of([np.asarray(r).reshape(-1, 2) for r in s[1]]) for s in samples]
        feet = np.array([[(b[0] + b[2]) / 2, b[3]] for b in boxes])
        rows = np.zeros((len(samples), 10), np.float32)
        rows[:, 6:8] = feet
        key = 'd%d' % item['id']
        floor = day_movie._floor_share(item['cam'], rows)
        out[key] = {'key': key, 'clip': '_drawn', 'cam': item['cam'], 'rows': np.zeros(0, np.int64), 'times': times,
                    'first': float(times[0]), 'last': float(times[-1]), 'box_first': np.array(boxes[0]),
                    'box_last': np.array(boxes[-1]), 'enters_at_edge': day_movie._at_edge(boxes[0]),
                    'leaves_at_edge': day_movie._at_edge(boxes[-1]),
                    'shop': bool(floor > 0 or times[-1] - times[0] >= day_movie.SHOP_SECONDS),
                    'fp': 'drawn:%d' % item['id'], 'drawn': True}
        found = item.get('joins') or []
        for part in [found] if isinstance(found, str) else found:
            joins.append(('join', key, part))
    return out, joins


def view(day, root=ROOT, unit=1280):
    """What the page draws: every drawn outline at `unit` width, and how far carrying got."""
    k = unit / RAW_W
    out = []
    for item in drawn(day, root)['items'].values():
        if item['deleted']:
            continue
        out.append({'id': item['id'], 'track': 'd%d' % item['id'], 'cam': item['cam'], 'kind': item['kind'],
                    'replaces': item['replaces'], 'carry': item['carry'],
                    'samples': [[s[0], [[int(round(v * k)) for v in r] for r in s[1]]] for s in item['samples']]})
    return out


def register(app, root):
    from flask import jsonify, request
    import re
    here = root if callable(root) else (lambda: root)

    def valid(day):
        if not re.fullmatch(r'\d{8}', day):
            raise ValueError('Неверный день')

    @app.post('/api/movie/<day>/segment')
    def movie_segment(day):
        body = request.get_json(force=True) or {}
        try:
            valid(day)
            points = [p for p in body.get('points') or [] if isinstance(p, list) and len(p) == 3]
            if not points and not body.get('box'):
                raise ValueError('Щёлкните по человеку')
            return jsonify(segment(day, body.get('cam'), float(body.get('at')), points, body.get('box'), Path(here())))
        except (ValueError, TypeError, KeyError) as exc:
            return jsonify({'error': str(exc)}), 400

    @app.post('/api/movie/<day>/warm')
    def movie_warm(day):
        try:
            valid(day)
            warm(day, float((request.get_json(force=True) or {}).get('at')), Path(here()))
        except (ValueError, TypeError) as exc:
            return jsonify({'error': str(exc)}), 400
        return jsonify({'queued': True})

    @app.post('/api/movie/<day>/draw')
    def movie_draw(day):
        try:
            valid(day)
            return jsonify(draw(day, request.get_json(force=True) or {}, Path(here())))
        except (ValueError, TypeError) as exc:
            return jsonify({'error': str(exc)}), 400

    @app.post('/api/movie/<day>/undraw')
    def movie_undraw(day):
        try:
            valid(day)
            return jsonify(undraw(day, int((request.get_json(force=True) or {}).get('id')), Path(here())))
        except (ValueError, TypeError) as exc:
            return jsonify({'error': str(exc)}), 400
