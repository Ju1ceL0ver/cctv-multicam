"""The same person on both cameras: the owner links them, one key per question: /xcam.

A question is one moment of the day and one person camera 1 sees then; the answer is which of the
people camera 2 sees at that same moment is him (their number), none of them (0), or can't tell (X).
One answer gives a "same" pair and a "different" pair for everyone else on camera 2 -- training and
test pairs for joining the two cameras, and for the model's identity vectors.

People and times come from the night passes (day_movie.index: the teacher's tracks on the film's clock,
camera 2 already shifted onto camera 1's clock); frames from the raw recording (seg_compare.raw_at),
saved once at build time at 1280x720 so the page answers instantly.

usage: xcam.py build [N] [DAY ...]   -> data/xcam/samples.json, data/xcam/img/*.jpg
       xcam.py model teacher|CKPT [N_PER_DAY] DAY ...   days the night passes never labelled, moments spread over the
                                                day; people and outlines from the teacher + SAM (as the drafts), or a slot model

Storage: data/xcam/labels.json {sample: {match, at}} (match: 1..n camera-2 person, 0 none, -1 unsure),
every answer also in history.jsonl."""
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
FULL_W, FULL_H = 2560, 1440
W, H = 1280, 720
GAP = 20.0             # s between two moments of one day
MAX_CAM2 = 9           # people offered on camera 2 (keys 1..9)
NEAR_T = 0.4           # s: a track counts as seen at t if it has a detection this close
UNSURE = -1
DUP_IOU = 0.5          # two outlines overlapping this much on one camera are one person


def folder(root):
    return Path(root) / 'data' / 'xcam'


def _key(ident):
    return hashlib.md5(ident.encode()).hexdigest()


def samples(root):
    p = folder(root) / 'samples.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def labels(root):
    p = folder(root) / 'labels.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def seen_at(day, root, t):
    """{cam: [(track, box 2560 px)]} of the shop's people (not passers-by) seen at film time t."""
    import day_movie
    idx = day_movie.index(day, root)
    out = {'cam1': [], 'cam2': []}
    for name, tr in idx['tracks'].items():
        if not tr['shop'] or not (tr['first'] - NEAR_T <= t <= tr['last'] + NEAR_T):
            continue
        k = int(np.argmin(np.abs(tr['times'] - t)))
        if abs(tr['times'][k] - t) > NEAR_T:
            continue
        out[tr['cam']].append((name, tr['rows'][k], tr['clip'], tr['last'] - tr['first']))
    for cam in out:                          # the longer track first: it is the one a duplicate gives way to
        out[cam].sort(key=lambda r: -r[3])
    boxes, polys = {}, {}
    for cam, items in out.items():
        rows = []
        for name, row, clip, _ in items:
            key = (clip, cam)
            if key not in boxes:
                base = Path(root) / 'data' / 'raw_clips' / clip
                with np.load(base / ('dets_%s.npz' % day_movie.TAG)) as a:
                    boxes[key] = a[cam][:, 1:5].astype(np.float64)
                p = base / ('polys_%s.npz' % day_movie.TAG)
                if p.exists():
                    with np.load(p) as a:
                        polys[key] = (a[cam + '_pts'], a[cam + '_off'])
            poly = []
            if key in polys:
                pts, off = polys[key]
                poly = [[round(float(x) * W / FULL_W, 1), round(float(y) * H / FULL_H, 1)] for x, y in pts[off[row]:off[row + 1]]]
            rows.append((name, [round(float(v), 1) for v in boxes[key][row]], poly))
        out[cam] = dedup(rows)
    return out


def _shape(person, size=(H // 4, W // 4)):
    import cv2
    m = np.zeros(size, np.uint8)
    name, box, poly = person
    if poly and len(poly) >= 3:
        cv2.fillPoly(m, [np.round(np.array(poly, np.float32) / 4).astype(np.int32)], 1)
    else:
        x1, y1, x2, y2 = [int(v * W / FULL_W / 4) for v in box]
        m[y1:y2 + 1, x1:x2 + 1] = 1
    return m.astype(bool)


def dedup(rows, iou=DUP_IOU):
    """One person, one entry: two tracks whose outlines overlap at IoU >= iou are one person seen by two
    night passes (an early clip on top of the day's grid); the first (longer) one stays."""
    kept, shapes = [], []
    for r in rows:
        m = _shape(r)
        if all((m & k).sum() < iou * max(1, (m | k).sum()) for k in shapes):
            kept.append(r); shapes.append(m)
    return kept


def candidate_times(day, root):
    """Film times (every 0.5 s) when camera 1 and camera 2 both see somebody of the shop."""
    import day_movie
    idx = day_movie.index(day, root)
    spans = {'cam1': [], 'cam2': []}
    for tr in idx['tracks'].values():
        if tr['shop']:
            spans[tr['cam']].append((tr['first'], tr['last']))
    def covered(cam, t):
        return any(a <= t <= b for a, b in spans[cam])
    if not spans['cam1'] or not spans['cam2']:
        return []
    lo = max(min(a for a, _ in spans['cam1']), min(a for a, _ in spans['cam2']))
    hi = min(max(b for _, b in spans['cam1']), max(b for _, b in spans['cam2']))
    return [t for t in np.arange(lo, hi, 0.5) if covered('cam1', t) and covered('cam2', t)]


def build(root, n=300, days=('20260917', '20260918'), seed=0, frame_at=None, log=print):
    """n questions spread evenly over the days; keeps the samples already answered."""
    import cv2
    if frame_at is None:
        import seg_compare
        frame_at = lambda day, cam, t: seg_compare.raw_at(day, cam, t, root)[0]
    rng = random.Random(seed)
    f = folder(root)
    (f / 'img').mkdir(parents=True, exist_ok=True)
    old, answered = samples(root), labels(root)
    out = {k: v for k, v in old.items() if k in answered}
    per_day = max(1, n // len(days))
    for day in days:
        times = candidate_times(day, root)
        rng.shuffle(times)
        chosen = []
        for t in times:
            if all(abs(t - c) >= GAP for c in chosen):
                chosen.append(t)
            if len(chosen) >= per_day:
                break
        made = 0
        for t in sorted(chosen):
            seen = seen_at(day, root, t)
            if not seen['cam1'] or not seen['cam2']:
                continue
            tag = '%s_%07d' % (day, int(round(t * 10)))
            frames = {}
            for cam in ('cam1', 'cam2'):
                p = f / 'img' / ('%s_%s.jpg' % (tag, cam))
                if not p.exists():
                    img = frame_at(day, cam, t)
                    if img is None:
                        break
                    cv2.imwrite(str(p), cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
                frames[cam] = p.name
            if len(frames) < 2:
                continue
            cam2 = seen['cam2'][:MAX_CAM2]
            for k, (track, box, poly) in enumerate(seen['cam1']):
                sid = '%s_%d' % (tag, k)
                if sid in out:
                    continue
                out[sid] = {'day': day, 't': round(float(t), 2), 'frames': frames,
                            'cam1': {'track': track, 'box': box, 'poly': poly},
                            'cam2': [{'track': tr, 'box': b, 'poly': pl} for tr, b, pl in cam2]}
            made += 1
        log('%s: %d moments' % (day, made))
    from storage import atomic_json
    atomic_json(f / 'samples.json', out)
    return out


def model_people(model, frames, dev, img_path, day, cam, thr=0.5):
    """People the slot model finds on a saved 1280x720 frame: [(name, box 2560 px, outline 1280 px)]."""
    import cv2
    import torch
    import torch.nn.functional as F
    import slot_data
    import eval_slot_heads as E
    bgs = E.backgrounds(day, cam)
    s = slot_data.load(frames, 'xcam', False, image=img_path, label=None,
                       background=bgs[len(bgs) // 2][1] if bgs else None, cam_day=(day, cam))
    from train_slots import predict, dedup as dedup_masks
    x = torch.from_numpy(s['x'])[None].to(dev)
    with torch.autocast('cuda', torch.bfloat16, enabled=dev == 'cuda'):
        ppl = predict(model, x, thr)[0]
    logits = [F.interpolate(p[1][None, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy() for p in ppl]
    keep = dedup_masks([lg > 0 for lg in logits], [p[0] for p in ppl])
    rows = []
    for k in keep:                           # the logits upsampled smoothly, then cut: no stairs
        m = (logits[k] > 0).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) < 150:
            continue
        x, y, w, h = cv2.boundingRect(c)
        poly = [[float(p[0][0]), float(p[0][1])] for p in cv2.approxPolyDP(c, 1.0, True)]
        rows.append(('m:%s:%d' % (cam, k), [x * FULL_W / W, y * FULL_H / H, (x + w) * FULL_W / W, (y + h) * FULL_H / H], poly))
    return dedup(rows)


class Teacher:
    """The drafts' own teacher on a full 2560x1440 frame: yolo26x-seg @1536 finds the people, SAM 2.1 Large
    outlines them from their boxes (fp16 autocast: 2.3x faster, masks IoU 0.9995), the nearer person drawn
    last so two people standing close stay two."""

    def __init__(self):
        import seg_compare
        from ultralytics import SAM, YOLO
        self.yolo = YOLO(r'C:\Users\ArykovAA\cctv_ai\retail_analytics\models\yolo26x-seg.pt', task='segment')
        self.sam = SAM(seg_compare.SAM_WEIGHTS)

    def __call__(self, frame, cam):
        import cv2
        import torch
        r = self.yolo.predict(frame, imgsz=1536, conf=0.25, classes=[0], verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        if not len(boxes):
            return []
        with torch.autocast('cuda', dtype=torch.float16):
            res = self.sam.predict(frame, bboxes=boxes.tolist(), imgsz=1024, verbose=False)[0]
        masks = (res.masks.data.float().cpu().numpy() > 0.5) if res.masks is not None else []
        labels = np.zeros((H, W), np.uint8)
        for k in np.argsort(boxes[:, 3]):                        # the nearer person is painted last
            if k < len(masks) and k < 254:
                m = cv2.resize(masks[k].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                labels[m] = int(k) + 1
        rows = []
        for v in np.unique(labels):
            if not v:
                continue
            cs, _ = cv2.findContours((labels == v).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cs, key=cv2.contourArea)
            if cv2.contourArea(c) < 150:
                continue
            x, y, w, h = cv2.boundingRect(c)
            poly = [[float(p[0][0]), float(p[0][1])] for p in cv2.approxPolyDP(c, 1.0, True)]
            rows.append(('s:%s:%d' % (cam, int(v)), [x * FULL_W / W, y * FULL_H / H, (x + w) * FULL_W / W, (y + h) * FULL_H / H], poly))
        return dedup(rows)


def build_model(root, ckpt, days, per_day=100, spacing=60.0, seed=0, frame_at=None, log=print, detector='teacher'):
    """Questions for days without the night passes' tracks: moments every `spacing` s at least, spread
    over the whole day (fewer repeats of the same customers), people and outlines from the slot model."""
    import cv2
    import torch
    import day_movie
    import slot_data
    from slot_model import SlotModel
    if frame_at is None:
        import seg_compare
        frame_at = lambda day, cam, t: seg_compare.raw_at(day, cam, t, root)[0]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames = slot_data.Frames()
    if detector == 'teacher':
        teacher = Teacher()
    else:
        ck = torch.load(ckpt, map_location='cpu')
        model = SlotModel(ck.get('backbone', 'deimv2_vit_tiny'), pretrained=False, mean=frames.mean, std=frames.std, teacher_dims=frames.teacher_dims)
        model.load_state_dict(ck.get('ema', ck['model']), strict=False)
        model = model.to(dev).eval()
    rng = random.Random(seed)
    f = folder(root)
    (f / 'img').mkdir(parents=True, exist_ok=True)
    out = samples(root)
    for day in days:
        start, end = day_movie.clock(day, root)
        span = end - start
        slots = list(np.arange(30.0, span - 30.0, spacing))
        rng.shuffle(slots)
        made = tried = 0
        for t in slots:
            if made >= per_day or tried >= per_day * 6:
                break
            tried += 1
            t = float(t + rng.uniform(0, spacing / 2))
            tag = '%s_%07d' % (day, int(round(t * 10)))
            names, people = {}, {}
            for cam in ('cam1', 'cam2'):
                p = f / 'img' / ('%s_%s.jpg' % (tag, cam))
                img = frame_at(day, cam, t)
                if img is None:
                    break
                cv2.imwrite(str(p), cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
                names[cam] = p.name
                people[cam] = teacher(img, cam) if detector == 'teacher' else model_people(model, frames, dev, p, day, cam)
                if not people[cam]:
                    break
            if len(people) < 2 or not people['cam1'] or not people['cam2']:
                for cam in ('cam1', 'cam2'):                      # nobody on one camera: keep no pictures
                    q = f / 'img' / ('%s_%s.jpg' % (tag, cam))
                    if q.exists() and not any(s_['frames'].get(cam) == q.name for s_ in out.values()):
                        q.unlink()
                continue
            for k, (track, box, poly) in enumerate(people['cam1']):
                sid = '%s_%d' % (tag, k)
                if sid in out:
                    continue
                out[sid] = {'day': day, 't': round(t, 2), 'frames': names, 'source': detector,
                            'cam1': {'track': '%s@%s' % (track, tag), 'box': box, 'poly': poly},
                            'cam2': [{'track': '%s@%s' % (tr, tag), 'box': b, 'poly': pl} for tr, b, pl in people['cam2'][:MAX_CAM2]]}
            made += 1
        log('%s: %d moments (%d looked at)' % (day, made, tried))
        from storage import atomic_json
        atomic_json(f / 'samples.json', out)
    return out


def answer(root, sid, match):
    """match: 1..n (that camera-2 person), 0 (none of them), -1 (can't tell), None (take it back)."""
    from storage import atomic_json, file_lock
    f = folder(root)
    f.mkdir(parents=True, exist_ok=True)
    with file_lock(str(f / 'labels.json') + '.lock'):
        r = labels(root)
        now = time.strftime('%Y-%m-%dT%H:%M:%S')
        if match is None:
            r.pop(sid, None)
        else:
            r[sid] = {'match': int(match), 'at': now}
        atomic_json(f / 'labels.json', r)
        with open(f / 'history.jsonl', 'a', encoding='utf-8') as h:
            h.write(json.dumps({'id': sid, 'match': match, 'at': now}) + '\n')
    return r


def pairs(root):
    """The answers as pairs of tracks: [(day, t, cam1 track, cam2 track, same: bool)]."""
    ss, r = samples(root), labels(root)
    out = []
    for sid, v in r.items():
        s = ss.get(sid)
        if s is None or v['match'] == UNSURE:
            continue
        chosen = s['cam2'][v['match'] - 1] if v['match'] > 0 else None
        cm = _shape((None, chosen['box'], chosen.get('poly'))) if chosen else None
        for k, p in enumerate(s['cam2'], 1):
            if chosen is not None and k != v['match']:
                m = _shape((None, p['box'], p.get('poly')))
                if (m & cm).sum() >= DUP_IOU * max(1, (m | cm).sum()):
                    continue                 # a duplicate of the chosen one (questions built before dedup)
            out.append((s['day'], s['t'], s['cam1']['track'], p['track'], v['match'] == k))
    return out


def _to_small(box):
    return [v * W / FULL_W for v in box]


COLOURS = [(66, 197, 245), (80, 220, 100), (245, 120, 66), (220, 90, 220), (70, 70, 240), (240, 220, 70),
           (160, 100, 255), (60, 200, 200), (200, 160, 90)]


def _outline(img, person, colour, thick):
    """The person's outline (the teacher's mask contour, 1280x720 px); his box when there is none."""
    import cv2
    if person.get('poly') and len(person['poly']) >= 3:
        pts = np.array(person['poly'], np.float32).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, colour, thick, cv2.LINE_AA)
    else:
        x1, y1, x2, y2 = [int(v) for v in _to_small(person['box'])]
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)


def render(root, s, what, k=None):
    """JPEG bytes. 'frame1'/'frame2': the camera's frame, 640 wide, the asked person (cam 1) or every
    offered person outlined in his own colour and numbered (cam 2); 'crop1': the asked person; 'crop2'
    with k: camera-2 person k -- only that person outlined, so two people standing close are told apart."""
    import cv2
    f = folder(root) / 'img'
    cam = 'cam1' if what.endswith('1') else 'cam2'
    img = cv2.imread(str(f / s['frames'][cam]))
    if what.startswith('frame'):
        out = img.copy()
        people = [s['cam1']] if cam == 'cam1' else s['cam2']
        for i, p in enumerate(people, 1):
            colour = COLOURS[0] if cam == 'cam1' else COLOURS[(i - 1) % len(COLOURS)]
            _outline(out, p, colour, 3)
            if cam == 'cam2':
                x1, y1 = [int(v) for v in _to_small(p['box'])[:2]]
                cv2.rectangle(out, (x1, max(0, y1 - 34)), (x1 + 30, max(34, y1)), colour, -1)
                cv2.putText(out, str(i), (x1 + 6, max(28, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        out = cv2.resize(out, (640, 360), interpolation=cv2.INTER_AREA)
        q = 82
    else:
        p = s['cam1'] if cam == 'cam1' else s['cam2'][k - 1]
        colour = COLOURS[0] if cam == 'cam1' else COLOURS[(k - 1) % len(COLOURS)]
        x1, y1, x2, y2 = _to_small(p['box'])
        w, h = x2 - x1, y2 - y1
        X1, Y1 = max(0, int(x1 - 0.25 * w)), max(0, int(y1 - 0.08 * h))
        X2, Y2 = min(W, int(x2 + 0.25 * w)), min(H, int(y2 + 0.08 * h))
        scale = 300 / max(1, Y2 - Y1)
        big = cv2.resize(img[Y1:Y2, X1:X2], (max(1, int((X2 - X1) * scale)), 300), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        shifted = dict(p, poly=[[(x - X1) * scale, (y - Y1) * scale] for x, y in p.get('poly') or []],
                       box=[(v - o) * scale * FULL_W / W for v, o in zip(_to_small(p['box']), (X1, Y1, X1, Y1))])
        _outline(big, shifted, colour, 2)
        out = big
        q = 90
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


def register(app, root):
    from flask import Response, abort, jsonify, render_template, request
    here = root if callable(root) else (lambda: root)

    def sample(sid):
        if not re.fullmatch(r'[0-9_]{1,40}', sid):
            abort(404)
        s = samples(here()).get(sid)
        if s is None:
            abort(404)
        return s

    @app.get('/xcam')
    def xcam_page():
        return render_template('xcam.html')

    @app.get('/api/xcam')
    def xcam_list():
        ss, r = samples(here()), labels(here())
        todo = sorted((i for i in ss if i not in r), key=_key)
        done = sorted((i for i in r if i in ss), key=lambda i: r[i]['at'])
        n = int(request.args.get('n', 200))
        meta = {i: {'day': ss[i]['day'], 't': ss[i]['t'], 'offered': len(ss[i]['cam2'])} for i in todo[:n] + done[-50:]}
        same = sum(1 for v in r.values() if v['match'] > 0)
        return jsonify({'todo': todo[:n], 'recent': done[-50:], 'answers': {i: r[i]['match'] for i in done[-50:]}, 'meta': meta,
                        'total': len(ss), 'done': len(r), 'same': same,
                        'none': sum(1 for v in r.values() if v['match'] == 0), 'unsure': sum(1 for v in r.values() if v['match'] == UNSURE)})

    @app.post('/api/xcam')
    def xcam_answer():
        body = request.get_json(force=True) or {}
        sid, match = str(body.get('id', '')), body.get('match')
        s = sample(sid)
        if match is not None and (not isinstance(match, int) or isinstance(match, bool) or not (UNSURE <= match <= len(s['cam2']))):
            return jsonify({'error': 'Ответ — номер человека на камере 2, 0 — никого из них, X — не понять'}), 400
        try:
            r = answer(here(), sid, match)
        except TimeoutError as exc:
            return jsonify({'error': str(exc)}), 503
        return jsonify({'ok': True, 'done': len(r)})

    @app.get('/xcam-img/<sid>/<what>')
    @app.get('/xcam-img/<sid>/<what>/<int:k>')
    def xcam_img(sid, what, k=None):
        s = sample(sid)
        if what not in ('frame1', 'frame2', 'crop1', 'crop2') or (what == 'crop2' and not (k and 1 <= k <= len(s['cam2']))):
            abort(404)
        resp = Response(render(here(), s, what, k), mimetype='image/jpeg')
        resp.headers['Cache-Control'] = 'private, max-age=3600'
        return resp


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'build':
        t0 = time.time()
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        days = tuple(sys.argv[3:]) or ('20260917', '20260918')
        got = build(str(ROOT), n, days)
        print('%d questions from %s, %.0f s' % (len(got), ', '.join(days), time.time() - t0))
    elif len(sys.argv) >= 4 and sys.argv[1] == 'model':
        t0 = time.time()
        rest = sys.argv[3:]
        per = int(rest.pop(0)) if rest and len(rest[0]) < 8 else 100          # a count, not a day (YYYYMMDD)
        days = rest
        got = build_model(str(ROOT), sys.argv[2], days, per, detector='model' if sys.argv[2].endswith('.pt') else 'teacher')
        print('%d questions in all, %.0f s' % (len(got), time.time() - t0))
    else:
        print(__doc__)
