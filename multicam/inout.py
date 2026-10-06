"""Inside the shop or behind the glass: the owner labels single people, one key each: /inout.

The question for a tiny classifier that will see only a crop of the person with some room around
it: can "inside / outside" be told from the look alone (the glass haze, reflections, the frame
of the door), with no feet and no full frame? These labels answer it and are its training set.

People come from the machine drafts (teacher + SAM label maps over 1280x720 frames of every
recorded day and both cameras). `build` picks them spread over day x camera, half of them with
the feet near an edge of the traced shop floor (the doubtful ones), half at random.

usage: inout.py build [N]      -> data/inout/samples.json

Storage: data/inout/labels.json {sample: {label, at}}, every answer also in history.jsonl.
The full frame shown next to the crop is for the owner, to answer right; the classifier will
not get it."""
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
W, H = 1280, 720
MIN_PX = 60            # the specks nobody meant, as in gold.py
MIN_H = 24             # px of height in the 1280 frame
NEAR = 40              # px from an edge of the floor: a doubtful place
CTX_W, CTX_H = 1.0, 0.25   # room around the person: +100% of its width, +25% of its height, split evenly
LABELS = {1: 'внутри', 2: 'снаружи', 3: 'в проёме двери', 4: 'отражение', 5: 'не человек', 0: 'не понять'}


def _key(ident):
    return hashlib.md5(ident.encode()).hexdigest()


def folder(root):
    return Path(root) / 'data' / 'inout'


def samples(root):
    p = folder(root) / 'samples.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def labels(root):
    p = folder(root) / 'labels.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def people(lab):
    """[(value, box x1 y1 x2 y2, foot x y)] of every person in a label map."""
    out = []
    for v in np.unique(lab):
        if not v:
            continue
        ys, xs = np.nonzero(lab == v)
        if len(ys) < MIN_PX or ys.max() - ys.min() < MIN_H:
            continue
        low = ys >= ys.max() - 2
        out.append((int(v), [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    [int(xs[low].mean()), int(ys.max())]))
    return out


def floor_distance(cam):
    """Signed px distance to the edge of the traced floor at 1280x720: + on the floor, - off it."""
    import cv2
    import rooms
    m = cv2.resize(rooms.mask(cam), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    inside = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
    return np.where(m, inside, -outside)


def build(root, n=2000, seed=0):
    """Pick n people spread over day x camera, half near an edge of the floor. Keeps the samples
    that already have an answer, so a rebuild never loses the owner's work."""
    import cv2
    import rate
    rng = random.Random(seed)
    frames = rate.index(root, force=True)
    strata = {}
    for ident in sorted(frames):
        parts = ident.split('_')
        cam = next((p for p in parts if p in ('cam1', 'cam2')), None)
        if cam and parts[0].isdigit():
            strata.setdefault((parts[0], cam), []).append(ident)
    for v in strata.values():
        rng.shuffle(v)
    dist = {}
    old, answered = samples(root), labels(root)
    out = {k: v for k, v in old.items() if k in answered}
    want = {True: n // 2, False: n - n // 2}
    have = {True: sum(v['near'] for v in out.values()), False: sum(not v['near'] for v in out.values())}
    order = sorted(strata)
    while any(strata[s] for s in order) and (have[True] < want[True] or have[False] < want[False]):
        for s in order:
            if not strata[s]:
                continue
            ident = strata[s].pop()
            day, cam = s
            lab = cv2.imread(str(frames[ident][1]), cv2.IMREAD_UNCHANGED)
            if lab is None:
                continue
            if lab.ndim == 3:
                lab = lab[:, :, 0]
            if lab.shape != (H, W):
                lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
            if cam not in dist:
                dist[cam] = floor_distance(cam)
            ppl = people(lab)
            rng.shuffle(ppl)
            for v, box, foot in ppl[:2]:          # at most two from a frame: more frames, more variety
                d = float(dist[cam][min(H - 1, foot[1]), min(W - 1, foot[0])])
                near = abs(d) < NEAR
                sid = '%s_p%d' % (ident, v)
                if sid in out or have[near] >= want[near]:
                    continue
                out[sid] = {'frame': ident, 'value': v, 'day': day, 'cam': cam, 'box': box, 'foot': foot,
                            'floor_px': round(d, 1), 'near': near}
                have[near] += 1
    from storage import atomic_json
    folder(root).mkdir(parents=True, exist_ok=True)
    atomic_json(folder(root) / 'samples.json', out)
    return out


def answer(root, sid, label):
    """label from LABELS, or None to take the answer back."""
    from storage import atomic_json, file_lock
    f = folder(root)
    f.mkdir(parents=True, exist_ok=True)
    with file_lock(str(f / 'labels.json') + '.lock'):
        r = labels(root)
        now = time.strftime('%Y-%m-%dT%H:%M:%S')
        if label is None:
            r.pop(sid, None)
        else:
            r[sid] = {'label': int(label), 'at': now}
        atomic_json(f / 'labels.json', r)
        with open(f / 'history.jsonl', 'a', encoding='utf-8') as h:
            h.write(json.dumps({'id': sid, 'label': label, 'at': now}) + '\n')
    return r


def crop_box(box, shape=(H, W)):
    """The crop the classifier will see: the person with room around it, clipped to the frame."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1, X2 = x1 - CTX_W * w / 2, x2 + CTX_W * w / 2
    Y1, Y2 = y1 - CTX_H * h / 2, y2 + CTX_H * h / 2
    return [max(0, int(X1)), max(0, int(Y1)), min(shape[1], int(np.ceil(X2))), min(shape[0], int(np.ceil(Y2)))]


def render(root, s, what, outline=True):
    """JPEG bytes: 'crop' (the classifier's view, the person outlined for the owner) or 'frame'
    (the whole frame, small, the person boxed)."""
    import cv2
    import rate
    image, draft = rate.index(root)[s['frame']]
    img = cv2.imread(str(image))
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H))
    if what == 'crop':
        X1, Y1, X2, Y2 = crop_box(s['box'])
        out = img[Y1:Y2, X1:X2].copy()
        if outline:
            lab = cv2.imread(str(draft), cv2.IMREAD_UNCHANGED)
            if lab.ndim == 3:
                lab = lab[:, :, 0]
            if lab.shape != (H, W):
                lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
            m = (lab[Y1:Y2, X1:X2] == s['value']).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cs, -1, (66, 197, 245), 1)
        quality = 92
    else:
        x1, y1, x2, y2 = s['box']
        out = img.copy()
        cv2.rectangle(out, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (66, 197, 245), 3)
        out = cv2.resize(out, (640, 360), interpolation=cv2.INTER_AREA)
        quality = 80
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def register(app, root):
    from flask import Response, abort, jsonify, render_template, request
    here = root if callable(root) else (lambda: root)

    def sample(sid):
        if not re.fullmatch(r'[A-Za-z0-9_]{1,90}', sid):
            abort(404)
        s = samples(here()).get(sid)
        if s is None:
            abort(404)
        return s

    @app.get('/inout')
    def inout_page():
        return render_template('inout.html', labels=LABELS)

    @app.get('/api/inout')
    def inout_list():
        """The next unlabelled people in a fixed shuffle (days and cameras mixed), the last
        answered ones (to go back), the counts."""
        ss, r = samples(here()), labels(here())
        todo = sorted((i for i in ss if i not in r), key=lambda i: (0 if ss[i].get('door') else 1, -ss[i].get('doubt', 0), _key(i)))   # 06.10: the door batch first, the least sure first
        done = sorted((i for i in r if i in ss), key=lambda i: r[i]['at'])
        hist = {str(k): 0 for k in LABELS}
        for v in r.values():
            hist[str(v['label'])] = hist.get(str(v['label']), 0) + 1
        meta = {i: {k: ss[i][k] for k in ('day', 'cam', 'near')} for i in todo[:200] + done[-50:]}
        return jsonify({'todo': todo[:int(request.args.get('n', 200))], 'recent': done[-50:],
                        'labels': {i: r[i]['label'] for i in done[-50:]}, 'meta': meta,
                        'total': len(ss), 'done': len(r), 'hist': hist})

    @app.post('/api/inout')
    def inout_answer():
        body = request.get_json(force=True) or {}
        sid, label = str(body.get('id', '')), body.get('label')
        sample(sid)
        if label is not None and (not isinstance(label, int) or isinstance(label, bool) or label not in LABELS):
            return jsonify({'error': 'Ответ — одна из клавиш 0–5'}), 400
        try:
            r = answer(here(), sid, label)
        except TimeoutError as exc:
            return jsonify({'error': str(exc)}), 503
        return jsonify({'ok': True, 'done': len(r)})

    @app.get('/inout-img/<sid>/<what>')
    def inout_img(sid, what):
        s = sample(sid)
        if what not in ('crop', 'plain', 'frame'):
            abort(404)
        data = render(here(), s, 'frame' if what == 'frame' else 'crop', outline=what == 'crop')
        resp = Response(data, mimetype='image/jpeg')
        resp.headers['Cache-Control'] = 'private, max-age=3600'
        return resp


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'build':
        t0 = time.time()
        got = build(str(ROOT), int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
        near = sum(v['near'] for v in got.values())
        days = sorted({v['day'] for v in got.values()})
        print('%d people (%d near an edge of the floor) from %s, %.0f s' % (len(got), near, ', '.join(days), time.time() - t0))
    else:
        print(__doc__)
