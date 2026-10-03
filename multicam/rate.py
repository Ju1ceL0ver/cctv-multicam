"""The owner scores machine drafts 0-5, one key per frame: /rate.

Drafts are the teacher's people outlined by SAM (data/seg_datasets/*/drafts/<id>.png, label maps,
0 = nobody). Scores are the ground truth for a light quality model that will pick only the best
drafts for distillation, so wrong drafts do not breed wrong students. New drafts are picked up by
a rescan once a minute. Order: a fixed shuffle (a hash of the id), so the frames are spread over
days and cameras, and the same after a reload.

Storage: data/rate/ratings.json {id: {score, at}}, every answer also appended to history.jsonl."""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

W, H = 1280, 720
RESCAN_S = 60
PALETTE = [(66, 135, 245), (245, 66, 93), (66, 245, 149), (245, 197, 66), (179, 66, 245), (66, 230, 245),
           (245, 132, 66), (156, 245, 66), (245, 66, 212), (66, 90, 245), (245, 245, 66), (66, 245, 90)]
_index = {'at': 0.0, 'root': None, 'items': {}}


def _key(ident):
    return hashlib.md5(ident.encode()).hexdigest()


def index(root, force=False):
    """{id: (image path, draft path)} of every draft that has its frame, rescanned once a minute."""
    root = Path(root)
    if not force and _index['root'] == str(root) and time.time() - _index['at'] < RESCAN_S:
        return _index['items']
    base = root / 'data' / 'seg_datasets'
    images = {}
    for pattern in ('*/*/images/*.jpg', '*/images/*.jpg'):
        for p in base.glob(pattern):
            images.setdefault(p.stem, p)
    items = {}
    for p in base.glob('*/drafts/*.png'):
        if p.stem in images:
            items[p.stem] = (images[p.stem], p)
    _index.update(at=time.time(), root=str(root), items=items)
    return items


def ratings(root):
    p = Path(root) / 'data' / 'rate' / 'ratings.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def rate(root, ident, score):
    """score 0..5, or None to take the answer back."""
    from storage import atomic_json, file_lock
    folder = Path(root) / 'data' / 'rate'
    folder.mkdir(parents=True, exist_ok=True)
    with file_lock(str(folder / 'ratings.json') + '.lock'):
        r = ratings(root)
        if score is None:
            r.pop(ident, None)
        else:
            r[ident] = {'score': int(score), 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        atomic_json(folder / 'ratings.json', r)
        with open(folder / 'history.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': ident, 'score': score, 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}) + '\n')
    return r


def overlay(image_path, draft_path):
    """The frame with every person filled in its own colour and outlined, as JPEG bytes."""
    import cv2
    img = cv2.imread(str(image_path))
    lab = cv2.imread(str(draft_path), cv2.IMREAD_UNCHANGED)
    if lab.ndim == 3:
        lab = lab[:, :, 0]
    if lab.shape != img.shape[:2]:
        lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = img.copy()
    people = [v for v in np.unique(lab) if v]
    for k, v in enumerate(people):
        m = lab == v
        colour = np.array(PALETTE[k % len(PALETTE)], np.float32)
        out[m] = (0.55 * out[m] + 0.45 * colour).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, tuple(int(c) for c in colour), 2)
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes(), len(people)


def register(app, root):
    from flask import Response, abort, jsonify, render_template, request, send_file
    here = root if callable(root) else (lambda: root)

    def item(ident):
        if not re.fullmatch(r'[A-Za-z0-9_]{1,80}', ident):
            abort(404)
        it = index(here()).get(ident)
        if it is None:
            abort(404)
        return it

    @app.get('/rate')
    def rate_page():
        return render_template('rate.html')

    @app.get('/api/rate')
    def rate_list():
        """The next unrated frames in the fixed order, the last rated ones (to go back), the counts."""
        items, r = index(here()), ratings(here())
        todo = sorted((i for i in items if i not in r), key=_key)
        done = sorted((i for i in r if i in items), key=lambda i: r[i]['at'])
        hist = [0] * 6
        for v in r.values():
            hist[v['score']] += 1
        return jsonify({'todo': todo[:int(request.args.get('n', 200))], 'recent': done[-50:],
                        'ratings': {i: r[i]['score'] for i in done[-50:]},
                        'total': len(items), 'rated': len(r), 'hist': hist})

    @app.post('/api/rate')
    def rate_answer():
        body = request.get_json(force=True) or {}
        ident, score = str(body.get('id', '')), body.get('score')
        item(ident)
        if score is not None and (not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 5):
            return jsonify({'error': 'Оценка — целое от 0 до 5'}), 400
        try:
            r = rate(here(), ident, score)
        except TimeoutError as exc:
            return jsonify({'error': str(exc)}), 503
        return jsonify({'ok': True, 'rated': len(r)})

    @app.get('/rate-img/<ident>/<what>')
    def rate_img(ident, what):
        image, draft = item(ident)
        if what == 'raw':
            return send_file(image, mimetype='image/jpeg', max_age=3600)
        if what != 'drawn':
            abort(404)
        data, n = overlay(image, draft)
        resp = Response(data, mimetype='image/jpeg')
        resp.headers['Cache-Control'] = 'private, max-age=600'
        resp.headers['X-People'] = str(n)
        return resp
