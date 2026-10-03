"""RE-LABELLING from scratch (/keep2). Keep or delete: the owner decides, one person of the SAM 3.1 drafts at a time: /keep.

Why: the owner's own frames (data/fix) show that he throws out ~17 % of SAM's people -- the ones in the neighbouring
shops, the specks, the ghosts -- and position alone cannot tell them apart. A small network "keep / delete" trained on
his decisions (his 1849 people of /fix, plus these) will do it for the thousands of frames he cannot look at. The
criterion for its threshold: not one person he keeps is thrown out (on his test).

What he sees: the person zoomed with room around it and its mask filled, and the whole frame with every person
outlined and this one filled yellow, so it is clear what is where.

Candidates: the people of data/sam31_stills/drafts/<frame>.png (SAM 3.1 on single frames, maps 1280x720), at least
MIN_PX pixels (300 px of the 2176x1224 frame = 104 here). Half of them in the upper part of the frame (where he
deleted most: the glass wall to the gallery and the shops across), half anywhere; at most two from a frame.

usage: keep.py build [N]      -> data/keep/samples.json
       keep.py review [THR]   -> the kept people the network is sure about, asked again (ids ..._r)

Storage: data/keep/labels.json {sample: {label, at}}; every answer also in history.jsonl."""
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
W, H = 1280, 720
MIN_PX = 104                       # = 300 px of the 2176 x 1224 frame
TOP = 0.40                         # "upper part": the foot above this share of the frame height
LABELS = {1: 'оставить', 2: 'удалить (другое)', 3: 'соседний магазин', 4: 'не человек (предмет, плакат, тележка)',
          5: 'плохая маска (рваная, неровная, пятна)', 6: 'слиты двое', 7: 'только часть человека (обрывок)',
          8: 'дубль, призрак, отражение', 9: 'плакат, реклама, манекен', 0: 'не понять'}
# what a training run does with the person: TAKE as it is, DELETE (the place is background: nobody there / not a person of
# this shop), IGNORE (a real person with a bad outline: no loss on it at all -- neither "person" nor "nobody")
TAKE = (1,)
DELETE = (2, 4, 8, 9)
IGNORE = (3, 5, 6, 7, 0)
SCALE = (2176 * 1224) / (W * H)


def _key(s):
    return hashlib.md5(s.encode()).hexdigest()


def folder(root):
    return Path(root) / 'data' / 'keep2'


def samples(root):
    p = folder(root) / 'samples.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def priority(root):
    p = folder(root) / 'priority.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def labels(root):
    p = folder(root) / 'labels.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def _map(path):
    import cv2
    lab = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if lab is None:
        return None
    if lab.ndim == 3:
        lab = lab[:, :, 0]
    if lab.shape != (H, W):
        lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    return lab


def people(lab):
    """[(value, area, box x1 y1 x2 y2, foot x y)] of every person of a map with at least MIN_PX pixels."""
    out = []
    for v in np.unique(lab):
        if not v:
            continue
        ys, xs = np.nonzero(lab == v)
        if len(ys) < MIN_PX:
            continue
        low = ys >= ys.max() - 2
        out.append((int(v), int(len(ys)), [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    [int(xs[low].mean()), int(ys.max())]))
    return out


def draft_path(root, ident):
    return Path(root) / 'data' / 'sam31_stills' / 'drafts' / ('%s.png' % ident)


def paths(root, s):
    """(image, label map) of a sample: SAM 3.1's single frames, or a window frame of /fix (full size, its draft)."""
    if s.get('src') == 'fix':
        f = Path(root) / 'data' / 'fix'
        return f / ('%s.jpg' % s['frame']), f / ('%s_init.png' % s['frame'])
    import rate
    return rate.index(root)[s['frame']][0], draft_path(root, s['frame'])


def build_all(root):
    """Everyone again, from scratch: every person of the /keep samples (single frames of SAM 3.1) and every person of the window
    frames of /fix (full size, SAM's draft), at least MIN_PX. The old answers are not shown. Answered samples are kept."""
    from storage import atomic_json
    out = {}
    old = json.load(open(Path(root) / 'data' / 'keep' / 'samples.json', encoding='utf-8'))
    for sid, s in old.items():
        if not s.get('review'):
            out[sid] = dict(s, rl='keep')
    f = Path(root) / 'data' / 'fix'
    st = json.load(open(f / 'state.json', encoding='utf-8'))
    items = {x['id']: x for x in json.load(open(f / 'manifest.json', encoding='utf-8'))['items']}
    for fid, v in st.items():
        if v.get('verdict') != 'take' or fid not in items:
            continue
        lab = _map(f / ('%s_init.png' % fid))
        if lab is None:
            continue
        it = items[fid]
        for val, area, box, foot in people(lab):
            out['fx_%s_p%d' % (fid, val)] = {'frame': fid, 'value': val, 'day': (it.get('tag') or fid)[:8], 'cam': it['cam'], 'box': box,
                                              'foot': foot, 'area': area, 'area_full': int(area * SCALE), 'top': foot[1] < TOP * H,
                                              'src': 'fix', 'rl': 'fix'}
    folder(root).mkdir(parents=True, exist_ok=True)
    keep_answers = labels(root)
    out = {k: v for k, v in out.items()}
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


def render(root, s, what):
    """JPEG bytes: 'zoom' (the person with room around it, mask filled) or 'frame' (the whole frame, everyone outlined,
    this person filled yellow and boxed)."""
    import cv2
    image, mp = paths(root, s)
    img = cv2.imread(str(image))
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H))
    lab = _map(mp)
    me = lab == s['value']
    x1, y1, x2, y2 = s['box']
    yellow = np.array((66, 197, 245), np.float32)
    if what == 'zoom':
        w, h = x2 - x1, y2 - y1
        mx, my = max(60, int(0.6 * w)), max(60, int(0.35 * h))
        X1, Y1, X2, Y2 = max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)
        crop = img[Y1:Y2, X1:X2].copy()
        m = me[Y1:Y2, X1:X2]
        f = 640.0 / max(crop.shape[:2])
        f = min(f, 6.0)
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        m = cv2.resize(m.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        crop[m] = (0.62 * crop[m] + 0.38 * yellow).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, cs, -1, (66, 197, 245), 2)
        out, q = crop, 90
    else:
        out = img.copy()
        for v in np.unique(lab):
            if v and v != s['value']:
                cs, _ = cv2.findContours((lab == v).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, cs, -1, (235, 235, 235), 1)
        out[me] = (0.55 * out[me] + 0.45 * yellow).astype(np.uint8)
        cs, _ = cv2.findContours(me.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (66, 197, 245), 2)
        cv2.rectangle(out, (max(0, x1 - 8), max(0, y1 - 8)), (min(W - 1, x2 + 8), min(H - 1, y2 + 8)), (66, 197, 245), 2)
        q = 84
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


def register(app, root):
    from flask import Response, abort, jsonify, render_template, request
    here = root if callable(root) else (lambda: root)

    def sample(sid):
        if not re.fullmatch(r'[A-Za-z0-9_]{1,100}', sid):
            abort(404)
        s = samples(here()).get(sid)
        if s is None:
            abort(404)
        return s

    @app.get('/keep2')
    def keep2_page():
        return render_template('keep2.html', labels=LABELS)

    @app.get('/api/keep2')
    def keep2_list():
        ss, r = samples(here()), labels(here())
        pri = priority(here())
        todo = sorted((i for i in ss if i not in r), key=lambda i: (i not in pri, pri.get(i, 0), _key(i)))
        done = sorted((i for i in r if i in ss), key=lambda i: r[i]['at'])
        hist = {str(k): 0 for k in LABELS}
        for v in r.values():
            hist[str(v['label'])] = hist.get(str(v['label']), 0) + 1
        meta = {i: dict({k: ss[i][k] for k in ('day', 'cam', 'top', 'area_full')}, review=bool(ss[i].get('review')), score=ss[i].get('score')) for i in todo[:200] + done[-50:]}
        return jsonify({'todo': todo[:int(request.args.get('n', 200))], 'recent': done[-50:],
                        'labels': {i: r[i]['label'] for i in done[-50:]}, 'meta': meta,
                        'total': len(ss), 'done': len(r), 'hist': hist})

    @app.post('/api/keep2')
    def keep2_answer():
        body = request.get_json(force=True) or {}
        sid, label = str(body.get('id', '')), body.get('label')
        sample(sid)
        if label is not None and (not isinstance(label, int) or isinstance(label, bool) or label not in LABELS):
            return jsonify({'error': 'Ответ — 1 оставить, 2–9 причина, 0 не понять'}), 400
        try:
            r = answer(here(), sid, label)
        except TimeoutError as exc:
            return jsonify({'error': str(exc)}), 503
        return jsonify({'ok': True, 'done': len(r)})

    @app.get('/keep2-img/<sid>/<what>')
    def keep2_img(sid, what):
        s = sample(sid)
        if what not in ('zoom', 'frame'):
            abort(404)
        resp = Response(render(here(), s, what), mimetype='image/jpeg')
        resp.headers['Cache-Control'] = 'private, max-age=3600'
        return resp


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'build':
        t0 = time.time()
        got = build_all(str(ROOT))
        print('%d people (%d from /keep, %d from /fix), %.0f s' % (len(got), sum(v['rl'] == 'keep' for v in got.values()), sum(v['rl'] == 'fix' for v in got.values()), time.time() - t0))
    else:
        print(__doc__)
