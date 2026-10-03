"""The owner's verdict on the teacher's frames, at the stored size (2176 x 1224): /fix.

One frame at a time with SAM 3.1's people painted over it. `1` or Enter -- take it (as it is, or with whatever he
repainted first: the /paint tools), `2` -- do not take it. A taken frame is a verified one: the test of 23.09
becomes a test of his eyes, and the training frames he took or corrected are what the "take / correct" scorer
learns from (the draft against his version, person by person).

  data/fix/manifest.json     items: id, batch, set, tag, cam, tick, w, h, people, persons (SAM's person per label)
  data/fix/<id>.jpg          the frame;  <id>_init.png  the draft (label map, 0 = nobody);  <id>_mask.png  his edit
  data/fix/state.json        {id: {verdict: take | skip | None (edits, no verdict yet), edited, people, at}}

usage: fix.py test                 the 160 moments of the teacher test (data/v2_test/v2b.json) -> batch 1
       fix.py train N [PER]        N moments of the training windows (half the busiest, half any with people,
                                   spread over windows and cameras) -> batches of PER (100)"""
import base64
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SPACE_S = 20.0
HELD_DAY = '20260923'


def folder(root=ROOT):
    return Path(root) / 'data' / 'fix'


def manifest(root=ROOT):
    p = folder(root) / 'manifest.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {'items': [], 'batches': {}}


def state(root=ROOT):
    p = folder(root) / 'state.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def label_map(w, tick, shape):
    """SAM's people of a tick as one label map at the stored size (bigger first, a small one stays on top)."""
    rows = [int(i) for i in w.rows_at(tick)]
    got = []
    for i in rows:
        r = int(w.t['r'][i])
        x1, y1, x2, y2 = w.masks.rows[r, 4:8].astype(int)
        crop = np.asarray(w.masks.crop(r)).astype(bool)
        got.append((int(crop.sum()), x1, y1, crop, int(w.t['person'][i])))
    got.sort(key=lambda g: -g[0])
    lab = np.zeros(shape, np.uint8)
    persons = []
    for k, (_, x1, y1, crop, person) in enumerate(got[:255]):
        h, wd = min(crop.shape[0], shape[0] - y1), min(crop.shape[1], shape[1] - x1)
        lab[y1:y1 + h, x1:x1 + wd][crop[:h, :wd]] = k + 1
        persons.append(person)
    return lab, persons


def add(picks, batch, name, kind, log=print):
    """picks: [(tag, cam, tick)] -> frames and drafts on disk, the manifest extended (known ids are left alone)."""
    import cv2
    import v2_data as VD
    from storage import atomic_json
    out = folder()
    out.mkdir(parents=True, exist_ok=True)
    m = manifest()
    have = {it['id'] for it in m['items']}
    wins, n = {}, 0
    for tag, cam, tick in sorted(picks):
        ident = '%s_%s_%05d' % (tag, cam, tick)
        if ident in have:
            continue
        if (tag, cam) not in wins:
            for w in wins.values():                       # one decoder at a time is enough here
                if w.cap is not None:
                    w.cap.release(); w.cap = None
            wins[(tag, cam)] = VD.Window(tag, cam)
        w = wins[(tag, cam)]
        img = w.frame(tick)
        if img is None:
            continue
        lab, persons = label_map(w, tick, img.shape[:2])
        cv2.imwrite(str(out / ('%s.jpg' % ident)), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        cv2.imwrite(str(out / ('%s_init.png' % ident)), lab)
        m['items'].append({'id': ident, 'batch': batch, 'set': kind, 'tag': tag, 'cam': cam, 'tick': int(tick),
                           'w': int(img.shape[1]), 'h': int(img.shape[0]), 'people': len(persons), 'persons': persons})
        n += 1
        if n % 20 == 0:
            m['batches'][str(batch)] = name
            atomic_json(out / 'manifest.json', m)
            log('%d frames' % n)
    m['batches'][str(batch)] = name
    atomic_json(out / 'manifest.json', m)
    log('batch %s (%s): %d frames added' % (batch, name, n))
    return n


def pick_train(n, seed=1):
    """n moments of the training windows: per window-camera the same share, half of them the busiest."""
    import v2_data as VD
    rng = random.Random(seed)
    cams = [(t, c) for t, c in VD.window_cams() if not t.startswith(HELD_DAY)]
    per = max(2, -(-n // len(cams)))
    taken_all = []
    for tag, cam in cams:
        w = VD.Window(tag, cam)
        ticks, counts = np.unique(w.t['tick'], return_counts=True)
        cands = [(int(c), rng.random(), int(t)) for t, c in zip(ticks.tolist(), counts.tolist())]
        taken = []
        free = lambda t: all(abs(t - u) * VD.TICK >= SPACE_S for u in taken)
        for _, _, t in sorted(cands, key=lambda x: (-x[0], x[1])):
            if len(taken) >= per // 2:
                break
            if free(t):
                taken.append(t)
        for _, _, t in sorted(cands, key=lambda x: x[1]):
            if len(taken) >= per:
                break
            if free(t):
                taken.append(t)
        taken_all += [(tag, cam, t) for t in taken]
    rng.shuffle(taken_all)
    return taken_all[:n]


def save(root, ident, verdict, png_b64):
    """One answer. verdict: 'take' | 'skip' | None (edits kept, no verdict yet). The previous edit goes to history."""
    import cv2
    from storage import atomic_json, file_lock
    out = folder(root)
    it = next((i for i in manifest(root)['items'] if i['id'] == ident), None)
    if it is None:
        raise ValueError('Нет такого кадра')
    if verdict not in ('take', 'skip', None):
        raise ValueError('Не тот ответ')
    labels = None
    if png_b64:
        raw = base64.b64decode(png_b64.split(',', 1)[-1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError('Не картинка')
        labels = img if img.ndim == 2 else img[:, :, 2 if img.shape[2] >= 3 else 0]
        if labels.shape != (it['h'], it['w']):
            raise ValueError('Не тот размер: %s' % (labels.shape,))
    with file_lock(str(out / 'state.json') + '.lock'):
        target = out / ('%s_mask.png' % ident)
        if labels is not None:
            if target.exists():
                hist = out / 'history'
                hist.mkdir(exist_ok=True)
                target.replace(hist / ('%s_%s.png' % (ident, time.strftime('%Y%m%d_%H%M%S'))))
            cv2.imwrite(str(target), labels.astype(np.uint8))
        cur = cv2.imread(str(target if target.exists() else out / ('%s_init.png' % ident)), cv2.IMREAD_UNCHANGED)
        init = cv2.imread(str(out / ('%s_init.png' % ident)), cv2.IMREAD_UNCHANGED)
        st = state(root)
        st[ident] = {'verdict': verdict, 'edited': bool(target.exists() and (cur != init).any()),
                     'people': int(len([v for v in np.unique(cur) if v])), 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        atomic_json(out / 'state.json', st)
        with open(out / 'history.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(st[ident], id=ident)) + '\n')
    return st[ident]


def register(app, root):
    import re
    from flask import abort, jsonify, render_template, request, send_file
    here = root if callable(root) else (lambda: root)

    def valid(ident):
        if not re.fullmatch(r'[0-9A-Za-z_]{1,80}', ident):
            abort(404)

    @app.get('/fix')
    def fix_page():
        return render_template('fix.html')

    @app.get('/api/fix')
    def fix_list():
        m = manifest(here())
        live_p = folder(here()) / 'live.json'
        live = json.load(open(live_p, encoding='utf-8')) if live_p.exists() else None
        return jsonify({'live': live, 'items': [{k: it.get(k) for k in ('id', 'batch', 'set', 'w', 'h', 'people', 'tag', 'cam', 'rank')} for it in m['items']],
                        'batches': m.get('batches', {}), 'state': state(here())})

    @app.get('/fix-img/<ident>/<what>')
    def fix_img(ident, what):
        valid(ident)
        d = folder(here())
        if what == 'frame':
            p, mime = d / ('%s.jpg' % ident), 'image/jpeg'
        elif what == 'labels':                   # his edit if there is one, else the draft
            p = d / ('%s_mask.png' % ident)
            p, mime = (p if p.exists() else d / ('%s_init.png' % ident)), 'image/png'
        elif what == 'init':
            p, mime = d / ('%s_init.png' % ident), 'image/png'
        else:
            abort(404)
        if not p.exists():
            abort(404)
        r = send_file(p, mimetype=mime)
        r.headers['Cache-Control'] = 'no-store' if what == 'labels' else 'private, max-age=3600'
        return r

    @app.post('/api/fix/<ident>')
    def fix_save(ident):
        valid(ident)
        body = request.get_json(force=True) or {}
        try:
            return jsonify(save(here(), ident, body.get('verdict'), body.get('png')))
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        except TimeoutError as exc:
            return jsonify({'error': str(exc)}), 503


def export_yolo(rep=8):
    """The owner's taken training frames as a YOLO-seg set next to the SAM windows' one (each frame `rep` times: 170
    frames against 10 800), every outline of a person kept -> data/seg_datasets/fix_yolo/data.yaml."""
    import cv2
    import v2_teacher_test as TT
    base = ROOT / 'data' / 'seg_datasets'
    out = base / 'fix_yolo'
    for sub in ('images/train', 'labels/train'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for it, lab_p in TT.fix_items('take', 'train'):
        img = cv2.imread(str(lab_p.parent / ('%s.jpg' % it['id'])))
        lab = cv2.imread(str(lab_p), cv2.IMREAD_UNCHANGED)
        if img is None or lab is None:
            continue
        if lab.ndim == 3:
            lab = lab[:, :, 0]
        lab = cv2.resize(lab, (1280, 720), interpolation=cv2.INTER_NEAREST)
        lines = []
        for v in np.unique(lab):
            if not v:
                continue
            cs, _ = cv2.findContours((lab == v).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if cv2.contourArea(c) < 20:
                    continue
                c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(np.float32)
                if len(c) < 3:
                    continue
                c[:, 0] /= 1280
                c[:, 1] /= 720
                lines.append('0 ' + ' '.join('%.5f %.5f' % (x, y) for x, y in np.clip(c, 0, 1)))
        small = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
        for r in range(rep):
            cv2.imwrite(str(out / 'images' / 'train' / ('%s_r%d.jpg' % (it['id'], r))), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / 'labels' / 'train' / ('%s_r%d.txt' % (it['id'], r))).write_text('\n'.join(lines))
        n += 1
    sam = base / 'sam31_yolo'
    (out / 'data.yaml').write_text('train:\n  - %s\n  - %s\nval: %s\nnames:\n  0: person\n'
                                   % ((sam / 'images' / 'train').as_posix(), (out / 'images' / 'train').as_posix(), (sam / 'images' / 'val').as_posix()))
    print('fix_yolo:', n, 'frames x', rep, flush=True)
    return n


if __name__ == '__main__':
    os.chdir(ROOT)
    say = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)
    if sys.argv[1] == 'test':
        spec = json.load(open(ROOT / 'data' / 'v2_test' / 'v2b.json'))['items']
        add([(i['tag'], i['cam'], int(i['tick'])) for i in spec], 1, 'тест 23.09', 'test', say)
    elif sys.argv[1] == 'train':
        n = int(sys.argv[2])
        per = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        picks = pick_train(n)
        first = max([int(b) for b in manifest()['batches']] + [1]) + 1
        for k in range(0, len(picks), per):
            add(picks[k:k + per], first + k // per, 'обучение %d' % (k // per + 1), 'train', say)
    elif sys.argv[1] == 'yolo':
        export_yolo(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
