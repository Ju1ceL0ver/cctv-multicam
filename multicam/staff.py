"""Staff or customer: the owner labels single people, one key each, and a classifier learns as he goes: /staff (05.10.2026).

People are those of the machine drafts (teacher + SAM label maps over 1280x720 frames of every recorded day, both
cameras) that have teacher ReID vectors (data/teacher_emb/<frame>.npz: TransReID cloth + CSCI shape). `build` turns
them into one feature table: both vectors (L2-normalised, PCA), the camera and where the person stands.

The queue asks what teaches most:
  - cold start (fewer than MIN_EACH answers of a class): a person whose nearest look-alikes come from many different
    days goes first, alternating with random ones -- staff are the same few people every day, customers are not;
  - then a logistic regression on the answers scores everybody; the queue alternates the most doubtful (score near
    0.5) with random people (they keep the accuracy estimate honest), and the page shows the guess: Enter takes it.
Accuracy: leave-one-day-out on the answers (the same person shows up in many frames of one day; a random split would
flatter the number).

usage: staff.py build        -> data/staff/feats.npz, people.json
       staff.py eval         -> accuracy of the current answers, and how many people of all are called staff
Storage: data/staff/labels.json {person: {label, at}}, every answer also in history.jsonl."""
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
W, H = 1280, 720
LABELS = {1: 'покупатель', 2: 'сотрудник', 0: 'не понять / не человек'}
MIN_EACH = 5
PCA_CLOTH, PCA_SHAPE = 128, 64
MASKED = os.environ.get('RA_STAFF_MASKED', '1') == '1'   # 08.10: mask vectors 86.4 % vs box 86.2 %, counter 74 vs 65 %
_model = {'n': -1, 'p': None, 'cv': None}
_lock = threading.Lock()


def folder(root):
    return Path(root) / 'data' / 'staff'


def _key(ident):
    return hashlib.md5(ident.encode()).hexdigest()


def labels(root):
    p = folder(root) / 'labels.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def _pca(X, k):
    X = X - X.mean(0)
    rng = np.random.default_rng(0)
    sub = X[rng.choice(len(X), min(len(X), 6000), replace=False)]
    _, _, vt = np.linalg.svd(sub, full_matrices=False)
    return X @ vt[:k].T


def _norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)


def build(root):
    """One row per person with teacher vectors and a frame on disk."""
    import rate
    frames = rate.index(root, force=True)
    ids, meta, cloth, shape = [], {}, [], []
    for f in sorted((Path(root) / 'data' / 'teacher_emb').glob('*.npz')):
        frame = f.stem
        if frame not in frames:
            continue
        parts = frame.split('_')
        cam = next((p for p in parts if p in ('cam1', 'cam2')), None)
        if not cam or not parts[0].isdigit():
            continue
        z = np.load(f)
        for v, box, c, s in zip(z['labels'], z['boxes'], z['cloth'], z['shape']):
            pid = '%s_p%d' % (frame, int(v))
            ids.append(pid)
            meta[pid] = {'frame': frame, 'value': int(v), 'day': parts[0], 'cam': cam, 'box': [int(x) for x in box]}
            cloth.append(c.astype(np.float32))
            shape.append(s.astype(np.float32))
    cz = folder(root) / 'counter_emb.npz'               # 08.10: the live counter's door snapshots (staff_counter.py)
    if cz.exists():
        z = np.load(cz)
        cm = json.load(open(folder(root) / 'counter_people.json', encoding='utf-8'))
        for i, c, s_ in zip(z['ids'], z['cloth'], z['shape']):
            i = str(i)
            if i in cm:
                ids.append(i); meta[i] = cm[i]; cloth.append(c); shape.append(s_)
    mz = folder(root) / 'masked_emb.npz'                # 08.10: vectors of the crop on the person's mask (staff_masked.py)
    if MASKED and mz.exists():
        z = np.load(mz)
        mc, ms = z['cloth'], z['shape']                  # once: every z[...] unpacks the whole array again
        at = {str(i): k for k, i in enumerate(z['ids'])}
        for k, i in enumerate(ids):
            if i in at:
                cloth[k], shape[k] = mc[at[i]], ms[at[i]]
    C = _norm(_pca(_norm(np.array(cloth)), PCA_CLOTH))
    S = _norm(_pca(_norm(np.array(shape)), PCA_SHAPE))
    box = np.array([meta[i]['box'] for i in ids], np.float32)
    where = np.stack([(box[:, 0] + box[:, 2]) / 2 / W, box[:, 3] / H, (box[:, 3] - box[:, 1]) / H,
                      np.array([meta[i]['cam'] == 'cam2' for i in ids], np.float32)], 1)
    # cold-start prior: how many distinct days among the 20 nearest look-alikes (staff come back every day)
    days = np.array([meta[i]['day'] for i in ids])
    E = np.concatenate([C, S], 1) / np.sqrt(2)
    prior = np.zeros(len(ids), np.float32)
    for a in range(0, len(ids), 2048):
        sim = E[a:a + 2048] @ E.T
        nn = np.argpartition(-sim, 21, axis=1)[:, :21]
        prior[a:a + 2048] = [len(set(days[r])) for r in nn]
    f = folder(root)
    f.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f / 'feats.npz', ids=np.array(ids), X=np.concatenate([C, S, where], 1).astype(np.float32), prior=prior)
    from storage import atomic_json
    atomic_json(f / 'people.json', meta)
    return len(ids)


_feats = {}


def feats(root):
    p = folder(root) / 'feats.npz'
    st = p.stat().st_mtime
    if _feats.get('mtime') != st:
        z = np.load(p)
        _feats.update(mtime=st, ids=[str(i) for i in z['ids']], X=z['X'], prior=z['prior'],
                      meta=json.load(open(folder(root) / 'people.json', encoding='utf-8')))
        _feats['row'] = {i: k for k, i in enumerate(_feats['ids'])}
    return _feats


def _clf(n=1000):
    # 08.10: a small net (256, 64) -- day-held-out 89.0 % on the owner's 1930 answers against 80.5 % of the logistic
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier((256, 64), max_iter=400, early_stopping=n >= 100, random_state=0)


def fit(root, force=False):
    """Scores of everybody (None in the cold start) and the leave-one-day-out accuracy; cached by the answers."""
    F, r = feats(root), labels(root)
    ans = [(i, v['label']) for i, v in r.items() if v['label'] in (1, 2) and i in F['row']]
    with _lock:
        if not force and _model['n'] == len(r):
            return _model
        y = np.array([a == 2 for _, a in ans])
        if y.sum() < MIN_EACH or (~y).sum() < MIN_EACH:
            _model.update(n=len(r), p=None, cv=None)
            return _model
        rows = [F['row'][i] for i, _ in ans]
        X = F['X'][rows]
        p = _clf(len(y)).fit(X, y).predict_proba(F['X'])[:, 1]
        if _model.get('cv') and abs(len(r) - _model.get('cv_n', 0)) < 50 and not force:
            _model.update(n=len(r), p=p)
            return _model
        days = np.array([F['meta'][i]['day'] for i, _ in ans])
        right = total = 0
        fp = fn = 0
        for d in sorted(set(days)):
            te = days == d
            tr = ~te
            if y[tr].sum() < 2 or (~y[tr]).sum() < 2:
                continue
            q = _clf(int(tr.sum())).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] >= 0.5
            right += int((q == y[te]).sum()); total += int(te.sum())
            fp += int((q & ~y[te]).sum()); fn += int((~q & y[te]).sum())
        cv = {'acc': round(right / total, 4) if total else None, 'n': total, 'staff_as_customer': fn, 'customer_as_staff': fp}
        _model.update(n=len(r), p=p, cv=cv, cv_n=len(r))
        return _model


ONLY_CAM = os.environ.get('RA_STAFF_CAM', 'cam1')
DEDUP = 0.95
COUNTER_FIRST = True                  # cosine of the ReID vectors above which a candidate is the same person as one already queued     # 08.10: the owner asked for the door camera only


def queue(root, n=100):
    """The next people to ask, in order: doubtful and random alternating (or the cold-start prior)."""
    F, r = feats(root), labels(root)
    m = fit(root)
    todo = np.array([i not in r and (not ONLY_CAM or F['meta'][i]['cam'] == ONLY_CAM) for i in F['ids']])
    idx = np.flatnonzero(todo)
    rand = sorted(idx, key=lambda k: _key(F['ids'][k]))
    if m['p'] is None:
        first = idx[np.argsort(-F['prior'][idx], kind='stable')]
    else:
        first = idx[np.argsort(np.abs(m['p'][idx] - 0.5), kind='stable')]
    if COUNTER_FIRST:                                   # 08.10: the counter's snapshots of the unlabelled week first
        cnt = np.array([F['meta'][F['ids'][k]].get('src') == 'counter' for k in first], bool)
        if cnt.any():
            first = np.concatenate([first[cnt], first[~cnt]])
            rand = [k for k in rand if F['meta'][F['ids'][k]].get('src') == 'counter'] + \
                   [k for k in rand if F['meta'][F['ids'][k]].get('src') != 'counter']
    emb = F['X'][:, :192] / np.linalg.norm(F['X'][:, :192], axis=1, keepdims=True).clip(1e-6)
    out, seen = [], set()
    a = b = 0
    picked = []
    while len(out) < n and (a < len(first) or b < len(rand)):
        for src, pos in ((first, 'a'), (rand, 'b')):
            k = a if pos == 'a' else b
            while k < len(src) and (src[k] in seen or (picked and float(np.max(emb[picked] @ emb[src[k]])) > DEDUP)):
                seen.add(src[k]); k += 1            # 08.10: the same person from a neighbouring frame -- skip
            if k < len(src):
                seen.add(src[k]); out.append(int(src[k])); picked.append(int(src[k])); k += 1
            if pos == 'a':
                a = k
            else:
                b = k
    return [F['ids'][k] for k in out[:n]], m


def answer(root, pid, label):
    from storage import atomic_json, file_lock
    f = folder(root)
    f.mkdir(parents=True, exist_ok=True)
    with file_lock(str(f / 'labels.json') + '.lock'):
        r = labels(root)
        now = time.strftime('%Y-%m-%dT%H:%M:%S')
        if label is None:
            r.pop(pid, None)
        else:
            r[pid] = {'label': int(label), 'at': now}
        atomic_json(f / 'labels.json', r)
        with open(f / 'history.jsonl', 'a', encoding='utf-8') as h:
            h.write(json.dumps({'id': pid, 'label': label, 'at': now}) + '\n')
    return r


def register(app, root):
    from flask import Response, abort, jsonify, render_template, request
    import inout
    here = root if callable(root) else (lambda: root)

    def person(pid):
        if not re.fullmatch(r'[A-Za-z0-9_]{1,90}', pid):
            abort(404)
        s = feats(here())['meta'].get(pid)
        if s is None:
            abort(404)
        return s

    @app.get('/staff')
    def staff_page():
        return render_template('staff.html', labels=LABELS)

    @app.get('/api/staff')
    def staff_list():
        F, r = feats(here()), labels(here())
        todo, m = queue(here(), int(request.args.get('n', 60)))
        done = sorted((i for i in r if i in F['row']), key=lambda i: r[i]['at'])[-50:]
        guess = {}
        if m['p'] is not None:
            guess = {i: round(float(m['p'][F['row'][i]]), 3) for i in todo + done}
        meta = {i: {k: F['meta'][i][k] for k in ('day', 'cam')} for i in todo + done}
        hist = {str(k): 0 for k in LABELS}
        for v in r.values():
            hist[str(v['label'])] = hist.get(str(v['label']), 0) + 1
        staff_all = int((m['p'] >= 0.5).sum()) if m['p'] is not None else None
        return jsonify({'todo': todo, 'recent': done, 'labels': {i: r[i]['label'] for i in done}, 'guess': guess,
                        'meta': meta, 'total': len(F['ids']), 'done': len(r), 'hist': hist, 'cv': m['cv'],
                        'staff_all': staff_all})

    @app.post('/api/staff')
    def staff_answer():
        body = request.get_json(force=True) or {}
        pid, label = str(body.get('id', '')), body.get('label')
        person(pid)
        if label is not None and (not isinstance(label, int) or isinstance(label, bool) or label not in LABELS):
            return jsonify({'error': 'Ответ — 1, 2 или 0'}), 400
        try:
            r = answer(here(), pid, label)
        except TimeoutError as exc:
            return jsonify({'error': str(exc)}), 503
        return jsonify({'ok': True, 'done': len(r)})

    @app.get('/staff-img/<pid>/<what>')
    def staff_img(pid, what):
        s = person(pid)
        if what not in ('crop', 'plain', 'frame'):
            abort(404)
        if s.get('src') == 'counter':                   # the counter's own snapshot: its crop, or its frame with the box
            import cv2
            if what == 'frame' and s.get('full') and Path(s['full']).exists():
                img = cv2.imread(s['full'])
                x1, y1, x2, y2 = [int(v) for v in s['box_raw']]
                cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (66, 197, 245), 3)
                img = cv2.resize(img, (960, int(960 * img.shape[0] / img.shape[1])))
            else:
                mc = folder(here()) / 'counter_masked' / (pid + '.jpg')
                img = cv2.imread(str(mc) if mc.exists() else s['crop'])
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            resp = Response(buf.tobytes(), mimetype='image/jpeg')
            resp.headers['Cache-Control'] = 'private, max-age=3600'
            return resp
        data = inout.render(here(), s, 'frame' if what == 'frame' else 'crop', outline=what == 'crop')
        resp = Response(data, mimetype='image/jpeg')
        resp.headers['Cache-Control'] = 'private, max-age=3600'
        return resp


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'build':
        t0 = time.time()
        print('%d people, %.0f s' % (build(str(ROOT)), time.time() - t0))
    elif len(sys.argv) >= 2 and sys.argv[1] == 'eval':
        m = fit(str(ROOT), force=True)
        print(m['cv'], 'staff of all:', None if m['p'] is None else int((m['p'] >= 0.5).sum()), 'of', len(feats(str(ROOT))['ids']))
    else:
        print(__doc__)
