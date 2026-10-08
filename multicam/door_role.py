"""Staff or customer at the door by accumulation (08.10.2026): every person keeps its best views over the whole track
(big mask, nobody else touching it), and the role is the mean of the owner's staff model (data/staff/current_role.pkl,
staff_model.py) over up to KEEP of them, recomputed as views pile up -- not one frame. The ReID teachers and the
model live in one long-lived process (venv_rfdetr), loaded once.

  Bank   -- in the live loop: offer(key, score, crop_fn, box) keeps the KEEP best views of each person key
  Worker -- a client of `door_role.py serve` (stdin/stdout, one JSON line each way)
  Roles  -- role(keys): the person's views from all its keys (pieces stitched into one person), cached

usage: door_role.py serve   (venv_rfdetr; the live loop starts it itself)"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
RF = r'C:\Users\ArykovAA\cctv_ai\venv_rfdetr\Scripts\python.exe'
KEEP = 8                       # views per person for the mean
MIN_AREA = 6000                # mask pixels on the 2176 x 1224 frame: smaller views say little about clothes
ISOLATED_IOU = 0.05            # another person's box overlapping more than this: the view is shared


def score(area, isolated, cut):
    """The worth of one view: its size, much less when someone else overlaps it or the frame edge cuts it."""
    if area < MIN_AREA:
        return 0.0
    return float(area) * (1.0 if isolated else 0.15) * (0.3 if cut else 1.0)


def box_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0.0


class Bank:
    def __init__(self, folder, keep=KEEP):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.views = {}                                   # key -> [(score, path, box_1280)]
        self.n = 0

    def wants(self, key, s):
        v = self.views.get(key, [])
        return s > 0 and (len(v) < self.keep or s > v[-1][0])

    def offer(self, key, s, make_crop, box_1280):
        """make_crop() -> BGR image, called only when the view is kept."""
        import cv2
        if not self.wants(key, s):
            return False
        img = make_crop()
        if img is None:
            return False
        self.n += 1
        path = self.folder / ('%d.jpg' % self.n)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        v = sorted(self.views.get(key, []) + [(s, str(path), [float(x) for x in box_1280])], key=lambda z: -z[0])
        for _, p, _ in v[self.keep:]:
            try:
                os.remove(p)
            except OSError:
                pass
        self.views[key] = v[:self.keep]
        return True

    def best(self, keys, n=KEEP):
        allv = sorted((x for k in keys for x in self.views.get(k, [])), key=lambda z: -z[0])
        return allv[:n]


class Worker:
    def __init__(self):
        self.p = subprocess.Popen([RF, str(ROOT / 'door_role.py'), 'serve'], cwd=str(ROOT), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
                                  env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        self.lock = threading.Lock()
        self.threshold = float(self._read().get('ready', 0.45))   # the model's own threshold

    def _read(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('role worker died')
            if line.startswith('{'):
                return json.loads(line)

    def probs(self, paths, boxes, vectors=False):
        with self.lock:
            self.p.stdin.write(json.dumps({'crops': paths, 'boxes': boxes, 'vectors': vectors}) + '\n')
            self.p.stdin.flush()
            r = self._read()
            return (r['p'], r.get('v')) if vectors else r['p']


GALLERY_SEED = 0.7              # 08.10: a person the role model is this sure of (>= MIN_SEED_VIEWS views) seeds the day's gallery
MIN_SEED_VIEWS = 4
GALLERY_MAX = 40
GALLERY_MIN = 5                 # the gallery is used once it has this many views
GALLERY_SIM = 0.65              # mean of the 3 best cosines to the gallery (staff_gallery_test.py: 0.651-0.684 by day)


class Roles:
    """08.10: staff by the model OR by the day's gallery -- staff wear the same clothes all day; the gallery is seeded by
    people the model is sure of (on the owner's /staff answers: 93.7 % against 91.3 % for the model on the same views)."""

    def __init__(self, bank, worker, threshold=None):
        self.bank, self.worker = bank, worker
        self.threshold = threshold if threshold is not None else getattr(worker, 'threshold', 0.45)
        self.cache = {}                                       # view path -> p_staff of that view
        self.vec = {}                                         # view path -> appearance vector
        self.gallery, self.gallery_day, self.seeded = [], None, set()

    def role(self, keys):
        v = self.bank.best(keys)
        if not v:
            return {'role': None, 'p_staff': None, 'views': 0}
        new = [(p, b) for _, p, b in v if p not in self.cache]   # one model pass per view, ever (08.10: was per set)
        if new:
            try:
                ps, vs = self.worker.probs([p for p, _ in new], [b for _, b in new], vectors=True)
                if len(ps) != len(new):
                    raise RuntimeError('%d probabilities for %d views' % (len(ps), len(new)))
                self.cache.update({p: float(x) for (p, _), x in zip(new, ps)})
                if vs:
                    self.vec.update({p: np.asarray(x, np.float32) for (p, _), x in zip(new, vs)})
            except Exception as exc:                          # the role never blocks the door
                return {'role': None, 'p_staff': None, 'views': len(v), 'error': str(exc)[:100]}
        p = float(np.mean([self.cache[p_] for _, p_, _ in v]))
        day = time.strftime('%Y%m%d')
        if day != self.gallery_day:                           # a new day: the staff wear other clothes
            self.gallery, self.gallery_day, self.seeded = [], day, set()
        mine = [self.vec[p_] for _, p_, _ in v if p_ in self.vec]
        sim = None
        if len(self.gallery) >= GALLERY_MIN and mine:
            G = np.stack([g for g, owner in self.gallery if owner != tuple(keys)]) if any(o != tuple(keys) for _, o in self.gallery) else None
            if G is not None and len(G) >= GALLERY_MIN:
                S = np.stack(mine) @ G.T
                sim = float(np.mean(np.sort(S, 1)[:, -3:].mean(1)))
        if p >= GALLERY_SEED and len(mine) >= MIN_SEED_VIEWS and tuple(keys) not in self.seeded and len(self.gallery) < GALLERY_MAX:
            self.seeded.add(tuple(keys))
            self.gallery += [(m, tuple(keys)) for m in mine[:GALLERY_MAX - len(self.gallery)]]
        staff = p >= self.threshold or (sim is not None and sim >= GALLERY_SIM)
        out = {'role': 'staff' if staff else 'customer', 'p_staff': round(p, 3), 'views': len(v)}
        if sim is not None:
            out['gallery_sim'] = round(sim, 3)
        return out


def serve():
    import cv2
    import staff_model
    import track_emb as T
    clf = staff_model.load_current(ROOT)
    embed = T.teachers()
    print(json.dumps({'ready': float(getattr(clf, 'threshold', 0.45))}), flush=True)
    for line in sys.stdin:
        try:
            q = json.loads(line)
            crops = [cv2.imread(p) for p in q['crops']]
            ok = [i for i, c in enumerate(crops) if c is not None]
            if not ok:
                print(json.dumps({'p': []}), flush=True)
                continue
            v1, v2 = embed([crops[i] for i in ok])
            p = clf.predict_proba(np.asarray(v1), np.asarray(v2), [q['boxes'][i] for i in ok], ['cam1'] * len(ok))[:, 1]
            out = {'p': [float(x) for x in p]}
            if q.get('vectors'):                          # 08.10: the day's staff gallery compares these
                a1 = np.asarray(v1, np.float32); a2 = np.asarray(v2, np.float32)
                a1 /= np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), 1e-6)
                a2 /= np.maximum(np.linalg.norm(a2, axis=1, keepdims=True), 1e-6)
                v = np.concatenate([a1, a2], 1) / np.sqrt(2)
                out['v'] = [[round(float(x), 4) for x in row] for row in v]
            print(json.dumps(out), flush=True)
        except Exception as exc:
            print(json.dumps({'p': [], 'error': str(exc)[:200]}), flush=True)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'serve':
    serve()
