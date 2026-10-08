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

    def probs(self, paths, boxes):
        with self.lock:
            self.p.stdin.write(json.dumps({'crops': paths, 'boxes': boxes}) + '\n')
            self.p.stdin.flush()
            return self._read()['p']


class Roles:
    def __init__(self, bank, worker, threshold=None):
        self.bank, self.worker = bank, worker
        self.threshold = threshold if threshold is not None else getattr(worker, 'threshold', 0.45)
        self.cache = {}                                       # frozenset of view paths -> p

    def role(self, keys):
        v = self.bank.best(keys)
        if not v:
            return {'role': None, 'p_staff': None, 'views': 0}
        sig = frozenset(p for _, p, _ in v)
        if sig not in self.cache:
            try:
                ps = self.worker.probs([p for _, p, _ in v], [b for _, _, b in v])
                if not ps:
                    raise RuntimeError('no probabilities')
                self.cache[sig] = float(np.mean(ps))
            except Exception as exc:                          # the role never blocks the door
                return {'role': None, 'p_staff': None, 'views': len(v), 'error': str(exc)[:100]}
        p = self.cache[sig]
        return {'role': 'staff' if p >= self.threshold else 'customer', 'p_staff': round(p, 3), 'views': len(v)}


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
            print(json.dumps({'p': [float(x) for x in p]}), flush=True)
        except Exception as exc:
            print(json.dumps({'p': [], 'error': str(exc)[:200]}), flush=True)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'serve':
    serve()
