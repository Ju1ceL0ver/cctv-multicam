"""09.10: the side features computed in threads (door_combo.Live.add -> events) give the same model votes as one by one."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))


class FakeClf:
    def features_rgb_crop(self, frame, crop, x0, y0):
        if crop.sum() == 0:
            raise ValueError('empty')
        return np.array([float(crop.sum()), float(x0), float(y0), float(frame[0, 0, 0])])

    def p_inside_batch(self, X):
        return np.array([1.0 / (1.0 + np.exp(-(x[1] - 500) / 100.0)) for x in X])


def make(threads):
    import os
    import door_combo
    os.environ['RA_SIDE_THREADS'] = str(threads)
    live = door_combo.Live.__new__(door_combo.Live)
    live.clf = FakeClf()
    live.side = {'mode': 'model', 'thr': 0.9, 'h_in': 0, 'h_out': 0, 'trunc': 0.0, 'legs': 'none', 'conf': 2, 'band': 60,
                 'near': 150, 'birth': False, 'alt': 'first', 'gate': False, 'table': None}
    live.line = {'p1': [300, 200], 'p2': [900, 260], 'inside': [600, 500]}
    live.raw, live.obs, live.pending = {}, {}, {}
    live.cfg = {'stitch': None, 'conf': 2, 'thr': 0.95, 'move': 0, 'rad': 1, 'lo': 2}
    live.u, live.hz = np.array([0.0, 1.0]), np.zeros((0, 2))
    live.line_cfg = None
    return live


def test_threads_same_votes():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        crop = (rng.random((40, 20)) > (0.3 if i % 50 else 1.1)).astype(np.uint8)   # every 50th empty
        frame = np.full((720, 1280, 3), i % 255, np.uint8)
        rows.append((('w', i, 0), float(i), frame, (crop, int(rng.integers(0, 1200)), int(rng.integers(0, 680))), [0.5, 0.5]))
    res = []
    for threads in (1, 4):
        live = make(threads)
        for key, t, fr, m, foot in rows:
            live.add(key, t, fr, m, foot)
        live.events(None, {})
        res.append({k: v[5] for k, v in live.raw.items()})
    assert res[0] == res[1]
    assert 280 <= len(res[0]) < 300                           # the empty crops are skipped, nothing broke
    assert all(v is not None for v in res[0].values())
