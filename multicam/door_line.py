"""The owner's shop line on camera 1 (08.10.2026): a line drawn on the 1280 x 720 frame and a point on its 'inside the
shop' side; a person with any piece of mask on that side is inside, whatever the model says.
-> data/door_v2/door_line.json {'p1': [x, y], 'p2': [x, y], 'inside': [x, y], 'min_px': n} (1280 x 720)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FILE = ROOT / 'data' / 'door_v2' / 'door_line.json'


def load():
    return json.load(open(FILE)) if FILE.exists() else None


def side_sign(line, x, y):
    (x1, y1), (x2, y2) = line['p1'], line['p2']
    return np.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))


def inside(line, mask, min_px=None):
    """mask: H x W bool of any size (the whole frame) -> True when at least min_px of its pixels lie on the inside
    side of the line (min_px: the line's own setting, default 20 at 1280 x 720, scaled to the mask's size)."""
    if line is None or not mask.any():
        return False
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    sx, sy = 1280.0 / w, 720.0 / h
    want = side_sign(line, *line['inside'])
    s = side_sign(line, xs * sx, ys * sy)
    n = int((s == want).sum())
    need = (min_px if min_px is not None else line.get('min_px', 20)) / (sx * sy)
    return n >= max(1, need)


def depth(line, mask):
    """How deep the mask goes past the line, in pixels of the 1280 x 720 frame: > 0 -- its deepest pixel is that far on
    the inside; < 0 -- the whole mask stays on the outside, its nearest pixel that far from the line (08.10: for a
    side with hysteresis, so a person standing on the line does not flicker)."""
    if line is None or not mask.any():
        return None
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    sx, sy = 1280.0 / w, 720.0 / h
    (x1, y1), (x2, y2) = line['p1'], line['p2']
    n = float(np.hypot(x2 - x1, y2 - y1))
    want = side_sign(line, *line['inside'])
    d = ((x2 - x1) * (ys * sy - y1) - (y2 - y1) * (xs * sx - x1)) / n * want
    return float(d.max())
