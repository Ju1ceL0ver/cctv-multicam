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


def extent(mask):
    """(top, height) of the mask in pixels of the 1280 x 720 frame."""
    ys = np.nonzero(mask.any(1))[0]
    if not len(ys):
        return None, None
    sy = 720.0 / mask.shape[0]
    return float(ys[0] * sy), float((ys[-1] - ys[0] + 1) * sy)


def full_height_table(obs, bin_px=20, q=90):
    """The height a whole person has with the head at a given row: per bin of the mask's top, the q-th percentile of the
    masks' heights (most masks there are whole people) -- from any runs, no labels. obs: [(top, height)]."""
    a = np.array([o for o in obs if o[0] is not None], float)
    edges = np.arange(0, 721, bin_px)
    full = []
    for lo in edges[:-1]:
        h = a[(a[:, 0] >= lo) & (a[:, 0] < lo + bin_px), 1] if len(a) else []
        full.append(float(np.percentile(h, q)) if len(h) >= 20 else None)
    # empty bins borrow the nearest filled one
    filled = [i for i, v in enumerate(full) if v is not None]
    for i, v in enumerate(full):
        if v is None and filled:
            full[i] = full[min(filled, key=lambda j: abs(j - i))]
    return {'bin_px': bin_px, 'full': full}


def truncated(table, top, height, ratio):
    """A mask much shorter than a whole person with its head there: its lower part is hidden (a stand, another person)."""
    if not table or ratio <= 0 or top is None:
        return False
    i = min(len(table['full']) - 1, max(0, int(top // table['bin_px'])))
    f = table['full'][i]
    return bool(f) and height < ratio * f


def vote(cfg, table, d, top, height):
    """1 inside / 0 outside / 0.5 not sure, with hysteresis and the hidden-legs rule (door_line_tune.py)."""
    if d is None:
        return 0.5
    if d >= cfg['h_in']:
        return 1.0
    if d <= -cfg['h_out'] and not truncated(table, top, height, cfg.get('trunc', 0)):
        return 0.0
    return 0.5


def bottom_x(mask):
    """x (1280 frame) of the mask's lowest tenth -- where the feet would be."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    lo = ys >= ys.max() - max(1, (ys.max() - ys.min()) // 10)
    return float(xs[lo].mean() * 1280.0 / mask.shape[1])


def point_depth(line, x, y):
    """Signed distance (1280 frame) of a point past the line, > 0 on the inside."""
    (x1, y1), (x2, y2) = line['p1'], line['p2']
    n = float(np.hypot(x2 - x1, y2 - y1))
    return float(((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / n * side_sign(line, *line['inside']))


def full_at(table, top):
    if not table or top is None:
        return None
    return table['full'][min(len(table['full']) - 1, max(0, int(top // table['bin_px'])))]


def vote2(cfg, table, line, d, top, height, bx):
    """Like vote(), with cfg['legs']: 'unsure' -- a mask with hidden legs gives no outside vote; 'virtual' -- its feet
    are put where a whole person's would be (head + the usual height there) and that point is judged."""
    if d is None:
        return 0.5
    if d >= cfg['h_in']:
        return 1.0
    if truncated(table, top, height, cfg.get('trunc', 0)):
        if cfg.get('legs') == 'virtual' and bx is not None:
            dv = point_depth(line, bx, top + full_at(table, top))
            return 1.0 if dv >= cfg['h_in'] else (0.0 if dv <= -cfg['h_out'] else 0.5)
        return 0.5
    return 0.0 if d <= -cfg['h_out'] else 0.5


def side_vote(cfg, table, line, d, top, height, bx, p):
    """One observation's vote by a door_side_tune.py setting (mode, thr, h_in, h_out, trunc, legs): 1 / 0 / 0.5."""
    lv = vote2(cfg, table, line, d, top, height, bx)
    thr = cfg.get('thr')
    mv = 0.5 if (p is None or not thr) else (1.0 if p >= thr else (0.0 if p <= 1 - thr else 0.5))
    m = cfg['mode']
    if m == 'line':
        return lv
    if m == 'model':
        return mv
    if m == 'model+line_in':
        return 1.0 if (d is not None and d >= cfg['h_in']) else mv
    if m == 'agree':
        return lv if lv == mv else 0.5
    if m == 'line_in+model_out':
        return 1.0 if lv == 1 else (0.0 if mv == 0 else 0.5)
    if m == 'line_in+both_out':
        return 1.0 if lv == 1 else (0.0 if (mv == 0 and lv == 0) else 0.5)
    if m == 'either_in+both_out':
        return 1.0 if (lv == 1 or mv == 1) else (0.0 if (lv == 0 and mv == 0) else 0.5)
    raise ValueError(m)
