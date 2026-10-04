"""Do a track's own points follow it? TAPNext++ on the door runs (04.10.2026): the strong version of door_lk.py.

Each stretch is cut into pieces of CHUNK ticks (2 s). At a piece's first tick, 16 points on the body of every person
(4 x 4 grid over the middle half of the box width, 15-80 % of its height); TAPNext++ (DeepMind, online, re-finds a
point after an occlusion) carries them through the piece. At every tick the person's points that are visible vote for
the box that holds them: a track that slid onto somebody else finds its points in another box.

usage (cctv_base, the card): door_tap.py RUN.jsonl.gz [CKPT]
  -> RUN.tap.json.gz {"<stretch>|<t>|<world>": [visible points, share in the same world's box, best other share, that world]}"""
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
TAP = Path(os.environ.get('RA_TAP_DIR', str(ROOT.parent / 'tapnet')))
CHUNK = 25
GRID = 4


def body_points(q):
    cx, cy, w, h = q['box']
    x1, y1 = (cx - w / 2) * 2176, (cy - h / 2) * 1248
    W, H = w * 2176, h * 1248
    xs = x1 + W * np.linspace(0.3, 0.7, GRID)
    ys = y1 + H * np.linspace(0.15, 0.8, GRID)
    return np.array([[x, y] for y in ys for x in xs], np.float32)


def box_of(q):
    cx, cy, w, h = q['box']
    return np.array([(cx - w / 2) * 2176, (cy - h / 2) * 1248, (cx + w / 2) * 2176, (cy + h / 2) * 1248])


def main(path, ckpt=None):
    import cv2
    import torch
    sys.path.insert(0, str(TAP / 'src'))
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP
    import door_v2 as D
    import sam31_segment as SS
    model = TAPNextPP.from_checkpoint(str(ckpt or TAP / 'tapnextpp_ckpt.pt'), device='cuda')
    rows = []
    try:
        for l in gzip.open(path, 'rt'):
            rows.append(json.loads(l))
    except (EOFError, json.JSONDecodeError):
        pass
    head, ticks = rows[0], rows[1:]
    day, cam = head['day'], head.get('cam', 'cam1')
    raw = D.Raw(day, cam)
    by_span = {}
    for r in ticks:
        by_span.setdefault(r['s'], []).append(r)
    out, t0, frames = {}, time.time(), 0
    status = Path(str(path).replace('.jsonl.gz', '.tap.status.json'))
    for si, rs in sorted(by_span.items()):
        a, _ = head['spans'][si]
        rs = sorted(rs, key=lambda r: r['t'])
        at = {int(round((r['t'] - a) / D.TICK)): r for r in rs}
        n = max(at) + 1
        tks = SS.tick_frames(day, cam, a, n)
        for c0 in range(0, n, CHUNK):
            r0 = at.get(c0)
            if r0 is None or not r0['p']:
                continue
            owner, pts = [], []
            for q in r0['p']:
                p = body_points(q)
                pts.append(p); owner += [q['w']] * len(p)
            pts, owner = np.concatenate(pts), np.array(owner)
            state = None
            for i in range(c0, min(n, c0 + CHUNK)):
                img = raw.get(tks[i])
                if img is None:
                    break
                g = cv2.resize(img, (2176, 1224), interpolation=cv2.INTER_AREA)
                with torch.no_grad():
                    if state is None:
                        pos, vis, state = model.track_frame(g, pts)
                    else:
                        pos, vis, state = model.track_frame(g, state=state)
                frames += 1
                r = at.get(i)
                if r is None or i == c0:
                    continue
                boxes = [(q['w'], box_of(q)) for q in r['p']]
                pos, vis = np.asarray(pos).reshape(-1, 2), np.asarray(vis).reshape(-1).astype(bool)
                for w in set(owner.tolist()):
                    sel = (owner == w) & vis
                    key = '%d|%.2f|%d' % (si, r['t'], w)
                    if sel.sum() < 3 or not any(wb == w for wb, _ in boxes):
                        continue
                    votes = {}
                    for x, y in pos[sel]:
                        inside = [(wb, b) for wb, b in boxes if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
                        if inside:
                            wb = min(inside, key=lambda z: np.hypot((z[1][0] + z[1][2]) / 2 - x, (z[1][1] + z[1][3]) / 2 - y))[0]
                            votes[wb] = votes.get(wb, 0) + 1
                    k = int(sel.sum())
                    others = {kk: v for kk, v in votes.items() if kk != w}
                    ow = max(others, key=others.get) if others else None
                    out[key] = [k, round(votes.get(w, 0) / k, 3), round(others[ow] / k, 3) if ow is not None else 0.0, ow]
        json.dump({'span': si, 'spans': len(by_span), 'frames': frames, 'fps': round(frames / max(1, time.time() - t0), 2),
                   'elapsed_s': round(time.time() - t0), 'updated': time.strftime('%H:%M:%S')}, open(status, 'w'))
    with gzip.open(str(path).replace('.jsonl.gz', '.tap.json.gz'), 'wt') as f:
        json.dump(out, f)
    json.dump({'finished': time.strftime('%H:%M:%S'), 'people_ticks': len(out), 'frames': frames,
               'elapsed_s': round(time.time() - t0)}, open(status, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
