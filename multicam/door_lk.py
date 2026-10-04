"""Do a track's own points follow it? Lucas-Kanade on the door runs (04.10.2026), a cheap check of the point-tracking
idea (TAPNext++ would be the strong version).

Per tick of a door run, inside every person's box (the middle of the body: the box shrunk 25 % on the sides, its
upper 85 %), up to NPTS good corners; Lucas-Kanade (pyramid) to the next tick with a forward-backward check; each
surviving point votes for the person whose box holds it at the next tick (the nearest box centre when several do).
A track that jumps to another person shows it: its points land in somebody else's box.

usage: door_lk.py RUN.jsonl.gz  -> RUN.lk.json.gz {"<stretch>|<t>|<world>": [points, share in the same world's box
                                    next tick, share in the best other box, that world]}"""
import gzip
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
NPTS = 20
SCALE = 0.5                      # LK on a 1088 x 612 grey frame
FB_PX = 1.5                      # forward-backward error allowed (px of the small frame)


def boxes_of(r):
    out = []
    for q in r['p']:
        cx, cy, w, h = q['box']
        out.append((q['w'], np.array([(cx - w / 2) * 2176, (cy - h / 2) * 1248, (cx + w / 2) * 2176, (cy + h / 2) * 1248]) * SCALE))
    return out


def main(path):
    import door_v2 as D
    import sam31_segment as SS
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
    out, t0 = {}, time.time()
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    status = Path(str(path).replace('.jsonl.gz', '.lk.status.json'))
    for si, rs in sorted(by_span.items()):
        a, _ = head['spans'][si]
        rs = sorted(rs, key=lambda r: r['t'])
        n = int(round((rs[-1]['t'] - a) / D.TICK)) + 1
        tks = SS.tick_frames(day, cam, a, n)
        prev, prev_r = None, None
        for r in rs:
            i = int(round((r['t'] - a) / D.TICK))
            img = raw.get(tks[i])
            if img is None:
                prev = None
                continue
            g = cv2.cvtColor(cv2.resize(img, None, fx=SCALE * 2176 / img.shape[1], fy=SCALE * 1224 / img.shape[0],
                                        interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            if prev is not None and prev_r is not None and abs(r['t'] - prev_r['t'] - D.TICK) < 0.05 and prev_r['p']:
                nb = boxes_of(r)
                for w, b in boxes_of(prev_r):
                    x1, y1, x2, y2 = b
                    bw, bh = x2 - x1, y2 - y1
                    m = np.zeros_like(prev)
                    m[int(max(0, y1)):int(min(m.shape[0], y1 + 0.85 * bh)), int(max(0, x1 + 0.25 * bw)):int(min(m.shape[1], x2 - 0.25 * bw))] = 255
                    p0 = cv2.goodFeaturesToTrack(prev, NPTS, 0.01, 4, mask=m)
                    key = '%d|%.2f|%d' % (si, prev_r['t'], w)
                    if p0 is None or len(p0) < 3:
                        out[key] = [0, None, None, None]
                        continue
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, **lk)
                    pb, stb, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, **lk)
                    ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((pb - p0)[:, 0], axis=1) < FB_PX)
                    pts = p1[ok, 0]
                    if len(pts) < 3:
                        out[key] = [int(len(pts)), None, None, None]
                        continue
                    votes = {}
                    for x, y in pts:
                        inside = [(wn, bn) for wn, bn in nb if bn[0] <= x <= bn[2] and bn[1] <= y <= bn[3]]
                        if not inside:
                            continue
                        wn = min(inside, key=lambda z: np.hypot((z[1][0] + z[1][2]) / 2 - x, (z[1][1] + z[1][3]) / 2 - y))[0]
                        votes[wn] = votes.get(wn, 0) + 1
                    same = votes.get(w, 0) / len(pts)
                    others = {k: v for k, v in votes.items() if k != w}
                    ow = max(others, key=others.get) if others else None
                    out[key] = [int(len(pts)), round(same, 3), round(others[ow] / len(pts), 3) if ow is not None else 0.0, ow]
            prev, prev_r = g, r
        json.dump({'span': si, 'spans': len(by_span), 'elapsed_s': round(time.time() - t0), 'updated': time.strftime('%H:%M:%S')},
                  open(status, 'w'))
    with gzip.open(str(path).replace('.jsonl.gz', '.lk.json.gz'), 'wt') as f:
        json.dump(out, f)
    json.dump({'finished': time.strftime('%H:%M:%S'), 'people_ticks': len(out), 'elapsed_s': round(time.time() - t0)}, open(status, 'w'))


if __name__ == '__main__':
    main(sys.argv[1])
