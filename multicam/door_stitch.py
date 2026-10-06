"""One person, one number at the door (06.10.2026): the small SAM sometimes loses a person standing close to another
for a frame or two and its detector starts a new track on the same spot -- a new number for somebody who never left.
Two repairs on door_v2's run format (ticks), before any rule sees them:

1. duplicates: at one tick two people whose boxes overlap by DUP_IOU or more are one; the older track keeps it.
2. continuation: a track that starts where another one has just ended (that one's last box and this one's first
   overlap by CONT_IOU, the gap at most CONT_GAP s) and is never seen together with it after that, is the same
   person: it takes the older number.

Applied by door_v2._read when RA_DOOR_STITCH=1, and by the live door."""
import numpy as np

DUP_IOU = 0.7
CONT_IOU = 0.35
CONT_GAP = 1.5


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / max(1e-9, a[2] * a[3] + b[2] * b[3] - inter)


def stitch(ticks):
    """In place; returns (duplicates dropped, tracks joined)."""
    # spans of each person per stretch, before the repairs
    first, last = {}, {}
    for r in ticks:
        for q in r['p']:
            key = (r['s'], q['w'])
            first.setdefault(key, (r['t'], q['box']))
            last[key] = (r['t'], q['box'])
    dropped = 0
    for r in ticks:                                   # 1. duplicates: the older track keeps the person
        ps = sorted(r['p'], key=lambda q: first[(r['s'], q['w'])][0])
        keep = []
        for q in ps:
            if any(iou(q['box'], k['box']) >= DUP_IOU for k in keep):
                dropped += 1
                continue
            keep.append(q)
        r['p'] = keep
    # 2. continuation, earliest first; a chain of breaks ends with the first number
    seen = {}
    for r in ticks:
        for q in r['p']:
            seen.setdefault((r['s'], q['w']), []).append(r['t'])
    times = {k: np.array(v) for k, v in seen.items()}
    by_stretch = {}
    for k in times:
        by_stretch.setdefault(k[0], []).append(k)
    parent = {}
    root = lambda k: root(parent[k]) if k in parent else k
    starts = sorted(times, key=lambda k: times[k][0])
    joined = 0
    for b in starts:
        tb0 = times[b][0]
        fb = first[b][1]
        best = None
        for a in by_stretch[b[0]]:
            if a == b or a[0] != b[0] or times[a][0] >= tb0:
                continue
            ta1 = times[a][-1]
            if not (tb0 - CONT_GAP <= ta1 <= tb0 + 0.3):
                continue
            if (times[a] > tb0 + 0.3).any():          # still seen after b started: two people
                continue
            la = last.get(a)
            if la is None:
                continue
            v = iou(la[1], fb)
            if v >= CONT_IOU and (best is None or v > best[0]):
                best = (v, a)
        if best is not None:
            parent[b] = best[1]
            joined += 1
    if parent:
        for r in ticks:
            for q in r['p']:
                k = (r['s'], q['w'])
                if k in parent:
                    q['w'] = root(k)[1]
    return dropped, joined
