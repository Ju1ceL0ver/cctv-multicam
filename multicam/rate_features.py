"""Inputs for the draft-quality model: for every frame the owner scored on /rate, three
background-difference maps that are cheap to keep up to date in real time (§36):

  short  |frame - selective background of the last 5 min|  (people who move; do not soak in)
  long   |frame - median of the last 30 min|                 (people who stand for minutes)
  days   |frame - median of empty frames of other days| / their pixel spread (MAD)
                                                              (people who sit for hours)

Offline the past is read from the raw recording. Seeking frame by frame cost ~20 s a frame
(stat_bg2: 118 frames in 2253 s); decoding key frames only (every 2 s in these files) reads a
15-minute file in ~11 s, so each camera-day is walked once, in time order, keeping the last
30 minutes of key frames. The frame itself is the /rate JPEG, so the maps line up with it.

Output: data/rate/features/<id>.npz, `d` float16 (3, 360, 640) in the order above, and `n` =
key frames used for (short, long), so thin histories at opening time are visible.

  python rate_features.py [--workers 8] [--force] [--all | --queue N]    (--all: every draft, ~1 MB each)
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
W, H = 640, 360
SHORT, LONG = 300, 1800          # s of history
LONG_STEP = 6                    # s between key frames in the 30-minute median
K = 2.5                          # a pixel within K spreads counts as background and may update it
KEY_GAP = 2.0                    # s between key frames of the raw files
LAB_W = np.array([0.5, 1.0, 1.0], np.float32)   # lightness counts half: shadows and exposure


def lab(bgr):
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def dist(a, b, sd=None):
    d = (a - b) if sd is None else (a - b) / sd
    return np.sqrt(((d * LAB_W) ** 2).sum(2))


def selective(frames, alpha):
    """Running background that only takes pixels it already agrees with (people do not soak in).
    frames: BGR uint8, oldest first; starts from their median."""
    mu = lab(np.median(np.stack(frames), axis=0).astype(np.uint8))
    var = np.full_like(mu, 36.0)
    for f in frames:
        d = lab(f) - mu
        z = np.sqrt(((d * d) / var * [0.25, 1, 1]).sum(2) / 2.25)
        bgm = (z < K)[..., None]
        mu += alpha * d * bgm
        var += alpha * (d * d - var) * bgm
    return mu


def maps(cur, shorts, longs, gmu, gsd):
    """cur: LAB float (H, W, 3); shorts/longs: BGR uint8 frames, oldest first. -> float32 (3, H, W).
    Frames stay uint8 until needed: 30 minutes of float Lab would be ~0.8 GB per camera-day."""
    out = np.zeros((3, H, W), np.float32)
    if shorts:
        out[0] = dist(cur, selective(shorts, KEY_GAP / 120.0))
    if longs:
        out[1] = dist(cur, lab(np.median(np.stack(longs), axis=0).astype(np.uint8)))
    out[2] = dist(cur, gmu, gsd)
    return out


def keyframes(path, ffmpeg):
    """(seconds into the file, BGR uint8 640x360) of every key frame."""
    o = subprocess.run([ffmpeg, '-hide_banner', '-nostdin', '-skip_frame', 'nokey', '-i', path,
                        '-vf', 'scale=%d:%d:flags=area,showinfo' % (W, H), '-fps_mode', 'passthrough',
                        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True)
    ts = [float(x) for x in re.findall(rb'pts_time:([\d.]+)', o.stderr)]
    n = len(o.stdout) // (W * H * 3)
    frames = np.frombuffer(o.stdout[:n * W * H * 3], np.uint8).reshape(n, H, W, 3)
    return list(zip(ts[:n], frames))


def parse(ident):
    day, cam, hms, seg, sec = ident.split('_')
    return day, cam, '%s_%s.mp4' % (hms, seg), int(sec)


def global_stats(root, cam, day, n=200):
    """Median and MAD spread of empty frames of this camera on other days (Lab, 640x360)."""
    import cv2
    import random
    base = os.path.join(root, 'data', 'seg_datasets', 'rf_20260925')
    empt = []
    for lab_txt in glob.glob(os.path.join(base, '*', 'labels', '*.txt')):
        i = os.path.basename(lab_txt)[:-4]
        if ('_%s_' % cam) in i and not i.startswith(day) and os.path.getsize(lab_txt) == 0:
            empt.append(lab_txt.replace(os.sep + 'labels' + os.sep, os.sep + 'images' + os.sep)[:-4] + '.jpg')
    random.Random(1).shuffle(empt)
    E = np.stack([lab(cv2.resize(cv2.imread(p), (W, H), interpolation=cv2.INTER_AREA)) for p in empt[:n]])
    gmu = np.median(E, axis=0)
    gsd = np.maximum(cv2.GaussianBlur(1.4826 * np.median(np.abs(E - gmu), axis=0), (9, 9), 0), 3.0)
    return gmu, gsd, len(empt[:n])


def camera_day(job):
    """Walk one camera-day in time order and write the maps of its scored frames."""
    import cv2
    import imageio_ffmpeg
    from rawsource import segments
    cv2.setNumThreads(2)
    root, day, cam, items, out_dir = job           # items: [(ident, jpg path)]
    t0 = time.time()
    segs = segments(cam, day)
    if not segs:
        return day, cam, [], 'no recording'
    day0 = segs[0][1].replace(hour=0, minute=0, second=0)
    start = {os.path.basename(p): (st - day0).total_seconds() for p, st in segs}
    want = []                                      # (absolute s of day, ident, jpg)
    for ident, jpg in items:
        _, _, fname, sec = parse(ident)
        if fname in start:
            want.append((start[fname] + sec, ident, jpg))
    want.sort()
    missing = [i for i, _ in items if i not in {w[1] for w in want}]
    gmu, gsd, n_empty = global_stats(root, cam, day)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    hist = deque()                                 # (abs s, BGR uint8), the last LONG seconds
    done, k = [], 0
    for p, st in segs:
        s0 = (st - day0).total_seconds()
        if k >= len(want):
            break
        if s0 + 900 < want[k][0] - LONG:           # nothing before the next frame needs this file
            continue
        for t, f in keyframes(p, ffmpeg):
            T = s0 + t
            while k < len(want) and want[k][0] <= T:
                done.append(_write(want[k], hist, gmu, gsd, out_dir))
                k += 1
            hist.append((T, f))
            while hist and hist[0][0] < T - LONG:
                hist.popleft()
    while k < len(want):                           # frames after the last key frame of the day
        done.append(_write(want[k], hist, gmu, gsd, out_dir))
        k += 1
    return day, cam, done, 'ok %d frames, %d empty frames for the day map, %.0f s, missing %s' % (
        len(done), n_empty, time.time() - t0, missing)


def _write(w, hist, gmu, gsd, out_dir):
    import cv2
    T, ident, jpg = w
    cur = lab(cv2.resize(cv2.imread(jpg), (W, H), interpolation=cv2.INTER_AREA))
    shorts = [f for t, f in hist if T - SHORT <= t < T]
    longs, last = [], -1e9
    for t, f in hist:
        if T - LONG <= t < T and t - last >= LONG_STEP - 0.01:
            longs.append(f)
            last = t
    d = maps(cur, shorts, longs, gmu, gsd)
    np.savez_compressed(os.path.join(out_dir, ident + '.npz'), d=d.astype(np.float16),
                        n=np.array([len(shorts), len(longs)]))
    return ident, len(shorts), len(longs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--all', action='store_true', help='every draft, not only the scored ones (for rate_boost.py select)')
    ap.add_argument('--queue', type=int, default=0, help='also the next N unscored frames in /rate order, ready before they are scored')
    a = ap.parse_args()
    from multiprocessing import Pool
    import rate
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, 'data', 'rate', 'features')
    os.makedirs(out_dir, exist_ok=True)
    items, scored = rate.index(root, force=True), rate.ratings(root)
    groups = {}
    queue = sorted((i for i in items if i not in scored), key=rate._key)[:a.queue]
    for ident in (items if a.all else list(scored) + queue):
        if ident not in items or (not a.force and os.path.exists(os.path.join(out_dir, ident + '.npz'))):
            continue
        day, cam, _, _ = parse(ident)
        groups.setdefault((day, cam), []).append((ident, str(items[ident][0])))
    print('%d frames to do in %d camera-days' % (sum(map(len, groups.values())), len(groups)), flush=True)
    jobs = [(root, d, c, v, out_dir) for (d, c), v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
    t0, report = time.time(), {}
    with Pool(a.workers) as pool:
        for day, cam, done, msg in pool.imap_unordered(camera_day, jobs):
            thin = [x for x in done if x[1] < 60 or x[2] < 150]
            print('%s %s: %s; short history < 2 min or long < 15 min: %d' % (day, cam, msg, len(thin)), flush=True)
            report['%s_%s' % (day, cam)] = {'msg': msg, 'frames': done}
    report['seconds'] = round(time.time() - t0)
    json.dump(report, open(os.path.join(root, 'data', 'rate', 'features_all.json' if a.all else 'features.json'), 'w'), indent=1)
    print('all done in %d s' % report['seconds'], flush=True)


if __name__ == '__main__':
    main()
