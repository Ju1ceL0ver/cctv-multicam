"""The owner's two-class inside/outside model (inout_lab/binary_door.py, 07.10.2026) on a door stretch: every person of
SAM 3.1's tracks on every shown tick gets the model's side; drawn in acid colours: INSIDE green, OUTSIDE magenta,
uncertain (< 0.9) yellow with '?'; the side of a track smoothed over +-1.5 s next to the raw one; the owner's entries
and exits as banners. CPU only.

usage: door_binary_video.py TAG [SPAN_S] [OUT.mp4]  (TAG = door_<day>_<start>, or 'busiest' for the busiest of 19.09)"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inout_lab'))
GREEN, MAGENTA, YELLOW = (0, 255, 0), (255, 0, 255), (0, 255, 255)        # BGR


def main(tag='busiest', span='90', out=None):
    import cv2
    import day_proxy
    import door_compare_video as V
    import door_learn as L
    import door_v2 as D
    from binary_door import BinaryDoorClassifier
    clf = BinaryDoorClassifier()
    base_all = ROOT / 'data' / 'sam31_door'
    if tag == 'busiest':
        best = None
        for d in base_all.glob('door_20260919_*/cam1/report.json'):
            n = len(json.load(open(d))['person_of_piece'])
            if best is None or n > best[0]:
                best = (n, d.parent.parent.name)
        tag = best[1]
    base = base_all / tag / 'cam1'
    M, by_tick, info = V.people(base)
    day, a = tag.split('_')[1], float(tag.split('_')[2])
    a = next((x for x, y in D.stretches(day) if int(round(x)) == int(a)), a)
    owner = [{'kind': t['kind'], 't': t['t'] - L.SHIFT} for t in D.truth(day, True) if t['kind'] in ('in', 'out')]
    n = int(info['ticks'])
    ts = [a + k * D.TICK for k in range(n)]
    span = float(span)
    lo, hi = a, a + n * D.TICK
    if span > 0:
        ev = np.array([e['t'] for e in owner if lo <= e['t'] <= hi])
        if len(ev):
            c = max(ev, key=lambda t0: ((ev >= t0 - 5) & (ev <= t0 - 5 + span)).sum())
            lo, hi = c - 5, c - 5 + span
    stride = 3
    cap = cv2.VideoCapture(str(base / 'video.mp4'))
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # pass 1: the model on every person of every shown tick
    raw = {}                                               # (k, pid) -> (label, p_inside, uncertain)
    frames = {}
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = a + k * D.TICK
        if t > hi:
            break
        if k % stride == 0 and t >= lo:
            frames[k] = f
            rgb = f[:, :, ::-1].copy()
            for r, pid in by_tick.get(k, []):
                x1, y1, x2, y2 = M.rows[r, 4:8].astype(int)
                m = np.zeros((H, W), np.uint8)
                c_ = M.crop(r)
                m[y1:y1 + c_.shape[0], x1:x1 + c_.shape[1]] = c_[:H - y1, :W - x1]
                try:
                    p = clf.predict_rgb(rgb, m)
                except ValueError:
                    continue
                raw[(k, pid)] = (p['label'], p['p_inside'], p['uncertain'])
        k += 1
    cap.release()
    # smoothing per track: mean p_inside over +-1.5 s
    by_pid = {}
    for (k, pid), v in raw.items():
        by_pid.setdefault(pid, []).append((k, v[1]))
    smooth = {}
    for pid, xs in by_pid.items():
        ks = np.array([x[0] for x in xs]); ps = np.array([x[1] for x in xs])
        for kk, _ in xs:
            smooth[(kk, pid)] = float(ps[np.abs(ks - kk) * D.TICK <= 1.5].mean())
    out = out or str(ROOT / 'data' / 'logs' / ('door_binary_%s.mp4' % tag))
    Wd, Hd = 1280, 720
    enc = subprocess.Popen([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (Wd, Hd),
                            '-r', str(12.5 / stride), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p', out],
                           stdin=subprocess.PIPE)
    for k in sorted(frames):
        img = frames[k].copy()
        over = img.copy()
        for r, pid in by_tick.get(k, []):
            if (k, pid) not in raw:
                continue
            lab, p_in, unc = raw[(k, pid)]
            ps = smooth[(k, pid)]
            side = 'INSIDE' if ps >= 0.5 else 'OUTSIDE'
            sure = max(ps, 1 - ps) >= clf.threshold
            col = (GREEN if side == 'INSIDE' else MAGENTA) if sure else YELLOW
            x1, y1, x2, y2 = M.rows[r, 4:8].astype(int)
            m = np.zeros(img.shape[:2], bool)
            c_ = M.crop(r)
            m[y1:y1 + c_.shape[0], x1:x1 + c_.shape[1]] = c_[:img.shape[0] - y1, :img.shape[1] - x1]
            over[m] = col
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, col, 7)
            txt = 'P%d %s%s %d%%' % (pid, side, '' if sure else '?', round(100 * max(ps, 1 - ps)))
            ly = y1 - 14 if y1 > 70 else y2 + 50
            cv2.putText(img, txt, (max(5, (x1 + x2) // 2 - 140), ly), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 12)
            cv2.putText(img, txt, (max(5, (x1 + x2) // 2 - 140), ly), cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 5)
        img = cv2.addWeighted(img, 0.55, over, 0.45, 0)
        now = a + k * D.TICK
        on = [e for e in owner if 0 <= now - e['t'] <= 2.5]
        cv2.rectangle(img, (0, 0), (img.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(img, 'binary inside/outside  t=%.1f  green=INSIDE magenta=OUTSIDE yellow=unsure' % (k * D.TICK),
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        if on:
            txt = 'OWNER: ' + ' + '.join('ENTRY' if e['kind'] == 'in' else 'EXIT' for e in on)
            cv2.putText(img, txt, (20, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 12)
            cv2.putText(img, txt, (20, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 160, 255), 5)
        enc.stdin.write(cv2.resize(img, (Wd, Hd)).tobytes())
    enc.stdin.close(); enc.wait()
    print(json.dumps({'tag': tag, 'people_ticks': len(raw), 'unsure': sum(max(v, 1 - v) < clf.threshold for v in smooth.values()),
                      'out': out}), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
