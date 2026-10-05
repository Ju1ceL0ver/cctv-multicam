"""The same short door clip as door_video.py, but every mask, number and side comes from a v2 checkpoint (05.10.2026).

usage: door_video_model.py DAY FROM_S TO_S CKPT [OUT.mp4]   (film seconds; the card; ~3 s of warm-up before FROM_S)"""
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
COL = [(66, 197, 245), (80, 220, 100), (245, 160, 66), (220, 90, 220), (240, 220, 70), (160, 100, 255), (60, 200, 200),
       (200, 160, 90), (120, 230, 180), (250, 120, 160), (90, 140, 250), (230, 230, 230)]
SIDE = ['OUT', 'IN', 'DOOR']
WARM = 3.0


def main(day, a, b, ckpt, out=None):
    import day_movie
    import day_proxy
    import door_learn as L
    import door_v2 as D
    import sam31_segment as SS
    import slot_v2 as V
    a, b = float(a), float(b)
    cam = 'cam1'
    model = V.load(str(ROOT / ckpt)).to('cuda').eval()
    oc = D.OneCam(model, 'cuda', cam)
    raw = D.Raw(day, cam)
    start, _ = day_movie.clock(day, str(ROOT))
    lt = time.localtime(start)
    sod0 = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
    a0 = a - WARM
    samples = {}                                       # the recent hall, as in door_v2.run
    for k in range(int(D.BG_SPAN / D.BG_EVERY)):
        key = round((a0 - D.BG_SPAN + k * D.BG_EVERY) / D.BG_EVERY) * D.BG_EVERY
        img = raw.get(SS.tick_frames(day, cam, key, 1)[0])
        if img is None:
            continue
        g = cv2.resize(img, (2176, 1224), interpolation=cv2.INTER_AREA)
        bl = D.bg_long(day, sod0 + key, cam=cam)
        oc.reset()
        _, cut = oc.step(key, {'img': g, 'bg_long': bl, 'bg_now': cv2.resize(bl, (D.BG_W, D.BG_H))}, 0)
        samples[key] = D._sample(g, cut)
    oc.reset()
    now_bg = np.nan_to_num(np.nanmedian(np.stack(list(samples.values())), 0), nan=127).astype(np.uint8)
    truth = [(t['t'] - L.SHIFT, t['kind']) for t in D.truth(day, True) if a - 1 <= t['t'] - L.SHIFT <= b and t['kind'] in ('in', 'out')]
    out = out or str(ROOT / 'data' / 'door_v2' / ('door_clip_%s_%d_%s.mp4' % (day, a, Path(ckpt).parent.name)))
    W, H = 1280, 720
    enc = subprocess.Popen([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H),
                            '-r', '12.5', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', out],
                           stdin=subprocess.PIPE)
    n = int(round((b - a0) / D.TICK))
    ticks = SS.tick_frames(day, cam, a0, n)
    frames = []                                        # (t, frame, people, masks): two passes, so motionless "people" can go
    for i, tk in enumerate(ticks):
        t = a0 + i * D.TICK
        img = raw.get(tk)
        if img is None:
            continue
        g = cv2.resize(img, (2176, 1224), interpolation=cv2.INTER_AREA)
        people, _ = oc.step(i, {'img': g, 'bg_long': D.bg_long(day, sod0 + t, cam=cam), 'bg_now': now_bg}, i)
        if t >= a:
            D.add_io([{'s': 0, 't': t, 'p': people}])
            frames.append((t, cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA), people, [np.packbits(m) for m in oc.last_masks],
                           [m.shape for m in oc.last_masks]))
    # the advertising stand: the model, like its teacher, calls it a person; it never moves (door_sam.static_people's rule)
    # by place, not by number: the tracker may renumber it. A spot where some box of the same size sits in >= 70 % of
    # the frames is the stand; every box on it is dropped
    px = np.array([2176, 1248, 2176, 1248])
    boxes = [np.array([q['box'] for q in people]).reshape(-1, 4) * px for _, _, people, _, _ in frames]
    same = lambda B, c: len(B) and (np.abs(B - c) < [10, 10, 0.1 * c[2] + 4, 0.1 * c[3] + 4]).all(1)
    spots = []
    for B in boxes[::10]:
        for c in B:
            if not any((np.abs(c - s_) < 10).all() for s_ in spots) and np.mean([bool(np.any(same(B2, c))) for B2 in boxes]) >= 0.7:
                spots.append(c)
    POSTER = np.array([883, 87, 933, 246])            # cam1, 2176 px (section 42)

    def iou_poster(q):
        cx, cy, w, h = np.array(q['box']) * px
        b_ = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        i_ = max(0, min(b_[2], POSTER[2]) - max(b_[0], POSTER[0])) * max(0, min(b_[3], POSTER[3]) - max(b_[1], POSTER[1]))
        return i_ / ((b_[2] - b_[0]) * (b_[3] - b_[1]) + 50 * 159 - i_)
    on_stand = lambda q: 850 <= q['box'][0] * 2176 <= 1000 and 60 <= q['box'][1] * 1248 <= 280 and q['box'][3] * 1248 < 260
    static = lambda q: on_stand(q) or iou_poster(q) >= 0.4 or any(bool(same(np.array(q['box'])[None] * px, s_)[0]) for s_ in spots)
    wid = {}
    for t, g, people, packed, shapes in frames:
        sc = W / 2176
        vis, over, labels = g.copy(), g.copy(), []
        for q, pm, sh in zip(people, packed, shapes):
            if static(q):
                continue
            m4 = np.unpackbits(pm)[:sh[0] * sh[1]].reshape(sh)
            pid = wid.setdefault(q['w'], len(wid) + 1)
            m = cv2.resize(m4.astype(np.uint8), (W, int(round(m4.shape[0] * 4 * sc))), interpolation=cv2.INTER_LINEAR)[:H] > 0
            col = COL[pid % len(COL)]
            over[m] = col
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, col, 2)
            ys, xs = np.nonzero(m)
            if not len(ys):
                continue
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            ly = y1 - 8 if y1 - 8 > 75 else y2 + 30
            labels.append(((x1 + x2) // 2, ly, 'P%d %s' % (pid, SIDE[int(np.argmax(q['io']))]), col))
        vis = cv2.addWeighted(vis, 0.65, over, 0.35, 0)
        for x, y, txt, col in labels:
            cv2.putText(vis, txt, (x - 40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5)
            cv2.putText(vis, txt, (x - 40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)
        cv2.rectangle(vis, (0, 0), (W, 44), (0, 0, 0), -1)
        cv2.putText(vis, '%s door t=%.1f  model %s' % (day, t, Path(ckpt).parent.name), (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        gt = [kind for te, kind in truth if 0 <= t - te <= 2.0]
        if gt:
            cv2.putText(vis, 'OWNER: ' + ' + '.join('ENTRY' if e == 'in' else 'EXIT' for e in gt), (860, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 200, 255), 2)
        enc.stdin.write(vis.tobytes())
    enc.stdin.close(); enc.wait()
    print(out, len(wid), 'people', len(spots), 'static spots', len(truth), 'owner events')


if __name__ == '__main__':
    main(*sys.argv[1:])
