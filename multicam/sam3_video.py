"""A short video of SAM 3 tracking people by itself (text "person"), for the owner's eye: every person in a
colour of his own and his track number, from the raw 2560x1440 recording at 12.5 fps.

usage: sam3_video.py DAY CAM FILM_SECONDS [LENGTH_S] [CONF]   -> data/logs/sam3_video_DAY_CAM_T.mp4 (H.264)"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
COL = [(66, 197, 245), (80, 220, 100), (245, 160, 66), (220, 90, 220), (240, 220, 70), (160, 100, 255), (60, 200, 200),
       (200, 160, 90), (120, 230, 180), (250, 120, 160), (90, 140, 250), (230, 230, 230)]


def raw_run(day, cam, film, seconds, step=2):
    """Frames of the raw recording from `film` (the day film's clock) for `seconds`, every `step`-th."""
    import cv2
    import day_masks
    import day_player
    from rawsource import segments
    small, info = day_masks.FRAMES.at(day, cam, film, str(ROOT))
    seg = next(s for s in day_player.segments(day, str(ROOT))[cam] if s.name == info['segment'])
    target = float(seg.times()[info['index']])
    path = next(p for p, _ in segments(cam, day) if os.path.basename(p) == info['segment'])
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, target * 1000))
    out, k = [], 0
    while len(out) < int(seconds * 25 / step):
        if not cap.grab():
            break
        if k % step == 0:
            ok, f = cap.retrieve()
            if ok:
                out.append(f)
        k += 1
    cap.release()
    return out


def main():
    import cv2
    import sam3_crop as C
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
    day, cam, film = sys.argv[1], sys.argv[2], float(sys.argv[3])
    length = float(sys.argv[4]) if len(sys.argv) > 4 else 15.0
    conf = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4
    t0 = time.time()
    frames = raw_run(day, cam, film, length)
    tmp = ROOT / 'data' / 'logs' / 'sam3_video_in.mp4'
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), 12.5, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    vp = SAM3VideoSemanticPredictor(overrides=dict(conf=conf, task='segment', mode='predict', model=str(C.W3), half=True, save=False, verbose=False, imgsz=1008))
    raw_out = ROOT / 'data' / 'logs' / 'sam3_video_raw.mp4'
    ow, oh = 1920, 1080
    vo = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*'mp4v'), 12.5, (ow, oh))
    seen, t1 = set(), time.time()
    for k, r in enumerate(vp(source=str(tmp), text=['person'], stream=True)):
        C.patch(vp)
        img = frames[k].copy() if k < len(frames) else r.orig_img.copy()
        if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for m, i, c in zip(r.masks.data.cpu().numpy() > 0.5, ids, cf):
                seen.add(int(i))
                if m.shape != img.shape[:2]:
                    m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                col = COL[int(i) % len(COL)]
                over = img.copy(); over[m] = col
                img = cv2.addWeighted(img, 0.72, over, 0.28, 0)
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, (0, 0, 0), 6); cv2.drawContours(img, cs, -1, col, 3)
                ys, xs = np.nonzero(m)
                if len(ys):
                    x, y = int(xs.mean()), max(40, int(ys.min()) - 12)
                    cv2.putText(img, '#%d %.2f' % (i, c), (x - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 7)
                    cv2.putText(img, '#%d %.2f' % (i, c), (x - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
        cv2.rectangle(img, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(img, 'SAM 3 tracking, text "person", conf >= %.2f  |  %s %s  frame %d  |  track numbers so far: %d' % (conf, day, cam, k, len(seen)),
                    (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        vo.write(cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA))
    vo.release()
    out = ROOT / 'data' / 'logs' / ('sam3_video_%s_%s_%d.mp4' % (day, cam, int(film)))
    import day_proxy
    subprocess.run([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-i', str(raw_out), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24',
                    '-pix_fmt', 'yuv420p', str(out)])
    rep = {'frames': len(frames), 'tracks': len(seen), 'read_s': round(t1 - t0), 'track_s': round(time.time() - t1), 'out': str(out),
           'mb': round(out.stat().st_size / 1e6, 1) if out.exists() else None}
    json.dump(rep, open(ROOT / 'data' / 'logs' / 'sam3_video.json', 'w'), indent=1)
    print(rep)


if __name__ == '__main__':
    main()
